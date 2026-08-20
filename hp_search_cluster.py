from train_cluster import build_cluster_partition_map
import fcntl
import os
import numpy as np
import traceback
import torch
import optuna
import random
import time

from argparse import ArgumentParser, Namespace
from constants import EDGE_MODELS, NODE_EDGE_MODELS
from contextlib import redirect_stdout
from optuna.visualization import plot_optimization_history, plot_slice, plot_pareto_front
from pprint import pformat
from training import trainer_factory
from testing import DualAutoregressiveTester, NodeAutoregressiveTester, EdgeAutoregressiveTester
from typing import List, Tuple, Dict
from utils import Logger, file_utils, hp_search_utils, train_utils

def parse_args() -> Namespace:
    parser = ArgumentParser(description='')
    parser.add_argument("--config", type=str, required=True, help='Path to training config file')
    parser.add_argument("--hparam_config", type=str, required=True, help='Path to hyperparameter config file')
    parser.add_argument("--model", type=str, required=True, help='Model to use for training')
    parser.add_argument("--seed", type=int, default=42, help='Seed for random number generators')
    parser.add_argument("--device", type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'), help='Device to run on')
    
    parser.add_argument("--study_name", type=str, help='Name of the Optuna study')
    parser.add_argument("--storage", type=str, help='SQLite DB URL (e.g. sqlite:///my_study.db)')
    parser.add_argument("--n_trials_per_job", type=int, default=None, help='Number of trials this job should run')
    parser.add_argument("--use_cluster_gcn", action="store_true", help="Use Cluster-GCN for training")
    parser.add_argument("--num_clusters", type=int, default=10, help="Number of clusters for Cluster-GCN")
    parser.add_argument("--clusters_per_batch", type=int, default=1, help="Number of clusters per batch for Cluster-GCN")
    parser.add_argument("--sliding", action="store_true", help="Enable sliding cluster generation for Cluster-GCN")
    parser.add_argument("--batching_strategy", choices=("same_run", "cross_event"), default="same_run", help="Batching strategy for variable-topology mSWE Cluster-GCN")
    parser.add_argument("--objective_metric", choices=("depth_mae", "depth_rmse", "node_rmse", "edge_rmse", "unit_discharge_rmse", "unit_discharge_mae"), default=None,
                        help="Scalar metric to minimize. Overrides hparam_config objective_metric/objective_metrics.")
    parser.add_argument("--objective_source", choices=("validation", "test"), default=None,
                        help="Use validation metrics for fast search or full test rollout metrics for full confirmation.")
    parser.add_argument("--wandb_project", type=str, default=None, help="Weights & Biases project name for per-trial logging")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases entity/team name")
    parser.add_argument("--wandb_mode", choices=("online", "offline", "disabled"), default=None, help="Weights & Biases logging mode")
    parser.add_argument("--wandb_tags", nargs="*", default=None, help="Optional W&B tags added to every trial run")
    parser.add_argument("--wandb_run_name_prefix", type=str, default=None, help="Prefix for per-trial W&B run names")
    parser.add_argument("--fixed_params_file", type=str, default=None, help="YAML file containing fixed candidate params for confirmation runs")
    parser.add_argument("--fixed_params_index", type=int, default=None, help="Index into fixed_params_file candidates")
    parser.add_argument("--fixed_params_indices", type=str, default=None, help="Comma/range list of fixed candidate indices, e.g. 0-5,8")
    return parser.parse_args()


def _get_objective_metric_name() -> str:
    return _get_objective_metric_names()[0]


def _get_objective_metric_names() -> Tuple[str, ...]:
    if args.objective_metric:
        return (args.objective_metric,)
    metric_names = hparam_config.get('objective_metrics', hparam_config.get('objective_metric', 'depth_mae'))
    if isinstance(metric_names, str):
        metric_names = [metric_names]
    valid_metrics = {"depth_mae", "depth_rmse", "node_rmse", "edge_rmse", "unit_discharge_rmse", "unit_discharge_mae"}
    unknown = set(metric_names) - valid_metrics
    if unknown:
        raise ValueError(f"Unknown objective metric(s): {sorted(unknown)}. Valid metrics: {sorted(valid_metrics)}")
    return tuple(metric_names)


def _is_multi_objective() -> bool:
    return len(_get_objective_metric_names()) > 1


def _get_objective_source() -> str:
    return args.objective_source or hparam_config.get('objective_source', 'validation')


def _get_hard_prune_threshold() -> float | None:
    threshold = hparam_config.get('hard_prune_max_objective')
    return None if threshold is None else float(threshold)


def _select_objective_value(metric_name: str, metrics: Dict[str, float]) -> float:
    if metric_name not in metrics:
        raise KeyError(f"Unknown objective metric {metric_name}. Available metrics: {sorted(metrics)}")
    return metrics[metric_name]


def _select_objective_values(metrics: Dict[str, float]) -> float | Tuple[float, ...]:
    objective_metrics = _get_objective_metric_names()
    values = tuple(_select_objective_value(metric_name, metrics) for metric_name in objective_metrics)
    return values[0] if len(values) == 1 else values


def _objective_value_for_logging(value):
    return list(value) if isinstance(value, tuple) else value


def _primary_objective_value(value) -> float:
    return float(value[0] if isinstance(value, tuple) else value)


def _infinite_objective_value() -> float | Tuple[float, ...]:
    values = tuple(float('inf') for _ in _get_objective_metric_names())
    return values[0] if len(values) == 1 else values


def _primary_trial_value(trial):
    return trial.values[0] if _is_multi_objective() else trial.value

def _best_component(training_stats, key: str, fallback: float | None = None) -> float:
    values = training_stats.epoch_val_loss_components.get(key, [])
    if values:
        return float(np.nanmin(np.asarray(values, dtype=float)))
    if fallback is not None:
        return float(fallback)
    return float('nan')


def _validation_metrics_from_trainer(trainer) -> Dict[str, float]:
    stats = trainer.training_stats
    objective_metric = _get_objective_metric_name()
    objective_component_by_metric = {
        'depth_mae': 'val_node_depth_mae',
        'depth_rmse': 'val_node_depth_rmse',
        'node_rmse': 'val_node_rmse',
        'edge_rmse': 'val_edge_rmse',
        'unit_discharge_rmse': 'val_edge_unit_discharge_rmse',
        'unit_discharge_mae': 'val_edge_unit_discharge_mae',
    }
    objective_component = objective_component_by_metric[objective_metric]
    objective_values = np.asarray(stats.epoch_val_loss_components.get(objective_component, []), dtype=float)
    if objective_values.size == 0 or not np.isfinite(objective_values).any():
        raise ValueError(f'No finite validation values found for objective component: {objective_component}')
    best_epoch_idx = int(np.nanargmin(objective_values))

    def component_at(key: str, fallback_key: str | None = None) -> float:
        values = stats.epoch_val_loss_components.get(key, [])
        if values and best_epoch_idx < len(values):
            return float(values[best_epoch_idx])
        if fallback_key is not None:
            return component_at(fallback_key)
        return float('nan')

    node_rmse = component_at('val_node_rmse')
    edge_rmse = component_at('val_edge_rmse')
    depth_rmse = component_at('val_node_depth_rmse', 'val_node_rmse')
    depth_mae = component_at('val_node_depth_mae', 'val_node_depth_rmse')
    unit_discharge_rmse = component_at('val_edge_unit_discharge_rmse', 'val_edge_rmse')
    unit_discharge_mae = component_at('val_edge_unit_discharge_mae', 'val_edge_unit_discharge_rmse')
    event_metrics = []
    if getattr(trainer, 'validation_event_metrics_history', None) and best_epoch_idx < len(trainer.validation_event_metrics_history):
        event_metrics = trainer.validation_event_metrics_history[best_epoch_idx]
    return {
        'node_rmse': node_rmse,
        'depth_rmse': depth_rmse,
        'depth_mae': depth_mae,
        'edge_rmse': edge_rmse,
        'unit_discharge_rmse': unit_discharge_rmse,
        'unit_discharge_mae': unit_discharge_mae,
        'best_epoch': best_epoch_idx + 1,
        'event_metrics': event_metrics,
    }


def _maybe_prune_fold(trial: optuna.Trial | None,
                      fold_idx: int,
                      objective_value: float,
                      metric_name: str,
                      step: int | None = None) -> None:
    """Stop a trial early on a non-finite or clearly hopeless objective.

    Note: Optuna's own pruner only participates in single-objective studies. Multi-objective
    studies do not support trial.report()/should_prune(), so for those the hard threshold and
    the NaN/Inf check below are the only pruning in effect.
    """
    hard_threshold = _get_hard_prune_threshold()
    if not np.isfinite(objective_value):
        raise optuna.TrialPruned(f"NAN or Inf {metric_name} encountered after fold {fold_idx + 1}.")
    if hard_threshold is not None and objective_value > hard_threshold:
        raise optuna.TrialPruned(
            f"{metric_name}={objective_value:.4e} exceeds hard prune threshold {hard_threshold:.4e} "
            f"after fold {fold_idx + 1}."
        )
    if trial is not None and not _is_multi_objective():
        trial.report(objective_value, step=fold_idx if step is None else step)
        if trial.should_prune():
            raise optuna.TrialPruned(
                f"Optuna pruned trial at fold {fold_idx + 1} with {metric_name}={objective_value:.4e}."
            )


def _enforce_memory_tier(updated_config: Dict, trial: optuna.Trial) -> None:
    memory_tier = hparam_config.get('memory_tier', 'a100_40')
    hidden = int(updated_config['model_parameters'][args.model]['hidden_features'])
    clusters_per_batch = int(updated_config.get('clusters_per_batch', args.clusters_per_batch))
    if memory_tier in ('a100_40', 'a10040') and hidden >= 256 and clusters_per_batch >= 10:
        raise optuna.TrialPruned(
            'Skipping A100-40-incompatible config: '
            f'hidden_features={hidden}, clusters_per_batch={clusters_per_batch}. '
            'Use memory_tier=large on A100-80/H100/H200 to include this case.'
        )



DEFAULT_WANDB_PROJECT = 'dual-flood-gnn-cluster'


def _wandb_mode() -> str:
    return args.wandb_mode or os.getenv('WANDB_MODE', 'online')


def _wandb_project() -> str | None:
    project = args.wandb_project or os.getenv('WANDB_PROJECT', DEFAULT_WANDB_PROJECT)
    if project in (None, '', 'false', 'False', 'disabled'):
        return None
    return project


def _trial_wandb_config(trial: optuna.Trial, updated_config: Dict) -> Dict:
    train_cfg = updated_config.get('training_parameters', {})
    ar_cfg = train_cfg.get('autoregressive', {})
    loss_cfg = updated_config.get('loss_func_parameters', {})
    model_cfg = updated_config.get('model_parameters', {}).get(args.model, {})
    return {
        'trial_number': trial.number,
        'study_name': args.study_name,
        'objective_metric': _get_objective_metric_name(),
        'objective_metrics': list(_get_objective_metric_names()),
        'multi_objective': _is_multi_objective(),
        'objective_source': _get_objective_source(),
        'hard_prune_max_objective': _get_hard_prune_threshold(),
        'memory_tier': hparam_config.get('memory_tier', 'a100_40'),
        'config_path': args.config,
        'hparam_config_path': args.hparam_config,
        'model': args.model,
        'seed': args.seed,
        'slurm_job_id': os.getenv('SLURM_JOB_ID'),
        'slurm_array_job_id': os.getenv('SLURM_ARRAY_JOB_ID'),
        'slurm_array_task_id': os.getenv('SLURM_ARRAY_TASK_ID'),
        'use_cluster_gcn': args.use_cluster_gcn,
        'num_clusters': args.num_clusters,
        'clusters_per_batch_arg': args.clusters_per_batch,
        'sliding': args.sliding,
        'trial_params': dict(trial.params),
        'model_parameters': model_cfg,
        'training_parameters': {
            'learning_rate': train_cfg.get('learning_rate'),
            'adam_weight_decay': train_cfg.get('adam_weight_decay'),
            'gradient_clip_value': train_cfg.get('gradient_clip_value'),
            'num_epochs': train_cfg.get('num_epochs'),
            'early_stopping_patience': train_cfg.get('early_stopping_patience'),
            'autoregressive': ar_cfg,
        },
        'loss_func_parameters': {
            'edge_loss_weight': loss_cfg.get('edge_loss_weight'),
            'edge_pred_loss_scale': loss_cfg.get('edge_pred_loss_scale'),
            'use_local_mass_loss': loss_cfg.get('use_local_mass_loss'),
            'local_mass_loss_scale': loss_cfg.get('local_mass_loss_scale'),
            'local_mass_loss_weight': loss_cfg.get('local_mass_loss_weight'),
            'use_global_mass_loss': loss_cfg.get('use_global_mass_loss'),
        },
        'sampled_clusters_per_batch': updated_config.get('clusters_per_batch', args.clusters_per_batch),
    }


def _init_wandb_trial_run(trial: optuna.Trial, updated_config: Dict):
    project = _wandb_project()
    mode = _wandb_mode()
    if project is None or mode == 'disabled':
        return None
    try:
        import wandb
    except Exception as exc:
        logger.log(f'Warning: W&B requested but import failed: {exc}')
        return None

    tags = list(args.wandb_tags or [])
    tags.extend(['optuna', 'cluster-gcn', 'hp-search'])
    if args.study_name:
        tags.append(args.study_name)
    run_prefix = args.wandb_run_name_prefix or args.study_name or 'cluster-hp-search'
    run_name = f'{run_prefix}-trial-{trial.number}'
    try:
        run = wandb.init(
            project=project,
            entity=args.wandb_entity or os.getenv('WANDB_ENTITY'),
            name=run_name,
            mode=mode,
            tags=tags,
            config=_trial_wandb_config(trial, updated_config),
            reinit=True,
        )
        try:
            run.define_metric('epoch')
            run.define_metric('train/*', step_metric='epoch')
            run.define_metric('val/*', step_metric='epoch')
            run.define_metric('fold/*')
            run.define_metric('trial/*')
        except Exception as exc:
            logger.log(f'Warning: failed to define W&B metrics for trial {trial.number}: {exc}')
        return run
    except Exception as exc:
        logger.log(f'Warning: failed to initialize W&B run for trial {trial.number}: {exc}')
        return None


def _wandb_log(wandb_run, payload: Dict) -> None:
    if wandb_run is None or not payload:
        return
    try:
        wandb_run.log(payload)
    except Exception as exc:
        logger.log(f'Warning: failed to log to W&B: {exc}')


def _wandb_summary_update(wandb_run, payload: Dict) -> None:
    if wandb_run is None or not payload:
        return
    try:
        for key, value in payload.items():
            wandb_run.summary[key] = value
    except Exception as exc:
        logger.log(f'Warning: failed to update W&B summary: {exc}')


def _finish_wandb_run(wandb_run) -> None:
    if wandb_run is None:
        return
    try:
        wandb_run.finish()
    except Exception as exc:
        logger.log(f'Warning: failed to finish W&B run: {exc}')


def _log_training_history_to_wandb(wandb_run, training_stats, fold_idx: int, group_id: str) -> None:
    if wandb_run is None:
        return
    train_components = training_stats.epoch_loss_components
    val_components = training_stats.epoch_val_loss_components
    max_epochs = max(
        [len(training_stats.total_epoch_loss)]
        + [len(values) for values in train_components.values()]
        + [len(values) for values in val_components.values()]
        + [0]
    )
    for epoch_idx in range(max_epochs):
        payload = {
            'epoch': epoch_idx + 1,
            'fold/index': fold_idx,
            'fold/id': group_id,
        }
        if epoch_idx < len(training_stats.total_epoch_loss):
            payload['train/loss'] = float(training_stats.total_epoch_loss[epoch_idx])
        for key, values in train_components.items():
            if epoch_idx < len(values):
                payload[f'train/{key}'] = float(values[epoch_idx])
        for key, values in val_components.items():
            if epoch_idx < len(values):
                payload[f'val/{key}'] = float(values[epoch_idx])
        _wandb_log(wandb_run, payload)


def _make_epoch_pruning_callback(trial: optuna.Trial | None,
                                 wandb_run,
                                 fold_idx: int,
                                 group_id: str,
                                 completed_fold_objectives: List[float],
                                 num_epochs: int):
    objective_metric = _get_objective_metric_name()
    metric_key_by_objective = {
        'depth_mae': 'val/node_depth_mae',
        'depth_rmse': 'val/node_depth_rmse',
        'node_rmse': 'val/node_rmse',
        'edge_rmse': 'val/edge_rmse',
        'unit_discharge_rmse': 'val/edge_unit_discharge_rmse',
        'unit_discharge_mae': 'val/edge_unit_discharge_mae',
    }
    metric_key = metric_key_by_objective[objective_metric]

    def callback(trainer, epoch: int, metrics: Dict[str, float]) -> None:
        if metric_key not in metrics:
            return
        epoch_objective = float(metrics[metric_key])
        cumulative = float(np.mean(completed_fold_objectives + [epoch_objective]))
        step = fold_idx * (num_epochs + 1) + epoch + 1
        payload = {
            'cv/step': step,
            'cv/fold_index': fold_idx,
            'cv/epoch': epoch + 1,
            'cv/current_fold_objective': epoch_objective,
            'cv/cumulative_objective': cumulative,
            'fold/index': fold_idx,
            'fold/id': group_id,
            'fold/epoch': epoch + 1,
        }
        _wandb_log(wandb_run, payload)
        _maybe_prune_fold(trial, fold_idx, cumulative, objective_metric, step=step)

    return callback


def cross_validate(_config: Dict, cross_val_groups: List[str], trial: optuna.Trial = None, wandb_run=None) -> float | Tuple[float, ...]:
    fold_metrics = []
    completed_fold_objectives = []
    objective_metric = _get_objective_metric_name()
    objective_source = _get_objective_source()
    if objective_source != 'validation':
        raise ValueError('Hyperparameter search must use validation objective; final test is reserved for selected configs.')
    load_test = False
    for fold_idx, group_id in enumerate(cross_val_groups):
        fold_seed = args.seed + fold_idx if args.seed is not None else None
        if fold_seed is not None:
            random.seed(fold_seed)
            np.random.seed(fold_seed)
            torch.manual_seed(fold_seed)
            torch.cuda.manual_seed_all(fold_seed)
        logger.log(f'Cross-validating with Group {group_id} as the validation set...\n')

        train_dataset, test_dataset, val_dataset = hp_search_utils.load_datasets(
            group_id, _config, logger, load_test=load_test
        )
        model = hp_search_utils.load_model(args.model, _config, train_dataset, args.device)

        # ============ Training Phase ============
        logger.log('\nTraining model...')
        _train_config = _config['training_parameters']
        optimizer = torch.optim.Adam(model.parameters(), lr=_train_config['learning_rate'], weight_decay=_train_config['adam_weight_decay'])

        base_trainer_params = train_utils.get_trainer_config(args.model, _config)
        trainer_params = {
            'model': model,
            'dataset': train_dataset,
            'val_dataset': val_dataset,
            'optimizer': optimizer,
            'logger': None,
            'device': args.device,
            'same_run_batching': args.use_cluster_gcn and args.batching_strategy == 'same_run',
            'epoch_end_callback': _make_epoch_pruning_callback(
                trial,
                wandb_run,
                fold_idx,
                group_id,
                completed_fold_objectives,
                int(_train_config['num_epochs']),
            ),
            **base_trainer_params,
        }

        autoregressive_train_config = _train_config['autoregressive']
        autoregressive_enabled = autoregressive_train_config.get('enabled', False)

        if args.use_cluster_gcn:
            partition_map = build_cluster_partition_map(train_dataset, args.num_clusters, _config, logger)
            trainer_params['partition_map'] = partition_map
            trainer_params['clusters_per_batch'] = _config.get('clusters_per_batch', args.clusters_per_batch)
            trainer_params['sliding'] = args.sliding
            trainer_params['cross_event_batching'] = args.batching_strategy == 'cross_event'

        trainer = trainer_factory(args.model, autoregressive_enabled, isCluster=args.use_cluster_gcn, **trainer_params)
        with open(os.devnull, "w") as f, redirect_stdout(f):
            trainer.train()
        trainer.training_stats.log = logger.log  # Restore logging to console
        trainer.print_stats_summary()
        _log_training_history_to_wandb(wandb_run, trainer.training_stats, fold_idx, group_id)

        if objective_source == 'validation':
            logger.log('\nUsing best validation metrics as Optuna objective; skipping full test rollout for this fold.')
            metrics = _validation_metrics_from_trainer(trainer)
        else:
            # ============ Testing Phase ============
            logger.log('\nTesting model...')

            _test_config = _config['testing_parameters']
            tester_params = {
                'model': model,
                'dataset': test_dataset,
                'rollout_start': _test_config['rollout_start'],
                'rollout_timesteps': _test_config['rollout_timesteps'],
                'include_physics_loss': False,
                'logger': None,
                'device': args.device,
            }

            if is_dual_model:
                tester = DualAutoregressiveTester(**tester_params)
            elif model.__class__.__name__ in EDGE_MODELS:
                tester = EdgeAutoregressiveTester(**tester_params)
            else:
                tester = NodeAutoregressiveTester(**tester_params)
            with open(os.devnull, "w") as f, redirect_stdout(f):
                tester.test()

            metrics = {
                'node_rmse': float(tester.get_avg_node_rmse()),
                'depth_rmse': float(tester.get_avg_node_depth_rmse()) if hasattr(tester, 'get_avg_node_depth_rmse') else float(tester.get_avg_node_rmse()),
                'depth_mae': float(tester.get_avg_node_depth_mae()) if hasattr(tester, 'get_avg_node_depth_mae') else float(tester.get_avg_node_mae()),
                'edge_rmse': float(tester.get_avg_edge_rmse()) if is_dual_model else 0.0,
                'unit_discharge_rmse': float(tester.get_avg_edge_rmse()) if is_dual_model else 0.0,
                'unit_discharge_mae': float(tester.get_avg_edge_mae()) if is_dual_model and hasattr(tester, 'get_avg_edge_mae') else 0.0,
                'best_epoch': float('nan'),
                'event_metrics': [],
            }
        objective_value = _select_objective_value(objective_metric, metrics)
        objective_values = _select_objective_values(metrics)
        for metric_name in _get_objective_metric_names():
            metric_value = _select_objective_value(metric_name, metrics)
            if not np.isfinite(metric_value):
                raise optuna.TrialPruned(f"NAN or Inf {metric_name} encountered after fold {fold_idx + 1}.")
        fold_payload = {
            'fold/index': fold_idx,
            'fold/id': group_id,
            'fold/objective_value': objective_value,
            'fold/depth_mae': metrics['depth_mae'],
            'fold/depth_rmse': metrics['depth_rmse'],
            'fold/node_rmse': metrics['node_rmse'],
            'fold/edge_rmse': metrics['edge_rmse'],
            'fold/unit_discharge_rmse': metrics['unit_discharge_rmse'],
            'fold/unit_discharge_mae': metrics['unit_discharge_mae'],
            'fold/best_epoch': metrics.get('best_epoch', float('nan')),
        }
        if isinstance(objective_values, tuple):
            for metric_name, metric_value in zip(_get_objective_metric_names(), objective_values):
                fold_payload[f'fold/objective_{metric_name}'] = metric_value
        for event_metric in metrics.get('event_metrics', []):
            _wandb_log(wandb_run, {
                'event/fold_index': fold_idx,
                'event/run_id': event_metric['run_id'],
                'event/depth_mae': event_metric['depth_mae'],
                'event/depth_rmse': event_metric['depth_rmse'],
                'event/node_rmse': event_metric['node_rmse'],
                'event/edge_rmse': event_metric['edge_rmse'],
                'event/unit_discharge_rmse': event_metric.get('unit_discharge_rmse', float('nan')),
                'event/unit_discharge_mae': event_metric.get('unit_discharge_mae', float('nan')),
            })
        _wandb_log(wandb_run, fold_payload)
        _wandb_summary_update(wandb_run, {
            f'{group_id}_depth_mae': metrics['depth_mae'],
            f'{group_id}_depth_rmse': metrics['depth_rmse'],
            f'{group_id}_node_rmse': metrics['node_rmse'],
            f'{group_id}_edge_rmse': metrics['edge_rmse'],
            f'{group_id}_unit_discharge_rmse': metrics['unit_discharge_rmse'],
            f'{group_id}_unit_discharge_mae': metrics['unit_discharge_mae'],
        })
        cumulative_objective = float(np.mean(completed_fold_objectives + [objective_value]))
        _maybe_prune_fold(
            trial,
            fold_idx,
            cumulative_objective,
            objective_metric,
            step=fold_idx * (int(_train_config['num_epochs']) + 1) + int(_train_config['num_epochs']),
        )
        completed_fold_objectives.append(objective_value)
        fold_metrics.append(metrics)
        logger.log(
            f'Group {group_id} metrics: '
            f'depth_mae={metrics["depth_mae"]:.4e}, '
            f'depth_rmse={metrics["depth_rmse"]:.4e}, '
            f'node_rmse={metrics["node_rmse"]:.4e}, '
            f'edge_rmse={metrics["edge_rmse"]:.4e}, '
            f'unit_discharge_rmse={metrics["unit_discharge_rmse"]:.4e}, '
            f'unit_discharge_mae={metrics["unit_discharge_mae"]:.4e}'
        )
        if trial is not None:
            trial.set_user_attr(f'{group_id}_depth_mae', metrics['depth_mae'])
            trial.set_user_attr(f'{group_id}_depth_rmse', metrics['depth_rmse'])
            trial.set_user_attr(f'{group_id}_node_rmse', metrics['node_rmse'])
            trial.set_user_attr(f'{group_id}_edge_rmse', metrics['edge_rmse'])
            trial.set_user_attr(f'{group_id}_unit_discharge_rmse', metrics['unit_discharge_rmse'])
            trial.set_user_attr(f'{group_id}_unit_discharge_mae', metrics['unit_discharge_mae'])

    averages = {
        key: float(np.mean([metrics[key] for metrics in fold_metrics]))
        for key in ('node_rmse', 'depth_rmse', 'depth_mae', 'edge_rmse', 'unit_discharge_rmse', 'unit_discharge_mae')
    }
    fold_std = {
        key: float(np.std([metrics[key] for metrics in fold_metrics], ddof=1)) if len(fold_metrics) > 1 else 0.0
        for key in averages
    }
    fold_worst = {
        key: float(np.max([metrics[key] for metrics in fold_metrics]))
        for key in averages
    }
    logger.log(
        f'\nAverage metrics across folds: '
        f'depth_mae={averages["depth_mae"]:.4e}, '
        f'depth_rmse={averages["depth_rmse"]:.4e}, '
        f'node_rmse={averages["node_rmse"]:.4e}, '
        f'edge_rmse={averages["edge_rmse"]:.4e}, '
        f'unit_discharge_rmse={averages["unit_discharge_rmse"]:.4e}, '
        f'unit_discharge_mae={averages["unit_discharge_mae"]:.4e}'
    )
    objective_value = _select_objective_value(objective_metric, averages)
    objective_values = _select_objective_values(averages)
    if trial is not None:
        for key, value in averages.items():
            trial.set_user_attr(f'avg_{key}', value)
        for key, value in fold_std.items():
            trial.set_user_attr(f'std_{key}', value)
        for key, value in fold_worst.items():
            trial.set_user_attr(f'worst_{key}', value)
        trial.set_user_attr('objective_metric', objective_metric)
        trial.set_user_attr('objective_metrics', list(_get_objective_metric_names()))
    _wandb_log(wandb_run, {
        'trial/objective_value': objective_value,
        'trial/avg_depth_mae': averages['depth_mae'],
        'trial/avg_depth_rmse': averages['depth_rmse'],
        'trial/avg_node_rmse': averages['node_rmse'],
        'trial/avg_edge_rmse': averages['edge_rmse'],
        'trial/avg_unit_discharge_rmse': averages['unit_discharge_rmse'],
        'trial/avg_unit_discharge_mae': averages['unit_discharge_mae'],
        'trial/std_depth_mae': fold_std['depth_mae'],
        'trial/std_unit_discharge_mae': fold_std['unit_discharge_mae'],
        'trial/worst_depth_mae': fold_worst['depth_mae'],
        'trial/worst_unit_discharge_mae': fold_worst['unit_discharge_mae'],
    })
    _wandb_summary_update(wandb_run, {
        'objective_value': objective_value,
        'objective_values': _objective_value_for_logging(objective_values),
        'best_validation_depth_mae': averages['depth_mae'],
        'best_validation_depth_rmse': averages['depth_rmse'],
        'best_validation_unit_discharge_mae': averages['unit_discharge_mae'],
        'best_validation_unit_discharge_rmse': averages['unit_discharge_rmse'],
        'avg_node_rmse': averages['node_rmse'],
        'avg_edge_rmse': averages['edge_rmse'],
        'std_depth_mae': fold_std['depth_mae'],
        'std_unit_discharge_mae': fold_std['unit_discharge_mae'],
        'worst_depth_mae': fold_worst['depth_mae'],
        'worst_unit_discharge_mae': fold_worst['unit_discharge_mae'],
        'objective_metric': objective_metric,
        'objective_metrics': list(_get_objective_metric_names()),
    })
    return objective_values

def create_objective(cross_val_groups: List[str]):
    def objective(trial: optuna.Trial) -> float | Tuple[float, float]:
        hyperparamters = hparam_config['hyperparameters']
        start_time = time.time()
        wandb_run = None
        try:
            updated_config = hp_search_utils.suggest_hyperparamters(trial, hyperparamters, config, logger)
            wandb_run = _init_wandb_trial_run(trial, updated_config)
            _wandb_log(wandb_run, {'trial/status_code': 0, 'trial/number': trial.number})
            _enforce_memory_tier(updated_config, trial)
            objective_value = cross_validate(updated_config, cross_val_groups, trial=trial, wandb_run=wandb_run)
            duration_s = time.time() - start_time
            primary_objective_value = _primary_objective_value(objective_value)
            _wandb_log(wandb_run, {
                'trial/status_code': 1,
                'trial/duration_s': duration_s,
                'trial/objective_value': primary_objective_value,
            })
            _wandb_summary_update(wandb_run, {
                'trial_status': 'complete',
                'duration_s': duration_s,
                'objective_value': primary_objective_value,
                'objective_values': _objective_value_for_logging(objective_value),
                'trial_number': trial.number,
            })
            return objective_value
        except optuna.TrialPruned as e:
            duration_s = time.time() - start_time
            reason = str(e)
            logger.log(f"Trial {trial.number} pruned: {reason}")
            _wandb_log(wandb_run, {
                'trial/status_code': -1,
                'trial/duration_s': duration_s,
                'trial/pruned': 1,
            })
            _wandb_summary_update(wandb_run, {
                'trial_status': 'pruned',
                'pruning_reason': reason,
                'duration_s': duration_s,
                'trial_number': trial.number,
            })
            raise
        except torch.OutOfMemoryError as e:
            duration_s = time.time() - start_time
            reason = str(e)
            failed_objective = _infinite_objective_value()
            primary_failed_objective = _primary_objective_value(failed_objective)
            logger.log(f"Trial {trial.number} failed due to CUDA Out Of Memory: {e}")
            _wandb_log(wandb_run, {
                'trial/status_code': -2,
                'trial/duration_s': duration_s,
                'trial/oom': 1,
                'trial/objective_value': primary_failed_objective,
            })
            _wandb_summary_update(wandb_run, {
                'trial_status': 'failed',
                'failure_type': 'cuda_oom',
                'failure_reason': reason,
                'duration_s': duration_s,
                'objective_value': primary_failed_objective,
                'objective_values': _objective_value_for_logging(failed_objective),
                'trial_number': trial.number,
            })
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            return failed_objective
        except (RuntimeError, AssertionError, TypeError, ValueError, KeyError) as e:
            duration_s = time.time() - start_time
            reason = str(e)
            failed_objective = _infinite_objective_value()
            primary_failed_objective = _primary_objective_value(failed_objective)
            logger.log(f"Trial {trial.number} failed due to recoverable error: {e}")
            logger.log(traceback.format_exc())
            _wandb_log(wandb_run, {
                'trial/status_code': -3,
                'trial/duration_s': duration_s,
                'trial/recoverable_error': 1,
                'trial/objective_value': primary_failed_objective,
            })
            _wandb_summary_update(wandb_run, {
                'trial_status': 'failed',
                'failure_type': e.__class__.__name__,
                'failure_reason': reason,
                'duration_s': duration_s,
                'objective_value': primary_failed_objective,
                'objective_values': _objective_value_for_logging(failed_objective),
                'trial_number': trial.number,
            })
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return failed_objective
        finally:
            _finish_wandb_run(wandb_run)
    return objective

def plot_hyperparameter_search_results(study: optuna.Study):
    output_dir = hparam_config['output_dir']
    if output_dir is None:
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    node_plot_params = {
        'study': study,
        'target_name': _get_objective_metric_name(),
        'target': lambda t: _primary_trial_value(t),
    }

    fig = plot_optimization_history(**node_plot_params)
    fig.write_html(os.path.join(output_dir, f'{study.study_name}_optimization_history.html'))

    fig = plot_slice(**node_plot_params)
    fig.write_html(os.path.join(output_dir, f'{study.study_name}_slice_plot.html'))


def _parse_fixed_param_indices(indices_arg: str, num_candidates: int) -> List[int]:
    indices = []
    for chunk in indices_arg.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        if '-' in chunk:
            start_text, end_text = chunk.split('-', 1)
            start_idx = int(start_text)
            end_idx = int(end_text)
            if end_idx < start_idx:
                raise ValueError(f'Invalid fixed index range: {chunk}')
            indices.extend(range(start_idx, end_idx + 1))
        else:
            indices.append(int(chunk))
    if not indices:
        raise ValueError('--fixed_params_indices did not contain any indices.')
    unique_indices = []
    seen = set()
    for idx in indices:
        if idx in seen:
            continue
        if idx < 0 or idx >= num_candidates:
            raise IndexError(f'fixed params index {idx} is outside candidate range 0..{num_candidates - 1}')
        seen.add(idx)
        unique_indices.append(idx)
    return unique_indices


def _load_fixed_params_list() -> List[Dict] | None:
    if args.fixed_params_file is None:
        return None
    if args.fixed_params_index is not None and args.fixed_params_indices is not None:
        raise ValueError('Use only one of --fixed_params_index or --fixed_params_indices.')
    if args.fixed_params_index is None and args.fixed_params_indices is None:
        raise ValueError('--fixed_params_index or --fixed_params_indices is required when --fixed_params_file is set.')
    data = file_utils.read_yaml_file(args.fixed_params_file)
    candidates = data.get('candidates', data if isinstance(data, list) else None)
    if not isinstance(candidates, list):
        raise ValueError('Fixed params file must be a list or contain a candidates list.')
    if args.fixed_params_index is not None:
        indices = _parse_fixed_param_indices(str(args.fixed_params_index), len(candidates))
    else:
        indices = _parse_fixed_param_indices(args.fixed_params_indices, len(candidates))
    fixed_params_list = []
    for idx in indices:
        candidate = candidates[idx]
        fixed_params_list.append(dict(candidate.get('params', candidate)))
    return fixed_params_list


def _storage_lock_path(storage_url: str | None) -> str | None:
    if not storage_url or not storage_url.startswith('sqlite:///'):
        return None
    return storage_url.replace('sqlite:///', '', 1) + '.create.lock'

def main():
    try:
        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            logger.log(f'Setting random seed to {args.seed}')

        current_device = torch.cuda.get_device_name(args.device) if args.device != 'cpu' else 'CPU'
        logger.log(f'Using device: {current_device}')

        # Begin hyperparameter search
        temp_dir_name = hparam_config.get('temp_dir_name')
        if temp_dir_name:
            array_task_id = os.getenv('SLURM_ARRAY_TASK_ID')
            if args.storage and array_task_id:
                hp_search_utils.TEMP_DIR_NAME = f"{temp_dir_name}_{array_task_id}"
            else:
                hp_search_utils.TEMP_DIR_NAME = temp_dir_name
        logger.log(f'Using hyperparameter-search temp dir: {hp_search_utils.TEMP_DIR_NAME}')

        dataset_parameters = config['dataset_parameters']
        root_dir = dataset_parameters['root_dir']
        dataset_summary_file = dataset_parameters['training']['dataset_summary_file']
        early_stopping_patience = train_config['early_stopping_patience']
        if early_stopping_patience is None:
            raise ValueError('CV hyperparameter search requires early_stopping_patience so validation metrics are computed.')
        if _get_objective_source() != 'validation':
            raise ValueError('CV hyperparameter search must use objective_source=validation.')
        validation_parameters = dataset_parameters.get('validation')
        if validation_parameters is not None and validation_parameters.get('dataset_summary_file') is not None:
            val_summary_file = validation_parameters['dataset_summary_file']
            logger.log(
                f'Creating explicit train/validation dataset files: '
                f'train={dataset_summary_file}, validation={val_summary_file}'
            )
            cross_val_groups, temp_dir_paths = hp_search_utils.create_explicit_train_val_dataset_files(
                root_dir,
                dataset_summary_file,
                val_summary_file,
            )
        else:
            num_folds = hparam_config['num_folds']
            if num_folds == 1:
                percent_validation = train_config['val_split_percent']
                if percent_validation is None:
                    raise ValueError('num_folds=1 requires training_parameters.val_split_percent.')
                logger.log(
                    f'Creating single train/validation split from {dataset_summary_file} '
                    f'with val_split_percent={percent_validation}.'
                )
                train_summary_file, val_summary_file = train_utils.split_dataset_events(
                    root_dir,
                    dataset_summary_file,
                    percent_validation,
                    temp_dir_name=f'{hp_search_utils.TEMP_DIR_NAME}_train_val_split',
                )
                cross_val_groups, temp_dir_paths = hp_search_utils.create_explicit_train_val_dataset_files(
                    root_dir,
                    train_summary_file,
                    val_summary_file,
                )
            else:
                logger.log(f'Creating {num_folds}-fold cross-validation dataset files from {dataset_summary_file}...')
                cross_val_groups, temp_dir_paths, manifest_path = hp_search_utils.create_cross_val_dataset_files(
                    root_dir,
                    dataset_summary_file,
                    num_folds,
                    fold_seed=int(hparam_config.get('fold_seed', args.seed if args.seed is not None else 42)),
                    manifest_dir=hparam_config.get('manifest_dir', 'cv_manifests'),
                )
                logger.log(f'Using fold manifest: {manifest_path}')

        study_name = "_".join(hparam_config['hyperparameters'].keys())
        if args.study_name:
            study_name = args.study_name
        study_kwargs = {'study_name': study_name }
        if args.storage:
            study_kwargs['storage'] = args.storage
            study_kwargs['load_if_exists'] = True
            array_task_id = int(os.getenv('SLURM_ARRAY_TASK_ID', '0') or 0)
            sampler_seed = None if args.seed is None else args.seed + array_task_id
            study_kwargs['sampler'] = optuna.samplers.TPESampler(
                seed=sampler_seed,
                n_startup_trials=30,
                multivariate=True,
                group=True,
                constant_liar=True,
            )
            pruner_config = hparam_config.get('pruner', {})
            study_kwargs['pruner'] = optuna.pruners.MedianPruner(
                n_startup_trials=int(pruner_config.get('n_startup_trials', 10)),
                n_warmup_steps=int(pruner_config.get('n_warmup_steps', 3)),
                interval_steps=int(pruner_config.get('interval_steps', 1)),
            )
        if _is_multi_objective():
            study_kwargs['directions'] = ['minimize' for _ in _get_objective_metric_names()]
        else:
            study_kwargs['direction'] = 'minimize'
        num_trials = args.n_trials_per_job if args.n_trials_per_job is not None else hparam_config['num_trials']
        lock_path = _storage_lock_path(args.storage)
        if lock_path is None:
            study = optuna.create_study(**study_kwargs)
        else:
            os.makedirs(os.path.dirname(lock_path), exist_ok=True)
            with open(lock_path, 'w') as lock_file:
                logger.log(f'Acquiring Optuna storage creation lock: {lock_path}')
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                try:
                    study = optuna.create_study(**study_kwargs)
                finally:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)
        fixed_params_list = _load_fixed_params_list()
        if fixed_params_list is not None:
            for fixed_params in fixed_params_list:
                study.enqueue_trial(fixed_params, skip_if_exists=False)
            num_trials = len(fixed_params_list)
            logger.log(f'Enqueued {num_trials} fixed confirmation params: {fixed_params_list}')
        logger.log(f'Using sampler: {study.sampler.__class__.__name__ if study.sampler else None}')
        logger.log(f'Using pruner: {study.pruner.__class__.__name__ if study.pruner else None}')
        logger.log(f'Objective source: {_get_objective_source()}')
        logger.log(f'Objective metrics: {list(_get_objective_metric_names())}')
        logger.log(f'Hard prune threshold: {_get_hard_prune_threshold()}')
        logger.log(f'Memory tier: {hparam_config.get("memory_tier", "a100_40")}')

        objective = create_objective(cross_val_groups)
        logger.log(f'Running hyperparameter search for {num_trials} trials...')
        study.optimize(objective, n_trials=num_trials)

        if _is_multi_objective():
            logger.log('Pareto-optimal trials found:')
            for best_trial in study.best_trials:
                metrics = ', '.join(
                    f'{metric_name}={metric_value:.4e}'
                    for metric_name, metric_value in zip(_get_objective_metric_names(), best_trial.values)
                )
                logger.log(f'Trial {best_trial.number}: {metrics}, params={best_trial.params}')
                logger.log(f'Trial {best_trial.number} secondary metrics: {best_trial.user_attrs}')
        else:
            logger.log('Best hyperparameters found:')
            for key, value in study.best_params.items():
                logger.log(f'{key}: {value}')
            logger.log(f'Best {_get_objective_metric_name()} objective value: {study.best_value:.4e}')
            logger.log(f'Best trial secondary metrics: {study.best_trial.user_attrs}')

        # Plot hyperparameter search results
        plot_hyperparameter_search_results(study)
    except Exception:
        logger.log(f"Unexpected error:\n{traceback.format_exc()}")
        import sys
        sys.exit(1)
    finally:
        if 'temp_dir_paths' in locals() and not args.storage:
            # Clean up temporary directories ONLY if not using distributed storage, otherwise it might delete shared caches!
            file_utils.delete_temp_dirs(temp_dir_paths)

if __name__ == '__main__':
    args = parse_args()
    config = file_utils.read_yaml_file(args.config)
    hparam_config = file_utils.read_yaml_file(args.hparam_config)
    config['model_parameters'] = {args.model: config['model_parameters'][args.model]}

    # Initialize logger
    train_config = config['training_parameters']
    log_path = train_config['log_path']
    logger = Logger(log_path=log_path)

    logger.log('================================================')
    logger.log(f'Running Hyperparameter Search with {args.model} model')
    logger.log(f'Configuration: {pformat(config)}')
    logger.log(f'Hyperparameter Search Configuration: {pformat(hparam_config)}')

    is_dual_model = args.model in NODE_EDGE_MODELS

    main()

    logger.log('================================================')
