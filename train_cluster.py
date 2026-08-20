import numpy as np
import json
import os
import subprocess
import sys
import traceback
import torch
import gc
import random
import torch.optim as optim # <<< CLUSTER-GCN MODIFICATION >>> (Ensure optim is imported)

from argparse import ArgumentParser, Namespace, BooleanOptionalAction
from datetime import datetime
from torch_geometric.loader import ClusterLoader, DataLoader # <<< CLUSTER-GCN MODIFICATION >>> (Import ClusterLoader)
from torch_geometric.data import Data # <<< CLUSTER-GCN MODIFICATION >>>

from data import dataset_factory, FloodEventDataset
from models import model_factory
from test import get_test_dataset_config, run_test
# <<< CLUSTER-GCN MODIFICATION >>> (Import partitioning utils and the cluster trainer)
print("Attempting to import Cluster-GCN specific components...", flush=True)
try:
    from utils.cluster_utils import load_base_graph_structure, partition_graph, visualize_partitions
    from training import trainer_factory, ClusterDualAutoregressiveTrainer 
    print("Cluster-GCN specific components imported successfully.", flush=True)
except ImportError as e:
    print(f"Warning: Could not import Cluster-GCN specific components: {e}", flush=True)
    # Define dummy functions/classes if needed for script to load
    def load_base_graph_structure(*args, **kwargs): raise NotImplementedError("Import failed")
    def partition_graph(*args, **kwargs): raise NotImplementedError("Import failed")
    class ClusterDualAutoregressiveTrainer: pass # Dummy class

from training import trainer_factory # Keep original factory for non-cluster case
from typing import Dict, Optional, Tuple
from utils import Logger, file_utils, train_utils

DEFAULT_PROJECT = "dual-flood-gnn-cluster"

def parse_args() -> Namespace:
    parser = ArgumentParser(description='')
    parser.add_argument("--config", type=str, required=True, help='Path to training config file')
    parser.add_argument("--model", type=str, required=True, help='Model to use for training')
    parser.add_argument("--with_test", action=BooleanOptionalAction, default=False, help='Whether to run test after training')
    parser.add_argument("--collect_regression_metrics", action="store_true", help="Collect regression metrics after test outputs are generated")
    parser.add_argument("--depth_target_dir", type=str, default=None, help="Optional directory containing notebook-compatible depth targets")
    parser.add_argument("--seed", type=int, default=42, help='Seed for random number generators')
    parser.add_argument("--device", type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'), help='Device to run on')
    parser.add_argument("--debug", type=bool, default=False, help='Add debug messages to output')
    # <<< CLUSTER-GCN MODIFICATION >>> (Add arg for enabling cluster GCN)
    parser.add_argument("--use_cluster_gcn", action='store_true', help='Enable Cluster-GCN training strategy')
    parser.add_argument("--num_clusters", type=int, default=30, help='Number of clusters for Cluster-GCN (if enabled)')
    parser.add_argument("--clusters_per_batch", type=int, default=5, help='Number of clusters per batch for Cluster-GCN (if enabled)')
    parser.add_argument("--sliding", action=BooleanOptionalAction, default=False, help='Use sliding window for cluster selection (if enabled). Use --sliding or --no-sliding')
    parser.add_argument(
        "--batching_strategy",
        choices=("same_run", "cross_event"),
        default="same_run",
        help="Batching strategy for variable-topology mSWE Cluster-GCN. same_run preserves existing behavior; cross_event allows multiple events per optimizer step.",
    )
    parser.add_argument("--amp", choices=("none", "bf16", "fp16"), default="none", help="Mixed precision mode for CUDA training")
    parser.add_argument("--compile_model", action="store_true", help="Enable torch.compile for the model")
    parser.add_argument("--fused_adam", action="store_true", help="Use fused Adam when supported by this PyTorch/CUDA build")
    parser.add_argument("--wandb_project", type=str, default=None, help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases entity/team name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Weights & Biases run name")
    parser.add_argument("--wandb_mode", choices=("online", "offline", "disabled"), default=None, help="Weights & Biases logging mode")
    parser.add_argument("--wandb_tags", nargs="*", default=None, help="Optional Weights & Biases tags")
    return parser.parse_args()


def init_wandb_run(args: Namespace, config: Dict, logger: Logger):
    wandb_project = args.wandb_project or os.getenv("WANDB_PROJECT", DEFAULT_PROJECT)
    if wandb_project is None:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb logging requested, but the wandb package is not installed.") from exc

    wandb_mode = args.wandb_mode or os.getenv("WANDB_MODE", "online")
    wandb_entity = args.wandb_entity or os.getenv("WANDB_ENTITY")
    run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=args.wandb_run_name,
        mode=wandb_mode,
        tags=args.wandb_tags,
        config={
            "config_path": args.config,
            "model": args.model,
            "seed": args.seed,
            "amp": args.amp,
            "compile_model": args.compile_model,
            "fused_adam": args.fused_adam,
            "use_cluster_gcn": args.use_cluster_gcn,
            "num_clusters": args.num_clusters,
            "clusters_per_batch": args.clusters_per_batch,
            "sliding": args.sliding,
            "batching_strategy": args.batching_strategy,
            "training_config": config.get("training_parameters", {}),
            "loss_config": config.get("loss_func_parameters", {}),
        },
    )
    logger.log(f"Initialized W&B run: project={wandb_project}, name={run.name}, id={run.id}")
    return run


def log_test_metrics_to_wandb(wandb_run, metrics_dir: str, logger: Logger, metrics_prefix: str = None):
    """Aggregate the per-event test .npz files into test/mean_* and test/std_* metrics.

    Key names match dual_flood_gnn/train.py so cluster runs and single-mesh runs are
    directly comparable in W&B. Mass-conservation losses are summed and taken absolute
    per event (signed values cancel); everything else is averaged over timesteps.
    """
    if wandb_run is None or not metrics_dir or not os.path.isdir(metrics_dir):
        return

    metric_keys = [
        "rmse", "rmse_flooded", "mae", "mae_flooded", "nse", "nse_flooded", "csi",
        "edge_rmse", "edge_mae", "edge_nse", "global_mass_loss", "local_mass_loss",
        "inference_time",
    ]
    event_rows = []
    for filename in os.listdir(metrics_dir):
        if not filename.endswith(".npz"):
            continue
        if metrics_prefix and not filename.startswith(metrics_prefix):
            continue
        path = os.path.join(metrics_dir, filename)
        try:
            data = np.load(path)
            row = {}
            for key in metric_keys:
                if key not in data.files:
                    continue
                arr = np.asarray(data[key])
                if arr.size == 0:
                    continue
                if arr.shape == ():
                    row[key] = float(arr)
                elif key in {"global_mass_loss", "local_mass_loss"}:
                    row[key] = float(abs(np.sum(arr)))
                else:
                    row[key] = float(np.mean(arr))
            if row:
                event_rows.append(row)
        except Exception as exc:
            logger.log(f"Warning: failed to read test metrics for W&B from {path}: {exc}")

    if not event_rows:
        return

    summary = {}
    for key in metric_keys:
        values = [row[key] for row in event_rows if key in row]
        if values:
            summary[f"test/mean_{key}"] = float(np.mean(values))
            summary[f"test/std_{key}"] = float(np.std(values))
    summary["test/event_count"] = len(event_rows)

    try:
        wandb_run.log(summary)
        for key, value in summary.items():
            wandb_run.summary[key] = value
        logger.log(f"Logged aggregated test metrics from {len(event_rows)} events to W&B.")
    except Exception as exc:
        logger.log(f"Warning: failed to log test metrics to W&B: {exc}")


def log_summary_json_to_wandb(wandb_run, output_json: str, logger: Logger):
    if wandb_run is None:
        return

    try:
        with open(output_json, "r") as f:
            payload = json.load(f)
    except Exception as exc:
        logger.log(f"Warning: failed to read summary metrics JSON for W&B logging: {exc}")
        return

    wandb_metrics = {}
    for group in ("mean", "std"):
        for key, value in payload.get(group, {}).items():
            if key == "run_id" or not isinstance(value, (int, float)):
                continue
            wandb_metrics[f"test/{group}_{key}"] = float(value)

    if not wandb_metrics:
        return

    try:
        wandb_run.log(wandb_metrics)
        for key, value in wandb_metrics.items():
            wandb_run.summary[key] = value
        logger.log(f"Logged {len(wandb_metrics)} standardized summary metrics to W&B.")
    except Exception as exc:
        logger.log(f"Warning: failed to log standardized summary metrics to W&B: {exc}")


def collect_regression_metrics(args: Namespace, config: Dict, logger: Logger, metrics_dir: str, wandb_run=None, metrics_glob: str | None = None):
    dataset_parameters = config["dataset_parameters"]
    dataset_type = get_dataset_type(config)

    if dataset_type == "mswegnn":
        output_csv = os.path.join(metrics_dir, "mswegnn_paper_metrics_summary.csv")
        output_json = os.path.join(metrics_dir, "mswegnn_paper_metrics_summary.json")
        cmd = [
            sys.executable,
            "collect_mswegnn_test_metrics.py",
            "--metrics_dir",
            metrics_dir,
            "--root_dir",
            dataset_parameters["root_dir"],
            "--config",
            args.config,
            "--output_csv",
            output_csv,
            "--output_json",
            output_json,
        ]
        if metrics_glob is not None:
            cmd.extend(["--metrics_glob", metrics_glob])
        logger.log(f"Collecting standardized mSWE-GNN paper metrics into: {output_csv}")
    else:
        output_csv = os.path.join(metrics_dir, "regression_metrics_summary.csv")
        output_json = os.path.join(metrics_dir, "regression_metrics_summary.json")
        cmd = [
            sys.executable,
            "collect_regression_metrics.py",
            "--metrics_dir",
            metrics_dir,
            "--root_dir",
            dataset_parameters["root_dir"],
            "--config",
            args.config,
            "--output_csv",
            output_csv,
            "--output_json",
            output_json,
        ]
        if metrics_glob is not None:
            cmd.extend(["--metrics_glob", metrics_glob])
        if args.depth_target_dir is not None:
            cmd.extend(["--depth_target_dir", args.depth_target_dir])
        logger.log(f"Collecting regression metrics into: {output_csv}")

    subprocess.run(cmd, check=True)
    log_summary_json_to_wandb(wandb_run, output_json, logger)


def get_dataset_type(config: Dict) -> str:
    return config.get('dataset_parameters', {}).get('dataset_type', 'hecras')


def build_base_dataset_config(config: Dict, args: Namespace, logger: Logger) -> Dict:
    dataset_parameters = config['dataset_parameters']
    loss_func_parameters = config['loss_func_parameters']
    base_config = {
        'root_dir': dataset_parameters['root_dir'],
        'features_stats_file': dataset_parameters['features_stats_file'],
        'previous_timesteps': dataset_parameters['previous_timesteps'],
        'normalize': dataset_parameters['normalize'],
        'boundary_aware_features': dataset_parameters.get('boundary_aware_features', False),
        'boundary_aware_feature_groups': dataset_parameters.get('boundary_aware_feature_groups', None),
        'boundary_aware_cache_prefix': dataset_parameters.get('boundary_aware_cache_prefix', None),
        'timestep_interval': dataset_parameters['timestep_interval'],
        'spin_up_time': dataset_parameters['spin_up_time'],
        'time_from_peak': dataset_parameters['time_from_peak'],
        'inflow_boundary_nodes': dataset_parameters.get('inflow_boundary_nodes', []),
        'outflow_boundary_nodes': dataset_parameters.get('outflow_boundary_nodes', []),
        'with_global_mass_loss': loss_func_parameters.get('use_global_mass_loss', False),
        'with_local_mass_loss': loss_func_parameters.get('use_local_mass_loss', False),
        'debug': args.debug,
        'logger': logger,
        'force_reload': False,
    }
    if get_dataset_type(config) != 'mswegnn':
        base_config['nodes_shp_file'] = dataset_parameters['nodes_shp_file']
        base_config['edges_shp_file'] = dataset_parameters['edges_shp_file']
    return base_config


def ensure_dataset_run_id_attrs(dataset) -> None:
    """Make an mSWE dataset look like the HEC-RAS one to older call sites.

    Aliases event_run_ids -> hec_ras_run_ids, exposes a single boundary_condition, and stamps
    run_id/event_idx onto every sample so batching and per-event partition lookup can tell
    which topology a graph came from.
    """
    if hasattr(dataset, 'event_run_ids'):
        dataset.hec_ras_run_ids = dataset.event_run_ids
    if hasattr(dataset, 'boundary_conditions') and dataset.boundary_conditions:
        dataset.boundary_condition = dataset.boundary_conditions[0]
    if not hasattr(dataset, 'data_list') or not hasattr(dataset, 'event_start_idx') or not hasattr(dataset, 'event_run_ids'):
        return
    total = len(dataset.data_list)
    for event_idx, start_idx in enumerate(dataset.event_start_idx):
        end_idx = dataset.event_start_idx[event_idx + 1] if event_idx + 1 < len(dataset.event_start_idx) else total
        run_id = int(dataset.event_run_ids[event_idx])
        for sample_idx in range(start_idx, end_idx):
            dataset.data_list[sample_idx].run_id = torch.tensor(run_id, dtype=torch.long)
            dataset.data_list[sample_idx].event_idx = torch.tensor(event_idx, dtype=torch.long)


def build_cluster_partition_map(dataset, num_clusters: int, config: Dict, logger: Logger):
    """Partition the graph(s) for Cluster-GCN.

    HEC-RAS has one shared mesh, so a single partition tensor is returned. mSWE-GNN has one
    mesh per event, so a {run_id: partition_map} dict is returned instead; maps are cached on
    disk per (run_id, num_clusters) because METIS partitioning is expensive to redo.
    """
    dataset_type = get_dataset_type(config)
    if dataset_type != 'mswegnn':
        logger.log("Performing one-time graph partitioning...")
        constant_values_path = 'data/datasets/processed/constant_values.npz'
        constant_values = np.load(constant_values_path)
        edge_index = torch.from_numpy(constant_values['edge_index']).long()
        num_nodes = constant_values['static_nodes'].shape[0]
        partition_map = partition_graph(edge_index, num_nodes, num_clusters)
        logger.log(f"Partition complete: {partition_map.shape[0]} nodes assigned to {num_clusters} clusters")
        return partition_map

    if not hasattr(dataset, 'data_list'):
        raise TypeError("mSWE Cluster-GCN currently requires memory storage mode with data_list.")
    ensure_dataset_run_id_attrs(dataset)
    partition_dir = os.path.join(dataset.processed_dir, 'partitions')
    if os.getenv('HP_SEARCH_UNIQUE_CACHE', '0') == '1':
        job_id = os.getenv('SLURM_ARRAY_JOB_ID') or os.getenv('SLURM_JOB_ID') or 'manual'
        task_id = os.getenv('SLURM_ARRAY_TASK_ID') or '0'
        partition_dir = os.path.join(partition_dir, f'job{job_id}_task{task_id}')
    os.makedirs(partition_dir, exist_ok=True)
    partition_maps = {}
    logger.log(f"Building/loading per-event mSWE partitions in {partition_dir}")
    for event_idx, start_idx in enumerate(dataset.event_start_idx):
        run_id = int(dataset.event_run_ids[event_idx])
        graph = dataset.data_list[start_idx]
        num_nodes = int(graph.num_nodes)
        cache_path = os.path.join(partition_dir, f'run_{run_id}_clusters_{num_clusters}.pt')
        if os.path.exists(cache_path):
            part = torch.load(cache_path, map_location='cpu')
            logger.log(f"Loaded cached partition for run_id={run_id}: {num_nodes} nodes")
        else:
            part = partition_graph(graph.edge_index.cpu().long(), num_nodes, num_clusters)
            torch.save(part.cpu(), cache_path)
            logger.log(f"Saved partition for run_id={run_id}: {num_nodes} nodes -> {cache_path}")
        if int(part.numel()) != num_nodes:
            raise ValueError(f"Partition length mismatch for run_id={run_id}: {part.numel()} != {num_nodes}")
        partition_maps[run_id] = part.long()
    logger.log(f"Prepared {len(partition_maps)} per-event mSWE partition maps.")
    return partition_maps

# --- load_dataset function remains largely the same, prepares dataset objects ---
# --- It might need adjustment if validation also needs ClusterLoader ---
def load_dataset(config: Dict, args: Namespace, logger: Logger, use_cluster_gcn: bool) -> Tuple[FloodEventDataset, Optional[FloodEventDataset]]:
    dataset_parameters = config['dataset_parameters']
    root_dir = dataset_parameters['root_dir']
    train_dataset_parameters = dataset_parameters['training']
    loss_func_parameters = config['loss_func_parameters']
    dataset_type = get_dataset_type(config)
    base_datset_config = build_base_dataset_config(config, args, logger)

    dataset_summary_file = train_dataset_parameters['dataset_summary_file']
    event_stats_file = train_dataset_parameters['event_stats_file']
    # <<< CLUSTER-GCN MODIFICATION >>> (Force memory mode if using Cluster GCN)
    storage_mode = 'memory' if use_cluster_gcn else dataset_parameters['storage_mode']
    if use_cluster_gcn and dataset_parameters['storage_mode'] != 'memory':
        logger.log("Warning: Forcing 'memory' storage mode for Cluster-GCN compatibility.")

    train_config = config['training_parameters']
    early_stopping_patience = train_config['early_stopping_patience']

    # --- Determine if autoregressive training is enabled ---
    autoregressive_train_params = train_config.get('autoregressive', {})
    autoregressive_enabled = autoregressive_train_params.get('enabled', False)
    num_label_timesteps = autoregressive_train_params.get('total_num_timesteps', 1) # Default to 1 if not specified

    # --- Handle Validation Split ---
    # Validation is constructed after the train dataset so train-mode preprocessing
    # can create feature-normalization stats before validation loads in test mode.
    val_dataset = None
    val_dataset_config = None
    val_storage_mode = None
    if early_stopping_patience is not None:
        percent_validation = train_config['val_split_percent']
        assert percent_validation is not None, 'Validation split percentage must be specified if early stopping is used.'
        logger.log(f'Splitting dataset events with {percent_validation * 100}% for validation')
        train_summary_file, val_summary_file = train_utils.split_dataset_events(root_dir, dataset_summary_file, percent_validation)
        val_event_stats_file = val_summary_file.replace(os.path.basename(dataset_summary_file), event_stats_file) # Use basename

        # Config for Validation Dataset (usually not autoregressive for validation step)
        val_dataset_config = {
            'mode': 'test', # Use test mode for validation loading logic
            'dataset_summary_file': val_summary_file,
            'event_stats_file': val_event_stats_file,
            **base_datset_config,
            # Validation typically doesn't need num_label_timesteps unless tester uses it
        }
        logger.log(f'Prepared validation dataset configuration: {val_dataset_config}')
        # <<< CLUSTER-GCN MODIFICATION >>> (Validation dataset storage mode: try configured mode, fallback to disk on OOM)
        val_storage_mode = dataset_parameters.get('validation_storage_mode', storage_mode)
        if use_cluster_gcn and val_storage_mode != 'memory':
            logger.log(
                f"Using validation storage_mode='{val_storage_mode}' for Cluster-GCN to avoid eager validation RAM pressure."
            )

    else:
        # No validation split needed
        train_summary_file = dataset_summary_file


    # --- Config for Training Dataset ---
    #train_event_stats_file = os.path.basename(train_summary_file).replace(os.path.basename(dataset_summary_file), event_stats_file) # Use basename
    train_event_stats_file = train_summary_file.replace(dataset_summary_file, event_stats_file)

    train_dataset_config = {
        'mode': 'train',
        'dataset_summary_file': train_summary_file,
        'event_stats_file': train_event_stats_file,
        **base_datset_config,
    }
    if autoregressive_enabled:
        train_dataset_config['num_label_timesteps'] = num_label_timesteps # Add AR specific arg

    logger.log(f'Using training dataset configuration: {train_dataset_config}')
    train_dataset = dataset_factory(storage_mode, autoregressive=autoregressive_enabled, dataset_type=dataset_type, **train_dataset_config)

    if val_dataset_config is not None:
        logger.log(f'Using validation dataset configuration: {val_dataset_config}')
        try:
            val_dataset = dataset_factory(val_storage_mode, autoregressive=False, dataset_type=dataset_type, **val_dataset_config) # Val dataset usually not AR itself
        except Exception as e:
            logger.log(f"Warning: Failed to load validation dataset with storage_mode='{val_storage_mode}': {e}")
            # Fallback to disk-backed loading to avoid memory allocation failures
            fallback_mode = dataset_parameters.get('validation_storage_mode', 'disk') if not use_cluster_gcn else 'disk'
            logger.log(f"Attempting to load validation dataset using fallback storage_mode='{fallback_mode}'")
            try:
                val_dataset = dataset_factory(fallback_mode, autoregressive=False, dataset_type=dataset_type, **val_dataset_config)
                logger.log("Loaded validation dataset using fallback disk-backed mode.")
            except Exception:
                logger.log(f"Failed to load validation dataset in fallback mode:\n{traceback.format_exc()}")
                raise

    ensure_dataset_run_id_attrs(train_dataset)
    if val_dataset is not None:
        ensure_dataset_run_id_attrs(val_dataset)

    logger.log(f'Loaded train dataset with {len(train_dataset)} samples.')
    if val_dataset:
        logger.log(f'Loaded validation dataset with {len(val_dataset)} samples.')

    return train_dataset, val_dataset


# --- run_train function remains largely the same, prepares and calls trainer ---
# def run_train(model: torch.nn.Module,
#               model_name: str,
#               # <<< CLUSTER-GCN MODIFICATION >>> (Accept loader instead of dataset)
#               train_loader: torch.utils.data.DataLoader, # Accepts ClusterLoader or DataLoader
#               logger: Logger,
#               config: Dict,
#               val_dataset: Optional[FloodEventDataset] = None, # Keep val_dataset for trainer validation logic
#               stats_dir: Optional[str] = None,
#               model_dir: Optional[str] = None,
#               device: str = 'cpu',
#               use_cluster_gcn: bool = False) -> str: # <<< CLUSTER-GCN MODIFICATION >>>
#         train_config = config['training_parameters']

#         # Loss function and optimizer
#         optimizer = torch.optim.Adam(model.parameters(), lr=train_config['learning_rate'], weight_decay=train_config['adam_weight_decay'])
#         logger.log(f'Using Adam optimizer with learning rate {train_config["learning_rate"]} and weight decay {train_config["adam_weight_decay"]}')

#         base_trainer_params = train_utils.get_trainer_config(model_name, config, logger)
#         trainer_params = {
#             'model': model,
#             'dataloader': train_loader, # <<< CLUSTER-GCN MODIFICATION >>> (Pass the loader)
#             'val_dataset': val_dataset,
#             'optimizer': optimizer,
#             'logger': logger,
#             'device': device,
#             **base_trainer_params,
#         }

#         autoregressive_train_config = train_config.get('autoregressive', {})
#         autoregressive_enabled = autoregressive_train_config.get('enabled', False)

#         # <<< CLUSTER-GCN MODIFICATION >>> (Select the correct trainer class)
#         if use_cluster_gcn:
#             # Ensure the specific trainer class is imported and available
#             try:
#                 # Assuming ClusterDualAutoregressiveTrainer exists in training package
#                 from training import ClusterDualAutoregressiveTrainer
#                 logger.log("Using ClusterDualAutoregressiveTrainer.")
#                 trainer = ClusterDualAutoregressiveTrainer(**trainer_params)
#             except ImportError:
#                  raise ImportError("ClusterDualAutoregressiveTrainer not found. Make sure it's defined and imported.")
#             except Exception as e:
#                  raise RuntimeError(f"Error initializing ClusterDualAutoregressiveTrainer: {e}")
#         else:
#              logger.log("Using standard trainer factory.")
#              trainer = trainer_factory(model_name, autoregressive_enabled, **trainer_params)
#         # <<< END CLUSTER-GCN MODIFICATION >>>

#         trainer.train() # Call the train method of the selected trainer

#         trainer.print_stats_summary()

#         # Save training stats and model
#         curr_date_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
#         if stats_dir is not None:
#             if not os.path.exists(stats_dir):
#                 os.makedirs(stats_dir)

#             saved_metrics_path = os.path.join(stats_dir, f'{model_name}_{curr_date_str}_train_stats.npz')
#             trainer.save_stats(saved_metrics_path)

#         model_path = f'{model_name}_{curr_date_str}.pt'
#         if model_dir is not None:
#             if not os.path.exists(model_dir):
#                 os.makedirs(model_dir)

#             model_path = os.path.join(model_dir, f'{model_name}_{curr_date_str}.pt')
#             trainer.save_model(model_path)

#         return model_path

def run_train(model: torch.nn.Module,
              model_name: str,
              train_dataset: FloodEventDataset,
              logger: Logger,
              config: Dict,
              val_dataset: Optional[FloodEventDataset] = None,
              stats_dir: Optional[str] = None,
              model_dir: Optional[str] = None,
              device: str = 'cpu',
              partition_map: Optional[torch.Tensor] = None,
              use_cluster_gcn: bool = False,
              clusters_per_batch: Optional[int] = 5,
              sliding: bool = True,
              batching_strategy: str = "same_run",
              amp_mode: str = "none",
              fused_adam: bool = False,
              wandb_run=None) -> str:
        train_config = config['training_parameters']

        # Loss function and optimizer
        adam_kwargs = {
            "lr": train_config["learning_rate"],
            "weight_decay": train_config["adam_weight_decay"],
        }
        if fused_adam and device == "cuda":
            try:
                optimizer = torch.optim.Adam(model.parameters(), fused=True, **adam_kwargs)
                logger.log("Using fused Adam optimizer.")
            except TypeError as e:
                logger.log(f"Fused Adam is not supported by this PyTorch build; falling back to standard Adam. Error: {e}")
                optimizer = torch.optim.Adam(model.parameters(), **adam_kwargs)
            except RuntimeError as e:
                logger.log(f"Fused Adam failed to initialize; falling back to standard Adam. Error: {e}")
                optimizer = torch.optim.Adam(model.parameters(), **adam_kwargs)
        else:
            if fused_adam:
                logger.log("Fused Adam requested but device is not CUDA; using standard Adam.")
            optimizer = torch.optim.Adam(model.parameters(), **adam_kwargs)
        logger.log(f'Using Adam optimizer with learning rate {train_config["learning_rate"]} and weight decay {train_config["adam_weight_decay"]}')

        base_trainer_params = train_utils.get_trainer_config(model_name, config, logger)
        if batching_strategy not in {"same_run", "cross_event"}:
            raise ValueError(f"Unsupported batching_strategy={batching_strategy}")
        same_run_batching = use_cluster_gcn and get_dataset_type(config) == 'mswegnn' and batching_strategy == "same_run"
        cross_event_batching = use_cluster_gcn and get_dataset_type(config) == 'mswegnn' and batching_strategy == "cross_event"
        trainer_params = {
            'model': model,
            'dataset': train_dataset,
            'val_dataset': val_dataset,
            'optimizer': optimizer,
            'logger': logger,
            'device': device,
            'same_run_batching': same_run_batching,
            'amp_mode': amp_mode,
            'wandb_run': wandb_run,
            **base_trainer_params,
        }
        if same_run_batching:
            logger.log("Enabled same-run batching for mSWE Cluster-GCN.")
        if cross_event_batching:
            logger.log("Enabled cross-event batching for mSWE Cluster-GCN.")
        logger.log(f"Trainer parameters prepared: {trainer_params.keys()}")
        if use_cluster_gcn:
            logger.log("Using ClusterDualAutoregressiveTrainer with Cluster-GCN strategy.")
            trainer_params['partition_map'] = partition_map
            trainer_params['clusters_per_batch'] = clusters_per_batch
            trainer_params['sliding'] = sliding
            trainer_params['cross_event_batching'] = cross_event_batching

        autoregressive_train_config = train_config['autoregressive']
        autoregressive_enabled = autoregressive_train_config.get('enabled', False)
        trainer = trainer_factory(model_name, autoregressive_enabled, isCluster=use_cluster_gcn, **trainer_params)
        trainer.train()

        trainer.print_stats_summary()

        # Save training stats and model
        curr_date_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if stats_dir is not None:
            if not os.path.exists(stats_dir):
                os.makedirs(stats_dir)

            saved_metrics_path = os.path.join(stats_dir, f'{model_name}_{curr_date_str}_train_stats.npz')
            trainer.save_stats(saved_metrics_path)

        model_path = f'{model_name}_{curr_date_str}.pt'
        if model_dir is not None:
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            model_path = os.path.join(model_dir, f'{model_name}_{curr_date_str}.pt')
            trainer.save_model(model_path)

        return model_path


def main():
    print("Starting main function...", flush=True)
    args = parse_args()
    config = file_utils.read_yaml_file(args.config)

    train_config = config['training_parameters']
    log_path = train_config['log_path']
    logger = Logger(log_path=log_path)
    wandb_run = None

    try:
        logger.log('================================================')

        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            logger.log(f'Setting random seed to {args.seed}')

        current_device = torch.cuda.get_device_name(args.device) if args.device != 'cpu' else 'CPU'
        logger.log(f'Using device: {current_device}')
        logger.log(
            f'Optimization settings: amp={args.amp}, compile_model={args.compile_model}, '
            f'fused_adam={args.fused_adam}, batching_strategy={args.batching_strategy}'
        )
        wandb_run = init_wandb_run(args, config, logger)

        # Dataset
        use_cluster_gcn = args.use_cluster_gcn
        train_dataset, val_dataset = load_dataset(config, args, logger, use_cluster_gcn)
        logger.log(f"use cluster {use_cluster_gcn}")
        logger.log(f"cluster sliding mode: {args.sliding}")
        logger.log(
            f'Dataset feature dimensions: static_node={train_dataset.num_static_node_features}, '
            f'dynamic_node={train_dataset.num_dynamic_node_features}, '
            f'static_edge={train_dataset.num_static_edge_features}, '
            f'dynamic_edge={train_dataset.num_dynamic_edge_features}, '
            f'boundary_aware_features={config.get("dataset_parameters", {}).get("boundary_aware_features", False)}'
        )
        # Partition graph once before training. mSWE uses one partition map per event topology.
        partition_map = None
        if use_cluster_gcn:
            partition_map = build_cluster_partition_map(train_dataset, args.num_clusters, config, logger)

        # Model
        model_params = config['model_parameters'][args.model]
        base_model_params = {
            'static_node_features': train_dataset.num_static_node_features,
            'dynamic_node_features': train_dataset.num_dynamic_node_features,
            'static_edge_features': train_dataset.num_static_edge_features,
            'dynamic_edge_features': train_dataset.num_dynamic_edge_features,
            'previous_timesteps': train_dataset.previous_timesteps,
            'device': args.device,
        }
        model_config = {**model_params, **base_model_params}
        model = model_factory(args.model, **model_config)
        logger.log(f'Using model: {args.model}')
        logger.log(f'Using model configuration: {model_config}')
        num_train_params = model.get_model_size()
        logger.log(f'Number of trainable model parameters: {num_train_params}')

        checkpoint_path = train_config.get('checkpoint_path', None)
        if checkpoint_path is not None:
            logger.log(f'Loading model from checkpoint: {checkpoint_path}')
            model.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location=args.device))

        if args.compile_model:
            if args.device != "cuda":
                logger.log("torch.compile requested on non-CUDA device; compiling anyway may not improve performance.")
            logger.log("Compiling model with torch.compile(mode='reduce-overhead', dynamic=True).")
            model = torch.compile(model, mode="reduce-overhead", dynamic=True)

        stats_dir = train_config['stats_dir']
        model_dir = train_config['model_dir']
        model_path = run_train(model=model,
                               model_name=args.model,
                               train_dataset=train_dataset,
                               val_dataset=val_dataset,
                               logger=logger,
                               config=config,
                               stats_dir=stats_dir,
                               model_dir=model_dir,
                               device=args.device,
                               partition_map=partition_map,
                               use_cluster_gcn=use_cluster_gcn,
                               clusters_per_batch=args.clusters_per_batch,
                               sliding=args.sliding,
                               batching_strategy=args.batching_strategy,
                               amp_mode=args.amp,
                               fused_adam=args.fused_adam,
                               wandb_run=wandb_run)
        if wandb_run is not None:
            wandb_run.summary['model_path'] = model_path

        logger.log('================================================')

        if not args.with_test:
            return

        # =================== Testing ===================
        logger.log(f'Starting testing for model: {model_path}')

        dataset_parameters = config['dataset_parameters']
        dataset_type = get_dataset_type(config)
        base_datset_config = build_base_dataset_config(config, args, logger)
        base_datset_config['force_reload'] = True
        test_dataset_config = get_test_dataset_config(base_datset_config, config)
        logger.log(f'Using test dataset configuration: {test_dataset_config}')

        # Clear memory before loading test dataset
        del train_dataset
        if val_dataset is not None:
            del val_dataset
        gc.collect()
        if args.device == 'cuda':
            torch.cuda.empty_cache()

        storage_mode = dataset_parameters['storage_mode']
        dataset = dataset_factory(storage_mode, autoregressive=False, dataset_type=dataset_type, **test_dataset_config)
        ensure_dataset_run_id_attrs(dataset)
        logger.log(f'Loaded test dataset with {len(dataset)} samples')
        logger.log(
            'Test dataset feature dimensions: '
            f'static_node={dataset.num_static_node_features}, '
            f'dynamic_node={dataset.num_dynamic_node_features}, '
            f'static_edge={dataset.num_static_edge_features}, '
            f'dynamic_edge={dataset.num_dynamic_edge_features}, '
            f'boundary_aware_features={dataset_parameters.get("boundary_aware_features", False)}'
        )

        logger.log(f'Using model checkpoint for {args.model}: {model_path}')
        logger.log(f'Using model configuration: {model_config}')

        test_config = config['testing_parameters']
        rollout_start = test_config['rollout_start']
        rollout_timesteps = test_config['rollout_timesteps']
        output_dir = test_config['output_dir']
        include_global_mass_loss = bool(
            config.get('loss_func_parameters', {}).get('use_global_mass_loss', True)
        )
        include_local_mass_loss = bool(
            config.get('loss_func_parameters', {}).get('use_local_mass_loss', True)
        )
        run_test(model=model,
                 model_path=model_path,
                 dataset=dataset,
                 logger=logger,
                 rollout_start=rollout_start,
                 rollout_timesteps=rollout_timesteps,
                 output_dir=output_dir,
                 device=args.device,
                 include_global_mass_loss=include_global_mass_loss,
                 include_local_mass_loss=include_local_mass_loss)
        metrics_prefix = os.path.splitext(os.path.basename(model_path))[0]
        log_test_metrics_to_wandb(wandb_run, output_dir, logger, metrics_prefix=metrics_prefix)
        if args.collect_regression_metrics:
            metrics_glob = f"{metrics_prefix}_runid_*_test_metrics.npz"
            collect_regression_metrics(args, config, logger, output_dir, wandb_run=wandb_run, metrics_glob=metrics_glob)

        logger.log('================================================')

    except Exception:
        logger.log(f'Unexpected error:\n{traceback.format_exc()}')
        raise
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == '__main__':
    main()
