import os
import copy
import struct
import tempfile
import numpy as np
import pandas as pd
import optuna

from data import dataset_factory, FloodEventDataset
from models import model_factory
from models.base_model import BaseModel
from typing import Dict, List, Tuple, Optional
from utils import Logger, file_utils

TEMP_DIR_NAME = 'hp_search_cross_val'
dataset_cache = {}

def _resolve_raw_path(raw_dir_path: str, relative_path: str) -> str:
    path = str(relative_path)
    return path if os.path.isabs(path) else os.path.join(raw_dir_path, path)


def _read_peak_inflow(raw_dir_path: str, hydrograph_file: str) -> float:
    values = np.loadtxt(_resolve_raw_path(raw_dir_path, hydrograph_file), ndmin=2)
    if values.shape[1] < 2 or values.shape[0] == 0:
        raise ValueError(f'Invalid hydrograph file: {hydrograph_file}')
    peak = float(np.nanmax(values[:, 1]))
    if not np.isfinite(peak):
        raise ValueError(f'Non-finite peak inflow in hydrograph file: {hydrograph_file}')
    return peak


def _read_dbf_record_count(raw_dir_path: str, nodes_shp_file: str) -> int:
    dbf_path = os.path.splitext(_resolve_raw_path(raw_dir_path, nodes_shp_file))[0] + '.dbf'
    with open(dbf_path, 'rb') as dbf_file:
        header = dbf_file.read(8)
    if len(header) != 8:
        raise ValueError(f'Invalid DBF header: {dbf_path}')
    return int(struct.unpack('<I', header[4:8])[0])


def _build_fold_manifest(summary_df: pd.DataFrame,
                         raw_dir_path: str,
                         num_folds: int,
                         seed: int) -> pd.DataFrame:
    """Assign events to folds, stratified by peak inflow and balanced by graph size.

    Contiguous index slicing put hydraulically similar events in the same fold. Here events
    are sorted by peak inflow (read from the hydrograph) and dealt out one per fold from each
    block, with larger graphs (node count read from the shapefile's .dbf header) going to the
    folds holding fewest nodes so far. The result is seeded and cached so folds are stable
    across workers and reruns.
    """
    required_columns = {'Run_ID', 'Hydrograph_Filepath', 'Nodes_Shp_Filepath'}
    missing_columns = required_columns.difference(summary_df.columns)
    if missing_columns:
        raise ValueError(f'Missing columns required for stratified folds: {sorted(missing_columns)}')
    if summary_df['Run_ID'].duplicated().any():
        duplicates = summary_df.loc[summary_df['Run_ID'].duplicated(), 'Run_ID'].tolist()
        raise ValueError(f'Duplicate Run_ID values in development manifest: {duplicates}')

    manifest = summary_df[['Run_ID']].copy()
    manifest['peak_inflow'] = [
        _read_peak_inflow(raw_dir_path, path)
        for path in summary_df['Hydrograph_Filepath']
    ]
    manifest['num_nodes'] = [
        _read_dbf_record_count(raw_dir_path, path)
        for path in summary_df['Nodes_Shp_Filepath']
    ]

    # Consecutive peak-flow blocks contain hydraulically similar events. Give
    # each fold one event from each block, assigning larger graphs to folds
    # with fewer accumulated nodes.
    rng = np.random.default_rng(seed)
    manifest['_tie_breaker'] = rng.random(len(manifest))
    ordered = manifest.sort_values(['peak_inflow', '_tie_breaker']).reset_index()
    fold_node_totals = np.zeros(num_folds, dtype=np.int64)
    fold_event_counts = np.zeros(num_folds, dtype=np.int64)
    fold_by_original_index = {}
    for start_idx in range(0, len(ordered), num_folds):
        block = ordered.iloc[start_idx:start_idx + num_folds].sort_values(
            ['num_nodes', '_tie_breaker'], ascending=[False, True]
        )
        fold_order = sorted(
            range(num_folds),
            key=lambda fold_idx: (fold_event_counts[fold_idx], fold_node_totals[fold_idx], fold_idx),
        )
        for (_, event), fold_idx in zip(block.iterrows(), fold_order):
            fold_by_original_index[int(event['index'])] = int(fold_idx + 1)
            fold_event_counts[fold_idx] += 1
            fold_node_totals[fold_idx] += int(event['num_nodes'])

    manifest['fold_id'] = [fold_by_original_index[idx] for idx in manifest.index]
    return manifest.drop(columns=['_tie_breaker'])


def _write_manifest_atomically(manifest: pd.DataFrame, manifest_path: str) -> None:
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    temp_fd, temp_path = tempfile.mkstemp(
        prefix=f'.{os.path.basename(manifest_path)}.',
        dir=os.path.dirname(manifest_path),
        text=True,
    )
    os.close(temp_fd)
    try:
        manifest.to_csv(temp_path, index=False)
        os.replace(temp_path, manifest_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def create_explicit_train_val_dataset_files(root_dir: str,
                                            train_summary_file: str,
                                            val_summary_file: str,
                                            group_id: str = 'split') -> Tuple[List[str], List[str]]:
    raw_dir_path = os.path.join(root_dir, 'raw')
    processed_dir_path = os.path.join(root_dir, 'processed')
    temp_dir_paths = file_utils.create_temp_dirs(paths=[raw_dir_path, processed_dir_path],
                                                 folder_name=TEMP_DIR_NAME)
    raw_temp_dir_path = temp_dir_paths[0]

    for summary_file, prefix in ((train_summary_file, 'train'), (val_summary_file, 'val')):
        src_path = os.path.join(raw_dir_path, summary_file)
        if not os.path.exists(src_path):
            raise FileNotFoundError(f'{prefix} summary file does not exist: {src_path}')
        summary_df = pd.read_csv(src_path)
        if summary_df.empty:
            raise ValueError(f'{prefix} summary file is empty: {src_path}')
        summary_df.to_csv(os.path.join(raw_temp_dir_path, f'{prefix}_{group_id}.csv'), index=False)

    return [group_id], temp_dir_paths


def create_cross_val_dataset_files(root_dir: str,
                                   dataset_summary_file: str,
                                   num_folds: int,
                                   fold_seed: int = 42,
                                   manifest_dir: str = 'cv_manifests') -> Tuple[List[str], List[str], str]:
    raw_dir_path = os.path.join(root_dir, 'raw')
    processed_dir_path = os.path.join(root_dir, 'processed')
    temp_dir_paths = file_utils.create_temp_dirs(paths=[raw_dir_path, processed_dir_path],
                                                 folder_name=TEMP_DIR_NAME)

    dataset_summary_path = os.path.join(raw_dir_path, dataset_summary_file)

    assert os.path.exists(dataset_summary_path), f'Dataset summary file does not exist: {dataset_summary_path}'
    summary_df = pd.read_csv(dataset_summary_path)
    assert len(summary_df) > 0, f'No data found in summary file: {dataset_summary_path}'
    if num_folds < 2:
        raise ValueError('Proper cross-validation requires num_folds >= 2.')
    assert len(summary_df) >= num_folds, f'Number of flood events ({len(summary_df)}) must be greater than or equal to number of folds ({num_folds})'

    raw_temp_dir_path = temp_dir_paths[0]

    manifest_path = os.path.join(
        raw_dir_path,
        manifest_dir,
        f'{os.path.splitext(os.path.basename(dataset_summary_file))[0]}_cv{num_folds}_seed{fold_seed}.csv',
    )
    if os.path.exists(manifest_path):
        manifest = pd.read_csv(manifest_path)
        expected_run_ids = set(summary_df['Run_ID'].astype(int))
        if set(manifest['Run_ID'].astype(int)) != expected_run_ids:
            raise ValueError(f'Existing fold manifest does not match development events: {manifest_path}')
    else:
        manifest = _build_fold_manifest(summary_df, raw_dir_path, num_folds, fold_seed)
        _write_manifest_atomically(manifest, manifest_path)

    fold_by_run_id = dict(zip(manifest['Run_ID'].astype(int), manifest['fold_id'].astype(int)))
    summary_df = summary_df.copy()
    summary_df['fold_id'] = summary_df['Run_ID'].astype(int).map(fold_by_run_id)
    if summary_df['fold_id'].isna().any():
        raise ValueError('At least one development event was not assigned to a fold.')

    groups = [f'fold{fold_idx}' for fold_idx in range(1, num_folds + 1)]
    for fold_idx, group_id in enumerate(groups, start=1):
        val_rows = summary_df[summary_df['fold_id'] == fold_idx].drop(columns=['fold_id'])
        train_rows = summary_df[summary_df['fold_id'] != fold_idx].drop(columns=['fold_id'])
        if val_rows.empty or train_rows.empty:
            raise ValueError(f'Invalid split for {group_id}: train={len(train_rows)}, val={len(val_rows)}')
        train_rows.to_csv(os.path.join(raw_temp_dir_path, f'train_{group_id}.csv'), index=False)
        val_rows.to_csv(os.path.join(raw_temp_dir_path, f'val_{group_id}.csv'), index=False)

    return groups, temp_dir_paths, manifest_path

def suggest_hyperparamters(trial: optuna.Trial, hyperparameters: Dict, config: Dict, logger: Logger = None) -> Dict:
    updated_config = copy.deepcopy(config)
    suggested_values = {}
    for param_name, param_info in hyperparameters.items():
        # A parameter may declare a condition on an earlier one, e.g. only suggest
        # learning_rate_decay when init_num_timesteps leaves room for curriculum steps.
        # Skipping it keeps the trial out of a region where the value has no effect.
        condition = param_info.get('condition')
        if condition is not None:
            condition_param = condition['param']
            condition_values = condition.get('values', [True])
            if suggested_values.get(condition_param) not in condition_values:
                if logger is not None:
                    logger.log(
                        f"Skipping parameter {param_name} because condition "
                        f"{condition_param}={suggested_values.get(condition_param)} "
                        f"is not in {condition_values}"
                    )
                continue

        param_type = param_info['type']
        if param_type == 'int':
            suggested_value = trial.suggest_int(param_name, param_info['min'], param_info['max'], step=param_info.get('step', 1), log=param_info.get('log', False))
        elif param_type == 'float':
            suggested_value = trial.suggest_float(param_name, param_info['min'], param_info['max'], step=param_info.get('step', None), log=param_info.get('log', False))
        elif param_type == 'categorical':
            suggested_value = trial.suggest_categorical(param_name, param_info['choices'])
        else:
            raise ValueError(f'Unsupported hyperparameter type: {param_type} for parameter: {param_name}')

        if logger is not None:
            logger.log(f'Testing value {suggested_value} for parameter {param_name}')
        suggested_values[param_name] = suggested_value

        # Traverse the config dictionary to set the suggested value
        path = param_info['path']
        keys = path.split('.')
        d = updated_config
        for key in keys[:-1]:
            if key not in d:
                raise KeyError(f'Key {key} not found in configuration path: {path}')
            d = d[key]
        d[keys[-1]] = suggested_value

    return updated_config

def load_datasets(group_id: str, config: Dict, logger: Logger, load_test: bool = True) -> Tuple[FloodEventDataset, Optional[FloodEventDataset], Optional[FloodEventDataset]]:
    train_config = config['training_parameters']
    early_stopping_patience = train_config['early_stopping_patience']

    dataset_parameters = config['dataset_parameters']
    loss_func_parameters = config['loss_func_parameters']
    features_stats_file = os.path.join(TEMP_DIR_NAME, f'features_stats_{group_id}.yaml')
    with_global_mass_loss = loss_func_parameters['use_global_mass_loss']
    with_local_mass_loss = loss_func_parameters['use_local_mass_loss']
    prepare_local_mass_for_search = bool(
        dataset_parameters.get('prepare_local_mass_for_search', with_local_mass_loss)
    )
    prepare_global_mass_for_search = bool(
        dataset_parameters.get('prepare_global_mass_for_search', with_global_mass_loss)
    )
    storage_mode = dataset_parameters['storage_mode']
    boundary_aware_cache_prefix = dataset_parameters.get('boundary_aware_cache_prefix', None)
    if os.getenv('HP_SEARCH_UNIQUE_CACHE', '0') == '1' and boundary_aware_cache_prefix is not None:
        job_id = os.getenv('SLURM_ARRAY_JOB_ID') or os.getenv('SLURM_JOB_ID') or 'manual'
        task_id = os.getenv('SLURM_ARRAY_TASK_ID') or '0'
        previous_timesteps = dataset_parameters.get('previous_timesteps', 'default')
        boundary_aware_cache_prefix = (
            f'{boundary_aware_cache_prefix}job{job_id}_task{task_id}_pt{previous_timesteps}_'
        )
    # Cache key must cover every dataset-shaping option a trial can vary, not just the fold
    # id, or a trial would silently reuse a dataset built for a different feature layout.
    cache_signature = '_'.join([
        f'local{int(prepare_local_mass_for_search)}',
        f'global{int(prepare_global_mass_for_search)}',
        f'target{dataset_parameters.get("prediction_target_mode", "default")}',
        f'prev{dataset_parameters.get("previous_timesteps", "default")}',
        f'ba{int(bool(dataset_parameters.get("boundary_aware_features", False)))}',
        str(boundary_aware_cache_prefix or 'cache'),
    ])
    train_key = f'train_{group_id}_{cache_signature}'
    test_key = f'test_{group_id}_{cache_signature}'
    val_key = f'val_{group_id}_{cache_signature}'
    if train_key in dataset_cache and (not load_test or test_key in dataset_cache):
        train_dataset = dataset_cache[train_key]
        test_dataset = dataset_cache[test_key] if load_test else None
        if early_stopping_patience is not None:
            if val_key not in dataset_cache:
                raise ValueError(f'Validation dataset for group {group_id} not found in cache.')
            val_dataset = dataset_cache[val_key]
            return train_dataset, test_dataset, val_dataset
        return train_dataset, test_dataset, None

    base_dataset_config = {
        'root_dir': dataset_parameters['root_dir'],
        'dataset_type': dataset_parameters.get('dataset_type', 'hecras'),
        'nodes_shp_file': dataset_parameters.get('nodes_shp_file'),
        'edges_shp_file': dataset_parameters.get('edges_shp_file'),
        'features_stats_file': features_stats_file,
        'previous_timesteps': dataset_parameters['previous_timesteps'],
        'normalize': dataset_parameters['normalize'],
        'boundary_aware_features': dataset_parameters.get('boundary_aware_features', False),
        'boundary_aware_feature_groups': dataset_parameters.get('boundary_aware_feature_groups', None),
        'boundary_aware_cache_prefix': boundary_aware_cache_prefix,
        'timestep_interval': dataset_parameters['timestep_interval'],
        'spin_up_time': dataset_parameters['spin_up_time'],
        'time_from_peak': dataset_parameters['time_from_peak'],
        'inflow_boundary_nodes': dataset_parameters['inflow_boundary_nodes'],
        'outflow_boundary_nodes': dataset_parameters['outflow_boundary_nodes'],
        'logger': logger,
        'force_reload': False,
    }
    for optional_key in (
        'prediction_target_mode',
        'node_target_feature',
        'edge_target_feature',
        'edge_unit_discharge_mode',
    ):
        if optional_key in dataset_parameters:
            base_dataset_config[optional_key] = dataset_parameters[optional_key]
    base_dataset_config = {k: v for k, v in base_dataset_config.items() if v is not None}

    train_summary_file = os.path.join(TEMP_DIR_NAME, f'train_{group_id}.csv')
    train_event_stats_file = os.path.join(TEMP_DIR_NAME, f'train_event_stats_{group_id}.yaml')
    train_dataset_config = {
        **base_dataset_config,
        'mode': 'train',
        'dataset_summary_file': train_summary_file,
        'event_stats_file': train_event_stats_file,
        'with_global_mass_loss': prepare_global_mass_for_search,
        'with_local_mass_loss': prepare_local_mass_for_search,
    }
    autoregressive_train_params = train_config['autoregressive']
    autoregressive_enabled = autoregressive_train_params.get('enabled', False)
    if autoregressive_enabled:
        train_dataset_config.update({
            'num_label_timesteps': autoregressive_train_params['total_num_timesteps'],
        })
    logger.log(f'Using train dataset configuration: {train_dataset_config}')
    train_dataset = dataset_factory(storage_mode, autoregressive=autoregressive_enabled, **train_dataset_config)
    dataset_cache[train_key] = train_dataset
    logger.log(f'Loaded train dataset with {len(train_dataset)} samples')

    test_dataset = None
    if load_test:
        test_summary_file = os.path.join(TEMP_DIR_NAME, f'test_{group_id}.csv')
        test_event_stats_file = os.path.join(TEMP_DIR_NAME, f'test_event_stats_{group_id}.yaml')
        test_dataset_config = {
            **base_dataset_config,
            'mode': 'test',
            'dataset_summary_file': test_summary_file,
            'event_stats_file': test_event_stats_file,
            # Exclude computation of physics loss for hyperparameter search
            'with_global_mass_loss': False,
            'with_local_mass_loss': prepare_local_mass_for_search,
        }
        logger.log(f'Using test dataset configuration: {test_dataset_config}')
        test_dataset = dataset_factory(storage_mode, autoregressive=False, **test_dataset_config)
        dataset_cache[test_key] = test_dataset
        logger.log(f'Loaded test dataset with {len(test_dataset)} samples')

    if early_stopping_patience is None:
        return train_dataset, test_dataset, None

    val_summary_file = os.path.join(TEMP_DIR_NAME, f'val_{group_id}.csv')
    val_event_stats_file = os.path.join(TEMP_DIR_NAME, f'val_event_stats_{group_id}.yaml')
    val_dataset_config = {
        **base_dataset_config,
        'mode': 'test',
        'dataset_summary_file': val_summary_file,
        'event_stats_file': val_event_stats_file,
        # Exclude computation of physics loss for hyperparameter search
        'with_global_mass_loss': False,
        'with_local_mass_loss': prepare_local_mass_for_search,
    }
    logger.log(f'Using validation dataset configuration: {val_dataset_config}')
    val_dataset = dataset_factory(storage_mode, autoregressive=False, **val_dataset_config)
    dataset_cache[val_key] = val_dataset
    logger.log(f'Loaded validation dataset with {len(val_dataset)} samples')

    return train_dataset, test_dataset, val_dataset

def load_model(model_name: str, config: Dict, dataset: FloodEventDataset, device: str) -> BaseModel:
    model_params = config['model_parameters'][model_name]
    base_model_params = {
        'static_node_features': dataset.num_static_node_features,
        'dynamic_node_features': dataset.num_dynamic_node_features,
        'static_edge_features': dataset.num_static_edge_features,
        'dynamic_edge_features': dataset.num_dynamic_edge_features,
        'previous_timesteps': dataset.previous_timesteps,
        'device': device,
    }
    model_config = {**model_params, **base_model_params}
    model = model_factory(model_name, **model_config)
    return model
