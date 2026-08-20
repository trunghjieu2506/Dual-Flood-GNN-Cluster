import os
import numpy as np

from numpy import ndarray
from .base_flood_event_dataset import FloodEventDataset
from .base_boundary_condition import BoundaryCondition
from data.shp_data_retrieval import get_cell_elevation, get_cell_position_x, get_cell_position_y,\
    get_cell_position, get_edge_index, get_edge_length, get_edge_slope
from data.dem_data_retrieval import get_filled_dem, get_aspect, get_curvature, get_flow_accumulation
from typing import Callable, List, Literal, Tuple

from .hydrograph_data_retrieval import get_event_timesteps, get_inflow_hydrograph
from .mswegnn_boundary_condition import mSWEGNNBoundaryCondition
from .nc_data_retrieval import get_face_flow, get_water_depth
from .shp_data_retrieval import get_node_types, get_edge_types, get_cell_area, get_face_length

class mSWEGNNFloodEventDataset(FloodEventDataset):
    EVENT_FILE_KEYS = [*FloodEventDataset.EVENT_FILE_KEYS, 'Cells_Shp_Filepath', 'Hydrograph_Filepath']
    BOUNDARY_AWARE_STATIC_NODE_FEATURES = [
        'is_inflow_boundary_node',
        'is_domain_boundary_node',
        'distance_to_inflow',
        'distance_to_boundary',
    ]
    BOUNDARY_AWARE_STATIC_EDGE_FEATURES = [
        'is_inflow_edge',
        'is_wall_boundary_edge',
        'is_boundary_edge',
        'boundary_normal_x',
        'boundary_normal_y',
    ]
    BOUNDARY_AWARE_DYNAMIC_EDGE_FEATURES = ['bc_flow']
    DEFAULT_BOUNDARY_AWARE_FEATURE_GROUPS = {
        'node_type_flags': True,
        'node_distances': True,
        'edge_type_flags': True,
        'edge_boundary_normals': True,
        'edge_bc_flow': True,
    }
    BOUNDARY_AWARE_FEATURES_BY_GROUP = {
        'node_type_flags': {
            'static_node': ['is_inflow_boundary_node', 'is_domain_boundary_node'],
            'static_edge': [],
            'dynamic_edge': [],
        },
        'node_distances': {
            'static_node': ['distance_to_inflow', 'distance_to_boundary'],
            'static_edge': [],
            'dynamic_edge': [],
        },
        'edge_type_flags': {
            'static_node': [],
            'static_edge': ['is_inflow_edge', 'is_wall_boundary_edge', 'is_boundary_edge'],
            'dynamic_edge': [],
        },
        'edge_boundary_normals': {
            'static_node': [],
            'static_edge': ['boundary_normal_x', 'boundary_normal_y'],
            'dynamic_edge': [],
        },
        'edge_bc_flow': {
            'static_node': [],
            'static_edge': [],
            'dynamic_edge': ['bc_flow'],
        },
    }

    def __init__(self, *args, **kwargs):
        self.boundary_aware_features = bool(kwargs.pop('boundary_aware_features', False))
        raw_feature_groups = kwargs.pop('boundary_aware_feature_groups', None)
        raw_cache_prefix = kwargs.pop('boundary_aware_cache_prefix', None)
        self.boundary_aware_feature_groups = self._resolve_boundary_aware_feature_groups(raw_feature_groups)
        self.boundary_aware_cache_prefix = raw_cache_prefix
        if self.boundary_aware_cache_prefix is None:
            self.boundary_aware_cache_prefix = 'boundary_aware_' if self.boundary_aware_features else ''

        self.enabled_boundary_aware_static_node_features = []
        self.enabled_boundary_aware_static_edge_features = []
        self.enabled_boundary_aware_dynamic_edge_features = []
        self._logged_boundary_summaries = set()
        if self.boundary_aware_features:
            self.enabled_boundary_aware_static_node_features = self._enabled_features('static_node')
            self.enabled_boundary_aware_static_edge_features = self._enabled_features('static_edge')
            self.enabled_boundary_aware_dynamic_edge_features = self._enabled_features('dynamic_edge')
            self.STATIC_NODE_FEATURES = [
                *FloodEventDataset.STATIC_NODE_FEATURES,
                *self.enabled_boundary_aware_static_node_features,
            ]
            self.STATIC_EDGE_FEATURES = [
                *FloodEventDataset.STATIC_EDGE_FEATURES,
                *self.enabled_boundary_aware_static_edge_features,
            ]
            self.DYNAMIC_EDGE_FEATURES = [
                *FloodEventDataset.DYNAMIC_EDGE_FEATURES,
                *self.enabled_boundary_aware_dynamic_edge_features,
            ]
        super().__init__(*args, **kwargs)
        self.time_from_peak = None  # Currently not implemented in mSWEGNNFloodEventDataset
        # Compatibility with original DualFlood/Cluster trainer and tester code.
        self.hec_ras_run_ids = self.event_run_ids
        self.boundary_condition = self.boundary_conditions[0] if self.boundary_conditions else None
        self.log_func(
            f'mSWE boundary-aware features enabled: {self.boundary_aware_features}. '
            f'Feature groups: {self.boundary_aware_feature_groups}. '
            f'Feature counts: static_node={self.num_static_node_features}, '
            f'dynamic_node={self.num_dynamic_node_features}, '
            f'static_edge={self.num_static_edge_features}, '
            f'dynamic_edge={self.num_dynamic_edge_features}'
        )

    def _resolve_boundary_aware_feature_groups(self, feature_groups):
        resolved = dict(self.DEFAULT_BOUNDARY_AWARE_FEATURE_GROUPS)
        if feature_groups is not None:
            unknown = set(feature_groups) - set(resolved)
            if unknown:
                raise ValueError(f'Unknown boundary_aware_feature_groups keys: {sorted(unknown)}')
            resolved.update({key: bool(value) for key, value in feature_groups.items()})
        return resolved

    def _enabled_features(self, feature_type: str):
        features = []
        for group_name, enabled in self.boundary_aware_feature_groups.items():
            if enabled:
                features.extend(self.BOUNDARY_AWARE_FEATURES_BY_GROUP[group_name][feature_type])
        return features

    @property
    def processed_file_names(self):
        event_files = []
        prefix = self.boundary_aware_cache_prefix if self.boundary_aware_features else ''
        for run_id in self.event_run_ids:
            event_files.extend([
                f'{prefix}static_values_event_{run_id}.npz',
                f'{prefix}dynamic_values_event_{run_id}.npz',
                f'boundary_condition_event_{run_id}.npz',
            ])
        return [
            self.event_stats_file,
            self.features_stats_file,
            *event_files,
        ]

    def _create_boundary_conditions(self, root_dir: str) -> List[BoundaryCondition]:
        bc_list = []
        for paths, run_id in zip(self.event_file_paths, self.event_run_ids):
            simulation_path, nodes_shp_path = paths[0], paths[1]
            npz_filename = f'boundary_condition_event_{run_id}.npz'
            bc = mSWEGNNBoundaryCondition(root_dir=root_dir,
                                          simulation_file=simulation_path,
                                          nodes_shp_file=nodes_shp_path,
                                          inflow_boundary_nodes=None, # Set within class
                                          outflow_boundary_nodes=None, # Set within class
                                          saved_npz_file=npz_filename)
            bc_list.append(bc)
        return bc_list

    def _set_event_properties(self):
        self._event_peak_idx = []
        self._event_base_timestep_interval = []
        self.event_start_idx = []

        current_total_ts = 0
        for event_idx in range(len(self.event_run_ids)):
            paths = self._get_event_file_paths(event_idx)

            timesteps = get_event_timesteps(paths[self.EVENT_FILE_KEYS[5]])
            event_ts_interval = int((timesteps[1] - timesteps[0]))
            assert self.timestep_interval % event_ts_interval == 0, f'Event {self.event_run_ids[event_idx]} has a timestep interval of {event_ts_interval} seconds, which is not compatible with the dataset timestep interval of {self.timestep_interval} seconds.'
            self._event_base_timestep_interval.append(event_ts_interval)

            # water_volume = get_water_volume(paths[self.EVENT_FILE_KEYS[0]])
            # total_water_volume = water_volume.sum(axis=1)
            # peak_idx = np.argmax(total_water_volume).item()
            # num_timesteps_after_peak = self.time_from_peak // event_ts_interval if self.time_from_peak is not None else 0
            # assert peak_idx + num_timesteps_after_peak < len(timesteps), "Timesteps after peak exceeds the available timesteps."
            # self._event_peak_idx.append(peak_idx)

            timesteps = self._get_trimmed_dynamic_data(timesteps, event_idx, aggr='first')
            trim_num_timesteps = len(timesteps)

            event_total_rollout_ts = trim_num_timesteps - self.previous_timesteps - 1  # First timestep starts at self.previous_timesteps; Last timestep is used for labels
            assert event_total_rollout_ts > 0, f'Event {event_idx} has too few timesteps.'
            self.event_start_idx.append(current_total_ts)

            current_total_ts += event_total_rollout_ts

        self.total_rollout_timesteps = current_total_ts

        # assert len(self._event_peak_idx) == len(self.event_run_ids), 'Mismatch in number of events and peak indices.'
        assert len(self.event_start_idx) == len(self.event_run_ids), 'Mismatch in number of events and start indices.'

    def _get_event_timesteps(self, event_idx: int) -> ndarray:
        paths = self._get_event_file_paths(event_idx)
        timesteps = get_event_timesteps(paths[self.EVENT_FILE_KEYS[5]])
        timesteps = self._get_trimmed_dynamic_data(timesteps, event_idx, aggr='first')
        return timesteps

    def _get_boundary_metadata(self, event_idx: int):
        paths = self._get_event_file_paths(event_idx)
        node_types = get_node_types(paths[self.EVENT_FILE_KEYS[1]])
        edge_types = get_edge_types(paths[self.EVENT_FILE_KEYS[2]])
        edge_index = get_edge_index(paths[self.EVENT_FILE_KEYS[2]])
        pos = get_cell_position(paths[self.EVENT_FILE_KEYS[1]])
        return node_types, edge_types, edge_index, pos

    def _normalize_distance(self, distance: ndarray) -> ndarray:
        max_distance = float(np.max(distance)) if distance.size else 0.0
        if max_distance <= 0.0:
            return np.zeros_like(distance, dtype=np.float32)
        return (distance / max_distance).astype(np.float32)

    def _distance_to_points(self, pos: ndarray, points: ndarray) -> ndarray:
        if points.size == 0:
            return np.zeros((pos.shape[0],), dtype=np.float32)
        min_dist = np.full((pos.shape[0],), np.inf, dtype=np.float64)
        chunk_size = 4096
        for start in range(0, pos.shape[0], chunk_size):
            chunk = pos[start:start + chunk_size]
            dist = np.linalg.norm(chunk[:, None, :] - points[None, :, :], axis=-1)
            min_dist[start:start + chunk_size] = dist.min(axis=1)
        return min_dist.astype(np.float32)

    def _get_domain_boundary_node_mask(self, node_types: ndarray, edge_types: ndarray, edge_index: ndarray) -> ndarray:
        mask = node_types != 1
        boundary_edges = edge_types > 1
        if np.any(boundary_edges):
            boundary_nodes = np.unique(edge_index[:, boundary_edges].reshape(-1))
            mask[boundary_nodes] = True
        return mask.astype(np.float32)

    def _get_boundary_normals(self, node_types: ndarray, edge_types: ndarray, edge_index: ndarray, pos: ndarray) -> ndarray:
        normals = np.zeros((edge_index.shape[1], 2), dtype=np.float32)
        boundary_edges = np.where(edge_types > 1)[0]
        for edge_id in boundary_edges:
            src, dst = edge_index[:, edge_id]
            src_is_boundary = node_types[src] != 1
            dst_is_boundary = node_types[dst] != 1
            if src_is_boundary and not dst_is_boundary:
                vec = pos[dst] - pos[src]
            elif dst_is_boundary and not src_is_boundary:
                vec = pos[src] - pos[dst]
            else:
                vec = pos[dst] - pos[src]
            norm = np.linalg.norm(vec)
            if norm > 0:
                normals[edge_id] = (vec / norm).astype(np.float32)
        return normals

    def _get_node_boundary_static_features(self, event_idx: int) -> dict:
        node_types, edge_types, edge_index, pos = self._get_boundary_metadata(event_idx)
        inflow_nodes = np.where(node_types == 2)[0]
        boundary_nodes = np.where(node_types != 1)[0]

        distance_to_inflow = self._distance_to_points(pos, pos[inflow_nodes])
        distance_to_boundary = self._distance_to_points(pos, pos[boundary_nodes])
        return {
            'is_inflow_boundary_node': (node_types == 2).astype(np.float32),
            'is_domain_boundary_node': self._get_domain_boundary_node_mask(node_types, edge_types, edge_index),
            'distance_to_inflow': self._normalize_distance(distance_to_inflow),
            'distance_to_boundary': self._normalize_distance(distance_to_boundary),
        }

    def _get_edge_boundary_static_features(self, event_idx: int) -> dict:
        node_types, edge_types, edge_index, pos = self._get_boundary_metadata(event_idx)
        normals = self._get_boundary_normals(node_types, edge_types, edge_index, pos)

        wall_adjacent_nodes = set()
        boundary_adjacent_nodes = set()
        node_normal_sum = np.zeros((node_types.shape[0], 2), dtype=np.float32)
        node_normal_count = np.zeros((node_types.shape[0],), dtype=np.float32)
        for edge_id in np.where(edge_types > 1)[0]:
            src, dst = edge_index[:, edge_id]
            normal_nodes = [node for node in (src, dst) if node_types[node] == 1]
            for node in normal_nodes:
                boundary_adjacent_nodes.add(int(node))
                if edge_types[edge_id] == 3:
                    wall_adjacent_nodes.add(int(node))
                    node_normal_sum[node] += normals[edge_id]
                    node_normal_count[node] += 1.0

        is_wall_boundary_edge = edge_types == 3
        is_boundary_edge = edge_types > 1
        edge_context_normals = normals.copy()
        for edge_id in range(edge_index.shape[1]):
            src, dst = edge_index[:, edge_id]
            if src in wall_adjacent_nodes or dst in wall_adjacent_nodes:
                is_wall_boundary_edge[edge_id] = True
                adjacent_normals = []
                for node in (src, dst):
                    if node_normal_count[node] > 0:
                        adjacent_normals.append(node_normal_sum[node] / node_normal_count[node])
                if adjacent_normals and np.linalg.norm(edge_context_normals[edge_id]) == 0:
                    edge_context_normals[edge_id] = np.mean(adjacent_normals, axis=0)
            if src in boundary_adjacent_nodes or dst in boundary_adjacent_nodes:
                is_boundary_edge[edge_id] = True

        return {
            'is_inflow_edge': (edge_types == 2).astype(np.float32),
            'is_wall_boundary_edge': is_wall_boundary_edge.astype(np.float32),
            'is_boundary_edge': is_boundary_edge.astype(np.float32),
            'boundary_normal_x': edge_context_normals[:, 0],
            'boundary_normal_y': edge_context_normals[:, 1],
        }

    def _get_dynamic_bc_flow(self, hydrograph_path: str, edges_shp_path: str) -> ndarray:
        edge_types = get_edge_types(edges_shp_path)
        inflow = get_inflow_hydrograph(hydrograph_path)[:, None]
        bc_flow = np.zeros((inflow.shape[0], edge_types.shape[0]), dtype=inflow.dtype)
        bc_flow[:, edge_types == 2] = inflow
        return bc_flow

    def _log_boundary_summary_once(self, event_idx: int) -> None:
        if event_idx in self._logged_boundary_summaries:
            return
        paths = self._get_event_file_paths(event_idx)
        node_types = get_node_types(paths[self.EVENT_FILE_KEYS[1]])
        edge_types = get_edge_types(paths[self.EVENT_FILE_KEYS[2]])
        inflow_edges = int(np.sum(edge_types == 2))
        wall_edges = int(np.sum(edge_types == 3))
        retained_boundary_nodes = int(np.sum(node_types == 2))
        dropped_boundary_ghosts = int(np.sum(node_types == 3))
        assert inflow_edges > 0, f'Run {self.event_run_ids[event_idx]} has no inflow boundary edge (edge_type == 2).'
        assert retained_boundary_nodes > 0, f'Run {self.event_run_ids[event_idx]} has no inflow boundary node (node_type == 2).'
        self.log_func(
            f'mSWE boundary summary run_id={self.event_run_ids[event_idx]}: '
            f'inflow_edges={inflow_edges}, outflow_edges=0, wall_boundary_edges={wall_edges}, '
            f'retained_boundary_nodes={retained_boundary_nodes}, dropped_boundary_ghosts={dropped_boundary_ghosts}'
        )
        self._logged_boundary_summaries.add(event_idx)

    def _get_static_node_features(self, event_idx: int) -> ndarray:
        def _get_roughness(node_shp_path: str) -> np.ndarray:
            num_nodes = get_node_types(node_shp_path).shape[0]
            ROUGHNESS_VALUE = 0.023  # Manning's n
            roughness = np.full((num_nodes,), ROUGHNESS_VALUE, dtype=np.float32)
            return roughness

        def _get_dem_based_feature(node_shp_path: str,
                                   dem_path: str,
                                   feature_func: Callable,
                                   *output_filenames: Tuple[str]) -> ndarray:
            pos = get_cell_position(node_shp_path)
            dem_folder = os.path.dirname(dem_path)
            dem_filename = os.path.splitext(os.path.basename(dem_path))[0]
            filled_dem_path = os.path.join(dem_folder, f'{dem_filename}_filled.tif')
            filled_dem = get_filled_dem(dem_path, filled_dem_path)

            output_paths = [os.path.join(dem_folder, fn) for fn in output_filenames]
            return feature_func(filled_dem, *output_paths, pos)

        def _get_aspect(nodes_shp_path: str, dem_path: str):
            dem_filename = os.path.splitext(os.path.basename(dem_path))[0]
            return _get_dem_based_feature(nodes_shp_path, dem_path, get_aspect, f'{dem_filename}_aspect.tif')

        def _get_curvature(nodes_shp_path: str, dem_path: str):
            dem_filename = os.path.splitext(os.path.basename(dem_path))[0]
            return _get_dem_based_feature(nodes_shp_path, dem_path, get_curvature, f'{dem_filename}_curvature.tif')

        def _get_flow_accumulation(nodes_shp_path: str, dem_path: str):
            dem_filename = os.path.splitext(os.path.basename(dem_path))[0]
            return _get_dem_based_feature(nodes_shp_path, dem_path, get_flow_accumulation,
                                          f'{dem_filename}_flow_dir.tif', f'{dem_filename}_flow_acc_dem.tif')

        paths = self._get_event_file_paths(event_idx)
        node_boundary_features = self._get_node_boundary_static_features(event_idx) if self.boundary_aware_features else {}
        STATIC_NODE_RETRIEVAL_MAP = {
            "area": lambda: get_cell_area(paths[self.EVENT_FILE_KEYS[4]]),
            "roughness": lambda: _get_roughness(paths[self.EVENT_FILE_KEYS[1]]),
            "elevation": lambda: get_cell_elevation(paths[self.EVENT_FILE_KEYS[1]]),
            "position_x": lambda: get_cell_position_x(paths[self.EVENT_FILE_KEYS[1]]),
            "position_y": lambda: get_cell_position_y(paths[self.EVENT_FILE_KEYS[1]]),
            "aspect": lambda: _get_aspect(paths[self.EVENT_FILE_KEYS[1]], paths[self.EVENT_FILE_KEYS[3]]),
            "curvature": lambda: _get_curvature(paths[self.EVENT_FILE_KEYS[1]], paths[self.EVENT_FILE_KEYS[3]]),
            "flow_accumulation": lambda: _get_flow_accumulation(paths[self.EVENT_FILE_KEYS[1]], paths[self.EVENT_FILE_KEYS[3]]),
            **{feature: (lambda feature=feature: node_boundary_features[feature]) for feature in node_boundary_features},
        }

        static_features = self._get_features(feature_list=self.STATIC_NODE_FEATURES,
                                  feature_retrieval_map=STATIC_NODE_RETRIEVAL_MAP)
        static_features = np.array(static_features).transpose()
        if self.boundary_aware_features:
            event_bc = self.boundary_conditions[event_idx]
            event_bc._boundary_static_nodes = static_features[event_bc.init_inflow_boundary_nodes].copy()
            self._log_boundary_summary_once(event_idx)
        return static_features

    def _get_static_edge_features(self, event_idx: int) -> ndarray:
        def get_relative_position(coord: Literal['x', 'y'], nodes_shp_path: str, edges_shp_path: str) -> ndarray:
            pos_retrieval_func = get_cell_position_x if coord == 'x' else get_cell_position_y
            position = pos_retrieval_func(nodes_shp_path)
            edge_index = get_edge_index(edges_shp_path)
            row, col = edge_index
            relative_pos = position[row] - position[col]
            return relative_pos

        paths = self._get_event_file_paths(event_idx)
        edge_boundary_features = self._get_edge_boundary_static_features(event_idx) if self.boundary_aware_features else {}
        STATIC_EDGE_RETRIEVAL_MAP = {
            "face_length": lambda: get_face_length(paths[self.EVENT_FILE_KEYS[2]]),
            "length": lambda: get_edge_length(paths[self.EVENT_FILE_KEYS[2]]),
            "slope": lambda: get_edge_slope(paths[self.EVENT_FILE_KEYS[2]]),
            "relative_position_x": lambda: get_relative_position('x', paths[self.EVENT_FILE_KEYS[1]], paths[self.EVENT_FILE_KEYS[2]]),
            "relative_position_y": lambda: get_relative_position('y', paths[self.EVENT_FILE_KEYS[1]], paths[self.EVENT_FILE_KEYS[2]]),
            **{feature: (lambda feature=feature: edge_boundary_features[feature]) for feature in edge_boundary_features},
        }

        static_features = self._get_features(feature_list=self.STATIC_EDGE_FEATURES,
                                  feature_retrieval_map=STATIC_EDGE_RETRIEVAL_MAP)
        static_features = np.array(static_features).transpose()
        if self.boundary_aware_features:
            event_bc = self.boundary_conditions[event_idx]
            boundary_nodes = np.concat([np.array(event_bc.init_inflow_boundary_nodes), np.array(event_bc.init_outflow_boundary_nodes)])
            edge_index = get_edge_index(paths[self.EVENT_FILE_KEYS[2]])
            boundary_edges_mask = np.any(np.isin(edge_index, boundary_nodes), axis=0)
            event_bc._boundary_static_edges = static_features[boundary_edges_mask].copy()
        return static_features

    def _get_dynamic_node_features(self, event_idx: int) -> ndarray:
        def _get_rainfall(simulation_path: str, node_shp_path: str):
            water_depth = get_water_depth(simulation_path)
            num_timesteps = water_depth.shape[0]
            node_types = get_node_types(node_shp_path)
            num_nodes = node_types.shape[0]

            # No rainfall in mSWEGNN dataset
            rainfall = np.zeros((num_timesteps, num_nodes), dtype=water_depth.dtype)
            return rainfall

        def _get_inflow_hydrograph(hydrograph_path: str, node_shp_path: str):
            inflow = get_inflow_hydrograph(hydrograph_path)[:, None]
            assert np.all(inflow >= 0), "Inflow hydrograph contains negative values."
            node_types = get_node_types(node_shp_path)
            num_nodes = node_types.shape[0]
            inflow = np.repeat(inflow, repeats=num_nodes, axis=-1)
            return inflow

        def _get_water_depth(simulation_path: str, node_shp_path: str):
            water_depth = get_water_depth(simulation_path)
            # Create water_depth for ghost nodes as zero
            node_types = get_node_types(node_shp_path)
            num_ghost_nodes = (node_types != 1).sum()
            ghost_nodes_depth = np.zeros((water_depth.shape[0], num_ghost_nodes), dtype=water_depth.dtype)
            water_depth = np.concatenate([water_depth, ghost_nodes_depth], axis=1)
            return water_depth

        def _get_water_volume(simulation_path: str, node_shp_path: str, cells_shp_path: str):
            water_depth = _get_water_depth(simulation_path, node_shp_path)

            cell_area = get_cell_area(cells_shp_path)[None, :]
            cell_area = np.repeat(cell_area, repeats=water_depth.shape[0], axis=0)
            water_volume = water_depth * cell_area
            return water_volume

        paths = self._get_event_file_paths(event_idx)
        DYNAMIC_NODE_RETRIEVAL_MAP = {
            "inflow": lambda: self._get_event_dynamic(event_idx, _get_inflow_hydrograph, aggr='first',
                                                      hydrograph_path=paths[self.EVENT_FILE_KEYS[5]],
                                                      node_shp_path=paths[self.EVENT_FILE_KEYS[1]]),
            "rainfall": lambda: self._get_event_dynamic(event_idx, _get_rainfall, aggr='first',
                                                        simulation_path=paths[self.EVENT_FILE_KEYS[0]],
                                                        node_shp_path=paths[self.EVENT_FILE_KEYS[1]]),
            "water_volume": lambda: self._get_event_dynamic(event_idx, _get_water_volume, aggr='first',
                                                            simulation_path=paths[self.EVENT_FILE_KEYS[0]],
                                                            node_shp_path=paths[self.EVENT_FILE_KEYS[1]],
                                                            cells_shp_path=paths[self.EVENT_FILE_KEYS[4]]),
            "water_depth": lambda: self._get_event_dynamic(event_idx, _get_water_depth, aggr='first',
                                                           simulation_path=paths[self.EVENT_FILE_KEYS[0]],
                                                           node_shp_path=paths[self.EVENT_FILE_KEYS[1]]),
        }

        dynamic_features = self._get_features(feature_list=self.DYNAMIC_NODE_FEATURES,
                                  feature_retrieval_map=DYNAMIC_NODE_RETRIEVAL_MAP)
        dynamic_features = np.array(dynamic_features).transpose(1, 2, 0)
        return dynamic_features

    def _get_dynamic_edge_features(self, event_idx: int) -> ndarray:
        def _get_face_flow(simulation_path: str, hydrograph_path: str, edges_shp_path: str):
            face_flow = get_face_flow(simulation_path)

            # Overwrite boundary edge flows with inflow hydrograph
            # Found that the simulation output differs for the first timestep
            inflow = get_inflow_hydrograph(hydrograph_path)[:, None]
            edge_types = get_edge_types(edges_shp_path)
            boundary_edge_indices = np.where(edge_types == 2)[0]
            face_flow[:, boundary_edge_indices] = inflow

            return face_flow

        paths = self._get_event_file_paths(event_idx)
        DYNAMIC_EDGE_RETRIEVAL_MAP = {
            "face_flow": lambda: self._get_event_dynamic(event_idx, _get_face_flow, aggr='first',
                                                         simulation_path=paths[self.EVENT_FILE_KEYS[0]],
                                                         hydrograph_path=paths[self.EVENT_FILE_KEYS[5]],
                                                         edges_shp_path=paths[self.EVENT_FILE_KEYS[2]]),
            "bc_flow": lambda: self._get_event_dynamic(event_idx, self._get_dynamic_bc_flow, aggr='first',
                                                        hydrograph_path=paths[self.EVENT_FILE_KEYS[5]],
                                                        edges_shp_path=paths[self.EVENT_FILE_KEYS[2]]),
        }

        dynamic_features = self._get_features(feature_list=self.DYNAMIC_EDGE_FEATURES,
                                  feature_retrieval_map=DYNAMIC_EDGE_RETRIEVAL_MAP)
        dynamic_features = np.array(dynamic_features).transpose(1, 2, 0)
        return dynamic_features
