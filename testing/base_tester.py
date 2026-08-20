import os
import numpy as np
import torch

from torch.nn import Module
from data import FloodEventDataset
from utils import Logger
from utils.validation_stats import ValidationStats
from typing import List, Optional

class BaseTester:
    def __init__(self,
                 model: Module,
                 dataset: FloodEventDataset,
                 rollout_start: int = 0,
                 rollout_timesteps: Optional[int] = None,
                 include_global_mass_loss: bool = True,
                 include_local_mass_loss: bool = True,
                 logger: Logger = None,
                 device: str = 'cpu'):
        self.model = model
        self.dataset = dataset
        self.rollout_start = rollout_start
        self.rollout_timesteps = rollout_timesteps
        self.include_global_mass_loss = include_global_mass_loss
        self.include_local_mass_loss = include_local_mass_loss
        self.include_physics_loss = include_global_mass_loss or include_local_mass_loss
        self.logger = logger
        self.device = device
        self.events_validation_stats: List[ValidationStats] = []

        self.log = print
        if logger is not None and hasattr(logger, 'log'):
            self.log = logger.log

        boundary_condition = getattr(dataset, 'boundary_condition', None)
        self.boundary_nodes_mask = getattr(boundary_condition, 'boundary_nodes_mask', None)
        self.non_boundary_nodes_mask = None if self.boundary_nodes_mask is None else ~self.boundary_nodes_mask
        self.boundary_edges_mask = getattr(boundary_condition, 'boundary_edges_mask', None)
        self.event_run_ids = getattr(dataset, 'event_run_ids', getattr(dataset, 'hec_ras_run_ids', []))

        # Get sliding window indices
        previous_timesteps = dataset.previous_timesteps
        sliding_window_length = previous_timesteps + 1

        target_nodes_idx = dataset.DYNAMIC_NODE_FEATURES.index(dataset.NODE_TARGET_FEATURE)
        self.start_node_target_idx = dataset.num_static_node_features + (target_nodes_idx * sliding_window_length)
        self.end_node_target_idx = self.start_node_target_idx + sliding_window_length

        target_edges_idx = dataset.DYNAMIC_EDGE_FEATURES.index(dataset.EDGE_TARGET_FEATURE)
        self.start_edge_target_idx = dataset.num_static_edge_features + (target_edges_idx * sliding_window_length)
        self.end_edge_target_idx = self.start_edge_target_idx + sliding_window_length

    def test(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def _get_graph_boundary_nodes_mask(self, graph):
        if hasattr(graph, 'boundary_nodes_mask'):
            return graph.boundary_nodes_mask.bool()
        if self.boundary_nodes_mask is None:
            return torch.zeros(graph.num_nodes, dtype=torch.bool, device=graph.x.device)
        return torch.as_tensor(self.boundary_nodes_mask, dtype=torch.bool, device=graph.x.device)

    def _get_graph_boundary_edges_mask(self, graph):
        if hasattr(graph, 'boundary_edges_mask'):
            return graph.boundary_edges_mask.bool()
        if self.boundary_edges_mask is None:
            return torch.zeros(graph.num_edges, dtype=torch.bool, device=graph.edge_attr.device)
        return torch.as_tensor(self.boundary_edges_mask, dtype=torch.bool, device=graph.edge_attr.device)

    def _get_cell_thresholds(self, graph, non_boundary_nodes_mask=None):
        if self.dataset.NODE_TARGET_FEATURE == 'water_depth':
            if non_boundary_nodes_mask is None:
                non_boundary_nodes_mask = ~self._get_graph_boundary_nodes_mask(graph)
            return torch.full(
                (int(non_boundary_nodes_mask.sum().item()), 1),
                0.05,
                dtype=graph.x.dtype,
                device=graph.x.device,
            )
        area_nodes_idx = self.dataset.STATIC_NODE_FEATURES.index('area')
        area = graph.x[:, area_nodes_idx].clone()
        if self.dataset.is_normalized:
            area = self.dataset.normalizer.denormalize('area', area)
        if non_boundary_nodes_mask is None:
            non_boundary_nodes_mask = ~self._get_graph_boundary_nodes_mask(graph)
        area = area[non_boundary_nodes_mask, None]
        return area * 0.05 # 5% of cell area

    def _node_target_to_depth(self, graph, values, non_boundary_nodes_mask):
        """Convert a node target to water depth (m) so errors compare across meshes.

        When the target is water_volume, raw RMSE is area-weighted and therefore not
        comparable between events whose cells differ in size. Dividing by the denormalised
        cell area gives the depth-space error the mSWE-GNN paper reports.
        """
        if self.dataset.NODE_TARGET_FEATURE == 'water_depth':
            return values[non_boundary_nodes_mask]
        if self.dataset.NODE_TARGET_FEATURE != 'water_volume' or 'area' not in self.dataset.STATIC_NODE_FEATURES:
            return values[non_boundary_nodes_mask]

        area_nodes_idx = self.dataset.STATIC_NODE_FEATURES.index('area')
        area = graph.x[:, area_nodes_idx].clone()
        if self.dataset.is_normalized:
            area = self.dataset.normalizer.denormalize('area', area)
        area = torch.clamp(area[non_boundary_nodes_mask, None], min=1e-12)
        return values[non_boundary_nodes_mask] / area

    def get_avg_node_rmse(self) -> float:
        rmses = [stat.get_avg_rmse() for stat in self.events_validation_stats]
        return np.mean(rmses) if rmses else 0.0

    def get_avg_node_mae(self) -> float:
        maes = [stat.get_avg_mae() for stat in self.events_validation_stats]
        return np.mean(maes) if maes else 0.0

    def get_avg_node_depth_rmse(self) -> float:
        rmses = [stat.get_avg_depth_rmse() for stat in self.events_validation_stats]
        return np.mean(rmses) if rmses else self.get_avg_node_rmse()

    def get_avg_node_depth_mae(self) -> float:
        maes = [stat.get_avg_depth_mae() for stat in self.events_validation_stats]
        return np.mean(maes) if maes else self.get_avg_node_mae()

    def get_avg_node_nse(self) -> float:
        nses = [stat.get_avg_nse() for stat in self.events_validation_stats]
        return np.mean(nses) if nses else 0.0

    def get_avg_edge_rmse(self) -> float:
        edge_rmses = [stat.get_avg_edge_rmse() for stat in self.events_validation_stats]
        return np.mean(edge_rmses) if edge_rmses else 0.0

    def get_avg_edge_mae(self) -> float:
        edge_maes = [stat.get_avg_edge_mae() for stat in self.events_validation_stats]
        return np.mean(edge_maes) if edge_maes else 0.0

    def get_avg_edge_nse(self) -> float:
        edge_nses = [stat.get_avg_edge_nse() for stat in self.events_validation_stats]
        return np.mean(edge_nses) if edge_nses else 0.0

    def get_avg_global_mass_loss(self) -> float:
        losses = [stat.get_total_global_mass_loss() for stat in self.events_validation_stats]
        return np.mean(losses) if losses else 0.0

    def get_avg_local_mass_loss(self) -> float:
        losses = [stat.get_total_local_mass_loss() for stat in self.events_validation_stats]
        return np.mean(losses) if losses else 0.0

    def get_avg_abs_global_mass_loss(self) -> float:
        losses = [abs(stat.get_total_global_mass_loss()) for stat in self.events_validation_stats]
        return np.mean(losses) if losses else 0.0

    def get_avg_abs_local_mass_loss(self) -> float:
        losses = [abs(stat.get_total_local_mass_loss()) for stat in self.events_validation_stats]
        return np.mean(losses) if losses else 0.0

    def save_stats(self, output_dir: str, stats_filename_prefix: Optional[str] = None):
        for event_idx, run_id in enumerate(self.event_run_ids):
            validation_stats = self.events_validation_stats[event_idx]
            saved_metrics_path = os.path.join(output_dir, f'{stats_filename_prefix}_runid_{run_id}_test_metrics.npz')
            validation_stats.save_stats(saved_metrics_path)
