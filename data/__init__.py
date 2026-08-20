from typing import Literal

from .autoregressive_flood_dataset import AutoregressiveFloodDataset
from .flood_event_dataset import FloodEventDataset
from .in_memory_autoregressive_flood_dataset import InMemoryAutoregressiveFloodDataset
from .in_memory_flood_dataset import InMemoryFloodDataset

def dataset_factory(storage_mode: Literal["memory", "disk"], autoregressive: bool, *args, dataset_type: str = "hecras", **kwargs) -> FloodEventDataset:
    """Build a dataset for one of two backends.

    'hecras' (default) keeps the original behaviour: every event shares one fixed mesh.
    'mswegnn' loads the mSWE-GNN NetCDF/shapefile events, where each event carries its own
    topology, so downstream code must not assume a single global node/edge ordering.
    """
    if dataset_type == "mswegnn":
        from data_mswegnn import (
            AutomSWEGNNFloodEventDataset,
            MemAutomSWEGNNFloodDataset,
            MemmSWEGNNFloodDataset,
            mSWEGNNFloodEventDataset,
        )

        if autoregressive:
            if storage_mode == "memory":
                return MemAutomSWEGNNFloodDataset(*args, **kwargs)
            if storage_mode == "disk":
                return AutomSWEGNNFloodEventDataset(*args, **kwargs)
        else:
            if storage_mode == "memory":
                return MemmSWEGNNFloodDataset(*args, **kwargs)
            if storage_mode == "disk":
                return mSWEGNNFloodEventDataset(*args, **kwargs)

        raise ValueError(f"Dataset class is not defined for storage_mode={storage_mode!r}, autoregressive={autoregressive!r}.")

    if autoregressive:
        if storage_mode == "memory":
            return InMemoryAutoregressiveFloodDataset(*args, **kwargs)
        elif storage_mode == "disk":
            return AutoregressiveFloodDataset(*args, **kwargs)

    if storage_mode == "memory":
        return InMemoryFloodDataset(*args, **kwargs)
    elif storage_mode == "disk":
        return FloodEventDataset(*args, **kwargs)

    raise ValueError("Dataset class is not defined.")

__all__ = [
    "AutoregressiveFloodDataset",
    "FloodEventDataset",
    "InMemoryAutoregressiveFloodDataset",
    "InMemoryFloodDataset",
    "dataset_factory",
]
