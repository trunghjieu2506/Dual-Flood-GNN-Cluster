import argparse
import csv
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data.hecras_data_retrieval import (
    get_cell_area,
    get_event_timesteps,
    get_water_level,
    get_water_volume,
    get_wl_vol_interp_points_for_cell,
)
from data.shp_data_retrieval import get_cell_elevation
from data_mswegnn.hydrograph_data_retrieval import get_event_timesteps as get_mswe_event_timesteps
from data_mswegnn.nc_data_retrieval import get_water_depth as get_mswe_water_depth
from data_mswegnn.shp_data_retrieval import (
    get_cell_area as get_mswe_cell_area,
    get_node_types as get_mswe_node_types,
)

EPS = 1e-12


def rmse(pred, target, axis=None):
    return np.sqrt(np.mean((pred - target) ** 2, axis=axis))


def mae(pred, target, axis=None):
    return np.mean(np.abs(pred - target), axis=axis)


def nse(pred, target):
    denom = np.sum((target - np.mean(target)) ** 2)
    if denom <= EPS:
        return float("nan")
    return float(1.0 - np.sum((target - pred) ** 2) / denom)


def csi(pred_depth, target_depth, threshold):
    binary_pred = pred_depth > threshold
    binary_target = target_depth > threshold
    tp = np.logical_and(binary_pred, binary_target).sum()
    fp = np.logical_and(binary_pred, ~binary_target).sum()
    fn = np.logical_and(~binary_pred, binary_target).sum()
    denom = tp + fp + fn
    return float(tp / denom) if denom else float("nan")


def mean_timestep_metric(pred, target, metric_func):
    return float(np.nanmean([metric_func(pred[t], target[t]) for t in range(pred.shape[0])]))


def volume_to_depth(volume, hec_ras_path, elevation):
    volume = np.asarray(volume, dtype=np.float64)
    if volume.ndim == 3:
        volume = volume[..., 0]
    num_timesteps, num_nodes = volume.shape
    area = get_cell_area(str(hec_ras_path)).astype(np.float64)
    water_level = np.zeros_like(volume, dtype=np.float64)
    for cell_idx in range(num_nodes):
        wl_interp, vol_interp = get_wl_vol_interp_points_for_cell(cell_idx, str(hec_ras_path))
        wl_interp = wl_interp.astype(np.float64)
        vol_interp = vol_interp.astype(np.float64)
        cell_vol = volume[:, cell_idx]
        water_level[:, cell_idx] = np.interp(cell_vol, vol_interp, wl_interp)
        over = cell_vol > vol_interp[-1]
        if np.any(over):
            water_level[over, cell_idx] = wl_interp[-1] + ((cell_vol[over] - vol_interp[-1]) / area[cell_idx])
    return np.clip(water_level - elevation[None, :num_nodes], a_min=0.0, a_max=None)


def get_trimmed_direct_depth(hec_ras_path, elevation, run_id, config, num_timesteps):
    dataset_config = config["dataset_parameters"]
    previous_timesteps = int(dataset_config["previous_timesteps"])
    timestep_interval = int(dataset_config["timestep_interval"])
    spin_up_time = dataset_config.get("spin_up_time")
    time_from_peak = dataset_config.get("time_from_peak")
    test_config = config.get("testing_parameters", {})
    rollout_start = int(test_config.get("rollout_start") or 0)
    rollout_timesteps = test_config.get("rollout_timesteps")

    water_level = get_water_level(str(hec_ras_path)).astype(np.float64)
    direct_depth = np.clip(water_level - elevation[None, : water_level.shape[1]], a_min=0.0, a_max=None)

    timesteps = get_event_timesteps(str(hec_ras_path))
    base_timestep_interval = int((timesteps[1] - timesteps[0]).total_seconds())
    if timestep_interval % base_timestep_interval != 0:
        raise ValueError(
            f"Configured timestep interval {timestep_interval}s is not divisible by native "
            f"event interval {base_timestep_interval}s for run {run_id}."
        )

    start = 0
    if isinstance(spin_up_time, int):
        start = spin_up_time // base_timestep_interval
    elif isinstance(spin_up_time, dict):
        spin_up_seconds = spin_up_time.get(run_id, spin_up_time.get(str(run_id), spin_up_time.get("default", 0)))
        start = spin_up_seconds // base_timestep_interval

    end = None
    if time_from_peak is not None:
        # Match FloodEventDataset: end at peak + configured horizon after peak.
        volume = get_water_volume(str(hec_ras_path)).astype(np.float64)
        peak_idx = int(np.argmax(volume.sum(axis=1)))
        end = peak_idx + (int(time_from_peak) // base_timestep_interval)

    trimmed = direct_depth[start:end]
    step = timestep_interval // base_timestep_interval
    if step > 1:
        trimmed_length = (trimmed.shape[0] // step) * step
        trimmed = trimmed[:trimmed_length].reshape(-1, step, trimmed.shape[1]).mean(axis=1)

    label_start = previous_timesteps + 1 + rollout_start
    label_end = label_start + num_timesteps if rollout_timesteps is None else label_start + int(rollout_timesteps)
    direct_target = trimmed[label_start:label_end]
    if direct_target.shape[0] != num_timesteps:
        raise ValueError(
            f"Direct depth target for run {run_id} has {direct_target.shape[0]} timesteps, "
            f"but prediction has {num_timesteps}. Check config alignment."
        )
    return direct_target


def load_notebook_depth_target(run_id, depth_target_dir):
    if depth_target_dir is None:
        return None
    depth_target_dir = Path(depth_target_dir)
    candidates = sorted(depth_target_dir.glob(f"*runid_{run_id}_*metrics_wd.npz"))
    candidates.extend(sorted(depth_target_dir.glob(f"*runid_{run_id}_*.npz")))
    for candidate in candidates:
        data = np.load(candidate, allow_pickle=True)
        if "target" in data:
            target = data["target"]
            if target.ndim == 3 and target.shape[-1] == 1:
                target = target[..., 0]
            return np.asarray(target, dtype=np.float64)
    return None


def get_dataset_type(config):
    return config.get("dataset_parameters", {}).get("dataset_type", "hecras")


def volume_to_depth_mswe(volume, cells_shp_path):
    volume = np.asarray(volume, dtype=np.float64)
    if volume.ndim == 3:
        volume = volume[..., 0]
    area = np.asarray(get_mswe_cell_area(str(cells_shp_path)), dtype=np.float64).reshape(-1)
    depth = np.zeros_like(volume, dtype=np.float64)
    num_nodes = min(volume.shape[1], area.shape[0])
    safe_area = np.where(area[:num_nodes] > EPS, area[:num_nodes], 1.0)
    depth[:, :num_nodes] = volume[:, :num_nodes] / safe_area[None, :]
    return np.clip(depth, a_min=0.0, a_max=None)


def get_trimmed_direct_depth_mswe(simulation_path, nodes_shp_path, hydrograph_path, run_id, config, num_timesteps):
    dataset_config = config["dataset_parameters"]
    previous_timesteps = int(dataset_config["previous_timesteps"])
    timestep_interval = int(dataset_config["timestep_interval"])
    spin_up_time = dataset_config.get("spin_up_time")
    test_config = config.get("testing_parameters", {})
    rollout_start = int(test_config.get("rollout_start") or 0)
    rollout_timesteps = test_config.get("rollout_timesteps")

    water_depth = get_mswe_water_depth(str(simulation_path)).astype(np.float64)
    node_types = np.asarray(get_mswe_node_types(str(nodes_shp_path)), dtype=np.int32).reshape(-1)
    num_ghost_nodes = int(np.sum(node_types != 1))
    if num_ghost_nodes > 0:
        ghost = np.zeros((water_depth.shape[0], num_ghost_nodes), dtype=np.float64)
        water_depth = np.concatenate([water_depth, ghost], axis=1)

    timesteps = get_mswe_event_timesteps(str(hydrograph_path)).astype(np.float64)
    base_timestep_interval = int(round(float(timesteps[1] - timesteps[0])))
    if timestep_interval % base_timestep_interval != 0:
        raise ValueError(
            f"Configured timestep interval {timestep_interval}s is not divisible by native event interval {base_timestep_interval}s for run {run_id}."
        )

    start = 0
    if isinstance(spin_up_time, int):
        start = spin_up_time // base_timestep_interval
    elif isinstance(spin_up_time, dict):
        spin_up_seconds = spin_up_time.get(run_id, spin_up_time.get(str(run_id), spin_up_time.get("default", 0)))
        start = spin_up_seconds // base_timestep_interval

    trimmed = water_depth[start:]
    step = timestep_interval // base_timestep_interval
    if step > 1:
        trimmed_length = (trimmed.shape[0] // step) * step
        trimmed = trimmed[:trimmed_length].reshape(-1, step, trimmed.shape[1]).mean(axis=1)

    label_start = previous_timesteps + 1 + rollout_start
    label_end = label_start + num_timesteps if rollout_timesteps is None else label_start + int(rollout_timesteps)
    direct_target = trimmed[label_start:label_end]
    if direct_target.shape[0] != num_timesteps:
        raise ValueError(
            f"Direct depth target for run {run_id} has {direct_target.shape[0]} timesteps, but prediction has {num_timesteps}. Check config alignment."
        )
    return direct_target


def build_run_file_map(root_dir):
    raw_dir = Path(root_dir) / "raw"
    frames = []
    for csv_name in ("train.csv", "test.csv"):
        csv_path = raw_dir / csv_name
        if csv_path.exists():
            frames.append(pd.read_csv(csv_path, encoding="utf-8-sig"))
    if not frames:
        raise FileNotFoundError(f"No train.csv/test.csv found under {raw_dir}")
    df = pd.concat(frames, ignore_index=True)
    records = {}
    for row in df.to_dict(orient="records"):
        run_id = str(row["Run_ID"])
        records[run_id] = {
            key: (raw_dir / value if isinstance(value, str) and key != "Run_ID" else value)
            for key, value in row.items()
        }
    return records


def summarize_metric_file(path, run_file_map, elevation, config, depth_target_dir):
    match = re.search(r"runid_([^_]+)_test_metrics\.npz$", path.name)
    if not match:
        raise ValueError(f"Cannot parse run id from {path}")
    run_id = match.group(1)
    dataset_type = get_dataset_type(config)
    row_info = run_file_map[run_id]
    data = np.load(path, allow_pickle=True)
    pred = data["pred"]
    target = data["target"]
    edge_pred = data["edge_pred"]
    edge_target = data["edge_target"]

    row = {
        "metric_file": str(path),
        "run_id": run_id,
        "volume_rmse": float(np.mean(data["rmse"])),
        "volume_mae": float(np.mean(data["mae"])),
        "volume_nse": float(np.mean(data["nse"])),
        "flow_rmse": float(np.mean(data["edge_rmse"])),
        "flow_mae": float(np.mean(data["edge_mae"])),
        "flow_nse": float(np.mean(data["edge_nse"])),
    }

    if dataset_type == "mswegnn":
        cells_shp_path = row_info["Cells_Shp_Filepath"]
        simulation_path = row_info["Simulation_Filepath"]
        nodes_shp_path = row_info["Nodes_Shp_Filepath"]
        hydrograph_path = row_info["Hydrograph_Filepath"]
        depth_pred = volume_to_depth_mswe(pred, cells_shp_path)
        depth_target = load_notebook_depth_target(run_id, depth_target_dir)
        if depth_target is None:
            depth_target = get_trimmed_direct_depth_mswe(simulation_path, nodes_shp_path, hydrograph_path, run_id, config, depth_pred.shape[0])
    else:
        hec_ras_path = row_info["HECRAS_Filepath"]
        depth_pred = volume_to_depth(pred, hec_ras_path, elevation)
        depth_target = load_notebook_depth_target(run_id, depth_target_dir)
        if depth_target is None:
            depth_target = get_trimmed_direct_depth(hec_ras_path, elevation, run_id, config, depth_pred.shape[0])

    num_nodes = depth_pred.shape[1]
    depth_target = depth_target[: depth_pred.shape[0], :num_nodes]
    if depth_target.shape != depth_pred.shape:
        raise ValueError(f"Depth target shape {depth_target.shape} does not match prediction shape {depth_pred.shape} for run {run_id}")

    row.update({
        "depth_rmse": float(np.mean(rmse(depth_pred, depth_target, axis=1))),
        "depth_mae": float(np.mean(mae(depth_pred, depth_target, axis=1))),
        "depth_nse": mean_timestep_metric(depth_pred, depth_target, nse),
        "depth_csi_0.05m": mean_timestep_metric(depth_pred, depth_target, lambda p, t: csi(p, t, 0.05)),
        "depth_csi_0.30m": mean_timestep_metric(depth_pred, depth_target, lambda p, t: csi(p, t, 0.30)),
    })
    return row


def main():
    parser = argparse.ArgumentParser(description="Collect water volume, water depth, and flow regression metrics.")
    parser.add_argument("--metrics_dir", default="saved_metrics")
    parser.add_argument("--root_dir", default="data/datasets")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--depth_target_dir", default=None)
    parser.add_argument("--output_csv", default="saved_metrics/regression_metrics_summary.csv")
    parser.add_argument("--output_json", default="saved_metrics/regression_metrics_summary.json")
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    metric_files = sorted(metrics_dir.glob("*_runid_*_test_metrics.npz"), key=os.path.getmtime)
    if not metric_files:
        raise FileNotFoundError(f"No *_runid_*_test_metrics.npz files found in {metrics_dir}")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    dataset_type = get_dataset_type(config)
    run_file_map = build_run_file_map(args.root_dir)
    elevation = None
    if dataset_type != "mswegnn":
        nodes_shp_file = config["dataset_parameters"].get("nodes_shp_file", "GEOMETRY/cell_centers_with_ele.shp")
        elevation = get_cell_elevation(str(Path(args.root_dir) / "raw" / nodes_shp_file)).astype(np.float64)

    rows = [summarize_metric_file(path, run_file_map, elevation, config, args.depth_target_dir) for path in metric_files]
    metric_keys = [k for k in rows[0] if k not in {"metric_file", "run_id"}]
    aggregate = {"run_id": "mean"}
    aggregate.update({k: float(np.nanmean([r[k] for r in rows])) for k in metric_keys})
    std = {"run_id": "std"}
    std.update({k: float(np.nanstd([r[k] for r in rows])) for k in metric_keys})

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({k: aggregate.get(k, "") for k in fieldnames})
        writer.writerow({k: std.get(k, "") for k in fieldnames})

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"events": rows, "mean": aggregate, "std": std}, indent=2))

    print("Saved regression metrics summary to:", output_csv)
    print("Saved regression metrics JSON to:", output_json)
    print(json.dumps({"mean": aggregate, "std": std}, indent=2))


if __name__ == "__main__":
    main()
