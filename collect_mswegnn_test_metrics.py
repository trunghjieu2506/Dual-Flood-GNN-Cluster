#!/usr/bin/env python3
"""Collect paper-style mSWE-GNN test metrics from saved DualFloodGNN outputs.

The script expects ``*_runid_*_test_metrics.npz`` files produced by ``test.py``.
It does not rerun inference. Node predictions/targets are treated as water
volumes and converted back to water depth using the event ``Cells.shp``
``area_m2`` column. Edge predictions/targets are treated as discharge and
converted to the mSWE-GNN paper's unit discharge q by dividing by
``Links.shp`` ``fc_length``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

EPS = 1e-12
DEFAULT_THRESHOLDS = (0.05, 0.30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect standardized mSWE-GNN benchmark metrics from saved test .npz files."
    )
    parser.add_argument("--config", default=None, help="Optional training/testing YAML config.")
    parser.add_argument("--metrics_dir", default=None, help="Directory containing *_runid_*_test_metrics.npz files.")
    parser.add_argument(
        "--metrics_glob",
        default="*_runid_*_test_metrics.npz",
        help="Glob pattern inside metrics_dir. Use this to select one checkpoint when a directory contains multiple runs.",
    )
    parser.add_argument("--root_dir", default=None, help="mSWE dataset root directory, e.g. data_mswegnn/datasets.")
    parser.add_argument("--split_csv", default=None, help="Split CSV under root_dir/raw, usually test.csv.")
    parser.add_argument("--output_csv", default=None, help="Output per-event CSV summary.")
    parser.add_argument("--output_json", default=None, help="Output JSON summary.")
    parser.add_argument(
        "--target_source",
        choices=("saved_volume", "raw_depth"),
        default="saved_volume",
        help=(
            "Depth target source. saved_volume converts the saved target tensor using cell area; "
            "raw_depth reads the original NetCDF water depth and aligns using the config."
        ),
    )
    parser.add_argument(
        "--csi_aggregation",
        choices=("timestep_mean", "global"),
        default="timestep_mean",
        help="Aggregate CSI by averaging per-timestep CSI values or by computing one global contingency table.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
        help="Water-depth thresholds in metres for CSI metrics.",
    )
    parser.add_argument("--wandb_project", default=None, help="Optional W&B project for logging summary metrics.")
    parser.add_argument("--wandb_entity", default=None, help="Optional W&B entity.")
    parser.add_argument("--wandb_run_id", default=None, help="Optional W&B run id to resume.")
    parser.add_argument("--wandb_run_name", default=None, help="Optional W&B run name if creating a run.")
    parser.add_argument("--wandb_mode", default=None, help="Optional W&B mode, e.g. online/offline/disabled.")
    parser.add_argument("--wandb_prefix", default="test", help="Prefix for W&B metric names.")
    parser.add_argument(
        "--wandb_resume",
        default="allow",
        choices=("allow", "must", "never", "auto"),
        help="W&B resume policy when wandb_run_id is provided.",
    )
    parser.add_argument(
        "--allow_duplicate_run_ids",
        action="store_true",
        help="Allow multiple metric files with the same run_id. By default this is treated as accidental run mixing.",
    )
    return parser.parse_args()


def read_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def resolve_paths(args: argparse.Namespace, config: dict[str, Any]) -> tuple[Path, Path, str, Path, Path]:
    dataset_params = config.get("dataset_parameters", {})
    testing_params = config.get("testing_parameters", {})

    root_dir = Path(args.root_dir or dataset_params.get("root_dir", "data_mswegnn/datasets"))
    split_csv = args.split_csv or dataset_params.get("testing", {}).get("dataset_summary_file", "test.csv")
    metrics_dir = Path(args.metrics_dir or testing_params.get("output_dir", "saved_metrics"))
    output_csv = Path(args.output_csv) if args.output_csv else metrics_dir / "regression_metrics_summary.csv"
    output_json = Path(args.output_json) if args.output_json else metrics_dir / "regression_metrics_summary.json"
    return root_dir, metrics_dir, split_csv, output_csv, output_json


def load_split_records(root_dir: Path, split_csv: str) -> dict[str, dict[str, Any]]:
    raw_dir = root_dir / "raw"
    split_path = raw_dir / split_csv
    if not split_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_path}")

    df = pd.read_csv(split_path, encoding="utf-8-sig")
    required = {"Run_ID", "Cells_Shp_Filepath", "Edges_Shp_Filepath"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{split_path} is missing required columns: {sorted(missing)}")

    records: dict[str, dict[str, Any]] = {}
    for row in df.to_dict(orient="records"):
        run_id = str(row["Run_ID"])
        records[run_id] = {
            key: (raw_dir / value if isinstance(value, str) and key != "Run_ID" else value)
            for key, value in row.items()
        }
    return records


def parse_run_id(path: Path) -> str:
    match = re.search(r"runid_([^_]+)_test_metrics\.npz$", path.name)
    if not match:
        raise ValueError(f"Cannot parse run id from metrics filename: {path}")
    return match.group(1)


def run_sort_key(path: Path) -> tuple[int, str]:
    run_id = parse_run_id(path)
    return (int(run_id), run_id) if run_id.isdigit() else (10**12, run_id)


def squeeze_last_dim(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 3 and array.shape[-1] == 1:
        return array[..., 0]
    return array


def read_dbf_column(shp_path: Path, column: str) -> np.ndarray:
    dbf_path = shp_path.with_suffix(".dbf")
    if not dbf_path.exists():
        raise FileNotFoundError(f"DBF sidecar not found for shapefile: {dbf_path}")

    data = dbf_path.read_bytes()
    num_records = int.from_bytes(data[4:8], "little")
    header_length = int.from_bytes(data[8:10], "little")
    record_length = int.from_bytes(data[10:12], "little")

    fields = []
    offset = 1
    pos = 32
    while pos < header_length and data[pos] != 0x0D:
        descriptor = data[pos:pos + 32]
        name = descriptor[:11].split(b"\x00", 1)[0].decode("ascii", errors="ignore")
        field_type = chr(descriptor[11])
        length = descriptor[16]
        fields.append((name, field_type, length, offset))
        offset += length
        pos += 32

    selected = next((field for field in fields if field[0] == column), None)
    if selected is None:
        raise KeyError(f"Column {column!r} not found in {dbf_path}")

    _, field_type, field_length, field_offset = selected
    values = []
    for idx in range(num_records):
        record_start = header_length + idx * record_length
        record = data[record_start:record_start + record_length]
        if not record or record[0:1] == b"*":
            continue
        raw = record[field_offset:field_offset + field_length].decode("latin1", errors="ignore").strip()
        if raw == "":
            values.append(np.nan)
        elif field_type in {"N", "F", "B"}:
            values.append(float(raw))
        else:
            values.append(raw)
    return np.asarray(values)


def read_shp_column(shp_path: Path, column: str) -> np.ndarray:
    try:
        import geopandas as gpd
    except ModuleNotFoundError:
        return read_dbf_column(shp_path, column)

    return gpd.read_file(shp_path)[column].to_numpy()


def cell_area(cells_shp_path: Path) -> np.ndarray:
    area = np.asarray(read_shp_column(cells_shp_path, "area_m2"), dtype=np.float64).reshape(-1)
    if area.size == 0:
        raise ValueError(f"No cell areas found in {cells_shp_path}")
    return area


def face_length(edges_shp_path: Path) -> np.ndarray:
    length = np.asarray(read_shp_column(edges_shp_path, "fc_length"), dtype=np.float64).reshape(-1)
    if length.size == 0:
        raise ValueError(f"No face lengths found in {edges_shp_path}")
    return length


def volume_to_depth(volume: np.ndarray, cells_shp_path: Path) -> np.ndarray:
    volume = squeeze_last_dim(np.asarray(volume, dtype=np.float64))
    if volume.ndim != 2:
        raise ValueError(f"Expected volume with shape [timesteps, nodes], got {volume.shape}")

    area = cell_area(cells_shp_path)
    num_nodes = min(volume.shape[1], area.shape[0])
    safe_area = np.where(area[:num_nodes] > EPS, area[:num_nodes], 1.0)
    depth = volume[:, :num_nodes] / safe_area[None, :]
    return np.clip(depth, a_min=0.0, a_max=None)


def discharge_to_unit_q(discharge: np.ndarray, edges_shp_path: Path) -> np.ndarray:
    discharge = squeeze_last_dim(np.asarray(discharge, dtype=np.float64))
    if discharge.ndim != 2:
        raise ValueError(f"Expected discharge with shape [timesteps, edges], got {discharge.shape}")

    length = face_length(edges_shp_path)
    num_edges = min(discharge.shape[1], length.shape[0])
    safe_length = np.where(length[:num_edges] > EPS, length[:num_edges], 1.0)
    return discharge[:, :num_edges] / safe_length[None, :]


def raw_depth_target(row: dict[str, Any], config: dict[str, Any], num_timesteps: int) -> np.ndarray:
    """Rebuild the ground-truth depth straight from the source NetCDF.

    Independent of the saved targets: re-applies spin-up trimming, timestep aggregation and
    the label offset, and re-pads ghost nodes with zero depth. Use it to check that the
    dataset pipeline trimmed the series the way the metrics assume.
    """
    import xarray as xr

    dataset_config = config.get("dataset_parameters", {})
    previous_timesteps = int(dataset_config.get("previous_timesteps", 1))
    timestep_interval = int(dataset_config.get("timestep_interval", 7200))
    spin_up_time = dataset_config.get("spin_up_time", 0)
    test_config = config.get("testing_parameters", {})
    rollout_start = int(test_config.get("rollout_start") or 0)
    rollout_timesteps = test_config.get("rollout_timesteps")

    with xr.open_dataset(row["Simulation_Filepath"]) as ds:
        water_depth = np.asarray(ds["mesh2d_waterdepth"].values, dtype=np.float64)
    node_types = np.asarray(read_shp_column(row["Nodes_Shp_Filepath"], "node_type"), dtype=np.int32).reshape(-1)
    num_ghost_nodes = int(np.sum(node_types != 1))
    if num_ghost_nodes > 0:
        ghost_depth = np.zeros((water_depth.shape[0], num_ghost_nodes), dtype=np.float64)
        water_depth = np.concatenate([water_depth, ghost_depth], axis=1)

    timesteps = np.asarray(np.loadtxt(row["Hydrograph_Filepath"])[:, 0], dtype=np.float64)
    base_interval = int(round(float(timesteps[1] - timesteps[0])))
    if timestep_interval % base_interval != 0:
        raise ValueError(f"Configured timestep_interval={timestep_interval} is incompatible with {base_interval}.")

    start = 0
    run_id = str(row["Run_ID"])
    if isinstance(spin_up_time, int):
        start = spin_up_time // base_interval
    elif isinstance(spin_up_time, dict):
        start = int(spin_up_time.get(run_id, spin_up_time.get("default", 0))) // base_interval

    trimmed = water_depth[start:]
    step = timestep_interval // base_interval
    if step > 1:
        trimmed_length = (trimmed.shape[0] // step) * step
        trimmed = trimmed[:trimmed_length].reshape(-1, step, trimmed.shape[1]).mean(axis=1)

    label_start = previous_timesteps + 1 + rollout_start
    label_end = label_start + (num_timesteps if rollout_timesteps is None else int(rollout_timesteps))
    target = trimmed[label_start:label_end]
    if target.shape[0] != num_timesteps:
        raise ValueError(
            f"Raw depth target for run {run_id} has {target.shape[0]} timesteps; expected {num_timesteps}."
        )
    return target


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.nanmean(np.abs(pred - target)))


# The paper averages the error within each timestep first, then across timesteps. That is
# not the same as pooling every cell-timestep into one mean, so keep both forms distinct.
def timestep_mean_mae(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.ndim == 1:
        return mae(pred, target)
    return float(np.nanmean(np.nanmean(np.abs(pred - target), axis=1)))


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((pred - target) ** 2)))


def timestep_mean_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.ndim == 1:
        return rmse(pred, target)
    return float(np.nanmean(np.sqrt(np.nanmean((pred - target) ** 2, axis=1))))


def nse(pred: np.ndarray, target: np.ndarray) -> float:
    denom = np.nansum((target - np.nanmean(target)) ** 2)
    if denom <= EPS:
        return float("nan")
    return float(1.0 - np.nansum((target - pred) ** 2) / denom)


def timestep_mean_nse(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.ndim == 1:
        return nse(pred, target)
    return float(np.nanmean([nse(pred[t], target[t]) for t in range(pred.shape[0])]))


def csi_global(pred_depth: np.ndarray, target_depth: np.ndarray, threshold: float) -> float:
    binary_pred = pred_depth > threshold
    binary_target = target_depth > threshold
    tp = np.logical_and(binary_pred, binary_target).sum()
    fp = np.logical_and(binary_pred, ~binary_target).sum()
    fn = np.logical_and(~binary_pred, binary_target).sum()
    denom = tp + fp + fn
    return float(tp / denom) if denom else float("nan")


def csi_timestep_mean(pred_depth: np.ndarray, target_depth: np.ndarray, threshold: float) -> float:
    return float(np.nanmean([csi_global(pred_depth[t], target_depth[t], threshold) for t in range(pred_depth.shape[0])]))


def summarize_file(
    metric_file: Path,
    split_records: dict[str, dict[str, Any]],
    config: dict[str, Any],
    target_source: str,
    thresholds: list[float],
    csi_aggregation: str,
) -> dict[str, Any]:
    run_id = parse_run_id(metric_file)
    if run_id not in split_records:
        raise KeyError(f"Run id {run_id} from {metric_file.name} is not present in the split CSV.")

    row = split_records[run_id]
    data = np.load(metric_file, allow_pickle=True)
    pred_volume = data["pred"]
    target_volume = data["target"]
    edge_pred = squeeze_last_dim(np.asarray(data["edge_pred"], dtype=np.float64))
    edge_target = squeeze_last_dim(np.asarray(data["edge_target"], dtype=np.float64))

    depth_pred = volume_to_depth(pred_volume, row["Cells_Shp_Filepath"])
    if target_source == "saved_volume":
        depth_target = volume_to_depth(target_volume, row["Cells_Shp_Filepath"])
    else:
        depth_target = raw_depth_target(row, config, depth_pred.shape[0])

    num_timesteps = min(depth_pred.shape[0], depth_target.shape[0])
    num_nodes = min(depth_pred.shape[1], depth_target.shape[1])
    depth_pred = depth_pred[:num_timesteps, :num_nodes]
    depth_target = depth_target[:num_timesteps, :num_nodes]

    num_edge_timesteps = min(edge_pred.shape[0], edge_target.shape[0])
    num_edges = min(edge_pred.shape[1], edge_target.shape[1])
    edge_pred = edge_pred[:num_edge_timesteps, :num_edges]
    edge_target = edge_target[:num_edge_timesteps, :num_edges]
    q_pred = discharge_to_unit_q(edge_pred, row["Edges_Shp_Filepath"])
    q_target = discharge_to_unit_q(edge_target, row["Edges_Shp_Filepath"])
    num_q_timesteps = min(q_pred.shape[0], q_target.shape[0])
    num_q_edges = min(q_pred.shape[1], q_target.shape[1])
    q_pred = q_pred[:num_q_timesteps, :num_q_edges]
    q_target = q_target[:num_q_timesteps, :num_q_edges]

    result: dict[str, Any] = {
        "metric_file": str(metric_file),
        "run_id": run_id,
        "num_timesteps": int(num_timesteps),
        "num_nodes": int(num_nodes),
        "num_edges": int(num_edges),
        "depth_mae_m": timestep_mean_mae(depth_pred, depth_target),
        "depth_rmse_m": timestep_mean_rmse(depth_pred, depth_target),
        "depth_nse": timestep_mean_nse(depth_pred, depth_target),
        "flow_mae_m2_s": timestep_mean_mae(q_pred, q_target),
        "flow_rmse_m2_s": timestep_mean_rmse(q_pred, q_target),
        "flow_nse": timestep_mean_nse(q_pred, q_target),
        "discharge_mae_m3_s": timestep_mean_mae(edge_pred, edge_target),
        "discharge_rmse_m3_s": timestep_mean_rmse(edge_pred, edge_target),
        "discharge_nse": timestep_mean_nse(edge_pred, edge_target),
    }

    csi_func = csi_timestep_mean if csi_aggregation == "timestep_mean" else csi_global
    for threshold in thresholds:
        key = f"depth_csi_{threshold:.2f}m"
        value = csi_func(depth_pred, depth_target, threshold)
        result[key] = value
        result[f"{key}_percent"] = value * 100.0 if np.isfinite(value) else float("nan")

    return result


def aggregate(rows: list[dict[str, Any]], label: str, reducer) -> dict[str, Any]:
    keys = [k for k in rows[0].keys() if k not in {"metric_file", "run_id"}]
    out: dict[str, Any] = {"run_id": label}
    for key in keys:
        values = [row[key] for row in rows]
        out[key] = float(reducer(values))
    return out


def write_outputs(rows: list[dict[str, Any]], mean: dict[str, Any], std: dict[str, Any], output_csv: Path, output_json: Path, metadata: dict[str, Any]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({key: mean.get(key, "") for key in fieldnames})
        writer.writerow({key: std.get(key, "") for key in fieldnames})

    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "events": rows, "mean": mean, "std": std}
    output_json.write_text(json.dumps(payload, indent=2))


def log_wandb(args: argparse.Namespace, mean: dict[str, Any], std: dict[str, Any], event_count: int) -> None:
    if not args.wandb_project:
        return
    import wandb

    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode

    init_kwargs: dict[str, Any] = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "name": args.wandb_run_name,
    }
    if args.wandb_run_id:
        init_kwargs["id"] = args.wandb_run_id
        init_kwargs["resume"] = args.wandb_resume

    run = wandb.init(**{k: v for k, v in init_kwargs.items() if v is not None})
    prefix = args.wandb_prefix.strip("/")
    payload = {f"{prefix}/event_count": event_count}
    for scope, summary in (("mean", mean), ("std", std)):
        for key, value in summary.items():
            if key == "run_id":
                continue
            payload[f"{prefix}/{scope}_{key}"] = value
    # Let W&B append at the next valid history step when resuming an old run.
    # Explicitly choosing epoch + 1 can be out-of-order after previous resumes.
    run.log(payload)
    run.summary.update(payload)
    print(f"Logged {len(payload)} metrics to W&B history and summary.")
    run.finish()


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    root_dir, metrics_dir, split_csv, output_csv, output_json = resolve_paths(args, config)
    split_records = load_split_records(root_dir, split_csv)
    metric_files = sorted(metrics_dir.glob(args.metrics_glob), key=run_sort_key)
    if not metric_files:
        raise FileNotFoundError(f"No files matching {args.metrics_glob!r} found in {metrics_dir}")

    run_ids = [parse_run_id(path) for path in metric_files]
    duplicate_run_ids = sorted({run_id for run_id in run_ids if run_ids.count(run_id) > 1})
    if duplicate_run_ids and not args.allow_duplicate_run_ids:
        raise ValueError(
            "Multiple metric files have the same run_id. This usually means metrics_dir contains outputs "
            f"from multiple checkpoints. Duplicate run_ids: {duplicate_run_ids[:10]}. "
            "Pass --metrics_glob with a checkpoint-specific prefix, or use --allow_duplicate_run_ids."
        )

    rows = [
        summarize_file(
            metric_file=path,
            split_records=split_records,
            config=config,
            target_source=args.target_source,
            thresholds=args.thresholds,
            csi_aggregation=args.csi_aggregation,
        )
        for path in metric_files
    ]
    mean = aggregate(rows, "mean", np.nanmean)
    std = aggregate(rows, "std", np.nanstd)
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metrics_dir": str(metrics_dir),
        "metrics_glob": args.metrics_glob,
        "root_dir": str(root_dir),
        "split_csv": split_csv,
        "config": args.config,
        "target_source": args.target_source,
        "csi_aggregation": args.csi_aggregation,
        "thresholds_m": args.thresholds,
        "event_count": len(rows),
    }

    write_outputs(rows, mean, std, output_csv, output_json, metadata)
    print(f"Saved CSV summary to: {output_csv}")
    print(f"Saved JSON summary to: {output_json}")
    print(json.dumps({"mean": mean, "std": std}, indent=2))
    log_wandb(args, mean, std, len(rows))


if __name__ == "__main__":
    main()
