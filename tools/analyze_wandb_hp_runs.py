#!/usr/bin/env python3
"""Export and summarize W&B runs for ClusterGNN hyperparameter analysis."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_HISTORY_METRICS = [
    "train/loss",
    "train/node_prediction_loss",
    "train/edge_prediction_loss",
    "train/epoch_duration_s",
    "val/node_rmse",
    "val/node_depth_rmse",
    "val/node_depth_mae",
    "val/edge_rmse",
    "val/node_loss",
    "val/edge_loss",
    "test/mean_depth_mae_m",
    "test/mean_depth_rmse_m",
    "test/mean_flow_mae_m2_s",
    "test/mean_flow_rmse_m2_s",
]


DEFAULT_CONFIG_KEYS = [
    "model",
    "seed",
    "use_cluster_gcn",
    "num_clusters",
    "clusters_per_batch",
    "sliding",
    "config_path",
    "training_config.batch_size",
    "training_config.learning_rate",
    "training_config.adam_weight_decay",
    "training_config.gradient_clip_value",
    "training_config.autoregressive.init_num_timesteps",
    "training_config.autoregressive.total_num_timesteps",
    "training_config.autoregressive.timestep_increment",
    "training_config.autoregressive.learning_rate_decay",
    "loss_config.edge_loss_weight",
    "loss_config.edge_pred_loss_scale",
    "loss_config.use_local_mass_loss",
    "loss_config.use_global_mass_loss",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze W&B ClusterGNN hyperparameter runs.")
    parser.add_argument("--project", required=True, help="W&B project, optionally entity/project.")
    parser.add_argument("--entity", default=None, help="W&B entity if project is not entity/project.")
    parser.add_argument("--name_regex", default=None, help="Only include runs whose name matches this regex.")
    parser.add_argument("--tag", action="append", default=[], help="Require this W&B tag. Can be repeated.")
    parser.add_argument("--state", default=None, help="Optional run state filter, e.g. finished.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum runs to fetch.")
    parser.add_argument("--history_samples", type=int, default=10000, help="Max history rows per run.")
    parser.add_argument("--metrics", nargs="*", default=DEFAULT_HISTORY_METRICS, help="History/summary metrics to export.")
    parser.add_argument("--config_keys", nargs="*", default=DEFAULT_CONFIG_KEYS, help="Config keys to flatten into columns.")
    parser.add_argument("--output_csv", default="result_summaries/wandb_hp_runs.csv")
    parser.add_argument("--output_json", default="result_summaries/wandb_hp_runs.json")
    return parser.parse_args()


def nested_get(mapping: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def safe_float(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def summarize_history(history: pd.DataFrame, metric: str) -> dict[str, Any]:
    if metric not in history.columns:
        return {}
    values = pd.to_numeric(history[metric], errors="coerce").dropna()
    if values.empty:
        return {}
    out = {
        f"{metric}__first": float(values.iloc[0]),
        f"{metric}__last": float(values.iloc[-1]),
        f"{metric}__min": float(values.min()),
        f"{metric}__max": float(values.max()),
        f"{metric}__mean": float(values.mean()),
    }
    try:
        out[f"{metric}__argmin_step"] = int(history.loc[values.idxmin()].get("_step", values.idxmin()))
    except Exception:
        pass
    return out


def run_to_row(run, metrics: list[str], config_keys: list[str], history_samples: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_id": run.id,
        "run_name": run.name,
        "state": run.state,
        "created_at": str(run.created_at),
        "url": run.url,
    }
    for key in config_keys:
        row[f"config.{key}"] = nested_get(run.config, key)
    for metric in metrics:
        value = safe_float(run.summary.get(metric))
        if value is not None:
            row[f"summary.{metric}"] = value

    history_keys = ["_step", *metrics]
    try:
        history = run.history(keys=history_keys, samples=history_samples, pandas=True)
    except Exception as exc:
        row["history_error"] = str(exc)
        return row

    row["history_rows"] = int(len(history))
    for metric in metrics:
        row.update(summarize_history(history, metric))
    return row


def main() -> None:
    args = parse_args()
    import wandb

    api = wandb.Api()
    project_path = args.project if "/" in args.project else f"{args.entity}/{args.project}" if args.entity else args.project
    filters: dict[str, Any] = {}
    if args.state:
        filters["state"] = args.state
    if args.tag:
        filters["tags"] = {"$all": args.tag}

    runs = api.runs(project_path, filters=filters, per_page=100)
    name_re = re.compile(args.name_regex) if args.name_regex else None
    rows = []
    for run in runs:
        if name_re and not name_re.search(run.name or ""):
            continue
        rows.append(run_to_row(run, args.metrics, args.config_keys, args.history_samples))
        if args.limit is not None and len(rows) >= args.limit:
            break

    df = pd.DataFrame(rows)
    if not df.empty:
        sort_key = "summary.test/mean_depth_mae_m"
        if sort_key not in df.columns:
            sort_key = "val/node_depth_mae__min" if "val/node_depth_mae__min" in df.columns else None
        if sort_key:
            df = df.sort_values(sort_key, na_position="last")

    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    output_json.write_text(json.dumps(rows, indent=2, default=str))

    print(f"Exported {len(df)} W&B runs to {output_csv}")
    print(f"Exported JSON to {output_json}")
    if not df.empty:
        display_cols = [
            col for col in [
                "run_name",
                "state",
                "summary.test/mean_depth_mae_m",
                "summary.test/mean_depth_rmse_m",
                "val/node_depth_mae__min",
                "val/node_depth_rmse__min",
                "val/node_rmse__min",
                "val/edge_rmse__min",
                "config.training_config.learning_rate",
                "config.clusters_per_batch",
            ] if col in df.columns
        ]
        print(df[display_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
