#!/usr/bin/env python3
"""Export fixed Optuna candidates for the 60-vs-150 epoch pilot.

The pilot reruns the same parameter vectors at two training budgets and then
compares their rankings. This exporter supports both older scalar Optuna
studies and newer multi-objective studies.
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Any

import optuna
import yaml


METRIC_ALIASES = {
    "depth_mae": ("avg_depth_mae", "best_validation_depth_mae"),
    "depth_rmse": ("avg_depth_rmse", "best_validation_depth_rmse"),
    "node_rmse": ("avg_node_rmse",),
    "edge_rmse": ("avg_edge_rmse",),
    "unit_discharge_mae": ("avg_unit_discharge_mae", "best_validation_unit_discharge_mae"),
    "unit_discharge_rmse": ("avg_unit_discharge_rmse", "best_validation_unit_discharge_rmse"),
}


def _param_signature(params: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(params.items()))


def _objective_metric_names(trial: optuna.trial.FrozenTrial) -> list[str]:
    names = trial.user_attrs.get("objective_metrics") or trial.user_attrs.get("objective_metric")
    if isinstance(names, str):
        return [names]
    if isinstance(names, (list, tuple)):
        return list(names)
    if trial.values is not None and len(trial.values) == 2:
        return ["depth_mae", "unit_discharge_mae"]
    return ["depth_mae"]


def _metric_value(trial: optuna.trial.FrozenTrial, metric: str) -> float | None:
    for attr_name in METRIC_ALIASES.get(metric, (f"avg_{metric}",)):
        value = trial.user_attrs.get(attr_name)
        if value is not None:
            value = float(value)
            return value if math.isfinite(value) else None

    names = _objective_metric_names(trial)
    if trial.values is not None and metric in names:
        value = float(trial.values[names.index(metric)])
        return value if math.isfinite(value) else None

    if metric == names[0] and trial.value is not None:
        value = float(trial.value)
        return value if math.isfinite(value) else None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Export top distinct configs for the budget-ranking pilot.")
    parser.add_argument("--storage", required=True, help="Optuna storage URL, e.g. sqlite:///optuna_results/study.db")
    parser.add_argument("--study_name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top_k", type=int, default=12)
    parser.add_argument("--rank_metric", default="depth_mae", choices=sorted(METRIC_ALIASES))
    parser.add_argument("--secondary_metric", default="unit_discharge_mae", choices=sorted(METRIC_ALIASES))
    args = parser.parse_args()

    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    rows = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE or not trial.params:
            continue
        primary = _metric_value(trial, args.rank_metric)
        if primary is None:
            continue
        secondary = _metric_value(trial, args.secondary_metric)
        rows.append((primary, float("inf") if secondary is None else secondary, trial))

    rows.sort(key=lambda item: (item[0], item[1], item[2].number))

    candidates = []
    seen = set()
    for primary, secondary, trial in rows:
        signature = _param_signature(trial.params)
        if signature in seen:
            continue
        seen.add(signature)
        candidates.append({
            "source_trial_number": trial.number,
            "source_study_name": args.study_name,
            "source_storage": args.storage,
            "rank_metric": args.rank_metric,
            "rank_metric_value": float(primary),
            "secondary_metric": args.secondary_metric,
            "secondary_metric_value": None if math.isinf(secondary) else float(secondary),
            "source_values": [float(value) for value in trial.values] if trial.values is not None else None,
            "source_user_attrs": dict(trial.user_attrs),
            "params": dict(trial.params),
        })
        if len(candidates) >= args.top_k:
            break

    if not candidates:
        raise RuntimeError("No complete finite trials found; cannot prepare pilot candidates.")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as output_file:
        yaml.safe_dump({"candidates": candidates}, output_file, sort_keys=False)
    print(f"Wrote {len(candidates)} candidates to {args.output}")
    for idx, candidate in enumerate(candidates):
        print(
            f"{idx}: source_trial={candidate['source_trial_number']} "
            f"{args.rank_metric}={candidate['rank_metric_value']:.6g} "
            f"params={candidate['params']}"
        )


if __name__ == "__main__":
    main()
