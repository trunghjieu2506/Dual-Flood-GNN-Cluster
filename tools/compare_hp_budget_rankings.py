#!/usr/bin/env python3
"""Compare fixed-config Optuna rankings between two training budgets."""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any

import optuna
import pandas as pd


METRIC_ALIASES = {
    "depth_mae": ("avg_depth_mae", "best_validation_depth_mae"),
    "depth_rmse": ("avg_depth_rmse", "best_validation_depth_rmse"),
    "node_rmse": ("avg_node_rmse",),
    "edge_rmse": ("avg_edge_rmse",),
    "unit_discharge_mae": ("avg_unit_discharge_mae", "best_validation_unit_discharge_mae"),
    "unit_discharge_rmse": ("avg_unit_discharge_rmse", "best_validation_unit_discharge_rmse"),
}


def _signature(params: dict[str, Any]) -> str:
    return json.dumps(dict(sorted(params.items())), sort_keys=True)


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


def _study_rows(storage: str, study_name: str, label: str) -> pd.DataFrame:
    study = optuna.load_study(study_name=study_name, storage=storage)
    rows = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE or not trial.params:
            continue
        row = {
            "signature": _signature(trial.params),
            f"{label}_trial_number": trial.number,
            f"{label}_duration_s": trial.duration.total_seconds() if trial.duration else None,
        }
        for key, value in trial.params.items():
            row[f"param_{key}"] = value
        for metric in METRIC_ALIASES:
            row[f"{label}_{metric}"] = _metric_value(trial, metric)
        rows.append(row)
    return pd.DataFrame(rows)


def _fmt(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "nan"
    return f"{value:.6g}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare pilot rankings between two Optuna studies.")
    parser.add_argument("--budget_a_storage", required=True)
    parser.add_argument("--budget_a_study", required=True)
    parser.add_argument("--budget_a_label", default="epoch60")
    parser.add_argument("--budget_b_storage", required=True)
    parser.add_argument("--budget_b_study", required=True)
    parser.add_argument("--budget_b_label", default="epoch150")
    parser.add_argument("--primary_metric", default="depth_mae", choices=sorted(METRIC_ALIASES))
    parser.add_argument("--secondary_metric", default="unit_discharge_mae", choices=sorted(METRIC_ALIASES))
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--output_csv", default="result_summaries/hp_pilot_budget_rankings.csv")
    parser.add_argument("--output_txt", default="result_summaries/hp_pilot_budget_rankings.txt")
    args = parser.parse_args()

    budget_a = _study_rows(args.budget_a_storage, args.budget_a_study, args.budget_a_label)
    budget_b = _study_rows(args.budget_b_storage, args.budget_b_study, args.budget_b_label)
    if budget_a.empty or budget_b.empty:
        raise RuntimeError(
            f"Cannot compare empty studies: {args.budget_a_label}={len(budget_a)}, "
            f"{args.budget_b_label}={len(budget_b)}"
        )

    merged = budget_a.merge(budget_b, on="signature", suffixes=("", "_b"))
    if merged.empty:
        raise RuntimeError("No identical parameter signatures found between studies.")

    primary_a = f"{args.budget_a_label}_{args.primary_metric}"
    primary_b = f"{args.budget_b_label}_{args.primary_metric}"
    secondary_a = f"{args.budget_a_label}_{args.secondary_metric}"
    secondary_b = f"{args.budget_b_label}_{args.secondary_metric}"
    merged[f"{args.budget_a_label}_rank"] = merged[primary_a].rank(method="min")
    merged[f"{args.budget_b_label}_rank"] = merged[primary_b].rank(method="min")
    merged["rank_delta"] = merged[f"{args.budget_b_label}_rank"] - merged[f"{args.budget_a_label}_rank"]
    merged["primary_metric_delta"] = merged[primary_b] - merged[primary_a]

    param_cols = sorted([col for col in merged.columns if col.startswith("param_")])
    keep_cols = [
        "signature",
        f"{args.budget_a_label}_trial_number",
        f"{args.budget_b_label}_trial_number",
        f"{args.budget_a_label}_rank",
        f"{args.budget_b_label}_rank",
        "rank_delta",
        primary_a,
        primary_b,
        "primary_metric_delta",
        secondary_a,
        secondary_b,
        f"{args.budget_a_label}_duration_s",
        f"{args.budget_b_label}_duration_s",
        *param_cols,
    ]
    merged = merged[keep_cols].sort_values([f"{args.budget_b_label}_rank", f"{args.budget_a_label}_rank"])

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    merged.to_csv(args.output_csv, index=False)

    spearman = merged[[f"{args.budget_a_label}_rank", f"{args.budget_b_label}_rank"]].corr(method="spearman").iloc[0, 1]
    top_a = set(merged.nsmallest(args.top_k, f"{args.budget_a_label}_rank")["signature"])
    top_b = set(merged.nsmallest(args.top_k, f"{args.budget_b_label}_rank")["signature"])
    overlap = len(top_a.intersection(top_b))

    lines = [
        "Hyperparameter Budget Ranking Pilot",
        f"Compared configs: {len(merged)}",
        f"Primary metric: {args.primary_metric}",
        f"Secondary metric: {args.secondary_metric}",
        f"Spearman rank correlation: {_fmt(float(spearman))}",
        f"Top-{args.top_k} overlap: {overlap}/{args.top_k}",
        "",
        "Top rows by longer-budget rank:",
    ]
    display_cols = [
        f"{args.budget_a_label}_rank",
        f"{args.budget_b_label}_rank",
        primary_a,
        primary_b,
        secondary_a,
        secondary_b,
        "rank_delta",
        *param_cols,
    ]
    lines.append(merged[display_cols].head(20).to_string(index=False))

    with open(args.output_txt, "w") as output_file:
        output_file.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Wrote CSV to {args.output_csv}")
    print(f"Wrote text summary to {args.output_txt}")


if __name__ == "__main__":
    main()
