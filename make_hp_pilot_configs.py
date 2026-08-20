#!/usr/bin/env python3
"""Create 60/150 epoch pilot base configs from the current Stage 1 base."""

from __future__ import annotations

from pathlib import Path

import yaml


def main() -> None:
    base_path = Path("configs/mswegnn_cluster_hp_cv_stage1_base.yaml")
    for budget, max_curriculum in ((60, 10), (150, 25)):
        cfg = yaml.safe_load(base_path.read_text())
        cfg["dataset_parameters"]["training"]["dataset_summary_file"] = "train_split/train_split.csv"
        cfg["dataset_parameters"]["validation"] = {
            "dataset_summary_file": "train_split/val_split.csv",
            "event_stats_file": "boundary_aware_cv_val_event_stats.yaml",
        }
        cfg["training_parameters"]["log_path"] = f"logs/mswegnn_cluster_hp_pilot{budget}.log"
        cfg["training_parameters"]["model_dir"] = f"saved_models/mswegnn_cluster_hp_pilot{budget}"
        cfg["training_parameters"]["stats_dir"] = f"training_stats/mswegnn_cluster_hp_pilot{budget}"
        cfg["training_parameters"]["num_epochs"] = budget
        cfg["training_parameters"]["early_stopping_patience"] = 15
        cfg["training_parameters"]["autoregressive"]["max_curriculum_epochs"] = max_curriculum
        cfg["testing_parameters"]["output_dir"] = f"saved_metrics/mswegnn_cluster_hp_pilot{budget}"
        output_path = Path(f"configs/mswegnn_cluster_hp_pilot{budget}_base.yaml")
        output_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
