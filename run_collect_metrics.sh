#!/bin/bash
#SBATCH --job-name=upload_metrics
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/upload_metrics-%j.out
#SBATCH --error=logs/upload_metrics-%j.err

set -euo pipefail
cd /mnt/scratch/n/nthieu/Dual-Flood-GNN-Cluster

source ../dual_flood_gnn/venv/bin/activate
export WANDB_SILENT=true

python3 collect_mswegnn_test_metrics.py \
  --config configs/mswegnn_cluster_test.yaml \
  --metrics_dir saved_metrics/mswegnn_cluster_best/ \
  --metrics_glob "DUALFloodGNN_2026-07-16_22-51-43_*_test_metrics.npz" \
  --wandb_project dual-flood-gnn-cluster \
  --wandb_run_id jgag2tmm \
  --wandb_resume allow \
  --wandb_prefix "test/"
