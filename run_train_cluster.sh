#!/bin/bash
#SBATCH --job-name=train-cluster
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40:1
#SBATCH --mem=64G
#SBATCH --time=1440
#SBATCH --output=logs/train_cluster-%j.out
#SBATCH --error=logs/train_cluster-%j.err

set -euo pipefail

mkdir -p logs saved_metrics training_stats saved_models

if [ -f "$HOME/.bashrc" ]; then
  . "$HOME/.bashrc"
fi

export WANDB_PROJECT="${WANDB_PROJECT:-dual-flood-gnn-cluster}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-cluster-train-batch-${SLURM_JOB_ID:-manual}}"

if [ -f venv/bin/activate ]; then
  . venv/bin/activate
else
  . ../dual_flood_gnn/venv/bin/activate
fi

cmd=(
  python train_cluster.py
  --config configs/mswegnn_cluster.yaml
  --model DUALFloodGNN
  --device cuda
  --use_cluster_gcn
  --num_clusters 30
  --clusters_per_batch 5
  --sliding
  --seed 42
  --with_test
  --collect_regression_metrics
)

if [ -n "${WANDB_PROJECT:-}" ]; then
  cmd+=(--wandb_project "${WANDB_PROJECT}")
fi

if [ -n "${WANDB_ENTITY:-}" ]; then
  cmd+=(--wandb_entity "${WANDB_ENTITY}")
fi

if [ -n "${WANDB_RUN_NAME:-}" ]; then
  cmd+=(--wandb_run_name "${WANDB_RUN_NAME}")
fi

if [ -n "${WANDB_MODE:-}" ]; then
  cmd+=(--wandb_mode "${WANDB_MODE}")
fi

srun "${cmd[@]}"
