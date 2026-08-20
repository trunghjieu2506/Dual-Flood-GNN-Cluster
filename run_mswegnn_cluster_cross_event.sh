#!/bin/bash
#SBATCH --job-name=mswe-cl-xevt-bs32
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40:1
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --output=logs/mswegnn_cluster_boundary_aware_cross_event_bs32-%j.out
#SBATCH --error=logs/mswegnn_cluster_boundary_aware_cross_event_bs32-%j.err

set -euo pipefail
cd /mnt/scratch/n/nthieu/Dual-Flood-GNN-Cluster
mkdir -p logs saved_metrics/mswegnn_cluster_boundary_aware_cross_event_bs32 training_stats/mswegnn_cluster_boundary_aware_cross_event_bs32 saved_models/mswegnn_cluster_boundary_aware_cross_event_bs32

if [ -f "$HOME/.bashrc" ]; then
  . "$HOME/.bashrc"
fi

export WANDB_PROJECT="${WANDB_PROJECT:-dual-flood-gnn-cluster}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-mswe-cluster-boundary-aware-cross-event-bs32-${SLURM_JOB_ID:-manual}}"

if [ -f venv/bin/activate ]; then
  . venv/bin/activate
else
  . ../dual_flood_gnn/venv/bin/activate
fi

cmd=(
  python train_cluster.py
  --config configs/mswegnn_cluster_cross_event.yaml
  --model DUALFloodGNN
  --device cuda
  --use_cluster_gcn
  --num_clusters 30
  --clusters_per_batch 5
  --sliding
  --batching_strategy cross_event
  --seed 42
  --amp none
  --with_test
  --collect_regression_metrics
  --wandb_tags mswe cluster boundary-aware cross-event bs32
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
