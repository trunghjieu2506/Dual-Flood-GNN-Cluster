#!/bin/bash
#SBATCH --job-name=mswe-cl-best
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40:1
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --output=logs/mswegnn_cluster_best-%j.out
#SBATCH --error=logs/mswegnn_cluster_best-%j.err

set -euo pipefail
cd /mnt/scratch/n/nthieu/Dual-Flood-GNN-Cluster
mkdir -p logs saved_metrics/mswegnn_cluster_best training_stats/mswegnn_cluster_best saved_models/mswegnn_cluster_best configs/runtime

if [ -f "$HOME/.bashrc" ]; then
  . "$HOME/.bashrc"
fi

export WANDB_PROJECT="${WANDB_PROJECT:-dual-flood-gnn-cluster}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-mswe-cluster-physics-timesteps_larger_local_loss_scale${SLURM_JOB_ID:-manual}}"

if [ -f venv/bin/activate ]; then
  . venv/bin/activate
else
  . ../dual_flood_gnn/venv/bin/activate
fi


cmd=(
  python train_cluster.py
  --config configs/rerun150_test_backfill/trial_10.yaml
  --model DUALFloodGNN
  --device cuda
  --use_cluster_gcn
  --num_clusters 20
  --batching_strategy cross_event
  --clusters_per_batch 5
  --sliding
  --seed 42
  --amp none
  --collect_regression_metrics
  --with_test
  --wandb_tags mswe cluster boundary-aware best
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

printf 'Resolved command:\n  %q' "${cmd[0]}"
for arg in "${cmd[@]:1}"; do
  printf ' %q' "$arg"
done
printf '\n'

srun "${cmd[@]}"
