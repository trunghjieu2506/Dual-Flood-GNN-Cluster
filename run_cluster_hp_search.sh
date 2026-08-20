#!/bin/bash
#SBATCH --job-name=mswe-cluster-screen
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=100G
#SBATCH --time=72:00:00
#SBATCH --array=0-7
#SBATCH --output=slurm_logs/mswe_cluster_screening_%A_%a.out
#SBATCH --error=slurm_logs/mswe_cluster_screening_%A_%a.err

set -euo pipefail

cd /mnt/scratch/n/nthieu/Dual-Flood-GNN-Cluster
mkdir -p logs slurm_logs optuna_results saved_models/mswegnn_cluster_hp_screening training_stats/mswegnn_cluster_hp_screening saved_metrics/mswegnn_cluster_hp_screening

source ~/.bashrc
if [ -f ../dual_flood_gnn/venv/bin/activate ]; then
  source ../dual_flood_gnn/venv/bin/activate
elif [ -f venv/bin/activate ]; then
  source venv/bin/activate
fi

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HP_SEARCH_UNIQUE_CACHE=1
export WANDB_PROJECT="${WANDB_PROJECT:-dual-flood-gnn-cluster}"
export WANDB_MODE="${WANDB_MODE:-online}"

TOTAL_TRIALS="${TOTAL_TRIALS:-160}"
NUM_WORKERS="${SLURM_ARRAY_TASK_COUNT:-4}"
TRIALS_PER_WORKER=$(((TOTAL_TRIALS + NUM_WORKERS - 1) / NUM_WORKERS))

echo "Worker ${SLURM_ARRAY_TASK_ID:-0}/${NUM_WORKERS}: ${TRIALS_PER_WORKER} trials toward ${TOTAL_TRIALS} total."

python hp_search_cluster.py \
  --config configs/mswegnn_cluster_hp.yaml \
  --hparam_config configs/hparam_config/cluster_hp.yaml \
  --model DUALFloodGNN \
  --seed 42 \
  --device cuda \
  --study_name mswe_cluster_screening_60epoch_dual \
  --storage sqlite:///optuna_results/mswe_cluster_screening_60epoch_dual.db \
  --n_trials_per_job "${TRIALS_PER_WORKER}" \
  --use_cluster_gcn \
  --num_clusters 20 \
  --clusters_per_batch 5 \
  --sliding \
  --batching_strategy cross_event \
  --objective_source validation \
  --wandb_run_name_prefix mswe-cluster-screening60 \
  --wandb_tags screening budget60 mswe cluster boundary-aware no-local-loss
