#!/bin/bash -l
#
# Step 1 (CPU): build ~360k top-up prompts + hf-disk dataset (~300M tokens).
# Submit:  sbatch slurm/01_prepare_topup.sbatch
#
#SBATCH --job-name=dm-synth-topup-prepare
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH -p alpha
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
set -euo pipefail

WS_ROOT="/data/horse/ws/hama901h-whittle"
ROOT="${WS_ROOT}/inference-hive/examples/datamix_1b_synthetic"
cd "${ROOT}"
mkdir -p logs topup

source "${WS_ROOT}/cache.sh"
export TOKENIZERS_PARALLELISM=false

echo "=== top-up prepare: $(date) ==="
./run_pipeline_topup.sh plan
./run_pipeline_topup.sh prepare

echo "=== done: $(date) ==="
echo "Dataset: ${ROOT}/topup/dataset/datamix-synth-completion"
echo "Next: sbatch slurm/02_validate_create_run_topup.sbatch"
