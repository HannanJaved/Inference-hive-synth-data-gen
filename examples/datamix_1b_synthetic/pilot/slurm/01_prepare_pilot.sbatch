#!/bin/bash -l
#
# Step 1 (CPU): build pilot prompts + hf-disk dataset.
# Submit:  sbatch pilot/slurm/01_prepare_pilot.sbatch
#
# SLURM CONFIG (aligned with whittle-paper/launch_search.sh) -----------------
#SBATCH --job-name=dm-synth-pilot-prepare
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH -p alpha
#SBATCH --requeue
#SBATCH --gres=gpu:1
# Tip: %x = job name, %j = job ID
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# -----------------------------------------------------------------------------
#
# Optional: chain step 2 after this job succeeds (replace JOBID after submitting):
#   sbatch --dependency=afterok:JOBID slurm/02_validate_create_run.sbatch
#
set -euo pipefail

WS_ROOT="/data/horse/ws/hama901h-whittle"
ROOT="${WS_ROOT}/inference-hive/examples/datamix_1b_synthetic/pilot"
cd "${ROOT}"
mkdir -p logs

source "${WS_ROOT}/cache.sh"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== pilot prepare: $(date) ==="
./run_pilot_pipeline.sh plan
./run_pilot_pipeline.sh prepare
echo "=== done: $(date) ==="
echo "Dataset: ${ROOT}/dataset/datamix-synth-completion"
echo "Next: sbatch slurm/02_validate_create_run.sbatch"
