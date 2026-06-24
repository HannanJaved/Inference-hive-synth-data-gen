#!/bin/bash -l
#
# Step 1 (CPU): build ~1.2M prompts + hf-disk dataset for the 1B run.
# Submit:  sbatch slurm/01_prepare.sbatch
#
# Optional chain step 2:
#   sbatch --dependency=afterok:JOBID slurm/02_validate_create_run.sbatch
#
#SBATCH --job-name=dm-synth-1b-prepare
#SBATCH --time=08:00:00
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
IH_ROOT="${WS_ROOT}/inference-hive"
cd "${ROOT}"
mkdir -p logs

source "${WS_ROOT}/cache.sh"
export TOKENIZERS_PARALLELISM=false

echo "=== 1B prepare: $(date) ==="
"${IH_ROOT}/.venv/bin/python" prepare_dataset.py \
  --mix-config mix_config.yaml \
  --root "${ROOT}" \
  --plan-only
"${IH_ROOT}/.venv/bin/python" prepare_dataset.py \
  --mix-config mix_config.yaml \
  --root "${ROOT}"

echo "=== done: $(date) ==="
echo "Dataset: ${ROOT}/dataset/datamix-synth-completion"
echo "Next: sbatch slurm/02_validate_create_run.sbatch"
