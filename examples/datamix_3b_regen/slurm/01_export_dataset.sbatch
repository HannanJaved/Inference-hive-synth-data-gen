#!/bin/bash -l
#
# Export prompts.jsonl -> dataset/datamix-synth-completion (after prepare).
# Use when prepare wrote prompts but HF export failed or was skipped.
#
#   sbatch slurm/01_export_dataset.sbatch
#   EXPORT=$(sbatch slurm/01_export_dataset.sbatch | awk '{print $NF}')
#   sbatch --dependency=afterok:${EXPORT} slurm/02_validate_create_run.sbatch
#
#SBATCH --job-name=dm-3b-regen-export
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH -p alpha
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
set -euo pipefail

WS_ROOT="/data/horse/ws/hama901h-whittle"
ROOT="${WS_ROOT}/inference-hive/examples/datamix_3b_regen"
cd "${ROOT}"
mkdir -p logs

source "${WS_ROOT}/cache.sh"
export TOKENIZERS_PARALLELISM=false

echo "=== 3B regen export HF dataset: $(date) ==="
if [[ ! -s prompts/prompts.jsonl ]]; then
  echo "ERROR: prompts/prompts.jsonl missing or empty. Run prepare first."
  exit 1
fi
wc -l prompts/prompts.jsonl
./run_pipeline.sh export-dataset
echo "=== done: $(date) ==="
echo "Next: sbatch slurm/02_validate_create_run.sbatch"
