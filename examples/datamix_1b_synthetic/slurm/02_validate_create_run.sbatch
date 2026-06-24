#!/bin/bash -l
#
# Step 2 (CPU): validate config/data and generate runs/datamix_1b/ih_job.slurm.
# Submit after step 1:
#   sbatch slurm/02_validate_create_run.sbatch
# Or:
#   sbatch --dependency=afterok:<PREPARE_JOBID> slurm/02_validate_create_run.sbatch
#
#SBATCH --job-name=dm-synth-1b-setup
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH -p alpha
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
set -euo pipefail

WS_ROOT="/data/horse/ws/hama901h-whittle"
ROOT="${WS_ROOT}/inference-hive/examples/datamix_1b_synthetic"
IH_ROOT="${WS_ROOT}/inference-hive"
RUN_DIR="${ROOT}/runs/datamix_1b"
cd "${ROOT}"
mkdir -p logs

source "${WS_ROOT}/cache.sh"
export TOKENIZERS_PARALLELISM=false

PYTHON="${IH_ROOT}/.venv/bin/python"

echo "=== 1B validate + create-run: $(date) ==="
cd "${IH_ROOT}"
"${PYTHON}" validate_config.py --config "${ROOT}/config_datamix_1b.yaml"
"${PYTHON}" validate_data.py --config "${ROOT}/config_datamix_1b.yaml"
"${PYTHON}" create_run.py \
  --config "${ROOT}/config_datamix_1b.yaml" \
  --output "${RUN_DIR}" \
  --force

echo ""
echo "=== Review before GPU submit ==="
echo "  Config copy:  ${RUN_DIR}/ih_config.yaml"
echo "  SLURM script: ${RUN_DIR}/ih_job.slurm"
echo ""
echo "Inspect those files, then submit inference (login node):"
echo "  ${ROOT}/submit_inference.sh"
echo "=== done: $(date) ==="
