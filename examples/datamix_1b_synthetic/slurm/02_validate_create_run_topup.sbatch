#!/bin/bash -l
#
# Step 2 (CPU): validate top-up config/data and generate runs/datamix_1b_topup/.
# Submit after step 1:
#   sbatch slurm/02_validate_create_run_topup.sbatch
#
#SBATCH --job-name=dm-synth-topup-setup
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
cd "${ROOT}"
mkdir -p logs

source "${WS_ROOT}/cache.sh"
export TOKENIZERS_PARALLELISM=false

echo "=== top-up validate + create-run: $(date) ==="
./run_pipeline_topup.sh validate
./run_pipeline_topup.sh create-run --force

echo ""
echo "=== Review before GPU submit ==="
echo "  Config copy:  ${ROOT}/runs/datamix_1b_topup/ih_config.yaml"
echo "  SLURM script: ${ROOT}/runs/datamix_1b_topup/ih_job.slurm"
echo ""
echo "Inspect those files, then submit inference (login node):"
echo "  ${ROOT}/submit_inference_topup.sh"
echo "=== done: $(date) ==="
