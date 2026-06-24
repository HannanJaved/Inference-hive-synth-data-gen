#!/bin/bash -l
#
# Step 2 (CPU): validate inference-hive config/data and generate ih_job.slurm.
# Submit after step 1 completes:
#   sbatch pilot/slurm/02_validate_create_run.sbatch
# Or with dependency:
#   sbatch --dependency=afterok:<PREPARE_JOBID> slurm/02_validate_create_run.sbatch
#
# SLURM CONFIG (aligned with whittle-paper/launch_search.sh) -----------------
#SBATCH --job-name=dm-synth-pilot-setup
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH -p alpha
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# -----------------------------------------------------------------------------
#
set -euo pipefail

WS_ROOT="/data/horse/ws/hama901h-whittle"
ROOT="${WS_ROOT}/inference-hive/examples/datamix_1b_synthetic/pilot"
RUN_DIR="${ROOT}/runs/datamix_pilot"
cd "${ROOT}"
mkdir -p logs

source "${WS_ROOT}/cache.sh"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== pilot validate + create-run: $(date) ==="
./run_pilot_pipeline.sh validate
./run_pilot_pipeline.sh create-run

echo ""
echo "=== Review before GPU submit ==="
echo "  Config copy:  ${RUN_DIR}/ih_config.yaml"
echo "  SLURM script: ${RUN_DIR}/ih_job.slurm"
echo ""
echo "Inspect those files, then submit inference (login node):"
echo "  ${ROOT}/submit_inference.sh"
echo "=== done: $(date) ==="
