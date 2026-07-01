#!/bin/bash -l
#
# Submit 16-GPU inference array, then chain postprocess when all shards finish.
# Requires runs/datamix_3b_regen/ih_job.slurm from step 02.
#
#   SETUP=$(sbatch slurm/02_validate_create_run.sbatch | awk '{print $NF}')
#   sbatch --dependency=afterok:${SETUP} slurm/03_submit_inference.sbatch
#
#SBATCH --job-name=dm-3b-regen-infer
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -p alpha
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
set -euo pipefail

WS_ROOT="/data/horse/ws/hama901h-whittle"
ROOT="${WS_ROOT}/inference-hive/examples/datamix_3b_regen"
IH_ROOT="${WS_ROOT}/inference-hive"
RUN_DIR="${ROOT}/runs/datamix_3b_regen"
IH_JOB="${RUN_DIR}/ih_job.slurm"

cd "${ROOT}"
mkdir -p logs "${RUN_DIR}/logs"

if [[ ! -f "${IH_JOB}" ]]; then
  echo "ERROR: missing ${IH_JOB} — run slurm/02_validate_create_run.sbatch first."
  exit 1
fi

echo "=== 3B regen submit inference: $(date) ==="
# ih_job.slurm runs validate_data.py and run_inference.py relative to inference-hive root
cd "${IH_ROOT}"
ARRAY_JOB=$(sbatch --parsable --array=0-15 "${IH_JOB}")
echo "Inference array job: ${ARRAY_JOB}"

POST_JOB=$(sbatch --parsable --dependency="afterok:${ARRAY_JOB}" --chdir="${ROOT}" slurm/04_postprocess.sbatch)
echo "Postprocess job (after inference): ${POST_JOB}"
echo "=== done: $(date) ==="
