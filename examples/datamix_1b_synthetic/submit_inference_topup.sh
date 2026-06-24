#!/usr/bin/env bash
#
# Submit 16 GPU shard jobs for the 300M top-up run.
#
#   cd .../datamix_1b_synthetic
#   ./submit_inference_topup.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${ROOT}/runs/datamix_1b_topup"
IH_JOB="${RUN_DIR}/ih_job.slurm"
IH_ROOT="$(cd "${ROOT}/../.." && pwd)"

if [[ ! -f "${IH_JOB}" ]]; then
  echo "Missing ${IH_JOB}"
  echo "Run step 2 first: sbatch slurm/02_validate_create_run_topup.sbatch"
  exit 1
fi

echo "=== Files to review ==="
echo "  ${ROOT}/config_datamix_topup.yaml"
echo "  ${RUN_DIR}/ih_config.yaml"
echo "  ${IH_JOB}"
echo ""
echo "This will sbatch up to 16 GPU shards (gpu:1 each, array 0-15)."
echo "Outputs go to: ${ROOT}/topup/ih_outputs/"
echo ""
read -r -p "Submit 300M top-up inference shard jobs? [y/N] " confirm
if [[ "${confirm}" != [yY] ]]; then
  echo "Aborted."
  exit 0
fi

cd "${IH_ROOT}"
"${IH_ROOT}/.venv/bin/python" submit.py --run-dir "${RUN_DIR}"

echo ""
echo "Monitor: cd ${ROOT} && ./run_pipeline_topup.sh status"
echo "Logs:    ${RUN_DIR}/logs/"
