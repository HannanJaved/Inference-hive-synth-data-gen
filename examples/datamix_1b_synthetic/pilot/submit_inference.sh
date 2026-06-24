#!/usr/bin/env bash
#
# Step 3 (login node): submit the inference-hive GPU job(s).
# DOES NOT auto-run — review ih_job.slurm first, then execute:
#
#   cd /data/horse/ws/hama901h-whittle/inference-hive/examples/datamix_1b_synthetic/pilot
#   ./submit_inference.sh
#
# This calls inference-hive submit.py, which sbatch's runs/datamix_pilot/ih_job.slurm
# with --array 0 (single shard for pilot).
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${ROOT}/runs/datamix_pilot"
IH_JOB="${RUN_DIR}/ih_job.slurm"

if [[ ! -f "${IH_JOB}" ]]; then
  echo "Missing ${IH_JOB}"
  echo "Run step 2 first: sbatch slurm/02_validate_create_run.sbatch"
  exit 1
fi

echo "=== Files to review ==="
echo "  ${ROOT}/config_datamix_pilot.yaml"
echo "  ${RUN_DIR}/ih_config.yaml"
echo "  ${IH_JOB}"
echo ""
read -r -p "Submit pilot inference job? [y/N] " confirm
if [[ "${confirm}" != [yY] ]]; then
  echo "Aborted."
  exit 0
fi

cd "${ROOT}"
./run_pilot_pipeline.sh submit 1

echo ""
echo "Monitor: ./run_pilot_pipeline.sh status"
echo "Logs:    ${RUN_DIR}/logs/"
