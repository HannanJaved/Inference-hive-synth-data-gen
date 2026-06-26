#!/bin/bash -l
#
# One-time (or refresh): build DCLM CORE 13-gram benchmark index for decontamination.
# Submit:  sbatch slurm/00_prepare_benchmarks.sbatch
#
#SBATCH --job-name=dm-synth-bench
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -p alpha
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
set -euo pipefail

WS_ROOT="/data/horse/ws/hama901h-whittle"
ROOT="${WS_ROOT}/inference-hive/examples/datamix_1b_synthetic"
cd "${ROOT}"
mkdir -p logs

source "${WS_ROOT}/cache.sh"
export TOKENIZERS_PARALLELISM=false

echo "=== DCLM CORE benchmark index: $(date) ==="
./run_pipeline.sh prepare-benchmarks
echo "=== done: $(date) ==="
echo "Index: ${ROOT}/benchmarks/dclm_core_13gram_index.pkl"
