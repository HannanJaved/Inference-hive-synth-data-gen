#!/bin/bash -l
#SBATCH --job-name=dm-3b-regen-prepare
#SBATCH --time=72:00:00
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

echo "=== 3B regen prepare: $(date) ==="
./run_pipeline.sh plan
./run_pipeline.sh prepare
echo "=== done: $(date) ==="
echo "Check prompts/prompt_dedup_stats.json before GPU submit."
