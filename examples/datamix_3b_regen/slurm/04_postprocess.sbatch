#!/bin/bash -l
#SBATCH --job-name=dm-3b-regen-post
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
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

echo "=== 3B regen postprocess: $(date) ==="
./run_pipeline.sh postprocess
./run_pipeline.sh tokenize
echo "=== corpus: ${ROOT}/corpus/synthetic_corpus.jsonl ==="
echo "=== stats:  ${ROOT}/corpus/join_stats.json ==="
