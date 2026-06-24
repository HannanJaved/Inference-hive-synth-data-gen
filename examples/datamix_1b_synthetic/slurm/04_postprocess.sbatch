#!/bin/bash -l
#
# Step 4 (CPU): join ih_outputs, QC, tokenize — after all GPU shards finish.
# Submit manually:
#   sbatch slurm/04_postprocess.sbatch
#
#SBATCH --job-name=dm-synth-1b-post
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH -p alpha
#SBATCH --requeue
#SBATCH --gres=gpu:1
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

echo "=== 1B postprocess: $(date) ==="
./run_pipeline.sh postprocess
./run_pipeline.sh tokenize

echo "=== outputs ==="
echo "  corpus: ${ROOT}/corpus/synthetic_corpus.jsonl"
echo "  stats:  ${ROOT}/corpus/join_stats.json"
echo "  tokens: ${ROOT}/corpus/tokenized/tokens.jsonl"
echo "=== done: $(date) ==="
