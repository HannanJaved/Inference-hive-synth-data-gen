#!/bin/bash -l
#
# Step 4 (CPU, optional): postprocess after inference-hive run completes.
# Submit manually once ih_outputs/ has parquet shards:
#   sbatch pilot/slurm/04_postprocess_pilot.sbatch
#
# SLURM CONFIG (aligned with whittle-paper/launch_search.sh) -----------------
#SBATCH --job-name=dm-synth-pilot-post
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -p alpha
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# -----------------------------------------------------------------------------
#
set -euo pipefail

WS_ROOT="/data/horse/ws/hama901h-whittle"
ROOT="${WS_ROOT}/inference-hive/examples/datamix_1b_synthetic/pilot"
cd "${ROOT}"
mkdir -p logs

source "${WS_ROOT}/cache.sh"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== pilot postprocess: $(date) ==="
./run_pilot_pipeline.sh postprocess
./run_pilot_pipeline.sh tokenize

echo "=== outputs ==="
echo "  corpus: ${ROOT}/corpus/synthetic_corpus.jsonl"
echo "  stats:  ${ROOT}/corpus/join_stats.json"
echo "  tokens: ${ROOT}/corpus/tokenized/tokenize_stats.json"
echo "=== done: $(date) ==="
