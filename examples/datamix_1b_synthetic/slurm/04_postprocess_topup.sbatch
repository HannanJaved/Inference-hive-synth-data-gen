#!/bin/bash -l
#
# Step 4 (CPU): postprocess top-up ih_outputs, merge with main corpus, tokenize.
# Submit after all top-up GPU shards finish:
#   sbatch slurm/04_postprocess_topup.sbatch
#
#SBATCH --job-name=dm-synth-topup-post
#SBATCH --time=08:00:00
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

echo "=== top-up postprocess: $(date) ==="
./run_pipeline_topup.sh postprocess
./run_pipeline_topup.sh tokenize
./run_pipeline_topup.sh merge
./run_pipeline_topup.sh tokenize-merged

echo "=== outputs ==="
echo "  topup corpus: ${ROOT}/topup/corpus/synthetic_corpus.jsonl"
echo "  merged:       ${ROOT}/corpus/synthetic_corpus_merged.jsonl"
echo "  merged stats: ${ROOT}/corpus/tokenized_merged/tokenize_stats.json"
echo "=== done: $(date) ==="
