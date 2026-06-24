#!/usr/bin/env bash
# Concatenate main + top-up corpora after both postprocess steps complete.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN="${ROOT}/corpus/synthetic_corpus.jsonl"
TOPUP="${ROOT}/topup/corpus/synthetic_corpus.jsonl"
OUT="${ROOT}/corpus/synthetic_corpus_merged.jsonl"

for f in "$MAIN" "$TOPUP"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing $f"
    exit 1
  fi
done

cat "$MAIN" "$TOPUP" > "$OUT"

MAIN_N=$(wc -l < "$MAIN")
TOPUP_N=$(wc -l < "$TOPUP")
MERGED_N=$(wc -l < "$OUT")

echo "Merged corpus written to: $OUT"
echo "  main:   ${MAIN_N} rows"
echo "  topup:  ${TOPUP_N} rows"
echo "  merged: ${MERGED_N} rows"
echo ""
echo "Tokenize merged: ./run_pipeline_topup.sh tokenize-merged"
