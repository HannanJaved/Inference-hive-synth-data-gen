#!/usr/bin/env bash
# Orchestrate the 300M-token top-up generation (separate from main 1B run).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOPUP_ROOT="${ROOT}/topup"
IH_ROOT="$(cd "$ROOT/../.." && pwd)"
export DATAMIX_SYNTH_ROOT="$TOPUP_ROOT"

PYTHON="${IH_ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Activate or create inference-hive venv at $IH_ROOT/.venv"
  exit 1
fi

MIX_CONFIG="${ROOT}/mix_config_topup.yaml"
IH_CONFIG="${ROOT}/config_datamix_topup.yaml"
RUN_DIR="${ROOT}/runs/datamix_1b_topup"

step="${1:-help}"

case "$step" in
  plan)
    "$PYTHON" "$ROOT/prepare_dataset.py" --mix-config "$MIX_CONFIG" --root "$TOPUP_ROOT" --plan-only
    ;;
  prepare)
    "$PYTHON" "$ROOT/prepare_dataset.py" --mix-config "$MIX_CONFIG" --root "$TOPUP_ROOT"
    ;;
  validate)
    cd "$IH_ROOT"
    "$PYTHON" validate_config.py --config "$IH_CONFIG"
    "$PYTHON" validate_data.py --config "$IH_CONFIG"
    ;;
  create-run)
    cd "$IH_ROOT"
    FORCE="${2:-}"
    if [[ "$FORCE" == "--force" ]]; then
      "$PYTHON" create_run.py --config "$IH_CONFIG" --output "$RUN_DIR" --force
    else
      "$PYTHON" create_run.py --config "$IH_CONFIG" --output "$RUN_DIR"
    fi
    ;;
  submit)
    cd "$IH_ROOT"
    LIMIT="${2:-}"
    if [[ -n "$LIMIT" ]]; then
      "$PYTHON" submit.py --run-dir "$RUN_DIR" --limit "$LIMIT"
    else
      "$PYTHON" submit.py --run-dir "$RUN_DIR"
    fi
    ;;
  status)
    cd "$IH_ROOT"
    "$PYTHON" status.py --run-dir "$RUN_DIR" --detailed
    ;;
  postprocess)
    "$PYTHON" "$ROOT/postprocess_outputs.py" \
      --mix-config "$MIX_CONFIG" \
      --root "$TOPUP_ROOT" \
      --ih-output "$TOPUP_ROOT/ih_outputs" \
      "${@:2}"
    ;;
  tokenize)
    "$PYTHON" "$ROOT/postprocess_outputs.py" \
      --mix-config "$MIX_CONFIG" \
      --root "$TOPUP_ROOT" \
      --ih-output "$TOPUP_ROOT/ih_outputs" \
      --tokenize
    ;;
  merge)
    "$ROOT/merge_corpus.sh"
    ;;
  tokenize-merged)
    MERGED="${ROOT}/corpus/synthetic_corpus_merged.jsonl"
    if [[ ! -f "$MERGED" ]]; then
      echo "Missing $MERGED — run ./run_pipeline_topup.sh merge first"
      exit 1
    fi
    "$PYTHON" -c "
from pathlib import Path
from datamix_synth.config import load_mix_config
from datamix_synth.tokenize import tokenize_corpus
import json
cfg = load_mix_config('${MIX_CONFIG}')
cfg.root = Path('${ROOT}')
cfg.corpus_dir = Path('${ROOT}/corpus')
print(json.dumps(tokenize_corpus(
    cfg,
    corpus_jsonl=Path('${MERGED}'),
    output_subdir='tokenized_merged',
), indent=2))
"
    ;;
  help|*)
    cat <<'EOF'
Datamix 300M top-up pipeline (16 GPUs, separate ih_outputs)

  ./run_pipeline_topup.sh plan          # show ~300M token budget / prompt counts
  ./run_pipeline_topup.sh prepare       # build topup/prompts + topup/dataset
  ./run_pipeline_topup.sh validate      # validate inference-hive config + data
  ./run_pipeline_topup.sh create-run    # generate runs/datamix_1b_topup/
  ./run_pipeline_topup.sh submit        # submit 16 GPU shards (or submit N)
  ./run_pipeline_topup.sh status        # monitor top-up run
  ./run_pipeline_topup.sh postprocess   # QC + export topup/corpus/
  ./run_pipeline_topup.sh tokenize      # token counts for top-up corpus
  ./run_pipeline_topup.sh merge         # cat main + topup → corpus/synthetic_corpus_merged.jsonl
  ./run_pipeline_topup.sh tokenize-merged

SLURM (login node):
  sbatch slurm/01_prepare_topup.sbatch
  sbatch slurm/02_validate_create_run_topup.sbatch
  ./submit_inference_topup.sh
  sbatch slurm/04_postprocess_topup.sbatch   # after inference completes
EOF
    ;;
esac
