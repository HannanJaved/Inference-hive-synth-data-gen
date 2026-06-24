#!/usr/bin/env bash
# Orchestrate the Datamix 1B synthetic data pipeline.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IH_ROOT="$(cd "$ROOT/../.." && pwd)"
export DATAMIX_SYNTH_ROOT="$ROOT"

PYTHON="${IH_ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Activate or create inference-hive venv at $IH_ROOT/.venv"
  exit 1
fi

step="${1:-help}"

case "$step" in
  plan)
    "$PYTHON" "$ROOT/prepare_dataset.py" --mix-config "$ROOT/mix_config.yaml" --root "$ROOT" --plan-only
    ;;
  prepare)
    "$PYTHON" "$ROOT/prepare_dataset.py" --mix-config "$ROOT/mix_config.yaml" --root "$ROOT"
    ;;
  validate)
    cd "$IH_ROOT"
    "$PYTHON" validate_config.py --config "$ROOT/config_datamix_1b.yaml"
    "$PYTHON" validate_data.py --config "$ROOT/config_datamix_1b.yaml"
    ;;
  create-run)
    cd "$IH_ROOT"
    "$PYTHON" create_run.py --config "$ROOT/config_datamix_1b.yaml" --output "$ROOT/runs/datamix_1b"
    ;;
  submit)
    cd "$IH_ROOT"
    LIMIT="${2:-}"
    if [[ -n "$LIMIT" ]]; then
      "$PYTHON" submit.py --run-dir "$ROOT/runs/datamix_1b" --limit "$LIMIT"
    else
      "$PYTHON" submit.py --run-dir "$ROOT/runs/datamix_1b"
    fi
    ;;
  status)
    cd "$IH_ROOT"
    "$PYTHON" status.py --run-dir "$ROOT/runs/datamix_1b" --detailed
    ;;
  postprocess)
    "$PYTHON" "$ROOT/postprocess_outputs.py" --mix-config "$ROOT/mix_config.yaml" --root "$ROOT" "${@:2}"
    ;;
  tokenize)
    "$PYTHON" "$ROOT/postprocess_outputs.py" --mix-config "$ROOT/mix_config.yaml" --root "$ROOT" --tokenize
    ;;
  help|*)
    cat <<'EOF'
Datamix 1B synthetic pipeline (inference-hive)

  ./run_pipeline.sh plan          # show token budget / prompt counts
  ./run_pipeline.sh prepare       # build prompts + hf-disk dataset
  ./run_pipeline.sh validate      # validate inference-hive config + data
  ./run_pipeline.sh create-run    # generate SLURM run directory
  ./run_pipeline.sh submit [N]    # submit jobs (optional --limit N)
  ./run_pipeline.sh status        # monitor run progress
  ./run_pipeline.sh postprocess   # join outputs, QC, export corpus
  ./run_pipeline.sh tokenize      # postprocess + Datamix tokenizer token counts

Before running:
  1. Edit mix_config.yaml (optional: fractions, languages)
  2. Edit config_datamix_1b.yaml (SLURM partition/account)
  3. Ensure Datamix model exists in HF cache (see config_datamix_pilot.yaml)
EOF
    ;;
esac
