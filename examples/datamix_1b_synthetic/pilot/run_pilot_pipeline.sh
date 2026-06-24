#!/usr/bin/env bash
# Pilot pipeline wrapper (same steps as run_pipeline.sh, pilot configs/paths).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(cd "$ROOT/.." && pwd)"
IH_ROOT="$(cd "$PIPELINE_ROOT/../.." && pwd)"
export DATAMIX_SYNTH_ROOT="$ROOT"

PYTHON="${IH_ROOT}/.venv/bin/python"
MIX_CONFIG="${ROOT}/mix_config_pilot.yaml"
IH_CONFIG="${ROOT}/config_datamix_pilot.yaml"
RUN_DIR="${ROOT}/runs/datamix_pilot"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing inference-hive venv: ${IH_ROOT}/.venv"
  exit 1
fi

step="${1:-help}"

case "$step" in
  plan)
    "$PYTHON" "$PIPELINE_ROOT/prepare_dataset.py" --mix-config "$MIX_CONFIG" --root "$ROOT" --plan-only
    ;;
  prepare)
    "$PYTHON" "$PIPELINE_ROOT/prepare_dataset.py" --mix-config "$MIX_CONFIG" --root "$ROOT"
    ;;
  validate)
    cd "$IH_ROOT"
    "$PYTHON" validate_config.py --config "$IH_CONFIG"
    "$PYTHON" validate_data.py --config "$IH_CONFIG"
    ;;
  create-run)
    cd "$IH_ROOT"
    "$PYTHON" create_run.py --config "$IH_CONFIG" --output "$RUN_DIR"
    ;;
  submit)
    cd "$IH_ROOT"
    LIMIT="${2:-1}"
    "$PYTHON" submit.py --run-dir "$RUN_DIR" --limit "$LIMIT"
    ;;
  status)
    cd "$IH_ROOT"
    "$PYTHON" status.py --run-dir "$RUN_DIR" --detailed
    ;;
  postprocess)
    "$PYTHON" "$PIPELINE_ROOT/postprocess_outputs.py" --mix-config "$MIX_CONFIG" --root "$ROOT" "${@:2}"
    ;;
  tokenize)
    "$PYTHON" "$PIPELINE_ROOT/postprocess_outputs.py" --mix-config "$MIX_CONFIG" --root "$ROOT" --tokenize
    ;;
  help|*)
    cat <<EOF
Datamix PILOT pipeline (~500k tokens, 1 GPU node)

  ./run_pilot_pipeline.sh plan
  ./run_pilot_pipeline.sh prepare
  ./run_pilot_pipeline.sh validate
  ./run_pilot_pipeline.sh create-run
  ./run_pilot_pipeline.sh submit        # default: 1 shard
  ./run_pilot_pipeline.sh status
  ./run_pilot_pipeline.sh postprocess
  ./run_pilot_pipeline.sh tokenize

SLURM wrappers in pilot/slurm/ — see pilot/README_PILOT.md
EOF
    ;;
esac
