#!/usr/bin/env bash
# 3B regeneration pipeline (prompt dedup, 16-GPU inference, full postprocess QC).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IH_ROOT="$(cd "$ROOT/../.." && pwd)"
SYNTH_PKG="${ROOT}/../datamix_1b_synthetic"
export DATAMIX_SYNTH_ROOT="$ROOT"
export PYTHONPATH="${SYNTH_PKG}:${PYTHONPATH:-}"

PYTHON="${IH_ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Activate or create inference-hive venv at $IH_ROOT/.venv"
  exit 1
fi

step="${1:-help}"

case "$step" in
  plan)
    "$PYTHON" "${SYNTH_PKG}/prepare_dataset.py" --mix-config "$ROOT/mix_config.yaml" --root "$ROOT" --plan-only
    ;;
  prepare-benchmarks)
    "$PYTHON" "${SYNTH_PKG}/prepare_dclm_benchmark_ngrams.py" --root "${SYNTH_PKG}"
    ;;
  prepare)
    "$PYTHON" "${SYNTH_PKG}/prepare_dataset.py" --mix-config "$ROOT/mix_config.yaml" --root "$ROOT"
    ;;
  validate)
    cd "$IH_ROOT"
    "$PYTHON" validate_config.py --config "$ROOT/config_datamix_3b.yaml"
    "$PYTHON" validate_data.py --config "$ROOT/config_datamix_3b.yaml"
    ;;
  create-run)
    cd "$IH_ROOT"
    FORCE="${2:-}"
    if [[ "$FORCE" == "--force" ]]; then
      "$PYTHON" create_run.py --config "$ROOT/config_datamix_3b.yaml" --output "$ROOT/runs/datamix_3b_regen" --force
    else
      "$PYTHON" create_run.py --config "$ROOT/config_datamix_3b.yaml" --output "$ROOT/runs/datamix_3b_regen"
    fi
    ;;
  submit)
    cd "$IH_ROOT"
    LIMIT="${2:-}"
    if [[ -n "$LIMIT" ]]; then
      "$PYTHON" submit.py --run-dir "$ROOT/runs/datamix_3b_regen" --limit "$LIMIT"
    else
      "$PYTHON" submit.py --run-dir "$ROOT/runs/datamix_3b_regen"
    fi
    ;;
  status)
    cd "$IH_ROOT"
    "$PYTHON" status.py --run-dir "$ROOT/runs/datamix_3b_regen" --detailed
    ;;
  postprocess)
    "$PYTHON" "${SYNTH_PKG}/postprocess_outputs.py" --mix-config "$ROOT/mix_config.yaml" --root "$ROOT" "${@:2}"
    ;;
  tokenize)
    "$PYTHON" "${SYNTH_PKG}/postprocess_outputs.py" --mix-config "$ROOT/mix_config.yaml" --root "$ROOT" --tokenize
    ;;
  help|*)
    cat <<'EOF'
Datamix 3B regeneration (dedup + decontam, 16 GPUs)

  ./run_pipeline.sh plan
  ./run_pipeline.sh prepare-benchmarks   # once, or use index from datamix_1b_synthetic
  ./run_pipeline.sh prepare              # deduped prompts (~3.6M rows planned)
  ./run_pipeline.sh validate
  ./run_pipeline.sh create-run
  ./submit_inference.sh                  # 16 GPU shards
  ./run_pipeline.sh status
  ./run_pipeline.sh postprocess
  ./run_pipeline.sh tokenize

SLURM: see slurm/README in this directory.
EOF
    ;;
esac
