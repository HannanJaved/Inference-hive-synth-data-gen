#!/usr/bin/env python3
"""Prepare Datamix-aligned synthetic prompt dataset for inference-hive."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datamix_synth.budget import write_plan
from datamix_synth.build import build_all
from datamix_synth.config import MixConfig, load_mix_config


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Datamix synthetic prompts for inference-hive")
    ap.add_argument("--mix-config", default="mix_config.yaml",
                    help="YAML with token budget and domain fractions")
    ap.add_argument("--root", default=None,
                    help="Pipeline root (default: $DATAMIX_SYNTH_ROOT or cwd)")
    ap.add_argument("--plan-only", action="store_true",
                    help="Only write budget plan, do not build prompts")
    ap.add_argument("--dry-run", action="store_true",
                    help="Alias for --plan-only")
    args = ap.parse_args()

    root = Path(args.root or os.environ.get("DATAMIX_SYNTH_ROOT", Path.cwd())).resolve()
    cfg = load_mix_config(Path(args.mix_config))
    cfg.root = root
    cfg.prompts_dir = root / "prompts"
    cfg.dataset_dir = root / "dataset"
    cfg.outputs_dir = root / "ih_outputs"
    cfg.corpus_dir = root / "corpus"

    plan_path = cfg.root / "budget_plan.json"
    summary = write_plan(cfg, plan_path)
    print(json.dumps(summary, indent=2))
    print(f"\nWrote budget plan to {plan_path}")
    print(f"Total prompts planned: {summary['total_prompts']:,}")
    print(f"Estimated completion tokens: {summary['estimated_completion_tokens']:,}")

    if args.plan_only or args.dry_run:
        return

    prompts_path, dataset_path, dedup_stats = build_all(cfg)
    print(f"\nWrote prompts: {prompts_path}")
    print(f"Wrote inference-hive dataset: {dataset_path}")
    if dedup_stats.get("dedup_prompts"):
        print(f"Prompt dedup stats: {json.dumps(dedup_stats, indent=2)}")
    print("\nNext steps:")
    print("  1. Edit config_datamix_1b.yaml (SLURM paths, model)")
    print("  2. python ../../validate_config.py --config config_datamix_1b.yaml")
    print("  3. python ../../validate_data.py --config config_datamix_1b.yaml")
    print("  4. python ../../create_run.py --config config_datamix_1b.yaml --output runs/datamix_1b")
    print("  5. python ../../submit.py --run-dir runs/datamix_1b")


if __name__ == "__main__":
    main()
