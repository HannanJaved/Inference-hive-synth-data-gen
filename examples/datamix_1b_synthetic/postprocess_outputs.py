#!/usr/bin/env python3
"""Post-process inference-hive outputs into a Datamix-style synthetic corpus."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datamix_synth.config import load_mix_config
from datamix_synth.postprocess import join_and_export
from datamix_synth.tokenize import tokenize_corpus


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix-config", default="mix_config.yaml")
    ap.add_argument("--root", default=None)
    ap.add_argument("--ih-output", default=None, help="inference-hive output_path directory")
    ap.add_argument("--tokenize", action="store_true",
                    help="Also count tokens with the Datamix model tokenizer")
    args = ap.parse_args()

    root = Path(args.root or os.environ.get("DATAMIX_SYNTH_ROOT", Path.cwd())).resolve()
    cfg = load_mix_config(Path(args.mix_config))
    cfg.root = root
    cfg.prompts_dir = root / "prompts"
    cfg.dataset_dir = root / "dataset"
    cfg.outputs_dir = Path(args.ih_output) if args.ih_output else root / "ih_outputs"
    cfg.corpus_dir = root / "corpus"

    stats = join_and_export(cfg)
    print(json.dumps(stats, indent=2))

    if args.tokenize:
        tok_stats = tokenize_corpus(cfg)
        print(json.dumps(tok_stats, indent=2))


if __name__ == "__main__":
    main()
