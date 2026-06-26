#!/usr/bin/env python3
"""Build cached n-gram index for DCLM CORE benchmark decontamination."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from datamix_synth.dclm_core_benchmarks import (
    DCLM_CORE_BENCHMARK_NAMES,
    collect_benchmark_texts,
)
from datamix_synth.decontam import BenchmarkNgramIndex, build_ngram_index


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Build DCLM CORE benchmark n-gram index")
    ap.add_argument("--root", default=None, help="Pipeline root (default: $DATAMIX_SYNTH_ROOT or cwd)")
    ap.add_argument("--ngram-size", type=int, default=13)
    ap.add_argument(
        "--output",
        default=None,
        help="Output pickle path (default: <root>/benchmarks/dclm_core_13gram_index.pkl)",
    )
    ap.add_argument(
        "--benchmark",
        action="append",
        dest="benchmarks",
        help="Restrict to specific benchmark(s); repeatable",
    )
    args = ap.parse_args()

    root = Path(args.root or os.environ.get("DATAMIX_SYNTH_ROOT", Path.cwd())).resolve()
    out = Path(args.output) if args.output else root / "benchmarks" / f"dclm_core_{args.ngram_size}gram_index.pkl"
    names = args.benchmarks or list(DCLM_CORE_BENCHMARK_NAMES)

    by_benchmark = collect_benchmark_texts(names)
    all_texts: list[str] = []
    counts = {}
    for name, texts in by_benchmark.items():
        counts[name] = len(texts)
        all_texts.extend(texts)

    ngrams = build_ngram_index(all_texts, ngram_size=args.ngram_size)
    metadata = {
        "benchmarks": names,
        "strings_per_benchmark": counts,
        "total_strings": len(all_texts),
        "ngram_count": len(ngrams),
        "ngram_size": args.ngram_size,
        "method": "word_ngram_overlap",
    }
    index = BenchmarkNgramIndex(
        ngram_size=args.ngram_size,
        ngrams=ngrams,
        metadata=metadata,
    )
    index.save(out)

    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    print(f"\nWrote index: {out}")
    print(f"Wrote metadata: {meta_path}")


if __name__ == "__main__":
    main()
