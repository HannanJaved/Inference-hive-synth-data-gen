"""Tokenize synthetic corpus with the Datamix model tokenizer (Gemma-3 family, 262k vocab)."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from datamix_synth.config import MixConfig


def tokenize_corpus(
    cfg: MixConfig,
    *,
    corpus_jsonl: Path | None = None,
    output_subdir: str = "tokenized",
    batch_size: int = 256,
) -> dict:
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise SystemExit("pip install transformers") from e

    corpus_jsonl = corpus_jsonl or (cfg.corpus_dir / "synthetic_corpus.jsonl")
    tok_dir = cfg.corpus_dir / output_subdir
    tok_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name_or_path)

    total_tokens = 0
    by_domain: Counter = Counter()
    n_docs = 0

    out_path = tok_dir / "tokens.jsonl"
    with corpus_jsonl.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        batch_texts: list[str] = []
        batch_meta: list[dict] = []

        def flush() -> None:
            nonlocal total_tokens, n_docs
            if not batch_texts:
                return
            enc = tokenizer(batch_texts, add_special_tokens=True, truncation=False)
            for meta, ids in zip(batch_meta, enc["input_ids"]):
                n_tok = len(ids)
                total_tokens += n_tok
                by_domain[meta["domain"]] += n_tok
                n_docs += 1
                fout.write(json.dumps({
                    "id": meta["id"],
                    "domain": meta["domain"],
                    "lang": meta["lang"],
                    "num_tokens": n_tok,
                }) + "\n")
            batch_texts.clear()
            batch_meta.clear()

        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            batch_texts.append(rec["text"])
            batch_meta.append(rec)
            if len(batch_texts) >= batch_size:
                flush()
        flush()

    summary = {
        "tokenizer": cfg.tokenizer_name_or_path,
        "documents": n_docs,
        "total_tokens": total_tokens,
        "by_domain_tokens": dict(by_domain),
        "token_counts_jsonl": str(out_path),
    }
    (tok_dir / "tokenize_stats.json").write_text(json.dumps(summary, indent=2))
    return summary
