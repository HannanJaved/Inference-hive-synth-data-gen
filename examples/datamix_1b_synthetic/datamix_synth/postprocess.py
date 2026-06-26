"""Join inference-hive outputs, QC, and export pretraining corpus."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from datamix_synth.config import MixConfig
from datamix_synth.decontam import BenchmarkNgramIndex, load_benchmark_index
from datamix_synth.dedup import DedupFilter

_LEAK_PHRASES = re.compile(
    r"(as an ai|i cannot|here is the code|sure!|certainly!|let me|chatgpt|language model)",
    re.IGNORECASE,
)
_CODE_FENCE = re.compile(r"^```", re.MULTILINE)
# Imperative / rubric leakage from failed instruction-style runs or web junk.
_RUBRIC_PHRASES = re.compile(
    r"(write original|requirements:|target length:|output only|similar to nemotron|"
    r"similar in spirit|format exactly|primary language:|pretraining|annotation guidelines|"
    r"do not include|maximum output length|payment details|freelance|upwork|"
    r"please contact me|verifiable constraint|training example)",
    re.IGNORECASE,
)
_BULLET_LINE = re.compile(r"^\s*[-*•]\s+\S", re.MULTILINE)

_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("text", pa.string()),
    ("domain", pa.string()),
    ("lang", pa.string()),
    ("completion_tokens", pa.int32()),
])


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _load_prompt_index(prompts_path: Path) -> dict[str, dict]:
    return {r["id"]: r for r in _iter_jsonl(prompts_path)}


def _extract_text(response_obj: dict) -> str | None:
    try:
        choices = response_obj.get("choices") or []
        if not choices:
            return None
        choice = choices[0]
        msg = choice.get("message") or {}
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        text = choice.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
    except (AttributeError, TypeError, KeyError):
        return None
    return None


def _prompt_overlap_ratio(text: str, prompt: str) -> float:
    """Fraction of prompt lines substantially repeated in the output."""
    if not prompt or not text:
        return 0.0
    prompt_lines = [ln.strip() for ln in prompt.splitlines() if ln.strip()]
    if not prompt_lines:
        return 0.0
    hits = sum(1 for ln in prompt_lines if ln in text)
    return hits / len(prompt_lines)


def _looks_like_rubric(text: str) -> bool:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return True
    bullet_lines = len(_BULLET_LINE.findall(text))
    if bullet_lines >= 5 and bullet_lines / len(lines) >= 0.35:
        return True
    return bool(_RUBRIC_PHRASES.search(text))


def _qc_text(text: str, domain: str, meta_prompt: str = "") -> tuple[bool, str | None]:
    if len(text) < 120:
        return False, "too_short"
    if _prompt_overlap_ratio(text, meta_prompt) >= 0.5:
        return False, "prompt_echo"
    if _looks_like_rubric(text):
        return False, "rubric_leak"
    if _LEAK_PHRASES.search(text[:400]):
        return False, "assistant_leak"
    if domain == "code" and _CODE_FENCE.search(text):
        return False, "markdown_fence"
    if domain == "math":
        # Prefix already contains Problem/Solution header; completion extends the solution.
        if not re.search(r"\b(step|\d+\s*[-+*/=]|therefore|answer|cm\^2|\$\d)", text, re.I):
            return False, "math_format"
    return True, None


def _parquet_output_files(output_path: Path) -> list[Path]:
    """List finalized/checkpoint parquet files; exclude atomic .incomplete merge stubs."""
    files = sorted(
        p for p in output_path.glob("**/*.parquet")
        if p.is_file() and p.stat().st_size > 0
    )
    if not files:
        raise FileNotFoundError(f"No parquet files found under {output_path}")
    return files


def _iter_ih_responses(output_path: Path) -> Iterator[dict[str, Any]]:
    # Parts first, then checkpoints (newer files win on duplicate ids).
    files = _parquet_output_files(output_path)
    files.sort(key=lambda p: (0 if "_part" in p.name else 1, p.name))

    by_id: dict[str, dict[str, Any]] = {}
    for path in files:
        table = pq.read_table(path, columns=["id", "response"])
        for row_id, resp in zip(table["id"].to_pylist(), table["response"].to_pylist()):
            by_id[row_id] = {"id": row_id, "response": resp}

    yield from by_id.values()


def _write_domain_parquet(cfg: MixConfig, domain_bufs: dict[str, list[dict]]) -> None:
    for domain, rows in domain_bufs.items():
        if not rows:
            continue
        p = cfg.corpus_dir / f"domain={domain}" / "data.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows, schema=_SCHEMA), p, compression="zstd")


def join_and_export(
    cfg: MixConfig,
    *,
    prompts_path: Path | None = None,
    ih_output_path: Path | None = None,
    min_completion_tokens: int = 150,
) -> dict:
    prompts_path = prompts_path or (cfg.prompts_dir / "prompts.jsonl")
    ih_output_path = ih_output_path or cfg.outputs_dir
    prompt_idx = _load_prompt_index(prompts_path)

    cfg.corpus_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = cfg.corpus_dir / "synthetic_corpus.jsonl"
    reject_jsonl = cfg.corpus_dir / "rejected.jsonl"
    stats_path = cfg.corpus_dir / "join_stats.json"

    kept = 0
    rejected = 0
    token_est = 0
    by_domain: Counter = Counter()
    by_lang: Counter = Counter()
    reject_reasons: Counter = Counter()
    domain_bufs: dict[str, list[dict]] = defaultdict(list)
    dedup_filters: dict[str, DedupFilter] = {}
    dedup_reject_reasons: Counter = Counter()
    decontam_rejected = 0

    benchmark_index: BenchmarkNgramIndex | None = None
    if cfg.decontam_enabled:
        benchmark_index = load_benchmark_index(cfg)

    if cfg.dedup_outputs:
        for domain in cfg.domain_fractions:
            dedup_filters[domain] = DedupFilter(
                near_threshold=cfg.dedup_output_near_threshold,
                shingle_size=cfg.dedup_shingle_size,
            )

    with out_jsonl.open("w", encoding="utf-8") as fout, \
         reject_jsonl.open("w", encoding="utf-8") as frej:
        for row in _iter_ih_responses(ih_output_path):
            row_id = row["id"]
            meta = prompt_idx.get(row_id)
            if meta is None:
                rejected += 1
                reject_reasons["missing_prompt"] += 1
                frej.write(json.dumps({"id": row_id, "reason": "missing_prompt"}) + "\n")
                continue

            text = _extract_text(row["response"])
            if not text:
                rejected += 1
                reject_reasons["empty_response"] += 1
                frej.write(json.dumps({"id": row_id, "reason": "empty_response"}) + "\n")
                continue

            usage = (row["response"] or {}).get("usage") or {}
            ct = int(usage.get("completion_tokens") or 0)
            if ct and ct < min_completion_tokens:
                rejected += 1
                reject_reasons["short_completion"] += 1
                frej.write(json.dumps({"id": row_id, "reason": "short_completion", "tokens": ct}) + "\n")
                continue

            ok, reason = _qc_text(text, meta["domain"], meta.get("meta_prompt", ""))
            if not ok:
                rejected += 1
                reject_reasons[reason or "qc_fail"] += 1
                frej.write(json.dumps({"id": row_id, "reason": reason, "domain": meta["domain"]}) + "\n")
                continue

            if cfg.dedup_outputs:
                dedup = dedup_filters[meta["domain"]]
                accepted, dedup_reason = dedup.try_add(text)
                if not accepted:
                    rejected += 1
                    reject_reasons[dedup_reason or "duplicate"] += 1
                    dedup_reject_reasons[dedup_reason or "duplicate"] += 1
                    frej.write(json.dumps({
                        "id": row_id,
                        "reason": dedup_reason,
                        "domain": meta["domain"],
                    }) + "\n")
                    continue

            if benchmark_index is not None and benchmark_index.is_contaminated(text):
                rejected += 1
                reject_reasons["benchmark_contamination"] += 1
                decontam_rejected += 1
                frej.write(json.dumps({
                    "id": row_id,
                    "reason": "benchmark_contamination",
                    "domain": meta["domain"],
                }) + "\n")
                continue

            rec = {
                "id": row_id,
                "text": text,
                "domain": meta["domain"],
                "lang": meta["lang"],
                "completion_tokens": ct,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            domain_bufs[meta["domain"]].append(rec)
            kept += 1
            by_domain[meta["domain"]] += 1
            by_lang[meta["lang"]] += 1
            token_est += ct or max(1, len(text.split()) * 4 // 3)

    _write_domain_parquet(cfg, domain_bufs)

    stats = {
        "kept": kept,
        "rejected": rejected,
        "estimated_completion_tokens": token_est,
        "by_domain": dict(by_domain),
        "by_lang": dict(by_lang),
        "reject_reasons": dict(reject_reasons),
        "dedup_outputs": cfg.dedup_outputs,
        "dedup_reject_reasons": dict(dedup_reject_reasons),
        "decontam_enabled": cfg.decontam_enabled,
        "decontam_rejected": decontam_rejected,
        "decontam_index": str(cfg.benchmark_index_path) if cfg.decontam_enabled else None,
        "corpus_jsonl": str(out_jsonl),
    }
    stats_path.write_text(json.dumps(stats, indent=2))
    return stats
