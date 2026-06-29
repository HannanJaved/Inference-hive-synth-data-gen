"""Join inference-hive outputs, QC, and export pretraining corpus."""
from __future__ import annotations

import json
import multiprocessing as mp
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from datamix_synth.config import MixConfig
from datamix_synth.decontam import BenchmarkNgramIndex, load_benchmark_index
from datamix_synth.dedup import DedupFilter

# Prompt echo almost always appears at the start of the completion.
_ECHO_HEAD_CHARS = 4096
_PROGRESS_EVERY = 25_000

# Set in parent before fork; workers inherit via copy-on-write (Linux).
_G_PROMPT_IDX: dict[str, dict] | None = None
_G_BENCH: BenchmarkNgramIndex | None = None

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
    head = text[:_ECHO_HEAD_CHARS]
    hits = sum(1 for ln in prompt_lines if ln in head)
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


def _load_all_responses(output_path: Path) -> list[dict[str, Any]]:
    return list(_iter_ih_responses(output_path))


@dataclass(frozen=True)
class _PostprocessParams:
    min_completion_tokens: int
    dedup_outputs: bool
    dedup_output_near_threshold: float
    dedup_shingle_size: int
    decontam_enabled: bool


@dataclass
class _ShardResult:
    kept: list[dict[str, Any]]
    rejected: list[dict[str, Any]]
    reject_reasons: Counter
    dedup_reject_reasons: Counter
    by_domain: Counter
    by_lang: Counter
    decontam_rejected: int
    token_est: int
    processed: int


def _process_shard(
    shard_rows: list[dict[str, Any]],
    prompt_idx: dict[str, dict],
    benchmark_index: BenchmarkNgramIndex | None,
    params: _PostprocessParams,
    shard_id: int,
) -> _ShardResult:
    dedup_filters: dict[str, DedupFilter] = {}
    if params.dedup_outputs:
        for domain in ("english", "code", "math", "multilingual"):
            dedup_filters[domain] = DedupFilter(
                near_threshold=params.dedup_output_near_threshold,
                shingle_size=params.dedup_shingle_size,
            )

    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reject_reasons: Counter = Counter()
    dedup_reject_reasons: Counter = Counter()
    by_domain: Counter = Counter()
    by_lang: Counter = Counter()
    decontam_rejected = 0
    token_est = 0

    for n_done, row in enumerate(shard_rows, start=1):
        row_id = row["id"]
        meta = prompt_idx.get(row_id)
        if meta is None:
            reject_reasons["missing_prompt"] += 1
            rejected.append({"id": row_id, "reason": "missing_prompt"})
            continue

        text = _extract_text(row["response"])
        if not text:
            reject_reasons["empty_response"] += 1
            rejected.append({"id": row_id, "reason": "empty_response"})
            continue

        usage = (row["response"] or {}).get("usage") or {}
        ct = int(usage.get("completion_tokens") or 0)
        if ct and ct < params.min_completion_tokens:
            reject_reasons["short_completion"] += 1
            rejected.append({"id": row_id, "reason": "short_completion", "tokens": ct})
            continue

        ok, reason = _qc_text(text, meta["domain"], meta.get("meta_prompt", ""))
        if not ok:
            reject_reasons[reason or "qc_fail"] += 1
            rejected.append({"id": row_id, "reason": reason, "domain": meta["domain"]})
            continue

        if params.dedup_outputs:
            dedup = dedup_filters[meta["domain"]]
            accepted, dedup_reason = dedup.try_add(text)
            if not accepted:
                reject_reasons[dedup_reason or "duplicate"] += 1
                dedup_reject_reasons[dedup_reason or "duplicate"] += 1
                rejected.append({
                    "id": row_id,
                    "reason": dedup_reason,
                    "domain": meta["domain"],
                })
                continue

        if benchmark_index is not None and benchmark_index.is_contaminated(text):
            reject_reasons["benchmark_contamination"] += 1
            decontam_rejected += 1
            rejected.append({
                "id": row_id,
                "reason": "benchmark_contamination",
                "domain": meta["domain"],
            })
            continue

        rec = {
            "id": row_id,
            "text": text,
            "domain": meta["domain"],
            "lang": meta["lang"],
            "completion_tokens": ct,
        }
        kept.append(rec)
        by_domain[meta["domain"]] += 1
        by_lang[meta["lang"]] += 1
        token_est += ct or max(1, len(text.split()) * 4 // 3)

        if n_done % _PROGRESS_EVERY == 0:
            print(f"  shard {shard_id}: processed {n_done:,} rows", flush=True)

    return _ShardResult(
        kept=kept,
        rejected=rejected,
        reject_reasons=reject_reasons,
        dedup_reject_reasons=dedup_reject_reasons,
        by_domain=by_domain,
        by_lang=by_lang,
        decontam_rejected=decontam_rejected,
        token_est=token_est,
        processed=len(shard_rows),
    )


def _shard_rows(
    rows: list[dict[str, Any]],
    n_workers: int,
) -> list[list[dict[str, Any]]]:
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(n_workers)]
    for row in rows:
        buckets[hash(row["id"]) % n_workers].append(row)
    return buckets


def _merge_shard_results(results: list[_ShardResult]) -> _ShardResult:
    merged = _ShardResult(
        kept=[],
        rejected=[],
        reject_reasons=Counter(),
        dedup_reject_reasons=Counter(),
        by_domain=Counter(),
        by_lang=Counter(),
        decontam_rejected=0,
        token_est=0,
        processed=0,
    )
    for res in results:
        merged.kept.extend(res.kept)
        merged.rejected.extend(res.rejected)
        merged.reject_reasons.update(res.reject_reasons)
        merged.dedup_reject_reasons.update(res.dedup_reject_reasons)
        merged.by_domain.update(res.by_domain)
        merged.by_lang.update(res.by_lang)
        merged.decontam_rejected += res.decontam_rejected
        merged.token_est += res.token_est
        merged.processed += res.processed
    return merged


def _write_domain_parquet(cfg: MixConfig, domain_bufs: dict[str, list[dict]]) -> None:
    for domain, rows in domain_bufs.items():
        if not rows:
            continue
        p = cfg.corpus_dir / f"domain={domain}" / "data.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows, schema=_SCHEMA), p, compression="zstd")


def _process_shard_forked(
    shard_rows: list[dict[str, Any]],
    shard_id: int,
    params: _PostprocessParams,
) -> _ShardResult:
    assert _G_PROMPT_IDX is not None
    return _process_shard(shard_rows, _G_PROMPT_IDX, _G_BENCH, params, shard_id)


def _run_parallel(
    rows: list[dict[str, Any]],
    prompt_idx: dict[str, dict],
    benchmark_index: BenchmarkNgramIndex | None,
    params: _PostprocessParams,
    n_workers: int,
) -> _ShardResult:
    global _G_PROMPT_IDX, _G_BENCH
    _G_PROMPT_IDX = prompt_idx
    _G_BENCH = benchmark_index

    shards = _shard_rows(rows, n_workers)
    tasks = [(shard, shard_id) for shard_id, shard in enumerate(shards) if shard]
    print(f"postprocess: {len(rows):,} rows across {len(tasks)} shards ({n_workers} workers)", flush=True)

    if not tasks:
        return _merge_shard_results([])

    if len(tasks) == 1:
        shard, shard_id = tasks[0]
        return _process_shard(shard, prompt_idx, benchmark_index, params, shard_id)

    ctx = mp.get_context("fork")
    with ctx.Pool(processes=min(n_workers, len(tasks))) as pool:
        results = pool.starmap(
            _process_shard_forked,
            [(shard, shard_id, params) for shard, shard_id in tasks],
        )
    return _merge_shard_results(results)


def join_and_export(
    cfg: MixConfig,
    *,
    prompts_path: Path | None = None,
    ih_output_path: Path | None = None,
    min_completion_tokens: int = 150,
    workers: int | None = None,
) -> dict:
    prompts_path = prompts_path or (cfg.prompts_dir / "prompts.jsonl")
    ih_output_path = ih_output_path or cfg.outputs_dir
    n_workers = workers if workers is not None else cfg.postprocess_workers
    n_workers = max(1, n_workers)

    print(f"Loading prompt index from {prompts_path}...", flush=True)
    prompt_idx = _load_prompt_index(prompts_path)
    print(f"  {len(prompt_idx):,} prompts", flush=True)

    benchmark_index: BenchmarkNgramIndex | None = None
    if cfg.decontam_enabled:
        print(f"Loading benchmark index from {cfg.benchmark_index_path}...", flush=True)
        benchmark_index = load_benchmark_index(cfg)

    print(f"Loading inference outputs from {ih_output_path}...", flush=True)
    all_rows = _load_all_responses(ih_output_path)
    print(f"  {len(all_rows):,} responses", flush=True)

    params = _PostprocessParams(
        min_completion_tokens=min_completion_tokens,
        dedup_outputs=cfg.dedup_outputs,
        dedup_output_near_threshold=cfg.dedup_output_near_threshold,
        dedup_shingle_size=cfg.dedup_shingle_size,
        decontam_enabled=cfg.decontam_enabled,
    )

    merged = _run_parallel(all_rows, prompt_idx, benchmark_index, params, n_workers)

    cfg.corpus_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = cfg.corpus_dir / "synthetic_corpus.jsonl"
    reject_jsonl = cfg.corpus_dir / "rejected.jsonl"
    stats_path = cfg.corpus_dir / "join_stats.json"

    domain_bufs: dict[str, list[dict]] = defaultdict(list)
    with out_jsonl.open("w", encoding="utf-8") as fout, \
         reject_jsonl.open("w", encoding="utf-8") as frej:
        for rec in merged.kept:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            domain_bufs[rec["domain"]].append(rec)
        for rec in merged.rejected:
            frej.write(json.dumps(rec, ensure_ascii=False) + "\n")

    _write_domain_parquet(cfg, domain_bufs)

    stats = {
        "kept": len(merged.kept),
        "rejected": len(merged.rejected),
        "estimated_completion_tokens": merged.token_est,
        "by_domain": dict(merged.by_domain),
        "by_lang": dict(merged.by_lang),
        "reject_reasons": dict(merged.reject_reasons),
        "dedup_outputs": cfg.dedup_outputs,
        "dedup_reject_reasons": dict(merged.dedup_reject_reasons),
        "decontam_enabled": cfg.decontam_enabled,
        "decontam_rejected": merged.decontam_rejected,
        "decontam_index": str(cfg.benchmark_index_path) if cfg.decontam_enabled else None,
        "postprocess_workers": n_workers,
        "corpus_jsonl": str(out_jsonl),
    }
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"postprocess done: kept={stats['kept']:,} rejected={stats['rejected']:,}", flush=True)
    return stats
