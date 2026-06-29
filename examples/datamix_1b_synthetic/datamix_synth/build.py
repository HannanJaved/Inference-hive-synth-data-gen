"""Build prompt rows and export inference-hive-ready datasets."""
from __future__ import annotations

import json
import multiprocessing as mp
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import datasets as hfds

from datamix_synth.budget import plan_budget
from datamix_synth.config import MixConfig
from datamix_synth.decontam import BenchmarkNgramIndex, load_benchmark_index
from datamix_synth.dedup import DedupFilter, normalize_prompt_for_dedup, normalize_text
from datamix_synth.prompts import build_meta_prompt

_PROGRESS_EVERY = 10_000

# Inherited by forked prepare workers (Linux copy-on-write).
_G_BENCH: BenchmarkNgramIndex | None = None


@dataclass(frozen=True)
class _PromptBuildOpts:
    seed: int
    id_prefix: str
    dedup_prompts: bool
    dedup_prompt_fold_templates: bool
    dedup_prompt_near_threshold: float
    dedup_shingle_size: int
    dedup_prompt_max_resamples: int


@dataclass(frozen=True)
class _SlotTask:
    order: int
    domain: str
    slot_i: int
    lang_code: str | None = None
    lang_name: str | None = None


@dataclass
class _PrepareShardResult:
    rows: list[dict[str, Any]]
    resample_exhausted: Counter
    decontam_stats: Counter
    by_scope: dict[str, dict[str, int]]
    written: int


def _opts_from_cfg(cfg: MixConfig) -> _PromptBuildOpts:
    return _PromptBuildOpts(
        seed=cfg.seed,
        id_prefix=cfg.id_prefix,
        dedup_prompts=cfg.dedup_prompts,
        dedup_prompt_fold_templates=cfg.dedup_prompt_fold_templates,
        dedup_prompt_near_threshold=cfg.dedup_prompt_near_threshold,
        dedup_shingle_size=cfg.dedup_shingle_size,
        dedup_prompt_max_resamples=cfg.dedup_prompt_max_resamples,
    )


def _slot_seed(opts: _PromptBuildOpts, domain: str, slot_i: int, lang_code: str | None) -> int:
    return hash((opts.seed, domain, slot_i, lang_code)) & 0x7FFFFFFF


def _dedup_filter_for_scope(opts: _PromptBuildOpts) -> DedupFilter | None:
    if not opts.dedup_prompts:
        return None
    normalize_fn = (
        normalize_prompt_for_dedup
        if opts.dedup_prompt_fold_templates
        else normalize_text
    )
    return DedupFilter(
        near_threshold=opts.dedup_prompt_near_threshold,
        shingle_size=opts.dedup_shingle_size,
        normalize_fn=normalize_fn,
    )


def _dedup_scope(domain: str, meta_fields: dict[str, Any]) -> str:
    subtopic = meta_fields.get("subtopic")
    if domain == "multilingual":
        return f"multilingual:{meta_fields.get('lang', 'unknown')}"
    if subtopic:
        return f"{domain}:{subtopic}"
    return domain


def _sample_unique_prompt(
    rng: random.Random,
    opts: _PromptBuildOpts,
    dedup_filters: dict[str, DedupFilter | None],
    benchmark_index: BenchmarkNgramIndex | None,
    decontam_stats: Counter,
    *,
    domain: str,
    lang_code: str | None = None,
    lang_name: str | None = None,
) -> tuple[str, dict[str, Any]] | None:
    for _ in range(opts.dedup_prompt_max_resamples):
        meta, meta_fields = build_meta_prompt(
            rng, domain, lang_code=lang_code, lang_name=lang_name,
        )
        if benchmark_index is not None and benchmark_index.is_contaminated(meta):
            decontam_stats["benchmark_contamination"] += 1
            continue
        if not opts.dedup_prompts:
            return meta, meta_fields
        scope = _dedup_scope(domain, meta_fields)
        dedup = dedup_filters.get(scope)
        if dedup is None:
            dedup = _dedup_filter_for_scope(opts)
            dedup_filters[scope] = dedup
        accepted, _reason = dedup.try_add(meta)
        if accepted:
            return meta, meta_fields
    return None


def _row_from_sample(
    opts: _PromptBuildOpts,
    task: _SlotTask,
    meta: str,
    meta_fields: dict[str, Any],
) -> dict[str, Any]:
    if task.domain == "multilingual":
        row_id = f"{opts.id_prefix}ml-{task.lang_code}-{task.slot_i:06d}"
        lang = task.lang_code
    else:
        row_id = f"{opts.id_prefix}{task.domain}-{task.slot_i:07d}"
        lang = meta_fields.get("lang", "en")
    return {
        "id": row_id,
        "domain": task.domain,
        "lang": lang,
        "meta_prompt": meta,
        "conversation": [{"role": "user", "content": meta}],
        **{k: v for k, v in meta_fields.items() if k not in ("domain", "lang")},
        "_order": task.order,
    }


def _all_slot_tasks(cfg: MixConfig, plans: dict) -> list[_SlotTask]:
    tasks: list[_SlotTask] = []
    order = 0
    for domain in ("english", "code", "math"):
        plan = plans[domain]
        for i in range(plan.n_prompts):
            tasks.append(_SlotTask(order=order, domain=domain, slot_i=i))
            order += 1

    ml_plans = plans["multilingual"]
    for (lang_code, lang_name), plan in zip(cfg.languages.items(), ml_plans):
        for i in range(plan.n_prompts):
            tasks.append(_SlotTask(
                order=order,
                domain="multilingual",
                slot_i=i,
                lang_code=lang_code,
                lang_name=lang_name,
            ))
            order += 1
    return tasks


def _shard_tasks(tasks: list[_SlotTask], n_workers: int) -> list[list[_SlotTask]]:
    buckets: list[list[_SlotTask]] = [[] for _ in range(n_workers)]
    for task in tasks:
        buckets[task.order % n_workers].append(task)
    return buckets


def _process_prepare_shard(
    shard_tasks: list[_SlotTask],
    opts: _PromptBuildOpts,
    shard_id: int,
) -> _PrepareShardResult:
    dedup_filters: dict[str, DedupFilter | None] = {}
    resample_exhausted: Counter = Counter()
    decontam_stats: Counter = Counter()
    rows: list[dict[str, Any]] = []
    benchmark_index = _G_BENCH

    for n_done, task in enumerate(shard_tasks, start=1):
        rng = random.Random(_slot_seed(opts, task.domain, task.slot_i, task.lang_code))
        sampled = _sample_unique_prompt(
            rng, opts, dedup_filters, benchmark_index, decontam_stats,
            domain=task.domain,
            lang_code=task.lang_code,
            lang_name=task.lang_name,
        )
        if sampled is None:
            scope = (
                f"multilingual:{task.lang_code}"
                if task.domain == "multilingual"
                else task.domain
            )
            resample_exhausted[scope] += 1
            continue
        meta, meta_fields = sampled
        rows.append(_row_from_sample(opts, task, meta, meta_fields))
        if n_done % _PROGRESS_EVERY == 0:
            print(f"  shard {shard_id}: {n_done:,} slots processed", flush=True)

    by_scope = {
        scope: dict(f.stats) if f is not None else {}
        for scope, f in dedup_filters.items()
    }
    return _PrepareShardResult(
        rows=rows,
        resample_exhausted=resample_exhausted,
        decontam_stats=decontam_stats,
        by_scope=by_scope,
        written=len(rows),
    )


def _merge_prepare_results(results: list[_PrepareShardResult]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    resample_exhausted: Counter = Counter()
    decontam_stats: Counter = Counter()
    by_scope: dict[str, dict[str, int]] = {}

    for res in results:
        rows.extend(res.rows)
        resample_exhausted.update(res.resample_exhausted)
        decontam_stats.update(res.decontam_stats)
        for scope, stats in res.by_scope.items():
            merged = by_scope.setdefault(scope, Counter())
            merged.update(stats)
            by_scope[scope] = dict(merged)

    rows.sort(key=lambda r: r["_order"])
    for row in rows:
        row.pop("_order", None)

    dedup_stats: dict[str, Any] = {
        "resample_exhausted": dict(resample_exhausted),
        "decontam_rejected": dict(decontam_stats),
        "by_scope": by_scope,
    }
    return rows, dedup_stats


def _run_parallel_prepare(
    cfg: MixConfig,
    plans: dict,
    opts: _PromptBuildOpts,
    n_workers: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    global _G_BENCH
    if cfg.decontam_enabled:
        print(f"Loading benchmark index from {cfg.benchmark_index_path}...", flush=True)
        _G_BENCH = load_benchmark_index(cfg)
    else:
        _G_BENCH = None

    tasks = _all_slot_tasks(cfg, plans)
    shards = _shard_tasks(tasks, n_workers)
    active = [(shard_id, shard) for shard_id, shard in enumerate(shards) if shard]
    print(
        f"prepare: {len(tasks):,} slots across {len(active)} shards ({n_workers} workers)",
        flush=True,
    )

    if len(active) <= 1:
        _, shard = active[0]
        results = [_process_prepare_shard(shard, opts, 0)]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=min(n_workers, len(active))) as pool:
            results = pool.starmap(
                _process_prepare_shard,
                [(shard, opts, shard_id) for shard_id, shard in active],
            )

    return _merge_prepare_results(results)


def _iter_prompt_rows(
    cfg: MixConfig,
    plans: dict,
    dedup_stats: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    opts = _opts_from_cfg(cfg)
    rng = random.Random(cfg.seed)
    dedup_filters: dict[str, DedupFilter | None] = {}
    resample_exhausted = Counter()
    decontam_stats: Counter = Counter()
    benchmark_index: BenchmarkNgramIndex | None = None
    if cfg.decontam_enabled:
        benchmark_index = load_benchmark_index(cfg)

    for domain in ("english", "code", "math"):
        plan = plans[domain]
        for i in range(plan.n_prompts):
            sampled = _sample_unique_prompt(
                rng, opts, dedup_filters, benchmark_index, decontam_stats, domain=domain,
            )
            if sampled is None:
                resample_exhausted[domain] += 1
                continue
            meta, meta_fields = sampled
            yield _row_from_sample(
                opts, _SlotTask(order=0, domain=domain, slot_i=i), meta, meta_fields,
            )

    ml_plans = plans["multilingual"]
    for (lang_code, lang_name), plan in zip(cfg.languages.items(), ml_plans):
        scope = f"multilingual:{lang_code}"
        for i in range(plan.n_prompts):
            sampled = _sample_unique_prompt(
                rng, opts, dedup_filters, benchmark_index, decontam_stats,
                domain="multilingual", lang_code=lang_code, lang_name=lang_name,
            )
            if sampled is None:
                resample_exhausted[scope] += 1
                continue
            meta, meta_fields = sampled
            row = _row_from_sample(
                opts,
                _SlotTask(order=0, domain="multilingual", slot_i=i,
                          lang_code=lang_code, lang_name=lang_name),
                meta, meta_fields,
            )
            yield row

    dedup_stats["resample_exhausted"] = dict(resample_exhausted)
    dedup_stats["decontam_rejected"] = dict(decontam_stats)
    dedup_stats["by_scope"] = {
        scope: dict(f.stats) if f is not None else {}
        for scope, f in dedup_filters.items()
    }


def build_prompts(cfg: MixConfig, *, workers: int | None = None) -> tuple[Path, dict[str, Any]]:
    """Write prompts.jsonl and return its path plus dedup stats."""
    cfg.prompts_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.prompts_dir / "prompts.jsonl"
    stats_path = cfg.prompts_dir / "prompt_dedup_stats.json"
    plans = plan_budget(cfg)
    n_workers = workers if workers is not None else cfg.prepare_workers
    n_workers = max(1, n_workers)
    dedup_stats: dict[str, Any] = {
        "dedup_prompts": cfg.dedup_prompts,
        "prepare_workers": n_workers,
    }

    if n_workers == 1:
        n_written = 0
        with out.open("w", encoding="utf-8") as f:
            for row in _iter_prompt_rows(cfg, plans, dedup_stats):
                row.pop("_order", None)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_written += 1
                if n_written % _PROGRESS_EVERY == 0:
                    print(f"  prompts written: {n_written:,}", flush=True)
    else:
        opts = _opts_from_cfg(cfg)
        rows, shard_stats = _run_parallel_prepare(cfg, plans, opts, n_workers)
        dedup_stats.update(shard_stats)
        n_written = 0
        with out.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_written += 1
                if n_written % _PROGRESS_EVERY == 0:
                    print(f"  prompts written: {n_written:,}", flush=True)

    dedup_stats["prompts_written"] = n_written
    stats_path.write_text(json.dumps(dedup_stats, indent=2))
    print(f"prepare done: {n_written:,} prompts written", flush=True)
    return out, dedup_stats


def export_hf_disk(cfg: MixConfig, prompts_path: Path | None = None) -> Path:
    """Convert prompts.jsonl to a HuggingFace dataset on disk for inference-hive."""
    prompts_path = prompts_path or (cfg.prompts_dir / "prompts.jsonl")
    print(f"Exporting HF dataset from {prompts_path}...", flush=True)
    ds = hfds.load_dataset("json", data_files=str(prompts_path), split="train")
    ds = ds.map(
        lambda r: {
            "id": r["id"],
            "domain": r["domain"],
            "lang": r["lang"],
            "prompt": r["meta_prompt"],
        },
        remove_columns=ds.column_names,
        desc="export_hf_disk",
    )

    cfg.dataset_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.dataset_dir / "datamix-synth-completion"
    ds.save_to_disk(str(out))
    print(f"Wrote HF dataset: {out}", flush=True)
    return out


def build_all(cfg: MixConfig, *, workers: int | None = None) -> tuple[Path, Path, dict[str, Any]]:
    prompts, dedup_stats = build_prompts(cfg, workers=workers)
    dataset = export_hf_disk(cfg, prompts)
    return prompts, dataset, dedup_stats
