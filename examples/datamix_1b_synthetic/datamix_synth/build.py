"""Build prompt rows and export inference-hive-ready datasets."""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import datasets as hfds

from datamix_synth.budget import plan_budget
from datamix_synth.config import MixConfig
from datamix_synth.decontam import BenchmarkNgramIndex, load_benchmark_index
from datamix_synth.dedup import DedupFilter, normalize_prompt_for_dedup
from datamix_synth.prompts import build_meta_prompt


def _dedup_filter_for_scope(cfg: MixConfig) -> DedupFilter | None:
    if not cfg.dedup_prompts:
        return None
    return DedupFilter(
        near_threshold=cfg.dedup_prompt_near_threshold,
        shingle_size=cfg.dedup_shingle_size,
        normalize_fn=normalize_prompt_for_dedup,
    )


def _sample_unique_prompt(
    rng: random.Random,
    cfg: MixConfig,
    dedup: DedupFilter | None,
    benchmark_index: BenchmarkNgramIndex | None,
    decontam_stats: Counter,
    *,
    domain: str,
    lang_code: str | None = None,
    lang_name: str | None = None,
) -> tuple[str, dict[str, Any]] | None:
    for _ in range(cfg.dedup_prompt_max_resamples):
        meta, meta_fields = build_meta_prompt(
            rng, domain, lang_code=lang_code, lang_name=lang_name,
        )
        if benchmark_index is not None and benchmark_index.is_contaminated(meta):
            decontam_stats["benchmark_contamination"] += 1
            continue
        if dedup is None:
            return meta, meta_fields
        accepted, _reason = dedup.try_add(meta)
        if accepted:
            return meta, meta_fields
    return None


def _iter_prompt_rows(
    cfg: MixConfig,
    plans: dict,
    dedup_stats: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    rng = random.Random(cfg.seed)
    domain_filters: dict[str, DedupFilter | None] = {}
    resample_exhausted = Counter()
    decontam_stats: Counter = Counter()
    benchmark_index: BenchmarkNgramIndex | None = None
    if cfg.decontam_enabled:
        benchmark_index = load_benchmark_index(cfg)

    for domain in ("english", "code", "math"):
        plan = plans[domain]
        scope = domain
        dedup = _dedup_filter_for_scope(cfg)
        domain_filters[scope] = dedup
        for i in range(plan.n_prompts):
            sampled = _sample_unique_prompt(
                rng, cfg, dedup, benchmark_index, decontam_stats, domain=domain,
            )
            if sampled is None:
                resample_exhausted[domain] += 1
                continue
            meta, meta_fields = sampled
            row_id = f"{cfg.id_prefix}{domain}-{i:07d}"
            yield {
                "id": row_id,
                "domain": domain,
                "lang": meta_fields.get("lang", "en"),
                "meta_prompt": meta,
                "conversation": [{"role": "user", "content": meta}],
                **{k: v for k, v in meta_fields.items() if k not in ("domain", "lang")},
            }

    ml_plans = plans["multilingual"]
    for (lang_code, lang_name), plan in zip(cfg.languages.items(), ml_plans):
        scope = f"multilingual:{lang_code}"
        dedup = _dedup_filter_for_scope(cfg)
        domain_filters[scope] = dedup
        for i in range(plan.n_prompts):
            sampled = _sample_unique_prompt(
                rng, cfg, dedup, benchmark_index, decontam_stats,
                domain="multilingual", lang_code=lang_code, lang_name=lang_name,
            )
            if sampled is None:
                resample_exhausted[scope] += 1
                continue
            meta, meta_fields = sampled
            row_id = f"{cfg.id_prefix}ml-{lang_code}-{i:06d}"
            yield {
                "id": row_id,
                "domain": "multilingual",
                "lang": lang_code,
                "meta_prompt": meta,
                "conversation": [{"role": "user", "content": meta}],
                **{k: v for k, v in meta_fields.items() if k not in ("domain", "lang")},
            }

    dedup_stats["resample_exhausted"] = dict(resample_exhausted)
    dedup_stats["decontam_rejected"] = dict(decontam_stats)
    dedup_stats["by_scope"] = {
        scope: dict(f.stats) if f is not None else {}
        for scope, f in domain_filters.items()
    }


def build_prompts(cfg: MixConfig) -> tuple[Path, dict[str, Any]]:
    """Write prompts.jsonl and return its path plus dedup stats."""
    cfg.prompts_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.prompts_dir / "prompts.jsonl"
    stats_path = cfg.prompts_dir / "prompt_dedup_stats.json"
    plans = plan_budget(cfg)
    dedup_stats: dict[str, Any] = {"dedup_prompts": cfg.dedup_prompts}

    n_written = 0
    with out.open("w", encoding="utf-8") as f:
        for row in _iter_prompt_rows(cfg, plans, dedup_stats):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_written += 1

    dedup_stats["prompts_written"] = n_written
    stats_path.write_text(json.dumps(dedup_stats, indent=2))
    return out, dedup_stats


def export_hf_disk(cfg: MixConfig, prompts_path: Path | None = None) -> Path:
    """Convert prompts.jsonl to a HuggingFace dataset on disk for inference-hive."""
    prompts_path = prompts_path or (cfg.prompts_dir / "prompts.jsonl")
    rows = [json.loads(line) for line in prompts_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    # inference-hive needs id + prompt strings (completion API; Datamix 9B has no chat template).
    ds = hfds.Dataset.from_list([
        {
            "id": r["id"],
            "domain": r["domain"],
            "lang": r["lang"],
            "prompt": r["meta_prompt"],
        }
        for r in rows
    ])

    cfg.dataset_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.dataset_dir / "datamix-synth-completion"
    ds.save_to_disk(str(out))
    return out


def build_all(cfg: MixConfig) -> tuple[Path, Path, dict[str, Any]]:
    prompts, dedup_stats = build_prompts(cfg)
    dataset = export_hf_disk(cfg, prompts)
    return prompts, dataset, dedup_stats
