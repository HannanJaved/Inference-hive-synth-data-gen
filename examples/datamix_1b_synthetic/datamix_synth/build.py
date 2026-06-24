"""Build prompt rows and export inference-hive-ready datasets."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterator

import datasets as hfds

from datamix_synth.budget import plan_budget
from datamix_synth.config import MixConfig
from datamix_synth.prompts import build_meta_prompt


def _iter_prompt_rows(cfg: MixConfig, plans: dict) -> Iterator[dict[str, Any]]:
    rng = random.Random(cfg.seed)
    counter = 0

    for domain in ("english", "code", "math"):
        plan = plans[domain]
        for i in range(plan.n_prompts):
            meta, meta_fields = build_meta_prompt(rng, domain)
            row_id = f"{cfg.id_prefix}{domain}-{i:07d}"
            yield {
                "id": row_id,
                "domain": domain,
                "lang": meta_fields.get("lang", "en"),
                "meta_prompt": meta,
                "conversation": [{"role": "user", "content": meta}],
                **{k: v for k, v in meta_fields.items() if k not in ("domain", "lang")},
            }
            counter += 1

    ml_plans = plans["multilingual"]
    for (lang_code, lang_name), plan in zip(cfg.languages.items(), ml_plans):
        for i in range(plan.n_prompts):
            meta, meta_fields = build_meta_prompt(
                rng, "multilingual", lang_code=lang_code, lang_name=lang_name,
            )
            row_id = f"{cfg.id_prefix}ml-{lang_code}-{i:06d}"
            yield {
                "id": row_id,
                "domain": "multilingual",
                "lang": lang_code,
                "meta_prompt": meta,
                "conversation": [{"role": "user", "content": meta}],
                **{k: v for k, v in meta_fields.items() if k not in ("domain", "lang")},
            }
            counter += 1


def build_prompts(cfg: MixConfig) -> Path:
    """Write prompts.jsonl and return its path."""
    cfg.prompts_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.prompts_dir / "prompts.jsonl"
    plans = plan_budget(cfg)

    with out.open("w", encoding="utf-8") as f:
        for row in _iter_prompt_rows(cfg, plans):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return out


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


def build_all(cfg: MixConfig) -> tuple[Path, Path]:
    prompts = build_prompts(cfg)
    dataset = export_hf_disk(cfg, prompts)
    return prompts, dataset
