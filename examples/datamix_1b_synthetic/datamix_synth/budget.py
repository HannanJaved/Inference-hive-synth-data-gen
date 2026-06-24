"""Token-budget planning for the 1B synthetic corpus."""
from __future__ import annotations

import json
import math
from pathlib import Path

from datamix_synth.config import DomainBudget, MixConfig


def plan_budget(cfg: MixConfig) -> dict[str, DomainBudget | list[DomainBudget]]:
    """Compute prompt counts per domain to hit cfg.total_tokens (completion tokens)."""
    cfg.validate()
    plans: dict[str, DomainBudget | list[DomainBudget]] = {}

    for domain, frac in cfg.domain_fractions.items():
        if domain == "multilingual":
            continue
        target = int(cfg.total_tokens * frac)
        avg = cfg.avg_tokens[domain]
        n = max(1, math.ceil(target / avg))
        plans[domain] = DomainBudget(domain=domain, token_target=target,
                                   avg_completion_tokens=avg, n_prompts=n)

    ml_target = int(cfg.total_tokens * cfg.multilingual_fraction)
    ml_avg = cfg.avg_tokens["multilingual"]
    langs = list(cfg.languages.items())
    if not langs:
        raise ValueError("no languages configured for multilingual domain")

    per_lang_target = ml_target // len(langs)
    per_lang_n = max(1, math.ceil(per_lang_target / ml_avg))
    ml_plans: list[DomainBudget] = []
    for code, _name in langs:
        ml_plans.append(DomainBudget(
            domain="multilingual",
            token_target=per_lang_target,
            avg_completion_tokens=ml_avg,
            n_prompts=per_lang_n,
        ))
    plans["multilingual"] = ml_plans

    return plans


def summarize_plan(cfg: MixConfig, plans: dict) -> dict:
    rows = []
    total_prompts = 0
    est_tokens = 0

    for domain, plan in plans.items():
        if domain == "multilingual":
            for i, p in enumerate(plan):
                lang = list(cfg.languages.keys())[i]
                rows.append({
                    "domain": domain, "lang": lang,
                    "n_prompts": p.n_prompts,
                    "token_target": p.token_target,
                    "avg_completion_tokens": p.avg_completion_tokens,
                    "est_completion_tokens": p.n_prompts * p.avg_completion_tokens,
                })
                total_prompts += p.n_prompts
                est_tokens += p.n_prompts * p.avg_completion_tokens
        else:
            rows.append({
                "domain": domain, "lang": "en" if domain != "multilingual" else None,
                "n_prompts": plan.n_prompts,
                "token_target": plan.token_target,
                "avg_completion_tokens": plan.avg_completion_tokens,
                "est_completion_tokens": plan.n_prompts * plan.avg_completion_tokens,
            })
            total_prompts += plan.n_prompts
            est_tokens += plan.n_prompts * plan.avg_completion_tokens

    return {
        "total_tokens_target": cfg.total_tokens,
        "total_prompts": total_prompts,
        "estimated_completion_tokens": est_tokens,
        "domain_fractions": cfg.domain_fractions,
        "by_shard": rows,
    }


def write_plan(cfg: MixConfig, path: Path) -> dict:
    plans = plan_budget(cfg)
    summary = summarize_plan(cfg, plans)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))
    return summary
