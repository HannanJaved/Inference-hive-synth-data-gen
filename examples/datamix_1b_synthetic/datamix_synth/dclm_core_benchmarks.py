"""Collect reference text for DCLM CORE benchmark decontamination."""
from __future__ import annotations

import json
import logging
import urllib.request
from collections.abc import Callable, Iterable, Iterator
from typing import Any

log = logging.getLogger(__name__)

BenchmarkCollector = Callable[[], list[str]]

DCLM_CORE_BENCHMARK_NAMES: tuple[str, ...] = (
    "agi_eval_lsat_ar",
    "arc_easy",
    "arc_challenge",
    "bigbench_qa_wikidata",
    "bigbench_dyck_languages",
    "bigbench_operators",
    "bigbench_repeat_copy_logic",
    "bigbench_cs_algorithms",
    "bigbench_language_identification",
    "boolq",
    "commonsense_qa",
    "copa",
    "coqa",
    "hellaswag",
    "jeopardy",
    "lambada",
    "openbookqa",
    "piqa",
    "squad",
    "winograd_wsc",
    "winogrande",
)


def _flatten_strings(*parts: Any) -> list[str]:
    out: list[str] = []
    for part in parts:
        if part is None:
            continue
        if isinstance(part, str):
            s = part.strip()
            if s:
                out.append(s)
        elif isinstance(part, (list, tuple, set)):
            for item in part:
                out.extend(_flatten_strings(item))
        elif isinstance(part, dict):
            for value in part.values():
                out.extend(_flatten_strings(value))
    return out


def _all_rows(ds_dict: Any) -> Iterator[dict[str, Any]]:
    for split in ds_dict:
        for row in ds_dict[split]:
            yield row


def _load_hf_dataset(*args: Any, **kwargs: Any) -> Any:
    from datasets import load_dataset

    kwargs.pop("trust_remote_code", None)
    return load_dataset(*args, **kwargs)


def _collect_arc(config: str) -> list[str]:
    ds = _load_hf_dataset("allenai/ai2_arc", config)
    texts: list[str] = []
    for row in _all_rows(ds):
        choices = row.get("choices") or {}
        texts.extend(_flatten_strings(row.get("question"), choices.get("text")))
    return texts


def _collect_bigbench(task: str) -> list[str]:
    ds = _load_hf_dataset("tasksource/bigbench", task)
    texts: list[str] = []
    for row in _all_rows(ds):
        texts.extend(_flatten_strings(
            row.get("inputs"),
            row.get("input"),
            row.get("targets"),
            row.get("target"),
            row.get("multiple_choice_targets"),
            row.get("multiple_choice_scores"),
        ))
        if "targets" in row and isinstance(row["targets"], list):
            for target in row["targets"]:
                if isinstance(target, dict):
                    texts.extend(_flatten_strings(target.get("choice"), target.get("answer")))
    return texts


def _collect_agi_eval_lsat_ar() -> list[str]:
    candidates = [
        ("sagnikrayc/AGIEval", "lsat-ar"),
        ("AI-ModelScope/AGIEval", "lsat-ar"),
        ("hails/agi_eval", "lsat-ar"),
    ]
    for path, config in candidates:
        try:
            ds = _load_hf_dataset(path, config)
            texts: list[str] = []
            for row in _all_rows(ds):
                texts.extend(_flatten_strings(
                    row.get("query"),
                    row.get("question"),
                    row.get("passage"),
                    row.get("options"),
                    row.get("choices"),
                    row.get("label"),
                ))
            if texts:
                return texts
        except Exception as exc:
            log.debug("AGIEval loader %s/%s failed: %s", path, config, exc)

    url = "https://raw.githubusercontent.com/ruixiangcui/AGIEval/main/data/v1/lsat-ar.jsonl"
    texts = []
    with urllib.request.urlopen(url, timeout=60) as resp:
        for line in resp:
            if not line.strip():
                continue
            row = json.loads(line)
            texts.extend(_flatten_strings(
                row.get("query"),
                row.get("question"),
                row.get("passage"),
                row.get("options"),
                row.get("choices"),
            ))
    return texts


def _collect_boolq() -> list[str]:
    ds = _load_hf_dataset("super_glue", "boolq")
    texts: list[str] = []
    for row in _all_rows(ds):
        texts.extend(_flatten_strings(row.get("question"), row.get("passage")))
    return texts


def _collect_commonsense_qa() -> list[str]:
    ds = _load_hf_dataset("commonsense_qa")
    texts: list[str] = []
    for row in _all_rows(ds):
        choices = row.get("choices") or {}
        texts.extend(_flatten_strings(row.get("question"), choices.get("text"), choices.get("label")))
    return texts


def _collect_copa() -> list[str]:
    ds = _load_hf_dataset("super_glue", "copa")
    texts: list[str] = []
    for row in _all_rows(ds):
        texts.extend(_flatten_strings(
            row.get("premise"),
            row.get("question"),
            row.get("choice1"),
            row.get("choice2"),
        ))
    return texts


def _collect_coqa() -> list[str]:
    ds = _load_hf_dataset("EleutherAI/coqa")
    texts: list[str] = []
    for row in _all_rows(ds):
        story = row.get("story", "")
        questions = (row.get("questions") or {}).get("input_text") or []
        answers = (row.get("answers") or {}).get("input_text") or []
        texts.append(story)
        for q, a in zip(questions, answers):
            texts.extend(_flatten_strings(q, a))
    return texts


def _collect_hellaswag() -> list[str]:
    ds = _load_hf_dataset("hellaswag")
    texts: list[str] = []
    for row in _all_rows(ds):
        ctx = f"{row.get('ctx_a', '')} {row.get('ctx_b', '')}".strip()
        texts.extend(_flatten_strings(row.get("activity_label"), ctx, row.get("endings")))
    return texts


def _collect_jeopardy() -> list[str]:
    ds = _load_hf_dataset("soldni/jeopardy", "all_questions")
    texts: list[str] = []
    for row in _all_rows(ds):
        texts.extend(_flatten_strings(
            row.get("question"),
            row.get("answer"),
            row.get("continuation"),
        ))
    return texts


def _collect_lambada() -> list[str]:
    ds = _load_hf_dataset("EleutherAI/lambada_openai")
    texts: list[str] = []
    for row in _all_rows(ds):
        texts.extend(_flatten_strings(row.get("text")))
    return texts


def _collect_openbookqa() -> list[str]:
    ds = _load_hf_dataset("allenai/openbookqa", "main")
    texts: list[str] = []
    for row in _all_rows(ds):
        choices = row.get("choices") or {}
        texts.extend(_flatten_strings(row.get("question_stem"), choices.get("text"), choices.get("label")))
    return texts


def _collect_piqa() -> list[str]:
    for path in ("baber/piqa", "piqa"):
        try:
            ds = _load_hf_dataset(path)
            texts: list[str] = []
            for row in _all_rows(ds):
                texts.extend(_flatten_strings(row.get("goal"), row.get("sol1"), row.get("sol2")))
            if texts:
                return texts
        except Exception as exc:
            log.debug("PIQA loader %s failed: %s", path, exc)
    raise RuntimeError("could not load PIQA from HuggingFace")


def _collect_squad() -> list[str]:
    ds = _load_hf_dataset("squad")
    texts: list[str] = []
    for row in _all_rows(ds):
        answers = row.get("answers") or {}
        texts.extend(_flatten_strings(row.get("context"), row.get("question"), answers.get("text")))
    return texts


def _collect_winograd_wsc() -> list[str]:
    try:
        ds = _load_hf_dataset("winograd_wsc", revision="refs/convert/parquet")
    except Exception as exc:
        log.debug("winograd_wsc parquet load failed: %s", exc)
        ds = _load_hf_dataset("winograd_wsc", "wsc273")
    texts: list[str] = []
    for row in _all_rows(ds):
        texts.extend(_flatten_strings(row.get("sentence"), row.get("text"), row.get("options")))
    return texts


def _collect_winogrande() -> list[str]:
    ds = _load_hf_dataset("allenai/winogrande", "winogrande_xl")
    texts: list[str] = []
    for row in _all_rows(ds):
        texts.extend(_flatten_strings(row.get("sentence"), row.get("option1"), row.get("option2")))
    return texts


_COLLECTORS: dict[str, BenchmarkCollector] = {
    "agi_eval_lsat_ar": _collect_agi_eval_lsat_ar,
    "arc_easy": lambda: _collect_arc("ARC-Easy"),
    "arc_challenge": lambda: _collect_arc("ARC-Challenge"),
    "bigbench_qa_wikidata": lambda: _collect_bigbench("qa_wikidata"),
    "bigbench_dyck_languages": lambda: _collect_bigbench("dyck_languages"),
    "bigbench_operators": lambda: _collect_bigbench("operators"),
    "bigbench_repeat_copy_logic": lambda: _collect_bigbench("repeat_copy_logic"),
    "bigbench_cs_algorithms": lambda: _collect_bigbench("cs_algorithms"),
    "bigbench_language_identification": lambda: _collect_bigbench("language_identification"),
    "boolq": _collect_boolq,
    "commonsense_qa": _collect_commonsense_qa,
    "copa": _collect_copa,
    "coqa": _collect_coqa,
    "hellaswag": _collect_hellaswag,
    "jeopardy": _collect_jeopardy,
    "lambada": _collect_lambada,
    "openbookqa": _collect_openbookqa,
    "piqa": _collect_piqa,
    "squad": _collect_squad,
    "winograd_wsc": _collect_winograd_wsc,
    "winogrande": _collect_winogrande,
}


def collect_benchmark_texts(
    names: Iterable[str] | None = None,
) -> dict[str, list[str]]:
    """Return raw reference strings per benchmark."""
    selected = list(names) if names is not None else list(DCLM_CORE_BENCHMARK_NAMES)
    out: dict[str, list[str]] = {}
    for name in selected:
        collector = _COLLECTORS.get(name)
        if collector is None:
            raise KeyError(f"unknown DCLM CORE benchmark: {name}")
        texts = collector()
        out[name] = texts
        log.info("Collected %s strings from %s", len(texts), name)
    return out
