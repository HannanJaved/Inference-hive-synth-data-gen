"""Domain-specific continuation prefixes for base-LM synthetic generation."""
from __future__ import annotations

import random
from typing import Any

from datamix_synth.prefixes import (
    code_prefix,
    english_prefix,
    english_topic,
    math_prefix,
    multilingual_prefix,
    prefix_metadata,
)

# Re-export subtopic lists for budget/tests.
ENGLISH_SUBTOPICS: list[tuple[str, float]] = [
    ("science_explainer", 0.18),
    ("history_culture", 0.14),
    ("technology_overview", 0.14),
    ("how_to_guide", 0.12),
    ("biography_profile", 0.10),
    ("health_wellness", 0.10),
    ("economics_society", 0.10),
    ("nature_environment", 0.12),
]

CODE_SUBTOPICS: list[tuple[str, float]] = [
    ("python_script", 0.30),
    ("data_processing", 0.20),
    ("algorithms", 0.15),
    ("web_api", 0.15),
    ("testing", 0.10),
    ("cli_tool", 0.10),
]

MATH_SUBTOPICS: list[tuple[str, float]] = [
    ("arithmetic", 0.20),
    ("fractions_decimals", 0.18),
    ("algebra", 0.22),
    ("geometry", 0.18),
    ("word_problem", 0.22),
]

CODE_LANGS = ["python", "javascript", "typescript", "java", "go", "rust", "c++", "sql"]


def _weighted_choice(rng: random.Random, pairs: list[tuple[str, float]]) -> str:
    r = rng.random()
    acc = 0.0
    for item, w in pairs:
        acc += w
        if r <= acc:
            return item
    return pairs[-1][0]


def _english_meta(rng: random.Random) -> tuple[str, dict[str, Any]]:
    sub = _weighted_choice(rng, ENGLISH_SUBTOPICS)
    topic = english_topic(rng, sub)
    prompt = english_prefix(rng, sub, topic=topic)
    return prompt, prefix_metadata("english", lang="en", subtopic=sub, topic=topic)


def _code_meta(rng: random.Random) -> tuple[str, dict[str, Any]]:
    sub = _weighted_choice(rng, CODE_SUBTOPICS)
    lang = rng.choice(CODE_LANGS)
    prompt = code_prefix(rng, sub, lang)
    return prompt, prefix_metadata("code", lang="en", subtopic=sub, code_lang=lang)


def _math_meta(rng: random.Random) -> tuple[str, dict[str, Any]]:
    sub = _weighted_choice(rng, MATH_SUBTOPICS)
    prompt = math_prefix(rng, sub)
    return prompt, prefix_metadata("math", lang="en", subtopic=sub)


def _multilingual_meta(rng: random.Random, lang_code: str, lang_name: str) -> tuple[str, dict[str, Any]]:
    prompt = multilingual_prefix(rng, lang_code, lang_name)
    return prompt, prefix_metadata("multilingual", lang=lang_code, lang_name=lang_name)


def build_meta_prompt(
    rng: random.Random,
    domain: str,
    *,
    lang_code: str | None = None,
    lang_name: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Return (continuation_prefix, metadata). The prefix is sent to the completion API."""
    if domain == "english":
        return _english_meta(rng)
    if domain == "code":
        return _code_meta(rng)
    if domain == "math":
        return _math_meta(rng)
    if domain == "multilingual":
        if not lang_code or not lang_name:
            raise ValueError("multilingual prompts require lang_code and lang_name")
        return _multilingual_meta(rng, lang_code, lang_name)
    raise ValueError(f"unknown domain: {domain}")
