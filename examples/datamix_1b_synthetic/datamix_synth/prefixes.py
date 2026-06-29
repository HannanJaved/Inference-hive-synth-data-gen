"""Continuation prefixes for base-LM synthetic generation.

Base models continue text; they do not follow imperative instructions well.
Each prefix mimics the *start* of a real pretraining document in the target domain.
"""
from __future__ import annotations

import random
from typing import Any

from datamix_synth.procedural import (
    procedural_code_task,
    procedural_english_topic,
    procedural_ml_title,
)

# --- English (Nemotron-CC-like web/educational prose) ---

ENGLISH_TOPICS: dict[str, list[str]] = {
    "science_explainer": [
        "Photosynthesis", "Plate tectonics", "The water cycle", "Newton's laws of motion",
        "Cell division", "The greenhouse effect", "Electric circuits", "The human immune system",
        "Sound waves and hearing", "Magnetism in everyday life",
    ],
    "history_culture": [
        "The Roman Republic", "The Silk Road", "The printing press", "The Industrial Revolution",
        "Ancient Egyptian agriculture", "The Han dynasty", "Medieval trade fairs",
        "The abolition of slavery", "Women's suffrage", "The fall of Constantinople",
    ],
    "technology_overview": [
        "How DNS works", "Solid-state drives", "Version control with Git", "Public-key cryptography",
        "Solar panel efficiency", "5G mobile networks", "Computer memory hierarchies",
        "Optical fiber communication", "Lithium-ion batteries", "Machine learning basics",
    ],
    "how_to_guide": [
        "How to prepare sourdough bread", "How to prune tomato plants", "How to read a topographic map",
        "How to balance a household budget", "How to pack a hiking backpack",
        "How to clean and maintain bicycle gears", "How to write a clear email",
        "How to organize digital photos", "How to measure room dimensions",
        "How to start seeds indoors",
    ],
    "biography_profile": [
        "Marie Curie", "Ada Lovelace", "Nelson Mandela", "Hypatia of Alexandria",
        "Ibn al-Haytham", "Rosalind Franklin", "Sitting Bull", "Grace Hopper",
        "Frida Kahlo", "Sun Yat-sen",
    ],
    "health_wellness": [
        "Sleep hygiene", "Hydration and exercise", "Posture at a desk",
        "Mindful breathing", "Stretching for office workers", "Balanced meal planning",
        "Walking as daily exercise", "Managing screen time", "Hand-washing technique",
        "Recovery after mild muscle soreness",
    ],
    "economics_society": [
        "Supply and demand", "Inflation and purchasing power", "Urban public transit",
        "Microfinance in rural areas", "Property taxes explained", "Trade agreements",
        "Volunteering in communities", "Recycling programs", "Local farmers markets",
        "Affordable housing policy",
    ],
    "nature_environment": [
        "The Amazon rainforest", "Coral reef ecosystems", "Alpine meadow wildflowers",
        "Migratory bird routes", "River delta formation", "Desert adaptation in plants",
        "Temperate deciduous forests", "Wetland filtration", "Volcanic soil fertility",
        "Coastal erosion",
    ],
}

ENGLISH_OPENINGS = [
    "{topic} is an important subject that helps explain how the world works. At its core, ",
    "Researchers and educators often return to {topic} because it connects theory with everyday experience. ",
    "Many readers first encounter {topic} in school, yet its implications extend far beyond the classroom. ",
    "Over the past century, understanding {topic} has shaped public debate, policy, and technology. ",
    "To appreciate {topic}, it helps to begin with a few basic facts and then examine how they fit together. ",
    "Historians, scientists, and practitioners have documented {topic} from several angles. One useful starting point is ",
    "In recent years, public interest in {topic} has grown as new evidence reshapes older assumptions. ",
    "A careful reading of {topic} reveals patterns that are easy to miss on a first pass. ",
    "When communities discuss {topic}, they often weigh practical trade-offs alongside long-term goals. ",
    "Textbooks introduce {topic} with simplified models, but real situations tend to be more nuanced. ",
    "Journalists covering {topic} frequently highlight case studies that illustrate broader principles. ",
    "Students sometimes find {topic} abstract until they connect it to familiar experiences. ",
    "Policy makers debating {topic} must balance scientific findings with local constraints. ",
    "Engineers working near {topic} routinely test assumptions against measured data. ",
    "Teachers explaining {topic} often begin with a concrete example before generalizing. ",
    "Archaeological and archival sources occasionally add surprising detail to {topic}. ",
    "Readers comparing international approaches to {topic} notice both shared themes and local variation. ",
    "Laboratory studies of {topic} complement field observations gathered over many seasons. ",
    "Small-scale experiments can clarify one mechanism involved in {topic}, though scale effects matter. ",
    "Debates about {topic} sometimes hinge on definitions that different experts use differently. ",
    "Practitioners who work with {topic} daily develop intuitions that formal models later capture. ",
    "Review articles on {topic} summarize decades of work while noting open questions. ",
    "Citizen groups monitoring {topic} contribute observations that professional surveys may overlook. ",
    "Historical chronicles show how attitudes toward {topic} shifted across generations. ",
    "Modern instruments allow finer measurements related to {topic} than were possible a century ago. ",
    "Comparing {topic} across regions highlights environmental and cultural factors that interact. ",
    "Introductory lectures on {topic} often map the main concepts before discussing exceptions. ",
    "Writers describing {topic} for a general audience avoid jargon while preserving accuracy. ",
    "Funding priorities influence which aspects of {topic} receive the most research attention. ",
    "Seasonal cycles sometimes obscure longer trends visible only after years of {topic} records. ",
    "Ethical questions surrounding {topic} arise whenever new applications reach the public. ",
]

ENGLISH_HEADERS = [
    "{topic}\n\n",
    "# {topic}\n\n",
    "## {topic}\n\n",
    "{topic}\n\nIntroduction\n\n",
    "{topic}\n\nOverview\n\n",
    "{topic}\n\nBackground\n\n",
    "{topic}\n\nSummary\n\n",
    "Topic: {topic}\n\n",
    "{topic}\n\nChapter 1\n\n",
    "{topic}\n\nNotes\n\n",
]

# --- Code (StarCoder-like file starts) ---

CODE_PYTHON_HEADERS = [
    '"""{task}"""\n\nimport json\nfrom pathlib import Path\nfrom typing import Any, Iterator\n\n',
    '"""{task}"""\n\nfrom __future__ import annotations\n\nimport csv\nfrom dataclasses import dataclass\n\n',
    '"""{task}"""\n\nimport argparse\nimport logging\nfrom collections import defaultdict\n\n',
]

CODE_PYTHON_BODY = [
    "def load_rows(path: str) -> Iterator[dict[str, Any]]:\n    \"\"\"Read records from a newline-delimited JSON file.\"\"\"\n    with Path(path).open(encoding=\"utf-8\") as f:\n        for line in f:\n            line = line.strip()\n            if line:\n                yield json.loads(line)\n\n\n",
    "def normalize(value: str) -> str:\n    \"\"\"Return a cleaned string suitable for downstream processing.\"\"\"\n    return value.strip().lower()\n\n\n",
    "class RecordStore:\n    \"\"\"Simple in-memory store keyed by identifier.\"\"\"\n\n    def __init__(self) -> None:\n        self._items: dict[str, Any] = {}\n\n    def put(self, key: str, value: Any) -> None:\n        self._items[key] = value\n\n",
]

CODE_JS_HEADERS = [
    "// {task}\n\n'use strict';\n\n",
    "/**\n * {task}\n */\n\n",
]

CODE_JS_BODY = [
    "function parseRecords(text) {\n  return text\n    .split('\\n')\n    .filter(Boolean)\n    .map((line) => JSON.parse(line));\n}\n\n",
    "function normalizeKey(value) {\n  return String(value).trim().toLowerCase();\n}\n\n",
]

CODE_SQL_HEADERS = [
    "-- {task}\n\n",
]

CODE_SQL_BODY = [
    "WITH cleaned AS (\n  SELECT\n    id,\n    lower(trim(name)) AS name,\n    created_at\n  FROM raw_events\n  WHERE created_at IS NOT NULL\n)\n",
    "SELECT\n  category,\n  count(*) AS n\nFROM sales\nGROUP BY category\nORDER BY n DESC;\n\n",
]

CODE_TASKS = {
    "python_script": [
        "Load and validate JSONL records",
        "Aggregate events by category",
        "Parse command-line arguments for a batch job",
    ],
    "data_processing": [
        "Normalize messy CSV columns before export",
        "Merge two JSON streams on a shared key",
        "Filter invalid rows from a dataset",
    ],
    "algorithms": [
        "Binary search on a sorted array",
        "Breadth-first traversal of a graph",
        "Compute rolling averages over a numeric series",
    ],
    "web_api": [
        "Fetch paginated JSON from a REST endpoint",
        "Handle HTTP errors when calling a remote service",
        "Build query parameters for a search API",
    ],
    "testing": [
        "Unit tests for a string normalization helper",
        "Tests for a small record parser",
        "Tests for a date formatting utility",
    ],
    "cli_tool": [
        "CLI entry point for converting CSV to JSON",
        "CLI utility that summarizes a log file",
        "CLI wrapper around a file processing pipeline",
    ],
}

# --- Multilingual article starts (HPLT-like) ---

ML_TITLES: dict[str, list[str]] = {
    "de": [
        "Die Geschichte der Windenergie in Deutschland",
        "Traditionelles Backen in Süddeutschland",
        "Der Rhein und seine Bedeutung für den Handel",
        "Städtische Mobilität und öffentlicher Nahverkehr",
        "Die Entwicklung der Buchdruckerkunst",
        "Alpine Landwirtschaft im Wandel",
        "Festivals und Brauchtum in Bayern",
        "Klimafreundliches Bauen mit Holz",
    ],
    "fr": [
        "Les châteaux de la Loire",
        "La viticulture en Bourgogne",
        "Le développement du métro parisien",
        "Les marchés alimentaires en Provence",
        "L'histoire de la presse écrite",
        "La faune des Pyrénées",
        "Les écoles de cuisine française",
        "La transition énergétique en Bretagne",
    ],
    "es": [
        "La historia del olivo en Andalucía",
        "El Camino de Santiago",
        "La arquitectura mudéjar",
        "El transporte urbano en Madrid",
        "La biodiversidad del Parque Nacional de Doñana",
        "La industria textil en Cataluña",
        "Las fiestas locales en Galicia",
        "La energía solar en España",
    ],
    "pl": [
        "Historia Wisły i handlu rzecznego",
        "Tradycyjne wypieki chleba na żurem",
        "Rozwój transportu publicznego w Warszawie",
        "Bioróżnorodność Puszczy Białowieskiej",
        "Sztuka współczesna w Krakowie",
        "Rolnictwo ekologiczne na Warmii",
        "Zabytki architektury drewnianej na Podlasiu",
        "Energia wiatrowa na Bałtyku",
    ],
}

ML_OPENINGS: dict[str, list[str]] = {
    "de": [
        "Schon seit Jahrhunderten spielt dieses Thema eine Rolle im Alltag vieler Menschen. Besonders deutlich wird das, wenn ",
        "Fachleute betonen, dass sich hier mehrere Entwicklungen überschneiden. Ein wichtiger Ausgangspunkt ist ",
        "In den letzten Jahrzehnten hat sich das Bild deutlich verändert. Heute lässt sich beobachten, dass ",
        "Für Besucher und Einheimische ist der Zusammenhang oft überraschend vielfältig. Zunächst ",
    ],
    "fr": [
        "Depuis longtemps, ce sujet occupe une place importante dans la vie locale. On observe notamment que ",
        "Les spécialistes rappellent que plusieurs facteurs entrent en jeu. Pour comprendre l'ensemble, il faut d'abord ",
        "Au fil des décennies, la situation a beaucoup évolué. Aujourd'hui, ",
        "Les visiteurs découvrent souvent des détails inattendus. En commençant par ",
    ],
    "es": [
        "Desde hace siglos, este tema forma parte de la vida cotidiana en la región. En particular, ",
        "Los expertos señalan que intervienen varios factores a la vez. Para entenderlo mejor, conviene ",
        "En las últimas décadas, la situación ha cambiado de forma notable. Hoy ",
        "Quienes visitan la zona suelen fijarse en detalles sorprendentes. En primer lugar, ",
    ],
    "pl": [
        "Od wielu lat temat ten ma znaczenie w życiu lokalnej społeczności. Szczególnie widać to, gdy ",
        "Specjaliści podkreślają, że na sytuację wpływa kilka czynników naraz. Aby to zrozumieć, warto zacząć od tego, że ",
        "W ostatnich dekadach obraz sytuacji wyraźnie się zmienił. Obecnie ",
        "Odwiedzający często zauważają detale, które łatwo pominąć. Na początek ",
    ],
}

# Fallback for other configured languages: reuse Spanish templates with localized title slot.
ML_GENERIC_OPENINGS = ML_OPENINGS["es"]


def _pick(rng: random.Random, items: list[str]) -> str:
    return items[rng.randrange(len(items))]


def english_topic(rng: random.Random, subtopic: str) -> str:
    """Pick a static or procedurally generated English topic (high combinatorial diversity)."""
    if rng.random() < 0.15:
        return _pick(rng, ENGLISH_TOPICS[subtopic])
    return procedural_english_topic(rng, subtopic)


def english_prefix(rng: random.Random, subtopic: str, *, topic: str | None = None) -> str:
    topic = topic or english_topic(rng, subtopic)
    header = _pick(rng, ENGLISH_OPENINGS).format(topic=topic)
    return _pick(rng, ENGLISH_HEADERS).format(topic=topic) + header


def code_task(rng: random.Random, subtopic: str) -> str:
    if rng.random() < 0.2:
        return _pick(rng, CODE_TASKS[subtopic])
    return procedural_code_task(rng, subtopic)


def code_prefix(rng: random.Random, subtopic: str, lang: str) -> str:
    task = code_task(rng, subtopic)
    if lang in ("javascript", "typescript"):
        return _pick(rng, CODE_JS_HEADERS).format(task=task) + _pick(rng, CODE_JS_BODY)
    if lang == "sql":
        return _pick(rng, CODE_SQL_HEADERS).format(task=task) + _pick(rng, CODE_SQL_BODY)
    # Default to Python-like snippets for python and other langs.
    return _pick(rng, CODE_PYTHON_HEADERS).format(task=task) + _pick(rng, CODE_PYTHON_BODY)


def _math_numbers(rng: random.Random, subtopic: str) -> tuple[str, str]:
    if subtopic == "arithmetic":
        a, b = rng.randint(12, 999), rng.randint(12, 999)
        op = rng.choice(["+", "-", "*"])
        expr = f"{a} {op} {b}"
        return f"Compute {expr}.", expr
    if subtopic == "fractions_decimals":
        n, d = rng.randint(2, 12), rng.randint(3, 15)
        pct = rng.choice([10, 15, 20, 25, 40, 50])
        return f"What is {pct}% of {n * d}?", f"{pct}% of {n * d}"
    if subtopic == "algebra":
        x = rng.randint(2, 15)
        a = rng.randint(2, 9)
        b = rng.randint(5, 40)
        c = a * x + b
        return f"Solve for x: {a}x + {b} = {c}", f"x = {x}"
    if subtopic == "geometry":
        w, h = rng.randint(4, 20), rng.randint(4, 20)
        return f"A rectangle has width {w} cm and height {h} cm. Find its area.", f"{w * h} cm^2"
    # word_problem
    price = rng.randint(2, 15)
    count = rng.randint(3, 20)
    extra = rng.randint(1, 9)
    total = price * count + extra
    return (
        f"A shop sells notebooks for ${price} each. Maya buys {count} notebooks and also pays ${extra} for a folder. How much does she spend in total?",
        f"${total}",
    )


def math_prefix(rng: random.Random, subtopic: str) -> str:
    statement, _ = _math_numbers(rng, subtopic)
    return f"Problem:\n{statement}\n\nSolution:\n"


def multilingual_title(rng: random.Random, lang_code: str, lang_name: str) -> str:
    titles = ML_TITLES.get(lang_code)
    if titles and rng.random() < 0.2:
        return _pick(rng, titles)
    return procedural_ml_title(rng, lang_code, lang_name)


def multilingual_prefix(rng: random.Random, lang_code: str, lang_name: str = "") -> str:
    openings = ML_OPENINGS.get(lang_code, ML_GENERIC_OPENINGS)
    title = multilingual_title(rng, lang_code, lang_name or lang_code)
    opening = _pick(rng, openings)
    return f"{title}\n\n{opening}"


def prefix_metadata(domain: str, **extra: Any) -> dict[str, Any]:
    meta = {"domain": domain, "prompt_style": "continuation"}
    meta.update(extra)
    return meta
