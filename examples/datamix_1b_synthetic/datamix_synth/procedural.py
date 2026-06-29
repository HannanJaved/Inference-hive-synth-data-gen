"""Procedural topic/title/task generation for large-scale prompt diversity."""
from __future__ import annotations

import random

# Shared word banks (combinatorial space >> millions).
_ADJECTIVES = [
    "ancient", "modern", "coastal", "urban", "rural", "tropical", "arctic", "volcanic",
    "renewable", "digital", "organic", "industrial", "colonial", "medieval", "classical",
    "experimental", "theoretical", "practical", "seasonal", "nocturnal", "microscopic",
    "planetary", "cellular", "genetic", "thermal", "magnetic", "acoustic", "optical",
    "hydraulic", "mechanical", "electrical", "chemical", "biological", "geological",
    "atmospheric", "marine", "freshwater", "alpine", "desert", "forest", "grassland",
    "suburban", "metropolitan", "regional", "national", "international", "interstellar",
]

_NOUNS = [
    "migration", "irrigation", "fermentation", "navigation", "architecture", "sculpture",
    "literacy", "diplomacy", "commerce", "metallurgy", "cartography", "astronomy",
    "ecology", "geography", "linguistics", "pedagogy", "nutrition", "immunology",
    "hydrology", "seismology", "ornithology", "botany", "zoology", "paleontology",
    "cryptography", "photography", "typography", "chronology", "genealogy", "mythology",
    "democracy", "federalism", "citizenship", "sovereignty", "jurisdiction",
    "infrastructure", "manufacturing", "agriculture", "horticulture", "viticulture",
    "beekeeping", "woodworking", "blacksmithing", "weaving", "pottery", "glassblowing",
]

_FIELDS = [
    "public health", "urban planning", "environmental policy", "marine biology",
    "renewable energy", "materials science", "cognitive psychology", "microeconomics",
    "macroeconomics", "international trade", "civil engineering", "software engineering",
    "data science", "library science", "museum studies", "archival research",
    "wildlife conservation", "watershed management", "disaster preparedness",
    "food security", "water sanitation", "rural development", "transport planning",
]

_PLACES = [
    "the Baltic coast", "the Andean highlands", "the Nile delta", "the Great Lakes",
    "the Iberian peninsula", "the Horn of Africa", "the Mekong basin", "the Caucasus",
    "the Caribbean", "the Pacific northwest", "the Sahel", "the Levant",
    "Scandinavia", "Patagonia", "Southeast Asia", "Central Europe", "West Africa",
    "the Appalachian range", "the Ganges plain", "the Australian outback",
]

_PEOPLE_ROLES = [
    "a village midwife", "a harbor pilot", "a court astronomer", "a monastery scribe",
    "a railway engineer", "a lighthouse keeper", "a market gardener", "a shipwright",
    "a school principal", "a museum curator", "a field geologist", "a choir director",
    "a textile merchant", "a map engraver", "a bridge inspector", "a park ranger",
]

_VERBS = [
    "Analyze", "Validate", "Transform", "Aggregate", "Normalize", "Serialize",
    "Deserialize", "Compress", "Decompress", "Index", "Filter", "Merge", "Split",
    "Encode", "Decode", "Hash", "Cache", "Batch", "Stream", "Paginate", "Retry",
    "Schedule", "Monitor", "Profile", "Benchmark", "Migrate", "Replicate", "Archive",
]

_CODE_OBJECTS = [
    "JSONL records", "CSV columns", "log lines", "user sessions", "sensor readings",
    "invoice rows", "geolocation points", "time-series samples", "API responses",
    "configuration files", "audit events", "search queries", "image metadata",
    "payment transactions", "inventory SKUs", "email headers", "webhook payloads",
]

_CODE_PURPOSES = [
    "a nightly ETL job", "a CLI reporting tool", "a data quality pipeline",
    "a backup verification script", "a metrics aggregation service",
    "an integration test harness", "a schema migration utility",
    "a batch export workflow", "a real-time alert processor",
]

_ML_THEMES = [
    "traditional crafts", "coastal fishing", "mountain railways", "river commerce",
    "harvest festivals", "monastic libraries", "market squares", "vineyard terraces",
    "stone bridges", "woodland paths", "island ferries", "border markets",
    "folk music", "ceramic workshops", "shepherd routes", "canal locks",
]

_ML_REGIONS = [
    "the northern provinces", "the southern coast", "the central highlands",
    "the eastern borderlands", "the western valleys", "the capital district",
    "the old town quarter", "the industrial belt", "the lake district",
]


def _pick(rng: random.Random, items: list[str]) -> str:
    return items[rng.randrange(len(items))]


def procedural_english_topic(rng: random.Random, subtopic: str) -> str:
    """Return a high-entropy English topic string for the given subtopic."""
    adj, noun, field, place = _pick(rng, _ADJECTIVES), _pick(rng, _NOUNS), _pick(rng, _FIELDS), _pick(rng, _PLACES)
    year = rng.randint(1400, 2024)
    pct = rng.randint(3, 97)

    patterns: dict[str, list[str]] = {
        "science_explainer": [
            f"{adj.capitalize()} {noun} in {field}",
            f"how {noun} shapes {field}",
            f"the chemistry of {adj} {noun}",
            f"{noun} observed in {place}",
            f"phase transitions in {adj} materials",
            f"field studies of {noun} since {year}",
        ],
        "history_culture": [
            f"{adj.capitalize()} {noun} in {place}",
            f"daily life and {noun} around {year}",
            f"trade routes linking {place} and {_pick(rng, _PLACES)}",
            f"{_pick(rng, _PEOPLE_ROLES)} and {noun}",
            f"archaeological evidence of {adj} {noun}",
            f"oral histories of {noun} in {place}",
        ],
        "technology_overview": [
            f"{adj.capitalize()} systems for {noun}",
            f"how engineers approach {noun} in {field}",
            f"benchmarking {noun} pipelines",
            f"open standards for {adj} {noun}",
            f"failure modes in {noun} deployments",
            f"scaling {noun} across {pct} nodes",
        ],
        "how_to_guide": [
            f"How to maintain {adj} {noun} at home",
            f"How to document {noun} for beginners",
            f"How to troubleshoot {noun} in {field}",
            f"How to plan a {adj} project involving {noun}",
            f"How to teach {noun} to students aged {rng.randint(8, 18)}",
            f"How to budget for {noun} improvements",
        ],
        "biography_profile": [
            f"{_pick(rng, _PEOPLE_ROLES).capitalize()} who advanced {noun}",
            f"early career work on {adj} {noun}",
            f"letters describing {noun} in {place}, {year}",
            f"a life shaped by {field} and {noun}",
            f"community leadership through {noun}",
            f"innovations in {adj} {noun} after {year}",
        ],
        "health_wellness": [
            f"{adj.capitalize()} habits supporting {noun}",
            f"community programs for {noun} in {field}",
            f"evidence on {noun} and daily routines",
            f"seasonal changes in {noun} for adults over {rng.randint(25, 75)}",
            f"practical guidance on {noun} at work",
            f"hydration, sleep, and {noun}",
        ],
        "economics_society": [
            f"{adj.capitalize()} markets for {noun} in {place}",
            f"policy debates on {noun} since {year}",
            f"household spending on {noun}",
            f"cooperatives managing {noun} in {field}",
            f"tax incentives for {adj} {noun}",
            f"labor trends in {noun} ({pct}% growth)",
        ],
        "nature_environment": [
            f"{adj.capitalize()} habitats supporting {noun}",
            f"restoration of {noun} near {place}",
            f"migratory patterns and {noun}",
            f"soil chemistry affecting {noun}",
            f"wetland {noun} after {year} reforms",
            f"citizen science tracking {adj} {noun}",
        ],
    }
    pool = patterns.get(subtopic, patterns["science_explainer"])
    return _pick(rng, pool)


def procedural_code_task(rng: random.Random, subtopic: str) -> str:
    verb, obj, purpose = _pick(rng, _VERBS), _pick(rng, _CODE_OBJECTS), _pick(rng, _CODE_PURPOSES)
    limit = rng.randint(10, 5000)
    key = f"field_{rng.randint(1, 99)}"

    patterns: dict[str, list[str]] = {
        "python_script": [
            f"{verb} {obj} for {purpose}",
            f"{verb} {obj} with a limit of {limit}",
            f"{verb} {obj} keyed by {key}",
        ],
        "data_processing": [
            f"{verb} and deduplicate {obj}",
            f"{verb} {obj} before loading a warehouse",
            f"Clean {obj} missing column {key}",
        ],
        "algorithms": [
            f"{verb} sorted {obj} in O(n log n)",
            f"{verb} {obj} using a sliding window of {rng.randint(3, 64)}",
            f"{verb} graph edges derived from {obj}",
        ],
        "web_api": [
            f"{verb} paginated {obj} from a REST API",
            f"{verb} {obj} with exponential backoff",
            f"{verb} {obj} and handle HTTP 429 responses",
        ],
        "testing": [
            f"{verb} unit tests for {obj}",
            f"{verb} fixtures covering {obj}",
            f"{verb} regression cases for {key} in {obj}",
        ],
        "cli_tool": [
            f"{verb} {obj} from a CLI with argparse",
            f"{verb} {obj} and print a summary table",
            f"{verb} {obj} for {purpose}",
        ],
    }
    return _pick(rng, patterns.get(subtopic, patterns["python_script"]))


def procedural_ml_title(rng: random.Random, lang_code: str, lang_name: str) -> str:
    theme, region = _pick(rng, _ML_THEMES), _pick(rng, _ML_REGIONS)
    year = rng.randint(1850, 2020)
    templates = [
        f"{theme.capitalize()} in {region}",
        f"{lang_name} perspectives on {theme}",
        f"local history of {theme} ({lang_code})",
        f"{theme} and community life since {year}",
        f"walking tours through {region} focusing on {theme}",
        f"archives documenting {theme} in {lang_name} towns",
    ]
    return _pick(rng, templates)
