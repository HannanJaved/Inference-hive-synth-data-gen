"""Mix configuration and language lists for Datamix-style synthetic generation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# HPLT 2.0-style EU + partner languages (36). Edit mix_config.yaml to override.
DEFAULT_HPLT_LANGUAGES: dict[str, str] = {
    "bg": "Bulgarian",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "es": "Spanish",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "ga": "Irish",
    "hr": "Croatian",
    "hu": "Hungarian",
    "is": "Icelandic",
    "it": "Italian",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mt": "Maltese",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sq": "Albanian",
    "sr": "Serbian",
    "sv": "Swedish",
    "uk": "Ukrainian",
    "ca": "Catalan",
    "eu": "Basque",
    "gl": "Galician",
    "mk": "Macedonian",
    "bs": "Bosnian",
    "be": "Belarusian",
    "cy": "Welsh",
    "lb": "Luxembourgish",
    "rm": "Romansh",
}


@dataclass
class DomainBudget:
    domain: str
    token_target: int
    avg_completion_tokens: int
    n_prompts: int


@dataclass
class MixConfig:
    total_tokens: int = 1_000_000_000
    # Datamix pretraining mix (must sum to 1.0).
    english_fraction: float = 0.72
    code_fraction: float = 0.072
    math_fraction: float = 0.104
    multilingual_fraction: float = 0.104

    seed: int = 42
    # Prepended to every prompt id (e.g. "topup-" for supplemental runs).
    id_prefix: str = ""
    max_seq_len: int = 2048
    # Tokens reserved for the user meta-prompt; completion must fit in max_seq_len - this.
    max_prompt_tokens_reserve: int = 512
    max_completion_tokens: int = 1536

    avg_tokens: dict[str, int] = field(default_factory=lambda: {
        "english": 900,
        "code": 700,
        "math": 600,
        "multilingual": 800,
    })

    languages: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_HPLT_LANGUAGES))

    # Generator + tokenizer: openeurollm/datamix-9b-80-20 (HF hub cache snapshot).
    generator_model_path: str = (
        "/data/horse/ws/hama901h-whittle/.cache/huggingface/hub/"
        "models--openeurollm--datamix-9b-80-20/snapshots/"
        "66c3229e708857a0bc54070131d4b1762d4f5279"
    )
    tokenizer_name_or_path: str = (
        "/data/horse/ws/hama901h-whittle/.cache/huggingface/hub/"
        "models--openeurollm--datamix-9b-80-20/snapshots/"
        "66c3229e708857a0bc54070131d4b1762d4f5279"
    )

    # Deduplication (prompt build + postprocess QC).
    dedup_prompts: bool = True
    dedup_outputs: bool = True
    # Prompt near-dedup folds topic/title slots before comparing templates.
    dedup_prompt_near_threshold: float = 0.85
    dedup_output_near_threshold: float = 0.78
    dedup_shingle_size: int = 5
    dedup_prompt_max_resamples: int = 150

    # DCLM CORE benchmark decontamination (13-gram word overlap, GPT-3 style).
    decontam_enabled: bool = True
    decontam_ngram_size: int = 13
    decontam_benchmark_index: str | None = None

    # Paths (resolved relative to SYNTH_ROOT).
    root: Path = field(default_factory=lambda: Path.cwd())
    prompts_dir: Path | None = None
    dataset_dir: Path | None = None
    outputs_dir: Path | None = None
    corpus_dir: Path | None = None

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        self.prompts_dir = self.prompts_dir or self.root / "prompts"
        self.dataset_dir = self.dataset_dir or self.root / "dataset"
        self.outputs_dir = self.outputs_dir or self.root / "ih_outputs"
        self.corpus_dir = self.corpus_dir or self.root / "corpus"

    @property
    def benchmark_index_path(self) -> Path:
        if self.decontam_benchmark_index:
            p = Path(self.decontam_benchmark_index)
            return p if p.is_absolute() else self.root / p
        return self.root / "benchmarks" / f"dclm_core_{self.decontam_ngram_size}gram_index.pkl"

    @property
    def domain_fractions(self) -> dict[str, float]:
        return {
            "english": self.english_fraction,
            "code": self.code_fraction,
            "math": self.math_fraction,
            "multilingual": self.multilingual_fraction,
        }

    def validate(self) -> None:
        total = sum(self.domain_fractions.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"domain fractions must sum to 1.0, got {total:.6f}")
        if self.total_tokens <= 0:
            raise ValueError("total_tokens must be positive")
        if self.max_completion_tokens > self.max_seq_len - self.max_prompt_tokens_reserve:
            raise ValueError(
                "max_completion_tokens exceeds max_seq_len - max_prompt_tokens_reserve"
            )


def load_mix_config(path: str | Path) -> MixConfig:
    data: dict[str, Any] = yaml.safe_load(Path(path).read_text())
    langs = data.pop("languages", None)
    cfg = MixConfig(**{k: v for k, v in data.items() if k != "paths"})
    if "paths" in data:
        for key, val in data["paths"].items():
            if hasattr(cfg, key):
                setattr(cfg, key, Path(val))
    if langs:
        # YAML 1.1 treats bare `no`/`yes` as booleans; normalize keys to strings.
        normalized: dict[str, str] = {}
        for k, v in langs.items():
            if k is False:
                k = "no"
            elif k is True:
                k = "yes"
            normalized[str(k)] = str(v)
        cfg.languages = normalized
    cfg.validate()
    return cfg
