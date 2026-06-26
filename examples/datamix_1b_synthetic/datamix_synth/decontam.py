"""Benchmark n-gram decontamination (GPT-3 / lm-eval style word n-grams)."""
from __future__ import annotations

import pickle
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


_TRANSLATION = str.maketrans(
    string.ascii_lowercase + string.ascii_uppercase,
    string.ascii_lowercase * 2,
    string.punctuation,
)


def normalize_for_decontam(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace (janitor-compatible)."""
    return " ".join(text.translate(_TRANSLATION).split())


def iter_word_ngrams(text: str, n: int = 13) -> Iterator[str]:
    tokens = normalize_for_decontam(text).split()
    if not tokens:
        return
    if len(tokens) < n:
        yield " ".join(tokens)
        return
    for i in range(len(tokens) - n + 1):
        yield " ".join(tokens[i : i + n])


def build_ngram_index(texts: list[str], ngram_size: int = 13) -> set[str]:
    index: set[str] = set()
    for text in texts:
        for ng in iter_word_ngrams(text, ngram_size):
            index.add(ng)
    return index


@dataclass
class BenchmarkNgramIndex:
    ngram_size: int
    ngrams: set[str]
    metadata: dict

    def is_contaminated(self, text: str) -> bool:
        return any(ng in self.ngrams for ng in iter_word_ngrams(text, self.ngram_size))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ngram_size": self.ngram_size,
            "ngrams": self.ngrams,
            "metadata": self.metadata,
        }
        with path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> BenchmarkNgramIndex:
        with path.open("rb") as f:
            payload = pickle.load(f)
        return cls(
            ngram_size=int(payload["ngram_size"]),
            ngrams=set(payload["ngrams"]),
            metadata=dict(payload.get("metadata") or {}),
        )


def load_benchmark_index(cfg) -> BenchmarkNgramIndex:
    path = cfg.benchmark_index_path
    if not path.is_file():
        raise FileNotFoundError(
            f"Benchmark n-gram index not found at {path}. "
            "Run ./run_pipeline.sh prepare-benchmarks first."
        )
    return BenchmarkNgramIndex.load(path)
