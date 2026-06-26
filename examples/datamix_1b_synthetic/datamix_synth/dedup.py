"""Exact and near-duplicate detection for prompts and generated text."""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def normalize_prompt_for_dedup(text: str) -> str:
    """Fold topic/title slots so shared prefix templates collide under dedup."""
    chunks = re.split(r"\n\s*\n", text.strip(), maxsplit=1)
    if len(chunks) == 2 and len(chunks[1].strip()) >= 40:
        norm = normalize_text(chunks[1])
    else:
        norm = normalize_text(text)

    replacements = [
        (r"^[^.\n]{3,80}\s+is an important subject", "<topic> is an important subject"),
        (r"return to [^.\n]{3,80}\s+because it connects", "return to <topic> because it connects"),
        (r"first encounter [^.\n]{3,80}\s+in school", "first encounter <topic> in school"),
        (r"understanding [^.\n]{3,80}\s+has shaped", "understanding <topic> has shaped"),
        (r"appreciate [^.\n]{3,80},\s+it helps", "appreciate <topic>, it helps"),
        (r"documented [^.\n]{3,80}\s+from several angles", "documented <topic> from several angles"),
        (r"problem:\s*[^.\n]{3,120}\n\nsolution:", "problem: <topic>\n\nsolution:"),
        (r"^[^.\n]{8,120}\n\n(?:schon seit|depuis longtemps|desde hace|od wielu lat)",
         "<title>\n\n<opening>"),
    ]
    for pat, repl in replacements:
        norm = re.sub(pat, repl, norm, count=1, flags=re.IGNORECASE)
    return norm


def char_shingles(text: str, n: int = 5) -> frozenset[str]:
    norm = normalize_text(text)
    if not norm:
        return frozenset()
    if len(norm) <= n:
        return frozenset([norm])
    return frozenset(norm[i : i + n] for i in range(len(norm) - n + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def minhash_signature(shingles: frozenset[str], num_perm: int = 128) -> tuple[int, ...]:
    if not shingles:
        return tuple(0 for _ in range(num_perm))
    mins = [2**64 - 1] * num_perm
    for perm in range(num_perm):
        salt = perm.to_bytes(4, "little", signed=False)
        for shingle in shingles:
            digest = hashlib.blake2b(shingle.encode("utf-8"), digest_size=8, person=salt).digest()
            h = int.from_bytes(digest, "little")
            if h < mins[perm]:
                mins[perm] = h
    return tuple(mins)


@dataclass
class NearDupIndex:
    """LSH MinHash index for streaming near-duplicate detection."""

    threshold: float = 0.85
    shingle_size: int = 5
    num_perm: int = 128
    band_size: int = 8
    _entries: list[tuple[tuple[int, ...], str]] = field(default_factory=list)
    _buckets: dict[tuple[int, tuple[int, ...]], list[int]] = field(default_factory=dict)

    def _bands(self, sig: tuple[int, ...]) -> list[tuple[int, tuple[int, ...]]]:
        keys: list[tuple[int, tuple[int, ...]]] = []
        for band_idx in range(self.num_perm // self.band_size):
            start = band_idx * self.band_size
            keys.append((band_idx, sig[start : start + self.band_size]))
        return keys

    def find_near_duplicate(self, text: str) -> bool:
        norm = text
        shingles = char_shingles(norm, self.shingle_size)
        if not shingles:
            return False
        sig = minhash_signature(shingles, self.num_perm)
        candidate_idxs: set[int] = set()
        for band_key in self._bands(sig):
            candidate_idxs.update(self._buckets.get(band_key, []))

        for idx in candidate_idxs:
            other_sig, other_norm = self._entries[idx]
            est = sum(a == b for a, b in zip(sig, other_sig)) / self.num_perm
            if est < self.threshold:
                continue
            other_shingles = char_shingles(other_norm, self.shingle_size)
            if jaccard(shingles, other_shingles) >= self.threshold:
                return True
        return False

    def add(self, text: str) -> None:
        norm = text
        shingles = char_shingles(norm, self.shingle_size)
        if not shingles:
            return
        sig = minhash_signature(shingles, self.num_perm)
        idx = len(self._entries)
        self._entries.append((sig, norm))
        for band_key in self._bands(sig):
            self._buckets.setdefault(band_key, []).append(idx)


@dataclass
class DedupFilter:
    """Reject exact or near duplicates within a scope (e.g. per domain)."""

    near_threshold: float = 0.85
    shingle_size: int = 5
    exact: bool = True
    near: bool = True
    normalize_fn: Callable[[str], str] | None = None
    stats: Counter = field(default_factory=Counter)
    _raw_seen: set[str] = field(default_factory=set)
    _key_seen: set[str] = field(default_factory=set)
    _near_index: NearDupIndex | None = None

    def __post_init__(self) -> None:
        if self.near:
            self._near_index = NearDupIndex(
                threshold=self.near_threshold,
                shingle_size=self.shingle_size,
            )

    def _dedup_key(self, text: str) -> str:
        if self.normalize_fn is not None:
            return self.normalize_fn(text)
        return normalize_text(text)

    def is_duplicate(self, text: str) -> tuple[bool, str | None]:
        raw = normalize_text(text)
        if not raw:
            self.stats["empty"] += 1
            return True, "empty"

        if self.exact and raw in self._raw_seen:
            self.stats["exact_duplicate"] += 1
            return True, "exact_duplicate"

        key = self._dedup_key(text)
        if not key:
            self.stats["empty"] += 1
            return True, "empty"

        if self.exact and key in self._key_seen:
            self.stats["template_duplicate"] += 1
            return True, "template_duplicate"

        if self.near and self._near_index is not None and self._near_index.find_near_duplicate(key):
            self.stats["near_duplicate"] += 1
            return True, "near_duplicate"

        return False, None

    def register(self, text: str) -> None:
        raw = normalize_text(text)
        if not raw:
            return
        self._raw_seen.add(raw)
        key = self._dedup_key(text)
        if not key:
            return
        self._key_seen.add(key)
        if self.near and self._near_index is not None:
            self._near_index.add(key)

    def try_add(self, text: str) -> tuple[bool, str | None]:
        is_dup, reason = self.is_duplicate(text)
        if is_dup:
            return False, reason
        self.register(text)
        self.stats["accepted"] += 1
        return True, None
