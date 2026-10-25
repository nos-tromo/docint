"""Deterministic sampling and text preparation for a topic fit.

Pure: numpy and the standard library only. The sample is chosen by hashing
point ids with a seed, so the same collection and seed select the same points
regardless of scroll order.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from docint.utils.topics_client import encode_texts

TEXT_CHARS = 2000
TOP_N_WORDS = 10
MIN_DF = 2

_BUDGET_OVERHEAD = 6


def rank_key(seed: int, point_id: str) -> bytes:
    """Return the sort key that ranks a point for sampling.

    Args:
        seed: Sampling seed.
        point_id: Point id.

    Returns:
        The SHA-256 digest of ``seed:point_id``.
    """
    return hashlib.sha256(f"{seed}:{point_id}".encode()).digest()


def top_ranked(ids: Iterable[str], *, seed: int, limit: int) -> list[str]:
    """Select the ``limit`` ids with the smallest rank keys.

    Args:
        ids: Candidate point ids, in any order.
        seed: Sampling seed.
        limit: Most ids to return.

    Returns:
        The selected ids in rank order.
    """
    ranked = heapq.nsmallest(limit, ((rank_key(seed, i), i) for i in ids))
    return [i for _, i in ranked]


def fit_text(text: str) -> str:
    """Prepare a chunk's text for the fit.

    Args:
        text: Raw chunk text.

    Returns:
        The text without control characters, whitespace collapsed to single
        spaces, stripped and cut to ``TEXT_CHARS``.
    """
    kept = "".join(c for c in text if c.isspace() or unicodedata.category(c) != "Cc")
    return " ".join(kept.split())[:TEXT_CHARS]


@dataclass(frozen=True)
class Candidate:
    """A point offered to the fit sample.

    Attributes:
        point_id: Qdrant point id.
        vector: Dense vector.
        text: Chunk text.
    """

    point_id: str
    vector: np.ndarray
    text: str


@dataclass(frozen=True)
class FitSample:
    """The points a fit runs on.

    Attributes:
        points: Eligible, distinct candidates sorted by point id, text prepared.
        copies: Dropped duplicate id mapped to its representative's id.
    """

    points: tuple[Candidate, ...]
    copies: Mapping[str, str]

    @property
    def n(self) -> int:
        """Return the number of fitted points.

        Returns:
            The sample size.
        """
        return len(self.points)


def _vector_key(vector: np.ndarray) -> bytes:
    """Return the exact-duplicate key of a vector.

    Args:
        vector: Dense vector.

    Returns:
        Its float32 bytes, with negative zero folded into zero.
    """
    return (np.asarray(vector, dtype="<f4") + 0.0).tobytes()


def build_sample(candidates: Iterable[Candidate], *, min_text_chars: int) -> FitSample:
    """Filter, order and deduplicate candidates into a fit sample.

    Args:
        candidates: Candidate points.
        min_text_chars: Shortest prepared text that is eligible.

    Returns:
        The sample; the first point per distinct vector represents its copies.
    """
    eligible = []
    for cand in candidates:
        text = fit_text(cand.text)
        if len(text) >= min_text_chars:
            eligible.append(Candidate(cand.point_id, cand.vector, text))
    eligible.sort(key=lambda c: c.point_id)
    seen: dict[bytes, str] = {}
    points: list[Candidate] = []
    copies: dict[str, str] = {}
    for cand in eligible:
        key = _vector_key(cand.vector)
        rep = seen.get(key)
        if rep is None:
            seen[key] = cand.point_id
            points.append(cand)
        else:
            copies[cand.point_id] = rep
    return FitSample(points=tuple(points), copies=copies)


def _json_len(text: str) -> int:
    """Return a text's encoded length as a JSON string.

    Args:
        text: The text.

    Returns:
        Bytes of its UTF-8 JSON form, quotes included.
    """
    return len(json.dumps(text, ensure_ascii=False).encode())


def _cut(text: str, budget: int) -> str:
    """Cut a text to the longest prefix within a JSON byte budget.

    Args:
        text: The text.
        budget: Most JSON-encoded bytes.

    Returns:
        The longest fitting prefix.
    """
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _json_len(text[:mid]) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo]


def texts_within(texts: Sequence[str], limit_bytes: int) -> list[str]:
    """Shorten texts until their encoded part fits the service limit.

    Args:
        texts: Prepared texts.
        limit_bytes: Largest accepted ``texts`` part.

    Returns:
        The texts unchanged when they fit, else each cut to a shared budget.
    """
    if len(encode_texts(texts)) <= limit_bytes:
        return list(texts)
    budget = (limit_bytes - 2 - len(texts)) // len(texts) - _BUDGET_OVERHEAD
    return [_cut(t, budget) for t in texts]


def auto_min_topic_size(n: int) -> int:
    """Return the automatic minimum topic size for a sample.

    Args:
        n: Sample size.

    Returns:
        ``n // 1000`` clamped to 10..100.
    """
    return min(100, max(10, n // 1000))


def min_eligible(min_topic_size: int) -> int:
    """Return how many eligible points a build needs.

    Args:
        min_topic_size: Minimum topic size.

    Returns:
        Three times the size, at least 50.
    """
    return max(3 * min_topic_size, 50)


def nr_topics_for(max_topics: int) -> int:
    """Map the topic cap to the service's ``nr_topics``.

    Args:
        max_topics: Configured cap, 0 or less for none.

    Returns:
        0 for no reduction, else the cap plus the outlier class.
    """
    return 0 if max_topics <= 0 else max_topics + 1
