"""Deterministic sampling and text preparation for a topic fit.

Pure: numpy and the standard library only. The sample is chosen by hashing
point ids with a seed, so the same collection and seed select the same points
regardless of scroll order.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from docint.utils.topics_client import SERVICE_MIN_POINTS, encode_texts

TEXT_CHARS = 2000
TEXT_CHARS_MAX = 8000
TEXT_BUDGET_CHARS = 200_000_000
TOP_N_WORDS = 10
MIN_DF = 2
MAX_DF = 0.8

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


def fit_text(text: str, limit: int = TEXT_CHARS) -> str:
    """Prepare a chunk's text for the fit.

    Args:
        text: Raw chunk text.
        limit: Most characters kept.

    Returns:
        The text without control characters, whitespace collapsed to single
        spaces, stripped and cut to ``limit``.
    """
    kept = "".join(c for c in text if c.isspace() or unicodedata.category(c) != "Cc")
    return " ".join(kept.split())[:limit].rstrip()


def text_chars_for(points: int) -> int:
    """Return how many characters of each text a fit of ``points`` chunks keeps.

    A small or medium fit keeps whole chunks; a large one shares a fixed total,
    so the largest fit sends no more text than a 2,000-character cut always did.

    Args:
        points: Chunks ranked into the sample.

    Returns:
        ``TEXT_BUDGET_CHARS // points``, between ``TEXT_CHARS`` and ``TEXT_CHARS_MAX``.
    """
    return max(TEXT_CHARS, min(TEXT_CHARS_MAX, TEXT_BUDGET_CHARS // max(1, points)))


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


def build_sample(candidates: Iterable[Candidate], *, min_text_chars: int, text_chars: int = TEXT_CHARS) -> FitSample:
    """Filter, order and deduplicate candidates into a fit sample.

    Args:
        candidates: Candidate points.
        min_text_chars: Shortest prepared text that is eligible.
        text_chars: Most characters of each text kept.

    Returns:
        The sample; the first point per distinct vector represents its copies.
    """
    eligible = []
    for cand in candidates:
        text = fit_text(cand.text, text_chars)
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

    A small sample gets small topics, so a few documents still split into
    several: ``round(sqrt(n) / 3)``, at least 2 and at most 10. From 813
    chunks on the size is ``n // 1000``, between 10 and 100.

    Args:
        n: Sample size.

    Returns:
        The size, from 2 to 100.
    """
    return min(100, max(2, min(10, round(math.sqrt(n) / 3)), n // 1000))


def min_eligible(min_topic_size: int) -> int:
    """Return how many eligible points a fit needs.

    Args:
        min_topic_size: Minimum topic size.

    Returns:
        Twice the size, at least ``SERVICE_MIN_POINTS``: the service's own
        minimum, so a sample it would refuse is refused before it is sent.
    """
    return max(SERVICE_MIN_POINTS, 2 * min_topic_size)


def nr_topics_for(max_topics: int) -> int:
    """Map the topic cap to the service's ``nr_topics``.

    Args:
        max_topics: Configured cap, 0 or less for none.

    Returns:
        0 for no reduction, else the cap plus the outlier class.
    """
    return 0 if max_topics <= 0 else max_topics + 1
