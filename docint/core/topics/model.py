"""The topic table: records, centres, admission floors and alignment.

Pure: numpy and the standard library only. Centres are unit rows stored at
float32 precision, so every alignment sees the values a saved run reloads.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import Any, Literal

import numpy as np

Origin = Literal["build", "discover"]
_ALIGN_BATCH_ROWS = 256


@dataclass(frozen=True)
class TopicInfo:
    """One topic's record.

    Attributes:
        id: Topic id, unique within a run.
        label: Short display label.
        description: One-line description.
        keywords: ``(word, score)`` pairs.
        origin: Whether the build or a later discovery made it.
        added_at: ISO timestamp the topic was added.
        fit_size: Members in the fit that produced it.
        floor: Admission floor on cosine to the centre.
        first_seen: Earliest member timestamp, ISO UTC.
        last_seen: Latest member timestamp, ISO UTC.
    """

    id: int
    label: str
    description: str
    keywords: tuple[tuple[str, float], ...]
    origin: Origin
    added_at: str
    fit_size: int
    floor: float
    first_seen: str | None = None
    last_seen: str | None = None

    def to_json(self) -> dict[str, Any]:
        """Serialise the record.

        Returns:
            A JSON-ready dict.
        """
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "keywords": [[w, s] for w, s in self.keywords],
            "origin": self.origin,
            "added_at": self.added_at,
            "fit_size": self.fit_size,
            "floor": self.floor,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> TopicInfo:
        """Rebuild a record from :meth:`to_json` output.

        Args:
            data: The serialised record.

        Returns:
            The record, keywords as tuples.
        """
        return cls(
            id=int(data["id"]),
            label=str(data["label"]),
            description=str(data["description"]),
            keywords=tuple((str(w), float(s)) for w, s in data["keywords"]),
            origin=data["origin"],
            added_at=str(data["added_at"]),
            fit_size=int(data["fit_size"]),
            floor=float(data["floor"]),
            first_seen=data.get("first_seen"),
            last_seen=data.get("last_seen"),
        )


def unit_rows(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise rows.

    Args:
        vectors: ``(n, dim)`` matrix.

    Returns:
        A float64 matrix of unit rows; an all-zero row stays zero.
    """
    arr = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norms == 0.0, 1.0, norms)


def _stored(rows: np.ndarray) -> np.ndarray:
    """Round rows through float32.

    Args:
        rows: Float matrix.

    Returns:
        A float64 matrix holding float32-representable values.
    """
    return np.asarray(rows, dtype=np.float64).astype(np.float32).astype(np.float64)


@dataclass(frozen=True, eq=False)
class TopicTable:
    """Topics with their centres, in ascending id order.

    Attributes:
        topics: Records with ascending unique ids.
        centres: ``(len(topics), dim)`` float64 unit rows, float32-representable.
    """

    topics: tuple[TopicInfo, ...]
    centres: np.ndarray

    def __post_init__(self) -> None:
        """Validate ordering and shape.

        Raises:
            ValueError: On unsorted or repeated ids, non-2-D centres, or a row count mismatch.
        """
        ids = [t.id for t in self.topics]
        if any(b <= a for a, b in pairwise(ids)):
            raise ValueError("topic ids must be ascending and unique")
        if self.centres.ndim != 2:
            raise ValueError("centres must be 2-D")
        if self.centres.shape[0] != len(self.topics):
            raise ValueError("centres must have one row per topic")

    @property
    def ids(self) -> np.ndarray:
        """Return the topic ids.

        Returns:
            An int64 array.
        """
        return np.array([t.id for t in self.topics], dtype=np.int64)

    @property
    def floors(self) -> np.ndarray:
        """Return the admission floors.

        Returns:
            A float64 array.
        """
        return np.array([t.floor for t in self.topics], dtype=np.float64)

    @property
    def dim(self) -> int:
        """Return the centre dimension.

        Returns:
            Columns of ``centres``.
        """
        return int(self.centres.shape[1])

    def _row(self, topic_id: int) -> int | None:
        """Find a topic's row.

        Args:
            topic_id: Topic id.

        Returns:
            The row index, or None.
        """
        for i, t in enumerate(self.topics):
            if t.id == topic_id:
                return i
        return None

    def topic(self, topic_id: int) -> TopicInfo | None:
        """Look a topic up.

        Args:
            topic_id: Topic id.

        Returns:
            The record, or None.
        """
        row = self._row(topic_id)
        return None if row is None else self.topics[row]

    def centre(self, topic_id: int) -> np.ndarray | None:
        """Look a centre up.

        Args:
            topic_id: Topic id.

        Returns:
            The centre row, or None.
        """
        row = self._row(topic_id)
        return None if row is None else self.centres[row]

    def align(self, vectors: np.ndarray) -> np.ndarray:
        """Assign each vector to its nearest topic when it clears that topic's floor.

        Args:
            vectors: ``(n, dim)`` matrix.

        Returns:
            An int64 array of topic ids, ``-1`` where unassigned.
        """
        n = int(np.asarray(vectors).shape[0])
        if not self.topics:
            return np.full(n, -1, dtype=np.int64)
        scores = unit_rows(vectors) @ self.centres.T
        best = scores.argmax(axis=1)
        rows = np.arange(n)
        admit = scores[rows, best] >= self.floors[best]
        return np.where(admit, self.ids[best], -1).astype(np.int64)

    def appended(self, topics: Sequence[TopicInfo], centres: np.ndarray) -> TopicTable:
        """Append topics with higher ids.

        Args:
            topics: New records.
            centres: Their centre rows.

        Returns:
            A new table; existing entries are untouched.
        """
        new = np.asarray(centres, dtype=np.float64).reshape(len(topics), -1)
        merged = np.vstack([self.centres, new]) if self.topics else new
        return TopicTable(topics=self.topics + tuple(topics), centres=merged)

    def with_seen(self, seen: Mapping[int, tuple[str, str]]) -> TopicTable:
        """Widen topics' first and last seen times.

        Args:
            seen: Topic id to ``(min, max)`` ISO-8601 UTC strings.

        Returns:
            A new table whose ranges only ever grow.
        """
        out: list[TopicInfo] = []
        for t in self.topics:
            rng = seen.get(t.id)
            if rng is None:
                out.append(t)
                continue
            lo, hi = rng
            first = lo if t.first_seen is None or lo < t.first_seen else t.first_seen
            last = hi if t.last_seen is None or hi > t.last_seen else t.last_seen
            out.append(replace(t, first_seen=first, last_seen=last))
        return TopicTable(topics=tuple(out), centres=self.centres)


def centres_for(vectors: np.ndarray, labels: np.ndarray, topic_ids: Sequence[int]) -> np.ndarray:
    """Compute topic centres.

    Args:
        vectors: ``(n, dim)`` matrix.
        labels: Topic id per row.
        topic_ids: Topics to compute, in output order.

    Returns:
        Unit means of each topic's members, at float32 precision.
    """
    units = unit_rows(vectors)
    dim = units.shape[1]
    out = np.zeros((len(topic_ids), dim), dtype=np.float64)
    for i, tid in enumerate(topic_ids):
        members = units[np.asarray(labels) == tid]
        if members.shape[0]:
            out[i] = unit_rows(members.mean(axis=0, keepdims=True))[0]
    return _stored(out)


def floors_for(
    vectors: np.ndarray,
    labels: np.ndarray,
    topic_ids: Sequence[int],
    centres: np.ndarray,
    percentile: float,
) -> np.ndarray:
    """Compute admission floors.

    Args:
        vectors: ``(n, dim)`` matrix.
        labels: Topic id per row.
        topic_ids: Topics to compute, matching ``centres`` rows.
        centres: Unit centres.
        percentile: Percentile of member cosines to use.

    Returns:
        One floor per topic.

    Raises:
        ValueError: If a topic has no members.
    """
    units = unit_rows(vectors)
    out = np.zeros(len(topic_ids), dtype=np.float64)
    for i, tid in enumerate(topic_ids):
        members = units[np.asarray(labels) == tid]
        if not members.shape[0]:
            raise ValueError("topic has no members")
        out[i] = float(np.percentile(members @ centres[i], percentile))
    return out


def kept_members(vectors: np.ndarray, labels: np.ndarray, existing: np.ndarray, *, min_topic_size: int) -> np.ndarray:
    """Keep only the members nearest their own topic.

    Each topic's provisional centre is the unit mean of all its members. A
    member stays only when its own topic's provisional centre is its argmax over
    the existing centres followed by the provisional ones, scored as
    :meth:`TopicTable.align` scores. A tie goes to the lowest row, so a tie with
    an existing centre sets the member aside. A topic left with fewer members
    than ``min_topic_size`` loses them all.

    Args:
        vectors: ``(n, dim)`` matrix.
        labels: Topic id per row, ``-1`` for a row in none.
        existing: ``(m, dim)`` unit centres the members are weighed against.
        min_topic_size: Fewest members a topic may keep.

    Returns:
        A new int64 array: each member kept keeps its topic, every other row is ``-1``.
    """
    given = np.asarray(labels, dtype=np.int64)
    out = given.copy()
    topic_ids = np.unique(given[given >= 0])
    if not topic_ids.size:
        return out
    provisional = centres_for(vectors, given, topic_ids.tolist())
    centres = np.vstack([np.asarray(existing, dtype=np.float64).reshape(-1, vectors.shape[1]), provisional])
    # The row of each member's own provisional centre.
    own = centres.shape[0] - topic_ids.size + np.searchsorted(topic_ids, given)
    # Bound the score matrix even when many topics meet a large sample.
    for start in range(0, given.size, _ALIGN_BATCH_ROWS):
        stop = start + _ALIGN_BATCH_ROWS
        nearest = (unit_rows(vectors[start:stop]) @ centres.T).argmax(axis=1)
        page = out[start:stop]
        page[(given[start:stop] >= 0) & (nearest != own[start:stop])] = -1
    for tid in topic_ids:
        members = out == tid
        if np.count_nonzero(members) < min_topic_size:
            out[members] = -1
    return out


def central_members(
    vectors: np.ndarray, labels: np.ndarray, topic_id: int, centre: np.ndarray, limit: int
) -> list[int]:
    """List a topic's members nearest its centre.

    Args:
        vectors: ``(n, dim)`` matrix.
        labels: Topic id per row.
        topic_id: Topic to read.
        centre: Its centre.
        limit: Most rows to return.

    Returns:
        Row indices by descending cosine, ties by ascending index.
    """
    rows = np.flatnonzero(np.asarray(labels) == topic_id)
    if not rows.size:
        return []
    scores = unit_rows(vectors[rows]) @ np.asarray(centre, dtype=np.float64)
    order = np.lexsort((rows, -scores))
    return [int(rows[i]) for i in order[:limit]]


def novel(existing: np.ndarray, candidates: np.ndarray, *, min_similarity: float) -> np.ndarray:
    """Flag candidates that resemble no existing centre.

    Args:
        existing: ``(m, dim)`` centres.
        candidates: ``(k, dim)`` candidate centres.
        min_similarity: Cosine at or above which a candidate is a repeat.

    Returns:
        A bool per candidate, true where its best cosine is below the bar.
    """
    k = int(np.asarray(candidates).shape[0])
    if not np.asarray(existing).shape[0]:
        return np.ones(k, dtype=bool)
    best = (unit_rows(candidates) @ unit_rows(existing).T).max(axis=1)
    return best < min_similarity


def merge_new(
    table: TopicTable,
    candidates: Sequence[TopicInfo],
    candidate_centres: np.ndarray,
    *,
    min_similarity: float,
) -> tuple[TopicTable, dict[int, int]]:
    """Append the candidates that are new topics.

    Args:
        table: Existing table.
        candidates: Candidate records.
        candidate_centres: Their centres, one row each.
        min_similarity: Cosine at or above which a candidate repeats an existing topic.

    Returns:
        The extended table and a map of candidate id to its new id.
    """
    if not candidates:
        return table, {}
    keep = novel(table.centres, candidate_centres, min_similarity=min_similarity)
    chosen = sorted((c.id, i) for i, c in enumerate(candidates) if keep[i])
    start = int(table.ids.max()) + 1 if table.topics else 0
    mapping = {cid: start + n for n, (cid, _) in enumerate(chosen)}
    infos = [replace(candidates[i], id=mapping[cid]) for cid, i in chosen]
    if not chosen:
        return table, mapping
    rows = np.asarray(candidate_centres, dtype=np.float64)[[i for _, i in chosen]]
    return table.appended(infos, _stored(rows)), mapping
