"""Qdrant reads and payload writes for topic runs.

The only Qdrant-facing module of the topics package. Topic membership lives in
two payload keys written payload-only, so assigning a point never touches its
vectors, text or search index.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger
from qdrant_client import models

from docint.core.search.fields import read_payload_schema
from docint.core.search.fulltext import not_coarse_condition
from docint.core.storage.scroll import iter_scroll
from docint.core.summary.units import payload_text
from docint.core.topics.sample import Candidate
from docint.utils.retry import is_transient_qdrant_error, retry_with_backoff

TOPIC_FIELD = "docint_topic"
RUN_FIELD = "docint_topic_run"
FILE_HASH_FIELD = "file_hash"
UNASSIGNED = -1
PAGE_SIZE = 256
TIMESTAMP_KEYS = ("reference_metadata.timestamp", "reference_metadata.posting_timestamp")

_DENSE_NAME = "text-dense"
_SEARCH_TEXT = "search_text"
_FALLBACK_TEXT_KEYS = ["text", "chunk_text", "chunk", "content", "_node_content"]
_INDEXED_KEYS = (TOPIC_FIELD, RUN_FIELD, FILE_HASH_FIELD)


def topic_value(run_id: str, topic_id: int) -> str:
    """Build the payload value that places a point in a topic.

    Args:
        run_id: Run id.
        topic_id: Topic id, ``-1`` for unassigned.

    Returns:
        ``"<run_id>:<topic_id>"``.
    """
    return f"{run_id}:{topic_id}"


def parse_topic_value(value: object) -> tuple[str, int] | None:
    """Split a payload value back into its run and topic.

    Args:
        value: A stored ``docint_topic`` value, or anything else.

    Returns:
        ``(run_id, topic_id)``, or ``None`` when the value is not a topic value.
    """
    if not isinstance(value, str):
        return None
    run_id, sep, tail = value.rpartition(":")
    if not sep or not run_id:
        return None
    try:
        return run_id, int(tail)
    except ValueError:
        return None


def as_point_id(point_id: str) -> str | int:
    """Restore a point id's native Qdrant type.

    Args:
        point_id: Point id as it travels through JSON.

    Returns:
        An integer for an all-digit id, else the id unchanged.
    """
    return int(point_id) if point_id.isdigit() else point_id


@dataclass(frozen=True)
class DenseVector:
    """Where a collection keeps its dense vector.

    Attributes:
        name: The vector's name, or ``None`` for an unnamed vector.
        dim: Its dimension.
    """

    name: str | None
    dim: int

    def selector(self) -> bool | list[str]:
        """Return the ``with_vectors`` argument that fetches only this vector.

        Returns:
            ``True`` when unnamed, else the one-name list.
        """
        return True if self.name is None else [self.name]

    def of(self, point: Any) -> np.ndarray | None:
        """Read this vector off a retrieved point.

        Args:
            point: A Qdrant record or scored point.

        Returns:
            A float32 array, or ``None`` when the point carries no such vector.
        """
        raw = getattr(point, "vector", None)
        if isinstance(raw, Mapping):
            raw = raw.get(self.name) if self.name is not None else None
        if raw is None or not isinstance(raw, (list, tuple, np.ndarray)):
            return None
        return np.asarray(raw, dtype=np.float32)


def dense_vector(client: Any, collection: str) -> DenseVector:
    """Find a collection's dense vector.

    Args:
        client: Qdrant client.
        collection: Physical collection name.

    Returns:
        The dense vector's name and dimension.

    Raises:
        ValueError: When several named vectors exist and none is ``text-dense``.
    """
    config = client.get_collection(collection_name=collection).config.params.vectors
    if isinstance(config, Mapping):
        if _DENSE_NAME in config:
            return DenseVector(_DENSE_NAME, int(config[_DENSE_NAME].size))
        if len(config) == 1:
            ((name, params),) = config.items()
            return DenseVector(str(name), int(params.size))
        raise ValueError(f"Collection '{collection}' has several named vectors and none is '{_DENSE_NAME}'.")
    return DenseVector(None, int(config.size))


def ensure_topic_indexes(client: Any, collection: str) -> None:
    """Give the three topic keys a KEYWORD index, replacing a wrong-kind one.

    Args:
        client: Qdrant client.
        collection: Physical collection name.
    """
    schema = read_payload_schema(client, collection)
    for key in _INDEXED_KEYS:
        entry = schema.get(key)
        data_type = getattr(entry, "data_type", None)
        kind = str(getattr(data_type, "value", data_type)).lower() if entry is not None else None
        if kind == "keyword":
            continue
        if entry is not None:
            client.delete_payload_index(collection_name=collection, field_name=key, wait=True)
        # A no-op (with a warning) in local mode; the warning is noise there.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Payload indexes have no effect", category=UserWarning)
            client.create_payload_index(
                collection_name=collection,
                field_name=key,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )


def eligible_filter() -> models.Filter:
    """Select the points a topic can be assigned to.

    Returns:
        Every point except coarse parent chunks.
    """
    return not_coarse_condition()


def pending_filter(run_id: str) -> models.Filter:
    """Select eligible points the run has not yet placed.

    Args:
        run_id: Run id.

    Returns:
        Eligible points whose run key is not ``run_id``.
    """
    return models.Filter(
        must=[not_coarse_condition()],
        must_not=[models.FieldCondition(key=RUN_FIELD, match=models.MatchValue(value=run_id))],
    )


def run_filter(run_id: str) -> models.Filter:
    """Select the points a run has placed.

    Args:
        run_id: Run id.

    Returns:
        Points whose run key is ``run_id``.
    """
    return models.Filter(must=[models.FieldCondition(key=RUN_FIELD, match=models.MatchValue(value=run_id))])


def topic_filter(run_id: str, topic_id: int) -> models.Filter:
    """Select the members of one topic.

    Args:
        run_id: Run id.
        topic_id: Topic id.

    Returns:
        Points whose topic key is that run's topic.
    """
    return models.Filter(
        must=[models.FieldCondition(key=TOPIC_FIELD, match=models.MatchValue(value=topic_value(run_id, topic_id)))]
    )


def and_filters(*filters: models.Filter | None) -> models.Filter:
    """Conjoin filters, skipping absent ones.

    Args:
        *filters: Filters to combine; ``None`` entries are ignored.

    Returns:
        One filter holding every clause of every given filter.
    """
    must: list[Any] = []
    must_not: list[Any] = []
    for flt in filters:
        if flt is None:
            continue
        must.extend(_clauses(flt.must))
        must_not.extend(_clauses(flt.must_not))
        if flt.should:
            # A should-group is only a disjunction on its own; nest it so it stays one.
            must.append(models.Filter(should=_clauses(flt.should)))
    return models.Filter(must=must or None, must_not=must_not or None)


def _clauses(raw: Any) -> list[Any]:
    """Normalise a filter clause slot to a list.

    Args:
        raw: A condition, a list of conditions or ``None``.

    Returns:
        The conditions as a list.
    """
    if raw is None:
        return []
    return list(raw) if isinstance(raw, list) else [raw]


def scan_ids(client: Any, collection: str, *, scroll_filter: models.Filter) -> Iterator[str]:
    """Stream the id of every point a filter selects.

    Args:
        client: Qdrant client.
        collection: Physical collection name.
        scroll_filter: Which points.

    Yields:
        Point ids as strings.
    """
    for page in iter_scroll(
        client,
        collection_name=collection,
        scroll_filter=scroll_filter,
        page_size=PAGE_SIZE,
        with_payload=False,
        with_vectors=False,
        on_error="raise",
    ):
        for point in page:
            yield str(point.id)


@dataclass(frozen=True)
class PointRow:
    """One scrolled point.

    Attributes:
        point_id: Point id as a string.
        vector: The dense vector, ``None`` when the point has none.
        payload: The selected payload keys as Qdrant returned them.
    """

    point_id: str
    vector: np.ndarray | None
    payload: Mapping[str, Any]


def point_text(payload: Mapping[str, Any]) -> str:
    """Pick a chunk's text from its payload.

    Args:
        payload: Raw Qdrant payload.

    Returns:
        ``search_text`` when non-blank, else the best-effort node text.
    """
    stored = payload.get(_SEARCH_TEXT)
    if isinstance(stored, str) and stored.strip():
        return stored
    return payload_text(dict(payload))


def retrieve_candidates(
    client: Any,
    collection: str,
    ids: Sequence[str],
    dense: DenseVector,
    *,
    progress: Callable[[int], None] | None = None,
) -> list[Candidate]:
    """Fetch fit candidates for known ids, in the order given.

    Args:
        client: Qdrant client.
        collection: Physical collection name.
        ids: Point ids to read.
        dense: The collection's dense vector.
        progress: Called with the number of ids read after each batch.

    Returns:
        A candidate per id found with a dense vector; the rest are skipped.
    """
    out: list[Candidate] = []
    read = 0
    for start in range(0, len(ids), PAGE_SIZE):
        batch = list(ids[start : start + PAGE_SIZE])
        points = client.retrieve(
            collection_name=collection,
            ids=[as_point_id(i) for i in batch],
            with_vectors=dense.selector(),
            with_payload=[_SEARCH_TEXT],
        )
        by_id = {str(p.id): p for p in points}
        blank = [
            pid
            for pid, p in by_id.items()
            if dense.of(p) is not None and not str((p.payload or {}).get(_SEARCH_TEXT) or "").strip()
        ]
        fallback: dict[str, str] = {}
        if blank:
            for p in client.retrieve(
                collection_name=collection,
                ids=[as_point_id(i) for i in blank],
                with_vectors=False,
                with_payload=_FALLBACK_TEXT_KEYS,
            ):
                fallback[str(p.id)] = point_text(p.payload or {})
        for pid in batch:
            point = by_id.get(pid)
            if point is None:
                continue
            vector = dense.of(point)
            if vector is None:
                continue
            text = fallback[pid] if pid in fallback else point_text(point.payload or {})
            out.append(Candidate(point_id=pid, vector=vector, text=text))
        read += len(batch)
        if progress is not None:
            progress(read)
    return out


def iter_rows(
    client: Any,
    collection: str,
    *,
    scroll_filter: models.Filter,
    dense: DenseVector,
    payload_keys: Sequence[str] = TIMESTAMP_KEYS,
) -> Iterator[list[PointRow]]:
    """Scroll points with their dense vector and selected payload keys.

    A forward pass over the scroll's own offset cursor, so payload writes
    behind the cursor never cause a point to be skipped.

    Args:
        client: Qdrant client.
        collection: Physical collection name.
        scroll_filter: Which points.
        dense: The collection's dense vector.
        payload_keys: Payload keys to fetch.

    Yields:
        One list of rows per scroll page.
    """
    for page in iter_scroll(
        client,
        collection_name=collection,
        scroll_filter=scroll_filter,
        page_size=PAGE_SIZE,
        with_payload=list(payload_keys),
        with_vectors=dense.selector(),
        on_error="raise",
    ):
        yield [PointRow(str(p.id), dense.of(p), p.payload or {}) for p in page]


def write_assignments(client: Any, collection: str, run_id: str, groups: Mapping[int, Sequence[str]]) -> int:
    """Stamp points with their topic, one operation per topic.

    Points are selected by id filter, not a points list, so one deleted
    meanwhile is skipped instead of failing the batch.

    Args:
        client: Qdrant client.
        collection: Physical collection name.
        run_id: Run id.
        groups: Topic id to the point ids placed in it.

    Returns:
        The number of ids written.
    """
    operations = [
        models.SetPayloadOperation(
            set_payload=models.SetPayload(
                payload={TOPIC_FIELD: topic_value(run_id, topic_id), RUN_FIELD: run_id},
                filter=models.Filter(must=[models.HasIdCondition(has_id=[as_point_id(i) for i in ids])]),
            )
        )
        for topic_id, ids in groups.items()
        if ids
    ]
    if not operations:
        return 0
    retry_with_backoff(
        "topics_write",
        lambda: client.batch_update_points(collection_name=collection, update_operations=operations, wait=True),
        max_retries=3,
        initial_backoff=0.5,
        max_backoff=4.0,
        is_retryable=is_transient_qdrant_error,
    )
    written = sum(len(ids) for ids in groups.values())
    logger.debug("Topic assignments written | topics={} points={}", len(operations), written)
    return written


def count(client: Any, collection: str, flt: models.Filter) -> int:
    """Count the points a filter selects.

    Args:
        client: Qdrant client.
        collection: Physical collection name.
        flt: Which points.

    Returns:
        The exact count.
    """
    return int(client.count(collection_name=collection, count_filter=flt, exact=True).count)


def topic_counts(client: Any, collection: str, run_id: str, *, limit: int) -> dict[int, int]:
    """Count members per topic for one run.

    Args:
        client: Qdrant client.
        collection: Physical collection name.
        run_id: Run id.
        limit: Most facet values to read.

    Returns:
        Topic id to member count; values of other runs are ignored.
    """
    hits = client.facet(
        collection_name=collection, key=TOPIC_FIELD, facet_filter=run_filter(run_id), exact=True, limit=limit
    ).hits
    out: dict[int, int] = {}
    for hit in hits:
        parsed = parse_topic_value(hit.value)
        if parsed is not None and parsed[0] == run_id:
            out[parsed[1]] = int(hit.count)
    return out


def document_counts(client: Any, collection: str, run_id: str, topic_id: int, *, limit: int) -> list[tuple[str, int]]:
    """Count a topic's members per source document.

    Args:
        client: Qdrant client.
        collection: Physical collection name.
        run_id: Run id.
        topic_id: Topic id.
        limit: Most facet values to read.

    Returns:
        ``(file_hash, count)`` pairs, largest first and by hash on ties.
    """
    hits = client.facet(
        collection_name=collection,
        key=FILE_HASH_FIELD,
        facet_filter=topic_filter(run_id, topic_id),
        exact=True,
        limit=limit,
    ).hits
    return sorted(((str(h.value), int(h.count)) for h in hits), key=lambda item: (-item[1], item[0]))


def central_points(
    client: Any,
    collection: str,
    *,
    centre: np.ndarray,
    dense: DenseVector,
    run_id: str,
    topic_id: int,
    limit: int,
    extra: models.Filter | None = None,
) -> list[Any]:
    """Find a topic's members closest to a centre.

    Args:
        client: Qdrant client.
        collection: Physical collection name.
        centre: The query vector.
        dense: The collection's dense vector.
        run_id: Run id.
        topic_id: Topic id.
        limit: Most points to return.
        extra: A further filter the members must satisfy.

    Returns:
        Scored points, closest first.
    """
    return list(
        client.query_points(
            collection_name=collection,
            query=centre.tolist(),
            using=dense.name,
            query_filter=and_filters(topic_filter(run_id, topic_id), extra),
            limit=limit,
            with_payload=True,
        ).points
    )


def member_page(
    client: Any, collection: str, *, run_id: str, topic_id: int, offset: Any, limit: int
) -> tuple[list[Any], Any]:
    """Read one page of a topic's members.

    Args:
        client: Qdrant client.
        collection: Physical collection name.
        run_id: Run id.
        topic_id: Topic id.
        offset: The scroll offset from the previous page, ``None`` for the first.
        limit: Page size.

    Returns:
        The points and the next offset (``None`` at the end).
    """
    points, next_offset = client.scroll(
        collection_name=collection,
        scroll_filter=topic_filter(run_id, topic_id),
        offset=offset,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return list(points), next_offset
