"""Tests for the topics package's Qdrant reads and payload writes."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException
from topics_fakes import DENSE_NAME, DIM, RecordingClient, add_points, make_collection, uuid_ids

from docint.core.topics import gather
from docint.core.topics.gather import (
    FILE_HASH_FIELD,
    RUN_FIELD,
    TOPIC_FIELD,
    DenseVector,
    and_filters,
    as_point_id,
    central_points,
    count,
    dense_vector,
    document_counts,
    ensure_topic_indexes,
    iter_rows,
    member_page,
    parse_topic_value,
    pending_filter,
    point_text,
    retrieve_candidates,
    run_filter,
    scan_ids,
    topic_counts,
    topic_filter,
    topic_value,
    write_assignments,
)

COLL = "c"


def _vec(i: int) -> list[float]:
    """Build a distinct vector.

    Args:
        i: Seed index.

    Returns:
        A DIM-wide vector.
    """
    v = [0.1] * DIM
    v[i % DIM] = 1.0 + i
    return v


@pytest.fixture
def client() -> Iterator[QdrantClient]:
    """Yield an in-memory Qdrant client.

    Yields:
        The client.
    """
    c = QdrantClient(location=":memory:")
    yield c
    c.close()


def _seed(client: QdrantClient, n: int = 6, *, named: bool = True) -> list[str]:
    """Create the collection and add ``n`` points with search text.

    Args:
        client: Qdrant client.
        n: Point count.
        named: Hybrid shape or unnamed.

    Returns:
        The point ids.
    """
    make_collection(client, COLL, named=named)
    ids = uuid_ids(n)
    add_points(
        client,
        COLL,
        [(pid, _vec(i), {"search_text": f"text {i}", FILE_HASH_FIELD: f"h{i % 2}"}) for i, pid in enumerate(ids)],
        named=named,
    )
    return ids


def test_dense_vector_reads_named_and_unnamed_configs(client: QdrantClient) -> None:
    """A named config yields its name; an unnamed one yields none.

    Args:
        client: Qdrant client.
    """
    make_collection(client, "named")
    make_collection(client, "plain", named=False, dim=4)
    client.create_collection(
        "other", vectors_config={"emb": models.VectorParams(size=3, distance=models.Distance.COSINE)}
    )
    assert dense_vector(client, "named") == DenseVector(DENSE_NAME, DIM)
    assert dense_vector(client, "plain") == DenseVector(None, 4)
    assert dense_vector(client, "other") == DenseVector("emb", 3)
    assert DenseVector(None, 4).selector() is True
    assert DenseVector("emb", 3).selector() == ["emb"]


def test_dense_vector_refuses_an_ambiguous_named_config(client: QdrantClient) -> None:
    """Several named vectors without text-dense cannot be told apart.

    Args:
        client: Qdrant client.
    """
    client.create_collection(
        "amb",
        vectors_config={
            "a": models.VectorParams(size=3, distance=models.Distance.COSINE),
            "b": models.VectorParams(size=3, distance=models.Distance.COSINE),
        },
    )
    with pytest.raises(ValueError, match="several named vectors"):
        dense_vector(client, "amb")


class _SchemaClient:
    """A client whose payload schema is scripted, recording index calls."""

    def __init__(self, schema: dict[str, str]) -> None:
        """Script the schema.

        Args:
            schema: Key to index data type.
        """
        self.schema = {k: SimpleNamespace(data_type=SimpleNamespace(value=v)) for k, v in schema.items()}
        self.calls: list[tuple[str, str]] = []

    def get_collection(self, collection_name: str) -> Any:
        """Return the scripted schema.

        Args:
            collection_name: Ignored.

        Returns:
            An object with a ``payload_schema``.
        """
        return SimpleNamespace(payload_schema=self.schema)

    def delete_payload_index(self, collection_name: str, field_name: str, wait: bool = True) -> None:
        """Record a delete.

        Args:
            collection_name: Ignored.
            field_name: Key.
            wait: Ignored.
        """
        self.calls.append(("delete", field_name))

    def create_payload_index(self, collection_name: str, field_name: str, field_schema: Any, wait: bool = True) -> None:
        """Record a create.

        Args:
            collection_name: Ignored.
            field_name: Key.
            field_schema: Must be KEYWORD.
            wait: Ignored.
        """
        assert field_schema == models.PayloadSchemaType.KEYWORD
        assert wait is True
        self.calls.append(("create", field_name))


def test_ensure_topic_indexes_creates_three_keyword_indexes() -> None:
    """Each unindexed key gets one KEYWORD index and nothing is deleted."""
    stub = _SchemaClient({})
    ensure_topic_indexes(stub, COLL)
    assert stub.calls == [("create", TOPIC_FIELD), ("create", RUN_FIELD), ("create", FILE_HASH_FIELD)]


def test_ensure_topic_indexes_replaces_a_wrong_kind_index() -> None:
    """A text index is dropped before the keyword one; a keyword one is left alone."""
    stub = _SchemaClient({TOPIC_FIELD: "keyword", RUN_FIELD: "text"})
    ensure_topic_indexes(stub, COLL)
    assert stub.calls == [("delete", RUN_FIELD), ("create", RUN_FIELD), ("create", FILE_HASH_FIELD)]


def test_ensure_topic_indexes_is_quiet_in_local_mode(client: QdrantClient, recwarn: pytest.WarningsRecorder) -> None:
    """The local-mode no-effect warning is suppressed.

    Args:
        client: Qdrant client.
        recwarn: Captured warnings.
    """
    make_collection(client, COLL)
    ensure_topic_indexes(client, COLL)
    assert not [w for w in recwarn if "Payload indexes" in str(w.message)]


def test_pending_filter_selects_points_outside_the_run_and_never_coarse_ones(client: QdrantClient) -> None:
    """Pending means not in the run and not a coarse parent.

    Args:
        client: Qdrant client.
    """
    make_collection(client, COLL)
    add_points(
        client,
        COLL,
        [
            (1, _vec(1), {}),
            (2, _vec(2), {RUN_FIELD: "r1"}),
            (3, _vec(3), {RUN_FIELD: "r0"}),
            (4, _vec(4), {"docint_hier_type": "coarse"}),
            (5, _vec(5), {"docint_hier_type": "fine"}),
        ],
    )
    assert sorted(scan_ids(client, COLL, scroll_filter=pending_filter("r1"))) == ["1", "3", "5"]


def test_scan_ids_streams_every_eligible_id(client: QdrantClient) -> None:
    """Ids come back as strings across more than one page.

    Args:
        client: Qdrant client.
    """
    make_collection(client, COLL)
    add_points(client, COLL, [(i + 1, _vec(i), {}) for i in range(gather.PAGE_SIZE + 5)])
    ids = list(scan_ids(client, COLL, scroll_filter=gather.eligible_filter()))
    assert len(ids) == gather.PAGE_SIZE + 5
    assert len(set(ids)) == len(ids)
    assert all(isinstance(i, str) for i in ids)


def test_candidates_prefer_search_text_and_fall_back_to_payload_text(client: QdrantClient) -> None:
    """Blank search text falls back to the node text; points without a vector or row are dropped.

    Args:
        client: Qdrant client.
    """
    make_collection(client, COLL)
    a, b, c, d = uuid_ids(4)
    node = json.dumps({"text": "from node"})
    add_points(
        client,
        COLL,
        [
            (a, _vec(0), {"search_text": "kept search"}),
            (b, _vec(1), {"search_text": "  ", "_node_content": node}),
            (c, _vec(2), {"text": "plain text"}),
        ],
    )
    seen: list[int] = []
    got = retrieve_candidates(client, COLL, [c, a, d, b], dense_vector(client, COLL), progress=seen.append)
    assert [(x.point_id, x.text) for x in got] == [(c, "plain text"), (a, "kept search"), (b, "from node")]
    assert got[0].vector.dtype == np.float32
    assert seen == [4]


def test_candidates_read_in_batches(client: QdrantClient) -> None:
    """More ids than a page are read page by page with progress after each.

    Args:
        client: Qdrant client.
    """
    make_collection(client, COLL, named=False)
    n = gather.PAGE_SIZE + 3
    add_points(client, COLL, [(i + 1, _vec(i), {"search_text": f"t{i}"}) for i in range(n)], named=False)
    seen: list[int] = []
    got = retrieve_candidates(
        client, COLL, [str(i + 1) for i in range(n)], dense_vector(client, COLL), progress=seen.append
    )
    assert len(got) == n
    assert seen == [gather.PAGE_SIZE, n]


def test_iter_rows_reads_named_and_unnamed_vectors_and_the_timestamp_keys(client: QdrantClient) -> None:
    """Rows carry the dense vector and only the selected payload keys, nested ones intact.

    Args:
        client: Qdrant client.
    """
    for name, named in (("n", True), ("u", False)):
        make_collection(client, name, named=named)
        add_points(
            client,
            name,
            [(1, _vec(1), {"reference_metadata": {"timestamp": "t1"}, "other": 1}), (2, _vec(2), {})],
            named=named,
        )
        rows = [
            r
            for page in iter_rows(
                client, name, scroll_filter=gather.eligible_filter(), dense=dense_vector(client, name)
            )
            for r in page
        ]
        by_id = {r.point_id: r for r in rows}
        assert set(by_id) == {"1", "2"}
        assert by_id["1"].vector is not None and by_id["1"].vector.shape == (DIM,)
        assert by_id["1"].payload == {"reference_metadata": {"timestamp": "t1"}}
        assert by_id["2"].payload == {}


def test_iter_rows_survives_writes_behind_its_cursor(client: QdrantClient) -> None:
    """Stamping each page as it is read never makes the scroll skip a point.

    Args:
        client: Qdrant client.
    """
    make_collection(client, COLL)
    n = gather.PAGE_SIZE * 2 + 10
    add_points(client, COLL, [(i + 1, _vec(i), {}) for i in range(n)])
    dense = dense_vector(client, COLL)
    seen: list[str] = []
    for page in iter_rows(client, COLL, scroll_filter=pending_filter("r1"), dense=dense):
        ids = [r.point_id for r in page]
        seen.extend(ids)
        write_assignments(client, COLL, "r1", {0: ids})
    assert len(seen) == n
    assert len(set(seen)) == n
    assert count(client, COLL, run_filter("r1")) == n


def test_write_assignments_sets_both_keys_with_one_operation_per_topic(client: QdrantClient) -> None:
    """One batch carries one operation per non-empty topic, and both keys land.

    Args:
        client: Qdrant client.
    """
    ids = _seed(client)
    rec = RecordingClient(client)
    written = write_assignments(rec, COLL, "r1", {0: ids[:2], 1: ids[2:5], 2: []})
    assert written == 5
    assert rec.calls == ["batch_update_points"]
    points = {str(p.id): p.payload or {} for p in client.retrieve(COLL, ids)}
    assert points[ids[0]][TOPIC_FIELD] == "r1:0" and points[ids[0]][RUN_FIELD] == "r1"
    assert points[ids[3]][TOPIC_FIELD] == "r1:1"
    assert TOPIC_FIELD not in points[ids[5]]
    assert points[ids[0]]["search_text"] == "text 0"
    assert write_assignments(rec, COLL, "r1", {}) == 0
    assert rec.calls == ["batch_update_points"]


def test_write_assignments_skips_points_deleted_meanwhile(client: QdrantClient) -> None:
    """A deleted id does not fail the batch.

    Args:
        client: Qdrant client.
    """
    ids = _seed(client)
    client.delete(COLL, points_selector=models.PointIdsList(points=[ids[1]]), wait=True)
    assert write_assignments(client, COLL, "r1", {0: ids[:3]}) == 3
    assert count(client, COLL, topic_filter("r1", 0)) == 2


def test_write_assignments_retries_a_transient_failure(client: QdrantClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A transient transport error is retried once and then succeeds.

    Args:
        client: Qdrant client.
        monkeypatch: Pytest monkeypatch.
    """
    ids = _seed(client)
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    class Flaky:
        """Fails the first batch."""

        def __init__(self) -> None:
            """Start with no attempts."""
            self.attempts = 0

        def batch_update_points(self, **kwargs: Any) -> Any:
            """Raise once, then delegate.

            Args:
                **kwargs: The call's arguments.

            Returns:
                The inner result.

            Raises:
                ResponseHandlingException: On the first call.
            """
            self.attempts += 1
            if self.attempts == 1:
                raise ResponseHandlingException(Exception("connection reset"))
            return client.batch_update_points(**kwargs)

    flaky = Flaky()
    assert write_assignments(flaky, COLL, "r1", {0: ids[:2]}) == 2
    assert flaky.attempts == 2
    assert count(client, COLL, topic_filter("r1", 0)) == 2


def test_topic_counts_read_the_runs_facet_and_ignore_other_runs(client: QdrantClient) -> None:
    """Counts are per topic of the asked run only.

    Args:
        client: Qdrant client.
    """
    ids = _seed(client)
    write_assignments(client, COLL, "r1", {0: ids[:3], 1: ids[3:5], -1: ids[5:]})
    write_assignments(client, COLL, "r0", {7: ids[:1]})
    # The first point now belongs to run r0, so r1 no longer counts it.
    assert topic_counts(client, COLL, "r1", limit=10) == {0: 2, 1: 2, -1: 1}
    assert topic_counts(client, COLL, "r0", limit=10) == {7: 1}


def test_topic_counts_count_only_the_members_an_extra_filter_selects(client: QdrantClient) -> None:
    """An extra filter narrows every topic's count, and a topic it leaves no member in drops out.

    Args:
        client: Qdrant client.
    """
    ids = _seed(client)
    write_assignments(client, COLL, "r1", {0: ids[:3], 1: ids[3:4], -1: ids[4:]})
    is_h1 = models.FieldCondition(key=FILE_HASH_FIELD, match=models.MatchValue(value="h1"))
    only_h1, not_h1 = models.Filter(must=[is_h1]), models.Filter(must_not=[is_h1])
    # Indices 0..5 carry hashes h0,h1,h0,h1,h0,h1.
    assert topic_counts(client, COLL, "r1", limit=10, extra=only_h1) == {0: 1, 1: 1, -1: 1}
    assert topic_counts(client, COLL, "r1", limit=10, extra=not_h1) == {0: 2, -1: 1}
    assert topic_counts(client, COLL, "r1", limit=10, extra=None) == topic_counts(client, COLL, "r1", limit=10)


def test_document_counts_facet_file_hash_within_a_topic(client: QdrantClient) -> None:
    """Documents are counted within the topic, largest first and by hash on ties.

    Args:
        client: Qdrant client.
    """
    ids = _seed(client)
    write_assignments(client, COLL, "r1", {0: ids[:5], 1: ids[5:]})
    # Indices 0..4 carry hashes h0,h1,h0,h1,h0.
    assert document_counts(client, COLL, "r1", 0, limit=10) == [("h0", 3), ("h1", 2)]
    assert document_counts(client, COLL, "r1", 1, limit=10) == [("h1", 1)]


def test_central_points_rank_by_closeness_within_the_topic_and_the_extra_filter(client: QdrantClient) -> None:
    """Only topic members are ranked, nearest first, narrowed by the extra filter.

    Args:
        client: Qdrant client.
    """
    ids = _seed(client)
    write_assignments(client, COLL, "r1", {0: ids[:4]})
    dense = dense_vector(client, COLL)
    centre = np.asarray(_vec(2), dtype=np.float32)
    got = central_points(client, COLL, centre=centre, dense=dense, run_id="r1", topic_id=0, limit=3)
    assert str(got[0].id) == ids[2]
    assert {str(p.id) for p in got} <= set(ids[:4])
    only_h1 = models.Filter(must=[models.FieldCondition(key=FILE_HASH_FIELD, match=models.MatchValue(value="h1"))])
    narrowed = central_points(client, COLL, centre=centre, dense=dense, run_id="r1", topic_id=0, limit=3, extra=only_h1)
    assert {str(p.id) for p in narrowed} == {ids[1], ids[3]}


def test_member_page_pages_with_the_scroll_offset(client: QdrantClient) -> None:
    """Pages cover every member exactly once and end with a None offset.

    Args:
        client: Qdrant client.
    """
    ids = _seed(client)
    write_assignments(client, COLL, "r1", {0: ids[:5]})
    seen: list[str] = []
    offset: Any = None
    while True:
        points, offset = member_page(client, COLL, run_id="r1", topic_id=0, offset=offset, limit=2)
        seen.extend(str(p.id) for p in points)
        if offset is None:
            break
    assert sorted(seen) == sorted(ids[:5])


def test_topic_values_round_trip_and_garbage_does_not_parse() -> None:
    """Values round-trip, including negative ids; anything else is None."""
    assert parse_topic_value(topic_value("20260101-000000-abc123", 4)) == ("20260101-000000-abc123", 4)
    assert parse_topic_value(topic_value("r", -1)) == ("r", -1)
    for bad in (None, 3, "", "r", ":3", "r:x", "r:"):
        assert parse_topic_value(bad) is None


def test_ids_and_text_helpers() -> None:
    """Digit ids become ints; text prefers search_text; filters conjoin."""
    assert as_point_id("12") == 12
    assert as_point_id("a-b") == "a-b"
    assert point_text({"search_text": "s", "text": "t"}) == "s"
    assert point_text({"search_text": " ", "text": "t"}) == "t"
    merged = and_filters(topic_filter("r", 1), None, pending_filter("r"))
    assert isinstance(merged.must, list) and len(merged.must) == 2
    assert isinstance(merged.must_not, list) and len(merged.must_not) == 1
