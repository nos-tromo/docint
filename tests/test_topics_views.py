"""Tests for the topics read side: overview, evidence, documents and member pages."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from qdrant_client import QdrantClient
from topics_fakes import DIM, FakeFit, RecordingClient, add_points, blob_vectors, make_collection, uuid_ids

from docint.core.rag import RAG
from docint.core.topics import views
from docint.core.topics.gather import dense_vector
from docint.core.topics.service import TopicsDeps, TopicsService
from docint.core.topics.store import TopicRun, TopicStore
from docint.utils.cursor import InvalidCursorError, encode_cursor
from docint.utils.env_cfg import TopicsConfig

COLL = "views-coll"
CREATED = "2026-01-01T00:00:00+00:00"
EMBED = "synthetic-embed"
BLOBS = np.eye(3, DIM, dtype=np.float64)
#: Points per blob; blob ``i`` becomes topic ``i``, so size order is 1, 2, 0.
SIZES = (20, 40, 30)
OUTLIERS = 3
LATE = 2
#: Of topic 0's members, the first this many come from ``hash-a``, the rest from ``hash-b``.
HASH_A = 12


def _cfg() -> TopicsConfig:
    """Build the test configuration.

    Returns:
        The configuration.
    """
    return TopicsConfig(
        enabled=True,
        align_on_ingest=True,
        labels=False,
        fit_max_points=100_000,
        min_text_chars=10,
        seed=42,
        min_topic_size=0,
        max_topics=0,
        admit_percentile=5.0,
        merge_min_similarity=0.7,
    )


def _text(pid: str, tag: str) -> str:
    """Build a synthetic chunk text.

    Args:
        pid: Point id.
        tag: Which group the point belongs to.

    Returns:
        Text long enough to join a fit.
    """
    return f"passage {pid} {tag} shared words for the synthetic topic"


def _fill(client: QdrantClient, name: str = COLL) -> None:
    """Create the collection with three blobs, outliers and a coarse parent.

    Args:
        client: Qdrant client.
        name: Collection name.
    """
    make_collection(client, name)
    ids = iter(uuid_ids(sum(SIZES) + OUTLIERS + 1))
    rows: list[tuple[str | int, Any, dict[str, Any]]] = []
    for blob, per in enumerate(SIZES):
        for i, vec in enumerate(blob_vectors(blob, BLOBS[blob : blob + 1], per, 0.05)):
            pid = next(ids)
            file_hash = ("hash-a" if i < HASH_A else "hash-b") if blob == 0 else f"hash-{blob}"
            rows.append((pid, vec, {"search_text": _text(pid, f"blob{blob}"), "file_hash": file_hash}))
    for _ in range(OUTLIERS):
        pid = next(ids)
        vec = np.zeros(DIM)
        vec[DIM - 1] = 1.0
        rows.append((pid, vec, {"search_text": _text(pid, "outlier"), "file_hash": "hash-x"}))
    pid = next(ids)
    rows.append((pid, BLOBS[0], {"search_text": _text(pid, "coarse"), "docint_hier_type": "coarse"}))
    add_points(client, name, rows)


def _add_late(client: QdrantClient, name: str = COLL) -> None:
    """Add points the run has not placed yet.

    Args:
        client: Qdrant client.
        name: Collection name.
    """
    rows: list[tuple[str | int, Any, dict[str, Any]]] = [
        (pid, BLOBS[1], {"search_text": _text(pid, "late"), "file_hash": "hash-late"})
        for pid in uuid_ids(1000 + LATE)[1000:]
    ]
    add_points(client, name, rows)


def _build(client: Any, root: Path, name: str = COLL) -> TopicRun:
    """Build a run over a collection with the scripted fit.

    Args:
        client: Qdrant client.
        root: Store root.
        name: Collection name.

    Returns:
        The active run.
    """
    store = TopicStore(root)
    service = TopicsService(
        TopicsDeps(
            client=client,
            fit=FakeFit(BLOBS),
            store=store,
            labeller=None,
            config=_cfg(),
            embed_model=EMBED,
            stop_words=("the",),
            stop_words_sha256="0" * 64,
            sleep=lambda _s: None,
        )
    )
    service.run(
        "build",
        name,
        logical="docs",
        owner_created_at=lambda: CREATED,
        progress=lambda _stage, _done, _total: None,
        warn=lambda _code, _message: None,
    )
    run = store.active(name, owner_created_at=CREATED)
    assert run is not None
    return run


def _source(point: Any) -> dict[str, Any]:
    """Shape a point as a minimal source.

    Args:
        point: A Qdrant record or scored point.

    Returns:
        The id, text and score.
    """
    return {"id": str(point.id), "text": point.payload["search_text"], "score": getattr(point, "score", None)}


def _no_ensure(_name: str) -> None:
    """Skip index creation.

    Args:
        _name: Ignored.
    """


@pytest.fixture
def client() -> Iterator[QdrantClient]:
    """Yield an in-memory Qdrant client holding the filled collection.

    Yields:
        The client.
    """
    c = QdrantClient(location=":memory:")
    _fill(c)
    yield c
    c.close()


@pytest.fixture
def run(client: QdrantClient, tmp_path: Path) -> TopicRun:
    """Build the run.

    Args:
        client: Qdrant client.
        tmp_path: Per-test directory.

    Returns:
        The active run.
    """
    return _build(client, tmp_path / "topics")


def test_overview_is_none_without_a_run(client: QdrantClient) -> None:
    """No run means no overview, and nothing is asked of Qdrant.

    Args:
        client: Qdrant client.
    """
    recording = RecordingClient(client)
    assert views.overview(recording, None, COLL, embed_model=EMBED, ensure_indexes=_no_ensure) is None
    assert recording.calls == []


def test_overview_counts_coverage_and_sorts_topics_by_size(client: QdrantClient, run: TopicRun) -> None:
    """Coverage counts eligible chunks only, and topics come largest first.

    Args:
        client: Qdrant client.
        run: The active run.
    """
    _add_late(client)
    out = views.overview(client, run, COLL, embed_model=EMBED, ensure_indexes=_no_ensure)

    assert out is not None
    total = sum(SIZES) + OUTLIERS + LATE
    assert out["coverage"] == {"total": total, "assigned": sum(SIZES), "unassigned": OUTLIERS, "pending": LATE}
    assert out["run_id"] == run.run_id
    assert out["created_at"] == run.record["created_at"]
    assert out["updated_at"] == run.record["updated_at"]
    assert out["params"] == run.record["params"]
    assert out["embed_model"] == EMBED
    assert out["embed_model_changed"] is False
    assert [t["id"] for t in out["topics"]] == [1, 2, 0]
    assert [t["size"] for t in out["topics"]] == [40, 30, 20]
    assert out["topics"][0]["share"] == round(40 / total, 4)
    assert set(out["topics"][0]) == {
        "id",
        "label",
        "description",
        "keywords",
        "size",
        "share",
        "origin",
        "added_at",
        "first_seen",
        "last_seen",
    }
    assert out["topics"][0]["origin"] == "build"
    assert all(isinstance(pair, list) and len(pair) == 2 for pair in out["topics"][0]["keywords"])
    json.dumps(out)


def test_overview_flags_a_changed_embed_model(client: QdrantClient, run: TopicRun) -> None:
    """A run built with another embedding model is flagged, naming the run's model.

    Args:
        client: Qdrant client.
        run: The active run.
    """
    out = views.overview(client, run, COLL, embed_model="another-embed", ensure_indexes=_no_ensure)

    assert out is not None
    assert out["embed_model"] == EMBED
    assert out["embed_model_changed"] is True


def test_overview_ensures_indexes_before_its_facet(client: QdrantClient, run: TopicRun) -> None:
    """A server facet needs its index, so the overview ensures indexes first.

    Args:
        client: Qdrant client.
        run: The active run.
    """
    recording = RecordingClient(client)

    def ensure(name: str) -> None:
        """Record the ensure among the client calls.

        Args:
            name: Collection name.
        """
        recording.calls.append(f"ensure:{name}")

    views.overview(recording, run, COLL, embed_model=EMBED, ensure_indexes=ensure)

    assert f"ensure:{COLL}" in recording.calls
    assert recording.calls.index(f"ensure:{COLL}") < recording.calls.index("facet")


def test_evidence_returns_central_passages_as_sources(client: QdrantClient, run: TopicRun) -> None:
    """Evidence is the topic's members nearest its centre, closest first.

    Args:
        client: Qdrant client.
        run: The active run.
    """
    out = views.evidence(client, run, COLL, 1, dense=dense_vector(client, COLL), limit=3, source_of=_source)

    assert out is not None
    assert len(out) == 3
    assert all("blob1" in source["text"] for source in out)
    scores = [source["score"] for source in out]
    assert scores == sorted(scores, reverse=True)


def test_unknown_or_unassigned_topics_have_no_views(client: QdrantClient, run: TopicRun) -> None:
    """Topic ``-1`` and ids the table lacks have no evidence, documents or chunks.

    Args:
        client: Qdrant client.
        run: The active run.
    """
    dense = dense_vector(client, COLL)
    for topic_id in (-1, 99):
        assert views.evidence(client, run, COLL, topic_id, dense=dense, limit=3, source_of=_source) is None
        assert views.documents(client, run, COLL, topic_id, limit=5, names=dict) is None
        assert views.chunks(client, run, COLL, topic_id, cursor=None, limit=5, source_of=_source) is None


def test_documents_list_file_counts_with_names_and_a_more_flag(client: QdrantClient, run: TopicRun) -> None:
    """Documents are counted per file, named when known and flagged when cut.

    Args:
        client: Qdrant client.
        run: The active run.
    """

    def names() -> dict[str, str]:
        """Name one of the two files.

        Returns:
            The hash-to-filename map.
        """
        return {"hash-a": "alpha.pdf"}

    cut = views.documents(client, run, COLL, 0, limit=1, names=names)
    assert cut == {"documents": [{"file_hash": "hash-a", "filename": "alpha.pdf", "count": HASH_A}], "more": True}

    full = views.documents(client, run, COLL, 0, limit=5, names=names)
    assert full == {
        "documents": [
            {"file_hash": "hash-a", "filename": "alpha.pdf", "count": HASH_A},
            {"file_hash": "hash-b", "filename": "hash-b", "count": SIZES[0] - HASH_A},
        ],
        "more": False,
    }


def test_chunks_page_with_a_cursor_and_a_bad_cursor_raises(client: QdrantClient, run: TopicRun) -> None:
    """Members page through an opaque cursor; a malformed one is refused.

    Args:
        client: Qdrant client.
        run: The active run.
    """
    first = views.chunks(client, run, COLL, 2, cursor=None, limit=25, source_of=_source)
    assert first is not None
    assert len(first["items"]) == 25
    assert first["next_cursor"] is not None

    second = views.chunks(client, run, COLL, 2, cursor=first["next_cursor"], limit=25, source_of=_source)
    assert second is not None
    assert len(second["items"]) == SIZES[2] - 25
    assert second["next_cursor"] is None
    ids = {item["id"] for item in [*first["items"], *second["items"]]}
    assert len(ids) == SIZES[2]
    assert all("blob2" in item["text"] for item in [*first["items"], *second["items"]])

    with pytest.raises(InvalidCursorError):
        views.chunks(client, run, COLL, 2, cursor="not a cursor", limit=25, source_of=_source)
    with pytest.raises(InvalidCursorError):
        views.chunks(client, run, COLL, 2, cursor=encode_cursor([1]), limit=25, source_of=_source)


def _rag(client: QdrantClient, root: Path, monkeypatch: pytest.MonkeyPatch, created: str) -> RAG:
    """Build a RAG over an in-memory client and a test store.

    Args:
        client: Qdrant client.
        root: Store root.
        monkeypatch: The monkeypatch fixture.
        created: What the ownership row's ``created_at`` reads as.

    Returns:
        The RAG.
    """
    rag = RAG(qdrant_collection="")
    rag._qdrant_client = client
    monkeypatch.setattr(RAG, "topic_store", lambda self: TopicStore(root))
    monkeypatch.setattr(RAG, "collection_created_at", lambda self, physical: created)
    return rag


def test_overview_is_none_for_a_stale_run(
    client: QdrantClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run of a recreated collection reads as absent, and is not deleted.

    Args:
        client: Qdrant client.
        tmp_path: Per-test directory.
        monkeypatch: The monkeypatch fixture.
    """
    root = tmp_path / "topics"
    run = _build(client, root)

    assert _rag(client, root, monkeypatch, "2026-02-02T00:00:00+00:00").topics_overview(COLL) is None
    assert TopicStore(root).active_id(COLL) == run.run_id

    fresh = _rag(client, root, monkeypatch, CREATED).topics_overview(COLL)
    assert fresh is not None
    assert fresh["run_id"] == run.run_id


def test_rag_topic_sources_never_name_the_physical_collection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Sources served by the topic reads carry no owner-namespaced collection name.

    Args:
        tmp_path: Per-test directory.
        monkeypatch: The monkeypatch fixture.
    """
    physical = "u0123456789ab__docs"
    client = QdrantClient(location=":memory:")
    _fill(client, physical)
    root = tmp_path / "topics"
    _build(client, root, physical)
    rag = _rag(client, root, monkeypatch, CREATED)

    evidence = rag.topic_evidence(physical, 1, limit=3)
    page = rag.topic_chunks(physical, 1, cursor=None, limit=5)

    assert evidence is not None
    assert page is not None
    assert evidence[0]["file_hash"] == "hash-1"
    assert physical not in json.dumps([evidence, page])
    client.close()
