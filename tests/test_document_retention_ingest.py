"""Tests for how ingest feeds the document retention index and stamps its clock on the points."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from llama_index.core.schema import MetadataMode, TextNode
from qdrant_client import QdrantClient, models
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from docint.core.rag import RAG, RETENTION_STAMP_KEY
from docint.core.retention import RetentionPeriodState
from docint.core.state.base import Base
from docint.core.state.collection_owner_manager import CollectionOwnerManager
from docint.core.state.document_retention_manager import DocumentRetentionManager

NOW = datetime(2026, 11, 5, 12, 0, tzinfo=UTC)
DAY = timedelta(days=1)


@pytest.fixture
def rag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> RAG:
    """A RAG with in-memory ownership and retention stores sharing one database.

    Args:
        monkeypatch: The monkeypatch fixture.
        tmp_path: Pytest-provided temporary directory.

    Returns:
        RAG: The RAG under test, on collection ``phys``.
    """
    monkeypatch.delenv("DOCUMENT_RETENTION_DAYS", raising=False)
    monkeypatch.delenv("RETENTION_NOTICE_DAYS", raising=False)
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    instance = RAG(qdrant_collection="phys")
    instance._qdrant_client = MagicMock()
    instance._qdrant_src_dir = tmp_path / "sources"
    owners = CollectionOwnerManager(instance)
    owners._SessionMaker = sessionmaker(bind=engine)
    instance.collection_owners = owners
    instance._collection_backfill_done = True
    retention = DocumentRetentionManager(instance)
    retention._SessionMaker = sessionmaker(bind=engine)
    instance.document_retention = retention
    return instance


def _node(file_hash: str, **extra: Any) -> TextNode:
    return TextNode(
        text="zz synthetic text", metadata={"file_hash": file_hash, "file_name": f"{file_hash}.pdf", **extra}
    )


def test_persist_stamps_the_ingest_time_and_hides_it(rag: RAG) -> None:
    """Every node carries its ingest time, which neither the embedding nor the prompt sees."""
    node = _node("h1")
    before = node.get_content(metadata_mode=MetadataMode.EMBED)
    rag._note_retention_metadata([node], now=NOW)
    assert node.metadata[RETENTION_STAMP_KEY] == NOW.timestamp()
    assert RETENTION_STAMP_KEY in node.excluded_embed_metadata_keys
    assert RETENTION_STAMP_KEY in node.excluded_llm_metadata_keys
    assert node.get_content(metadata_mode=MetadataMode.EMBED) == before
    rag._note_retention_metadata([node], now=NOW + DAY)
    assert node.metadata[RETENTION_STAMP_KEY] == NOW.timestamp()


def test_meta_collects_filename_and_point_hashes(rag: RAG) -> None:
    """A document's file name and every hash its points carry are gathered while it is persisted."""
    nodes = [
        _node("clip", media_file_hash="clip-media"),
        _node("clip", source_doc_id="figure-doc", posting_uuid="post-1"),
        _node("clip", source_doc_id="post-2", posting_uuid="post-2"),
        _node("other"),
    ]
    rag._note_retention_metadata(nodes, now=NOW)
    filename, hashes = rag._ingest_doc_meta[("phys", "clip")]
    assert filename == "clip.pdf"
    assert hashes == {"clip", "clip-media", "figure-doc"}


def test_completed_document_is_indexed_with_its_owner(rag: RAG) -> None:
    """When a document's ingest completes, its row names the collection's owner and logical name."""
    physical = rag.ensure_collection_owner_manager().register("alice", "docs")
    rag.qdrant_collection = physical
    rag.ensure_collection_owner_manager().record_retention_period(365, now=NOW - 400 * DAY)
    rag._note_retention_metadata([_node("h1")], now=NOW)
    rag._index_completed_document("h1", now=NOW)
    (row,) = rag.ensure_document_retention_manager().rows_for(physical)
    assert (row.owner, row.logical, row.file_hash, row.filename) == ("alice", "docs", "h1", "h1.pdf")
    assert row.expires_at == NOW + 365 * DAY


def test_a_cli_ingest_is_indexed_without_an_owner(rag: RAG) -> None:
    """A collection without an ownership row still gets its documents indexed."""
    rag._note_retention_metadata([_node("h1")], now=NOW)
    rag._index_completed_document("h1", now=NOW)
    (row,) = rag.ensure_document_retention_manager().rows_for("phys")
    assert (row.owner, row.logical) == (None, "phys")


def test_an_index_failure_never_fails_the_ingest(rag: RAG, monkeypatch: pytest.MonkeyPatch) -> None:
    """The points carry the clock; a failed index write is logged and repaired at the next startup."""
    monkeypatch.setattr(
        DocumentRetentionManager, "record_ingest", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db"))
    )
    rag._note_retention_metadata([_node("h1")], now=NOW)
    rag._index_completed_document("h1", now=NOW)


def test_payload_documents_groups_points_by_document(rag: RAG) -> None:
    """Qdrant's payloads give each document's hashes, name and stamped ingest time."""
    client = QdrantClient(location=":memory:")
    client.create_collection("phys", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE))
    stamp = (NOW - 300 * DAY).timestamp()
    client.upsert(
        "phys",
        points=[
            models.PointStruct(
                id=1, vector=[1, 0], payload={"file_hash": "a", "file_name": "a.pdf", RETENTION_STAMP_KEY: stamp}
            ),
            models.PointStruct(
                id=2,
                vector=[0, 1],
                payload={"file_hash": "a", "media_file_hash": "a-clip", RETENTION_STAMP_KEY: stamp + 5},
            ),
            models.PointStruct(id=3, vector=[1, 1], payload={"file_hash": "b", "filename": "b.txt"}),
        ],
    )
    rag._qdrant_client = client
    docs = {doc.file_hash: doc for doc in rag.payload_documents("phys")}
    assert docs["a"].point_hashes == frozenset({"a", "a-clip"})
    assert (docs["a"].filename, docs["a"].ingested_at) == ("a.pdf", NOW - 300 * DAY)
    assert (docs["b"].filename, docs["b"].ingested_at) == ("b.txt", None)


def test_repair_indexes_what_qdrant_holds_and_never_resets(rag: RAG, monkeypatch: pytest.MonkeyPatch) -> None:
    """At startup every collection's unindexed documents get rows; an indexed one keeps its clock."""
    from docint.core.state.document_retention_manager import PayloadDocument

    period = RetentionPeriodState(days=365, set_at=NOW - 400 * DAY)
    rag.ensure_document_retention_manager().record_ingest(
        "phys", "phys", None, "known", "k.pdf", ["known"], now=NOW - 5 * DAY, period=period, notice_days=30
    )
    monkeypatch.setattr(RAG, "list_collections", lambda self, **k: ["phys"])
    monkeypatch.setattr(
        RAG,
        "payload_documents",
        lambda self, physical: [
            PayloadDocument("known", frozenset({"known"}), "k.pdf", None),
            PayloadDocument("restored", frozenset({"restored"}), "r.pdf", None),
        ],
    )
    monkeypatch.setattr(RAG, "_manifest_completion_times", lambda self, physical: {})
    assert rag.repair_retention_index(period, notice_days=30) == 1
    rows = {row.file_hash: row for row in rag.ensure_document_retention_manager().rows_for("phys")}
    assert rows["known"].ingested_at == NOW - 5 * DAY
    assert rows["restored"].ingested_at == period.set_at


def test_metadata_of_two_collections_never_mix(rag: RAG) -> None:
    """Two ingests at once keep each document's name and hashes to their own collection."""
    rag._note_retention_metadata([_node("h1", file_name="a.pdf", media_file_hash="clip-a")], now=NOW)
    with rag.collection_scope("other"):
        rag._note_retention_metadata([_node("h1", file_name="b.pdf")], now=NOW)
        rag._index_completed_document("h1", now=NOW)
    rag._index_completed_document("h1", now=NOW)
    (mine,) = rag.ensure_document_retention_manager().rows_for("phys")
    (theirs,) = rag.ensure_document_retention_manager().rows_for("other")
    assert (mine.filename, sorted(mine.point_hashes)) == ("a.pdf", ["clip-a", "h1"])
    assert (theirs.filename, theirs.point_hashes) == ("b.pdf", ["h1"])


def test_points_without_a_file_hash_are_reported(rag: RAG) -> None:
    """A point no document claims is never indexed, so the repair says so."""
    from loguru import logger

    client = QdrantClient(location=":memory:")
    client.create_collection("phys", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE))
    client.upsert("phys", points=[models.PointStruct(id=1, vector=[1, 0], payload={"text": "zz"})])
    rag._qdrant_client = client
    lines: list[str] = []
    sink = logger.add(lines.append, level="WARNING")
    try:
        assert rag.payload_documents("phys") == []
    finally:
        logger.remove(sink)
    assert any("1 point" in line and "file hash" in line for line in lines)


def test_repair_never_indexes_a_collection_being_deleted(rag: RAG, monkeypatch: pytest.MonkeyPatch) -> None:
    """A collection skipped by the caller, or gone by the end, keeps no rows."""
    from docint.core.state.document_retention_manager import PayloadDocument

    period = RetentionPeriodState(days=365, set_at=NOW - 400 * DAY)
    monkeypatch.setattr(RAG, "list_collections", lambda self, **k: ["phys", "busy"])
    docs = [PayloadDocument("h1", frozenset({"h1"}), "a.pdf", None)]
    monkeypatch.setattr(RAG, "payload_documents", lambda self, physical: docs)
    monkeypatch.setattr(RAG, "_manifest_completion_times", lambda self, physical: {})
    monkeypatch.setattr(RAG, "_collection_is_empty", lambda self, physical: physical == "phys", raising=False)
    exists = {"phys": False, "busy": True}
    rag.qdrant_client.collection_exists.side_effect = lambda name: exists.get(name, False)
    rag.repair_retention_index(period, notice_days=30, skip=lambda physical: physical == "busy")
    manager = rag.ensure_document_retention_manager()
    assert manager.rows_for("busy") == [] and manager.rows_for("phys") == []


def test_repair_scrolls_every_collection_once_per_process(rag: RAG, monkeypatch: pytest.MonkeyPatch) -> None:
    """After the first full repair only collections whose index write failed are read again."""
    period = RetentionPeriodState(days=365, set_at=NOW - 400 * DAY)
    read: list[str] = []
    monkeypatch.setattr(RAG, "list_collections", lambda self, **k: ["phys", "two"])
    monkeypatch.setattr(RAG, "payload_documents", lambda self, physical: read.append(physical) or [])
    monkeypatch.setattr(RAG, "_manifest_completion_times", lambda self, physical: {})
    rag.qdrant_client.collection_exists.return_value = True
    rag.repair_retention_index(period, notice_days=30)
    rag.repair_retention_index(period, notice_days=30)
    assert read == ["phys", "two"]
    monkeypatch.setattr(
        DocumentRetentionManager, "record_ingest", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db"))
    )
    rag._note_retention_metadata([_node("h9")], now=NOW)
    rag._index_completed_document("h9", now=NOW)
    rag.repair_retention_index(period, notice_days=30)
    assert read == ["phys", "two", "phys"]


def test_an_ingest_run_leaves_no_metadata_behind(rag: RAG) -> None:
    """What a run gathered for a document it never indexed is dropped, and the repair reads the collection."""
    rag._note_retention_metadata([_node("h1")], now=NOW)
    with rag.collection_scope("other"):
        rag._note_retention_metadata([_node("h2")], now=NOW)
    rag._drop_ingest_meta("phys")
    assert list(rag._ingest_doc_meta) == [("other", "h2")]
    assert rag._retention_dirty == {"phys"}
