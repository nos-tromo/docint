"""Tests for deleting one document with everything that is only its."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from llama_index.core.schema import TextNode
from llama_index.core.storage.docstore.keyval_docstore import KVDocumentStore
from qdrant_client import QdrantClient, models
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from docint.core.rag import RAG
from docint.core.retention import RetentionPeriodState
from docint.core.state.base import Base
from docint.core.state.document_retention import DocumentRetention
from docint.core.state.document_retention_manager import DocumentRetentionManager
from docint.core.storage.ingest_manifest import IngestManifest

NOW = datetime(2026, 11, 5, 12, 0, tzinfo=UTC)
PERIOD = RetentionPeriodState(days=365, set_at=NOW - timedelta(days=400))
VECTORS = models.VectorParams(size=2, distance=models.Distance.COSINE)


@pytest.fixture
def rag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> RAG:
    """A RAG on collection ``c1`` with an in-memory Qdrant, real KV store, manifest and artifacts.

    Args:
        monkeypatch: The monkeypatch fixture.
        tmp_path: Pytest-provided temporary directory.

    Returns:
        RAG: The RAG under test.
    """
    monkeypatch.setenv("PIPELINE_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    instance = RAG(qdrant_collection="c1")
    instance._qdrant_client = QdrantClient(location=":memory:")
    instance._qdrant_src_dir = tmp_path / "sources"
    instance.ingest_manifest_enabled = True
    monkeypatch.setattr(RAG, "_invalidate_ner_cache", lambda self, collection: None)
    monkeypatch.setattr(RAG, "_bump_summary_revision", lambda self, collection=None, allow_create=True: 1)
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    manager = DocumentRetentionManager(instance)
    manager._SessionMaker = sessionmaker(bind=engine)
    instance.document_retention = manager
    for name in ("c1", "c1_dockv", "c1_images", "c1_entities"):
        instance.qdrant_client.create_collection(name, vectors_config=VECTORS)
    (tmp_path / "sources" / "c1").mkdir(parents=True)
    return instance


def _point(point_id: int, **payload: Any) -> models.PointStruct:
    return models.PointStruct(id=point_id, vector=[1.0, 0.5], payload=payload)


def _source(rag: RAG, name: str) -> str:
    path = rag.qdrant_src_dir / "c1" / name
    path.write_text("zz-sentinel-content", encoding="utf-8")
    return str(path)


def _row(rag: RAG, file_hash: str, point_hashes: list[str], filename: str = "x") -> DocumentRetention:
    rag.ensure_document_retention_manager().record_ingest(
        "c1", "docs", "alice", file_hash, filename, point_hashes, now=NOW, period=PERIOD, notice_days=30
    )
    (row,) = [r for r in rag.ensure_document_retention_manager().rows_for("c1") if r.file_hash == file_hash]
    return row


def _ids(rag: RAG, collection: str) -> set[int]:
    points, _ = rag.qdrant_client.scroll(collection, limit=100, with_payload=False)
    return {int(point.id) for point in points}


def _docstore(rag: RAG) -> KVDocumentStore:
    return KVDocumentStore(kvstore=rag._build_kv_store("c1"))


def _manifest(rag: RAG, *hashes: str) -> None:
    manifest = IngestManifest(rag._ingest_manifest_path("c1"))
    for file_hash in hashes:
        manifest.mark_started("c1", file_hash)
        manifest.mark_completed("c1", file_hash)


def _artifacts(rag: RAG, *names: str, source: str = "") -> None:
    for name in names:
        directory = rag.path_config.artifacts / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "manifest.json").write_text(json.dumps({"file_path": source or f"/elsewhere/{name}.pdf"}))


def test_pdf_document_goes_with_points_nodes_file_manifest_and_artifacts(rag: RAG) -> None:
    """Its points, docstore nodes, source file, manifest row, artifacts and index row go; the neighbour stays."""
    a, b = _source(rag, "a.pdf"), _source(rag, "b.pdf")
    rag.qdrant_client.upsert("c1", [_point(1, file_hash="h1", file_path=a), _point(2, file_hash="h2", file_path=b)])
    rag.qdrant_client.upsert("c1_dockv", [_point(1, file_hash="h1"), _point(2, file_hash="h2")])
    _docstore(rag).add_documents(
        [TextNode(id_="n1", text="x", metadata={"file_hash": "h1"}), TextNode(id_="n2", text="y", metadata={"file_hash": "h2"})]
    )
    _manifest(rag, "h1", "h2")
    _artifacts(rag, "h1", source=a)
    _artifacts(rag, "h2", source=b)
    row = _row(rag, "h1", ["h1"], "a.pdf")
    _row(rag, "h2", ["h2"], "b.pdf")
    result = rag.delete_documents("c1", [row])
    assert _ids(rag, "c1") == {2}
    assert _ids(rag, "c1_dockv") == {2}
    assert set(_docstore(rag).docs) == {"n2"}
    assert not Path(a).exists() and Path(b).exists()
    assert IngestManifest(rag._ingest_manifest_path("c1")).completed_files("c1") == {"h2"}
    assert not (rag.path_config.artifacts / "h1").exists() and (rag.path_config.artifacts / "h2").exists()
    assert [r.file_hash for r in rag.ensure_document_retention_manager().rows_for("c1")] == ["h2"]
    assert (result.points, result.nodes, result.source_files, result.artifacts, result.collection_empty) == (1, 1, 1, 1, False)


def test_video_with_transcript_and_keyframes_goes_whole(rag: RAG) -> None:
    """A clip's transcript segments and its keyframes go with it."""
    rag.qdrant_client.upsert("c1", [_point(1, file_hash="tj", media_file_hash="clip"), _point(2, file_hash="other")])
    rag.qdrant_client.upsert("c1_images", [_point(10, image_id="k1", media_file_hash="clip")])
    rag.delete_documents("c1", [_row(rag, "tj", ["tj", "clip"])])
    assert _ids(rag, "c1") == {2}
    assert _ids(rag, "c1_images") == set()


def test_social_export_goes_with_posting_images(rag: RAG) -> None:
    """Images linked to the export's postings go with it."""
    rag.qdrant_client.upsert("c1", [_point(1, file_hash="dossier", posting_uuid="p1")])
    rag.qdrant_client.upsert(
        "c1_images", [_point(10, image_id="i1", posting_uuid="p1", occurrences=[{"posting_uuid": "p1"}])]
    )
    rag.delete_documents("c1", [_row(rag, "dossier", ["dossier"])])
    assert _ids(rag, "c1_images") == set()


def test_shared_image_keeps_the_other_documents_occurrence(rag: RAG) -> None:
    """An image two documents contain loses one occurrence and keeps the other, which becomes its identity."""
    rag.qdrant_client.upsert("c1", [_point(1, file_hash="h1"), _point(2, file_hash="h2")])
    rag.qdrant_client.upsert(
        "c1_images",
        [
            _point(
                10,
                image_id="img",
                source_doc_id="h1",
                source_path="/x/a.pdf",
                occurrences=[{"source_doc_id": "h1", "source_path": "/x/a.pdf"}, {"source_doc_id": "h2", "source_path": "/x/b.pdf"}],
            )
        ],
    )
    result = rag.delete_documents("c1", [_row(rag, "h1", ["h1"])])
    (point,) = rag.qdrant_client.retrieve("c1_images", ids=[10], with_payload=True)
    assert point.payload["occurrences"] == [{"source_doc_id": "h2", "source_path": "/x/b.pdf"}]
    assert (point.payload["source_doc_id"], point.payload["source_path"]) == ("h2", "/x/b.pdf")
    assert (result.image_points, result.image_occurrences) == (0, 1)


def test_shared_pdf_artifact_survives_while_another_collection_uses_it(rag: RAG) -> None:
    """Two collections holding the same PDF share its artifacts; deleting one copy keeps them."""
    rag.qdrant_client.upsert("c1", [_point(1, file_hash="h1")])
    _manifest(rag, "h1")
    other = IngestManifest(rag._ingest_manifest_path("c2"))
    (rag.qdrant_src_dir / "c2").mkdir(parents=True, exist_ok=True)
    other.mark_started("c2", "h1")
    other.mark_completed("c2", "h1")
    _artifacts(rag, "h1")
    rag.delete_documents("c1", [_row(rag, "h1", ["h1"])])
    assert (rag.path_config.artifacts / "h1").exists()


def test_source_path_outside_the_collection_is_not_removed(rag: RAG, tmp_path: Path) -> None:
    """A payload path that escapes the collection's source directory is never deleted."""
    outside = tmp_path / "outside.pdf"
    outside.write_text("keep", encoding="utf-8")
    rag.qdrant_client.upsert("c1", [_point(1, file_hash="h1", file_path=str(outside))])
    result = rag.delete_documents("c1", [_row(rag, "h1", ["h1"])])
    assert outside.exists()
    assert result.source_files == 0


def test_entities_are_dropped(rag: RAG) -> None:
    """Canonical entities have no per-document provenance, so the companion is dropped for a rebuild."""
    rag.qdrant_client.upsert("c1", [_point(1, file_hash="h1"), _point(2, file_hash="h2")])
    rag.qdrant_client.upsert("c1_entities", [_point(1, canonical_name="Alice Example")])
    result = rag.delete_documents("c1", [_row(rag, "h1", ["h1"])])
    assert not rag.qdrant_client.collection_exists("c1_entities")
    assert result.entities_dropped is True


def test_second_run_is_a_noop(rag: RAG) -> None:
    """Every step tolerates what is already gone."""
    rag.qdrant_client.upsert("c1", [_point(1, file_hash="h1"), _point(2, file_hash="h2")])
    row = _row(rag, "h1", ["h1"])
    rag.delete_documents("c1", [row])
    again = rag.delete_documents("c1", [row])
    assert (again.points, again.nodes, again.source_files) == (0, 0, 0)
    assert _ids(rag, "c1") == {2}


def test_last_document_reports_collection_empty(rag: RAG) -> None:
    """When nothing is left in the collection, the caller deletes the collection itself."""
    rag.qdrant_client.upsert("c1", [_point(1, file_hash="h1")])
    assert rag.delete_documents("c1", [_row(rag, "h1", ["h1"])]).collection_empty is True


def test_deleting_a_collection_drops_its_index_rows(rag: RAG) -> None:
    """A whole collection deleted by hand leaves no deadline behind."""
    rag.qdrant_client.upsert("c1", [_point(1, file_hash="h1")])
    _row(rag, "h1", ["h1"])
    rag.delete_collection("c1")
    assert rag.ensure_document_retention_manager().rows_for("c1") == []
