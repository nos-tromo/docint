"""Tests for the core PDF pipeline reader and default RAG ingestion path."""

from __future__ import annotations

import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import docint.core.rag as rag_module
from docint.core.readers.documents import CorePDFPipelineReader
from docint.core.rag import RAG
from docint.utils.hashing import compute_file_hash


def test_reader_build_nodes_sets_expected_metadata(tmp_path: Path) -> None:
    """Core-pipeline chunks should map to query/preview-friendly metadata."""
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

    chunks = [
        {
            "chunk_id": "chunk-1",
            "text": "Hello from core pipeline.",
            "page_range": [0],
            "block_ids": ["b1"],
            "section_path": ["Intro"],
            "table_ids": [],
            "image_ids": [],
            "source_mix": "pdf_text",
            "bbox_refs": [],
            "metadata": {"custom": "value"},
        }
    ]

    docs, nodes = CorePDFPipelineReader._build_nodes(
        file_path=pdf_path,
        doc_id="abc123",
        pipeline_version="1.0.0",
        chunks=chunks,
    )

    assert len(docs) == 1
    assert len(nodes) == 1
    meta = nodes[0].metadata
    assert meta["file_hash"] == "abc123"
    assert meta["filename"] == "sample.pdf"
    assert meta["source"] == "document"
    assert meta["chunk_id"] == "chunk-1"
    assert meta["point_id"] == nodes[0].node_id
    assert meta["page_number"] == 1
    assert meta["origin"]["file_hash"] == "abc123"
    assert meta["custom"] == "value"
    assert str(uuid.UUID(nodes[0].node_id)) == nodes[0].node_id


def test_reader_skips_existing_hashes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Already-ingested PDFs should be skipped before pipeline processing."""
    pdf_path = tmp_path / "already.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    existing_hash = compute_file_hash(pdf_path)

    reader = CorePDFPipelineReader(data_dir=tmp_path)

    monkeypatch.setattr(
        CorePDFPipelineReader,
        "_iter_pdf_files",
        staticmethod(lambda _: [pdf_path]),
    )

    class FakeOrchestrator:
        """Raise if process is called; existing hash path should skip it."""

        def __init__(self) -> None:
            """Initialise with a stub artifacts directory."""
            self.config = SimpleNamespace(artifacts_dir=str(tmp_path / "artifacts"))

        def process(self, file_path: str | Path) -> SimpleNamespace:
            """Raise if called; existing hashes should prevent invocation.

            Args:
                file_path: Path to the PDF file.

            Returns:
                Never returns.

            Raises:
                AssertionError: Always, since this should not be reached.
            """
            raise AssertionError("process() should not run for existing hashes")

    monkeypatch.setattr(
        "docint.core.readers.documents.reader.DocumentPipelineOrchestrator",
        FakeOrchestrator,
    )

    batches = list(reader.build({existing_hash}))
    assert batches == []
    assert reader.discovered_hashes == {existing_hash}


def test_reader_applies_ner_metadata(tmp_path: Path) -> None:
    """Core reader should enrich nodes when an entity extractor is provided."""
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    chunks = [
        {
            "chunk_id": "chunk-1",
            "text": "Ada Lovelace worked with Charles Babbage.",
            "page_range": [0],
        }
    ]
    _, nodes = CorePDFPipelineReader._build_nodes(
        file_path=pdf_path,
        doc_id="abc123",
        pipeline_version="1.0.0",
        chunks=chunks,
    )

    reader = CorePDFPipelineReader(
        data_dir=tmp_path,
        entity_extractor=lambda text: (
            [{"text": "Ada Lovelace", "type": "person"}],
            [
                {
                    "head": "Ada Lovelace",
                    "tail": "Charles Babbage",
                    "label": "worked_with",
                }
            ],
        ),
        ner_max_workers=1,
    )
    reader._apply_ner(nodes)

    assert nodes[0].metadata["entities"] == [{"text": "Ada Lovelace", "type": "person"}]
    assert nodes[0].metadata["relations"] == [
        {"head": "Ada Lovelace", "tail": "Charles Babbage", "label": "worked_with"}
    ]


def test_reader_ingests_extracted_images_via_shared_service(tmp_path: Path) -> None:
    """Core reader should route pipeline-extracted images through shared service.

    Args:
        tmp_path (Path): Temporary directory path for the test.
    """
    doc_id = "doc-hash-1"
    artifacts_dir = tmp_path / "artifacts"
    images_dir = artifacts_dir / doc_id / "images"
    images_dir.mkdir(parents=True)

    image_path = images_dir / "image-1.png"
    image_path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc``\x00\x00\x00\x04"
        b"\x00\x01\xf3\x17\xbd\xa5\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    (images_dir / "image-1.json").write_text(
        (
            "{"
            '"image_id":"image-1",'
            '"page_index":2,'
            '"bbox":{"x0":1,"y0":2,"x1":3,"y1":4},'
            f'"image_path":"{image_path}",'
            '"metadata":{"block_id":"figure-2-a","confidence":0.91}'
            "}"
        ),
        encoding="utf-8",
    )

    calls: list[tuple[Any, Any]] = []

    class RecordingService:
        def ingest_image(self, asset, *, context):
            calls.append((asset, context))
            return SimpleNamespace(status="stored", error=None)

    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    reader = CorePDFPipelineReader(
        data_dir=tmp_path,
        source_collection="att-2",
        image_ingestion_service=RecordingService(),  # type: ignore[arg-type]
    )
    reader._ingest_pipeline_images(
        file_path=pdf_path,
        doc_id=doc_id,
        artifacts_dir=artifacts_dir,
    )

    assert len(calls) == 1
    asset, context = calls[0]
    assert asset.source_type == "document"
    assert asset.source_doc_id == doc_id
    assert asset.source_path == str(pdf_path)
    assert asset.page_number == 3
    assert asset.bbox == {"x0": 1.0, "y0": 2.0, "x1": 3.0, "y1": 4.0}
    assert asset.extra_metadata["pipeline_image_id"] == "image-1"
    assert context.source_collection == "att-2"


def test_rag_excludes_pdfs_from_legacy_ingestion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """RAG should pass discovered PDF hashes into legacy pipeline filters."""

    class FakeDocStore:
        """No-op document store for testing."""

        def add_documents(self, nodes, allow_update=True) -> None:
            """Accept documents without storing them."""
            return None

    class FakeIndex:
        """Minimal vector store index stub."""

        def __init__(self, **kwargs) -> None:
            """Initialise with a fake doc store."""
            self.docstore = FakeDocStore()

        def insert_nodes(self, nodes) -> None:
            """Accept nodes without indexing them."""
            return None

    class FakeCoreReader:
        """Core reader stub that reports pre-set discovered hashes."""

        def __init__(
            self,
            data_dir: Path,
            entity_extractor=None,
            ner_max_workers: int = 1,
            source_collection: str | None = None,
            image_ingestion_service=None,
        ) -> None:
            """Initialise and discard all arguments."""
            _ = (
                data_dir,
                entity_extractor,
                ner_max_workers,
                source_collection,
                image_ingestion_service,
            )
            self.discovered_hashes = {"pdf-hash-1"}

        def build(self, existing_hashes, progress_callback=None):
            """Yield nothing; the stub reader produces no batches."""
            if False:
                yield
            return

    class FakeLegacyPipeline:
        """Legacy ingestion pipeline stub that records seen hashes."""

        def __init__(self) -> None:
            """Initialise with empty state."""
            self.dir_reader = None
            self.seen_hashes: set[str] | None = None
            self.entity_extractor = None
            self.ner_max_workers = 1

        def build(self, existing_hashes):
            """Record the hashes passed into the legacy pipeline."""
            self.seen_hashes = set(existing_hashes)
            if False:
                yield
            return

    fake_pipeline = FakeLegacyPipeline()

    monkeypatch.setattr(rag_module, "VectorStoreIndex", FakeIndex)
    monkeypatch.setattr(rag_module, "CorePDFPipelineReader", FakeCoreReader)
    monkeypatch.setattr(RAG, "_prepare_sources_dir", lambda self, path: path)
    monkeypatch.setattr(RAG, "_vector_store", lambda self: object())
    monkeypatch.setattr(RAG, "_storage_context", lambda self, vector_store: object())
    monkeypatch.setattr(RAG, "_get_existing_file_hashes", lambda self: set())
    monkeypatch.setattr(
        RAG,
        "_build_ingestion_pipeline",
        lambda self, progress_callback=None: fake_pipeline,
    )
    monkeypatch.setattr(
        RAG,
        "embed_model",
        property(lambda self: object()),
    )

    rag = RAG(qdrant_collection="test")
    rag.ingest_docs(tmp_path, build_query_engine=False)

    assert fake_pipeline.seen_hashes is not None
    assert "pdf-hash-1" in fake_pipeline.seen_hashes


def test_reader_yields_and_ingests_images_when_no_text_chunks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Image-only PDFs (e.g. screenshots) must still ingest images and yield.

    Previously, the reader's ``build`` method would ``continue`` past
    image ingestion when no text chunks were produced, silently
    dropping the document and its images.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture to override environment.
        tmp_path (Path): Temporary directory path for the test.
    """
    pdf_path = tmp_path / "screenshot.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    doc_id = compute_file_hash(pdf_path)

    artifacts_dir = tmp_path / "artifacts"
    images_dir = artifacts_dir / doc_id / "images"
    images_dir.mkdir(parents=True)

    # Write a 1x1 PNG and its JSON sidecar so image ingestion finds it.
    image_path = images_dir / "img-0.png"
    image_path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc``\x00\x00\x00\x04"
        b"\x00\x01\xf3\x17\xbd\xa5\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    (images_dir / "img-0.json").write_text(
        '{"image_id":"img-0","page_index":0,"bbox":{"x0":0,"y0":0,"x1":100,"y1":100},'
        f'"image_path":"{image_path}","metadata":{{}}}}'
        "",
        encoding="utf-8",
    )

    # Write an empty chunks file (no text chunks).
    chunks_path = artifacts_dir / doc_id / "chunks.jsonl"
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_path.write_text("")

    # Fake manifest returned by the orchestrator.
    manifest = SimpleNamespace(
        doc_id=doc_id,
        status="completed",
        pipeline_version="1.0.0",
        images_found=1,
        error=None,
    )

    class FakeOrchestrator:
        """Pipeline orchestrator stub that returns a pre-built manifest."""

        def __init__(self) -> None:
            """Initialise with a stub artifacts directory."""
            self.config = SimpleNamespace(artifacts_dir=str(artifacts_dir))

        def process(self, _fp: Any) -> SimpleNamespace:
            """Return the pre-built manifest.

            Args:
                _fp: File path (unused).

            Returns:
                A ``SimpleNamespace`` manifest.
            """
            return manifest

    monkeypatch.setattr(
        "docint.core.readers.documents.reader.DocumentPipelineOrchestrator",
        FakeOrchestrator,
    )
    monkeypatch.setattr(
        CorePDFPipelineReader,
        "_iter_pdf_files",
        staticmethod(lambda _: [pdf_path]),
    )

    image_calls: list[tuple[Any, Any]] = []

    class RecordingImageService:
        """Image service stub that records ingest calls."""

        def ingest_image(self, asset: Any, *, context: Any) -> SimpleNamespace:
            """Record the ingest call and return a success status.

            Args:
                asset: The image asset to ingest.
                context: Ingestion context with collection info.

            Returns:
                A ``SimpleNamespace`` with ``status='stored'``.
            """
            image_calls.append((asset, context))
            return SimpleNamespace(status="stored", error=None)

    reader = CorePDFPipelineReader(
        data_dir=tmp_path,
        source_collection="test-col",
        image_ingestion_service=RecordingImageService(),  # type: ignore[arg-type]
    )
    batches = list(reader.build(existing_hashes=set()))

    # The reader should still yield one batch (with empty nodes).
    assert len(batches) == 1
    docs, nodes, returned_hash = batches[0]
    assert returned_hash == doc_id
    assert nodes == []

    # The image service should have been invoked for the extracted image.
    assert len(image_calls) == 1

    # The hash should be tracked so subsequent runs skip this file.
    assert doc_id in reader.discovered_hashes
