"""Tests for the document processing pipeline modules."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from docint.core.pipeline.config import PipelineConfig, load_pipeline_config
from docint.core.pipeline.models import (
    BBox,
    BlockType,
    ChunkResult,
    DocumentManifest,
    ImageResult,
    LayoutBlock,
    OCRSpan,
    PageInfo,
    PageText,
    TableResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def pipeline_config(tmp_path: Path) -> PipelineConfig:
    """Return a pipeline config pointing at a temp artifacts dir."""
    return PipelineConfig(
        text_coverage_threshold=0.01,
        pipeline_version="test-1.0.0",
        artifacts_dir=str(tmp_path / "artifacts"),
        max_retries=1,
        force_reprocess=True,
        max_workers=1,
    )


@pytest.fixture()
def sample_page_info() -> PageInfo:
    return PageInfo(
        page_index=0,
        has_text_layer=True,
        text_coverage=5.0,
        needs_ocr=False,
        width=612.0,
        height=792.0,
        status="completed",
    )


@pytest.fixture()
def sample_layout_block() -> LayoutBlock:
    return LayoutBlock(
        block_id="block-0-abc12345",
        page_index=0,
        type=BlockType.TEXT,
        bbox=BBox(x0=0, y0=0, x1=612, y1=792),
        reading_order=0,
        confidence=1.0,
        text="Hello world. This is a test document. It has multiple sentences.",
    )


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestBBox:
    def test_area(self) -> None:
        bbox = BBox(x0=0, y0=0, x1=10, y1=20)
        assert bbox.area == 200.0

    def test_area_degenerate(self) -> None:
        bbox = BBox(x0=5, y0=5, x1=5, y1=5)
        assert bbox.area == 0.0

    def test_overlaps_true(self) -> None:
        a = BBox(x0=0, y0=0, x1=10, y1=10)
        b = BBox(x0=5, y0=5, x1=15, y1=15)
        assert a.overlaps(b)
        assert b.overlaps(a)

    def test_overlaps_false(self) -> None:
        a = BBox(x0=0, y0=0, x1=5, y1=5)
        b = BBox(x0=10, y0=10, x1=20, y1=20)
        assert not a.overlaps(b)


class TestBlockType:
    def test_values(self) -> None:
        assert BlockType.TEXT.value == "text"
        assert BlockType.TABLE.value == "table"
        assert BlockType.FIGURE.value == "figure"
        assert BlockType.TITLE.value == "title"


class TestPageInfo:
    def test_default_status(self) -> None:
        p = PageInfo(
            page_index=0, has_text_layer=True, text_coverage=1.0, needs_ocr=False
        )
        assert p.status == "pending"
        assert p.error is None


class TestDocumentManifest:
    def test_defaults(self) -> None:
        m = DocumentManifest(
            doc_id="abc", file_path="/x.pdf", file_name="x.pdf",
            pipeline_version="1.0.0",
        )
        assert m.pages_total == 0
        assert m.status == "pending"


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestPipelineConfig:
    def test_load_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Clear any existing env overrides
        for key in [
            "PIPELINE_TEXT_COVERAGE_THRESHOLD",
            "PIPELINE_ARTIFACTS_DIR",
            "PIPELINE_MAX_RETRIES",
            "PIPELINE_FORCE_REPROCESS",
            "PIPELINE_MAX_WORKERS",
        ]:
            monkeypatch.delenv(key, raising=False)

        cfg = load_pipeline_config()
        assert cfg.text_coverage_threshold == 0.01
        assert cfg.max_retries == 2
        assert cfg.force_reprocess is False
        assert cfg.max_workers == 4

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PIPELINE_TEXT_COVERAGE_THRESHOLD", "0.5")
        monkeypatch.setenv("PIPELINE_FORCE_REPROCESS", "true")
        cfg = load_pipeline_config()
        assert cfg.text_coverage_threshold == 0.5
        assert cfg.force_reprocess is True


# ---------------------------------------------------------------------------
# Triage tests
# ---------------------------------------------------------------------------

class TestTriage:
    def test_digital_pdf(self, pipeline_config: PipelineConfig) -> None:
        """Pages with sufficient text should not need OCR."""
        from docint.core.pipeline.triage import triage_pdf

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "A" * 500
        mock_mediabox = MagicMock()
        mock_mediabox.width = 612.0
        mock_mediabox.height = 792.0
        mock_page.mediabox = mock_mediabox

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        with patch("docint.core.pipeline.triage.pypdf") as mock_pypdf:
            mock_pypdf.PdfReader.return_value = mock_reader
            pages = triage_pdf("/fake/doc.pdf", pipeline_config)

        assert len(pages) == 1
        assert pages[0].has_text_layer is True
        assert pages[0].needs_ocr is False
        assert pages[0].status == "completed"

    def test_scanned_pdf(self, pipeline_config: PipelineConfig) -> None:
        """Pages with no text should need OCR."""
        from docint.core.pipeline.triage import triage_pdf

        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        mock_mediabox = MagicMock()
        mock_mediabox.width = 612.0
        mock_mediabox.height = 792.0
        mock_page.mediabox = mock_mediabox

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        with patch("docint.core.pipeline.triage.pypdf") as mock_pypdf:
            mock_pypdf.PdfReader.return_value = mock_reader
            pages = triage_pdf("/fake/scan.pdf", pipeline_config)

        assert len(pages) == 1
        assert pages[0].has_text_layer is False
        assert pages[0].needs_ocr is True

    def test_mixed_pdf(self, pipeline_config: PipelineConfig) -> None:
        """A PDF with mixed pages should classify each correctly."""
        from docint.core.pipeline.triage import triage_pdf

        digital_page = MagicMock()
        digital_page.extract_text.return_value = "X" * 1000
        digital_mb = MagicMock()
        digital_mb.width = 612.0
        digital_mb.height = 792.0
        digital_page.mediabox = digital_mb

        scanned_page = MagicMock()
        scanned_page.extract_text.return_value = ""
        scanned_mb = MagicMock()
        scanned_mb.width = 612.0
        scanned_mb.height = 792.0
        scanned_page.mediabox = scanned_mb

        mock_reader = MagicMock()
        mock_reader.pages = [digital_page, scanned_page]

        with patch("docint.core.pipeline.triage.pypdf") as mock_pypdf:
            mock_pypdf.PdfReader.return_value = mock_reader
            pages = triage_pdf("/fake/mixed.pdf", pipeline_config)

        assert len(pages) == 2
        assert pages[0].needs_ocr is False
        assert pages[1].needs_ocr is True

    def test_bad_page_does_not_crash(self, pipeline_config: PipelineConfig) -> None:
        """A page that raises during extraction should be marked failed."""
        from docint.core.pipeline.triage import triage_pdf

        good_page = MagicMock()
        good_page.extract_text.return_value = "A" * 500
        good_mb = MagicMock()
        good_mb.width = 612.0
        good_mb.height = 792.0
        good_page.mediabox = good_mb

        bad_page = MagicMock()
        bad_page.extract_text.side_effect = RuntimeError("corrupt page")

        mock_reader = MagicMock()
        mock_reader.pages = [good_page, bad_page]

        with patch("docint.core.pipeline.triage.pypdf") as mock_pypdf:
            mock_pypdf.PdfReader.return_value = mock_reader
            pages = triage_pdf("/fake/bad.pdf", pipeline_config)

        assert len(pages) == 2
        assert pages[0].status == "completed"
        assert pages[1].status == "failed"
        assert pages[1].error is not None


# ---------------------------------------------------------------------------
# Chunking tests
# ---------------------------------------------------------------------------

class TestChunking:
    def test_basic_chunking(self, sample_layout_block: LayoutBlock) -> None:
        from docint.core.pipeline.chunking import chunk_document

        layout = {0: [sample_layout_block]}
        page_texts = {
            0: PageText(
                page_index=0,
                full_text=sample_layout_block.text,
                source_mix="pdf_text",
            )
        }
        chunks = chunk_document("doc123", layout, page_texts, [], [])
        assert len(chunks) >= 1
        assert chunks[0].doc_id == "doc123"
        assert chunks[0].source_mix == "pdf_text"
        assert chunks[0].section_path == []

    def test_section_path_tracking(self) -> None:
        from docint.core.pipeline.chunking import chunk_document

        title_block = LayoutBlock(
            block_id="title-0",
            page_index=0,
            type=BlockType.TITLE,
            bbox=BBox(x0=0, y0=700, x1=612, y1=792),
            reading_order=0,
            confidence=1.0,
            text="Chapter 1: Introduction",
        )
        text_block = LayoutBlock(
            block_id="text-0",
            page_index=0,
            type=BlockType.TEXT,
            bbox=BBox(x0=0, y0=0, x1=612, y1=700),
            reading_order=1,
            confidence=1.0,
            text="This is the introduction text.",
        )
        layout = {0: [title_block, text_block]}
        page_texts = {
            0: PageText(page_index=0, full_text="", source_mix="pdf_text")
        }
        chunks = chunk_document("doc456", layout, page_texts, [], [])
        # The text block chunk should have the section path from the title
        text_chunks = [c for c in chunks if "introduction text" in c.text.lower()]
        assert len(text_chunks) >= 1
        assert "Chapter 1: Introduction" in text_chunks[0].section_path

    def test_chunk_size_respected(self) -> None:
        from docint.core.pipeline.chunking import chunk_document

        long_text = ". ".join(["Sentence number " + str(i) for i in range(100)])
        block = LayoutBlock(
            block_id="long-0",
            page_index=0,
            type=BlockType.TEXT,
            bbox=BBox(x0=0, y0=0, x1=612, y1=792),
            reading_order=0,
            confidence=1.0,
            text=long_text,
        )
        layout = {0: [block]}
        page_texts = {
            0: PageText(page_index=0, full_text=long_text, source_mix="pdf_text")
        }
        chunks = chunk_document(
            "doc789", layout, page_texts, [], [], chunk_size=200
        )
        assert len(chunks) > 1
        for chunk in chunks:
            # Allow some tolerance for sentence boundaries
            assert len(chunk.text) <= 300

    def test_stable_chunk_ids(self) -> None:
        from docint.core.pipeline.chunking import chunk_document

        block = LayoutBlock(
            block_id="stable-block",
            page_index=0,
            type=BlockType.TEXT,
            bbox=BBox(x0=0, y0=0, x1=612, y1=792),
            reading_order=0,
            confidence=1.0,
            text="Deterministic content for testing.",
        )
        layout = {0: [block]}
        page_texts = {
            0: PageText(page_index=0, full_text="", source_mix="pdf_text")
        }
        c1 = chunk_document("same-doc", layout, page_texts, [], [])
        c2 = chunk_document("same-doc", layout, page_texts, [], [])
        assert len(c1) == len(c2)
        for a, b in zip(c1, c2):
            assert a.chunk_id == b.chunk_id

    def test_ocr_source_mix_propagated(self) -> None:
        from docint.core.pipeline.chunking import chunk_document

        block = LayoutBlock(
            block_id="ocr-block",
            page_index=0,
            type=BlockType.TEXT,
            bbox=BBox(x0=0, y0=0, x1=612, y1=792),
            reading_order=0,
            confidence=0.8,
            text="OCR extracted text content.",
        )
        layout = {0: [block]}
        page_texts = {
            0: PageText(page_index=0, full_text="", source_mix="ocr", confidence=0.8)
        }
        chunks = chunk_document("ocr-doc", layout, page_texts, [], [])
        assert len(chunks) >= 1
        assert chunks[0].source_mix == "ocr"


# ---------------------------------------------------------------------------
# Artifact tests
# ---------------------------------------------------------------------------

class TestArtifacts:
    def test_save_and_load_manifest(self, tmp_path: Path) -> None:
        from docint.core.pipeline.artifacts import load_manifest, save_manifest

        manifest = DocumentManifest(
            doc_id="test-doc-id",
            file_path="/some/file.pdf",
            file_name="file.pdf",
            pipeline_version="1.0.0",
            pages=[
                PageInfo(
                    page_index=0,
                    has_text_layer=True,
                    text_coverage=5.0,
                    needs_ocr=False,
                    status="completed",
                )
            ],
            pages_total=1,
            status="completed",
        )
        save_manifest(manifest, tmp_path)
        loaded = load_manifest("test-doc-id", tmp_path)
        assert loaded is not None
        assert loaded.doc_id == "test-doc-id"
        assert loaded.pages_total == 1
        assert loaded.status == "completed"
        assert len(loaded.pages) == 1
        assert loaded.pages[0].page_index == 0

    def test_save_layout(self, tmp_path: Path) -> None:
        from docint.core.pipeline.artifacts import save_layout

        blocks = [
            LayoutBlock(
                block_id="b1",
                page_index=0,
                type=BlockType.TEXT,
                bbox=BBox(x0=0, y0=0, x1=100, y1=100),
                reading_order=0,
                confidence=1.0,
                text="hello",
            )
        ]
        path = save_layout("doc1", 0, blocks, tmp_path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert len(data) == 1
        assert data[0]["block_id"] == "b1"

    def test_save_page_text(self, tmp_path: Path) -> None:
        from docint.core.pipeline.artifacts import save_page_text

        pt = PageText(
            page_index=0,
            full_text="hello world",
            source_mix="pdf_text",
            confidence=1.0,
        )
        path = save_page_text("doc1", pt, tmp_path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["full_text"] == "hello world"

    def test_save_chunks(self, tmp_path: Path) -> None:
        from docint.core.pipeline.artifacts import save_chunks

        chunks = [
            ChunkResult(
                doc_id="d1",
                chunk_id="c1",
                text="chunk text",
                page_range=[0],
                block_ids=["b1"],
                section_path=["Intro"],
                table_ids=[],
                image_ids=[],
                source_mix="pdf_text",
            )
        ]
        path = save_chunks("d1", chunks, tmp_path)
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["chunk_id"] == "c1"

    def test_save_table(self, tmp_path: Path) -> None:
        from docint.core.pipeline.artifacts import save_table

        table = TableResult(
            table_id="t1",
            page_index=0,
            bbox=BBox(x0=0, y0=0, x1=100, y1=50),
            raw_text="col1,col2\na,b",
            confidence=0.9,
        )
        path = save_table("doc1", table, tmp_path)
        assert path.exists()

    def test_save_image_metadata(self, tmp_path: Path) -> None:
        from docint.core.pipeline.artifacts import save_image_metadata

        image = ImageResult(
            image_id="img1",
            page_index=0,
            bbox=BBox(x0=0, y0=0, x1=100, y1=100),
        )
        path = save_image_metadata("doc1", image, tmp_path)
        assert path.exists()


# ---------------------------------------------------------------------------
# Extraction tests
# ---------------------------------------------------------------------------

class TestExtraction:
    def test_extract_tables_from_layout(self) -> None:
        from docint.core.pipeline.extraction import extract_tables

        layout = {
            0: [
                LayoutBlock(
                    block_id="tb1",
                    page_index=0,
                    type=BlockType.TABLE,
                    bbox=BBox(x0=0, y0=0, x1=100, y1=50),
                    reading_order=0,
                    confidence=0.9,
                    text="col1|col2\na|b",
                ),
                LayoutBlock(
                    block_id="txt1",
                    page_index=0,
                    type=BlockType.TEXT,
                    bbox=BBox(x0=0, y0=50, x1=100, y1=100),
                    reading_order=1,
                    confidence=1.0,
                    text="Regular text",
                ),
            ]
        }
        tables = extract_tables(layout)
        assert len(tables) == 1
        assert tables[0].page_index == 0
        assert "col1" in tables[0].raw_text

    def test_extract_images_from_layout(self) -> None:
        from docint.core.pipeline.extraction import extract_images

        layout = {
            0: [
                LayoutBlock(
                    block_id="fig1",
                    page_index=0,
                    type=BlockType.FIGURE,
                    bbox=BBox(x0=0, y0=0, x1=200, y1=200),
                    reading_order=0,
                    confidence=0.85,
                    text="",
                ),
            ]
        }
        images = extract_images(layout)
        assert len(images) == 1
        assert images[0].page_index == 0
        assert images[0].metadata["confidence"] == 0.85


# ---------------------------------------------------------------------------
# OCR tests
# ---------------------------------------------------------------------------

class TestOCR:
    def test_build_page_text_pdf_only(
        self, sample_page_info: PageInfo, sample_layout_block: LayoutBlock
    ) -> None:
        from docint.core.pipeline.ocr import build_page_text

        result = build_page_text(sample_page_info, [sample_layout_block], [])
        assert result.source_mix == "pdf_text"
        assert "Hello world" in result.full_text
        assert result.confidence == 1.0

    def test_build_page_text_ocr_only(self, sample_page_info: PageInfo) -> None:
        from docint.core.pipeline.ocr import build_page_text

        ocr_spans = [
            OCRSpan(
                text="OCR text",
                bbox=BBox(x0=0, y0=0, x1=100, y1=100),
                confidence=0.75,
                source="ocr",
            )
        ]
        result = build_page_text(sample_page_info, [], ocr_spans)
        assert result.source_mix == "ocr"
        assert "OCR text" in result.full_text
        assert result.confidence == 0.75

    def test_build_page_text_mixed(
        self, sample_page_info: PageInfo, sample_layout_block: LayoutBlock
    ) -> None:
        from docint.core.pipeline.ocr import build_page_text

        ocr_spans = [
            OCRSpan(
                text="Additional OCR text",
                bbox=BBox(x0=0, y0=0, x1=100, y1=100),
                confidence=0.8,
                source="ocr",
            )
        ]
        result = build_page_text(sample_page_info, [sample_layout_block], ocr_spans)
        assert result.source_mix == "mixed"
        assert "Hello world" in result.full_text
        assert "Additional OCR text" in result.full_text


# ---------------------------------------------------------------------------
# Orchestrator tests
# ---------------------------------------------------------------------------

class TestOrchestrator:
    def test_process_with_mocked_pdf(
        self, pipeline_config: PipelineConfig, tmp_path: Path
    ) -> None:
        from docint.core.pipeline.orchestrator import DocumentPipelineOrchestrator

        # Create a dummy file for hashing
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy content for hashing")

        # Mock pypdf
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Test document content. Second sentence."
        mock_mb = MagicMock()
        mock_mb.width = 612.0
        mock_mb.height = 792.0
        mock_mb.left = 0.0
        mock_mb.bottom = 0.0
        mock_mb.right = 612.0
        mock_mb.top = 792.0
        mock_page.mediabox = mock_mb

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        orch = DocumentPipelineOrchestrator(config=pipeline_config)

        with patch("docint.core.pipeline.triage.pypdf") as mock_triage_pypdf, \
             patch("docint.core.pipeline.layout.pypdf") as mock_layout_pypdf, \
             patch("docint.core.pipeline.ocr.pypdf") as mock_ocr_pypdf:
            mock_triage_pypdf.PdfReader.return_value = mock_reader
            mock_layout_pypdf.PdfReader.return_value = mock_reader
            mock_ocr_pypdf.PdfReader.return_value = mock_reader

            manifest = orch.process(pdf_file)

        assert manifest.status == "completed"
        assert manifest.pages_total == 1
        assert manifest.pages_failed == 0

        # Check artifacts were created
        from docint.utils.hashing import compute_file_hash
        doc_id = compute_file_hash(pdf_file)
        artifacts_dir = Path(pipeline_config.artifacts_dir)
        assert (artifacts_dir / doc_id / "manifest.json").exists()

    def test_idempotent_rerun(
        self, pipeline_config: PipelineConfig, tmp_path: Path
    ) -> None:
        """Second run should reuse artifacts when pipeline version matches."""
        from docint.core.pipeline.orchestrator import DocumentPipelineOrchestrator

        # Disable force to test idempotency
        config = PipelineConfig(
            text_coverage_threshold=pipeline_config.text_coverage_threshold,
            pipeline_version=pipeline_config.pipeline_version,
            artifacts_dir=pipeline_config.artifacts_dir,
            max_retries=pipeline_config.max_retries,
            force_reprocess=False,
            max_workers=pipeline_config.max_workers,
        )

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 idempotent test content")

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Idempotent test."
        mock_mb = MagicMock()
        mock_mb.width = 612.0
        mock_mb.height = 792.0
        mock_mb.left = 0.0
        mock_mb.bottom = 0.0
        mock_mb.right = 612.0
        mock_mb.top = 792.0
        mock_page.mediabox = mock_mb

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        orch = DocumentPipelineOrchestrator(config=config)

        # First run
        with patch("docint.core.pipeline.triage.pypdf") as m1, \
             patch("docint.core.pipeline.layout.pypdf") as m2, \
             patch("docint.core.pipeline.ocr.pypdf") as m3:
            m1.PdfReader.return_value = mock_reader
            m2.PdfReader.return_value = mock_reader
            m3.PdfReader.return_value = mock_reader
            manifest1 = orch.process(pdf_file)

        assert manifest1.status == "completed"

        # Second run — should skip processing
        manifest2 = orch.process(pdf_file)
        assert manifest2.status == "completed"
        assert manifest2.doc_id == manifest1.doc_id

    def test_page_failure_isolation(
        self, pipeline_config: PipelineConfig, tmp_path: Path
    ) -> None:
        """A failing page should not crash the whole document."""
        from docint.core.pipeline.orchestrator import DocumentPipelineOrchestrator

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 failure isolation test")

        good_page = MagicMock()
        good_page.extract_text.return_value = "Good page content."
        good_mb = MagicMock()
        good_mb.width = 612.0
        good_mb.height = 792.0
        good_mb.left = 0.0
        good_mb.bottom = 0.0
        good_mb.right = 612.0
        good_mb.top = 792.0
        good_page.mediabox = good_mb

        bad_page = MagicMock()
        bad_page.extract_text.side_effect = RuntimeError("corrupt")

        mock_reader = MagicMock()
        mock_reader.pages = [good_page, bad_page]

        orch = DocumentPipelineOrchestrator(config=pipeline_config)

        with patch("docint.core.pipeline.triage.pypdf") as m1, \
             patch("docint.core.pipeline.layout.pypdf") as m2, \
             patch("docint.core.pipeline.ocr.pypdf") as m3:
            m1.PdfReader.return_value = mock_reader
            m2.PdfReader.return_value = mock_reader
            m3.PdfReader.return_value = mock_reader
            manifest = orch.process(pdf_file)

        assert manifest.status == "completed"
        assert manifest.pages_total == 2
        # At least one page should be processed (the good one)
        assert any(p.status == "completed" for p in manifest.pages)

    def test_retry_logic(
        self, pipeline_config: PipelineConfig, tmp_path: Path
    ) -> None:
        """Stages should retry on transient failures."""
        from docint.core.pipeline.orchestrator import DocumentPipelineOrchestrator

        orch = DocumentPipelineOrchestrator(config=pipeline_config)

        call_count = 0

        def flaky_fn():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient error")
            return "success"

        result = orch._run_with_retry("test-stage", flaky_fn)
        assert result == "success"
        assert call_count == 2
