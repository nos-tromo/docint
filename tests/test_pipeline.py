"""Tests for the document processing pipeline modules."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from docint.core.readers.documents.artifacts import (
    load_manifest,
    save_chunks,
    save_image_metadata,
    save_layout,
    save_manifest,
    save_page_text,
    save_table,
)
from docint.core.readers.documents.chunking import chunk_document
from docint.utils.env_cfg import PipelineConfig, load_pipeline_config
from docint.core.readers.documents.extraction import extract_images, extract_tables
from docint.core.readers.documents.layout import (
    PypdfLayoutAnalyzer,
    _extract_image_bboxes_from_stream,
    _find_table_end,
    _multiply_matrices,
)
from docint.core.readers.documents.models import (
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
from docint.core.readers.documents.ocr import (
    build_page_text,
    extract_text_for_pages,
)
from docint.core.readers.documents.orchestrator import (
    DocumentPipelineOrchestrator,
)
from docint.core.readers.documents.triage import triage_pdf
from docint.utils.hashing import compute_file_hash

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def pipeline_config(tmp_path: Path) -> PipelineConfig:
    """Return a pipeline config pointing at a temp artifacts dir.

    Args:
        tmp_path (Path): Temporary directory path for the test.I
    """
    return PipelineConfig(
        text_coverage_threshold=0.01,
        pipeline_version="test-1.0.0",
        artifacts_dir=str(tmp_path / "artifacts"),
        max_retries=1,
        force_reprocess=True,
        max_workers=1,
        enable_vision_ocr=False,
        vision_ocr_timeout=60.0,
        vision_ocr_max_retries=1,
        vision_ocr_max_image_dimension=1024,
        vision_ocr_max_tokens=4096,
    )


@pytest.fixture()
def sample_page_info() -> PageInfo:
    """Return a sample completed PageInfo for a digital page."""
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
    """Return a sample full-page TEXT layout block."""
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
    """Tests for bounding-box geometry helpers."""

    def test_area(self) -> None:
        """Positive-area box returns correct area."""
        bbox = BBox(x0=0, y0=0, x1=10, y1=20)
        assert bbox.area == 200.0

    def test_area_degenerate(self) -> None:
        """Zero-size box returns zero area."""
        bbox = BBox(x0=5, y0=5, x1=5, y1=5)
        assert bbox.area == 0.0

    def test_overlaps_true(self) -> None:
        """Overlapping boxes report True symmetrically."""
        a = BBox(x0=0, y0=0, x1=10, y1=10)
        b = BBox(x0=5, y0=5, x1=15, y1=15)
        assert a.overlaps(b)
        assert b.overlaps(a)

    def test_overlaps_false(self) -> None:
        """Disjoint boxes report False."""
        a = BBox(x0=0, y0=0, x1=5, y1=5)
        b = BBox(x0=10, y0=10, x1=20, y1=20)
        assert not a.overlaps(b)


class TestBlockType:
    """Tests for the BlockType enum values."""

    def test_values(self) -> None:
        """All expected block-type string values are present."""
        assert BlockType.TEXT.value == "text"
        assert BlockType.TABLE.value == "table"
        assert BlockType.FIGURE.value == "figure"
        assert BlockType.TITLE.value == "title"


class TestPageInfo:
    """Tests for PageInfo dataclass defaults."""

    def test_default_status(self) -> None:
        """New PageInfo should default to 'pending' with no error."""
        p = PageInfo(
            page_index=0, has_text_layer=True, text_coverage=1.0, needs_ocr=False
        )
        assert p.status == "pending"
        assert p.error is None


class TestDocumentManifest:
    """Tests for DocumentManifest dataclass defaults."""

    def test_defaults(self) -> None:
        """New manifest should default to zero pages and 'pending' status."""
        m = DocumentManifest(
            doc_id="abc",
            file_path="/x.pdf",
            file_name="x.pdf",
            pipeline_version="1.0.0",
        )
        assert m.pages_total == 0
        assert m.status == "pending"


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestPipelineConfig:
    """Tests for pipeline configuration loading and env overrides."""

    def test_load_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default config values should match documented defaults."""
        # Clear any existing env overrides
        for key in [
            "PIPELINE_TEXT_COVERAGE_THRESHOLD",
            "PIPELINE_ARTIFACTS_DIR",
            "PIPELINE_VERSION",
            "PIPELINE_MAX_RETRIES",
            "PIPELINE_FORCE_REPROCESS",
            "PIPELINE_MAX_WORKERS",
        ]:
            monkeypatch.delenv(key, raising=False)

        cfg = load_pipeline_config()
        assert cfg.text_coverage_threshold == 0.01
        assert cfg.pipeline_version == "1.0.0"
        assert cfg.max_retries == 2
        assert cfg.force_reprocess is False
        assert cfg.max_workers == 4

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Environment variables should override default config values."""
        monkeypatch.setenv("PIPELINE_TEXT_COVERAGE_THRESHOLD", "0.5")
        monkeypatch.setenv("PIPELINE_FORCE_REPROCESS", "true")
        monkeypatch.setenv("PIPELINE_VERSION", "2.1.0")
        cfg = load_pipeline_config()
        assert cfg.text_coverage_threshold == 0.5
        assert cfg.force_reprocess is True
        assert cfg.pipeline_version == "2.1.0"

    def test_empty_pipeline_version_falls_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty ``PIPELINE_VERSION`` should fall back to default version."""
        monkeypatch.setenv("PIPELINE_VERSION", "   ")
        cfg = load_pipeline_config(default_pipeline_version="9.9.9")
        assert cfg.pipeline_version == "9.9.9"

    def test_artifacts_dir_from_env(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """PIPELINE_ARTIFACTS_DIR env var should propagate into PipelineConfig.

        Args:
            monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture for env manipulation.
            tmp_path (Path): The temporary path fixture for creating test directories.
        """
        custom = str(tmp_path / "custom-artifacts")
        monkeypatch.setenv("PIPELINE_ARTIFACTS_DIR", custom)
        cfg = load_pipeline_config()
        assert cfg.artifacts_dir == custom


# ---------------------------------------------------------------------------
# Triage tests
# ---------------------------------------------------------------------------


class TestTriage:
    """Tests for the PDF triage stage (digital, scanned, mixed detection)."""

    def test_digital_pdf(self, pipeline_config: PipelineConfig) -> None:
        """Pages with sufficient text should not need OCR.

        Args:
            pipeline_config (PipelineConfig): The pipeline configuration fixture.
        """
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "A" * 500
        mock_mediabox = MagicMock()
        mock_mediabox.width = 612.0
        mock_mediabox.height = 792.0
        mock_page.mediabox = mock_mediabox

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        with patch("docint.core.readers.documents.triage.pypdf") as mock_pypdf:
            mock_pypdf.PdfReader.return_value = mock_reader
            pages = triage_pdf("/fake/doc.pdf", pipeline_config)

        assert len(pages) == 1
        assert pages[0].has_text_layer is True
        assert pages[0].needs_ocr is False
        assert pages[0].status == "completed"

    def test_scanned_pdf(self, pipeline_config: PipelineConfig) -> None:
        """Pages with no text should need OCR.

        Args:
            pipeline_config (PipelineConfig): The pipeline configuration fixture.
        """
        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        mock_mediabox = MagicMock()
        mock_mediabox.width = 612.0
        mock_mediabox.height = 792.0
        mock_page.mediabox = mock_mediabox

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        with patch("docint.core.readers.documents.triage.pypdf") as mock_pypdf:
            mock_pypdf.PdfReader.return_value = mock_reader
            pages = triage_pdf("/fake/scan.pdf", pipeline_config)

        assert len(pages) == 1
        assert pages[0].has_text_layer is False
        assert pages[0].needs_ocr is True

    def test_mixed_pdf(self, pipeline_config: PipelineConfig) -> None:
        """A PDF with mixed pages should classify each correctly.

        Args:
            pipeline_config (PipelineConfig): The pipeline configuration fixture.
        """
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

        with patch("docint.core.readers.documents.triage.pypdf") as mock_pypdf:
            mock_pypdf.PdfReader.return_value = mock_reader
            pages = triage_pdf("/fake/mixed.pdf", pipeline_config)

        assert len(pages) == 2
        assert pages[0].needs_ocr is False
        assert pages[1].needs_ocr is True

    def test_bad_page_does_not_crash(self, pipeline_config: PipelineConfig) -> None:
        """A page that raises during extraction should be marked failed.

        Args:
            pipeline_config (PipelineConfig): The pipeline configuration fixture.
        """
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

        with patch("docint.core.readers.documents.triage.pypdf") as mock_pypdf:
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
    """Tests for the document chunking logic."""

    def test_basic_chunking(self, sample_layout_block: LayoutBlock) -> None:
        """Single-block layout produces at least one chunk with correct metadata."""
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
        """Chunks following a TITLE block should carry its section path."""
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
        page_texts = {0: PageText(page_index=0, full_text="", source_mix="pdf_text")}
        chunks = chunk_document("doc456", layout, page_texts, [], [])
        # The text block chunk should have the section path from the title
        text_chunks = [c for c in chunks if "introduction text" in c.text.lower()]
        assert len(text_chunks) >= 1
        assert "Chapter 1: Introduction" in text_chunks[0].section_path

    def test_chunk_size_respected(self) -> None:
        """Long text should be split so no chunk exceeds the size limit."""
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
        chunks = chunk_document("doc789", layout, page_texts, [], [], chunk_size=200)
        assert len(chunks) > 1
        for chunk in chunks:
            # Allow some tolerance for sentence boundaries
            assert len(chunk.text) <= 300

    def test_stable_chunk_ids(self) -> None:
        """Identical inputs should produce deterministic chunk IDs."""
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
        page_texts = {0: PageText(page_index=0, full_text="", source_mix="pdf_text")}
        c1 = chunk_document("same-doc", layout, page_texts, [], [])
        c2 = chunk_document("same-doc", layout, page_texts, [], [])
        assert len(c1) == len(c2)
        for a, b in zip(c1, c2):
            assert a.chunk_id == b.chunk_id

    def test_ocr_source_mix_propagated(self) -> None:
        """Chunks from OCR pages should carry source_mix='ocr'."""
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

    def test_figure_only_page_with_ocr_text_produces_chunks(self) -> None:
        """A page with only FIGURE blocks should chunk page_text.full_text via fallback."""
        figure_block = LayoutBlock(
            block_id="fig-0",
            page_index=0,
            type=BlockType.FIGURE,
            bbox=BBox(x0=0, y0=0, x1=612, y1=792),
            reading_order=0,
            confidence=0.9,
            text="",
        )
        text_block = LayoutBlock(
            block_id="ocr-text-0-synth",
            page_index=0,
            type=BlockType.TEXT,
            bbox=BBox(x0=0, y0=0, x1=612, y1=792),
            reading_order=1,
            confidence=0.7,
            text="Vision OCR extracted text from scanned page.",
        )
        layout = {0: [figure_block, text_block]}
        page_texts = {
            0: PageText(
                page_index=0,
                full_text="Vision OCR extracted text from scanned page.",
                source_mix="ocr",
                confidence=0.7,
            )
        }
        chunks = chunk_document("scan-doc", layout, page_texts, [], [])
        assert len(chunks) >= 1
        assert "Vision OCR" in chunks[0].text
        assert chunks[0].source_mix == "ocr"


# ---------------------------------------------------------------------------
# Artifact tests
# ---------------------------------------------------------------------------


class TestArtifacts:
    """Tests for artifact serialization and deserialization."""

    def test_save_and_load_manifest(self, tmp_path: Path) -> None:
        """Round-trip save/load of a DocumentManifest preserves all fields."""
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
        """Saved layout JSON should contain the serialized block."""
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
        """Saved page text JSON should preserve the full_text value."""
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
        """Chunks should be saved as one JSONL line per chunk."""
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
        """Saved table metadata file should be created on disk."""
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
        """Saved image metadata file should be created on disk."""
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
    """Tests for table and image extraction from layout blocks."""

    def test_extract_tables_from_layout(self) -> None:
        """TABLE blocks should be extracted; TEXT blocks should be ignored."""
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
        """FIGURE blocks should be extracted with confidence in metadata."""
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
# Layout analysis tests
# ---------------------------------------------------------------------------


class TestLayoutAnalysis:
    """Tests for PypdfLayoutAnalyzer image and table detection logic."""

    def test_detect_images_creates_figure_blocks(self) -> None:
        """Pages with embedded images should produce FIGURE blocks."""
        mock_page = MagicMock()
        mock_mb = MagicMock()
        mock_mb.left = 0.0
        mock_mb.bottom = 0.0
        mock_mb.right = 612.0
        mock_mb.top = 792.0
        mock_page.mediabox = mock_mb
        mock_page.extract_text.return_value = "Some text on the page."

        # Simulate XObject with an /Image
        mock_image_obj = MagicMock()
        mock_image_obj.get.side_effect = lambda k, d="": (
            "/Image" if k == "/Subtype" else d
        )
        mock_image_obj.get_object.return_value = mock_image_obj

        mock_xobj_dict = {"/Im1": mock_image_obj}
        mock_xobj = MagicMock()
        mock_xobj.get_object.return_value = mock_xobj_dict

        mock_resources = MagicMock()
        mock_resources.get.side_effect = lambda k, d=None: (
            mock_xobj if k == "/XObject" else d
        )
        mock_page.get.side_effect = lambda k, d=None: (
            mock_resources if k == "/Resources" else (None if k == "/Contents" else d)
        )

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        with patch("docint.core.readers.documents.layout.pypdf") as mock_pypdf:
            mock_pypdf.PdfReader.return_value = mock_reader
            analyzer = PypdfLayoutAnalyzer("/fake/doc.pdf")
            blocks = analyzer.analyze_page(0)

        figure_blocks = [b for b in blocks if b.type == BlockType.FIGURE]
        text_blocks = [b for b in blocks if b.type == BlockType.TEXT]
        assert len(figure_blocks) >= 1
        assert len(text_blocks) >= 1  # remaining text

    def test_detect_tables_via_caption(self) -> None:
        """Text containing 'Table N:' captions should produce TABLE blocks."""
        table_text = (
            "Some introductory text about the experiment.\n"
            "Table 1: Results summary\n"
            "Model    Accuracy   F1\n"
            "BERT     89.3       88.1\n"
            "GPT-2    91.0       90.5\n"
            "\n"
            "The results show clear improvement in accuracy."
        )

        mock_page = MagicMock()
        mock_mb = MagicMock()
        mock_mb.left = 0.0
        mock_mb.bottom = 0.0
        mock_mb.right = 612.0
        mock_mb.top = 792.0
        mock_page.mediabox = mock_mb
        mock_page.extract_text.return_value = table_text
        # No images
        mock_resources = MagicMock()
        mock_resources.get.return_value = None
        mock_page.get.side_effect = lambda k, d=None: (
            mock_resources if k == "/Resources" else d
        )

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        with patch("docint.core.readers.documents.layout.pypdf") as mock_pypdf:
            mock_pypdf.PdfReader.return_value = mock_reader
            analyzer = PypdfLayoutAnalyzer("/fake/doc.pdf")
            blocks = analyzer.analyze_page(0)

        table_blocks = [b for b in blocks if b.type == BlockType.TABLE]
        text_blocks = [b for b in blocks if b.type == BlockType.TEXT]
        assert len(table_blocks) == 1
        assert "Table 1:" in table_blocks[0].text
        assert "Model" in table_blocks[0].text
        # Remaining text should still exist (intro + conclusion)
        assert len(text_blocks) >= 1
        remaining = text_blocks[0].text
        assert "introductory" in remaining
        assert "Table 1:" not in remaining

    def test_no_images_no_tables_produces_text_only(self) -> None:
        """A plain text page should produce only TEXT blocks."""
        mock_page = MagicMock()
        mock_mb = MagicMock()
        mock_mb.left = 0.0
        mock_mb.bottom = 0.0
        mock_mb.right = 612.0
        mock_mb.top = 792.0
        mock_page.mediabox = mock_mb
        mock_page.extract_text.return_value = "Just plain text. Nothing special."
        mock_resources = MagicMock()
        mock_resources.get.return_value = None
        mock_page.get.side_effect = lambda k, d=None: (
            mock_resources if k == "/Resources" else d
        )

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        with patch("docint.core.readers.documents.layout.pypdf") as mock_pypdf:
            mock_pypdf.PdfReader.return_value = mock_reader
            analyzer = PypdfLayoutAnalyzer("/fake/doc.pdf")
            blocks = analyzer.analyze_page(0)

        assert len(blocks) == 1
        assert blocks[0].type == BlockType.TEXT
        assert "plain text" in blocks[0].text

    def test_empty_page_produces_fallback_block(self) -> None:
        """A page with no text or images should still produce a block."""
        mock_page = MagicMock()
        mock_mb = MagicMock()
        mock_mb.left = 0.0
        mock_mb.bottom = 0.0
        mock_mb.right = 612.0
        mock_mb.top = 792.0
        mock_page.mediabox = mock_mb
        mock_page.extract_text.return_value = ""
        mock_resources = MagicMock()
        mock_resources.get.return_value = None
        mock_page.get.side_effect = lambda k, d=None: (
            mock_resources if k == "/Resources" else d
        )

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        with patch("docint.core.readers.documents.layout.pypdf") as mock_pypdf:
            mock_pypdf.PdfReader.return_value = mock_reader
            analyzer = PypdfLayoutAnalyzer("/fake/doc.pdf")
            blocks = analyzer.analyze_page(0)

        assert len(blocks) == 1
        assert blocks[0].type == BlockType.TEXT
        assert blocks[0].confidence == 0.0

    def test_mixed_content_page(self) -> None:
        """A page with images, tables, and text should produce all block types."""
        mixed_text = (
            "Introduction paragraph.\n"
            "Table 1: Key metrics\n"
            "Metric   Value\n"
            "Loss     0.5\n"
            "\n"
            "Some concluding remarks."
        )

        mock_page = MagicMock()
        mock_mb = MagicMock()
        mock_mb.left = 0.0
        mock_mb.bottom = 0.0
        mock_mb.right = 612.0
        mock_mb.top = 792.0
        mock_page.mediabox = mock_mb
        mock_page.extract_text.return_value = mixed_text

        # Simulate image XObject
        mock_image_obj = MagicMock()
        mock_image_obj.get.side_effect = lambda k, d="": (
            "/Image" if k == "/Subtype" else d
        )
        mock_image_obj.get_object.return_value = mock_image_obj
        mock_xobj_dict = {"/Im1": mock_image_obj}
        mock_xobj = MagicMock()
        mock_xobj.get_object.return_value = mock_xobj_dict
        mock_resources = MagicMock()
        mock_resources.get.side_effect = lambda k, d=None: (
            mock_xobj if k == "/XObject" else d
        )
        mock_page.get.side_effect = lambda k, d=None: (
            mock_resources if k == "/Resources" else (None if k == "/Contents" else d)
        )

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        with patch("docint.core.readers.documents.layout.pypdf") as mock_pypdf:
            mock_pypdf.PdfReader.return_value = mock_reader
            analyzer = PypdfLayoutAnalyzer("/fake/doc.pdf")
            blocks = analyzer.analyze_page(0)

        block_types = {b.type for b in blocks}
        assert BlockType.FIGURE in block_types
        assert BlockType.TABLE in block_types
        assert BlockType.TEXT in block_types


class TestContentStreamParsing:
    """Tests for the image bounding box extraction from content streams."""

    def test_extract_image_bbox_from_simple_stream(self) -> None:
        """Should extract correct bbox from cm + Do operators."""
        stream = "q\n1 0 0 1 100 200 cm\n300 0 0 400 0 0 cm\n/Im1 Do\nQ\n"
        result = _extract_image_bboxes_from_stream(stream, {"/Im1"})
        assert "/Im1" in result
        bbox = result["/Im1"]
        assert bbox.x0 == pytest.approx(100.0)
        assert bbox.y0 == pytest.approx(200.0)
        assert bbox.x1 == pytest.approx(400.0)  # 100 + 300
        assert bbox.y1 == pytest.approx(600.0)  # 200 + 400

    def test_extract_with_scaling(self) -> None:
        """Should handle scale + translate combos correctly."""
        stream = (
            "q\n"
            "1 0 0 1 196.559 397.582 cm\n"
            ".6 0 0 .6 0 0 cm\n"
            "364.8 0 0 537.36 0 0 cm\n"
            "/Im1 Do\n"
            "Q\n"
        )
        result = _extract_image_bboxes_from_stream(stream, {"/Im1"})
        assert "/Im1" in result
        bbox = result["/Im1"]
        assert bbox.x0 == pytest.approx(196.559, abs=0.1)
        assert bbox.y0 == pytest.approx(397.582, abs=0.1)
        assert bbox.x1 == pytest.approx(415.439, abs=0.1)
        assert bbox.y1 == pytest.approx(720.0, abs=0.1)

    def test_unknown_image_name_ignored(self) -> None:
        """Images not in the lookup set should be skipped."""
        stream = "q\n300 0 0 400 100 200 cm\n/Im99 Do\nQ\n"
        result = _extract_image_bboxes_from_stream(stream, {"/Im1"})
        assert "/Im99" not in result
        assert len(result) == 0

    def test_multiple_images(self) -> None:
        """Multiple images on one page should all get bboxes."""
        stream = (
            "q\n200 0 0 300 50 100 cm\n/Im1 Do\nQ\n"
            "q\n150 0 0 200 400 500 cm\n/Im2 Do\nQ\n"
        )
        result = _extract_image_bboxes_from_stream(stream, {"/Im1", "/Im2"})
        assert len(result) == 2
        assert "/Im1" in result
        assert "/Im2" in result
        assert result["/Im1"].x0 == pytest.approx(50.0)
        assert result["/Im2"].x0 == pytest.approx(400.0)


class TestTableDetection:
    """Tests for the table region detection heuristic."""

    def test_find_table_end_basic(self) -> None:
        """Table end should be found after tabular rows."""

        lines = [
            "Table 1: Results",
            "A    B    C",
            "1    2    3",
            "4    5    6",
            "",
            "Regular paragraph text continues here with more content that is long enough.",
        ]
        end = _find_table_end(lines, 0)
        # Should stop before the blank + prose paragraph
        assert end == 3  # last data row

    def test_find_table_end_stops_at_section(self) -> None:
        """Table detection should stop at a new section heading."""
        lines = [
            "Table 2: More results",
            "X    Y",
            "1    2",
            "3.1 Next Section",
            "Text after heading.",
        ]
        end = _find_table_end(lines, 0)
        assert end == 2

    def test_detect_tables_removes_table_from_text(self) -> None:
        """Table regions should be excluded from the remaining text."""

        text = (
            "Introduction.\n"
            "Table 1: Data\n"
            "Col1  Col2  Col3\n"
            "A     B     C\n"
            "\n"
            "Conclusion paragraph with enough text to not look like a table row."
        )
        table_blocks, remaining = PypdfLayoutAnalyzer._detect_tables(
            text, 0, BBox(0, 0, 612, 792), 612.0, 792.0
        )
        assert len(table_blocks) == 1
        assert "Table 1:" in table_blocks[0].text
        assert "Introduction" in remaining
        assert "Conclusion" in remaining
        assert "Table 1:" not in remaining

    def test_no_table_returns_full_text(self) -> None:
        """Without table captions, all text should remain."""

        text = "Just regular text without any tables."
        table_blocks, remaining = PypdfLayoutAnalyzer._detect_tables(
            text, 0, BBox(0, 0, 612, 792), 612.0, 792.0
        )
        assert len(table_blocks) == 0
        assert remaining == text


class TestMatrixMultiplication:
    """Tests for the PDF affine matrix multiplication helper."""

    def test_identity(self) -> None:
        """Multiplying two identity matrices returns identity."""
        identity = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        result = _multiply_matrices(identity, identity)
        assert result == pytest.approx(identity)

    def test_translate(self) -> None:
        """Translation matrix preserves tx/ty offsets."""
        identity = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        translate = [1.0, 0.0, 0.0, 1.0, 100.0, 200.0]
        result = _multiply_matrices(identity, translate)
        assert result[4] == pytest.approx(100.0)
        assert result[5] == pytest.approx(200.0)

    def test_scale_then_translate(self) -> None:
        """Scaling after translation preserves the translation offset."""
        translate = [1.0, 0.0, 0.0, 1.0, 50.0, 50.0]
        scale = [2.0, 0.0, 0.0, 3.0, 0.0, 0.0]
        result = _multiply_matrices(translate, scale)
        # Scale should apply in current coord system, translation preserved
        assert result[0] == pytest.approx(2.0)
        assert result[3] == pytest.approx(3.0)
        assert result[4] == pytest.approx(50.0)
        assert result[5] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# OCR tests
# ---------------------------------------------------------------------------


class TestOCR:
    """Tests for OCR page-text assembly."""

    def test_build_page_text_pdf_only(
        self, sample_page_info: PageInfo, sample_layout_block: LayoutBlock
    ) -> None:
        """Page with layout blocks only should yield source_mix='pdf_text'."""
        result = build_page_text(sample_page_info, [sample_layout_block], [])
        assert result.source_mix == "pdf_text"
        assert "Hello world" in result.full_text
        assert result.confidence == 1.0

    def test_build_page_text_ocr_only(self, sample_page_info: PageInfo) -> None:
        """Page with OCR spans only should yield source_mix='ocr'."""
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
        """Page with both layout blocks and OCR spans should yield source_mix='mixed'."""
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

    def test_extract_text_for_pages_vision_fallback(self) -> None:
        """Vision engine should be tried when pypdf yields nothing on OCR pages."""
        page_info = PageInfo(
            page_index=0,
            has_text_layer=False,
            text_coverage=0.0,
            needs_ocr=True,
            width=612.0,
            height=792.0,
            status="completed",
        )
        layout: dict[int, list[LayoutBlock]] = {0: []}

        mock_vision = MagicMock()
        mock_vision.ocr_page.return_value = [
            OCRSpan(
                text="Vision-extracted text from scanned page.",
                bbox=BBox(x0=0, y0=0, x1=612, y1=792),
                confidence=0.7,
                source="vision_ocr",
            )
        ]

        with patch("docint.core.readers.documents.ocr.pypdf") as mock_pypdf:
            mock_page = MagicMock()
            mock_page.extract_text.return_value = ""
            mock_mb = MagicMock()
            mock_mb.left = 0.0
            mock_mb.bottom = 0.0
            mock_mb.right = 612.0
            mock_mb.top = 792.0
            mock_page.mediabox = mock_mb
            mock_reader = MagicMock()
            mock_reader.pages = [mock_page]
            mock_pypdf.PdfReader.return_value = mock_reader

            result = extract_text_for_pages(
                "/fake/scan.pdf",
                [page_info],
                layout,
                vision_engine=mock_vision,
            )

        assert 0 in result
        assert "Vision-extracted text" in result[0].full_text
        assert result[0].source_mix == "ocr"
        mock_vision.ocr_page.assert_called_once()

    def test_extract_text_for_pages_no_vision_when_text_found(self) -> None:
        """Vision engine should NOT be called when pypdf yields text."""
        page_info = PageInfo(
            page_index=0,
            has_text_layer=True,
            text_coverage=0.5,
            needs_ocr=True,
            width=612.0,
            height=792.0,
            status="completed",
        )
        layout: dict[int, list[LayoutBlock]] = {0: []}

        mock_vision = MagicMock()

        with patch("docint.core.readers.documents.ocr.pypdf") as mock_pypdf:
            mock_page = MagicMock()
            mock_page.extract_text.return_value = "Some actual text."
            mock_mb = MagicMock()
            mock_mb.left = 0.0
            mock_mb.bottom = 0.0
            mock_mb.right = 612.0
            mock_mb.top = 792.0
            mock_page.mediabox = mock_mb
            mock_reader = MagicMock()
            mock_reader.pages = [mock_page]
            mock_pypdf.PdfReader.return_value = mock_reader

            result = extract_text_for_pages(
                "/fake/doc.pdf",
                [page_info],
                layout,
                vision_engine=mock_vision,
            )

        assert 0 in result
        assert "Some actual text" in result[0].full_text
        mock_vision.ocr_page.assert_not_called()

    def test_vision_ocr_engine_downscales_large_images(self) -> None:
        """VisionOCREngine should resize images exceeding max_image_dimension."""
        from docint.core.readers.documents.ocr import VisionOCREngine

        mock_page = MagicMock()
        mock_page.get_width.return_value = 612.0
        mock_page.get_height.return_value = 792.0

        # Simulate a 3000×4000 rendered bitmap that exceeds the 1024 cap.
        from PIL import Image as PILImage

        large_img = PILImage.new("RGB", (3000, 4000), color="white")
        mock_bitmap = MagicMock()
        mock_bitmap.to_pil.return_value = large_img

        mock_page.render.return_value = mock_bitmap

        mock_pdf = MagicMock()
        mock_pdf.__getitem__ = MagicMock(return_value=mock_page)

        with (
            patch("docint.core.readers.documents.ocr.pypdfium2") as mock_pdfium,
            patch("docint.core.readers.documents.ocr.OpenAIPipeline") as MockPipeline,
            patch("docint.core.readers.documents.ocr._OpenAI"),
            patch("docint.core.readers.documents.ocr.load_openai_env"),
            patch("docint.core.readers.documents.ocr.load_model_env") as mock_model_env,
        ):
            mock_pdfium.PdfDocument.return_value = mock_pdf
            pipeline_instance = MagicMock()
            pipeline_instance.load_prompt.return_value = "Extract text"
            pipeline_instance.seed = 42
            pipeline_instance.temperature = 0.0
            pipeline_instance.top_p = 0.0
            MockPipeline.return_value = pipeline_instance

            mock_model_env.return_value.vision_model_file = "test-vision.gguf"

            # Mock the vision client response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Arabic text from OCR"

            engine = VisionOCREngine(
                "/fake/large.pdf",
                timeout=30.0,
                max_retries=0,
                max_image_dimension=1024,
                max_tokens=4096,
            )
            with patch.object(
                engine._vision_client.chat.completions,
                "create",
                return_value=mock_response,
            ) as mock_create:
                spans = engine.ocr_page(0)

        assert len(spans) == 1
        assert "Arabic text from OCR" in spans[0].text
        # Verify the render used scaled DPI (120/72 ≈ 1.667)
        mock_page.render.assert_called_once()
        call_kwargs = mock_page.render.call_args
        assert abs(call_kwargs[1]["scale"] - 120 / 72) < 0.01
        # Verify max_tokens was passed to the API call
        create_kwargs = mock_create.call_args[1]
        assert create_kwargs["max_tokens"] == 4096

    def test_vision_ocr_engine_respects_timeout_config(self) -> None:
        """VisionOCREngine should create client with OCR-specific timeout."""
        from docint.core.readers.documents.ocr import VisionOCREngine

        with (
            patch("docint.core.readers.documents.ocr.pypdfium2"),
            patch("docint.core.readers.documents.ocr.OpenAIPipeline") as MockPipeline,
            patch("docint.core.readers.documents.ocr._OpenAI") as MockOpenAI,
            patch("docint.core.readers.documents.ocr.load_openai_env") as mock_oai_env,
        ):
            mock_oai_env.return_value.api_key = "sk-test"
            mock_oai_env.return_value.api_base = "http://localhost:8080/v1"
            pipeline_instance = MagicMock()
            pipeline_instance.load_prompt.return_value = "Extract text"
            MockPipeline.return_value = pipeline_instance

            VisionOCREngine(
                "/fake/doc.pdf",
                timeout=60.0,
                max_retries=0,
                max_image_dimension=1024,
                max_tokens=2048,
            )

            MockOpenAI.assert_called_once_with(
                api_key="sk-test",
                base_url="http://localhost:8080/v1",
                timeout=60.0,
                max_retries=0,
            )

    def test_vision_ocr_retries_at_half_resolution_on_timeout(self) -> None:
        """On timeout, VisionOCREngine should retry at half the max dimension."""
        from docint.core.readers.documents.ocr import VisionOCREngine

        mock_page = MagicMock()
        mock_page.get_width.return_value = 612.0
        mock_page.get_height.return_value = 792.0

        from PIL import Image as PILImage

        img = PILImage.new("RGB", (800, 1000), color="white")
        mock_bitmap = MagicMock()
        mock_bitmap.to_pil.return_value = img
        mock_page.render.return_value = mock_bitmap

        mock_pdf = MagicMock()
        mock_pdf.__getitem__ = MagicMock(return_value=mock_page)

        with (
            patch("docint.core.readers.documents.ocr.pypdfium2") as mock_pdfium,
            patch("docint.core.readers.documents.ocr.OpenAIPipeline") as MockPipeline,
            patch("docint.core.readers.documents.ocr._OpenAI"),
            patch("docint.core.readers.documents.ocr.load_openai_env"),
            patch("docint.core.readers.documents.ocr.load_model_env") as mock_model_env,
        ):
            mock_pdfium.PdfDocument.return_value = mock_pdf
            pipeline_instance = MagicMock()
            pipeline_instance.load_prompt.return_value = "Extract text"
            pipeline_instance.seed = 42
            pipeline_instance.temperature = 0.0
            pipeline_instance.top_p = 0.0
            MockPipeline.return_value = pipeline_instance
            mock_model_env.return_value.vision_model_file = "test-vision.gguf"

            # First call raises timeout, second succeeds
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Recovered text"

            engine = VisionOCREngine(
                "/fake/doc.pdf",
                timeout=10.0,
                max_retries=0,
                max_image_dimension=1024,
                max_tokens=4096,
            )

            with patch.object(
                engine._vision_client.chat.completions,
                "create",
                side_effect=[
                    RuntimeError("Request timed out."),
                    mock_response,
                ],
            ) as mock_create:
                spans = engine.ocr_page(0)

        assert len(spans) == 1
        assert "Recovered text" in spans[0].text
        # Two API calls: initial attempt + reduced-resolution retry
        assert mock_create.call_count == 2

    def test_vision_ocr_retries_on_empty_with_higher_detail(self) -> None:
        """When OCR returns empty text, VisionOCREngine should retry at higher detail."""
        from docint.core.readers.documents.ocr import VisionOCREngine

        mock_page = MagicMock()
        mock_page.get_width.return_value = 612.0
        mock_page.get_height.return_value = 792.0

        from PIL import Image as PILImage

        # Tall page to mirror screenshot-style scanned PDFs.
        img = PILImage.new("RGB", (900, 3000), color="white")
        mock_bitmap = MagicMock()
        mock_bitmap.to_pil.return_value = img
        mock_page.render.return_value = mock_bitmap

        mock_pdf = MagicMock()
        mock_pdf.__getitem__ = MagicMock(return_value=mock_page)

        with (
            patch("docint.core.readers.documents.ocr.pypdfium2") as mock_pdfium,
            patch("docint.core.readers.documents.ocr.OpenAIPipeline") as MockPipeline,
            patch("docint.core.readers.documents.ocr._OpenAI"),
            patch("docint.core.readers.documents.ocr.load_openai_env"),
            patch("docint.core.readers.documents.ocr.load_model_env") as mock_model_env,
        ):
            mock_pdfium.PdfDocument.return_value = mock_pdf
            pipeline_instance = MagicMock()
            pipeline_instance.load_prompt.return_value = "Extract text"
            pipeline_instance.seed = 42
            pipeline_instance.temperature = 0.0
            pipeline_instance.top_p = 0.0
            MockPipeline.return_value = pipeline_instance
            mock_model_env.return_value.vision_model_file = "test-vision.gguf"

            # First call returns empty content; second (higher-detail) succeeds.
            first = MagicMock()
            first.choices = [MagicMock()]
            first.choices[0].message.content = ""

            second = MagicMock()
            second.choices = [MagicMock()]
            second.choices[0].message.content = "نص عربي مستخرج"

            engine = VisionOCREngine(
                "/fake/doc.pdf",
                timeout=10.0,
                max_retries=0,
                max_image_dimension=1024,
                max_tokens=4096,
            )

            with patch.object(
                engine._vision_client.chat.completions,
                "create",
                side_effect=[first, second],
            ) as mock_create:
                spans = engine.ocr_page(0)

        assert len(spans) == 1
        assert spans[0].text == "نص عربي مستخرج"
        assert mock_create.call_count == 2

    def test_vision_ocr_treats_refusal_as_empty_and_recovers(self) -> None:
        """Refusal text should be dropped so higher-detail retry can recover OCR text."""
        from docint.core.readers.documents.ocr import VisionOCREngine

        mock_page = MagicMock()
        mock_page.get_width.return_value = 612.0
        mock_page.get_height.return_value = 792.0

        from PIL import Image as PILImage

        img = PILImage.new("RGB", (900, 3000), color="white")
        mock_bitmap = MagicMock()
        mock_bitmap.to_pil.return_value = img
        mock_page.render.return_value = mock_bitmap

        mock_pdf = MagicMock()
        mock_pdf.__getitem__ = MagicMock(return_value=mock_page)

        with (
            patch("docint.core.readers.documents.ocr.pypdfium2") as mock_pdfium,
            patch("docint.core.readers.documents.ocr.OpenAIPipeline") as MockPipeline,
            patch("docint.core.readers.documents.ocr._OpenAI"),
            patch("docint.core.readers.documents.ocr.load_openai_env"),
            patch("docint.core.readers.documents.ocr.load_model_env") as mock_model_env,
        ):
            mock_pdfium.PdfDocument.return_value = mock_pdf
            pipeline_instance = MagicMock()
            pipeline_instance.load_prompt.return_value = "Extract text"
            pipeline_instance.seed = 42
            pipeline_instance.temperature = 0.0
            pipeline_instance.top_p = 0.0
            MockPipeline.return_value = pipeline_instance
            mock_model_env.return_value.vision_model_file = "test-vision.gguf"

            refusal = MagicMock()
            refusal.choices = [MagicMock()]
            refusal.choices[0].message.content = "I'm sorry, I can't assist with that."

            recovered = MagicMock()
            recovered.choices = [MagicMock()]
            recovered.choices[0].message.content = "Recovered OCR text"

            engine = VisionOCREngine(
                "/fake/doc.pdf",
                timeout=10.0,
                max_retries=0,
                max_image_dimension=1024,
                max_tokens=4096,
            )

            with patch.object(
                engine._vision_client.chat.completions,
                "create",
                side_effect=[refusal, recovered],
            ) as mock_create:
                spans = engine.ocr_page(0)

        assert len(spans) == 1
        assert spans[0].text == "Recovered OCR text"
        assert mock_create.call_count == 2

    def test_vision_ocr_strips_reasoning_and_keeps_ocr_text(self) -> None:
        """Reasoning scratchpads must be stripped, real OCR text must survive."""
        from docint.core.readers.documents.ocr import VisionOCREngine

        mock_page = MagicMock()
        mock_page.get_width.return_value = 612.0
        mock_page.get_height.return_value = 792.0

        from PIL import Image as PILImage

        img = PILImage.new("RGB", (900, 1200), color="white")
        mock_bitmap = MagicMock()
        mock_bitmap.to_pil.return_value = img
        mock_page.render.return_value = mock_bitmap

        mock_pdf = MagicMock()
        mock_pdf.__getitem__ = MagicMock(return_value=mock_page)

        with (
            patch("docint.core.readers.documents.ocr.pypdfium2") as mock_pdfium,
            patch("docint.core.readers.documents.ocr.OpenAIPipeline") as MockPipeline,
            patch("docint.core.readers.documents.ocr._OpenAI"),
            patch("docint.core.readers.documents.ocr.load_openai_env"),
            patch("docint.core.readers.documents.ocr.load_model_env") as mock_model_env,
        ):
            mock_pdfium.PdfDocument.return_value = mock_pdf
            pipeline_instance = MagicMock()
            pipeline_instance.load_prompt.return_value = "Extract text"
            pipeline_instance.seed = 42
            pipeline_instance.temperature = 0.0
            pipeline_instance.top_p = 0.0
            pipeline_instance.reasoning_effort = None
            MockPipeline.return_value = pipeline_instance
            mock_model_env.return_value.vision_model_file = "test-vision.gguf"

            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[
                0
            ].message.content = "<think>analyzing layout</think>ACTUAL_OCR_TEXT"

            engine = VisionOCREngine(
                "/fake/doc.pdf",
                timeout=10.0,
                max_retries=0,
                max_image_dimension=1024,
                max_tokens=4096,
            )

            with patch.object(
                engine._vision_client.chat.completions,
                "create",
                return_value=response,
            ) as mock_create:
                spans = engine.ocr_page(0)

        assert len(spans) == 1
        assert spans[0].text == "ACTUAL_OCR_TEXT"
        assert mock_create.call_count == 1

    def test_vision_ocr_reasoning_only_triggers_recovery(self) -> None:
        """A pure reasoning response should be treated empty so recovery retries."""
        from docint.core.readers.documents.ocr import VisionOCREngine

        mock_page = MagicMock()
        mock_page.get_width.return_value = 612.0
        mock_page.get_height.return_value = 792.0

        from PIL import Image as PILImage

        img = PILImage.new("RGB", (900, 3000), color="white")
        mock_bitmap = MagicMock()
        mock_bitmap.to_pil.return_value = img
        mock_page.render.return_value = mock_bitmap

        mock_pdf = MagicMock()
        mock_pdf.__getitem__ = MagicMock(return_value=mock_page)

        with (
            patch("docint.core.readers.documents.ocr.pypdfium2") as mock_pdfium,
            patch("docint.core.readers.documents.ocr.OpenAIPipeline") as MockPipeline,
            patch("docint.core.readers.documents.ocr._OpenAI"),
            patch("docint.core.readers.documents.ocr.load_openai_env"),
            patch("docint.core.readers.documents.ocr.load_model_env") as mock_model_env,
        ):
            mock_pdfium.PdfDocument.return_value = mock_pdf
            pipeline_instance = MagicMock()
            pipeline_instance.load_prompt.return_value = "Extract text"
            pipeline_instance.seed = 42
            pipeline_instance.temperature = 0.0
            pipeline_instance.top_p = 0.0
            pipeline_instance.reasoning_effort = None
            MockPipeline.return_value = pipeline_instance
            mock_model_env.return_value.vision_model_file = "test-vision.gguf"

            reasoning_only = MagicMock()
            reasoning_only.choices = [MagicMock()]
            reasoning_only.choices[
                0
            ].message.content = "<think>lots of reasoning and nothing else</think>"

            recovered = MagicMock()
            recovered.choices = [MagicMock()]
            recovered.choices[0].message.content = "Recovered OCR text"

            engine = VisionOCREngine(
                "/fake/doc.pdf",
                timeout=10.0,
                max_retries=0,
                max_image_dimension=1024,
                max_tokens=4096,
            )

            with patch.object(
                engine._vision_client.chat.completions,
                "create",
                side_effect=[reasoning_only, recovered],
            ) as mock_create:
                spans = engine.ocr_page(0)

        assert len(spans) == 1
        assert spans[0].text == "Recovered OCR text"
        assert mock_create.call_count == 2

    def test_vision_ocr_no_image_refusal_is_empty(self) -> None:
        """A no-image refusal should be dropped so recovery can retry."""
        from docint.core.readers.documents.ocr import VisionOCREngine

        mock_page = MagicMock()
        mock_page.get_width.return_value = 612.0
        mock_page.get_height.return_value = 792.0

        from PIL import Image as PILImage

        img = PILImage.new("RGB", (900, 3000), color="white")
        mock_bitmap = MagicMock()
        mock_bitmap.to_pil.return_value = img
        mock_page.render.return_value = mock_bitmap

        mock_pdf = MagicMock()
        mock_pdf.__getitem__ = MagicMock(return_value=mock_page)

        with (
            patch("docint.core.readers.documents.ocr.pypdfium2") as mock_pdfium,
            patch("docint.core.readers.documents.ocr.OpenAIPipeline") as MockPipeline,
            patch("docint.core.readers.documents.ocr._OpenAI"),
            patch("docint.core.readers.documents.ocr.load_openai_env"),
            patch("docint.core.readers.documents.ocr.load_model_env") as mock_model_env,
        ):
            mock_pdfium.PdfDocument.return_value = mock_pdf
            pipeline_instance = MagicMock()
            pipeline_instance.load_prompt.return_value = "Extract text"
            pipeline_instance.seed = 42
            pipeline_instance.temperature = 0.0
            pipeline_instance.top_p = 0.0
            pipeline_instance.reasoning_effort = None
            MockPipeline.return_value = pipeline_instance
            mock_model_env.return_value.vision_model_file = "test-vision.gguf"

            refusal = MagicMock()
            refusal.choices = [MagicMock()]
            refusal.choices[
                0
            ].message.content = "I don't see any image attached to your message."

            recovered = MagicMock()
            recovered.choices = [MagicMock()]
            recovered.choices[0].message.content = "Recovered OCR text"

            engine = VisionOCREngine(
                "/fake/doc.pdf",
                timeout=10.0,
                max_retries=0,
                max_image_dimension=1024,
                max_tokens=4096,
            )

            with patch.object(
                engine._vision_client.chat.completions,
                "create",
                side_effect=[refusal, recovered],
            ) as mock_create:
                spans = engine.ocr_page(0)

        assert len(spans) == 1
        assert spans[0].text == "Recovered OCR text"
        assert mock_create.call_count == 2

    def test_vision_ocr_forwards_reasoning_effort(self) -> None:
        """Reasoning effort from the pipeline should be passed to the vision call."""
        from docint.core.readers.documents.ocr import VisionOCREngine

        mock_page = MagicMock()
        mock_page.get_width.return_value = 612.0
        mock_page.get_height.return_value = 792.0

        from PIL import Image as PILImage

        img = PILImage.new("RGB", (900, 1200), color="white")
        mock_bitmap = MagicMock()
        mock_bitmap.to_pil.return_value = img
        mock_page.render.return_value = mock_bitmap

        mock_pdf = MagicMock()
        mock_pdf.__getitem__ = MagicMock(return_value=mock_page)

        with (
            patch("docint.core.readers.documents.ocr.pypdfium2") as mock_pdfium,
            patch("docint.core.readers.documents.ocr.OpenAIPipeline") as MockPipeline,
            patch("docint.core.readers.documents.ocr._OpenAI"),
            patch("docint.core.readers.documents.ocr.load_openai_env"),
            patch("docint.core.readers.documents.ocr.load_model_env") as mock_model_env,
        ):
            mock_pdfium.PdfDocument.return_value = mock_pdf
            pipeline_instance = MagicMock()
            pipeline_instance.load_prompt.return_value = "Extract text"
            pipeline_instance.seed = 42
            pipeline_instance.temperature = 0.0
            pipeline_instance.top_p = 0.0
            pipeline_instance.reasoning_effort = "high"
            MockPipeline.return_value = pipeline_instance
            mock_model_env.return_value.vision_model_file = "test-vision.gguf"

            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message.content = "extracted"

            engine = VisionOCREngine(
                "/fake/doc.pdf",
                timeout=10.0,
                max_retries=0,
                max_image_dimension=1024,
                max_tokens=4096,
            )

            with patch.object(
                engine._vision_client.chat.completions,
                "create",
                return_value=response,
            ) as mock_create:
                engine.ocr_page(0)

        assert mock_create.call_args_list[0].kwargs["reasoning_effort"] == "high"


# ---------------------------------------------------------------------------
# Orchestrator tests
# ---------------------------------------------------------------------------


class TestOrchestrator:
    """Tests for the document pipeline orchestrator."""

    def test_process_with_mocked_pdf(
        self, pipeline_config: PipelineConfig, tmp_path: Path
    ) -> None:
        """Processing a mocked PDF should produce a completed manifest with artifacts."""
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

        with (
            patch("docint.core.readers.documents.triage.pypdf") as mock_triage_pypdf,
            patch("docint.core.readers.documents.layout.pypdf") as mock_layout_pypdf,
            patch("docint.core.readers.documents.ocr.pypdf") as mock_ocr_pypdf,
        ):
            mock_triage_pypdf.PdfReader.return_value = mock_reader
            mock_layout_pypdf.PdfReader.return_value = mock_reader
            mock_ocr_pypdf.PdfReader.return_value = mock_reader

            manifest = orch.process(pdf_file)

        assert manifest.status == "completed"
        assert manifest.pages_total == 1
        assert manifest.pages_failed == 0

        # Check artifacts were created

        doc_id = compute_file_hash(pdf_file)
        artifacts_dir = Path(pipeline_config.artifacts_dir)
        assert (artifacts_dir / doc_id / "manifest.json").exists()

    def test_idempotent_rerun(
        self, pipeline_config: PipelineConfig, tmp_path: Path
    ) -> None:
        """Second run should reuse artifacts when pipeline version matches."""

        # Disable force to test idempotency
        config = PipelineConfig(
            text_coverage_threshold=pipeline_config.text_coverage_threshold,
            pipeline_version=pipeline_config.pipeline_version,
            artifacts_dir=pipeline_config.artifacts_dir,
            max_retries=pipeline_config.max_retries,
            force_reprocess=False,
            max_workers=pipeline_config.max_workers,
            enable_vision_ocr=False,
            vision_ocr_timeout=60.0,
            vision_ocr_max_retries=1,
            vision_ocr_max_image_dimension=1024,
            vision_ocr_max_tokens=4096,
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
        with (
            patch("docint.core.readers.documents.triage.pypdf") as m1,
            patch("docint.core.readers.documents.layout.pypdf") as m2,
            patch("docint.core.readers.documents.ocr.pypdf") as m3,
        ):
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

        with (
            patch("docint.core.readers.documents.triage.pypdf") as m1,
            patch("docint.core.readers.documents.layout.pypdf") as m2,
            patch("docint.core.readers.documents.ocr.pypdf") as m3,
        ):
            m1.PdfReader.return_value = mock_reader
            m2.PdfReader.return_value = mock_reader
            m3.PdfReader.return_value = mock_reader
            manifest = orch.process(pdf_file)

        assert manifest.status == "completed"
        assert manifest.pages_total == 2
        # At least one page should be processed (the good one)
        assert any(p.status == "completed" for p in manifest.pages)

    def test_retry_logic(self, pipeline_config: PipelineConfig, tmp_path: Path) -> None:
        """Stages should retry on transient failures."""
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

    def test_scanned_pdf_injects_text_block(self, tmp_path: Path) -> None:
        """A scanned PDF should get a synthetic TEXT block after vision OCR.

        Args:
            tmp_path: pytest fixture providing a temporary directory for test files.
        """
        config = PipelineConfig(
            text_coverage_threshold=0.01,
            pipeline_version="test-1.0.0",
            artifacts_dir=str(tmp_path / "artifacts"),
            max_retries=1,
            force_reprocess=True,
            max_workers=1,
            enable_vision_ocr=True,
            vision_ocr_timeout=30.0,
            vision_ocr_max_retries=0,
            vision_ocr_max_image_dimension=1024,
            vision_ocr_max_tokens=4096,
        )

        pdf_file = tmp_path / "scan.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 scanned page test")

        # Scanned page: no text, but an embedded image
        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        mock_mb = MagicMock()
        mock_mb.width = 612.0
        mock_mb.height = 792.0
        mock_mb.left = 0.0
        mock_mb.bottom = 0.0
        mock_mb.right = 612.0
        mock_mb.top = 792.0
        mock_page.mediabox = mock_mb

        # Image XObject so layout produces a FIGURE block
        mock_image_obj = MagicMock()
        mock_image_obj.get.side_effect = lambda k, d="": (
            "/Image" if k == "/Subtype" else d
        )
        mock_image_obj.get_object.return_value = mock_image_obj
        mock_xobj_dict = {"/Im1": mock_image_obj}
        mock_xobj = MagicMock()
        mock_xobj.get_object.return_value = mock_xobj_dict
        mock_resources = MagicMock()
        mock_resources.get.side_effect = lambda k, d=None: (
            mock_xobj if k == "/XObject" else d
        )
        mock_page.get.side_effect = lambda k, d=None: (
            mock_resources if k == "/Resources" else (None if k == "/Contents" else d)
        )

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        # Mock the vision OCR engine to return extracted text
        mock_vision_engine = MagicMock()
        mock_vision_engine.ocr_page.return_value = [
            OCRSpan(
                text="Text from scanned page via vision OCR.",
                bbox=BBox(x0=0, y0=0, x1=612, y1=792),
                confidence=0.7,
                source="vision_ocr",
            )
        ]

        orch = DocumentPipelineOrchestrator(config=config)

        with (
            patch("docint.core.readers.documents.triage.pypdf") as m1,
            patch("docint.core.readers.documents.layout.pypdf") as m2,
            patch("docint.core.readers.documents.ocr.pypdf") as m3,
            patch(
                "docint.core.readers.documents.orchestrator.VisionOCREngine",
                return_value=mock_vision_engine,
            ),
        ):
            m1.PdfReader.return_value = mock_reader
            m2.PdfReader.return_value = mock_reader
            m3.PdfReader.return_value = mock_reader
            manifest = orch.process(pdf_file)

        assert manifest.status == "completed"
        assert manifest.pages_total == 1
        assert manifest.pages_ocr == 1

        # Verify chunks were produced from the vision OCR text
        doc_id = compute_file_hash(pdf_file)
        artifacts_dir = Path(config.artifacts_dir)
        chunks_path = artifacts_dir / doc_id / "chunks.jsonl"
        assert chunks_path.exists(), "Expected chunks.jsonl to be created"
        lines = [
            line for line in chunks_path.read_text().strip().split("\n") if line.strip()
        ]
        assert len(lines) >= 1
        import json

        chunk_data = json.loads(lines[0])
        assert (
            "vision OCR" in chunk_data["text"].lower()
            or "scanned" in chunk_data["text"].lower()
        )
