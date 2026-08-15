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
from docint.core.readers.documents.chunking import build_coarse_units
from docint.core.readers.documents.extraction import extract_images, extract_tables
from docint.core.readers.documents.layout import (
    DoclingParseLayoutAnalyzer,
    _detect_table_regions,
    _find_table_end,
    analyze_document,
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
from docint.core.readers.documents.parse import ParsedPage, ParsedPdf
from docint.core.readers.documents.triage import triage_pdf
from docint.utils.env_cfg import PipelineConfig, load_pipeline_config
from docint.utils.hashing import compute_file_hash
from pdf_fixtures import ImageBox, PageSpec, TextRun, build_pdf, two_column_page

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
        p = PageInfo(page_index=0, has_text_layer=True, text_coverage=1.0, needs_ocr=False)
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
        assert cfg.pipeline_version == "2.0.0"
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

    def test_empty_pipeline_version_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty ``PIPELINE_VERSION`` should fall back to default version."""
        monkeypatch.setenv("PIPELINE_VERSION", "   ")
        cfg = load_pipeline_config(default_pipeline_version="9.9.9")
        assert cfg.pipeline_version == "9.9.9"

    def test_artifacts_dir_from_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """PIPELINE_ARTIFACTS_DIR env var should propagate into PipelineConfig.

        Args:
            monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture for env manipulation.
            tmp_path (Path): The temporary path fixture for creating test directories.
        """
        custom = str(tmp_path / "custom-artifacts")
        monkeypatch.setenv("PIPELINE_ARTIFACTS_DIR", custom)
        cfg = load_pipeline_config()
        assert cfg.artifacts_dir == custom

    def test_vision_ocr_timeout_inherits_openai_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without an explicit override the OCR budget follows ``OPENAI_TIMEOUT``.

        A hardcoded default silently contradicts the endpoint's configured
        budget: a slow vision model that chat tolerates gets cut off mid-flight
        and surfaces as ``Request timed out``.

        Args:
            monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture for env manipulation.
        """
        monkeypatch.delenv("PIPELINE_VISION_OCR_TIMEOUT", raising=False)
        monkeypatch.setenv("OPENAI_TIMEOUT", "240")
        cfg = load_pipeline_config()
        assert cfg.vision_ocr_timeout == 240.0

    def test_vision_ocr_timeout_override_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An explicit ``PIPELINE_VISION_OCR_TIMEOUT`` still overrides the inherited value.

        Args:
            monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture for env manipulation.
        """
        monkeypatch.setenv("OPENAI_TIMEOUT", "240")
        monkeypatch.setenv("PIPELINE_VISION_OCR_TIMEOUT", "45")
        cfg = load_pipeline_config()
        assert cfg.vision_ocr_timeout == 45.0


# ---------------------------------------------------------------------------
# Triage tests
# ---------------------------------------------------------------------------


class TestTriage:
    """Tests for the PDF triage stage (digital, scanned, mixed detection)."""

    def test_digital_pdf(self, pipeline_config: PipelineConfig, tmp_path: Path) -> None:
        """Pages with sufficient text should not need OCR."""
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(build_pdf([two_column_page()]))

        pages = triage_pdf(pdf, pipeline_config)

        assert len(pages) == 1
        assert pages[0].has_text_layer is True
        assert pages[0].needs_ocr is False
        assert pages[0].status == "completed"
        assert (pages[0].width, pages[0].height) == (612.0, 792.0)
        assert pages[0].text_coverage > pipeline_config.text_coverage_threshold

    def test_scanned_pdf(self, pipeline_config: PipelineConfig, tmp_path: Path) -> None:
        """Pages with no text should need OCR."""
        pdf = tmp_path / "scan.pdf"
        pdf.write_bytes(build_pdf([PageSpec(images=[ImageBox(x=0, y=0, w=612, h=792)])]))

        pages = triage_pdf(pdf, pipeline_config)

        assert len(pages) == 1
        assert pages[0].has_text_layer is False
        assert pages[0].needs_ocr is True
        assert pages[0].text_coverage == 0.0

    def test_mixed_pdf(self, pipeline_config: PipelineConfig, tmp_path: Path) -> None:
        """A PDF with mixed pages should classify each correctly."""
        pdf = tmp_path / "mixed.pdf"
        pdf.write_bytes(build_pdf([two_column_page(), PageSpec()]))

        pages = triage_pdf(pdf, pipeline_config)

        assert len(pages) == 2
        assert pages[0].needs_ocr is False
        assert pages[1].needs_ocr is True

    def test_bad_page_does_not_crash(self, pipeline_config: PipelineConfig, tmp_path: Path) -> None:
        """A page that raises during parsing should be marked failed, not abort triage."""
        pdf = tmp_path / "bad.pdf"
        pdf.write_bytes(build_pdf([two_column_page(), two_column_page()]))
        real_page = ParsedPdf.page

        def _flaky(self: ParsedPdf, page_index: int) -> ParsedPage:
            if page_index == 1:
                raise RuntimeError("corrupt page")
            return real_page(self, page_index)

        with patch.object(ParsedPdf, "page", _flaky):
            pages = triage_pdf(pdf, pipeline_config)

        assert len(pages) == 2
        assert pages[0].status == "completed"
        assert pages[1].status == "failed"
        assert pages[1].needs_ocr is True
        assert pages[1].error is not None

    def test_unreadable_file_yields_single_failed_page(self, pipeline_config: PipelineConfig, tmp_path: Path) -> None:
        """A file docling-parse cannot open degrades to one failed page."""
        bogus = tmp_path / "bogus.pdf"
        bogus.write_bytes(b"not a pdf")

        pages = triage_pdf(bogus, pipeline_config)

        assert len(pages) == 1
        assert pages[0].status == "failed"
        assert pages[0].needs_ocr is True

    def test_reuses_injected_parsed_document(self, pipeline_config: PipelineConfig, tmp_path: Path) -> None:
        """When the orchestrator hands over an open ``ParsedPdf`` it is used as-is."""
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(build_pdf([two_column_page()]))
        with ParsedPdf(pdf) as parsed:
            with patch("docint.core.readers.documents.triage.ParsedPdf") as ctor:
                pages = triage_pdf(pdf, pipeline_config, parsed=parsed)
            ctor.assert_not_called()
        assert len(pages) == 1
        assert pages[0].needs_ocr is False


# ---------------------------------------------------------------------------
# Chunking tests
# ---------------------------------------------------------------------------


class TestChunking:
    """Tests for the document chunking logic."""

    def test_basic_chunking(self, sample_layout_block: LayoutBlock) -> None:
        """Single-block layout produces one coarse unit with correct metadata."""
        layout = {0: [sample_layout_block]}
        page_texts = {
            0: PageText(
                page_index=0,
                full_text=sample_layout_block.text,
                source_mix="pdf_text",
            )
        }
        units = build_coarse_units("doc123", layout, page_texts, [], [])
        assert len(units) >= 1
        assert units[0].doc_id == "doc123"
        assert units[0].source_mix == "pdf_text"
        assert units[0].section_path == []
        # No heading present, so the body is the block text verbatim.
        assert sample_layout_block.text in units[0].text

    def test_section_path_tracking(self) -> None:
        """Units following a TITLE carry its section path and prepend the heading."""
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
        units = build_coarse_units("doc456", layout, page_texts, [], [])
        # The body unit carries the section path and prepends the heading text.
        text_units = [u for u in units if "introduction text" in u.text.lower()]
        assert len(text_units) >= 1
        assert "Chapter 1: Introduction" in text_units[0].section_path
        assert text_units[0].text.startswith("Chapter 1: Introduction")
        # The heading is folded into its section's unit, never emitted alone.
        assert all(u.text.strip() != "Chapter 1: Introduction" for u in units)

    def test_coarse_units_respect_size_cap(self) -> None:
        """Multiple blocks are grouped into units bounded by coarse_chunk_size."""
        blocks = [
            LayoutBlock(
                block_id=f"t{i}",
                page_index=0,
                type=BlockType.TEXT,
                bbox=BBox(x0=0, y0=0, x1=612, y1=792),
                reading_order=i,
                confidence=1.0,
                text=f"Block {i} sentence one. Block {i} sentence two.",
            )
            for i in range(12)
        ]
        layout = {0: blocks}
        page_texts = {0: PageText(page_index=0, full_text="", source_mix="pdf_text")}
        cap = 120
        units = build_coarse_units("doc789", layout, page_texts, [], [], coarse_chunk_size=cap)
        assert len(units) > 1
        longest_block = max(len(b.text) for b in blocks)
        for unit in units:
            # A unit may exceed the cap only by a single whole block.
            assert len(unit.text) <= cap + longest_block

    def test_units_never_start_mid_sentence(self) -> None:
        """Regression: units begin at a block boundary, not an overlap fragment.

        The retired char-overlap chunker prefixed every non-first chunk with
        the last 64 characters of the previous chunk, producing mid-sentence
        starts. build_coarse_units carries no overlap, so each unit body
        begins exactly at a block's first character.
        """
        blocks = [
            LayoutBlock(
                block_id=f"b{i}",
                page_index=0,
                type=BlockType.TEXT,
                bbox=BBox(x0=0, y0=0, x1=612, y1=792),
                reading_order=i,
                confidence=1.0,
                text=f"Alpha sentence {i} ends here.",
            )
            for i in range(6)
        ]
        layout = {0: blocks}
        page_texts = {0: PageText(page_index=0, full_text="", source_mix="pdf_text")}
        units = build_coarse_units("nodup", layout, page_texts, [], [], coarse_chunk_size=40)
        assert len(units) > 1
        for unit in units:
            assert unit.text.lstrip().startswith("Alpha sentence")

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
        c1 = build_coarse_units("same-doc", layout, page_texts, [], [])
        c2 = build_coarse_units("same-doc", layout, page_texts, [], [])
        assert len(c1) == len(c2)
        for a, b in zip(c1, c2, strict=False):
            assert a.chunk_id == b.chunk_id

    def test_ocr_source_mix_propagated(self) -> None:
        """Units from OCR pages should carry source_mix='ocr'."""
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
        page_texts = {0: PageText(page_index=0, full_text="", source_mix="ocr", confidence=0.8)}
        units = build_coarse_units("ocr-doc", layout, page_texts, [], [])
        assert len(units) >= 1
        assert units[0].source_mix == "ocr"

    def test_figure_only_page_with_ocr_text_produces_chunks(self) -> None:
        """A scanned page (FIGURE + synthetic OCR TEXT block) still yields a unit."""
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
        units = build_coarse_units("scan-doc", layout, page_texts, [], [])
        assert len(units) >= 1
        # Exactly one unit — the figure block must not duplicate the page text.
        assert len(units) == 1
        assert "Vision OCR" in units[0].text
        assert units[0].source_mix == "ocr"


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
    """Tests for DoclingParseLayoutAnalyzer block detection on real PDFs."""

    @staticmethod
    def _analyze(tmp_path: Path, spec: PageSpec) -> list[LayoutBlock]:
        """Write a one-page PDF and return its layout blocks."""
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(build_pdf([spec]))
        analyzer = DoclingParseLayoutAnalyzer(pdf)
        try:
            return analyzer.analyze_page(0)
        finally:
            analyzer.close()

    def test_detect_images_creates_figure_blocks_with_placement(self, tmp_path: Path) -> None:
        """Embedded images become FIGURE blocks carrying their placement bbox."""
        spec = PageSpec(
            runs=[TextRun("Some text on the page.", x=60, y=700)],
            images=[ImageBox(x=100, y=400, w=200, h=100)],
        )
        blocks = self._analyze(tmp_path, spec)

        figure_blocks = [b for b in blocks if b.type == BlockType.FIGURE]
        text_blocks = [b for b in blocks if b.type == BlockType.TEXT]
        assert len(figure_blocks) == 1
        fig = figure_blocks[0].bbox
        assert (fig.x0, fig.y0, fig.x1, fig.y1) == pytest.approx((100.0, 400.0, 300.0, 500.0))
        assert figure_blocks[0].confidence == pytest.approx(0.9)
        assert len(text_blocks) == 1
        assert text_blocks[0].text == "Some text on the page."

    def test_two_images_two_figure_blocks(self, tmp_path: Path) -> None:
        """Every embedded image gets its own FIGURE block."""
        spec = PageSpec(images=[ImageBox(x=50, y=600, w=100, h=50), ImageBox(x=300, y=100, w=120, h=80)])
        blocks = self._analyze(tmp_path, spec)
        figs = sorted((b for b in blocks if b.type == BlockType.FIGURE), key=lambda b: b.bbox.x0)
        assert [round(b.bbox.x0) for b in figs] == [50, 300]

    def test_detect_tables_via_caption(self, tmp_path: Path) -> None:
        """A 'Table N:' caption followed by short rows becomes a TABLE block with a tight bbox."""
        rows = [
            "Some introductory text about the experiment.",
            "Table 1: Results summary",
            "Model    Accuracy   F1",
            "BERT     89.3       88.1",
            "GPT-2    91.0       90.5",
        ]
        runs = [TextRun(t, x=60, y=700 - 14 * i) for i, t in enumerate(rows)]
        runs.append(TextRun("The results show clear improvement in accuracy.", x=60, y=580))
        blocks = self._analyze(tmp_path, PageSpec(runs=runs))

        table_blocks = [b for b in blocks if b.type == BlockType.TABLE]
        assert len(table_blocks) == 1
        table = table_blocks[0]
        assert "Table 1:" in table.text
        assert "BERT" in table.text
        # Tight bbox: the table does not span the whole page.
        assert table.bbox.y1 < 720 and table.bbox.y0 > 600
        text = "\n".join(b.text for b in blocks if b.type == BlockType.TEXT)
        assert "introductory" in text and "results show" in text
        assert "Table 1:" not in text

    def test_no_images_no_tables_produces_text_only(self, tmp_path: Path) -> None:
        """Plain prose yields a single TEXT block with the page text."""
        blocks = self._analyze(
            tmp_path,
            PageSpec(runs=[TextRun("Just plain prose.", x=60, y=700), TextRun("Second line.", x=60, y=686)]),
        )
        assert [b.type for b in blocks] == [BlockType.TEXT]
        assert blocks[0].text == "Just plain prose.\nSecond line."
        assert blocks[0].bbox.y0 > 600  # tight, not page-sized

    def test_empty_page_produces_fallback_block(self, tmp_path: Path) -> None:
        """A page with nothing on it still yields an empty TEXT block."""
        blocks = self._analyze(tmp_path, PageSpec())
        assert len(blocks) == 1
        assert blocks[0].type == BlockType.TEXT
        assert blocks[0].text == ""
        assert blocks[0].confidence == 0.0

    def test_two_columns_read_left_column_first(self, tmp_path: Path) -> None:
        """Multi-column text is emitted column by column, one TEXT block per column."""
        blocks = self._analyze(tmp_path, two_column_page())
        text_blocks = [b for b in blocks if b.type == BlockType.TEXT]
        assert len(text_blocks) == 2
        assert text_blocks[0].text.startswith("Left column line 1")
        assert text_blocks[0].text.endswith("Left column line 3")
        assert text_blocks[1].text.startswith("Right column line 1")
        assert text_blocks[0].reading_order < text_blocks[1].reading_order

    def test_large_bold_line_becomes_title_and_bold_line_header(self, tmp_path: Path) -> None:
        """Font-based heading detection: biggest heading → TITLE, others → HEADER."""
        spec = PageSpec(
            runs=[
                TextRun("Annual Report", x=60, y=740, size=20, bold=True),
                TextRun("This is the body of the report, set in the regular face.", x=60, y=700, size=11),
                TextRun("It continues for another line of ordinary prose.", x=60, y=686, size=11),
                TextRun("Financial Summary", x=60, y=650, size=11, bold=True),
                TextRun("Revenue rose in every quarter of the reporting period.", x=60, y=636, size=11),
                TextRun("Costs stayed flat across the same period.", x=60, y=622, size=11),
            ]
        )
        blocks = self._analyze(tmp_path, spec)
        kinds = [(b.type, b.text) for b in blocks]
        assert (BlockType.TITLE, "Annual Report") in kinds
        assert (BlockType.HEADER, "Financial Summary") in kinds
        text_blocks = [b for b in blocks if b.type == BlockType.TEXT]
        assert len(text_blocks) == 2  # split at the heading
        assert "Annual Report" not in text_blocks[0].text
        order = [b.type for b in sorted(blocks, key=lambda b: b.reading_order)]
        assert order == [BlockType.TITLE, BlockType.TEXT, BlockType.HEADER, BlockType.TEXT]

    def test_uniform_page_has_no_headings(self, tmp_path: Path) -> None:
        """When every line looks alike nothing is promoted to a heading."""
        runs = [TextRun(f"Line {i} of ordinary prose that ends here.", x=60, y=700 - 14 * i) for i in range(6)]
        blocks = self._analyze(tmp_path, PageSpec(runs=runs))
        assert all(b.type == BlockType.TEXT for b in blocks)

    def test_analyze_document_reuses_injected_parsed_document(self, tmp_path: Path) -> None:
        """analyze_document() uses the caller's ParsedPdf when given one."""
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(build_pdf([two_column_page()]))
        page_info = PageInfo(page_index=0, has_text_layer=True, text_coverage=1.0, needs_ocr=False)
        with ParsedPdf(pdf) as parsed:
            with patch("docint.core.readers.documents.layout.ParsedPdf") as ctor:
                layout = analyze_document(pdf, [page_info], parsed=parsed)
            ctor.assert_not_called()
        assert 0 in layout and layout[0]


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

    def test_detect_table_regions(self) -> None:
        """Caption + rows form one region; surrounding prose is left out."""
        lines = [
            "Introduction.",
            "Table 1: Data",
            "Col1  Col2  Col3",
            "A     B     C",
            "",
            "Conclusion paragraph with enough text to not look like a table row.",
        ]
        assert _detect_table_regions(lines) == [(1, 3)]

    def test_no_table_no_regions(self) -> None:
        """Without table captions, no regions are found."""
        assert _detect_table_regions(["Just regular text without any tables."]) == []


# ---------------------------------------------------------------------------
# OCR tests
# ---------------------------------------------------------------------------


class TestOCR:
    """Tests for OCR page-text assembly."""

    def test_build_page_text_pdf_only(self, sample_page_info: PageInfo, sample_layout_block: LayoutBlock) -> None:
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

    def test_build_page_text_mixed(self, sample_page_info: PageInfo, sample_layout_block: LayoutBlock) -> None:
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

        # Simulate a 3000x4000 rendered bitmap that exceeds the 1024 cap.
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
            patch("docint.core.readers.documents.ocr.time.sleep"),
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
            patch("docint.core.readers.documents.ocr.time.sleep"),
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

    def test_vision_ocr_exhausted_timeouts_skip_higher_detail_recovery(self) -> None:
        """When every attempt fails at transport level, no larger image is sent.

        A transport failure is not an empty answer.  Escalating to a
        higher-detail image after two timeouts sends *more* bytes to an
        endpoint that has already proven too slow.
        """
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
            patch("docint.core.readers.documents.ocr.time.sleep"),
        ):
            mock_pdfium.PdfDocument.return_value = mock_pdf
            pipeline_instance = MagicMock()
            pipeline_instance.load_prompt.return_value = "Extract text"
            pipeline_instance.seed = 42
            pipeline_instance.temperature = 0.0
            pipeline_instance.top_p = 0.0
            MockPipeline.return_value = pipeline_instance
            mock_model_env.return_value.vision_model_file = "test-vision.gguf"

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
                side_effect=RuntimeError("Request timed out."),
            ) as mock_create:
                spans = engine.ocr_page(0)

        assert spans == []
        # Initial attempt + half-resolution retry only - no third, larger call.
        assert mock_create.call_count == 2

    def test_vision_ocr_stops_after_consecutive_page_failures(self) -> None:
        """Repeated total failures disable the engine instead of retrying every page.

        An unreachable or overloaded endpoint otherwise costs minutes per page
        for the whole document while producing nothing.
        """
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
            patch("docint.core.readers.documents.ocr.time.sleep"),
        ):
            mock_pdfium.PdfDocument.return_value = mock_pdf
            pipeline_instance = MagicMock()
            pipeline_instance.load_prompt.return_value = "Extract text"
            pipeline_instance.seed = 42
            pipeline_instance.temperature = 0.0
            pipeline_instance.top_p = 0.0
            MockPipeline.return_value = pipeline_instance
            mock_model_env.return_value.vision_model_file = "test-vision.gguf"

            engine = VisionOCREngine(
                "/fake/doc.pdf",
                timeout=10.0,
                max_retries=0,
                max_image_dimension=1024,
                max_tokens=4096,
            )
            budget = VisionOCREngine._MAX_CONSECUTIVE_PAGE_FAILURES

            with patch.object(
                engine._vision_client.chat.completions,
                "create",
                side_effect=RuntimeError("Request timed out."),
            ) as mock_create:
                for page_index in range(budget + 3):
                    assert engine.ocr_page(page_index) == []

            # Two calls per page until the budget is spent, then nothing.
            assert mock_create.call_count == budget * 2

    def test_vision_ocr_success_resets_failure_budget(self) -> None:
        """A page that answers clears the consecutive-failure counter.

        Isolated slow pages must not accumulate into a permanent shutdown of an
        otherwise working endpoint.
        """
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
            patch("docint.core.readers.documents.ocr.time.sleep"),
        ):
            mock_pdfium.PdfDocument.return_value = mock_pdf
            pipeline_instance = MagicMock()
            pipeline_instance.load_prompt.return_value = "Extract text"
            pipeline_instance.seed = 42
            pipeline_instance.temperature = 0.0
            pipeline_instance.top_p = 0.0
            MockPipeline.return_value = pipeline_instance
            mock_model_env.return_value.vision_model_file = "test-vision.gguf"

            ok = MagicMock()
            ok.choices = [MagicMock()]
            ok.choices[0].message.content = "Page text"

            engine = VisionOCREngine(
                "/fake/doc.pdf",
                timeout=10.0,
                max_retries=0,
                max_image_dimension=1024,
                max_tokens=4096,
            )
            budget = VisionOCREngine._MAX_CONSECUTIVE_PAGE_FAILURES

            timeout = RuntimeError("Request timed out.")
            # Fail every page but the one immediately before the budget runs out.
            side_effects: list[object] = []
            for _ in range(budget - 1):
                side_effects.extend([timeout, timeout])
            side_effects.append(ok)
            for _ in range(budget - 1):
                side_effects.extend([timeout, timeout])

            with patch.object(
                engine._vision_client.chat.completions,
                "create",
                side_effect=side_effects,
            ) as mock_create:
                for page_index in range(2 * budget - 1):
                    engine.ocr_page(page_index)

            # The successful page reset the counter, so every page was attempted.
            assert mock_create.call_count == len(side_effects)

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
            patch("docint.core.readers.documents.ocr.time.sleep"),
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
            patch("docint.core.readers.documents.ocr.time.sleep"),
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
            patch("docint.core.readers.documents.ocr.time.sleep"),
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
            response.choices[0].message.content = "<think>analyzing layout</think>ACTUAL_OCR_TEXT"

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
            patch("docint.core.readers.documents.ocr.time.sleep"),
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
            reasoning_only.choices[0].message.content = "<think>lots of reasoning and nothing else</think>"

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
            patch("docint.core.readers.documents.ocr.time.sleep"),
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
            refusal.choices[0].message.content = "I don't see any image attached to your message."

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
            patch("docint.core.readers.documents.ocr.time.sleep"),
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

    def test_process_with_mocked_pdf(self, pipeline_config: PipelineConfig, tmp_path: Path) -> None:
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

    def test_idempotent_rerun(self, pipeline_config: PipelineConfig, tmp_path: Path) -> None:
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

    def test_page_failure_isolation(self, pipeline_config: PipelineConfig, tmp_path: Path) -> None:
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

        def flaky_fn() -> str:
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
        mock_image_obj.get.side_effect = lambda k, d="": "/Image" if k == "/Subtype" else d
        mock_image_obj.get_object.return_value = mock_image_obj
        mock_xobj_dict = {"/Im1": mock_image_obj}
        mock_xobj = MagicMock()
        mock_xobj.get_object.return_value = mock_xobj_dict
        mock_resources = MagicMock()
        mock_resources.get.side_effect = lambda k, d=None: mock_xobj if k == "/XObject" else d
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
        lines = [line for line in chunks_path.read_text().strip().split("\n") if line.strip()]
        assert len(lines) >= 1
        import json

        chunk_data = json.loads(lines[0])
        assert "vision OCR" in chunk_data["text"].lower() or "scanned" in chunk_data["text"].lower()
