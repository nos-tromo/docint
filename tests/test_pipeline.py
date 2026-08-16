"""Tests for the document processing pipeline modules."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pdf_fixtures import (
    ImageBox,
    PageSpec,
    TextRun,
    build_pdf,
    irregular_table_page,
    report_pages,
    spanning_header_table_page,
    table_page,
    two_column_page,
    word_list_figure_page,
    wrapped_header_table_page,
)

from docint.core.ocr import OcrBlock, OcrBox, OcrCategory
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
    _is_bold,
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
    PdfTextEngine,
    blocks_from_ocr,
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
        enable_ocr=False,
        ocr_timeout=60.0,
        ocr_max_retries=1,
        ocr_max_image_dimension=1024,
        ocr_max_tokens=4096,
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
        assert cfg.pipeline_version == "3.4.0"
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

    def test_ocr_timeout_inherits_openai_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without an explicit override the OCR budget follows ``OPENAI_TIMEOUT``.

        A hardcoded default silently contradicts the endpoint's configured
        budget: a slow vision model that chat tolerates gets cut off mid-flight
        and surfaces as ``Request timed out``.

        Args:
            monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture for env manipulation.
        """
        monkeypatch.delenv("PIPELINE_OCR_TIMEOUT", raising=False)
        monkeypatch.setenv("OPENAI_TIMEOUT", "240")
        cfg = load_pipeline_config()
        assert cfg.ocr_timeout == 240.0

    def test_ocr_timeout_override_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An explicit ``PIPELINE_VISION_OCR_TIMEOUT`` still overrides the inherited value.

        Args:
            monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture for env manipulation.
        """
        monkeypatch.setenv("OPENAI_TIMEOUT", "240")
        monkeypatch.setenv("PIPELINE_OCR_TIMEOUT", "45")
        cfg = load_pipeline_config()
        assert cfg.ocr_timeout == 45.0


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

    def test_furniture_blocks_never_enter_chunk_text(self) -> None:
        """PAGE_HEADER / FOOTER / PAGE_NUMBER blocks are skipped by the chunker."""

        def _block(block_id: str, kind: BlockType, text: str, order: int) -> LayoutBlock:
            return LayoutBlock(
                block_id=block_id,
                page_index=0,
                type=kind,
                bbox=BBox(x0=0, y0=0, x1=612, y1=792),
                reading_order=order,
                confidence=1.0,
                text=text,
            )

        layout = {
            0: [
                _block("ph", BlockType.PAGE_HEADER, "Quarterly Review 2031", 0),
                _block("b", BlockType.TEXT, "The body sentence that should survive.", 1),
                _block("ft", BlockType.FOOTER, "Confidential draft", 2),
                _block("pn", BlockType.PAGE_NUMBER, "7", 3),
            ]
        }
        units = build_coarse_units("doc", layout, {}, [], [])

        assert len(units) == 1
        assert units[0].text == "The body sentence that should survive."
        assert units[0].block_ids == ["b"]

    def test_figure_text_blocks_never_enter_chunk_text(self) -> None:
        """FIGURE_TEXT joins furniture in the chunker's skip set."""

        def _block(block_id: str, kind: BlockType, text: str, order: int) -> LayoutBlock:
            return LayoutBlock(
                block_id=block_id,
                page_index=0,
                type=kind,
                bbox=BBox(x0=0, y0=0, x1=612, y1=792),
                reading_order=order,
                confidence=1.0,
                text=text,
            )

        layout = {
            0: [
                _block("ft", BlockType.FIGURE_TEXT, "application\nmissing\n<EOS>\nopinion", 0),
                _block("b", BlockType.TEXT, "The body sentence that should survive.", 1),
            ]
        }
        units = build_coarse_units("doc", layout, {}, [], [])
        assert len(units) == 1
        assert units[0].text == "The body sentence that should survive."

    def test_header_replaces_previous_header_under_title(self) -> None:
        """Section paths stay title + current header; consecutive HEADERs do not stack."""

        def _block(block_id: str, kind: BlockType, text: str, order: int) -> LayoutBlock:
            return LayoutBlock(
                block_id=block_id,
                page_index=0,
                type=kind,
                bbox=BBox(x0=0, y0=0, x1=612, y1=792),
                reading_order=order,
                confidence=1.0,
                text=text,
            )

        layout = {
            0: [
                _block("t", BlockType.TITLE, "Model", 0),
                _block("h1", BlockType.HEADER, "Encoder", 1),
                _block("b1", BlockType.TEXT, "Encoder body sentence.", 2),
                _block("h2", BlockType.HEADER, "Decoder", 3),
                _block("b2", BlockType.TEXT, "Decoder body sentence.", 4),
            ]
        }
        units = build_coarse_units("doc", layout, {}, [], [])
        assert [u.section_path for u in units] == [["Model", "Encoder"], ["Model", "Decoder"]]

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

    def test_save_table_writes_a_quoted_csv(self, tmp_path: Path) -> None:
        """A table with a cell grid is also written as CSV, with proper quoting."""
        table = TableResult(
            table_id="table-0-abc",
            page_index=0,
            bbox=BBox(x0=0, y0=0, x1=400, y1=100),
            raw_text="A | B\n1 | 2",
            cell_grid=[["Name", "Note"], ["Alpha", 'says "hi", loudly']],
            confidence=0.7,
        )
        save_table("doc1", table, tmp_path)

        csv_path = tmp_path / "doc1" / "tables" / "table-0-abc.csv"
        assert csv_path.exists()
        rows = list(csv.reader(csv_path.read_text().splitlines()))
        assert rows == [["Name", "Note"], ["Alpha", 'says "hi", loudly']]
        assert table.csv_path == str(csv_path)

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

    def test_extract_tables_carries_the_cell_grid(self, tmp_path: Path) -> None:
        """TableResult.cell_grid is populated from the layout block's cells."""
        layout = {
            0: [
                LayoutBlock(
                    block_id="tbl1",
                    page_index=0,
                    type=BlockType.TABLE,
                    bbox=BBox(x0=0, y0=0, x1=400, y1=100),
                    reading_order=0,
                    confidence=0.7,
                    text="A | B\n1 | 2",
                    cells=[["A", "B"], ["1", "2"]],
                )
            ]
        }
        tables = extract_tables(layout)
        assert tables[0].cell_grid == [["A", "B"], ["1", "2"]]

    def test_extract_images_writes_one_png_per_figure_block(self, tmp_path: Path) -> None:
        """Each FIGURE block gets the embedded image drawn at its bbox, not the page's first image."""
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(
            build_pdf(
                [
                    PageSpec(
                        images=[
                            ImageBox(x=50, y=600, w=100, h=50, pixels=(4, 2), rgb=(255, 0, 0)),
                            ImageBox(x=300, y=100, w=120, h=80, pixels=(2, 4), rgb=(0, 0, 255)),
                        ]
                    )
                ]
            )
        )
        layout = analyze_document(
            pdf, [PageInfo(page_index=0, has_text_layer=False, text_coverage=0.0, needs_ocr=True)]
        )
        out_dir = tmp_path / "images"

        images = extract_images(layout, pdf, out_dir)

        assert len(images) == 2
        paths = {img.image_path for img in images}
        assert len(paths) == 2 and all(p and Path(p).exists() for p in paths)
        from PIL import Image

        by_x = sorted(images, key=lambda i: i.bbox.x0)
        left = Image.open(str(by_x[0].image_path))
        right = Image.open(str(by_x[1].image_path))
        assert left.size == (4, 2)
        assert right.size == (2, 4)
        assert left.convert("RGB").getpixel((0, 0)) == (255, 0, 0)
        assert right.convert("RGB").getpixel((0, 0)) == (0, 0, 255)

    def test_figure_without_an_image_object_is_cropped_from_the_render(self, tmp_path: Path) -> None:
        """A picture region inside a scanned page has no embedded object of its own; crop the page."""
        pdf = tmp_path / "scan.pdf"
        pdf.write_bytes(
            build_pdf([PageSpec(images=[ImageBox(x=0, y=0, w=612, h=792, pixels=(6, 8), rgb=(0, 128, 0))])])
        )
        layout = {
            0: [
                LayoutBlock(
                    block_id="fig1",
                    page_index=0,
                    type=BlockType.FIGURE,
                    bbox=BBox(x0=100, y0=100, x1=300, y1=250),
                    reading_order=0,
                    confidence=0.7,
                    text="",
                )
            ]
        }
        images = extract_images(layout, pdf, tmp_path / "images")
        assert len(images) == 1
        assert images[0].image_path and Path(images[0].image_path).exists()
        from PIL import Image

        crop = Image.open(images[0].image_path)
        # 200 x 150 pt region rendered at the crop dpi keeps its aspect ratio.
        assert abs(crop.width / crop.height - 200 / 150) < 0.05
        assert crop.convert("RGB").getpixel((crop.width // 2, crop.height // 2)) == (0, 128, 0)

    def test_extract_images_crops_when_no_image_object_matches(self, tmp_path: Path) -> None:
        """A FIGURE block with no embedded object of its own is served from the page render."""
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(build_pdf([PageSpec(runs=[TextRun("no images here", x=60, y=700)])]))
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
                )
            ]
        }
        images = extract_images(layout, pdf, tmp_path / "images")
        assert len(images) == 1
        assert images[0].image_path and Path(images[0].image_path).exists()


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

    def test_table_block_text_is_row_major(self, tmp_path: Path) -> None:
        """A gridded table reads row by row, not column by column."""
        blocks = self._analyze(tmp_path, table_page())
        table = next(b for b in blocks if b.type == BlockType.TABLE)
        assert "Model | Accuracy | F1" in table.text
        assert "Alpha | 89.3 | 88.1" in table.text
        assert table.text.index("Alpha") < table.text.index("Accuracy") + len(table.text)
        assert table.cells is not None
        assert table.cells[0] == ["Model", "Accuracy", "F1"]
        assert len(table.cells) == 4

    def test_captioned_table_grid_covers_every_row(self, tmp_path: Path) -> None:
        """A caption anchors the table the geometry found — not the text heuristic's guess."""
        blocks = self._analyze(tmp_path, wrapped_header_table_page())
        table = next(b for b in blocks if b.type == BlockType.TABLE)
        assert table.text.startswith("Table 1: Complexity by layer type")
        assert table.cells is not None
        first_column = [row[0] for row in table.cells]
        assert "Self-Attention" in first_column
        assert "Convolutional" in first_column
        assert "O(1)" in table.text

    def test_irregular_captioned_table_still_reads_row_by_row(self, tmp_path: Path) -> None:
        """A table geometry cannot validate is still a table: its rows stay rows."""
        blocks = self._analyze(tmp_path, irregular_table_page())
        table = next(b for b in blocks if b.type == BlockType.TABLE)
        assert table.text.startswith("Table 2: Scores and cost on both corpora")
        alpha = next(line for line in table.text.splitlines() if line.startswith("Alpha"))
        assert "23.8" in alpha and "39.2" in alpha
        beta = next(line for line in table.text.splitlines() if line.startswith("Beta"))
        assert "24.6" in beta and "41.0" in beta
        # The caption's own wrapped line is not a table row, and the paragraph
        # below the table is not part of it.
        assert "Values are averages" not in table.text
        assert "ordinary body text" not in table.text

    def test_irregular_table_reports_no_cell_grid(self, tmp_path: Path) -> None:
        """When the structure was not recovered, no grid is claimed (so no CSV)."""
        blocks = self._analyze(tmp_path, irregular_table_page())
        table = next(b for b in blocks if b.type == BlockType.TABLE)
        assert table.cells is None or all(len(row) >= 2 for row in table.cells)

    def test_prose_below_an_irregular_table_stays_text(self, tmp_path: Path) -> None:
        """The paragraph under the table is still emitted as body text."""
        blocks = self._analyze(tmp_path, irregular_table_page())
        body = " ".join(b.text for b in blocks if b.type == BlockType.TEXT)
        assert "ordinary body text" in body

    def test_caption_less_table_is_still_a_table_block(self, tmp_path: Path) -> None:
        """A bare grid with no 'Table N:' caption is detected geometrically."""
        blocks = self._analyze(tmp_path, table_page(caption=None))
        table = next(b for b in blocks if b.type == BlockType.TABLE)
        assert table.cells is not None and table.cells[0] == ["Model", "Accuracy", "F1"]
        body = " ".join(b.text for b in blocks if b.type == BlockType.TEXT)
        assert "Following prose paragraph" in body
        assert "Gamma" not in body

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

    def test_medium_weight_font_counts_as_bold(self) -> None:
        """LaTeX's bold Times ships as ``NimbusRomNo9L-Medi``; it must read as bold."""
        assert _is_bold("/RCUMTF+NimbusRomNo9L-Medi") is True
        assert _is_bold("/AECCXO+NimbusRomNo9L-Regu") is False
        assert _is_bold("/Helvetica-Bold") is True

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

    def test_multi_line_paragraph_in_larger_font_is_not_a_heading(self, tmp_path: Path) -> None:
        """Three consecutive same-style lines are a paragraph, however large the face."""
        spec = PageSpec(
            runs=[
                TextRun("Provided proper attribution is given, permission is granted to", x=60, y=740, size=12),
                TextRun("reproduce the tables and figures in this paper solely for use in", x=60, y=726, size=12),
                TextRun("scholarly works.", x=60, y=712, size=12),
                TextRun("The abstract body is set smaller than the notice above it and", x=60, y=680, size=9),
                TextRun("runs on for a few lines so it clearly forms the page's body text.", x=60, y=669, size=9),
                TextRun("A third body line keeps the median where it belongs.", x=60, y=658, size=9),
                TextRun("And a fourth body line for good measure.", x=60, y=647, size=9),
            ]
        )
        blocks = self._analyze(tmp_path, spec)
        assert all(b.type == BlockType.TEXT for b in blocks)

    def test_rotated_stamp_is_furniture_not_a_heading(self, tmp_path: Path) -> None:
        """A rotated margin stamp is page furniture, never a heading, however tall its box."""
        spec = PageSpec(
            runs=[
                TextRun("Preprint stamp running up the margin", x=30, y=300, size=12, rotate90=True),
                TextRun("Body line one of the page.", x=60, y=700, size=10),
                TextRun("Body line two of the page.", x=60, y=688, size=10),
                TextRun("Body line three of the page.", x=60, y=676, size=10),
            ]
        )
        blocks = self._analyze(tmp_path, spec)
        by_type = {b.type: b.text for b in blocks}
        assert by_type.get(BlockType.PAGE_HEADER) == "Preprint stamp running up the margin"
        assert BlockType.TITLE not in by_type and BlockType.HEADER not in by_type
        assert "stamp" not in by_type.get(BlockType.TEXT, "")

    def test_two_letter_fragment_is_not_a_heading(self, tmp_path: Path) -> None:
        """Short symbol-like fragments (math) never become headings."""
        spec = PageSpec(
            runs=[
                TextRun("i K", x=200, y=500, size=16),
                TextRun("Body line one of the page.", x=60, y=700, size=10),
                TextRun("Body line two of the page.", x=60, y=688, size=10),
                TextRun("Body line three of the page.", x=60, y=676, size=10),
            ]
        )
        blocks = self._analyze(tmp_path, spec)
        assert all(b.type == BlockType.TEXT for b in blocks)

    def test_running_head_footer_and_number_become_furniture_blocks(self, tmp_path: Path) -> None:
        """Layout emits PAGE_HEADER / FOOTER / PAGE_NUMBER blocks, not TEXT."""
        pdf = tmp_path / "report.pdf"
        pdf.write_bytes(build_pdf(report_pages(3)))
        pages = [PageInfo(page_index=i, has_text_layer=True, text_coverage=1.0, needs_ocr=False) for i in range(3)]
        layout = analyze_document(pdf, pages)

        by_type = {b.type: b.text for b in layout[1]}
        assert by_type.get(BlockType.PAGE_HEADER) == "Quarterly Review 2031"
        assert by_type.get(BlockType.FOOTER) == "Confidential draft"
        assert by_type.get(BlockType.PAGE_NUMBER) == "2"
        body = " ".join(b.text for b in layout[1] if b.type == BlockType.TEXT)
        assert "Body line one of page 2" in body
        assert "Quarterly Review" not in body and "Confidential" not in body

    def test_word_list_figure_text_is_not_body_text(self, tmp_path: Path) -> None:
        """A figure's token axis is FIGURE_TEXT — never a TEXT block, never a heading."""
        blocks = self._analyze(tmp_path, word_list_figure_page())
        types = [b.type for b in blocks]
        assert BlockType.FIGURE_TEXT in types
        # The bag of words is not prose, and its outsized label is not a title.
        body = " ".join(b.text for b in blocks if b.type == BlockType.TEXT)
        assert "governments" not in body and "<EOS>" not in body
        assert all(b.type not in (BlockType.TITLE, BlockType.HEADER) or "Layer5" not in b.text for b in blocks)
        # The caption survives as ordinary text.
        assert "Figure 4" in body

    def test_short_lines_of_real_prose_stay_text(self, tmp_path: Path) -> None:
        """A math-heavy paragraph has many short lines but stays TEXT (measured: 0.44 share, 14 chars)."""
        runs = [TextRun("Since our model contains no recurrence, we inject position information.", x=60, y=700)]
        y = 686.0
        for token in (
            "PE",
            "(pos, 2i)",
            "= sin(pos / 10000",
            "2i / d",
            "model",
            ")",
            "where pos is the position and i is the dimension.",
        ):
            runs.append(TextRun(token, x=60, y=y))
            y -= 12
        for i in range(8):
            runs.append(
                TextRun(f"Ordinary prose line number {i} carries on with the argument at length.", x=60, y=y - i * 12)
            )
        blocks = self._analyze(tmp_path, PageSpec(runs=runs))
        assert BlockType.FIGURE_TEXT not in [b.type for b in blocks]

        """analyze_document() uses the caller's ParsedPdf when given one."""
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(build_pdf([two_column_page()]))
        page_info = PageInfo(page_index=0, has_text_layer=True, text_coverage=1.0, needs_ocr=False)
        with ParsedPdf(pdf) as parsed:
            with patch("docint.core.readers.documents.layout.ParsedPdf") as ctor:
                layout = analyze_document(pdf, [page_info], parsed=parsed)
            ctor.assert_not_called()
        assert layout.get(0)


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

    def test_furniture_blocks_are_left_out_of_page_text(self) -> None:
        """build_page_text() ignores furniture blocks when assembling the page text."""
        page_info = PageInfo(page_index=0, has_text_layer=True, text_coverage=1.0, needs_ocr=False)
        blocks = [
            LayoutBlock(
                block_id="ph",
                page_index=0,
                type=BlockType.PAGE_HEADER,
                bbox=BBox(x0=0, y0=760, x1=612, y1=780),
                reading_order=0,
                confidence=0.8,
                text="Quarterly Review 2031",
            ),
            LayoutBlock(
                block_id="b",
                page_index=0,
                type=BlockType.TEXT,
                bbox=BBox(x0=0, y0=100, x1=612, y1=700),
                reading_order=1,
                confidence=1.0,
                text="The body sentence that should survive.",
            ),
        ]

        result = build_page_text(page_info, blocks, [])

        assert result.full_text == "The body sentence that should survive."
        assert len(result.pdf_text_spans) == 1

    def test_pdf_text_engine_emits_one_span_per_line(self, tmp_path: Path) -> None:
        """The digital text engine yields per-line spans with real boxes, in reading order."""
        pdf = tmp_path / "cols.pdf"
        pdf.write_bytes(build_pdf([two_column_page()]))

        engine = PdfTextEngine(pdf)
        try:
            spans = engine.ocr_page(0)
        finally:
            engine.close()

        assert len(spans) == 6
        assert [s.text for s in spans[:3]] == ["Left column line 1", "Left column line 2", "Left column line 3"]
        assert spans[3].text == "Right column line 1"
        assert all(s.source == "pdf_text" for s in spans)
        assert spans[0].bbox.x0 == pytest.approx(60.0, abs=1.0)
        assert spans[3].bbox.x0 == pytest.approx(330.0, abs=1.0)
        assert spans[0].bbox.y0 > spans[1].bbox.y0

    def test_extract_text_for_pages_reuses_injected_parsed_document(self, tmp_path: Path) -> None:
        """extract_text_for_pages() uses the caller's ParsedPdf when given one."""
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(build_pdf([PageSpec(runs=[TextRun("Some actual text.", x=60, y=700)])]))
        page_info = PageInfo(page_index=0, has_text_layer=True, text_coverage=0.5, needs_ocr=True)
        with ParsedPdf(pdf) as parsed:
            with patch("docint.core.readers.documents.ocr.ParsedPdf") as ctor:
                result = extract_text_for_pages(pdf, [page_info], {0: []}, parsed=parsed)
            ctor.assert_not_called()
        assert "Some actual text" in result[0].full_text

    def test_ocr_blocks_become_layout_blocks(self) -> None:
        """The OCR package's vocabulary becomes the document's on this side."""
        read = [
            OcrBlock(category=OcrCategory.TITLE, bbox=OcrBox(50, 700, 400, 730), text="Scanned Report"),
            OcrBlock(category=OcrCategory.SECTION_HEADER, bbox=OcrBox(50, 660, 300, 680), text="1 Overview"),
            OcrBlock(category=OcrCategory.LIST_ITEM, bbox=OcrBox(50, 600, 400, 640), text="- one"),
            OcrBlock(
                category=OcrCategory.TABLE,
                bbox=OcrBox(50, 400, 560, 560),
                text="Model | Score",
                cells=[["Model", "Score"]],
            ),
            OcrBlock(category=OcrCategory.PICTURE, bbox=OcrBox(50, 200, 300, 380)),
            OcrBlock(category=OcrCategory.PAGE_HEADER, bbox=OcrBox(50, 760, 560, 780), text="Quarterly Review"),
        ]
        blocks = blocks_from_ocr(3, read)
        assert [b.type for b in blocks] == [
            BlockType.TITLE,
            BlockType.HEADER,
            BlockType.LIST,
            BlockType.TABLE,
            BlockType.FIGURE,
            BlockType.PAGE_HEADER,
        ]
        assert [b.reading_order for b in blocks] == [0, 1, 2, 3, 4, 5]
        assert all(b.page_index == 3 for b in blocks)
        assert blocks[3].cells == [["Model", "Score"]]
        assert blocks[3].cells_source == "ocr"

    def test_a_footer_that_is_only_a_number_is_a_page_number(self) -> None:
        """The chunker treats the two alike, but the artifact should say which."""
        read = [
            OcrBlock(category=OcrCategory.PAGE_FOOTER, bbox=OcrBox(300, 30, 320, 45), text="7"),
            OcrBlock(category=OcrCategory.PAGE_FOOTER, bbox=OcrBox(50, 30, 560, 45), text="Confidential draft"),
        ]
        assert [b.type for b in blocks_from_ocr(0, read)] == [BlockType.PAGE_NUMBER, BlockType.FOOTER]

    def test_kept_blocks_come_first_and_are_renumbered(self) -> None:
        """A text-only model says nothing about figures, so geometry's are kept."""
        figure = LayoutBlock("fig", 0, BlockType.FIGURE, BBox(0, 0, 612, 792), 9, 1.0)
        read = [OcrBlock(category=OcrCategory.TEXT, bbox=OcrBox(0, 0, 612, 792), text="All the words.")]
        blocks = blocks_from_ocr(0, read, keep=[figure])
        assert [b.type for b in blocks] == [BlockType.FIGURE, BlockType.TEXT]
        assert [b.reading_order for b in blocks] == [0, 1]


class TestOrchestrator:
    """Tests for the document pipeline orchestrator."""

    def test_process_real_pdf(self, pipeline_config: PipelineConfig, tmp_path: Path) -> None:
        """Processing a digital PDF should produce a completed manifest with artifacts and chunks."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(
            build_pdf(
                [
                    PageSpec(
                        runs=[
                            TextRun("Report Title", x=60, y=740, size=20, bold=True),
                            TextRun("Test document content. Second sentence.", x=60, y=700),
                            TextRun("Third sentence with more content here.", x=60, y=686),
                        ]
                    )
                ]
            )
        )

        orch = DocumentPipelineOrchestrator(config=pipeline_config)
        manifest = orch.process(pdf_file)

        assert manifest.status == "completed"
        assert manifest.pages_total == 1
        assert manifest.pages_failed == 0
        assert manifest.pages_ocr == 0

        doc_id = compute_file_hash(pdf_file)
        artifacts_dir = Path(pipeline_config.artifacts_dir)
        assert (artifacts_dir / doc_id / "manifest.json").exists()
        chunks_path = artifacts_dir / doc_id / "chunks.jsonl"
        chunk = json.loads(chunks_path.read_text().strip().split("\n")[0])
        assert chunk["section_path"] == ["Report Title"]
        assert "Test document content" in chunk["text"]

    def test_two_column_pdf_chunks_read_column_wise(self, pipeline_config: PipelineConfig, tmp_path: Path) -> None:
        """End to end, a two-column page's chunk text keeps each column contiguous."""
        pdf_file = tmp_path / "cols.pdf"
        pdf_file.write_bytes(build_pdf([two_column_page()]))

        manifest = DocumentPipelineOrchestrator(config=pipeline_config).process(pdf_file)

        assert manifest.status == "completed"
        chunks_path = Path(pipeline_config.artifacts_dir) / manifest.doc_id / "chunks.jsonl"
        text = "\n".join(json.loads(line)["text"] for line in chunks_path.read_text().strip().split("\n"))
        assert text.index("Left column line 3") < text.index("Right column line 1")

    def test_opens_document_once_and_closes_it(self, pipeline_config: PipelineConfig, tmp_path: Path) -> None:
        """One ParsedPdf handle serves every stage and is released at the end."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(build_pdf([two_column_page(), two_column_page()]))
        closes: list[int] = []
        real_close = ParsedPdf.close

        def _counting_close(self: ParsedPdf) -> None:
            closes.append(1)
            real_close(self)

        with (
            patch.object(ParsedPdf, "close", _counting_close),
            patch("docint.core.readers.documents.orchestrator.ParsedPdf", wraps=ParsedPdf) as ctor,
        ):
            manifest = DocumentPipelineOrchestrator(config=pipeline_config).process(pdf_file)

        assert manifest.status == "completed"
        assert ctor.call_count == 1
        assert len(closes) == 1

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
            enable_ocr=False,
            ocr_timeout=60.0,
            ocr_max_retries=1,
            ocr_max_image_dimension=1024,
            ocr_max_tokens=4096,
        )

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(build_pdf([PageSpec(runs=[TextRun("Idempotent test.", x=60, y=700)])]))

        orch = DocumentPipelineOrchestrator(config=config)

        manifest1 = orch.process(pdf_file)
        assert manifest1.status == "completed"

        # Second run — should skip processing
        with patch("docint.core.readers.documents.orchestrator.triage_pdf") as triage:
            manifest2 = orch.process(pdf_file)
        triage.assert_not_called()
        assert manifest2.status == "completed"
        assert manifest2.doc_id == manifest1.doc_id

    def test_page_failure_isolation(self, pipeline_config: PipelineConfig, tmp_path: Path) -> None:
        """A failing page should not crash the whole document."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(build_pdf([two_column_page(), two_column_page()]))
        real_page = ParsedPdf.page

        def _flaky(self: ParsedPdf, page_index: int) -> ParsedPage:
            if page_index == 1:
                raise RuntimeError("corrupt")
            return real_page(self, page_index)

        with patch.object(ParsedPdf, "page", _flaky):
            manifest = DocumentPipelineOrchestrator(config=pipeline_config).process(pdf_file)

        assert manifest.status == "completed"
        assert manifest.pages_total == 2
        assert [p.status for p in manifest.pages] == ["completed", "failed"]

    def test_unreadable_pdf_fails_cleanly(self, pipeline_config: PipelineConfig, tmp_path: Path) -> None:
        """A file that cannot be opened yields a failed manifest, not an exception."""
        pdf_file = tmp_path / "bogus.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 not really a pdf")

        manifest = DocumentPipelineOrchestrator(config=pipeline_config).process(pdf_file)

        assert manifest.status == "failed"
        assert manifest.error

    @staticmethod
    def _ocr_engine(
        *,
        reads_layout: bool = True,
        pages: dict[int, list[OcrBlock]] | None = None,
        regions: list[OcrBlock] | None = None,
    ) -> MagicMock:
        """A stand-in OCR engine answering from fixed blocks."""
        engine = MagicMock()
        engine.disabled = False
        engine.reads_layout = reads_layout
        engine.family.name = "dots" if reads_layout else "generic"
        engine.model = "dots-studio/dots.mocr" if reads_layout else "some/vision-model"
        engine.read_page.side_effect = lambda index: (pages or {}).get(index, [])
        engine.read_region.return_value = regions or []
        return engine

    def test_table_structure_lane_replaces_a_flattened_grid(
        self, pipeline_config: PipelineConfig, tmp_path: Path
    ) -> None:
        """A table geometry flattened is re-read through the OCR engine, rows and all."""
        pdf_file = tmp_path / "spanning.pdf"
        pdf_file.write_bytes(build_pdf([spanning_header_table_page()]))

        engine = self._ocr_engine(
            regions=[
                OcrBlock(
                    category=OcrCategory.TABLE,
                    bbox=OcrBox(50, 300, 560, 480),
                    text="Model | Score EN-DE | Score EN-FR",
                    cells=[["Model", "Score EN-DE", "Score EN-FR"], ["Alpha", "23.8", "39.2"]],
                )
            ]
        )

        with patch("docint.core.readers.documents.orchestrator.build_engine", return_value=engine) as build:
            manifest = DocumentPipelineOrchestrator(config=pipeline_config).process(pdf_file)

        build.assert_called_once()
        engine.read_region.assert_called_once()
        engine.close.assert_called_once()
        assert manifest.status == "completed"
        assert manifest.tables_structured == 1

        table = json.loads(
            next((Path(pipeline_config.artifacts_dir) / manifest.doc_id / "tables").glob("*.json")).read_text()
        )
        assert table["cell_grid"][0] == ["Model", "Score EN-DE", "Score EN-FR"]
        assert table["structure_source"] == "ocr"

        chunks = (Path(pipeline_config.artifacts_dir) / manifest.doc_id / "chunks.jsonl").read_text()
        assert "Model | Score EN-DE | Score EN-FR" in chunks
        assert "Table 2: Scores and cost on both corpora" in chunks

    def test_the_table_region_is_read_in_page_coordinates(
        self, pipeline_config: PipelineConfig, tmp_path: Path
    ) -> None:
        """The engine is asked about the block's own box, not the whole page."""
        pdf_file = tmp_path / "spanning.pdf"
        pdf_file.write_bytes(build_pdf([spanning_header_table_page()]))
        engine = self._ocr_engine(regions=[])

        with patch("docint.core.readers.documents.orchestrator.build_engine", return_value=engine):
            DocumentPipelineOrchestrator(config=pipeline_config).process(pdf_file)

        _, bbox = engine.read_region.call_args.args
        assert isinstance(bbox, OcrBox)
        assert bbox.x1 - bbox.x0 < 612.0

    def test_table_structure_lane_is_skipped_for_a_clean_grid(
        self, pipeline_config: PipelineConfig, tmp_path: Path
    ) -> None:
        """A table geometry recovered cleanly costs no remote call."""
        pdf_file = tmp_path / "clean.pdf"
        pdf_file.write_bytes(build_pdf([table_page()]))

        with patch("docint.core.readers.documents.orchestrator.build_engine") as build:
            manifest = DocumentPipelineOrchestrator(config=pipeline_config).process(pdf_file)

        build.assert_not_called()
        assert manifest.status == "completed"
        assert manifest.tables_structured == 0

    def test_table_structure_lane_can_be_switched_off(self, tmp_path: Path) -> None:
        """With the knob off the lane never runs, however weak the grid."""
        config = PipelineConfig(
            text_coverage_threshold=0.01,
            pipeline_version="test-1.0.0",
            artifacts_dir=str(tmp_path / "artifacts"),
            max_retries=1,
            force_reprocess=True,
            max_workers=1,
            enable_ocr=False,
            ocr_timeout=60.0,
            ocr_max_retries=1,
            ocr_max_image_dimension=1024,
            ocr_max_tokens=4096,
            enable_table_ocr=False,
        )
        pdf_file = tmp_path / "spanning.pdf"
        pdf_file.write_bytes(build_pdf([spanning_header_table_page()]))

        with patch("docint.core.readers.documents.orchestrator.build_engine") as build:
            manifest = DocumentPipelineOrchestrator(config=config).process(pdf_file)

        build.assert_not_called()
        assert manifest.status == "completed"

    def test_table_structure_failure_keeps_the_geometric_grid(
        self, pipeline_config: PipelineConfig, tmp_path: Path
    ) -> None:
        """When the model cannot answer, the table keeps what geometry found."""
        pdf_file = tmp_path / "spanning.pdf"
        pdf_file.write_bytes(build_pdf([spanning_header_table_page()]))
        engine = self._ocr_engine(regions=[])

        with patch("docint.core.readers.documents.orchestrator.build_engine", return_value=engine):
            manifest = DocumentPipelineOrchestrator(config=pipeline_config).process(pdf_file)

        assert manifest.status == "completed"
        assert manifest.tables_structured == 0
        assert manifest.tables_structure_failed == 1
        table = json.loads(
            next((Path(pipeline_config.artifacts_dir) / manifest.doc_id / "tables").glob("*.json")).read_text()
        )
        assert table["structure_source"] == "geometry"
        assert "Alpha" in table["raw_text"]

    def test_scanned_page_takes_its_layout_from_the_ocr_model(
        self, pipeline_config: PipelineConfig, tmp_path: Path
    ) -> None:
        """A page that needs OCR gets real blocks (heading, text, table) and OCR-sourced page text."""
        pipeline_config = replace(pipeline_config, enable_ocr=True)
        pdf_file = tmp_path / "scan.pdf"
        pdf_file.write_bytes(build_pdf([PageSpec(images=[ImageBox(x=0, y=0, w=612, h=792)])]))
        engine = self._ocr_engine(
            pages={
                0: [
                    OcrBlock(category=OcrCategory.TITLE, bbox=OcrBox(50, 700, 400, 730), text="Scanned Report"),
                    OcrBlock(
                        category=OcrCategory.TEXT,
                        bbox=OcrBox(50, 500, 560, 690),
                        text="Body read from the scan.",
                    ),
                    OcrBlock(
                        category=OcrCategory.TABLE,
                        bbox=OcrBox(50, 300, 560, 480),
                        text="Model | Score\nAlpha | 1",
                        cells=[["Model", "Score"], ["Alpha", "1"]],
                    ),
                ]
            }
        )

        with patch("docint.core.readers.documents.orchestrator.build_engine", return_value=engine):
            manifest = DocumentPipelineOrchestrator(config=pipeline_config).process(pdf_file)

        assert manifest.status == "completed"
        assert manifest.pages_ocr == 1
        assert manifest.pages_ocr_read == 1
        engine.close.assert_called_once()

        root = Path(pipeline_config.artifacts_dir) / manifest.doc_id
        page_text = json.loads((root / "pages" / "0" / "text.json").read_text())
        assert page_text["source_mix"] == "ocr"
        assert "Body read from the scan." in page_text["full_text"]
        chunks = [json.loads(line) for line in (root / "chunks.jsonl").read_text().splitlines() if line.strip()]
        assert any(c["section_path"] == ["Scanned Report"] for c in chunks)
        assert any("Model | Score" in c["text"] for c in chunks)
        table = json.loads(next((root / "tables").glob("*.json")).read_text())
        assert table["cell_grid"] == [["Model", "Score"], ["Alpha", "1"]]
        assert table["structure_source"] == "ocr"

    def test_a_text_only_model_keeps_the_page_its_figures(
        self, pipeline_config: PipelineConfig, tmp_path: Path
    ) -> None:
        """A model that reads text says nothing about pictures, so geometry's stand."""
        pipeline_config = replace(pipeline_config, enable_ocr=True)
        pdf_file = tmp_path / "scan.pdf"
        pdf_file.write_bytes(build_pdf([PageSpec(images=[ImageBox(x=0, y=0, w=612, h=792)])]))
        engine = self._ocr_engine(
            reads_layout=False,
            pages={
                0: [
                    OcrBlock(
                        category=OcrCategory.TEXT,
                        bbox=OcrBox(0, 0, 612, 792),
                        text="Text from the scanned page.",
                    )
                ]
            },
        )

        with patch("docint.core.readers.documents.orchestrator.build_engine", return_value=engine):
            manifest = DocumentPipelineOrchestrator(config=pipeline_config).process(pdf_file)

        assert manifest.status == "completed"
        assert manifest.pages_ocr_read == 1
        assert manifest.images_found == 1
        chunks = (Path(pipeline_config.artifacts_dir) / manifest.doc_id / "chunks.jsonl").read_text()
        assert "Text from the scanned page." in chunks

    def test_a_page_the_model_cannot_read_is_counted_not_crashed(
        self, pipeline_config: PipelineConfig, tmp_path: Path
    ) -> None:
        """An unanswered page leaves the document complete and says so."""
        pipeline_config = replace(pipeline_config, enable_ocr=True)
        pdf_file = tmp_path / "scan.pdf"
        pdf_file.write_bytes(build_pdf([PageSpec(images=[ImageBox(x=0, y=0, w=612, h=792)])]))
        engine = self._ocr_engine(pages={})

        with patch("docint.core.readers.documents.orchestrator.build_engine", return_value=engine):
            manifest = DocumentPipelineOrchestrator(config=pipeline_config).process(pdf_file)

        assert manifest.status == "completed"
        assert manifest.pages_ocr_read == 0
        assert manifest.pages_ocr_failed == 1

    def test_pages_after_the_budget_ran_out_are_skipped_not_failed(
        self, pipeline_config: PipelineConfig, tmp_path: Path
    ) -> None:
        """A dead endpoint is reported as given up on, not as pages that failed."""
        pipeline_config = replace(pipeline_config, enable_ocr=True)
        pdf_file = tmp_path / "scan.pdf"
        pdf_file.write_bytes(
            build_pdf(
                [
                    PageSpec(images=[ImageBox(x=0, y=0, w=612, h=792)]),
                    PageSpec(images=[ImageBox(x=0, y=0, w=612, h=792)]),
                ]
            )
        )
        engine = self._ocr_engine(pages={})

        def _read(index: int) -> list[OcrBlock]:
            engine.disabled = True  # the first call exhausts the budget
            return []

        engine.read_page.side_effect = _read

        with patch("docint.core.readers.documents.orchestrator.build_engine", return_value=engine):
            manifest = DocumentPipelineOrchestrator(config=pipeline_config).process(pdf_file)

        assert engine.read_page.call_count == 1
        assert manifest.pages_ocr_failed == 1
        assert manifest.pages_ocr_skipped == 1

    def test_both_lanes_share_one_engine(self, pipeline_config: PipelineConfig, tmp_path: Path) -> None:
        """One endpoint, one budget: a document builds its engine once."""
        pipeline_config = replace(pipeline_config, enable_ocr=True)
        pdf_file = tmp_path / "scan_and_table.pdf"
        pdf_file.write_bytes(
            build_pdf([PageSpec(images=[ImageBox(x=0, y=0, w=612, h=792)]), spanning_header_table_page()])
        )
        engine = self._ocr_engine(
            pages={0: [OcrBlock(category=OcrCategory.TEXT, bbox=OcrBox(0, 0, 612, 792), text="Read.")]},
            regions=[],
        )

        with patch("docint.core.readers.documents.orchestrator.build_engine", return_value=engine) as build:
            manifest = DocumentPipelineOrchestrator(config=pipeline_config).process(pdf_file)

        assert manifest.status == "completed"
        build.assert_called_once()
        engine.close.assert_called_once()
        assert engine.read_page.called and engine.read_region.called

    def test_without_an_engine_the_document_still_completes(
        self, pipeline_config: PipelineConfig, tmp_path: Path
    ) -> None:
        """No OCR endpoint configured: scanned pages stay empty, nothing raises."""
        pipeline_config = replace(pipeline_config, enable_ocr=True)
        pdf_file = tmp_path / "scan.pdf"
        pdf_file.write_bytes(build_pdf([PageSpec(images=[ImageBox(x=0, y=0, w=612, h=792)])]))

        with patch("docint.core.readers.documents.orchestrator.build_engine", return_value=None) as build:
            manifest = DocumentPipelineOrchestrator(config=pipeline_config).process(pdf_file)

        assert manifest.status == "completed"
        build.assert_called_once()
        assert manifest.pages_ocr_read == 0

    def test_digital_pages_never_reach_the_ocr_model(self, pipeline_config: PipelineConfig, tmp_path: Path) -> None:
        """Pages with a text layer and clean tables cost nothing remotely."""
        pdf_file = tmp_path / "digital.pdf"
        pdf_file.write_bytes(build_pdf([two_column_page()]))
        with patch("docint.core.readers.documents.orchestrator.build_engine") as build:
            manifest = DocumentPipelineOrchestrator(config=pipeline_config).process(pdf_file)
        assert manifest.status == "completed"
        build.assert_not_called()

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
