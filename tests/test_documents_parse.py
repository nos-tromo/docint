"""Tests for the docling-parse backbone shared by the PDF pipeline stages."""

from __future__ import annotations

from pathlib import Path

import pytest

from docint.core.readers.documents.models import BBox
from docint.core.readers.documents.parse import (
    ParsedPdf,
    TextLine,
    lines_to_text,
    order_lines,
)
from pdf_fixtures import ImageBox, PageSpec, TextRun, build_pdf, two_column_page


def _line(text: str, x0: float, y0: float, x1: float, y1: float, size: float = 10.0) -> TextLine:
    """Build a ``TextLine`` with a plain bbox for ordering tests."""
    return TextLine(text=text, bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1), font_name="/Helvetica", font_size=size)


class TestParsedPdf:
    """Opening a PDF and reading its pages."""

    def test_page_count_and_dimensions(self, tmp_path: Path) -> None:
        pdf = tmp_path / "two.pdf"
        pdf.write_bytes(build_pdf([PageSpec(), PageSpec(width=400, height=300)]))
        with ParsedPdf(pdf) as parsed:
            assert parsed.page_count == 2
            page = parsed.page(1)
            assert page.page_index == 1
            assert (page.width, page.height) == (400.0, 300.0)

    def test_lines_carry_text_bbox_and_font(self, tmp_path: Path) -> None:
        pdf = tmp_path / "text.pdf"
        pdf.write_bytes(
            build_pdf(
                [
                    PageSpec(
                        runs=[
                            TextRun("Section Heading", x=60, y=740, size=18, bold=True),
                            TextRun("Body line one.", x=60, y=700, size=11),
                        ]
                    )
                ]
            )
        )
        with ParsedPdf(pdf) as parsed:
            page = parsed.page(0)
        by_text = {line.text: line for line in page.lines}
        heading = by_text["Section Heading"]
        body = by_text["Body line one."]
        assert "Bold" in heading.font_name
        assert heading.font_size > body.font_size
        # Bottom-left origin, points: the heading sits above the body line.
        assert heading.bbox.y0 > body.bbox.y1
        assert heading.bbox.x0 == pytest.approx(60.0, abs=1.0)

    def test_images_report_placement_bbox(self, tmp_path: Path) -> None:
        pdf = tmp_path / "img.pdf"
        pdf.write_bytes(build_pdf([PageSpec(images=[ImageBox(x=100, y=400, w=200, h=100)])]))
        with ParsedPdf(pdf) as parsed:
            page = parsed.page(0)
        assert len(page.images) == 1
        img = page.images[0]
        assert img.index == 0
        assert (img.bbox.x0, img.bbox.y0, img.bbox.x1, img.bbox.y1) == pytest.approx((100.0, 400.0, 300.0, 500.0))

    def test_empty_page_has_no_lines_or_images(self, tmp_path: Path) -> None:
        pdf = tmp_path / "empty.pdf"
        pdf.write_bytes(build_pdf([PageSpec()]))
        with ParsedPdf(pdf) as parsed:
            page = parsed.page(0)
        assert page.lines == []
        assert page.images == []

    def test_unreadable_file_raises(self, tmp_path: Path) -> None:
        bogus = tmp_path / "bogus.pdf"
        bogus.write_bytes(b"not a pdf at all")
        with pytest.raises(Exception):
            ParsedPdf(bogus)


class TestOrderLines:
    """Reading order via XY-cut."""

    def test_two_columns_read_left_column_first(self) -> None:
        lines = [
            _line("R1", 330, 700, 450, 710),
            _line("L1", 60, 700, 180, 710),
            _line("R2", 330, 686, 450, 696),
            _line("L2", 60, 686, 180, 696),
        ]
        assert [ln.text for ln in order_lines(lines)] == ["L1", "L2", "R1", "R2"]

    def test_full_width_heading_precedes_both_columns(self) -> None:
        lines = [
            _line("R1", 330, 700, 450, 710),
            _line("L1", 60, 700, 180, 710),
            _line("Heading", 60, 740, 450, 756, size=16),
        ]
        assert [ln.text for ln in order_lines(lines)] == ["Heading", "L1", "R1"]

    def test_single_column_is_top_to_bottom(self) -> None:
        lines = [_line("B", 60, 680, 300, 690), _line("A", 60, 700, 300, 710), _line("C", 60, 660, 300, 670)]
        assert [ln.text for ln in order_lines(lines)] == ["A", "B", "C"]

    def test_empty_input(self) -> None:
        assert order_lines([]) == []


class TestLinesToText:
    """Joining ordered lines into page text."""

    def test_lines_joined_with_newlines(self) -> None:
        lines = [_line("A", 60, 700, 300, 710), _line("B", 60, 686, 300, 696)]
        assert lines_to_text(lines) == "A\nB"

    def test_large_vertical_gap_becomes_paragraph_break(self) -> None:
        lines = [_line("A", 60, 700, 300, 710), _line("B", 60, 640, 300, 650)]
        assert lines_to_text(lines) == "A\n\nB"


class TestRealTwoColumnPage:
    """End-to-end: a real two-column PDF reads column by column."""

    def test_two_column_pdf_text_order(self, tmp_path: Path) -> None:
        pdf = tmp_path / "cols.pdf"
        pdf.write_bytes(build_pdf([two_column_page()]))
        with ParsedPdf(pdf) as parsed:
            text = lines_to_text(order_lines(parsed.page(0).lines))
        assert text.index("Left column line 3") < text.index("Right column line 1")
