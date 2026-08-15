"""Tests for the docling-parse backbone shared by the PDF pipeline stages."""

from __future__ import annotations

from pathlib import Path

import pytest
from pdf_fixtures import ImageBox, PageSpec, TextRun, build_pdf, two_column_page

from docint.core.readers.documents.models import BBox
from docint.core.readers.documents.parse import (
    ParsedPdf,
    TextLine,
    lines_to_text,
    order_lines,
)


def _line(text: str, x0: float, y0: float, x1: float, y1: float, size: float = 10.0) -> TextLine:
    """Build a ``TextLine`` with a plain bbox for ordering tests."""
    return TextLine(text=text, bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1), font_name="/Helvetica", font_size=size)


class TestParsedPdf:
    """Opening a PDF and reading its pages."""

    def test_page_count_and_dimensions(self, tmp_path: Path) -> None:
        """Page count and per-page dimensions come from the parsed document."""
        pdf = tmp_path / "two.pdf"
        pdf.write_bytes(build_pdf([PageSpec(), PageSpec(width=400, height=300)]))
        with ParsedPdf(pdf) as parsed:
            assert parsed.page_count == 2
            page = parsed.page(1)
            assert page.page_index == 1
            assert (page.width, page.height) == (400.0, 300.0)

    def test_lines_carry_text_bbox_and_font(self, tmp_path: Path) -> None:
        """Line cells expose text, a bottom-left-origin bbox and the font name/size."""
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
        """Embedded bitmaps report their on-page placement rectangle."""
        pdf = tmp_path / "img.pdf"
        pdf.write_bytes(build_pdf([PageSpec(images=[ImageBox(x=100, y=400, w=200, h=100)])]))
        with ParsedPdf(pdf) as parsed:
            page = parsed.page(0)
        assert len(page.images) == 1
        img = page.images[0]
        assert img.index == 0
        assert (img.bbox.x0, img.bbox.y0, img.bbox.x1, img.bbox.y1) == pytest.approx((100.0, 400.0, 300.0, 500.0))

    def test_empty_page_has_no_lines_or_images(self, tmp_path: Path) -> None:
        """A blank page yields no lines and no images."""
        pdf = tmp_path / "empty.pdf"
        pdf.write_bytes(build_pdf([PageSpec()]))
        with ParsedPdf(pdf) as parsed:
            page = parsed.page(0)
        assert page.lines == []
        assert page.images == []

    def test_unreadable_file_raises(self, tmp_path: Path) -> None:
        """Opening a non-PDF raises instead of returning an empty document."""
        bogus = tmp_path / "bogus.pdf"
        bogus.write_bytes(b"not a pdf at all")
        with pytest.raises(RuntimeError, match="Failed to load"):
            ParsedPdf(bogus)


class TestOrderLines:
    """Reading order via XY-cut."""

    def test_two_columns_read_left_column_first(self) -> None:
        """Two side-by-side columns are read left column first, then right."""
        lines = [
            _line("R1", 330, 700, 450, 710),
            _line("L1", 60, 700, 180, 710),
            _line("R2", 330, 686, 450, 696),
            _line("L2", 60, 686, 180, 696),
        ]
        assert [ln.text for ln in order_lines(lines)] == ["L1", "L2", "R1", "R2"]

    def test_full_width_heading_precedes_both_columns(self) -> None:
        """A full-width line above two columns is read before either column."""
        lines = [
            _line("R1", 330, 700, 450, 710),
            _line("L1", 60, 700, 180, 710),
            _line("Heading", 60, 740, 450, 756, size=16),
        ]
        assert [ln.text for ln in order_lines(lines)] == ["Heading", "L1", "R1"]

    def test_single_column_is_top_to_bottom(self) -> None:
        """A single column reads top to bottom regardless of input order."""
        lines = [_line("B", 60, 680, 300, 690), _line("A", 60, 700, 300, 710), _line("C", 60, 660, 300, 670)]
        assert [ln.text for ln in order_lines(lines)] == ["A", "B", "C"]

    def test_empty_input(self) -> None:
        """No lines in, no lines out."""
        assert order_lines([]) == []


class TestLineMerging:
    """Horizontally adjacent cells on one baseline are one line."""

    def test_section_number_and_title_merge(self, tmp_path: Path) -> None:
        """A number and its title drawn as separate runs on one baseline become one line."""
        pdf = tmp_path / "num.pdf"
        pdf.write_bytes(
            build_pdf(
                [
                    PageSpec(
                        runs=[
                            TextRun("1", x=60, y=700, size=12, bold=True),
                            TextRun("Introduction", x=78, y=700, size=12, bold=True),
                            TextRun("Body text follows on the next line.", x=60, y=680),
                        ]
                    )
                ]
            )
        )
        with ParsedPdf(pdf) as parsed:
            texts = [ln.text for ln in parsed.page(0).lines]
        assert "1 Introduction" in texts
        assert len(texts) == 2

    def test_distant_cells_on_one_baseline_stay_separate(self, tmp_path: Path) -> None:
        """Two columns' first lines share a baseline but are far apart — never merged."""
        pdf = tmp_path / "cols.pdf"
        pdf.write_bytes(build_pdf([two_column_page()]))
        with ParsedPdf(pdf) as parsed:
            texts = [ln.text for ln in parsed.page(0).lines]
        assert "Left column line 1" in texts and "Right column line 1" in texts

    def test_rotated_line_keeps_true_font_size(self, tmp_path: Path) -> None:
        """A 90-degree stamp reports its real font size and is flagged rotated."""
        pdf = tmp_path / "rot.pdf"
        pdf.write_bytes(
            build_pdf(
                [
                    PageSpec(
                        runs=[
                            TextRun("Vertical side stamp text", x=30, y=300, size=12, rotate90=True),
                            TextRun("Upright body line.", x=60, y=700, size=12),
                        ]
                    )
                ]
            )
        )
        with ParsedPdf(pdf) as parsed:
            by_text = {ln.text: ln for ln in parsed.page(0).lines}
        stamp = by_text["Vertical side stamp text"]
        body = by_text["Upright body line."]
        assert stamp.rotated is True and body.rotated is False
        assert stamp.font_size == pytest.approx(body.font_size, abs=1.5)
        assert stamp.bbox.y1 - stamp.bbox.y0 > 60  # the axis-aligned box is tall


class TestLinesToText:
    """Joining ordered lines into page text."""

    def test_lines_joined_with_newlines(self) -> None:
        """Adjacent lines are joined with a single newline."""
        lines = [_line("A", 60, 700, 300, 710), _line("B", 60, 686, 300, 696)]
        assert lines_to_text(lines) == "A\nB"

    def test_large_vertical_gap_becomes_paragraph_break(self) -> None:
        """A vertical gap well above line height inserts a blank line."""
        lines = [_line("A", 60, 700, 300, 710), _line("B", 60, 640, 300, 650)]
        assert lines_to_text(lines) == "A\n\nB"


class TestDehyphenation:
    """Soft hyphens at a line break are joined back into one word."""

    def test_german_compound_is_joined(self) -> None:
        """``Bundes-`` + ``regierung`` becomes one word, no hyphen, no newline."""
        lines = [_line("Die Bundes-", 60, 700, 300, 710), _line("regierung entschied.", 60, 686, 300, 696)]
        assert lines_to_text(lines) == "Die Bundesregierung entschied."

    def test_english_word_is_joined(self) -> None:
        """A hyphen splitting an ordinary English word is removed."""
        lines = [_line("inter-", 60, 700, 300, 710), _line("national trade", 60, 686, 300, 696)]
        assert lines_to_text(lines) == "international trade"

    def test_uppercase_continuation_keeps_the_hyphen(self) -> None:
        """``Ost-`` + ``West`` is a real compound, not a wrap: hyphen and break stay."""
        lines = [_line("Ost-", 60, 700, 300, 710), _line("West-Konflikt", 60, 686, 300, 696)]
        assert lines_to_text(lines) == "Ost-\nWest-Konflikt"

    def test_digit_continuation_keeps_the_hyphen(self) -> None:
        """A range like ``1990-`` / ``1995`` is not a wrapped word."""
        lines = [_line("Zeitraum 1990-", 60, 700, 300, 710), _line("1995 insgesamt", 60, 686, 300, 696)]
        assert lines_to_text(lines) == "Zeitraum 1990-\n1995 insgesamt"

    def test_en_dash_is_not_a_soft_hyphen(self) -> None:
        """An en dash ends a clause; it never joins two lines."""
        lines = [_line("der Vertrag –", 60, 700, 300, 710), _line("so hiess es", 60, 686, 300, 696)]
        assert lines_to_text(lines) == "der Vertrag –\nso hiess es"

    def test_paragraph_break_never_joins(self) -> None:
        """A hyphen before a paragraph gap keeps the blank line (and the hyphen)."""
        lines = [_line("Kapitel-", 60, 700, 300, 710), _line("uebersicht folgt", 60, 620, 300, 630)]
        assert lines_to_text(lines) == "Kapitel-\n\nuebersicht folgt"

    def test_unicode_hyphen_variants_are_joined(self) -> None:
        """U+2010 HYPHEN and U+00AD SOFT HYPHEN behave like ASCII ``-``."""
        assert (
            lines_to_text([_line("Fach\u2010", 60, 700, 300, 710), _line("bereich", 60, 686, 300, 696)])
            == "Fachbereich"
        )
        assert (
            lines_to_text([_line("Fach\u00ad", 60, 700, 300, 710), _line("bereich", 60, 686, 300, 696)])
            == "Fachbereich"
        )

    def test_lone_hyphen_line_is_left_alone(self) -> None:
        """A line that is only a hyphen has no word to join."""
        lines = [_line("-", 60, 700, 300, 710), _line("bullet item", 60, 686, 300, 696)]
        assert lines_to_text(lines) == "-\nbullet item"


class TestRealTwoColumnPage:
    """End-to-end: a real two-column PDF reads column by column."""

    def test_two_column_pdf_text_order(self, tmp_path: Path) -> None:
        """A real two-column PDF's text keeps the left column ahead of the right."""
        pdf = tmp_path / "cols.pdf"
        pdf.write_bytes(build_pdf([two_column_page()]))
        with ParsedPdf(pdf) as parsed:
            text = lines_to_text(order_lines(parsed.page(0).lines))
        assert text.index("Left column line 3") < text.index("Right column line 1")
