"""Tests for running head / footer / page-number detection."""

from __future__ import annotations

from pathlib import Path

from pdf_fixtures import PageSpec, TextRun, build_pdf, report_pages, two_column_page

from docint.core.readers.documents.furniture import detect_furniture
from docint.core.readers.documents.models import BlockType
from docint.core.readers.documents.parse import ParsedPdf


def _classified(pdf_path: Path) -> list[dict[str, BlockType]]:
    """Return per page a mapping of line text -> furniture type."""
    with ParsedPdf(pdf_path) as parsed:
        furniture = detect_furniture(parsed)
        out: list[dict[str, BlockType]] = []
        for idx in range(parsed.page_count):
            page = parsed.page(idx)
            marks = furniture.get(idx, {})
            out.append({page.lines[i].text: kind for i, kind in marks.items()})
    return out


class TestRepeatedBands:
    """Text repeated in the top/bottom band of most pages is furniture."""

    def test_running_head_and_footer_and_page_numbers(self, tmp_path: Path) -> None:
        """A running head, a footer line and per-page numbers are all classified."""
        pdf = tmp_path / "report.pdf"
        pdf.write_bytes(build_pdf(report_pages(3)))

        per_page = _classified(pdf)

        assert len(per_page) == 3
        for page_no, marks in enumerate(per_page, start=1):
            assert marks.get("Quarterly Review 2031") == BlockType.PAGE_HEADER
            assert marks.get("Confidential draft") == BlockType.FOOTER
            assert marks.get(str(page_no)) == BlockType.PAGE_NUMBER
            assert not any(text.startswith("Body line") for text in marks)

    def test_body_text_in_the_band_is_not_furniture(self, tmp_path: Path) -> None:
        """A unique line in the header band on one page only stays body text."""
        pages = report_pages(3)
        pages[1].runs.append(TextRun("A one-off note near the top of page two only.", x=60, y=745, size=9))
        pdf = tmp_path / "report.pdf"
        pdf.write_bytes(build_pdf(pages))

        per_page = _classified(pdf)

        assert "A one-off note near the top of page two only." not in per_page[1]

    def test_page_numbers_vary_but_still_classify(self, tmp_path: Path) -> None:
        """Numbering styles ('- 3 -', 'Seite 4 von 9', 'iv') all read as page numbers."""
        specs = []
        for text in ("- 3 -", "Seite 4 von 9", "iv"):
            specs.append(
                PageSpec(
                    runs=[
                        TextRun("Body line one with sufficient prose to anchor the page.", x=60, y=700),
                        TextRun("Body line two with sufficient prose to anchor the page.", x=60, y=686),
                        TextRun(text, x=300, y=30, size=9),
                    ]
                )
            )
        pdf = tmp_path / "numbers.pdf"
        pdf.write_bytes(build_pdf(specs))

        per_page = _classified(pdf)

        assert per_page[0].get("- 3 -") == BlockType.PAGE_NUMBER
        assert per_page[1].get("Seite 4 von 9") == BlockType.PAGE_NUMBER
        assert per_page[2].get("iv") == BlockType.PAGE_NUMBER


class TestSinglePageAndMargins:
    """Single-page documents and rotated margin stamps."""

    def test_single_page_number_still_detected(self, tmp_path: Path) -> None:
        """With no repetition to measure, a bare number in the footer band still counts."""
        pdf = tmp_path / "one.pdf"
        pdf.write_bytes(
            build_pdf(
                [
                    PageSpec(
                        runs=[
                            TextRun("Body line one with sufficient prose to anchor the page.", x=60, y=700),
                            TextRun("Body line two with sufficient prose to anchor the page.", x=60, y=686),
                            TextRun("7", x=300, y=30, size=9),
                        ]
                    )
                ]
            )
        )

        assert _classified(pdf)[0].get("7") == BlockType.PAGE_NUMBER

    def test_single_page_head_is_not_furniture(self, tmp_path: Path) -> None:
        """One page gives no evidence of repetition, so a top line stays body text."""
        pdf = tmp_path / "one.pdf"
        pdf.write_bytes(
            build_pdf(
                [
                    PageSpec(
                        runs=[
                            TextRun("Some line near the top edge", x=60, y=760, size=9),
                            TextRun("Body line one with sufficient prose to anchor the page.", x=60, y=700),
                        ]
                    )
                ]
            )
        )

        assert "Some line near the top edge" not in _classified(pdf)[0]

    def test_rotated_margin_stamp_is_furniture(self, tmp_path: Path) -> None:
        """A rotated stamp in the left margin (arXiv style) is page furniture."""
        pdf = tmp_path / "stamp.pdf"
        pdf.write_bytes(
            build_pdf(
                [
                    PageSpec(
                        runs=[
                            TextRun("arXiv:2031.01234v2 [cs.CL] 2 Aug 2031", x=30, y=300, size=9, rotate90=True),
                            TextRun("Body line one with sufficient prose to anchor the page.", x=120, y=700),
                            TextRun("Body line two with sufficient prose to anchor the page.", x=120, y=686),
                        ]
                    )
                ]
            )
        )

        marks = _classified(pdf)[0]
        assert marks.get("arXiv:2031.01234v2 [cs.CL] 2 Aug 2031") == BlockType.PAGE_HEADER

    def test_plain_document_has_no_furniture(self, tmp_path: Path) -> None:
        """A page of ordinary body text yields nothing."""
        pdf = tmp_path / "plain.pdf"
        pdf.write_bytes(build_pdf([two_column_page(), two_column_page()]))

        assert all(not marks for marks in _classified(pdf))
