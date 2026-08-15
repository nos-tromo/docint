"""Tests for table cell-grid reconstruction and caption-less table detection."""

from __future__ import annotations

from pathlib import Path

from pdf_fixtures import (
    PageSpec,
    TextRun,
    build_pdf,
    table_page,
    two_column_page,
    wrapped_header_table_page,
)

from docint.core.readers.documents.models import BBox
from docint.core.readers.documents.parse import ParsedPdf, TextLine
from docint.core.readers.documents.tables import (
    build_grid,
    detect_geometric_tables,
    grid_to_text,
)


def _cell(text: str, x0: float, y0: float, width: float = 40.0, height: float = 10.0) -> TextLine:
    """Build a ``TextLine`` for grid tests."""
    return TextLine(text=text, bbox=BBox(x0=x0, y0=y0, x1=x0 + width, y1=y0 + height), font_size=height)


class TestBuildGrid:
    """Cells inside a region become rows and columns."""

    def test_three_by_three_grid(self) -> None:
        """Cells aligned on baselines and column bands rebuild the original grid."""
        rows = [("Model", "Accuracy", "F1"), ("Alpha", "89.3", "88.1"), ("Beta", "91.0", "90.5")]
        cells = [
            _cell(text, x0=60 + col * 120, y0=680 - row * 14)
            for row, values in enumerate(rows)
            for col, text in enumerate(values)
        ]
        grid = build_grid(cells, BBox(x0=50, y0=640, x1=400, y1=700))
        assert grid == [list(r) for r in rows]

    def test_missing_cell_becomes_empty_string(self) -> None:
        """A gap in the table stays a gap — the row keeps its column count."""
        cells = [
            _cell("A", 60, 680),
            _cell("B", 180, 680),
            _cell("C", 300, 680),
            _cell("1", 60, 666),
            _cell("3", 300, 666),
        ]
        grid = build_grid(cells, BBox(x0=50, y0=660, x1=400, y1=700))
        assert grid == [["A", "B", "C"], ["1", "", "3"]]

    def test_cells_outside_the_region_are_ignored(self) -> None:
        """Only cells inside the table's bbox take part."""
        cells = [_cell("A", 60, 680), _cell("B", 180, 680), _cell("prose", 60, 500)]
        grid = build_grid(cells, BBox(x0=50, y0=660, x1=400, y1=700))
        assert grid == [["A", "B"]]

    def test_multi_word_cell_stays_one_cell(self) -> None:
        """Two runs inside one column band join with a space, not a new column."""
        cells = [_cell("Total", 60, 680, width=30), _cell("revenue", 92, 680, width=40), _cell("42", 300, 680)]
        grid = build_grid(cells, BBox(x0=50, y0=670, x1=400, y1=700))
        assert grid == [["Total revenue", "42"]]

    def test_empty_region(self) -> None:
        """No cells, no grid."""
        assert build_grid([], BBox(x0=0, y0=0, x1=10, y1=10)) == []


class TestGridToText:
    """Rendering a grid back to text keeps rows together."""

    def test_rows_are_pipe_separated(self) -> None:
        """A row reads left to right, cells separated by a pipe."""
        grid = [["Model", "Accuracy"], ["Alpha", "89.3"]]
        assert grid_to_text(grid) == "Model | Accuracy\nAlpha | 89.3"

    def test_empty_grid_is_empty_text(self) -> None:
        """No rows, no text."""
        assert grid_to_text([]) == ""


class TestGeometricDetection:
    """Tables without a caption are found by their geometry."""

    def test_caption_less_table_is_detected(self, tmp_path: Path) -> None:
        """Three-plus aligned rows of short cells form a table region."""
        pdf = tmp_path / "grid.pdf"
        pdf.write_bytes(build_pdf([table_page(caption=None)]))
        with ParsedPdf(pdf) as parsed:
            page = parsed.page(0)
            regions = detect_geometric_tables(page)
        assert len(regions) == 1
        grid = build_grid(page.cells, regions[0])
        assert grid[0] == ["Model", "Accuracy", "F1"]
        assert len(grid) == 4

    def test_wrapped_header_cell_does_not_split_the_table(self, tmp_path: Path) -> None:
        """A column heading spilling onto a second line stays part of the same table."""
        pdf = tmp_path / "wrapped.pdf"
        pdf.write_bytes(build_pdf([wrapped_header_table_page(caption=None)]))
        with ParsedPdf(pdf) as parsed:
            page = parsed.page(0)
            regions = detect_geometric_tables(page)
        assert len(regions) == 1
        grid = build_grid(page.cells, regions[0])
        assert grid[0][0] == "Layer Type"
        assert [row[0] for row in grid][-3:] == ["Self-Attention", "Recurrent", "Convolutional"]

    def test_two_column_prose_is_not_a_table(self, tmp_path: Path) -> None:
        """Long lines in two columns must never read as a table."""
        pdf = tmp_path / "cols.pdf"
        pdf.write_bytes(build_pdf([two_column_page()]))
        with ParsedPdf(pdf) as parsed:
            assert detect_geometric_tables(parsed.page(0)) == []

    def test_aligned_long_text_pairs_are_not_a_table(self, tmp_path: Path) -> None:
        """Rows of two long cells (the shape of two-column prose) need a short column."""
        runs = []
        for i in range(4):
            runs.append(TextRun(f"A fairly long left-hand phrase number {i}", x=60, y=680 - i * 14))
            runs.append(TextRun(f"A fairly long right-hand phrase number {i}", x=330, y=680 - i * 14))
        pdf = tmp_path / "pairs.pdf"
        pdf.write_bytes(build_pdf([PageSpec(runs=runs)]))
        with ParsedPdf(pdf) as parsed:
            assert detect_geometric_tables(parsed.page(0)) == []

    def test_numbered_reference_list_is_not_a_table(self, tmp_path: Path) -> None:
        """A bibliography has a short label column but ragged rows — never a table."""
        # Faithful to a real reference list: a bracketed label starts each
        # entry, lines split into several runs (italic titles, journal names,
        # page ranges), and those runs land in different places on every line.
        # So no column structure repeats from row to row.
        runs = []
        y = 700.0
        entries = [
            [("[6]", 100), ("Some Author. A paper title.", 130), ("In Proceedings", 330)],
            [(", pages 1-9, 2031.", 130), ("Publisher", 260), ("Berlin", 420)],
            [("[7]", 100), ("Another Author. A different title.", 130)],
            [("Journal of Things", 150), (", 12(3), 2030.", 300)],
            [("[8]", 100), ("Fourth Author. Yet another title.", 130), ("Transactions", 390)],
            [("on Systems", 200), (", pages 44-55.", 340), ("Society", 470)],
        ]
        for line in entries:
            for text, x in line:
                runs.append(TextRun(text, x=x, y=y))
            y -= 12
        pdf = tmp_path / "refs.pdf"
        pdf.write_bytes(build_pdf([PageSpec(runs=runs)]))
        with ParsedPdf(pdf) as parsed:
            assert detect_geometric_tables(parsed.page(0)) == []

    def test_ordinary_paragraph_is_not_a_table(self, tmp_path: Path) -> None:
        """Wrapped prose has one cell per line and never qualifies."""
        runs = [
            TextRun(f"Prose line {i} running the full width of the text block.", x=60, y=700 - 14 * i) for i in range(6)
        ]
        pdf = tmp_path / "prose.pdf"
        pdf.write_bytes(build_pdf([PageSpec(runs=runs)]))
        with ParsedPdf(pdf) as parsed:
            assert detect_geometric_tables(parsed.page(0)) == []
