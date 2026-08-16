"""Table geometry: rebuild a cell grid, render it row-major, find uncaptioned tables.

A PDF has no notion of a table — only text drawn at coordinates. Read in the
parser's own order (and, after the XY-cut, in *column* order) a table's text
comes out column by column: ``Model, Alpha, Beta, Gamma, Accuracy, 89.3 …``,
which loses exactly the row associations a reader (or a retrieval answer) needs.

This module reconstructs the grid from cell positions: baselines group cells
into rows, whitespace gaps group them into columns. The result is rendered
row-major for the chunk text and kept as ``cell_grid`` for the CSV artifact.
Detection of tables *without* a ``Table N:`` caption uses the same geometry:
several consecutive rows sharing column bands, made of short cells.

No models, no rules about borders — many tables have no ruling lines at all.
"""

from __future__ import annotations

import statistics

from docint.core.ocr.html_table import grid_to_text
from docint.core.readers.documents.models import BBox
from docint.core.readers.documents.parse import ParsedPage, TextLine

# ``grid_to_text`` is re-exported: it defines how a table reads as text, and
# that must be the same whether the grid came from geometry or from a model.
__all__ = [
    "build_grid",
    "caption_extent",
    "detect_geometric_tables",
    "grid_to_text",
    "group_rows",
    "needs_structure",
]

# Cells whose baselines differ by less than this share of the median cell
# height belong to one row.
_ROW_TOLERANCE = 0.5
# Horizontal whitespace of at least this many font sizes separates columns.
_COLUMN_GAP_FONT_SIZES = 0.5
# Geometric detection: a table needs at least this many rows, each with at
# least this many cells, and its cells must be short.
_MIN_TABLE_ROWS = 3
_MIN_ROW_CELLS = 2
_MAX_CELL_CHARS = 40
_MIN_SHORT_CELL_SHARE = 0.6
# A row wider than this share of the page is prose, not a table row.
_MAX_ROW_WIDTH_SHARE = 0.85
# Two-column prose is also "rows of two aligned cells", so shape alone cannot
# separate it from a table. What a table almost always has and prose never does
# is a column of labels or numbers: every cell in it short. Requiring one such
# column costs the rare all-long-text table (a captioned one is still caught by
# the caption rule) and buys immunity to reading prose as a grid.
_MAX_LABEL_COLUMN_CHARS = 15
# A table is rectangular: its rows hold the same number of cells. Text that
# merely happens to be aligned — a numbered bibliography whose entries wrap, a
# footnote block — is ragged, every line splitting into a different number of
# runs at different places. Measured on a real paper, without this check a
# reference list was read as 40 tables.
_MIN_FULL_ROW_SHARE = 0.7
# A vertical gap this many median line heights wide ends a table: the next
# block of text is something else.
_MAX_ROW_GAP_HEIGHTS = 2.5
# A caption's wrapped lines share its left edge and read as a sentence; table
# rows are indented differently and hold short cells.
_CAPTION_INDENT_TOLERANCE = 3.0
_CAPTION_LINE_MIN_CHARS = 60
# Share of empty cells above which a reconstructed grid is treated as having
# lost its structure (spanning headers flatten into blanks).
_MAX_EMPTY_CELL_SHARE = 0.25


def _median_height(cells: list[TextLine]) -> float:
    """Median cell height, or 10pt when there is nothing to measure."""
    heights = [c.bbox.y1 - c.bbox.y0 for c in cells if c.bbox.y1 > c.bbox.y0]
    return statistics.median(heights) if heights else 10.0


def _median_size(cells: list[TextLine]) -> float:
    """Median font size, falling back to the median height."""
    sizes = [c.font_size for c in cells if c.font_size > 0]
    return statistics.median(sizes) if sizes else _median_height(cells)


def group_rows(cells: list[TextLine]) -> list[list[TextLine]]:
    """Group cells into rows by baseline, top to bottom, each row left to right.

    Args:
        cells (list[TextLine]): Cells of one table region.

    Returns:
        list[list[TextLine]]: Rows, each sorted by x.
    """
    if not cells:
        return []
    tolerance = _ROW_TOLERANCE * _median_height(cells)
    rows: list[list[TextLine]] = []
    for cell in sorted(cells, key=lambda c: (-c.bbox.y0, c.bbox.x0)):
        if rows and abs(rows[-1][0].bbox.y0 - cell.bbox.y0) <= tolerance:
            rows[-1].append(cell)
        else:
            rows.append([cell])
    return [sorted(row, key=lambda c: c.bbox.x0) for row in rows]


def _column_bands(cells: list[TextLine]) -> list[tuple[float, float]]:
    """Merge the cells' x-intervals into column bands separated by whitespace.

    Args:
        cells (list[TextLine]): Cells of one table region.

    Returns:
        list[tuple[float, float]]: ``(x0, x1)`` per column, left to right.
    """
    if not cells:
        return []
    min_gap = _COLUMN_GAP_FONT_SIZES * _median_size(cells)
    bands: list[list[float]] = []
    for cell in sorted(cells, key=lambda c: c.bbox.x0):
        if bands and cell.bbox.x0 - bands[-1][1] < min_gap:
            bands[-1][1] = max(bands[-1][1], cell.bbox.x1)
        else:
            bands.append([cell.bbox.x0, cell.bbox.x1])
    return [(x0, x1) for x0, x1 in bands]


def _band_index(cell: TextLine, bands: list[tuple[float, float]]) -> int:
    """Index of the band whose span the cell's centre falls in (nearest wins)."""
    centre = (cell.bbox.x0 + cell.bbox.x1) / 2
    for idx, (x0, x1) in enumerate(bands):
        if x0 <= centre <= x1:
            return idx
    return min(range(len(bands)), key=lambda i: min(abs(centre - bands[i][0]), abs(centre - bands[i][1])))


def build_grid(cells: list[TextLine], bbox: BBox) -> list[list[str]]:
    """Rebuild the cell grid of the table occupying ``bbox``.

    Rows come from shared baselines, columns from whitespace-separated bands
    across the whole region (so an empty cell keeps its column). Two runs
    inside one band join with a space — a multi-word cell, not two columns.

    Args:
        cells (list[TextLine]): Candidate cells (the page's unmerged cells).
        bbox (BBox): The table region.

    Returns:
        list[list[str]]: Rows of cell texts; ``""`` where a cell is missing.
    """
    inside = [
        c
        for c in cells
        if c.text.strip()
        and c.bbox.x0 >= bbox.x0 - 1
        and c.bbox.x1 <= bbox.x1 + 1
        and c.bbox.y0 >= bbox.y0 - 1
        and c.bbox.y1 <= bbox.y1 + 1
    ]
    if not inside:
        return []
    bands = _column_bands(inside)
    grid: list[list[str]] = []
    for row in group_rows(inside):
        texts: list[list[str]] = [[] for _ in bands]
        for cell in row:
            texts[_band_index(cell, bands)].append(cell.text.strip())
        grid.append([" ".join(parts) for parts in texts])
    return grid


def needs_structure(grid: list[list[str]] | None) -> bool:
    """Whether a table's geometric grid is too weak to stand on its own.

    A missing grid, a single column, or a grid where a quarter of the cells are
    empty all point the same way: the structure was flattened, typically by a
    header spanning columns that cell positions cannot express. Those are the
    tables worth a remote vision call; a dense grid is left alone.

    Args:
        grid (list[list[str]] | None): The reconstructed grid, if any.

    Returns:
        bool: True when the table should be sent for structure recovery.
    """
    if not grid:
        return True
    if max((len(row) for row in grid), default=0) < _MIN_ROW_CELLS:
        return True
    cells = [cell for row in grid for cell in row]
    if not cells:
        return True
    empty = sum(1 for cell in cells if not cell.strip())
    return empty >= _MAX_EMPTY_CELL_SHARE * len(cells)


def caption_extent(page: ParsedPage, caption: BBox) -> BBox | None:
    """Where the table under a ``Table N:`` caption ends.

    A caption is already proof that what follows is a table, so this asks only
    about extent, not about whether the region qualifies as a grid. That
    matters for the tables geometry cannot validate — multi-level headers,
    spanning cells, cells split into several runs by mathematical notation —
    which are real tables all the same, and whose rows must still be rendered
    row by row rather than column by column.

    The caption's own wrapped lines are skipped (a lone cell, or a cell longer
    than a table cell would be, is prose), and the region ends at the first
    vertical gap wider than the table's line spacing.

    Args:
        page (ParsedPage): The page the caption sits on.
        caption (BBox): Bounding box of the caption line.

    Returns:
        BBox | None: The table's area, or ``None`` when nothing follows the caption.
    """
    cells = [c for c in page.cells if c.text.strip()]
    if not cells:
        return None
    max_gap = _MAX_ROW_GAP_HEIGHTS * _median_height(cells)
    rows = [row for row in group_rows(cells) if max(c.bbox.y1 for c in row) <= caption.y0 + 1]

    members: list[TextLine] = []
    previous_bottom: float | None = None
    for row in rows:
        looks_like_prose = (
            len(row) < _MIN_ROW_CELLS
            or any(len(c.text.strip()) > _MAX_CELL_CHARS for c in row)
            # A caption's wrapped line starts at the caption's own left edge and
            # carries a sentence's worth of text — even when inline maths chops
            # it into short runs, which is what makes the cell-length test miss it.
            or (
                abs(row[0].bbox.x0 - caption.x0) <= _CAPTION_INDENT_TOLERANCE
                and sum(len(c.text.strip()) for c in row) > _CAPTION_LINE_MIN_CHARS
            )
        )
        if not members:
            # Still inside the caption's own paragraph.
            if looks_like_prose:
                continue
        else:
            top = max(c.bbox.y1 for c in row)
            if previous_bottom is not None and previous_bottom - top > max_gap:
                break
        members.extend(row)
        previous_bottom = min(c.bbox.y0 for c in row)

    if not members:
        return None
    return BBox(
        x0=min(c.bbox.x0 for c in members),
        y0=min(c.bbox.y0 for c in members),
        x1=max(c.bbox.x1 for c in members),
        y1=max(c.bbox.y1 for c in members),
    )


def _has_label_column(cells: list[TextLine], bands: list[tuple[float, float]]) -> bool:
    """Whether some column consists entirely of short cells (labels or numbers).

    Args:
        cells (list[TextLine]): All cells of the candidate region.
        bands (list[tuple[float, float]]): Its column bands.

    Returns:
        bool: True when at least one column holds only short cells.
    """
    per_band: dict[int, list[int]] = {}
    for cell in cells:
        per_band.setdefault(_band_index(cell, bands), []).append(len(cell.text.strip()))
    return any(lengths and max(lengths) <= _MAX_LABEL_COLUMN_CHARS for lengths in per_band.values())


def detect_geometric_tables(page: ParsedPage) -> list[BBox]:
    """Find tables that carry no ``Table N:`` caption.

    A run of at least three consecutive rows, each holding at least two cells,
    sharing at least two column bands, made mostly of short cells and none of
    them spanning the page like prose, is a table.

    Args:
        page (ParsedPage): The parsed page (its unmerged ``cells`` are used).

    Returns:
        list[BBox]: One region per detected table, top to bottom.
    """
    cells = [c for c in page.cells if c.text.strip()]
    if len(cells) < _MIN_TABLE_ROWS * _MIN_ROW_CELLS:
        return []
    max_width = _MAX_ROW_WIDTH_SHARE * page.width
    max_gap = _MAX_ROW_GAP_HEIGHTS * _median_height(cells)
    rows = group_rows(cells)

    regions: list[BBox] = []
    run: list[list[TextLine]] = []

    def _flush() -> None:
        """Emit the accumulated row run as a table region when it qualifies."""
        # Rows of a single cell are wrapped continuations of the row above
        # (a column heading too long for its column); they belong to the table
        # but do not count as rows of their own.
        multi = [row for row in run if len(row) >= _MIN_ROW_CELLS]
        if len(multi) < _MIN_TABLE_ROWS:
            run.clear()
            return
        members = [c for row in run for c in row]
        bands = _column_bands(members)
        if len(bands) < _MIN_ROW_CELLS:
            run.clear()
            return
        short = sum(1 for c in members if len(c.text.strip()) <= _MAX_CELL_CHARS)
        if short < _MIN_SHORT_CELL_SHARE * len(members):
            run.clear()
            return
        if not _has_label_column(members, bands):
            run.clear()
            return
        counts = [len(row) for row in multi]
        modal = max(set(counts), key=counts.count)
        if counts.count(modal) < _MIN_FULL_ROW_SHARE * len(multi):
            run.clear()
            return
        regions.append(
            BBox(
                x0=min(c.bbox.x0 for c in members),
                y0=min(c.bbox.y0 for c in members),
                x1=max(c.bbox.x1 for c in members),
                y1=max(c.bbox.y1 for c in members),
            )
        )
        run.clear()

    previous_bottom: float | None = None
    for row in rows:
        width = max(c.bbox.x1 for c in row) - min(c.bbox.x0 for c in row)
        top = max(c.bbox.y1 for c in row)
        gap = (previous_bottom - top) if previous_bottom is not None else 0.0
        previous_bottom = min(c.bbox.y0 for c in row)
        if width > max_width or gap > max_gap:
            _flush()
            if width > max_width:
                continue
        if len(row) < _MIN_ROW_CELLS and not run:
            # A lone line with no table above it starts nothing.
            continue
        run.append(row)
    _flush()
    return regions
