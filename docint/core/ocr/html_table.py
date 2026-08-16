"""HTML tables from a model answer, expanded into a flat grid.

Both OCR families answer with HTML for a table — the dots family inside its
layout JSON, a general vision model directly — so the expansion of
``rowspan``/``colspan`` into a rectangular grid lives here, once.
"""

from __future__ import annotations

import re
from html import unescape
from html.parser import HTMLParser

from loguru import logger
from typing_extensions import override


class _TableHtmlParser(HTMLParser):
    """Collect ``<tr>``/``<td>`` structure, with spans, from the first table."""

    def __init__(self) -> None:
        """Start with no rows collected."""
        super().__init__(convert_charrefs=True)
        self.rows: list[list[tuple[str, int, int]]] = []  # (text, rowspan, colspan)
        self._depth = 0
        self._done = False
        self._cell: list[str] | None = None
        self._spans: tuple[int, int] = (1, 1)

    @staticmethod
    def _span(attrs: list[tuple[str, str | None]], name: str) -> int:
        """Read a span attribute, defaulting to 1 for anything unusable."""
        for key, value in attrs:
            if key == name and value:
                try:
                    return max(1, min(int(value.strip()), 64))
                except ValueError:
                    return 1
        return 1

    @override
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Open a table, row or cell."""
        if self._done:
            return
        if tag == "table":
            self._depth += 1
        elif tag == "tr" and self._depth:
            self.rows.append([])
        elif tag in ("td", "th") and self._depth:
            if not self.rows:
                self.rows.append([])
            self._cell = []
            self._spans = (self._span(attrs, "rowspan"), self._span(attrs, "colspan"))

    @override
    def handle_endtag(self, tag: str) -> None:
        """Close a cell or the table."""
        if self._done:
            return
        if tag in ("td", "th") and self._cell is not None:
            text = re.sub(r"\s+", " ", "".join(self._cell)).strip()
            self.rows[-1].append((text, self._spans[0], self._spans[1]))
            self._cell = None
        elif tag == "table":
            self._depth = max(0, self._depth - 1)
            if self._depth == 0:
                # Only the first table is the answer; anything after it is noise.
                self._done = True

    @override
    def handle_data(self, data: str) -> None:
        """Accumulate text inside the current cell."""
        if self._cell is not None:
            self._cell.append(data)


def parse_html_table(html: str) -> list[list[str]] | None:
    """Turn a model's HTML answer into a rectangular grid.

    ``rowspan``/``colspan`` are expanded by repeating the spanning cell's text
    into every position it covers — that is what turns a two-level header into
    columns that each name their group. Rows are padded to equal width.

    Args:
        html (str): The model's raw answer (code fences and prose tolerated).

    Returns:
        list[list[str]] | None: The grid, or ``None`` when no usable table was found.
    """
    if not html or "<table" not in html.lower():
        return None
    parser = _TableHtmlParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception as exc:  # pragma: no cover - HTMLParser is lenient by design
        logger.debug("Table HTML could not be parsed: {}", exc)
        return None

    grid: list[list[str]] = []
    # Cells still owed to later rows by a rowspan: (row index, column) -> text.
    pending: dict[tuple[int, int], str] = {}
    for row_index, row in enumerate(parser.rows):
        cells: list[str] = []
        column = 0
        for text, rowspan, colspan in row:
            while (row_index, column) in pending:
                cells.append(pending.pop((row_index, column)))
                column += 1
            value = unescape(text)
            for offset in range(colspan):
                cells.append(value)
                for extra in range(1, rowspan):
                    pending[(row_index + extra, column + offset)] = value
            column += colspan
        while (row_index, column) in pending:
            cells.append(pending.pop((row_index, column)))
            column += 1
        grid.append(cells)

    grid = [row for row in grid if row]
    if not grid:
        return None
    width = max(len(row) for row in grid)
    return [row + [""] * (width - len(row)) for row in grid]


def grid_to_text(grid: list[list[str]]) -> str:
    """Render a grid row-major, cells separated by ``" | "``.

    The one definition of how a table reads as text, so a table recovered from
    geometry and one read by a model look the same in a chunk.

    Args:
        grid (list[list[str]]): Rows of cell texts.

    Returns:
        str: One line per row.
    """
    return "\n".join(" | ".join(row).strip() for row in grid)
