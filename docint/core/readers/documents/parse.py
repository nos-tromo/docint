"""docling-parse backbone shared by every stage of the PDF pipeline.

One place opens the PDF and normalises what ``docling-parse`` returns —
line cells with rotation-aware rectangles and font names, and bitmap
resources with their placement rectangles — into the pipeline's own plain
dataclasses (``BBox`` in the bottom-left-origin point coordinates the rest
of the package already uses). Triage, layout analysis, digital text
extraction and image extraction all consume :class:`ParsedPdf`; none of
them touch ``docling_parse`` directly.

This deliberately drives ``docling_parse.pdf_parser.DoclingPdfParser``
itself, not ``docling.document_converter`` — the converter eagerly imports
the model-backed PDF pipeline (``docling_ibm_models`` → torch), which the
CPU-only image must not ship (see ``readers/docx.py`` for the same rule).
docling-parse is a cells-and-geometry parser: reading order and block
structure are computed here (:func:`order_lines`) with a deterministic
recursive XY-cut, not by a model.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from docling_core.types.doc.page import TextCellUnit
from docling_parse.pdf_parser import ContentConfig, ContentLevel, DoclingPdfParser
from loguru import logger

from docint.core.readers.documents.models import BBox


@dataclass(frozen=True)
class TextLine:
    """One line cell of a page.

    Attributes:
        text: The line's text.
        bbox: Axis-aligned bounding box in bottom-left-origin points.
        font_name: The PDF font name (e.g. ``/Helvetica-Bold``); empty when unknown.
        font_size: Approximate font size in points (the cell rectangle's height).
    """

    text: str
    bbox: BBox
    font_name: str = ""
    font_size: float = 0.0


@dataclass(frozen=True)
class ImagePlacement:
    """One embedded bitmap and where it is drawn on the page.

    Attributes:
        index: Zero-based index of the bitmap on its page (docling-parse order).
        bbox: Placement rectangle in bottom-left-origin points.
    """

    index: int
    bbox: BBox


@dataclass
class ParsedPage:
    """Normalised content of one page.

    Attributes:
        page_index: Zero-based page number.
        width: Page width in points.
        height: Page height in points.
        lines: Line cells in the parser's native (content-stream) order.
        images: Embedded bitmaps with placement boxes.
    """

    page_index: int
    width: float
    height: float
    lines: list[TextLine] = field(default_factory=list)
    images: list[ImagePlacement] = field(default_factory=list)


def _rect_to_bbox(rect: Any) -> BBox:
    """Convert a docling ``BoundingRectangle`` (4 corners) to an axis-aligned ``BBox``.

    Args:
        rect (Any): A ``docling_core.types.doc.page.BoundingRectangle``.

    Returns:
        BBox: The axis-aligned bounding box of the rectangle's corners.
    """
    xs = (float(rect.r_x0), float(rect.r_x1), float(rect.r_x2), float(rect.r_x3))
    ys = (float(rect.r_y0), float(rect.r_y1), float(rect.r_y2), float(rect.r_y3))
    return BBox(x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys))


class ParsedPdf:
    """A PDF opened with docling-parse, exposing normalised pages on demand.

    Pages are parsed lazily and cached, so opening a document is cheap and
    each page is parsed at most once no matter how many stages ask for it.
    Use as a context manager (or call :meth:`close`) to release the native
    document handle.
    """

    def __init__(self, file_path: str | Path) -> None:
        """Open ``file_path`` for parsing.

        Args:
            file_path (str | Path): Path to the PDF file.

        Raises:
            Exception: Whatever docling-parse raises for an unreadable file.
        """
        self._file_path = Path(file_path)
        self._parser = DoclingPdfParser(loglevel="fatal")
        self._doc = self._parser.load(
            path_or_stream=self._file_path,
            lazy=True,
            content_config=ContentConfig(
                char_cells_content_level=ContentLevel.SKIP,
                word_cells_content_level=ContentLevel.SKIP,
                line_cells_content_level=ContentLevel.COMPUTE_AND_MATERIALIZE,
                shapes_content_level=ContentLevel.SKIP,
                bitmaps_content_level=ContentLevel.COMPUTE_AND_MATERIALIZE,
                include_bitmap_bytes=False,
            ),
        )
        self._pages: dict[int, ParsedPage] = {}

    @property
    def file_path(self) -> Path:
        """Path of the parsed file."""
        return self._file_path

    @property
    def page_count(self) -> int:
        """Number of pages in the document."""
        return int(self._doc.number_of_pages())

    def page(self, page_index: int) -> ParsedPage:
        """Return the normalised content of page ``page_index`` (zero-based).

        Args:
            page_index (int): Zero-based page number.

        Returns:
            ParsedPage: The page's lines and images.
        """
        cached = self._pages.get(page_index)
        if cached is not None:
            return cached
        # docling-parse pages are 1-based.
        raw = self._doc.get_page(page_index + 1)
        lines: list[TextLine] = []
        for cell in raw.iterate_cells(unit_type=TextCellUnit.LINE):
            text = str(cell.text or "")
            if not text.strip():
                continue
            bbox = _rect_to_bbox(cell.rect)
            lines.append(
                TextLine(
                    text=text,
                    bbox=bbox,
                    font_name=str(getattr(cell, "font_name", "") or ""),
                    font_size=round(bbox.y1 - bbox.y0, 2),
                )
            )
        images = [
            ImagePlacement(index=int(getattr(bmp, "index", idx)), bbox=_rect_to_bbox(bmp.rect))
            for idx, bmp in enumerate(raw.bitmap_resources)
        ]
        parsed = ParsedPage(
            page_index=page_index,
            width=float(raw.dimension.width),
            height=float(raw.dimension.height),
            lines=lines,
            images=images,
        )
        self._pages[page_index] = parsed
        return parsed

    def close(self) -> None:
        """Release the native document handle."""
        try:
            self._doc.unload()
        except Exception as exc:  # pragma: no cover - best effort
            logger.debug("docling-parse unload failed for {}: {}", self._file_path, exc)
        self._pages.clear()

    def __enter__(self) -> ParsedPdf:
        """Enter the context manager."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the document on exit."""
        self.close()


# ---------------------------------------------------------------------------
# Reading order
# ---------------------------------------------------------------------------


def _median_height(lines: list[TextLine]) -> float:
    """Median line height, or 10pt when there is nothing to measure."""
    heights = [ln.bbox.y1 - ln.bbox.y0 for ln in lines if ln.bbox.y1 > ln.bbox.y0]
    return statistics.median(heights) if heights else 10.0


def _largest_gap(intervals: list[tuple[float, float]], min_gap: float) -> float | None:
    """Return the midpoint of the widest empty gap between merged ``intervals``.

    Args:
        intervals (list[tuple[float, float]]): ``(start, end)`` extents along one axis.
        min_gap (float): Minimum gap width to count as a separation.

    Returns:
        float | None: The gap's midpoint, or ``None`` when no gap is wide enough.
    """
    if len(intervals) < 2:
        return None
    ordered = sorted(intervals)
    best_width = 0.0
    best_mid: float | None = None
    _, reach = ordered[0]
    for start, end in ordered[1:]:
        gap = start - reach
        if gap >= min_gap and gap > best_width:
            best_width = gap
            best_mid = reach + gap / 2
        reach = max(reach, end)
    return best_mid


def _xy_cut(lines: list[TextLine], min_gap: float) -> list[TextLine]:
    """Recursively split ``lines`` at whitespace gaps, columns first, then rows.

    Args:
        lines (list[TextLine]): Lines to order.
        min_gap (float): Minimum whitespace width that separates blocks.

    Returns:
        list[TextLine]: Lines in reading order.
    """
    if len(lines) <= 1:
        return list(lines)
    # Vertical cut: a whitespace column across the whole group → left, then right.
    x_mid = _largest_gap([(ln.bbox.x0, ln.bbox.x1) for ln in lines], min_gap)
    if x_mid is not None:
        left = [ln for ln in lines if ln.bbox.x1 <= x_mid]
        right = [ln for ln in lines if ln.bbox.x1 > x_mid]
        if left and right:
            return _xy_cut(left, min_gap) + _xy_cut(right, min_gap)
    # Horizontal cut: a whitespace row → top, then bottom.
    y_mid = _largest_gap([(ln.bbox.y0, ln.bbox.y1) for ln in lines], min_gap)
    if y_mid is not None:
        top = [ln for ln in lines if ln.bbox.y0 >= y_mid]
        bottom = [ln for ln in lines if ln.bbox.y0 < y_mid]
        if top and bottom:
            return _xy_cut(top, min_gap) + _xy_cut(bottom, min_gap)
    # Leaf: top-to-bottom, left-to-right.
    return sorted(lines, key=lambda ln: (-round(ln.bbox.y1, 1), ln.bbox.x0))


def order_lines(lines: list[TextLine]) -> list[TextLine]:
    """Return ``lines`` in human reading order.

    Uses a recursive XY-cut over the line boxes: a whitespace column wide
    enough to separate text columns splits the group left-then-right; else a
    whitespace row splits it top-then-bottom; a group that cannot be split is
    read top-to-bottom, left-to-right. A full-width heading above two columns
    therefore comes first, followed by the whole left column, then the right.

    Args:
        lines (list[TextLine]): Lines in any order.

    Returns:
        list[TextLine]: The same lines in reading order.
    """
    if not lines:
        return []
    min_gap = max(0.5 * _median_height(lines), 1.0)
    return _xy_cut(list(lines), min_gap)


def lines_to_text(lines: list[TextLine]) -> str:
    """Join already-ordered lines into page text.

    Consecutive lines are separated by a newline; a vertical gap larger than
    1.5 × the median line height inserts a blank line (paragraph break).

    Args:
        lines (list[TextLine]): Lines in reading order.

    Returns:
        str: The assembled text.
    """
    if not lines:
        return ""
    para_gap = 1.5 * _median_height(lines)
    parts: list[str] = []
    prev: TextLine | None = None
    for ln in lines:
        if prev is not None:
            gap = prev.bbox.y0 - ln.bbox.y1
            parts.append("\n\n" if gap > para_gap else "\n")
        parts.append(ln.text)
        prev = ln
    return "".join(parts)
