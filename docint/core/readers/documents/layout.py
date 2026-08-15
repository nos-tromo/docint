"""Layout analysis interface and the docling-parse backed implementation."""

from __future__ import annotations

import re
import statistics
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

from loguru import logger
from typing_extensions import override

from docint.core.readers.documents.models import BBox, BlockType, LayoutBlock, PageInfo
from docint.core.readers.documents.parse import (
    ParsedPage,
    ParsedPdf,
    TextLine,
    lines_to_text,
    order_lines,
)


class LayoutAnalyzer(ABC):
    """Abstract interface for layout analysis backends."""

    @abstractmethod
    def analyze_page(self, page_index: int, *, file_path: Path | None = None) -> list[LayoutBlock]:
        """Detect layout blocks on a single page.

        Args:
            page_index (int): Zero-based page number.
            file_path (Path | None): Path to the source PDF (used by some backends).

        Returns:
            list[LayoutBlock]: List of ``LayoutBlock`` items sorted by ``reading_order``.
        """


# Heading detection thresholds. Deliberately conservative: a false heading
# splits a coarse unit and resets/nests the section path, which hurts more
# than a missed heading.
_HEADING_MAX_CHARS = 120
_HEADING_SIZE_RATIO = 1.15
_HEADING_MAX_SHARE = 0.5
_HEADING_MAX_RUN = 2  # a run of more than this many same-style lines is a paragraph
_BOLD_MARKERS = (
    "bold",
    "black",
    "heavy",
    "semibold",
    "demibold",
    "medi",
)  # "medi": NimbusRomNo9L-Medi, LaTeX bold Times
_WORD_RE = re.compile(r"[^\W\d_]{3,}")  # at least one real word (3+ letters)


class DoclingParseLayoutAnalyzer(LayoutAnalyzer):
    """Layout analyser backed by docling-parse line cells and bitmap placements.

    Emits five block types, in reading order:

    * **FIGURE** — one per embedded bitmap, with its placement rectangle.
    * **TITLE** / **HEADER** — short lines set noticeably larger than the
      page's body text, or in a bold face at body size. The largest heading
      size on the page is ``TITLE`` (resets the section path downstream), the
      rest ``HEADER`` (nests).
    * **TABLE** — a *"Table N:"* caption plus the short/tabular lines that
      follow it (caption heuristic), with the union of those lines' boxes.
    * **TEXT** — everything else, one block per run of lines between headings
      / tables / column jumps, so each block has a tight bbox and its own text.

    Reading order comes from :func:`~docint.core.readers.documents.parse.order_lines`
    (recursive XY-cut), so multi-column pages read column by column.
    """

    def __init__(self, file_path: str | Path, *, parsed: ParsedPdf | None = None) -> None:
        """Open the PDF (or adopt an already-open handle) for layout analysis.

        Args:
            file_path (str | Path): Path to the PDF.
            parsed (ParsedPdf | None): Open document handle to reuse; when
                ``None`` the file is opened here and released by :meth:`close`.
        """
        self._file_path = Path(file_path)
        self._owned = parsed is None
        self._doc = parsed if parsed is not None else ParsedPdf(self._file_path)

    def close(self) -> None:
        """Release the document handle if this analyser opened it."""
        if self._owned:
            self._doc.close()

    @override
    def analyze_page(self, page_index: int, *, file_path: Path | None = None) -> list[LayoutBlock]:
        """Return layout blocks for a single page.

        Args:
            page_index (int): Zero-based page number.
            file_path (Path | None): Ignored (present for interface compatibility).

        Returns:
            list[LayoutBlock]: List of ``LayoutBlock`` items sorted by ``reading_order``.
        """
        blocks: list[LayoutBlock] = []
        try:
            page = self._doc.page(page_index)
            page_bbox = BBox(x0=0.0, y0=0.0, x1=page.width, y1=page.height)

            # --- Embedded bitmaps → FIGURE blocks (first, as before) ---
            for img in page.images:
                blocks.append(
                    LayoutBlock(
                        block_id=f"figure-{page_index}-{uuid.uuid4().hex[:8]}",
                        page_index=page_index,
                        type=BlockType.FIGURE,
                        bbox=img.bbox,
                        reading_order=0,
                        confidence=0.9,
                        text="",
                    )
                )

            # --- Text: headings, tables, prose ---
            blocks.extend(_build_text_blocks(page, page_index))

            # Fallback: if nothing was detected at all, emit an empty TEXT block
            if not blocks:
                blocks.append(
                    LayoutBlock(
                        block_id=f"block-{page_index}-{uuid.uuid4().hex[:8]}",
                        page_index=page_index,
                        type=BlockType.TEXT,
                        bbox=page_bbox,
                        reading_order=0,
                        confidence=0.0,
                        text="",
                    )
                )

            for order, block in enumerate(blocks):
                block.reading_order = order
        except Exception as exc:
            logger.warning("Layout analysis failed for page {}: {}", page_index, exc)
        return blocks


# ---------------------------------------------------------------------------
# Block construction
# ---------------------------------------------------------------------------


def _union_bbox(lines: list[TextLine]) -> BBox:
    """Bounding box enclosing every line in ``lines`` (non-empty)."""
    return BBox(
        x0=min(ln.bbox.x0 for ln in lines),
        y0=min(ln.bbox.y0 for ln in lines),
        x1=max(ln.bbox.x1 for ln in lines),
        y1=max(ln.bbox.y1 for ln in lines),
    )


def _with_paragraph_breaks(lines: list[TextLine]) -> list[TextLine | None]:
    """Interleave ``None`` markers where a paragraph gap separates two lines.

    Mirrors the blank-line convention of ``lines_to_text`` so the text-based
    table heuristics (which look for blank lines) can run over ordered lines.
    """
    if not lines:
        return []
    heights = [ln.bbox.y1 - ln.bbox.y0 for ln in lines if ln.bbox.y1 > ln.bbox.y0]
    para_gap = 1.5 * (statistics.median(heights) if heights else 10.0)
    out: list[TextLine | None] = []
    prev: TextLine | None = None
    for ln in lines:
        if prev is not None and prev.bbox.y0 - ln.bbox.y1 > para_gap:
            out.append(None)
        out.append(ln)
        prev = ln
    return out


def _is_bold(font_name: str) -> bool:
    """Whether a PDF font name advertises a bold weight."""
    name = font_name.lower()
    return any(marker in name for marker in _BOLD_MARKERS)


def _style_run_lengths(lines: list[TextLine]) -> list[int]:
    """For each line, the length of the same-style run it belongs to.

    Consecutive lines with the same font name, the same size (within 0.5 pt)
    and normal leading (vertical gap below 0.8 x the font size) form a run —
    the shape of a wrapped paragraph.

    Args:
        lines (list[TextLine]): Lines in reading order.

    Returns:
        list[int]: Run length per line index.
    """
    lengths = [1] * len(lines)
    start = 0
    for idx in range(1, len(lines) + 1):
        if idx < len(lines):
            prev, cur = lines[idx - 1], lines[idx]
            same_style = prev.font_name == cur.font_name and abs(prev.font_size - cur.font_size) < 0.5
            gap = prev.bbox.y0 - cur.bbox.y1
            close = -0.5 * max(cur.font_size, 1.0) <= gap < 0.8 * max(cur.font_size, 1.0)
            if same_style and close:
                continue
        for j in range(start, idx):
            lengths[j] = idx - start
        start = idx
    return lengths


def _classify_headings(lines: list[TextLine], excluded: set[int]) -> dict[int, BlockType]:
    """Decide which of ``lines`` are headings.

    A line is a heading candidate when it is short (≤ 120 chars, not ending in
    sentence punctuation), contains a real word, is not rotated, and either set
    ≥ 1.15 x the body font size or set in a bold face at body size or larger.
    A candidate that sits in a run of more than two consecutive same-style
    lines at normal leading is a paragraph, not a heading, however large the
    face. Candidates with the page's largest heading size (when that size is
    itself larger than body) are ``TITLE``; the rest ``HEADER``. When more
    than half of the page's lines qualify, nothing is promoted — the page
    simply has a uniform (or uniformly bold) face.

    Args:
        lines (list[TextLine]): Ordered lines of the page.
        excluded (set[int]): Indices of lines that belong to tables.

    Returns:
        dict[int, BlockType]: Mapping of line index → ``TITLE`` / ``HEADER``.
    """
    sizes = [ln.font_size for ln in lines if ln.font_size > 0]
    if len(sizes) < 2:
        return {}
    body = statistics.median(sizes)
    run_lengths = _style_run_lengths(lines)
    candidates: dict[int, float] = {}
    for idx, ln in enumerate(lines):
        if idx in excluded or ln.rotated or run_lengths[idx] > _HEADING_MAX_RUN:
            continue
        text = ln.text.strip()
        if not text or len(text) > _HEADING_MAX_CHARS or text.endswith((".", ",", ";", ":")):
            continue
        if not _WORD_RE.search(text):
            continue
        large = ln.font_size >= _HEADING_SIZE_RATIO * body
        bold = _is_bold(ln.font_name) and ln.font_size >= body - 0.5
        if large or bold:
            candidates[idx] = ln.font_size
    if not candidates or len(candidates) > _HEADING_MAX_SHARE * len(lines):
        return {}
    top = max(candidates.values())
    title_size = top if top >= _HEADING_SIZE_RATIO * body else None
    return {
        idx: BlockType.TITLE if title_size is not None and abs(size - title_size) < 0.5 else BlockType.HEADER
        for idx, size in candidates.items()
    }


def _build_text_blocks(page: ParsedPage, page_index: int) -> list[LayoutBlock]:
    """Turn a page's lines into TITLE / HEADER / TABLE / TEXT blocks in reading order.

    Args:
        page (ParsedPage): The parsed page.
        page_index (int): Zero-based page number (for block ids).

    Returns:
        list[LayoutBlock]: Blocks in reading order (``reading_order`` unset).
    """
    ordered = order_lines(page.lines)
    if not ordered:
        return []

    marked = _with_paragraph_breaks(ordered)
    # Table regions are found on the text view (blank-line markers included),
    # then mapped back to line indices.
    text_view = [ln.text if ln is not None else "" for ln in marked]
    positions = [i for i, ln in enumerate(marked) if ln is not None]  # marked idx per ordered idx
    marked_to_ordered = {m: o for o, m in enumerate(positions)}
    table_lines: dict[int, int] = {}  # ordered idx → table number
    for table_no, (start, end) in enumerate(_detect_table_regions(text_view)):
        for m in range(start, end + 1):
            o = marked_to_ordered.get(m)
            if o is not None:
                table_lines[o] = table_no

    headings = _classify_headings(ordered, set(table_lines))

    blocks: list[LayoutBlock] = []
    run: list[TextLine] = []

    def _flush_text() -> None:
        """Emit the accumulated prose lines as one TEXT block."""
        if run:
            blocks.append(
                LayoutBlock(
                    block_id=f"block-{page_index}-{uuid.uuid4().hex[:8]}",
                    page_index=page_index,
                    type=BlockType.TEXT,
                    bbox=_union_bbox(run),
                    reading_order=0,
                    confidence=1.0,
                    text=lines_to_text(run),
                )
            )
            run.clear()

    idx = 0
    while idx < len(ordered):
        ln = ordered[idx]
        if idx in table_lines:
            _flush_text()
            table_no = table_lines[idx]
            members = []
            while idx < len(ordered) and table_lines.get(idx) == table_no:
                members.append(ordered[idx])
                idx += 1
            blocks.append(
                LayoutBlock(
                    block_id=f"table-{page_index}-{uuid.uuid4().hex[:8]}",
                    page_index=page_index,
                    type=BlockType.TABLE,
                    bbox=_union_bbox(members),
                    reading_order=0,
                    confidence=0.7,
                    text="\n".join(m.text for m in members),
                )
            )
            continue
        if idx in headings:
            _flush_text()
            blocks.append(
                LayoutBlock(
                    block_id=f"heading-{page_index}-{uuid.uuid4().hex[:8]}",
                    page_index=page_index,
                    type=headings[idx],
                    bbox=ln.bbox,
                    reading_order=0,
                    confidence=0.8,
                    text=ln.text.strip(),
                )
            )
            idx += 1
            continue
        # A jump back up the page means a new column: start a new TEXT block.
        if run and ln.bbox.y0 > run[-1].bbox.y1:
            _flush_text()
        run.append(ln)
        idx += 1
    _flush_text()
    return blocks


# ---------------------------------------------------------------------------
# Table heuristics (text-based)
# ---------------------------------------------------------------------------

_CAPTION_RE = re.compile(r"^Table\s+\d+\s*[:.]", re.IGNORECASE)


def _detect_table_regions(lines: list[str]) -> list[tuple[int, int]]:
    """Find caption-anchored table regions in a page's text lines.

    Looks for *"Table N:"* / *"Table N."* captions and extends each region
    over the tabular lines that follow (see :func:`_find_table_end`).

    Args:
        lines (list[str]): Page text, one entry per line (``""`` for a blank line).

    Returns:
        list[tuple[int, int]]: ``(start, end)`` inclusive line-index ranges.
    """
    regions: list[tuple[int, int]] = []
    i = 0
    while i < len(lines):
        if _CAPTION_RE.match(lines[i].strip()):
            end = _find_table_end(lines, i)
            regions.append((i, end))
            i = end + 1
        else:
            i += 1
    return regions


def _find_table_end(lines: list[str], start: int) -> int:
    """Heuristically find the last line of a table region.

    Scans forward from *start* looking for contiguous lines that are
    either short/tabular (lots of whitespace separation, numeric data,
    or column-like alignment) or continuation of a table caption.  Stops
    when a blank line followed by non-tabular prose, or a new
    heading-like line is encountered.

    Args:
        lines (list[str]): All lines of the page text.
        start (int): Index of the table caption line.

    Returns:
        int: Index of the last line belonging to the table.
    """
    end = start
    saw_blank = False

    for i in range(start + 1, len(lines)):
        line = lines[i].strip()

        if not line:
            saw_blank = True
            continue

        # Stop if we hit a new section/heading or another table/figure caption
        if re.match(r"^\d+(\.\d+)*\s+[A-Z]", line):
            break
        if re.match(r"^(Table|Figure)\s+\d+\s*[:.]", line, re.IGNORECASE):
            break

        # After a blank line, require strong tabular evidence to continue
        multi_space_gaps = len(re.findall(r"\s{2,}", line))
        is_tabular = multi_space_gaps >= 2 or (multi_space_gaps >= 1 and len(line) < 60)

        if saw_blank and not is_tabular:
            # Non-tabular line after a blank — table has ended
            break

        if is_tabular or (not saw_blank and len(line) < 60):
            end = i
            saw_blank = False
        else:
            break

    return end


def analyze_document(
    file_path: str | Path,
    pages: list[PageInfo],
    *,
    parsed: ParsedPdf | None = None,
) -> dict[int, list[LayoutBlock]]:
    """Run layout analysis on every page of *file_path*.

    Args:
        file_path (str | Path): Path to the PDF.
        pages (list[PageInfo]): Page triage results (used for page indices).
        parsed (ParsedPdf | None): Open document handle to reuse.

    Returns:
        dict[int, list[LayoutBlock]]: Mapping of ``page_index`` → list of ``LayoutBlock``.
    """
    file_path = Path(file_path)
    analyzer = DoclingParseLayoutAnalyzer(file_path, parsed=parsed)
    layout: dict[int, list[LayoutBlock]] = {}
    try:
        for page_info in pages:
            try:
                blocks = analyzer.analyze_page(page_info.page_index, file_path=file_path)
                layout[page_info.page_index] = blocks
            except Exception as exc:
                logger.warning("Layout analysis skipped for page {}: {}", page_info.page_index, exc)
                layout[page_info.page_index] = []
    finally:
        analyzer.close()
    return layout
