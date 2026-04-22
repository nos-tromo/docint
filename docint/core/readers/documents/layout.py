"""Layout analysis interface and implementation."""

from __future__ import annotations

import re
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pypdf
from loguru import logger

from docint.core.readers.documents.models import BBox, BlockType, LayoutBlock, PageInfo


class LayoutAnalyzer(ABC):
    """Abstract interface for layout analysis backends."""

    @abstractmethod
    def analyze_page(
        self, page_index: int, *, file_path: Path | None = None
    ) -> list[LayoutBlock]:
        """Detect layout blocks on a single page.

        Args:
            page_index (int): Zero-based page number.
            file_path (Path | None): Path to the source PDF (used by some backends).

        Returns:
            list[LayoutBlock]: List of ``LayoutBlock`` items sorted by ``reading_order``.
        """


class PypdfLayoutAnalyzer(LayoutAnalyzer):
    """Layout analyser backed by ``pypdf`` text extraction and heuristics.

    Detects three block types:

    * **FIGURE** – pages containing embedded images (via ``/XObject``
      inspection).  Bounding boxes are derived from the content-stream
      transformation matrices when available, otherwise from the image
      native dimensions.
    * **TABLE** – contiguous text regions that match table-like
      patterns (captions such as *Table 1:* followed by column-aligned
      data).
    * **TEXT** – everything else.

    When the project integrates a more capable backend (e.g. Docling
    layout) this class can be replaced transparently.
    """

    def __init__(self, file_path: str | Path) -> None:
        self._file_path = Path(file_path)
        self._reader = pypdf.PdfReader(self._file_path)

    def analyze_page(
        self, page_index: int, *, file_path: Path | None = None
    ) -> list[LayoutBlock]:
        """Return layout blocks for a single page.

        The method inspects embedded images and text content to produce
        ``FIGURE``, ``TABLE``, and ``TEXT`` blocks sorted by reading
        order.

        Args:
            page_index (int): Zero-based page number.
            file_path (Path | None): Ignored (present for interface compatibility).

        Returns:
            list[LayoutBlock]: List of ``LayoutBlock`` items sorted by ``reading_order``.
        """
        blocks: list[LayoutBlock] = []
        try:
            page = self._reader.pages[page_index]
            text = page.extract_text() or ""
            mb = page.mediabox
            page_bbox = BBox(
                x0=float(mb.left),
                y0=float(mb.bottom),
                x1=float(mb.right),
                y1=float(mb.top),
            )
            page_width = float(mb.right) - float(mb.left)
            page_height = float(mb.top) - float(mb.bottom)

            order = 0

            # --- Detect embedded images → FIGURE blocks ---
            image_blocks = self._detect_images(page, page_index, page_bbox)
            for ib in image_blocks:
                ib.reading_order = order
                order += 1
                blocks.append(ib)

            # --- Detect tables → TABLE blocks + remaining TEXT ---
            table_blocks, remaining_text = self._detect_tables(
                text, page_index, page_bbox, page_width, page_height
            )
            for tb in table_blocks:
                tb.reading_order = order
                order += 1
                blocks.append(tb)

            # --- TEXT block for non-table text ---
            if remaining_text.strip():
                blocks.append(
                    LayoutBlock(
                        block_id=f"block-{page_index}-{uuid.uuid4().hex[:8]}",
                        page_index=page_index,
                        type=BlockType.TEXT,
                        bbox=page_bbox,
                        reading_order=order,
                        confidence=1.0,
                        text=remaining_text,
                    )
                )

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

        except Exception as exc:
            logger.warning("Layout analysis failed for page {}: {}", page_index, exc)
        return blocks

    # ------------------------------------------------------------------
    # Image detection
    # ------------------------------------------------------------------

    def _detect_images(
        self,
        page: Any,
        page_index: int,
        page_bbox: BBox,
    ) -> list[LayoutBlock]:
        """Detect embedded images on *page* and return ``FIGURE`` blocks.

        Walks the page's ``/XObject`` resources for ``/Image`` entries
        and attempts to resolve their placement from the content stream
        (``cm`` + ``Do`` operators).  Falls back to the image's native
        dimensions when placement cannot be determined.

        Args:
            page (Any): A ``pypdf`` page object.
            page_index (int): Zero-based page number.
            page_bbox (BBox): Full page bounding box (used as fallback).

        Returns:
            list[LayoutBlock]: A list of ``LayoutBlock`` items with ``type=FIGURE``.
        """
        blocks: list[LayoutBlock] = []
        try:
            # Collect image XObject names
            image_names = self._get_image_xobject_names(page)
            if not image_names:
                return blocks

            # Try to resolve placement from content stream
            placements = self._parse_image_placements(page, image_names)

            for img_name in image_names:
                bbox = placements.get(img_name, page_bbox)
                blocks.append(
                    LayoutBlock(
                        block_id=f"figure-{page_index}-{uuid.uuid4().hex[:8]}",
                        page_index=page_index,
                        type=BlockType.FIGURE,
                        bbox=bbox,
                        reading_order=0,
                        confidence=0.9 if img_name in placements else 0.5,
                        text="",
                    )
                )
        except Exception as exc:
            logger.debug("Image detection failed for page {}: {}", page_index, exc)
        return blocks

    @staticmethod
    def _get_image_xobject_names(page: Any) -> list[str]:
        """Return names of ``/Image`` XObjects on *page*.

        Args:
            page (Any): A ``pypdf`` page object.

        Returns:
            list[str]: A list of XObject name strings (e.g. ``"/Im1"``).
        """
        names: list[str] = []
        try:
            resources = page.get("/Resources")
            if not resources:
                return names
            xobjects = resources.get("/XObject")
            if not xobjects:
                return names
            xobj_dict = (
                xobjects.get_object() if hasattr(xobjects, "get_object") else xobjects
            )
            for name in xobj_dict:
                obj = xobj_dict[name]
                resolved = obj.get_object() if hasattr(obj, "get_object") else obj
                subtype = str(resolved.get("/Subtype", ""))
                if subtype == "/Image":
                    names.append(str(name))
        except Exception as exc:
            logger.debug("XObject enumeration failed: {}", exc)
        return names

    @staticmethod
    def _parse_image_placements(page: Any, image_names: list[str]) -> dict[str, BBox]:
        """Parse the content stream to find placement matrices for images.

        Looks for the ``cm`` (concat matrix) + ``Do`` (draw XObject)
        operator pattern and multiplies the accumulated transformation
        matrices to derive the on-page bounding box.

        Args:
            page (Any): A ``pypdf`` page object.
            image_names (list[str]): Names to look for in ``Do`` operands.

        Returns:
            dict[str, BBox]: Mapping of image name → ``BBox``.
        """
        placements: dict[str, BBox] = {}
        try:
            contents = page.get("/Contents")
            if not contents:
                return placements
            raw = contents.get_object()

            # Determine whether *raw* is a single stream or an array of
            # stream references.  Stream objects (EncodedStreamObject /
            # DecodedStreamObject) are iterable over their *dict keys*,
            # so we must check for ``get_data`` first to avoid treating a
            # single stream as an array.
            if hasattr(raw, "get_data"):
                # Single content stream
                try:
                    stream_text = raw.get_data().decode("latin-1")
                except Exception:
                    return placements
            elif hasattr(raw, "__iter__"):
                # Array of indirect stream references
                try:
                    data_parts: list[str] = []
                    for item in raw:
                        obj = item.get_object() if hasattr(item, "get_object") else item
                        data_parts.append(obj.get_data().decode("latin-1"))
                    stream_text = "\n".join(data_parts)
                except Exception:
                    return placements
            else:
                return placements

            # Tokenise and walk operators
            placements = _extract_image_bboxes_from_stream(
                stream_text, set(image_names)
            )
        except Exception as exc:
            logger.debug("Content stream parsing failed: {}", exc)
        return placements

    # ------------------------------------------------------------------
    # Table detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_tables(
        text: str,
        page_index: int,
        page_bbox: BBox,
        page_width: float,
        page_height: float,
    ) -> tuple[list[LayoutBlock], str]:
        """Detect table regions in *text* using caption and structure heuristics.

        The detector looks for *"Table N:"* / *"Table N."* captions and
        then attempts to identify the extent of the tabular data
        following the caption.  The table block gets the relevant text;
        the remaining text is returned separately.

        Args:
            text (str): Full extracted text from the page.
            page_index (int): Zero-based page number.
            page_bbox (BBox): Page bounding box (used for the table bbox).
            page_width (float): Page width in points.
            page_height (float): Page height in points.

        Returns:
            tuple[list[LayoutBlock], str]: A 2-tuple ``(table_blocks, remaining_text)``.
        """
        table_blocks: list[LayoutBlock] = []
        if not text.strip():
            return table_blocks, text

        lines = text.split("\n")
        table_regions: list[tuple[int, int]] = []

        # Regex to match table captions like "Table 1:", "Table 12."
        caption_re = re.compile(r"^Table\s+\d+\s*[:.]", re.IGNORECASE)

        i = 0
        while i < len(lines):
            if caption_re.match(lines[i].strip()):
                start = i
                end = _find_table_end(lines, start)
                table_regions.append((start, end))
                i = end + 1
            else:
                i += 1

        if not table_regions:
            return table_blocks, text

        # Build table blocks and collect remaining text
        remaining_lines: list[str] = []
        region_set: set[int] = set()
        for start, end in table_regions:
            for j in range(start, end + 1):
                region_set.add(j)
            table_text = "\n".join(lines[start : end + 1])
            table_blocks.append(
                LayoutBlock(
                    block_id=f"table-{page_index}-{uuid.uuid4().hex[:8]}",
                    page_index=page_index,
                    type=BlockType.TABLE,
                    bbox=page_bbox,
                    reading_order=0,
                    confidence=0.7,
                    text=table_text,
                )
            )

        for idx, line in enumerate(lines):
            if idx not in region_set:
                remaining_lines.append(line)

        return table_blocks, "\n".join(remaining_lines)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

# Regex for matching PDF number tokens (integers and floats, incl. negative)
_NUM_RE = re.compile(r"^-?(?:\d+\.?\d*|\.\d+)$")


def _extract_image_bboxes_from_stream(
    stream_text: str, image_names: set[str]
) -> dict[str, BBox]:
    """Walk a raw content-stream string and extract image bounding boxes.

    The function tracks a simplified graphics-state stack (``q`` / ``Q``
    save/restore) and accumulates ``cm`` (concat-matrix) operations.
    When a ``Do`` operator is encountered for a known image name the
    current transformation matrix is used to compute the on-page
    bounding box.

    Args:
        stream_text (str): Decoded content-stream data.
        image_names (set[str]): Set of XObject names to look for (e.g. ``{"/Im1"}``).

    Returns:
        dict[str, BBox]: Mapping of image name → ``BBox``.
    """
    placements: dict[str, BBox] = {}
    tokens = stream_text.split()
    # Graphics state: accumulated CTM components [a b c d e f]
    ctm = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    ctm_stack: list[list[float]] = []
    buf: list[str] = []

    for token in tokens:
        if token == "q":
            ctm_stack.append(list(ctm))
            buf.clear()
        elif token == "Q":
            if ctm_stack:
                ctm = ctm_stack.pop()
            buf.clear()
        elif token == "cm":
            if len(buf) >= 6:
                try:
                    vals = [float(v) for v in buf[-6:]]
                    ctm = _multiply_matrices(ctm, vals)
                except ValueError:
                    pass
            buf.clear()
        elif token == "Do":
            if buf:
                name = buf[-1]
                if name in image_names:
                    # CTM maps the unit square to the image rectangle
                    a, b, c, d, e, f = ctm
                    x0, y0 = e, f
                    x1 = a + e
                    y1 = d + f
                    # Normalise to ensure x0 < x1, y0 < y1
                    placements[name] = BBox(
                        x0=min(x0, x1),
                        y0=min(y0, y1),
                        x1=max(x0, x1),
                        y1=max(y0, y1),
                    )
            buf.clear()
        else:
            buf.append(token)
    return placements


def _multiply_matrices(current: list[float], new: list[float]) -> list[float]:
    """Multiply two 2-D affine matrices stored as ``[a b c d e f]``.

    The matrix layout follows PDF convention::

        | a  b  0 |
        | c  d  0 |
        | e  f  1 |

    Args:
        current (list[float]): Current transformation matrix.
        new (list[float]): New matrix to concatenate.

    Returns:
        list[float]: The resulting 6-element matrix list.
    """
    a1, b1, c1, d1, e1, f1 = current
    a2, b2, c2, d2, e2, f2 = new
    return [
        a2 * a1 + b2 * c1,
        a2 * b1 + b2 * d1,
        c2 * a1 + d2 * c1,
        c2 * b1 + d2 * d1,
        e2 * a1 + f2 * c1 + e1,
        e2 * b1 + f2 * d1 + f1,
    ]


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
    file_path: str | Path, pages: list[PageInfo]
) -> dict[int, list[LayoutBlock]]:
    """Run layout analysis on every page of *file_path*.

    Args:
        file_path (str | Path): Path to the PDF.
        pages (list[PageInfo]): Page triage results (used for page indices).

    Returns:
        dict[int, list[LayoutBlock]]: Mapping of ``page_index`` → list of ``LayoutBlock``.
    """
    file_path = Path(file_path)
    analyzer = PypdfLayoutAnalyzer(file_path)
    layout: dict[int, list[LayoutBlock]] = {}
    for page_info in pages:
        try:
            blocks = analyzer.analyze_page(page_info.page_index, file_path=file_path)
            layout[page_info.page_index] = blocks
        except Exception as exc:
            logger.warning(
                "Layout analysis skipped for page {}: {}", page_info.page_index, exc
            )
            layout[page_info.page_index] = []
    return layout
