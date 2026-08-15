"""Detection of page furniture: running heads, footers, page numbers, margin stamps.

Furniture is text that belongs to the *page*, not to the document's argument:
the running head repeated on every page, the footer, the page number, the
rotated stamp down the margin of a preprint. Left in place it lands in the
first and last text block of every page, so it is embedded with the body text,
returned as part of a citation snippet, and repeated across every chunk of a
long document.

Detection is deterministic and needs no model: it looks at where a line sits
on the page (top/bottom band, side margin), whether it looks like a page
number, and — the decisive signal — whether the same text recurs in the same
band across the document. A single page carries no evidence of repetition, so
only the position-and-shape rules apply there.
"""

from __future__ import annotations

import re
from collections import defaultdict

from docint.core.readers.documents.models import BlockType
from docint.core.readers.documents.parse import ParsedPdf, TextLine

# Fraction of the page height counted as the header / footer band, and of the
# page width counted as the side margin.
_BAND_FRACTION = 0.08
_MARGIN_FRACTION = 0.08
# A repeated band line must appear on at least this many pages and this share
# of the document before it counts as a running head or footer.
_MIN_REPEAT_PAGES = 2
_MIN_REPEAT_SHARE = 0.3
# Furniture lines are short; a full sentence in the band is body text.
_MAX_FURNITURE_CHARS = 120

_PAGE_NUMBER_PATTERNS = (
    # "7", "Page 7", "Seite 4 von 9", "p. 12 / 30"
    re.compile(r"^(?:page|seite|s\.|p\.)?\s*\d{1,4}(?:\s*(?:/|of|von)\s*\d{1,4})?$", re.IGNORECASE),
    # "- 3 -", "\u2014 iv \u2014" (hyphen, en dash or em dash on both sides)
    re.compile(r"^[-\u2013\u2014]\s*[0-9ivxlcdm]{1,7}\s*[-\u2013\u2014]$", re.IGNORECASE),
    # bare roman numerals
    re.compile(r"^[ivxlcdm]{1,7}$", re.IGNORECASE),
)


def _looks_like_page_number(text: str) -> bool:
    """Whether ``text`` reads as a page number in any common style."""
    stripped = text.strip()
    if not stripped or len(stripped) > 20:
        return False
    return any(pattern.match(stripped) for pattern in _PAGE_NUMBER_PATTERNS)


def _repetition_key(text: str) -> str:
    """Normalise a band line so the same head reads alike on every page.

    Digits collapse to ``#`` (so ``Page 3`` and ``Page 4`` share a key) and
    whitespace and case are normalised.

    Args:
        text (str): Raw line text.

    Returns:
        str: The normalised key.
    """
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", text)).strip().lower()


def _band_of(line: TextLine, page_height: float, page_width: float) -> BlockType | None:
    """Which furniture band ``line`` sits in, if any.

    Args:
        line (TextLine): The line to place.
        page_height (float): Page height in points.
        page_width (float): Page width in points.

    Returns:
        BlockType | None: ``PAGE_HEADER`` for the top band or a side margin,
            ``FOOTER`` for the bottom band, ``None`` for the body area.
    """
    if page_height <= 0 or page_width <= 0:
        return None
    if line.rotated:
        margin = _MARGIN_FRACTION * page_width
        if line.bbox.x1 <= margin or line.bbox.x0 >= page_width - margin:
            return BlockType.PAGE_HEADER
        return None
    if line.bbox.y0 >= page_height * (1 - _BAND_FRACTION):
        return BlockType.PAGE_HEADER
    if line.bbox.y1 <= page_height * _BAND_FRACTION:
        return BlockType.FOOTER
    return None


def detect_furniture(parsed: ParsedPdf) -> dict[int, dict[int, BlockType]]:
    """Classify every page's furniture lines.

    Runs in two passes over the document: the first places each line in a band
    and records how often each normalised band text recurs; the second promotes
    page numbers (no repetition needed — the text differs by design), rotated
    margin stamps, and band lines whose text repeats across the document.

    Args:
        parsed (ParsedPdf): An open document handle.

    Returns:
        dict[int, dict[int, BlockType]]: Page index → line index (into
            ``ParsedPage.lines``) → furniture type. Pages and lines with no
            furniture are absent.
    """
    page_count = parsed.page_count
    # (page_index, line_index) -> band; plus repetition bookkeeping
    banded: dict[tuple[int, int], BlockType] = {}
    rotated_margin: set[tuple[int, int]] = set()
    page_numbers: set[tuple[int, int]] = set()
    key_pages: defaultdict[tuple[str, BlockType], set[int]] = defaultdict(set)
    keys: dict[tuple[int, int], tuple[str, BlockType]] = {}

    for page_index in range(page_count):
        page = parsed.page(page_index)
        for line_index, line in enumerate(page.lines):
            band = _band_of(line, page.height, page.width)
            if band is None:
                continue
            text = line.text.strip()
            if not text or len(text) > _MAX_FURNITURE_CHARS:
                continue
            if line.rotated:
                rotated_margin.add((page_index, line_index))
                continue
            banded[(page_index, line_index)] = band
            if _looks_like_page_number(text):
                page_numbers.add((page_index, line_index))
                continue
            key = (_repetition_key(text), band)
            keys[(page_index, line_index)] = key
            key_pages[key].add(page_index)

    min_pages = max(_MIN_REPEAT_PAGES, round(_MIN_REPEAT_SHARE * page_count))
    result: defaultdict[int, dict[int, BlockType]] = defaultdict(dict)

    for page_index, line_index in rotated_margin:
        result[page_index][line_index] = BlockType.PAGE_HEADER
    for page_index, line_index in page_numbers:
        result[page_index][line_index] = BlockType.PAGE_NUMBER
    for (page_index, line_index), key in keys.items():
        if len(key_pages[key]) >= min_pages:
            result[page_index][line_index] = banded[(page_index, line_index)]

    return dict(result)
