"""Reading a page's text: the embedded text layer, and OCR blocks as layout.

The OCR *client* lives in :mod:`docint.core.ocr` — one engine for every
caller. This module is the documents-side half: the text-layer engine, the
translation of the engine's answer into :class:`LayoutBlock` s, and the
aggregation of both into a page's :class:`PageText`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from pathlib import Path

from loguru import logger
from typing_extensions import override

from docint.core.ocr import OcrBlock, OcrCategory
from docint.core.readers.documents.furniture import looks_like_page_number
from docint.core.readers.documents.models import (
    BBox,
    BlockType,
    LayoutBlock,
    OCRSpan,
    PageInfo,
    PageText,
)
from docint.core.readers.documents.parse import ParsedPdf, order_lines

# Page furniture is not part of the page's text (mirrors chunking.py).
_FURNITURE_BLOCK_TYPES = frozenset(
    {BlockType.PAGE_HEADER, BlockType.FOOTER, BlockType.PAGE_NUMBER, BlockType.FIGURE_TEXT}
)

# What the OCR package's categories mean to a document. ``PAGE_FOOTER`` is
# resolved per block: a footer that reads as nothing but a number is a page
# number, and the chunker treats the two alike anyway.
_CATEGORY_TO_BLOCK = {
    OcrCategory.TITLE: BlockType.TITLE,
    OcrCategory.SECTION_HEADER: BlockType.HEADER,
    OcrCategory.TEXT: BlockType.TEXT,
    OcrCategory.LIST_ITEM: BlockType.LIST,
    OcrCategory.CAPTION: BlockType.CAPTION,
    # A footnote and a formula are body content, not furniture: they are
    # printed because they carry meaning, and a reader citing the page
    # expects them in it.
    OcrCategory.FOOTNOTE: BlockType.TEXT,
    OcrCategory.FORMULA: BlockType.TEXT,
    OcrCategory.TABLE: BlockType.TABLE,
    OcrCategory.PICTURE: BlockType.FIGURE,
    OcrCategory.PAGE_HEADER: BlockType.PAGE_HEADER,
    OcrCategory.PAGE_FOOTER: BlockType.FOOTER,
}


def blocks_from_ocr(
    page_index: int, blocks: list[OcrBlock], *, keep: list[LayoutBlock] | None = None
) -> list[LayoutBlock]:
    """Translate one page's OCR answer into layout blocks.

    The engine speaks its own vocabulary so that reading pixels does not
    depend on the document reader; this is where that vocabulary becomes a
    page's layout. Blocks arrive in the model's reading order and keep it.

    Args:
        page_index (int): Zero-based page number.
        blocks (list[OcrBlock]): What the OCR engine read on that page.
        keep (list[LayoutBlock] | None): Blocks from the geometric pass to
            retain ahead of the OCR ones — used for the figures on a page
            whose OCR model reads text only and so cannot report them.

    Returns:
        list[LayoutBlock]: Layout blocks in reading order, renumbered.
    """
    kept = list(keep or [])
    out: list[LayoutBlock] = []
    for order, block in enumerate(kept):
        out.append(replace(block, reading_order=order))

    for offset, block in enumerate(blocks):
        text = block.text.strip()
        block_type = _CATEGORY_TO_BLOCK.get(block.category, BlockType.TEXT)
        if block_type is BlockType.FOOTER and looks_like_page_number(text):
            block_type = BlockType.PAGE_NUMBER
        order = len(kept) + offset
        out.append(
            LayoutBlock(
                block_id=f"ocr-{page_index}-{offset}",
                page_index=page_index,
                type=block_type,
                bbox=BBox(x0=block.bbox.x0, y0=block.bbox.y0, x1=block.bbox.x1, y1=block.bbox.y1),
                reading_order=order,
                confidence=0.9,
                text=text,
                cells=block.cells,
                cells_source="ocr" if block.cells else "geometry",
            )
        )
    return out


class OCREngine(ABC):
    """Abstract OCR engine wrapper."""

    @abstractmethod
    def ocr_page(self, page_index: int, *, file_path: Path | None = None) -> list[OCRSpan]:
        """Run OCR on a single page and return text spans.

        Args:
            page_index (int): Zero-based page number.
            file_path (Path | None): Path to the source PDF.

        Returns:
            list[OCRSpan]: List of ``OCRSpan`` items.
        """


class PdfTextEngine(OCREngine):
    """Text-layer engine backed by the docling-parse backbone (no actual OCR).

    Serves as the baseline "OCR" for pages triage flagged as low-coverage:
    it re-reads the page's embedded text and returns one span per line, in
    reading order, with the line's real bounding box. When it yields nothing
    the vision engine is tried.
    """

    def __init__(self, file_path: str | Path, *, parsed: ParsedPdf | None = None) -> None:
        """Open the PDF (or adopt an already-open handle).

        Args:
            file_path (str | Path): Path to the PDF file.
            parsed (ParsedPdf | None): Open document handle to reuse; when
                ``None`` the file is opened here and released by :meth:`close`.
        """
        self._file_path = Path(file_path)
        self._owned = parsed is None
        self._doc = parsed if parsed is not None else ParsedPdf(self._file_path)

    def close(self) -> None:
        """Release the document handle if this engine opened it."""
        if self._owned:
            self._doc.close()

    @override
    def ocr_page(self, page_index: int, *, file_path: Path | None = None) -> list[OCRSpan]:
        """Extract the page's text layer as per-line spans.

        Args:
            page_index (int): Zero-based page number.
            file_path (Path | None): Ignored (present for interface compatibility).

        Returns:
            list[OCRSpan]: One ``OCRSpan`` per non-empty line, in reading order.
        """
        spans: list[OCRSpan] = []
        try:
            page = self._doc.page(page_index)
            for line in order_lines(page.lines):
                text = line.text.strip()
                if not text:
                    continue
                spans.append(OCRSpan(text=text, bbox=line.bbox, confidence=1.0, source="pdf_text"))
        except Exception as exc:
            logger.warning("Text extraction failed for page {}: {}", page_index, exc)
        return spans


def build_page_text(
    page_info: PageInfo,
    layout_blocks: list[LayoutBlock],
    ocr_spans: list[OCRSpan],
    *,
    block_source: str = "pdf_text",
) -> PageText:
    """Aggregate text sources for a page into a ``PageText`` result.

    Args:
        page_info (PageInfo): Triage info for the page.
        layout_blocks (list[LayoutBlock]): Layout blocks detected on the page.
        ocr_spans (list[OCRSpan]): OCR spans for pages that needed OCR.
        block_source (str): Where the blocks' text came from — ``"pdf_text"``
            for a text layer, ``"ocr"`` when the layout-OCR model read the page.

    Returns:
        PageText: A ``PageText`` combining all sources.
    """
    pdf_spans: list[OCRSpan] = []
    for block in layout_blocks:
        if block.type in _FURNITURE_BLOCK_TYPES:
            continue
        if block.text.strip():
            pdf_spans.append(
                OCRSpan(
                    text=block.text.strip(),
                    bbox=block.bbox,
                    confidence=block.confidence,
                    source=block_source,
                )
            )

    all_spans = pdf_spans + ocr_spans
    full_text = "\n".join(s.text for s in all_spans if s.text.strip())

    has_pdf = bool(pdf_spans) and block_source == "pdf_text"
    has_ocr = bool(ocr_spans) or (bool(pdf_spans) and block_source != "pdf_text")
    if has_pdf and has_ocr:
        source_mix = "mixed"
    elif has_ocr:
        source_mix = "ocr"
    else:
        source_mix = "pdf_text"

    avg_confidence = sum(s.confidence for s in all_spans) / len(all_spans) if all_spans else 0.0

    return PageText(
        page_index=page_info.page_index,
        pdf_text_spans=pdf_spans,
        ocr_spans=ocr_spans,
        full_text=full_text,
        source_mix=source_mix,
        confidence=round(avg_confidence, 4),
    )


def extract_text_for_pages(
    file_path: str | Path,
    pages: list[PageInfo],
    layout: dict[int, list[LayoutBlock]],
    *,
    parsed: ParsedPdf | None = None,
) -> dict[int, PageText]:
    """Extract each page's text from its own text layer.

    Pages this cannot answer for — a scan has no text layer — are read by the
    OCR engine, which the orchestrator drives afterwards: what it returns is a
    page's *layout*, not a set of spans, so it belongs on the stage that owns
    the layout.

    Args:
        file_path (str | Path): Path to the PDF.
        pages (list[PageInfo]): Page triage results.
        layout (dict[int, list[LayoutBlock]]): Layout blocks per page.
        parsed (ParsedPdf | None): Open document handle to reuse.

    Returns:
        dict[int, PageText]: Mapping of ``page_index`` → ``PageText``.
    """
    file_path = Path(file_path)
    engine = PdfTextEngine(file_path, parsed=parsed)
    result: dict[int, PageText] = {}

    try:
        for page_info in pages:
            result[page_info.page_index] = _extract_page_text(file_path, page_info, layout, engine)
    finally:
        engine.close()

    return result


def _extract_page_text(
    file_path: Path,
    page_info: PageInfo,
    layout: dict[int, list[LayoutBlock]],
    engine: OCREngine,
) -> PageText:
    """Extract one page's text from the text layer.

    Args:
        file_path (Path): Path to the PDF.
        page_info (PageInfo): Triage result for the page.
        layout (dict[int, list[LayoutBlock]]): Layout blocks per page.
        engine (OCREngine): Text-layer engine used for ``needs_ocr`` pages.

    Returns:
        PageText: The page's aggregated text.
    """
    ocr_spans: list[OCRSpan] = []
    if page_info.needs_ocr:
        try:
            ocr_spans = engine.ocr_page(page_info.page_index, file_path=file_path)
        except Exception as exc:
            logger.warning("OCR failed for page {}: {}", page_info.page_index, exc)

    blocks = layout.get(page_info.page_index, [])
    return build_page_text(page_info, blocks, ocr_spans)
