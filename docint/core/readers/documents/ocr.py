"""OCR fallback for scanned pages."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pypdf
from loguru import logger

from docint.core.readers.documents.models import (
    BBox,
    LayoutBlock,
    OCRSpan,
    PageInfo,
    PageText,
)


class OCREngine(ABC):
    """Abstract OCR engine wrapper."""

    @abstractmethod
    def ocr_page(
        self, page_index: int, *, file_path: Path | None = None
    ) -> list[OCRSpan]:
        """Run OCR on a single page and return text spans.

        Args:
            page_index: Zero-based page number.
            file_path: Path to the source PDF.

        Returns:
            List of ``OCRSpan`` items.
        """


class PypdfTextEngine(OCREngine):
    """Fallback engine that extracts text directly using ``pypdf``.

    This serves as the baseline "OCR" when a dedicated OCR backend
    (e.g. PaddleOCR) is not available.
    """

    def __init__(self, file_path: str | Path) -> None:
        """Initialize the engine with the PDF file path.

        Args:
            file_path (str | Path): Path to the PDF file.
        """
        self._file_path = Path(file_path)
        self._reader = pypdf.PdfReader(self._file_path)

    def ocr_page(
        self, page_index: int, *, file_path: Path | None = None
    ) -> list[OCRSpan]:
        """Extract text from a page using pypdf (no actual OCR).
        This method attempts to extract text directly from the PDF page
        without performing any OCR.

        Args:
            page_index: Zero-based page number.
            file_path: Ignored (present for interface compatibility).

        Returns:
            List of ``OCRSpan`` items containing the extracted text and its bounding box.
        """
        spans: list[OCRSpan] = []
        try:
            page = self._reader.pages[page_index]
            text = page.extract_text() or ""
            if text.strip():
                mb = page.mediabox
                spans.append(
                    OCRSpan(
                        text=text.strip(),
                        bbox=BBox(
                            x0=float(mb.left),
                            y0=float(mb.bottom),
                            x1=float(mb.right),
                            y1=float(mb.top),
                        ),
                        confidence=1.0,
                        source="pdf_text",
                    )
                )
        except Exception as exc:
            logger.warning("Text extraction failed for page {}: {}", page_index, exc)
        return spans


def build_page_text(
    page_info: PageInfo,
    layout_blocks: list[LayoutBlock],
    ocr_spans: list[OCRSpan],
) -> PageText:
    """Aggregate text sources for a page into a ``PageText`` result.

    Args:
        page_info: Triage info for the page.
        layout_blocks: Layout blocks detected on the page.
        ocr_spans: OCR spans for pages that needed OCR.

    Returns:
        A ``PageText`` combining all sources.
    """
    pdf_spans: list[OCRSpan] = []
    for block in layout_blocks:
        if block.text.strip():
            pdf_spans.append(
                OCRSpan(
                    text=block.text.strip(),
                    bbox=block.bbox,
                    confidence=block.confidence,
                    source="pdf_text",
                )
            )

    all_spans = pdf_spans + ocr_spans
    full_text = "\n".join(s.text for s in all_spans if s.text.strip())

    has_pdf = bool(pdf_spans)
    has_ocr = bool(ocr_spans)
    if has_pdf and has_ocr:
        source_mix = "mixed"
    elif has_ocr:
        source_mix = "ocr"
    else:
        source_mix = "pdf_text"

    avg_confidence = (
        sum(s.confidence for s in all_spans) / len(all_spans) if all_spans else 0.0
    )

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
) -> dict[int, PageText]:
    """Extract text for all pages, applying OCR fallback where needed.

    Args:
        file_path: Path to the PDF.
        pages: Page triage results.
        layout: Layout blocks per page.

    Returns:
        Mapping of ``page_index`` → ``PageText``.
    """
    file_path = Path(file_path)
    engine = PypdfTextEngine(file_path)
    result: dict[int, PageText] = {}

    for page_info in pages:
        ocr_spans: list[OCRSpan] = []
        if page_info.needs_ocr:
            try:
                ocr_spans = engine.ocr_page(page_info.page_index, file_path=file_path)
            except Exception as exc:
                logger.warning("OCR failed for page {}: {}", page_info.page_index, exc)

        blocks = layout.get(page_info.page_index, [])
        result[page_info.page_index] = build_page_text(page_info, blocks, ocr_spans)

    return result
