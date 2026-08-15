"""Page-level triage: classify each PDF page as digital, scanned, or mixed."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from docint.core.readers.documents.models import PageInfo
from docint.core.readers.documents.parse import ParsedPdf, lines_to_text, order_lines
from docint.utils.env_cfg import PipelineConfig


def triage_pdf(
    file_path: str | Path,
    config: PipelineConfig,
    *,
    parsed: ParsedPdf | None = None,
) -> list[PageInfo]:
    """Inspect every page of *file_path* and decide whether OCR is needed.

    Uses the docling-parse backbone (:class:`ParsedPdf`) to read each page's
    embedded text and compute a *text_coverage* heuristic
    (``chars / page_area * 1000``). Pages below
    ``config.text_coverage_threshold`` are flagged ``needs_ocr``.

    Args:
        file_path (str | Path): Path to the PDF file.
        config (PipelineConfig): Pipeline configuration (supplies the threshold).
        parsed (ParsedPdf | None): An already-open document handle to reuse
            (the orchestrator opens one per document). When ``None`` the file
            is opened here and closed before returning.

    Returns:
        list[PageInfo]: A list of ``PageInfo`` — one per page.
    """
    file_path = Path(file_path)
    pages: list[PageInfo] = []

    owned = parsed is None
    try:
        doc = parsed if parsed is not None else ParsedPdf(file_path)
    except Exception as exc:
        logger.error("Failed to open PDF for triage: {}: {}", file_path, exc)
        return [
            PageInfo(
                page_index=0,
                has_text_layer=False,
                text_coverage=0.0,
                needs_ocr=True,
                error=str(exc),
                status="failed",
            )
        ]

    try:
        for idx in range(doc.page_count):
            try:
                page = doc.page(idx)
                text = lines_to_text(order_lines(page.lines))
                width = page.width
                height = page.height
                area = width * height if (width > 0 and height > 0) else 1.0

                # Normalised coverage metric
                coverage = len(text.strip()) / area * 1000.0
                has_text = len(text.strip()) > 0
                needs_ocr = coverage < config.text_coverage_threshold

                pages.append(
                    PageInfo(
                        page_index=idx,
                        has_text_layer=has_text,
                        text_coverage=round(coverage, 4),
                        needs_ocr=needs_ocr,
                        width=width,
                        height=height,
                        status="completed",
                    )
                )
            except Exception as exc:
                logger.warning("Triage failed for page {}: {}", idx, exc)
                pages.append(
                    PageInfo(
                        page_index=idx,
                        has_text_layer=False,
                        text_coverage=0.0,
                        needs_ocr=True,
                        error=str(exc),
                        status="failed",
                    )
                )
    finally:
        if owned:
            doc.close()

    logger.info(
        "Triage complete for {}: {} pages, {} need OCR",
        file_path.name,
        len(pages),
        sum(1 for p in pages if p.needs_ocr),
    )
    return pages
