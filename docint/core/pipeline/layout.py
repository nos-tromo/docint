"""Layout analysis interface and implementation."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from pathlib import Path

import pypdf
from loguru import logger

from docint.core.pipeline.models import BBox, BlockType, LayoutBlock, PageInfo


class LayoutAnalyzer(ABC):
    """Abstract interface for layout analysis backends."""

    @abstractmethod
    def analyze_page(
        self, page_index: int, *, file_path: Path | None = None
    ) -> list[LayoutBlock]:
        """Detect layout blocks on a single page.

        Args:
            page_index: Zero-based page number.
            file_path: Path to the source PDF (used by some backends).

        Returns:
            List of ``LayoutBlock`` items sorted by ``reading_order``.
        """


class PypdfLayoutAnalyzer(LayoutAnalyzer):
    """Minimal layout analyser backed by ``pypdf`` text extraction.

    Each page is treated as a single text block.  When the project
    integrates a more capable backend (e.g. Docling layout) this class
    can be replaced transparently.
    """

    def __init__(self, file_path: str | Path) -> None:
        self._file_path = Path(file_path)
        self._reader = pypdf.PdfReader(self._file_path)

    def analyze_page(
        self, page_index: int, *, file_path: Path | None = None
    ) -> list[LayoutBlock]:
        """Return a single ``TEXT`` block per page containing all extracted text."""
        blocks: list[LayoutBlock] = []
        try:
            page = self._reader.pages[page_index]
            text = page.extract_text() or ""
            mb = page.mediabox
            bbox = BBox(
                x0=float(mb.left),
                y0=float(mb.bottom),
                x1=float(mb.right),
                y1=float(mb.top),
            )
            blocks.append(
                LayoutBlock(
                    block_id=f"block-{page_index}-{uuid.uuid4().hex[:8]}",
                    page_index=page_index,
                    type=BlockType.TEXT,
                    bbox=bbox,
                    reading_order=0,
                    confidence=1.0 if text.strip() else 0.0,
                    text=text,
                )
            )
        except Exception as exc:
            logger.warning("Layout analysis failed for page {}: {}", page_index, exc)
        return blocks


def analyze_document(
    file_path: str | Path, pages: list[PageInfo]
) -> dict[int, list[LayoutBlock]]:
    """Run layout analysis on every page of *file_path*.

    Args:
        file_path: Path to the PDF.
        pages: Page triage results (used for page indices).

    Returns:
        Mapping of ``page_index`` → list of ``LayoutBlock``.
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
