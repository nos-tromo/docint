"""Data models for the document processing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BlockType(str, Enum):
    """Types of layout blocks detected in a document page."""

    TITLE = "title"
    TEXT = "text"
    LIST = "list"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"


@dataclass
class BBox:
    """Axis-aligned bounding box in page coordinates."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def area(self) -> float:
        """Return the area of the bounding box."""
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)

    def overlaps(self, other: BBox) -> bool:
        """Return True if this bbox overlaps with *other*."""
        return not (
            self.x1 <= other.x0
            or other.x1 <= self.x0
            or self.y1 <= other.y0
            or other.y1 <= self.y0
        )


@dataclass
class LayoutBlock:
    """A detected layout block on a single page."""

    block_id: str
    page_index: int
    type: BlockType
    bbox: BBox
    reading_order: int
    confidence: float
    text: str = ""


@dataclass
class OCRSpan:
    """A single text span produced by OCR or direct text extraction."""

    text: str
    bbox: BBox
    confidence: float
    source: str = "ocr"  # "pdf_text" or "ocr"


@dataclass
class PageInfo:
    """Triage result for a single PDF page."""

    page_index: int
    has_text_layer: bool
    text_coverage: float
    needs_ocr: bool
    width: float = 0.0
    height: float = 0.0
    error: str | None = None
    status: str = "pending"  # "pending", "completed", "failed"


@dataclass
class PageText:
    """Aggregated text extraction result for a page."""

    page_index: int
    pdf_text_spans: list[OCRSpan] = field(default_factory=list)
    ocr_spans: list[OCRSpan] = field(default_factory=list)
    full_text: str = ""
    source_mix: str = "pdf_text"  # "pdf_text", "ocr", "mixed"
    confidence: float = 1.0


@dataclass
class TableResult:
    """Extracted table metadata and content."""

    table_id: str
    page_index: int
    bbox: BBox
    raw_text: str
    cell_grid: list[list[str]] | None = None
    confidence: float = 0.0
    csv_path: str | None = None


@dataclass
class ImageResult:
    """Extracted image metadata."""

    image_id: str
    page_index: int
    bbox: BBox
    image_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkResult:
    """A single chunk produced by layout-aware chunking."""

    doc_id: str
    chunk_id: str
    text: str
    page_range: list[int]
    block_ids: list[str]
    section_path: list[str]
    table_ids: list[str]
    image_ids: list[str]
    source_mix: str
    bbox_refs: list[dict[str, float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentManifest:
    """Top-level manifest describing a processed document."""

    doc_id: str
    file_path: str
    file_name: str
    pipeline_version: str
    pages: list[PageInfo] = field(default_factory=list)
    tables_found: int = 0
    images_found: int = 0
    pages_total: int = 0
    pages_ocr: int = 0
    pages_failed: int = 0
    status: str = "pending"
    error: str | None = None
