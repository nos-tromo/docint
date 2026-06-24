"""Layout-aware, section-aware coarse-unit assembly.

The page-level PDF pipeline produces layout blocks with reading order and
type. This module groups those blocks into *coarse units* — one per
section, bounded by ``coarse_chunk_size`` — preserving page, bbox,
section-path, and table/image associations.

Sentence-level *fine* splitting is intentionally **not** done here: it is
delegated to the shared
:class:`~docint.core.storage.hierarchical.HierarchicalNodeParser`, so PDFs
flow through the same chunking machinery (overlap-free,
sentence-boundary-respecting :class:`SentenceSplitter`) as every other
file type. This replaces the previous bespoke character-overlap chunker
whose raw tail-slice overlap produced chunks that started mid-sentence.
"""

from __future__ import annotations

import hashlib

from loguru import logger

from docint.core.readers.documents.models import (
    BlockType,
    ChunkResult,
    ImageResult,
    LayoutBlock,
    PageText,
    TableResult,
)

# Blocks that update the running section path. They are not emitted as body
# text; their text becomes the heading prefix of the following unit(s).
_HEADING_BLOCK_TYPES = frozenset({BlockType.TITLE, BlockType.HEADER})

# Blocks that contribute prose to a coarse unit's body and may fall back to
# the page's full text when their own ``text`` is empty.
_PROSE_BLOCK_TYPES = frozenset({BlockType.TEXT, BlockType.LIST, BlockType.CAPTION})


def _stable_chunk_id(doc_id: str, page_index: int, block_id: str, idx: int) -> str:
    """Produce a deterministic chunk ID from the given inputs.

    Args:
        doc_id (str): The unique identifier for the document (e.g., sha256 of file bytes).
        page_index (int): The index of the first page the unit is derived from.
        block_id (str): The identifier of the first layout block in the unit.
        idx (int): The index of the unit within the document.

    Returns:
        str: A deterministic chunk ID string.
    """
    raw = f"{doc_id}:{page_index}:{block_id}:{idx}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _merge_source_mix(values: list[str]) -> str:
    """Collapse the per-block ``source_mix`` values of a unit into one label.

    Args:
        values (list[str]): The ordered ``source_mix`` values seen while
            accumulating a coarse unit (e.g. ``"pdf_text"``, ``"ocr"``).

    Returns:
        str: ``"pdf_text"`` when no values were seen, the single value when
            the unit is homogeneous, or ``"mixed"`` when blocks of
            different provenance were combined.
    """
    seen = [v for v in dict.fromkeys(values) if v]
    if not seen:
        return "pdf_text"
    if len(seen) == 1:
        return seen[0]
    return "mixed"


def build_coarse_units(
    doc_id: str,
    layout: dict[int, list[LayoutBlock]],
    page_texts: dict[int, PageText],
    tables: list[TableResult],
    images: list[ImageResult],
    coarse_chunk_size: int = 8192,
) -> list[ChunkResult]:
    """Group layout blocks into section-bounded coarse units.

    The walker visits pages in order and blocks in reading order. Heading
    blocks (``TITLE``/``HEADER``) update a running ``section_path`` and
    flush the in-progress unit so each unit belongs to a single section.
    Consecutive prose blocks under the current heading are accumulated
    until adding another would exceed ``coarse_chunk_size``, at which point
    the unit is flushed. The active section heading is prepended to each
    unit's text so the unit is self-contained; the fine (sentence-level)
    split is performed downstream by ``HierarchicalNodeParser``.

    No character overlap is applied — parent context is reconstructed at
    query time from the coarse parent via ``hier.parent_id``.

    Args:
        doc_id (str): Deterministic document identifier (sha256 of file bytes).
        layout (dict[int, list[LayoutBlock]]): Mapping of page_index → layout blocks.
        page_texts (dict[int, PageText]): Mapping of page_index → ``PageText``.
        tables (list[TableResult]): Extracted tables (linked by bbox overlap).
        images (list[ImageResult]): Extracted images (linked by bbox overlap).
        coarse_chunk_size (int): Maximum characters of body text per unit
            before a flush. A single block larger than this is emitted whole
            and re-split downstream.

    Returns:
        list[ChunkResult]: One ``ChunkResult`` per coarse unit, in reading order.
    """
    table_by_page: dict[int, list[TableResult]] = {}
    for t in tables:
        table_by_page.setdefault(t.page_index, []).append(t)
    image_by_page: dict[int, list[ImageResult]] = {}
    for img in images:
        image_by_page.setdefault(img.page_index, []).append(img)

    units: list[ChunkResult] = []
    section_path: list[str] = []

    # In-progress unit accumulator.
    acc_parts: list[str] = []
    acc_section: list[str] = []
    acc_block_ids: list[str] = []
    acc_pages: list[int] = []
    acc_bbox_refs: list[dict[str, float]] = []
    acc_table_ids: list[str] = []
    acc_image_ids: list[str] = []
    acc_source: list[str] = []
    acc_first_page: int | None = None
    acc_first_block: str | None = None
    # Pages whose body text has already been captured, so an empty prose
    # block does not pull the whole-page fallback text a second time.
    pages_captured: set[int] = set()

    def _acc_body_len() -> int:
        """Return the rendered length of the accumulated body (with separators)."""
        if not acc_parts:
            return 0
        return sum(len(p) for p in acc_parts) + 2 * (len(acc_parts) - 1)

    def _flush() -> None:
        """Emit the accumulated unit (if it has body text) and reset state."""
        nonlocal acc_first_page, acc_first_block
        body = "\n\n".join(p for p in acc_parts if p).strip()
        if body:
            heading = " > ".join(s for s in acc_section if s)
            text = f"{heading}\n\n{body}" if heading else body
            units.append(
                ChunkResult(
                    doc_id=doc_id,
                    chunk_id=_stable_chunk_id(
                        doc_id,
                        acc_first_page if acc_first_page is not None else 0,
                        acc_first_block or "unit",
                        len(units),
                    ),
                    text=text.strip(),
                    page_range=sorted(dict.fromkeys(acc_pages)),
                    block_ids=list(acc_block_ids),
                    section_path=list(acc_section),
                    table_ids=list(acc_table_ids),
                    image_ids=list(acc_image_ids),
                    source_mix=_merge_source_mix(acc_source),
                    bbox_refs=list(acc_bbox_refs),
                )
            )
        acc_parts.clear()
        acc_section.clear()
        acc_block_ids.clear()
        acc_pages.clear()
        acc_bbox_refs.clear()
        acc_table_ids.clear()
        acc_image_ids.clear()
        acc_source.clear()
        acc_first_page = None
        acc_first_block = None

    for page_idx in sorted(layout.keys()):
        blocks = sorted(layout[page_idx], key=lambda b: b.reading_order)
        page_text = page_texts.get(page_idx)
        page_source_mix = page_text.source_mix if page_text else "pdf_text"
        page_tables = table_by_page.get(page_idx, [])
        page_images = image_by_page.get(page_idx, [])

        for block in blocks:
            # Heading blocks delimit sections: flush the current unit, then
            # update the running section path (do not emit the heading as body).
            if block.type in _HEADING_BLOCK_TYPES:
                heading_text = block.text.strip()
                _flush()
                if heading_text:
                    section_path = _update_section_path(section_path, heading_text, block.type)
                continue

            text = block.text.strip()
            if not text:
                # Fall back to the page's OCR/full text — but only for prose
                # blocks and only once per page, so figure blocks and repeated
                # empty blocks do not duplicate the page text.
                if (
                    block.type in _PROSE_BLOCK_TYPES
                    and page_text is not None
                    and page_idx not in pages_captured
                    and page_text.full_text.strip()
                ):
                    text = page_text.full_text.strip()
                else:
                    continue

            # Flush before exceeding the coarse budget. A single block larger
            # than the budget is still added whole (re-split downstream).
            if acc_parts and _acc_body_len() + len(text) + 2 > coarse_chunk_size:
                _flush()

            if not acc_parts:
                acc_first_page = page_idx
                acc_first_block = block.block_id
                acc_section.extend(section_path)

            acc_parts.append(text)
            acc_block_ids.append(block.block_id)
            acc_pages.append(page_idx)
            pages_captured.add(page_idx)
            acc_bbox_refs.append(
                {
                    "x0": block.bbox.x0,
                    "y0": block.bbox.y0,
                    "x1": block.bbox.x1,
                    "y1": block.bbox.y1,
                }
            )
            acc_source.append(page_source_mix)
            for tbl in page_tables:
                if tbl.bbox.overlaps(block.bbox) and tbl.table_id not in acc_table_ids:
                    acc_table_ids.append(tbl.table_id)
            for img in page_images:
                if img.bbox.overlaps(block.bbox) and img.image_id not in acc_image_ids:
                    acc_image_ids.append(img.image_id)

    _flush()

    logger.info("Produced {} coarse units for document {}", len(units), doc_id[:12])
    return units


def _update_section_path(
    current_path: list[str], new_heading: str, block_type: BlockType = BlockType.TITLE
) -> list[str]:
    """Update section path based on heading block type.

    ``TITLE`` blocks reset the path (top-level heading); ``HEADER``
    blocks nest under the current title.  This provides a simple but
    reliable heading hierarchy based on layout block semantics rather
    than string length.

    Args:
        current_path (list[str]): The current section path.
        new_heading (str): The text of the new heading to add to the path.
        block_type (BlockType): The type of the block (TITLE or HEADER) to determine how to update

    Returns:
        list[str]: The updated section path list.
    """
    if not current_path:
        return [new_heading]
    if block_type == BlockType.TITLE:
        # Titles reset to top level
        return [new_heading]
    # Headers nest under the current path
    return [*current_path, new_heading]
