"""Layout-aware, section-aware, sentence-level chunking."""

from __future__ import annotations

import hashlib
import re

from loguru import logger

from docint.core.readers.documents.models import (
    BlockType,
    ChunkResult,
    ImageResult,
    LayoutBlock,
    PageText,
    TableResult,
)


def _sentence_split(text: str) -> list[str]:
    """Split text into sentences using a simple regex heuristic.

    Args:
        text: The input text to split into sentences.

    Returns:
        A list of sentence strings extracted from the input text.
    """
    if not text.strip():
        return []
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _stable_chunk_id(doc_id: str, page_index: int, block_id: str, idx: int) -> str:
    """Produce a deterministic chunk ID from the given inputs.

    Args:
        doc_id: The unique identifier for the document (e.g., sha256 of file bytes).
        page_index: The index of the page the chunk is derived from.
        block_id: The identifier of the layout block the chunk is derived from.
        idx: The index of the chunk within the block (for multiple chunks from the same block

    Returns:
        A deterministic chunk ID string.
    """
    raw = f"{doc_id}:{page_index}:{block_id}:{idx}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def chunk_document(
    doc_id: str,
    layout: dict[int, list[LayoutBlock]],
    page_texts: dict[int, PageText],
    tables: list[TableResult],
    images: list[ImageResult],
    chunk_size: int = 1024,
    chunk_overlap: int = 64,
) -> list[ChunkResult]:
    """Produce layout-aware chunks from processed document data.

    The chunker walks pages in order, respects reading order within each
    page, tracks a *section_path* derived from heading/title blocks, and
    splits text at the sentence level inside each block.

    Args:
        doc_id: Deterministic document identifier (sha256 of file bytes).
        layout: Mapping of page_index → layout blocks.
        page_texts: Mapping of page_index → ``PageText``.
        tables: Extracted tables.
        images: Extracted images.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap characters between consecutive chunks.

    Returns:
        List of ``ChunkResult`` items.
    """
    table_by_page: dict[int, list[TableResult]] = {}
    for t in tables:
        table_by_page.setdefault(t.page_index, []).append(t)
    image_by_page: dict[int, list[ImageResult]] = {}
    for img in images:
        image_by_page.setdefault(img.page_index, []).append(img)

    chunks: list[ChunkResult] = []
    section_path: list[str] = []

    sorted_pages = sorted(layout.keys())
    for page_idx in sorted_pages:
        blocks = sorted(layout[page_idx], key=lambda b: b.reading_order)
        page_text = page_texts.get(page_idx)
        source_mix = page_text.source_mix if page_text else "pdf_text"

        for block in blocks:
            # Update section path on title/header blocks
            if block.type in {BlockType.TITLE, BlockType.HEADER}:
                title_text = block.text.strip()
                if title_text:
                    section_path = _update_section_path(
                        section_path, title_text, block.type
                    )

            # Determine related tables/images for this block
            page_tables = table_by_page.get(page_idx, [])
            page_images = image_by_page.get(page_idx, [])
            related_table_ids = [
                t.table_id for t in page_tables if t.bbox.overlaps(block.bbox)
            ]
            related_image_ids = [
                img.image_id for img in page_images if img.bbox.overlaps(block.bbox)
            ]

            # Get the text to chunk
            text = block.text.strip()
            if not text and page_text:
                text = page_text.full_text

            if not text:
                continue

            # Sentence-level splitting
            sentences = _sentence_split(text)
            if not sentences:
                continue

            # Build chunks respecting chunk_size
            current_text = ""
            current_sentences: list[str] = []
            chunk_idx = 0

            for sentence in sentences:
                if current_text and len(current_text) + len(sentence) + 1 > chunk_size:
                    # Emit chunk
                    chunks.append(
                        _make_chunk(
                            doc_id=doc_id,
                            page_idx=page_idx,
                            block=block,
                            text=current_text,
                            chunk_idx=chunk_idx,
                            section_path=list(section_path),
                            source_mix=source_mix,
                            table_ids=related_table_ids,
                            image_ids=related_image_ids,
                        )
                    )
                    chunk_idx += 1

                    # Apply overlap
                    if chunk_overlap > 0 and current_text:
                        overlap_text = current_text[-chunk_overlap:]
                        current_text = overlap_text + " " + sentence
                    else:
                        current_text = sentence
                    current_sentences = [sentence]
                else:
                    current_text = (
                        current_text + " " + sentence if current_text else sentence
                    )
                    current_sentences.append(sentence)

            # Emit remaining text
            if current_text.strip():
                chunks.append(
                    _make_chunk(
                        doc_id=doc_id,
                        page_idx=page_idx,
                        block=block,
                        text=current_text,
                        chunk_idx=chunk_idx,
                        section_path=list(section_path),
                        source_mix=source_mix,
                        table_ids=related_table_ids,
                        image_ids=related_image_ids,
                    )
                )

    logger.info("Produced {} chunks for document {}", len(chunks), doc_id[:12])
    return chunks


def _make_chunk(
    *,
    doc_id: str,
    page_idx: int,
    block: LayoutBlock,
    text: str,
    chunk_idx: int,
    section_path: list[str],
    source_mix: str,
    table_ids: list[str],
    image_ids: list[str],
) -> ChunkResult:
    """Create a single ``ChunkResult``.

    Args:
        doc_id: The unique identifier for the document (e.g., sha256 of file bytes).
        page_idx: The index of the page the chunk is derived from.
        block: The layout block the chunk is derived from.
        text: The text content of the chunk.
        chunk_idx: The index of the chunk within the block (for multiple chunks from the same block).
        section_path: The hierarchical section path derived from heading/title blocks.
        source_mix: The source of the text (e.g., "pdf_text", "ocr_text").
        table_ids: List of related table identifiers that overlap with the block.
        image_ids: List of related image identifiers that overlap with the block.

    Returns:
        A ``ChunkResult`` instance representing the chunked text and its metadata.
    """
    chunk_id = _stable_chunk_id(doc_id, page_idx, block.block_id, chunk_idx)
    return ChunkResult(
        doc_id=doc_id,
        chunk_id=chunk_id,
        text=text.strip(),
        page_range=[page_idx],
        block_ids=[block.block_id],
        section_path=section_path,
        table_ids=table_ids,
        image_ids=image_ids,
        source_mix=source_mix,
        bbox_refs=[
            {
                "x0": block.bbox.x0,
                "y0": block.bbox.y0,
                "x1": block.bbox.x1,
                "y1": block.bbox.y1,
            }
        ],
    )


def _update_section_path(
    current_path: list[str], new_heading: str, block_type: BlockType = BlockType.TITLE
) -> list[str]:
    """Update section path based on heading block type.

    ``TITLE`` blocks reset the path (top-level heading); ``HEADER``
    blocks nest under the current title.  This provides a simple but
    reliable heading hierarchy based on layout block semantics rather
    than string length.

    Args:
        current_path: The current section path.
        new_heading: The text of the new heading to add to the path.
        block_type: The type of the block (TITLE or HEADER) to determine how to update

    Returns:
        The updated section path list.
    """
    if not current_path:
        return [new_heading]
    if block_type == BlockType.TITLE:
        # Titles reset to top level
        return [new_heading]
    # Headers nest under the current path
    return current_path + [new_heading]
