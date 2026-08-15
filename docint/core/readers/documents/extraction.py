"""Table and image extraction from layout blocks."""

from __future__ import annotations

import uuid
from pathlib import Path

import pypdfium2
import pypdfium2.raw as pdfium_raw
from loguru import logger

from docint.core.readers.documents.models import (
    BBox,
    BlockType,
    ImageResult,
    LayoutBlock,
    TableResult,
)

# Placement boxes come from docling-parse and pdfium reads the same content
# stream, so a match is exact up to float rounding.
_BBOX_MATCH_TOLERANCE = 1.0


def extract_tables(
    layout: dict[int, list[LayoutBlock]],
) -> list[TableResult]:
    """Extract table regions from layout blocks.

    For each ``TABLE`` block the row-major text, the bounding box and the
    reconstructed cell grid are captured (the grid is ``None`` when the region
    was too irregular to rebuild, in which case only the text is stored).

    Args:
        layout (dict[int, list[LayoutBlock]]): Mapping of page index → list of ``LayoutBlock``.

    Returns:
        list[TableResult]: List of ``TableResult`` items.
    """
    tables: list[TableResult] = []
    for page_idx, blocks in layout.items():
        for block in blocks:
            if block.type != BlockType.TABLE:
                continue
            table_id = f"table-{page_idx}-{uuid.uuid4().hex[:8]}"
            tables.append(
                TableResult(
                    table_id=table_id,
                    page_index=page_idx,
                    bbox=block.bbox,
                    raw_text=block.text,
                    cell_grid=block.cells,
                    confidence=block.confidence,
                )
            )
    logger.info("Extracted {} tables from layout blocks", len(tables))
    return tables


def extract_images(
    layout: dict[int, list[LayoutBlock]],
    file_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> list[ImageResult]:
    """Extract figure/image regions from layout blocks.

    For each ``FIGURE`` block the bounding box is recorded.  When
    *output_dir* is set, the embedded image drawn at that box is
    extracted via ``pypdfium2`` and saved as a PNG.

    Args:
        layout (dict[int, list[LayoutBlock]]): Mapping of page index → list of ``LayoutBlock``.
        file_path (str | Path | None): Path to the PDF (used for embedded image extraction).
        output_dir (str | Path | None): Directory to write extracted images.

    Returns:
        list[ImageResult]: List of ``ImageResult`` items.
    """
    images: list[ImageResult] = []
    for page_idx, blocks in layout.items():
        for block in blocks:
            if block.type != BlockType.FIGURE:
                continue
            image_id = f"image-{page_idx}-{uuid.uuid4().hex[:8]}"
            image_path: str | None = None

            if file_path and output_dir:
                image_path = _try_extract_embedded_image(
                    Path(file_path), page_idx, image_id, Path(output_dir), block.bbox
                )

            images.append(
                ImageResult(
                    image_id=image_id,
                    page_index=page_idx,
                    bbox=block.bbox,
                    image_path=image_path,
                    metadata={
                        "block_id": block.block_id,
                        "confidence": block.confidence,
                    },
                )
            )
    logger.info("Extracted {} images from layout blocks", len(images))
    return images


def _try_extract_embedded_image(
    file_path: Path,
    page_index: int,
    image_id: str,
    output_dir: Path,
    bbox: BBox | None = None,
) -> str | None:
    """Best-effort extraction of the embedded image drawn at ``bbox``.

    Uses ``pypdfium2`` (pdfium decodes every image filter) to enumerate the
    page's image objects, picks the one whose placement bounds match the
    FIGURE block's ``bbox`` (the layout stage took that box from
    docling-parse, which reports the same placement rectangle), renders it
    to a PIL image and writes a PNG. When no object matches — or ``bbox``
    is ``None`` — nothing is written.

    Args:
        file_path (Path): Source PDF.
        page_index (int): Page to inspect.
        image_id (str): Identifier for naming the output file.
        output_dir (Path): Where to write the image.
        bbox (BBox | None): Placement box of the FIGURE block.

    Returns:
        str | None: Path string to the written PNG, or ``None`` on failure.
    """
    if bbox is None:
        return None
    try:
        pdf = pypdfium2.PdfDocument(str(file_path))
        try:
            page = pdf[page_index]
            match: pypdfium2.PdfImage | None = None
            for obj in page.get_objects(filter=(pdfium_raw.FPDF_PAGEOBJ_IMAGE,)):
                if not isinstance(obj, pypdfium2.PdfImage):
                    continue
                x0, y0, x1, y1 = obj.get_bounds()
                if (
                    abs(x0 - bbox.x0) <= _BBOX_MATCH_TOLERANCE
                    and abs(y0 - bbox.y0) <= _BBOX_MATCH_TOLERANCE
                    and abs(x1 - bbox.x1) <= _BBOX_MATCH_TOLERANCE
                    and abs(y1 - bbox.y1) <= _BBOX_MATCH_TOLERANCE
                ):
                    match = obj
                    break
            if match is None:
                logger.debug("No image object matches bbox {} on page {}", bbox, page_index)
                return None
            pil_image = match.get_bitmap().to_pil()
            output_dir.mkdir(parents=True, exist_ok=True)
            img_path = output_dir / f"{image_id}.png"
            pil_image.save(str(img_path), "PNG")
            logger.debug("Extracted image: {}", img_path)
            return str(img_path)
        finally:
            pdf.close()
    except Exception as exc:
        logger.debug("Embedded image extraction failed: {}", exc)
    return None


def _bbox_to_dict(bbox: BBox) -> dict[str, float]:
    """Serialise a ``BBox`` to a plain dict.

    Args:
        bbox (BBox): The bounding box to serialise.

    Returns:
        dict[str, float]: A dictionary with keys "x0", "y0", "x1", "y1" representing the bounding box coordinates.
    """
    return {"x0": bbox.x0, "y0": bbox.y0, "x1": bbox.x1, "y1": bbox.y1}
