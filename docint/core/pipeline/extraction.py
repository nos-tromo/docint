"""Table and image extraction from layout blocks."""

from __future__ import annotations

import uuid
from pathlib import Path

import pypdf
from loguru import logger

from docint.core.pipeline.models import (
    BBox,
    BlockType,
    ImageResult,
    LayoutBlock,
    TableResult,
)


def extract_tables(
    layout: dict[int, list[LayoutBlock]],
) -> list[TableResult]:
    """Extract table regions from layout blocks.

    For each ``TABLE`` block the raw text and bounding box are captured.
    Structure-level cell grids are populated when the layout backend
    provides them; otherwise only the text is stored (best-effort).

    Args:
        layout: Mapping of page index → list of ``LayoutBlock``.

    Returns:
        List of ``TableResult`` items.
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
    *output_dir* is set, embedded images are extracted via ``pypdf``
    and saved as PNGs.

    Args:
        layout: Mapping of page index → list of ``LayoutBlock``.
        file_path: Path to the PDF (used for embedded image extraction).
        output_dir: Directory to write extracted images.

    Returns:
        List of ``ImageResult`` items.
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
                    Path(file_path), page_idx, image_id, Path(output_dir)
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
    file_path: Path, page_index: int, image_id: str, output_dir: Path
) -> str | None:
    """Best-effort extraction of embedded images via ``pypdf``.

    Args:
        file_path: Source PDF.
        page_index: Page to inspect.
        image_id: Identifier for naming the output file.
        output_dir: Where to write the PNG.

    Returns:
        Path string to the written image, or ``None`` on failure.
    """
    try:
        reader = pypdf.PdfReader(file_path)
        page = reader.pages[page_index]
        x_objects = page.get("/Resources", {})
        if hasattr(x_objects, "get"):
            x_objects = x_objects.get("/XObject", {})
        else:
            return None

        for obj_name in x_objects:
            obj = x_objects[obj_name]
            resolved = obj.get_object() if hasattr(obj, "get_object") else obj
            subtype = resolved.get("/Subtype", "")
            if subtype == "/Image":
                output_dir.mkdir(parents=True, exist_ok=True)
                img_path = output_dir / f"{image_id}.png"
                try:
                    data = resolved.get_data()
                    img_path.write_bytes(data)
                    return str(img_path)
                except Exception:
                    pass
    except Exception as exc:
        logger.debug("Embedded image extraction failed: {}", exc)
    return None


def _bbox_to_dict(bbox: BBox) -> dict[str, float]:
    """Serialise a ``BBox`` to a plain dict."""
    return {"x0": bbox.x0, "y0": bbox.y0, "x1": bbox.x1, "y1": bbox.y1}
