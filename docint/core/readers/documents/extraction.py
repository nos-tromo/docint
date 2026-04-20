"""Table and image extraction from layout blocks."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pypdf
from loguru import logger

from docint.core.readers.documents.models import (
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

    Uses ``pypdf``'s ``page.images`` API which returns decoded image
    data ready for writing.  Falls back to manual ``/XObject``
    inspection when the higher-level API is unavailable.

    Args:
        file_path (Path): Source PDF.
        page_index (int): Page to inspect.
        image_id (str): Identifier for naming the output file.
        output_dir (Path): Where to write the image.

    Returns:
        str | None: Path string to the written image, or ``None`` on failure.
    """
    try:
        reader = pypdf.PdfReader(file_path)
        page = reader.pages[page_index]

        # pypdf >= 3.x exposes page.images with decoded data
        if hasattr(page, "images") and page.images:
            output_dir.mkdir(parents=True, exist_ok=True)
            for idx, img in enumerate(page.images):
                ext = Path(img.name).suffix or ".png"
                img_path = output_dir / f"{image_id}-{idx}{ext}"
                img_path.write_bytes(img.data)
                logger.debug("Extracted image: {}", img_path)
                # Return the first extracted image path
                return str(img_path)

        # Fallback: manual XObject extraction with Pillow decoding
        return _try_extract_xobject_image(page, image_id, output_dir)

    except Exception as exc:
        logger.debug("Embedded image extraction failed: {}", exc)
    return None


def _try_extract_xobject_image(
    page: object, image_id: str, output_dir: Path
) -> str | None:
    """Attempt to extract an image from page XObjects using Pillow.

    Args:
        page (object): A ``pypdf`` page object.
        image_id (str): Identifier for naming the output file.
        output_dir (Path): Where to write the PNG.

    Returns:
        str | None: Path string to the written image, or ``None`` on failure.
    """
    try:
        import io

        from PIL import Image

        x_objects: Any = getattr(page, "get", lambda *a: None)("/Resources", {})
        if hasattr(x_objects, "get"):
            x_objects = x_objects.get("/XObject", {})
        else:
            return None

        if hasattr(x_objects, "get_object"):
            x_objects = x_objects.get_object()

        for obj_name in x_objects:
            obj = x_objects[obj_name]
            resolved = obj.get_object() if hasattr(obj, "get_object") else obj
            subtype = str(resolved.get("/Subtype", ""))
            if subtype != "/Image":
                continue

            output_dir.mkdir(parents=True, exist_ok=True)
            try:
                data = resolved.get_data()
                width = int(resolved.get("/Width", 0))
                height = int(resolved.get("/Height", 0))
                color_space = str(resolved.get("/ColorSpace", "/DeviceRGB"))

                if width > 0 and height > 0:
                    mode = "RGB" if "RGB" in color_space else "L"
                    try:
                        img = Image.frombytes(mode, (width, height), data)
                    except Exception:
                        img = Image.open(io.BytesIO(data))
                else:
                    img = Image.open(io.BytesIO(data))

                img_path = output_dir / f"{image_id}.png"
                img.save(str(img_path), "PNG")
                return str(img_path)
            except Exception:
                # Last resort: write raw bytes
                img_path = output_dir / f"{image_id}.bin"
                img_path.write_bytes(data)
                return str(img_path)
    except Exception as exc:
        logger.debug("XObject image extraction failed: {}", exc)
    return None


def _bbox_to_dict(bbox: BBox) -> dict[str, float]:
    """Serialise a ``BBox`` to a plain dict.

    Args:
        bbox (BBox): The bounding box to serialise.

    Returns:
        dict[str, float]: A dictionary with keys "x0", "y0", "x1", "y1" representing the bounding box coordinates.
    """
    return {"x0": bbox.x0, "y0": bbox.y0, "x1": bbox.x1, "y1": bbox.y1}
