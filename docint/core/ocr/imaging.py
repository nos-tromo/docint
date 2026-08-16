"""Image bounding and encoding for the OCR engine.

Every OCR call sends pixels to the same endpoint, so how an image is bounded
and encoded lives in one place: a difference here shows up as a model behaving
differently for reasons that have nothing to do with the prompt.
"""

from __future__ import annotations

import base64
from io import BytesIO

from loguru import logger
from PIL import Image as PILImage

JPEG_QUALITY: int = 80


def cap_image(pil_image: PILImage.Image, max_dim: int, *, context: str = "") -> PILImage.Image:
    """Down-scale *pil_image* so neither axis exceeds *max_dim*.

    Args:
        pil_image (PILImage.Image): The image to bound.
        max_dim (int): Maximum allowed pixel dimension.
        context (str): Short description used in the debug log (e.g. ``"page 3"``).

    Returns:
        PILImage.Image: The original image, or a proportionally scaled copy.
    """
    cur_max = max(pil_image.width, pil_image.height)
    if cur_max > max_dim:
        ratio = max_dim / cur_max
        new_w = max(int(pil_image.width * ratio), 1)
        new_h = max(int(pil_image.height * ratio), 1)
        pil_image = pil_image.resize((new_w, new_h))
        logger.debug("Resized OCR image {} to {}x{}", context, new_w, new_h)
    return pil_image


def encode_jpeg(pil_image: PILImage.Image, quality: int = JPEG_QUALITY) -> str:
    """Encode a PIL image as JPEG and return its base64 representation.

    Args:
        pil_image (PILImage.Image): The image to encode.
        quality (int): JPEG quality.

    Returns:
        str: Base64-encoded JPEG bytes (no data-URL prefix).
    """
    buf = BytesIO()
    # Convert RGBA → RGB before JPEG encoding.
    if pil_image.mode in ("RGBA", "P"):
        pil_image = pil_image.convert("RGB")
    pil_image.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def image_from_bytes(data: bytes) -> PILImage.Image:
    """Decode image bytes into a PIL image in a mode JPEG can carry.

    Args:
        data (bytes): Raw image bytes of any Pillow-readable format.

    Returns:
        PILImage.Image: The decoded image.
    """
    image = PILImage.open(BytesIO(data))
    image.load()
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    return image
