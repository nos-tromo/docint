"""OCR fallback for scanned pages."""

from __future__ import annotations

import base64
import re
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path

import pypdf
import pypdfium2
from loguru import logger
from openai import OpenAI as _OpenAI
from openai.types.chat import ChatCompletionContentPartParam, ChatCompletionMessageParam
from PIL import Image as PILImage

from docint.core.readers.documents.models import (
    BBox,
    LayoutBlock,
    OCRSpan,
    PageInfo,
    PageText,
)
from docint.utils.env_cfg import load_model_env, load_openai_env
from docint.utils.openai_cfg import OpenAIPipeline


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


class VisionOCREngine(OCREngine):
    """OCR engine that renders pages to images and uses a vision LLM.

    Uses ``pypdfium2`` to rasterise the page and the OpenAI-compatible
    vision endpoint with the ``ocr`` prompt to extract text.  This is
    the fallback when ``PypdfTextEngine`` returns nothing for pages
    flagged as needing OCR (i.e. scanned / screenshot PDFs).

    To avoid blocking the pipeline on slow endpoints the engine:

    * renders at a conservative DPI and caps the pixel dimensions,
    * encodes the image as JPEG (much smaller payload than PNG),
    * sends the request through a dedicated OpenAI client with a short
      timeout and limited retries,
    * caps the response length with ``max_tokens``,
    * on a timeout, automatically retries once at half resolution.
    """

    # Default limits used when no explicit values are supplied.
    _DEFAULT_TIMEOUT: float = 60.0
    _DEFAULT_MAX_RETRIES: int = 1
    _DEFAULT_MAX_IMAGE_DIM: int = 1024
    _EMPTY_RETRY_MAX_IMAGE_DIM: int = 1536
    _DEFAULT_RENDER_DPI: int = 120
    _DEFAULT_MAX_TOKENS: int = 4096
    _JPEG_QUALITY: int = 80
    _REFUSAL_MAX_CHARS: int = 280
    _REFUSAL_MAX_LINES: int = 4
    _REFUSAL_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"i(?:'| a)?m sorry[, ]+i (?:can(?:not|'t)|won't) assist"),
        re.compile(r"i (?:can(?:not|'t)|won't) assist with that"),
        re.compile(r"i (?:can(?:not|'t)|won't) help with that"),
        re.compile(r"i(?:'| a)?m unable to help with that"),
        re.compile(r"as an ai(?: language model)?[, ]+i (?:can(?:not|'t)|won't)"),
        re.compile(r"i cannot comply with that request"),
    )

    def __init__(
        self,
        file_path: str | Path,
        *,
        timeout: float | None = None,
        max_retries: int | None = None,
        max_image_dimension: int | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """Initialize the vision OCR engine.

        Args:
            file_path: Path to the source PDF.
            timeout: Per-request timeout in seconds for vision API calls.
                Defaults to ``60`` seconds.
            max_retries: Maximum retries for a single vision API call.
                Defaults to ``1``.
            max_image_dimension: Maximum pixel width or height for the
                rendered page image.  Larger renders are proportionally
                down-scaled before being sent to the API.  Defaults to ``1024``.
            max_tokens: Maximum number of tokens the vision LLM may generate
                per request.  Defaults to ``4096``.

        Raises:
            ImportError: If ``pypdfium2`` is not installed.
            RuntimeError: If the vision pipeline or OCR prompt cannot be loaded.
        """
        self._file_path = Path(file_path)
        self._pdf = pypdfium2.PdfDocument(str(self._file_path))
        self._pipeline = OpenAIPipeline()
        self._ocr_prompt = self._pipeline.load_prompt(kw="ocr")

        self._timeout = timeout if timeout is not None else self._DEFAULT_TIMEOUT
        self._max_retries = (
            max_retries if max_retries is not None else self._DEFAULT_MAX_RETRIES
        )
        self._max_image_dim = (
            max_image_dimension
            if max_image_dimension is not None
            else self._DEFAULT_MAX_IMAGE_DIM
        )
        self._max_tokens = (
            max_tokens if max_tokens is not None else self._DEFAULT_MAX_TOKENS
        )

        # Build a dedicated OpenAI client with the OCR-specific
        # timeout / retry settings so we don't block the pipeline for
        # the full global ``OPENAI_TIMEOUT × (1 + OPENAI_MAX_RETRIES)``
        # duration on large or slow pages.
        _oai = load_openai_env()
        self._vision_client = _OpenAI(
            api_key=_oai.api_key,
            base_url=_oai.api_base,
            timeout=self._timeout,
            max_retries=self._max_retries,
        )

    def ocr_page(
        self, page_index: int, *, file_path: Path | None = None
    ) -> list[OCRSpan]:
        """Render *page_index* to an image and extract text via vision LLM.

        The page is rasterised at 120 DPI, capped to
        ``max_image_dimension`` pixels, and encoded as JPEG.  If the
        first attempt times out the engine retries once with the image
        at half resolution.

        Args:
            page_index: Zero-based page number.
            file_path: Ignored (present for interface compatibility).

        Returns:
            List of ``OCRSpan`` items with the extracted text.
        """
        spans: list[OCRSpan] = []
        try:
            page = self._pdf[page_index]
            # Render at configured DPI (scale = DPI / 72).
            bitmap = page.render(scale=self._DEFAULT_RENDER_DPI / 72)
            base_image = bitmap.to_pil()
            pil_image = base_image.copy()

            # Down-scale if either dimension exceeds the configured cap.
            pil_image = self._cap_image(pil_image, self._max_image_dim, page_index)

            # First attempt at configured resolution.
            img_b64 = self._encode_jpeg(pil_image)
            logger.debug(
                "Vision OCR page {} — image {}×{}, payload ~{:.0f} KiB",
                page_index,
                pil_image.width,
                pil_image.height,
                len(img_b64) * 3 / 4 / 1024,
            )

            text: str | None = None
            try:
                text = self._call_vision_ocr(img_b64)
            except RuntimeError:
                # On timeout / error, retry once at half resolution.
                half_dim = max(self._max_image_dim // 2, 256)
                logger.info(
                    "Vision OCR retrying page {} at reduced resolution (max {}px)",
                    page_index,
                    half_dim,
                )
                pil_image = self._cap_image(pil_image, half_dim, page_index)
                img_b64 = self._encode_jpeg(pil_image)
                try:
                    text = self._call_vision_ocr(img_b64)
                except RuntimeError:
                    pass  # logged inside _call_vision_ocr

            # Empty-output recovery pass at a higher detail setting.
            if not (text and text.strip()):
                recovery_dim = max(self._max_image_dim, self._EMPTY_RETRY_MAX_IMAGE_DIM)
                recovery_image = self._cap_image(
                    base_image.copy(), recovery_dim, page_index
                )
                if (
                    recovery_image.size != pil_image.size
                    or max(recovery_image.width, recovery_image.height)
                    > self._max_image_dim
                ):
                    recovery_b64 = self._encode_jpeg(recovery_image)
                    logger.info(
                        "Vision OCR returned empty text for page {}; "
                        "retrying with higher-detail image {}×{}",
                        page_index,
                        recovery_image.width,
                        recovery_image.height,
                    )
                    recovery_prompt = (
                        f"{self._ocr_prompt}\n\n"
                        "Important: The text may be non-Latin (for example Arabic). "
                        "Return all visible text exactly as it appears. "
                        "Do not summarize or translate."
                    )
                    try:
                        text = self._call_vision_ocr(
                            recovery_b64,
                            prompt_override=recovery_prompt,
                        )
                        pil_image = recovery_image
                        img_b64 = recovery_b64
                    except RuntimeError:
                        pass  # logged inside _call_vision_ocr

            if text and text.strip():
                width = float(page.get_width())
                height = float(page.get_height())
                spans.append(
                    OCRSpan(
                        text=text.strip(),
                        bbox=BBox(x0=0.0, y0=0.0, x1=width, y1=height),
                        confidence=0.7,
                        source="vision_ocr",
                    )
                )
                logger.info(
                    "Vision OCR produced {} chars for page {}",
                    len(text.strip()),
                    page_index,
                )
            else:
                logger.warning(
                    "Vision OCR returned empty text for page {} "
                    "(image {}×{}, payload ~{:.0f} KiB)",
                    page_index,
                    pil_image.width,
                    pil_image.height,
                    len(img_b64) * 3 / 4 / 1024,
                )
        except Exception as exc:
            logger.warning("Vision OCR failed for page {}: {}", page_index, exc)
        return spans

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cap_image(
        pil_image: PILImage.Image,
        max_dim: int,
        page_index: int,
    ) -> PILImage.Image:
        """Down-scale *pil_image* so neither axis exceeds *max_dim*.

        Args:
            pil_image: The PIL image to be down-scaled.
            max_dim: The maximum allowed dimension for the image.
            page_index: The index of the page being processed.

        Returns:
            The down-scaled PIL image.
        """
        cur_max = max(pil_image.width, pil_image.height)
        if cur_max > max_dim:
            ratio = max_dim / cur_max
            new_w = max(int(pil_image.width * ratio), 1)
            new_h = max(int(pil_image.height * ratio), 1)
            pil_image = pil_image.resize((new_w, new_h))
            logger.debug(
                "Resized OCR image for page {} to {}×{}",
                page_index,
                new_w,
                new_h,
            )
        return pil_image

    @classmethod
    def _encode_jpeg(cls, pil_image: PILImage.Image) -> str:
        """Encode a PIL image as JPEG and return its base64 representation.

        Args:
            pil_image: The PIL image to be encoded.

        Returns:
            Base64-encoded string of the JPEG image.
        """
        buf = BytesIO()
        # Convert RGBA → RGB before JPEG encoding.
        if pil_image.mode in ("RGBA", "P"):
            pil_image = pil_image.convert("RGB")
        pil_image.save(buf, format="JPEG", quality=cls._JPEG_QUALITY)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _call_vision_ocr(
        self, img_b64: str, *, prompt_override: str | None = None
    ) -> str:
        """Send the base64-encoded image to the vision LLM for OCR.

        Uses the dedicated ``_vision_client`` which has the shorter
        OCR-specific timeout and retry count, preventing the pipeline from
        blocking for ``OPENAI_TIMEOUT × (1 + OPENAI_MAX_RETRIES)`` on
        unresponsive endpoints.

        Args:
            img_b64: Base64-encoded JPEG image data.
            prompt_override: Optional prompt text for this specific OCR call.

        Returns:
            Extracted text string (may be empty).

        Raises:
            RuntimeError: If the vision inference fails.
        """
        vision_model_id = load_model_env().vision_model_file.removesuffix(".gguf")

        prompt_text = prompt_override or self._ocr_prompt
        content_parts: list[ChatCompletionContentPartParam] = [
            {"type": "text", "text": prompt_text},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
            },
        ]
        messages: list[ChatCompletionMessageParam] = [
            {"role": "user", "content": content_parts}
        ]

        try:
            response = self._vision_client.chat.completions.create(
                model=vision_model_id,
                messages=messages,
                max_tokens=self._max_tokens,
                seed=self._pipeline.seed,
                temperature=self._pipeline.temperature,
                top_p=self._pipeline.top_p,
            )
            text = response.choices[0].message.content or ""
            if self._looks_like_refusal(text):
                logger.warning(
                    "Vision OCR returned refusal-style output; treating as empty text"
                )
                return ""
            return text
        except Exception as e:
            logger.error("Error during vision OCR inference: {}", e)
            raise RuntimeError(f"Vision OCR inference failed: {e}")

    @classmethod
    def _looks_like_refusal(cls, text: str) -> bool:
        """Return whether *text* looks like a safety refusal/disclaimer.

        The check is intentionally conservative and only flags short,
        single-message responses that match common refusal phrases.

        Args:
            text: OCR model output.

        Returns:
            True when the text appears to be a refusal message.
        """
        normalized = " ".join(text.strip().lower().split())
        if not normalized:
            return False
        if len(normalized) > cls._REFUSAL_MAX_CHARS:
            return False
        non_empty_lines = [line for line in text.splitlines() if line.strip()]
        if len(non_empty_lines) > cls._REFUSAL_MAX_LINES:
            return False
        return any(pattern.search(normalized) for pattern in cls._REFUSAL_PATTERNS)

    def close(self) -> None:
        """Release the underlying ``pypdfium2`` document handle."""
        try:
            self._pdf.close()
        except Exception:
            pass


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
    *,
    vision_engine: OCREngine | None = None,
) -> dict[int, PageText]:
    """Extract text for all pages, applying OCR fallback where needed.

    When *vision_engine* is provided and the standard ``pypdf`` extraction
    yields no text for a page that ``needs_ocr``, the vision engine is
    tried as a secondary fallback.

    Args:
        file_path: Path to the PDF.
        pages: Page triage results.
        layout: Layout blocks per page.
        vision_engine: Optional secondary OCR engine (e.g. ``VisionOCREngine``)
            used when the primary pypdf extraction returns nothing.

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

            # Vision OCR fallback when standard extraction yields nothing.
            if not ocr_spans and vision_engine is not None:
                try:
                    logger.info(
                        "Attempting vision OCR fallback for page {}",
                        page_info.page_index,
                    )
                    ocr_spans = vision_engine.ocr_page(
                        page_info.page_index, file_path=file_path
                    )
                except Exception as exc:
                    logger.warning(
                        "Vision OCR fallback failed for page {}: {}",
                        page_info.page_index,
                        exc,
                    )

        blocks = layout.get(page_info.page_index, [])
        result[page_info.page_index] = build_page_text(page_info, blocks, ocr_spans)

    return result
