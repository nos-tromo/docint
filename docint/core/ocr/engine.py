"""The one place docint reads text out of pixels.

Whatever the source — a scanned PDF page, a table's region, a photographed
letter, a video keyframe — reading it is one task and goes through one engine:
one client, one prompt per task, one failure policy. What differs between
models lives in :mod:`docint.core.ocr.families`; what differs between callers
is only which entry point they use and what they do with the blocks.

Failure is soft by design. A call the endpoint answered badly costs that one
page (it is usually fine again a second later); a call nothing answered at all
counts toward a small budget, after which the engine stops calling for the
rest of the document rather than spend a full timeout per page. Callers keep
whatever they had when a read returns nothing.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

import pypdfium2
from loguru import logger
from openai import APIConnectionError, APIStatusError, APITimeoutError
from openai import OpenAI as _OpenAI
from openai.types.chat import ChatCompletionContentPartParam, ChatCompletionMessageParam
from PIL import Image as PILImage

from docint.core.ocr.families import (
    GenericFamily,
    OcrBlock,
    OcrBox,
    OcrFrame,
    OcrLimits,
    OcrModelFamily,
    OcrTask,
    family_for,
)
from docint.core.ocr.imaging import encode_jpeg, image_from_bytes
from docint.utils.env_cfg import load_model_env, load_ocr_client_env, load_openai_env
from docint.utils.llm_sanitize import looks_like_no_image_refusal, squeeze_char_runs, strip_reasoning
from docint.utils.openai_cfg import OpenAIPipeline


class OcrError(RuntimeError):
    """An OCR call did not produce usable text.

    Subclasses ``RuntimeError`` so callers that catch the broad type keep
    working; the two subclasses below carry the distinction that decides
    whether the whole document should be given up on.
    """


class OcrUnreachable(OcrError):
    """Nothing came back — a timeout, or the endpoint could not be reached.

    This is the failure the per-document budget exists for: every further call
    would spend the full timeout for the same nothing.
    """


class OcrRejected(OcrError):
    """The endpoint answered, with an error status.

    Costs one call, not the document. Measured on the dev stack, these come
    back in 0.5-1.0s (against 68-117s for a successful page) in bursts that
    recover within seconds, so the next page is usually fine.
    """


@dataclass
class OcrStats:
    """What the OCR engine did across one document.

    Attributes:
        pages_read: Calls that produced blocks.
        pages_failed: Calls that reached the engine but produced nothing.
        pages_skipped: Calls not attempted because the budget had given up.
    """

    pages_read: int = 0
    pages_failed: int = 0
    pages_skipped: int = 0


class DocumentOcrEngine:
    """Reads pixels through the configured OCR model.

    Args:
        file_path (str | Path | None): PDF to read pages and regions from.
            Omit for an engine that only reads images.
        timeout (float | None): Per-request timeout in seconds.
        max_retries (int | None): SDK-level retries per request.
        max_image_dimension (int | None): Longest side sent by the generic family.
        max_pixels (int | None): Total-pixel budget of the dots family. Must not
            exceed the server's own cap: that model returns boxes in the frame
            of the image it actually saw, so a larger image comes back with
            coordinates that cannot be mapped home.
        max_tokens (int | None): Generation budget per call.
    """

    _DEFAULT_TIMEOUT: float = 60.0
    _DEFAULT_MAX_RETRIES: int = 1
    _DEFAULT_MAX_IMAGE_DIM: int = 1024
    _DEFAULT_MAX_PIXELS: int = 2_000_000
    _DEFAULT_MAX_TOKENS: int = 4096
    # Consecutive calls nothing answered at all before the engine stops
    # calling the endpoint for the rest of the document.
    _MAX_CONSECUTIVE_FAILURES: int = 3
    # Pause before a retry. Upstream rejections arrive in bursts lasting a few
    # seconds; retrying immediately lands inside the same burst, which is what
    # made a transient blip cost whole pages.
    _RETRY_BACKOFF_SECONDS: float = 2.0
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
        file_path: str | Path | None = None,
        *,
        timeout: float | None = None,
        max_retries: int | None = None,
        max_image_dimension: int | None = None,
        max_pixels: int | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> None:
        """Build the client and pick the model's family."""
        self._file_path = Path(file_path) if file_path is not None else None
        self._pipeline = OpenAIPipeline()
        openai_cfg = load_openai_env()
        cfg = load_ocr_client_env(
            default_api_base=openai_cfg.api_base or "",
            default_api_key=openai_cfg.api_key,
            default_timeout=timeout if timeout is not None else self._DEFAULT_TIMEOUT,
        )
        # No document model configured: read with the general vision model, as
        # this pipeline always has.
        self.model = model or cfg.model or load_model_env().vision_model
        self.api_base = cfg.api_base or openai_cfg.api_base or ""
        self.family: OcrModelFamily = family_for(self.model, self._pipeline)
        self.limits = OcrLimits(
            max_pixels=max_pixels if max_pixels is not None else self._DEFAULT_MAX_PIXELS,
            max_dim=max_image_dimension if max_image_dimension is not None else self._DEFAULT_MAX_IMAGE_DIM,
            max_tokens=max_tokens if max_tokens is not None else self._DEFAULT_MAX_TOKENS,
        )
        self._timeout = timeout if timeout is not None else cfg.timeout
        self._pdf = pypdfium2.PdfDocument(str(self._file_path)) if self._file_path is not None else None
        self._client = _OpenAI(
            api_key=cfg.api_key,
            base_url=self.api_base,
            timeout=self._timeout,
            max_retries=max_retries if max_retries is not None else self._DEFAULT_MAX_RETRIES,
        )
        self.stats = OcrStats()
        self._consecutive_failures = 0
        self.disabled = False

    @property
    def reads_layout(self) -> bool:
        """Whether the configured model returns structure, not just text."""
        return self.family.name == "dots"

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    def read_page(self, page_index: int) -> list[OcrBlock]:
        """Read a whole PDF page.

        Args:
            page_index (int): Zero-based page number.

        Returns:
            list[OcrBlock]: Blocks in reading order — one text block for a
                model without layout — or empty when nothing was read.
        """
        page = self._page(page_index)
        width = float(page.get_width())
        height = float(page.get_height())
        base = self._render(page, width, height)
        frame = OcrFrame(width=width, height=height)
        return self._read(base, OcrTask.PAGE, frame, context=f"page {page_index}")

    def read_region(self, page_index: int, bbox: OcrBox) -> list[OcrBlock]:
        """Read one region of a PDF page.

        Args:
            page_index (int): Zero-based page number.
            bbox (OcrBox): The region, in page points (bottom-left origin).

        Returns:
            list[OcrBlock]: Blocks whose boxes are in page coordinates.
        """
        page = self._page(page_index)
        width = float(page.get_width())
        height = float(page.get_height())
        pad = 6.0
        crop = (
            max(0.0, bbox.x0 - pad),
            max(0.0, bbox.y0 - pad),
            max(0.0, width - bbox.x1 - pad),
            max(0.0, height - bbox.y1 - pad),
        )
        region_w = max(width - crop[0] - crop[2], 1.0)
        region_h = max(height - crop[1] - crop[3], 1.0)
        base = self._render(page, region_w, region_h, crop=crop)
        frame = OcrFrame(width=region_w, height=region_h, offset_x=crop[0], offset_y=crop[1])
        return self._read(base, OcrTask.TABLE, frame, context=f"region on page {page_index}")

    def read_image(self, image: PILImage.Image | bytes, *, context: str = "image") -> list[OcrBlock]:
        """Read an image that is not a PDF page.

        Args:
            image (PILImage.Image | bytes): The image, decoded or raw.
            context (str): Description used in logs.

        Returns:
            list[OcrBlock]: Blocks whose boxes are in image pixels
                (bottom-left origin) — an image has no page geometry.
        """
        pil = image_from_bytes(image) if isinstance(image, bytes) else image
        frame = OcrFrame(width=float(pil.width), height=float(pil.height))
        return self._read(pil, OcrTask.IMAGE, frame, context=context)

    def close(self) -> None:
        """Release the PDF handle, if this engine opened one."""
        if self._pdf is None:
            return
        try:
            self._pdf.close()
        except Exception as exc:  # pragma: no cover - best effort
            logger.debug("Closing the OCR document failed: {}", exc)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _page(self, page_index: int) -> pypdfium2.PdfPage:
        """Return a page handle, or fail loudly if this engine has no document."""
        if self._pdf is None:
            raise RuntimeError("This OCR engine was built without a document; use read_image().")
        return self._pdf[page_index]

    def _render(
        self,
        page: pypdfium2.PdfPage,
        width: float,
        height: float,
        *,
        crop: tuple[float, float, float, float] | None = None,
    ) -> PILImage.Image:
        """Rasterise a page (or a crop of it) at the family's resolution.

        Args:
            page (pypdfium2.PdfPage): The page.
            width (float): Width of the area being rendered, in points.
            height (float): Its height, in points.
            crop (tuple | None): Amount to cut off each side, in points.

        Returns:
            PILImage.Image: The rendered image.
        """
        target = self.family.target_pixels(width, height, self.limits)
        if target is not None:
            scale = max(target[0] / max(width, 1.0), target[1] / max(height, 1.0))
        else:
            scale = getattr(self.family, "render_dpi", 120) / 72
        kwargs = {"crop": crop} if crop is not None else {}
        return page.render(scale=scale, **kwargs).to_pil()

    def _read(self, base: PILImage.Image, task: OcrTask, frame: OcrFrame, *, context: str) -> list[OcrBlock]:
        """Send an image, retry as the family allows, and parse the answer.

        Args:
            base (PILImage.Image): The rendered image before bounding.
            task (OcrTask): What is being read.
            frame (OcrFrame): Coordinate frame for the answer.
            context (str): Description used in logs.

        Returns:
            list[OcrBlock]: Parsed blocks, or empty when nothing was read.
        """
        if self.disabled:
            self.stats.pages_skipped += 1
            logger.debug("OCR disabled for this document; skipping {}", context)
            return []

        prompt = self.family.prompt(task)
        sent = self.family.prepare(base, self.limits, context=context)
        answer: str | None = None
        responded = False
        reachable = False
        try:
            answer = self._call(sent, prompt, context=context)
            responded = True
        except OcrError as first_error:
            reachable = isinstance(first_error, OcrRejected)
            time.sleep(self._RETRY_BACKOFF_SECONDS)
            reduced = self.family.degrade(sent, self.limits, context=context)
            if reduced is not None:
                logger.info("OCR retrying {} at reduced resolution ({}x{})", context, reduced.width, reduced.height)
                sent = reduced
            try:
                answer = self._call(sent, prompt, context=context)
                responded = True
            except OcrError as retry_error:
                reachable = reachable or isinstance(retry_error, OcrRejected)

        if not responded:
            self._note_failure(context, reachable=reachable)
            return []
        self._consecutive_failures = 0

        if not (answer and answer.strip()):
            escalation = self.family.escalate(base, sent, self.limits, context=context)
            if escalation is not None:
                bigger, bigger_prompt = escalation
                logger.info("OCR answer for {} was empty; retrying at {}x{}", context, bigger.width, bigger.height)
                sent = bigger
                try:
                    answer = self._call(bigger, bigger_prompt, context=context)
                except OcrError:
                    pass  # logged in _call

        blocks: list[OcrBlock] = []
        if answer and answer.strip():
            blocks = self.family.parse(answer, task, image_size=sent.size, frame=frame)
        if not blocks:
            self.stats.pages_failed += 1
            logger.warning("OCR produced nothing usable for {} (image {}x{})", context, sent.width, sent.height)
            return []
        self.stats.pages_read += 1
        logger.info("OCR read {} blocks for {}", len(blocks), context)
        return blocks

    def _call(self, image: PILImage.Image, prompt: str, *, context: str) -> str:
        """Send one request and return the cleaned answer.

        Args:
            image (PILImage.Image): The image to send.
            prompt (str): The instruction.
            context (str): Description used in logs.

        Returns:
            str: The answer, possibly empty.

        Raises:
            OcrRejected: The endpoint answered with an error status.
            OcrUnreachable: Nothing answered.
        """
        img_b64 = encode_jpeg(image)
        # Image first, then the instruction: the order the document models'
        # own clients use.
        content_parts: list[ChatCompletionContentPartParam] = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            {"type": "text", "text": prompt},
        ]
        messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": content_parts}]
        request_kwargs: dict[str, object] = {}
        if self._pipeline.reasoning_effort is not None:
            request_kwargs["reasoning_effort"] = self._pipeline.reasoning_effort
        try:
            response = self._client.chat.completions.create(  # type: ignore[call-overload]
                model=self.model,
                messages=messages,
                max_tokens=self.limits.max_tokens,
                seed=self._pipeline.seed,
                temperature=self._pipeline.temperature,
                top_p=self._pipeline.top_p,
                **request_kwargs,
            )
        except APIStatusError as exc:
            logger.error("OCR endpoint rejected {}: {}", context, exc)
            raise OcrRejected(f"OCR inference failed: {exc}") from exc
        except (APITimeoutError, APIConnectionError) as exc:
            logger.error("OCR endpoint unreachable for {}: {}", context, exc)
            raise OcrUnreachable(f"OCR inference failed: {exc}") from exc
        except Exception as exc:
            # Unclassifiable: treat as unreachable so the per-document budget
            # still protects against a failure mode we have not seen yet.
            logger.error("OCR call failed for {}: {}", context, exc)
            raise OcrUnreachable(f"OCR inference failed: {exc}") from exc

        raw = response.choices[0].message.content or ""
        text, captured = strip_reasoning(raw)
        if captured:
            logger.debug("Stripped {} chars of reasoning from the OCR response", len(captured))
        if looks_like_no_image_refusal(text):
            logger.warning("OCR reported no image despite one being attached; treating as empty")
            return ""
        if self._looks_like_refusal(text):
            logger.warning("OCR returned refusal-style output; treating as empty")
            return ""
        # Degenerate-repetition guard: a form's dotted fill-in line can lock a
        # model into repeating the fill character until ``max_tokens`` —
        # observed live as 65392 dots on one page, stored and embedded as real
        # content.
        squeezed = squeeze_char_runs(text)
        if len(squeezed) < len(text) - 1000:
            logger.warning("OCR output was {} chars of repeated filler; squeezed to {} chars", len(text), len(squeezed))
        return squeezed

    def _note_failure(self, context: str, *, reachable: bool) -> None:
        """Record a call that produced nothing, and give up when nothing answers.

        Only a call nothing answered for counts toward the budget. That budget
        exists to stop spending a full timeout each time on an endpoint that
        never answers; an endpoint returning an error status costs about a
        second and typically recovers within a few, so it must cost its own
        page and no more.

        Args:
            context (str): Description used in logs.
            reachable (bool): Whether the endpoint answered at all.
        """
        self.stats.pages_failed += 1
        if reachable:
            # Do not reset the consecutive counter: an unreachable endpoint
            # interleaved with rejections is still unreachable.
            logger.warning("OCR endpoint rejected {}; skipping it and continuing", context)
            return
        self._consecutive_failures += 1
        logger.warning(
            "OCR got no response for {} ({}/{} consecutive failures)",
            context,
            self._consecutive_failures,
            self._MAX_CONSECUTIVE_FAILURES,
        )
        if self._consecutive_failures >= self._MAX_CONSECUTIVE_FAILURES:
            self.disabled = True
            logger.error(
                "OCR endpoint unresponsive after {} consecutive calls; disabling OCR for the rest of this document",
                self._consecutive_failures,
            )

    @classmethod
    def _looks_like_refusal(cls, text: str) -> bool:
        """Whether *text* looks like a safety refusal rather than a reading.

        Conservative on purpose: only short, single-message answers that match
        a known refusal phrase, so a page that genuinely says "I cannot help"
        is not discarded.

        Args:
            text (str): The model's answer.

        Returns:
            bool: True when the answer reads as a refusal.
        """
        stripped = (text or "").strip()
        if not stripped or len(stripped) > cls._REFUSAL_MAX_CHARS:
            return False
        if len(stripped.splitlines()) > cls._REFUSAL_MAX_LINES:
            return False
        lowered = stripped.lower()
        return any(pattern.search(lowered) for pattern in cls._REFUSAL_PATTERNS)


def build_engine(*args: object, **kwargs: object) -> DocumentOcrEngine | None:
    """Build an OCR engine, or ``None`` when one cannot be built.

    Callers treat OCR as optional, so the "no endpoint configured / client
    could not be constructed" case is not an error at the call site.

    Returns:
        DocumentOcrEngine | None: The engine, or ``None``.
    """
    try:
        return DocumentOcrEngine(*args, **kwargs)  # type: ignore[arg-type]
    except Exception as exc:
        logger.debug("OCR engine not available: {}", exc)
        return None


# The generic family is what an unconfigured stack uses; re-exported so a
# caller can tell "no document model" without importing families.
__all__ = [
    "DocumentOcrEngine",
    "GenericFamily",
    "OcrBlock",
    "OcrBox",
    "OcrError",
    "OcrRejected",
    "OcrStats",
    "OcrTask",
    "OcrUnreachable",
    "build_engine",
]
