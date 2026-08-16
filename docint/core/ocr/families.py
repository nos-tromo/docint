"""What each OCR model is asked, and how its answer becomes blocks.

One engine talks to whatever model is configured; a *family* holds everything
that differs between models — the prompt for a task, the resolution the model
wants, and how to read its answer. Two exist:

* ``dots`` (``dots.ocr`` / ``dots.mocr``): answers with the page's layout —
  one element per block with a bbox, a category and its text (tables as HTML),
  in reading order. This is the family that lets a scanned page produce the
  same block structure as a digital one.
* ``generic``: any chat VLM (GLM-OCR included). It reads text, not layout, so
  a page comes back as a single text block; a table region comes back as HTML.

A model is placed in a family by its id, so switching ``OCR_MODEL`` switches
the contract without a code change.
"""

from __future__ import annotations

import json
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

from loguru import logger
from PIL import Image as PILImage
from typing_extensions import override

from docint.core.ocr.html_table import grid_to_text, parse_html_table
from docint.core.ocr.imaging import cap_image
from docint.utils.openai_cfg import OpenAIPipeline


class OcrCategory(StrEnum):
    """What a document model says a block *is*.

    Deliberately the OCR package's own vocabulary rather than the PDF
    pipeline's ``BlockType``: this package must not depend on the document
    reader (which consumes it), and an image caller has no page blocks at all.
    The document reader maps these onto its own types.
    """

    TITLE = "title"
    SECTION_HEADER = "section_header"
    TEXT = "text"
    LIST_ITEM = "list_item"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    FORMULA = "formula"
    TABLE = "table"
    PICTURE = "picture"
    PAGE_HEADER = "page_header"
    PAGE_FOOTER = "page_footer"


@dataclass(frozen=True)
class OcrBox:
    """A rectangle in the caller's frame, bottom-left origin."""

    x0: float
    y0: float
    x1: float
    y1: float


class OcrTask(StrEnum):
    """What the caller wants read out of the pixels."""

    #: A whole page: everything on it, with structure where the model has it.
    PAGE = "page"
    #: One table's region: its cells.
    TABLE = "table"
    #: An image file, keyframe or figure: the text it contains.
    IMAGE = "image"


@dataclass(frozen=True)
class OcrBlock:
    """One block of a page as the OCR model reads it.

    Attributes:
        category: What the model says it is.
        bbox: Where it sits, in the caller's frame (page points, bottom-left
            origin, for PDF pages and regions; pixels for a bare image).
        text: Its text — for a table, rendered row by row.
        cells: A table's grid, when the model gave one.
    """

    category: OcrCategory
    bbox: OcrBox
    text: str = ""
    cells: list[list[str]] | None = None


@dataclass(frozen=True)
class OcrFrame:
    """The coordinate frame the caller wants boxes in.

    Attributes:
        width: Frame width (points for a page/region, pixels for an image).
        height: Frame height.
        offset_x: Added to every x after scaling — a region's left edge in page space.
        offset_y: Added to every y after scaling — a region's bottom edge in page space.
    """

    width: float
    height: float
    offset_x: float = 0.0
    offset_y: float = 0.0


@dataclass(frozen=True)
class OcrLimits:
    """Bounds on what may be sent to the model.

    Attributes:
        max_pixels: Total-pixel budget (the dots family renders to it exactly).
        max_dim: Longest-side budget (the generic family caps to it).
        max_tokens: Generation budget per call.
    """

    max_pixels: int = 2_007_040
    max_dim: int = 1024
    # A region is a fraction of a page, so the same longest side buys it far
    # more detail — and a table read at page resolution loses its digits.
    region_max_dim: int = 1536
    max_tokens: int = 4096


class OcrModelFamily(ABC):
    """The contract one kind of OCR model speaks."""

    name: str

    @abstractmethod
    def prompt(self, task: OcrTask) -> str:
        """Return the instruction for *task*."""

    @abstractmethod
    def parse(self, answer: str, task: OcrTask, *, image_size: tuple[int, int], frame: OcrFrame) -> list[OcrBlock]:
        """Turn the model's answer into blocks in *frame*'s coordinates."""

    def target_pixels(self, width: float, height: float, limits: OcrLimits) -> tuple[int, int] | None:
        """Pixel size to render at, when the family wants an exact one.

        Args:
            width (float): Source width (points).
            height (float): Source height (points).
            limits (OcrLimits): The active bounds.

        Returns:
            tuple[int, int] | None: ``(width, height)`` in pixels, or ``None``
                to let the engine render at its default DPI and then bound.
        """
        _ = (width, height, limits)
        return None

    def prepare(self, image: PILImage.Image, limits: OcrLimits, *, context: str = "") -> PILImage.Image:
        """Bound an image to what this family sends.

        Args:
            image (PILImage.Image): The rendered image.
            limits (OcrLimits): The active bounds.
            context (str): Description for the debug log.

        Returns:
            PILImage.Image: The image to send.
        """
        return cap_image(image, limits.max_dim, context=context)

    def degrade(self, image: PILImage.Image, limits: OcrLimits, *, context: str = "") -> PILImage.Image | None:
        """A smaller image to retry with after a failure, if that helps.

        Args:
            image (PILImage.Image): The image that failed.
            limits (OcrLimits): The active bounds.
            context (str): Description for the debug log.

        Returns:
            PILImage.Image | None: The reduced image, or ``None`` to retry as-is.
        """
        _ = (image, limits, context)
        return None

    def escalate(
        self, base_image: PILImage.Image, sent: PILImage.Image, limits: OcrLimits, *, context: str = ""
    ) -> tuple[PILImage.Image, str] | None:
        """A larger image (and prompt) to retry with when the answer was empty.

        Args:
            base_image (PILImage.Image): The unbounded render.
            sent (PILImage.Image): What was sent and came back empty.
            limits (OcrLimits): The active bounds.
            context (str): Description for the debug log.

        Returns:
            tuple[PILImage.Image, str] | None: Image and prompt, or ``None``.
        """
        _ = (base_image, sent, limits, context)
        return None


# ---------------------------------------------------------------------------
# dots.ocr / dots.mocr
# ---------------------------------------------------------------------------

# The canonical layout prompt from the model's own client
# (dots_mocr/utils/prompts.py, prompt_layout_all_en). It is the string the
# model was trained on — protocol, not prose, so it is not localized and must
# not be paraphrased.
DOTS_LAYOUT_PROMPT = (
    "Please output the layout information from the PDF image, including each layout element's "
    "bbox, its category, and the corresponding text content within the bbox.\n\n"
    "1. Bbox format: [x1, y1, x2, y2]\n\n"
    "2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', "
    "'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', "
    "'Title'].\n\n"
    "3. Text Extraction & Formatting Rules:\n"
    "    - Picture: For the 'Picture' category, the text field should be omitted.\n"
    "    - Formula: Format its text as LaTeX.\n"
    "    - Table: Format its text as HTML.\n"
    "    - All Others (Text, Title, etc.): Format their text as Markdown.\n\n"
    "4. Constraints:\n"
    "    - The output text must be the original text from the image, with no translation.\n"
    "    - All layout elements must be sorted according to human reading order.\n\n"
    "5. Final Output: The entire output must be a single JSON object."
)

# The model resizes what it is given: sides rounded to this many pixels, total
# pixels bounded. Rendering to exactly such a size makes that resize the
# identity, so the boxes it returns are in the frame we sent.
_PATCH_FACTOR = 28
_MIN_PIXELS = 3136

_DOTS_CATEGORIES: dict[str, OcrCategory] = {
    "Title": OcrCategory.TITLE,
    "Section-header": OcrCategory.SECTION_HEADER,
    "Text": OcrCategory.TEXT,
    "List-item": OcrCategory.LIST_ITEM,
    "Caption": OcrCategory.CAPTION,
    "Footnote": OcrCategory.FOOTNOTE,
    "Formula": OcrCategory.FORMULA,
    "Table": OcrCategory.TABLE,
    "Picture": OcrCategory.PICTURE,
    "Page-header": OcrCategory.PAGE_HEADER,
    "Page-footer": OcrCategory.PAGE_FOOTER,
}


def aligned_size(width: float, height: float, *, max_pixels: int) -> tuple[int, int]:
    """Pixel size within *max_pixels* whose sides are multiples of the patch factor.

    Args:
        width (float): Source width.
        height (float): Source height.
        max_pixels (int): Total-pixel budget.

    Returns:
        tuple[int, int]: ``(width, height)`` in pixels.
    """
    aspect = max(width, 1.0) / max(height, 1.0)
    target_h = math.sqrt(max(max_pixels, _MIN_PIXELS) / aspect)
    target_w = target_h * aspect
    w_bar = max(_PATCH_FACTOR, int(target_w // _PATCH_FACTOR) * _PATCH_FACTOR)
    h_bar = max(_PATCH_FACTOR, int(target_h // _PATCH_FACTOR) * _PATCH_FACTOR)
    while w_bar * h_bar > max_pixels and (w_bar > _PATCH_FACTOR or h_bar > _PATCH_FACTOR):
        if w_bar >= h_bar:
            w_bar -= _PATCH_FACTOR
        else:
            h_bar -= _PATCH_FACTOR
    return w_bar, h_bar


def clean_json_array(content: str) -> str:
    """Strip fences and prose around a JSON array; repair a truncated tail.

    A page dense enough to exhaust the generation budget comes back cut off
    mid-element. Everything that did arrive is still usable, so the tail is
    trimmed to the last complete element rather than thrown away.

    Args:
        content (str): The model's raw answer.

    Returns:
        str: Something ``json.loads`` can read, or ``""``.
    """
    text = (content or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("[")
    if start < 0:
        return ""
    text = text[start:]
    end = text.rfind("]")
    if end >= 0:
        candidate = text[: end + 1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
    last = text.rfind("}")
    while last > 0:
        candidate = text[: last + 1].rstrip().rstrip(",") + "]"
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            last = text.rfind("}", 0, last)
    return ""


class DotsFamily(OcrModelFamily):
    """A model that answers with the page's layout as JSON."""

    name = "dots"

    @override
    def prompt(self, task: OcrTask) -> str:
        """The layout task, for every input — a table crop is just a small page."""
        _ = task
        return DOTS_LAYOUT_PROMPT

    @override
    def target_pixels(self, width: float, height: float, limits: OcrLimits) -> tuple[int, int]:
        """Render to the exact size the model would resize to."""
        return aligned_size(width, height, max_pixels=limits.max_pixels)

    @override
    def prepare(self, image: PILImage.Image, limits: OcrLimits, *, context: str = "") -> PILImage.Image:
        """Bring any image onto the model's own pixel grid."""
        target = aligned_size(image.width, image.height, max_pixels=limits.max_pixels)
        if image.size != target:
            logger.debug("Sizing OCR image {} to {}x{}", context, target[0], target[1])
            image = image.resize(target)
        return image

    @override
    def parse(self, answer: str, task: OcrTask, *, image_size: tuple[int, int], frame: OcrFrame) -> list[OcrBlock]:
        """Map the layout JSON onto blocks in the caller's frame.

        Boxes arrive as pixels with a top-left origin and leave in the frame's
        units with a bottom-left origin — the convention the rest of the
        pipeline uses.

        Args:
            answer (str): The model's raw answer.
            task (OcrTask): What was asked (a table crop keeps only its table).
            image_size (tuple[int, int]): Size of the image the model saw.
            frame (OcrFrame): Target coordinate frame.

        Returns:
            list[OcrBlock]: Blocks in reading order.
        """
        cleaned = clean_json_array(answer)
        if not cleaned:
            return []
        try:
            elements = json.loads(cleaned)
        except json.JSONDecodeError:
            return []
        if not isinstance(elements, list):
            return []

        img_w, img_h = image_size
        if img_w <= 0 or img_h <= 0:
            return []
        sx = frame.width / img_w
        sy = frame.height / img_h

        blocks: list[OcrBlock] = []
        for elem in elements:
            if not isinstance(elem, dict):
                continue
            raw = elem.get("bbox")
            if not isinstance(raw, list) or len(raw) != 4:
                continue
            try:
                x1, y1, x2, y2 = (float(v) for v in raw)
            except (TypeError, ValueError):
                continue
            raw_category = str(elem.get("category") or "Text")
            text = str(elem.get("text") or "").strip()
            category = _DOTS_CATEGORIES.get(raw_category, OcrCategory.TEXT)
            bbox = OcrBox(
                x0=min(x1, x2) * sx + frame.offset_x,
                y0=frame.height - max(y1, y2) * sy + frame.offset_y,
                x1=max(x1, x2) * sx + frame.offset_x,
                y1=frame.height - min(y1, y2) * sy + frame.offset_y,
            )
            cells: list[list[str]] | None = None
            if category is OcrCategory.TABLE:
                grid = parse_html_table(text) if text else None
                if grid:
                    cells = grid
                    text = grid_to_text(grid)
            elif category is OcrCategory.PICTURE:
                text = ""
            blocks.append(OcrBlock(category=category, bbox=bbox, text=text, cells=cells))

        if task is OcrTask.TABLE:
            # A table crop was sent; the answer's other elements are the
            # caption and stray marks around it.
            tables = [b for b in blocks if b.category is OcrCategory.TABLE]
            return tables or blocks
        return blocks


# ---------------------------------------------------------------------------
# Any other vision model
# ---------------------------------------------------------------------------


class GenericFamily(OcrModelFamily):
    """A chat VLM that reads text, with no layout of its own.

    Keeps the behaviour the vision-OCR lane has had: a page comes back as one
    block of text, a failed call is retried once at half resolution, and an
    empty answer is retried once at higher detail with an explicit instruction
    (which is what non-Latin scripts needed).
    """

    name = "generic"

    _RENDER_DPI: int = 120
    _EMPTY_RETRY_MAX_DIM: int = 1536

    def __init__(self, pipeline: OpenAIPipeline) -> None:
        """Load the prompts this family sends.

        Args:
            pipeline (OpenAIPipeline): Supplies the locale's prompt files.
        """
        self._page_prompt = pipeline.load_prompt(kw="ocr")
        self._table_prompt = pipeline.load_prompt(kw="table_structure")

    @property
    def render_dpi(self) -> int:
        """DPI a PDF page is rasterised at for this family."""
        return self._RENDER_DPI

    @override
    def prompt(self, task: OcrTask) -> str:
        """The OCR instruction, or the table instruction for a table crop."""
        return self._table_prompt if task is OcrTask.TABLE else self._page_prompt

    @override
    def parse(self, answer: str, task: OcrTask, *, image_size: tuple[int, int], frame: OcrFrame) -> list[OcrBlock]:
        """Wrap the answer in one block covering the whole frame.

        Args:
            answer (str): The model's raw answer.
            task (OcrTask): What was asked.
            image_size (tuple[int, int]): Size of the image the model saw (unused).
            frame (OcrFrame): Target coordinate frame.

        Returns:
            list[OcrBlock]: One block, or none when the answer was empty.
        """
        _ = image_size
        text = (answer or "").strip()
        if not text:
            return []
        bbox = OcrBox(
            x0=frame.offset_x,
            y0=frame.offset_y,
            x1=frame.offset_x + frame.width,
            y1=frame.offset_y + frame.height,
        )
        if task is OcrTask.TABLE:
            grid = parse_html_table(text)
            if grid is None:
                return []
            return [OcrBlock(category=OcrCategory.TABLE, bbox=bbox, text=grid_to_text(grid), cells=grid)]
        return [OcrBlock(category=OcrCategory.TEXT, bbox=bbox, text=text)]

    @override
    def degrade(self, image: PILImage.Image, limits: OcrLimits, *, context: str = "") -> PILImage.Image | None:
        """Half resolution, floored — an endpoint that was too slow gets less to chew."""
        half = max(limits.max_dim // 2, 256)
        reduced = cap_image(image, half, context=context)
        return reduced if reduced.size != image.size else None

    @override
    def escalate(
        self, base_image: PILImage.Image, sent: PILImage.Image, limits: OcrLimits, *, context: str = ""
    ) -> tuple[PILImage.Image, str] | None:
        """More detail plus an explicit instruction, for an answer that came back empty."""
        recovery_dim = max(limits.max_dim, self._EMPTY_RETRY_MAX_DIM)
        recovery = cap_image(base_image.copy(), recovery_dim, context=context)
        if recovery.size == sent.size and max(recovery.width, recovery.height) <= limits.max_dim:
            return None
        prompt = (
            f"{self._page_prompt}\n\n"
            "Important: The text may be non-Latin (for example Arabic). "
            "Return all visible text exactly as it appears. "
            "Do not summarize or translate."
        )
        return recovery, prompt


def family_for(model: str, pipeline: OpenAIPipeline) -> OcrModelFamily:
    """Pick the contract a configured model speaks.

    Args:
        model (str): The model id.
        pipeline (OpenAIPipeline): Supplies prompts to families that use them.

    Returns:
        OcrModelFamily: The matching family.
    """
    normalized = (model or "").lower()
    if "dots.ocr" in normalized or "dots.mocr" in normalized or "dotsocr" in normalized.replace("-", ""):
        return DotsFamily()
    return GenericFamily(pipeline)
