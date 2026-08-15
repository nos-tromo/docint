"""Table structure recovered from a rendered image by a remote vision model.

Cell geometry can rebuild a table's rows and columns, but not the fact that a
header *spans* several of them: nothing in the text positions says that ``BLEU``
sits over both ``EN-DE`` and ``EN-FR``. That information exists only in the
rendering, so the tables whose structure the geometric pass could not recover
are rendered — region only, caption included — and handed to a vision model,
which is asked for HTML. Its ``rowspan``/``colspan`` are then expanded into the
flat grid the rest of the pipeline already speaks, giving every column a
self-describing header instead of a blank cell.

Like every other model call in docint this is remote and fail-soft: the model
runs on the shared inference host, and a failure leaves the geometric grid
exactly as it was. Which endpoint and model answer is a config seam
(``TABLE_VLM_API_BASE`` / ``TABLE_VLM_MODEL``), so a dedicated document-parsing
model can replace the general vision model without touching this code.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

import pypdfium2
from loguru import logger
from openai import APIConnectionError, APIStatusError, APITimeoutError
from openai import OpenAI as _OpenAI
from openai.types.chat import ChatCompletionContentPartParam, ChatCompletionMessageParam
from typing_extensions import override

from docint.core.readers.documents.imaging import cap_image, encode_jpeg
from docint.core.readers.documents.models import BBox
from docint.utils.env_cfg import load_openai_env, load_table_vlm_env
from docint.utils.llm_sanitize import strip_reasoning
from docint.utils.openai_cfg import OpenAIPipeline


@dataclass
class TableVlmStats:
    """What the table-structure lane did across one document.

    Attributes:
        tables_recovered: Tables whose structure came back usable.
        tables_failed: Tables the model could not answer for (error or junk).
        tables_skipped: Tables not attempted because the lane had been disabled.
    """

    tables_recovered: int = 0
    tables_failed: int = 0
    tables_skipped: int = 0


class _TableHtmlParser(HTMLParser):
    """Collect ``<tr>``/``<td>`` structure, with spans, from the first table."""

    def __init__(self) -> None:
        """Start with no rows collected."""
        super().__init__(convert_charrefs=True)
        self.rows: list[list[tuple[str, int, int]]] = []  # (text, rowspan, colspan)
        self._depth = 0
        self._done = False
        self._cell: list[str] | None = None
        self._spans: tuple[int, int] = (1, 1)

    @staticmethod
    def _span(attrs: list[tuple[str, str | None]], name: str) -> int:
        """Read a span attribute, defaulting to 1 for anything unusable."""
        for key, value in attrs:
            if key == name and value:
                try:
                    return max(1, min(int(value.strip()), 64))
                except ValueError:
                    return 1
        return 1

    @override
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Open a table, row or cell."""
        if self._done:
            return
        if tag == "table":
            self._depth += 1
        elif tag == "tr" and self._depth:
            self.rows.append([])
        elif tag in ("td", "th") and self._depth:
            if not self.rows:
                self.rows.append([])
            self._cell = []
            self._spans = (self._span(attrs, "rowspan"), self._span(attrs, "colspan"))

    @override
    def handle_endtag(self, tag: str) -> None:
        """Close a cell or the table."""
        if self._done:
            return
        if tag in ("td", "th") and self._cell is not None:
            text = re.sub(r"\s+", " ", "".join(self._cell)).strip()
            self.rows[-1].append((text, self._spans[0], self._spans[1]))
            self._cell = None
        elif tag == "table":
            self._depth = max(0, self._depth - 1)
            if self._depth == 0:
                # Only the first table is the answer; anything after it is noise.
                self._done = True

    @override
    def handle_data(self, data: str) -> None:
        """Accumulate text inside the current cell."""
        if self._cell is not None:
            self._cell.append(data)


def parse_html_table(html: str) -> list[list[str]] | None:
    """Turn a model's HTML answer into a rectangular grid.

    ``rowspan``/``colspan`` are expanded by repeating the spanning cell's text
    into every position it covers — that is what turns a two-level header into
    columns that each name their group. Rows are padded to equal width.

    Args:
        html (str): The model's raw answer (code fences and prose tolerated).

    Returns:
        list[list[str]] | None: The grid, or ``None`` when no usable table was found.
    """
    if not html or "<table" not in html.lower():
        return None
    parser = _TableHtmlParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception as exc:  # pragma: no cover - HTMLParser is lenient by design
        logger.debug("Table HTML could not be parsed: {}", exc)
        return None

    grid: list[list[str]] = []
    # Cells still owed to later rows by a rowspan: (row index, column) -> text.
    pending: dict[tuple[int, int], str] = {}
    for row_index, row in enumerate(parser.rows):
        cells: list[str] = []
        column = 0
        for text, rowspan, colspan in row:
            while (row_index, column) in pending:
                cells.append(pending.pop((row_index, column)))
                column += 1
            value = unescape(text)
            for offset in range(colspan):
                cells.append(value)
                for extra in range(1, rowspan):
                    pending[(row_index + extra, column + offset)] = value
            column += colspan
        while (row_index, column) in pending:
            cells.append(pending.pop((row_index, column)))
            column += 1
        grid.append(cells)

    grid = [row for row in grid if row]
    if not grid:
        return None
    width = max(len(row) for row in grid)
    return [row + [""] * (width - len(row)) for row in grid]


class TableStructureEngine:
    """Recovers one table's structure per call from a rendered region.

    The engine holds its own OpenAI client rather than going through
    :class:`~docint.utils.openai_cfg.OpenAIPipeline`, whose vision helper
    collapses every failure into one exception — this lane needs to tell an
    endpoint that answered badly (costs one table) from one that did not answer
    at all (costs a full timeout, and is worth giving up on).
    """

    _RENDER_DPI: int = 200
    _REGION_PADDING: float = 6.0
    _RETRY_BACKOFF_SECONDS: float = 2.0
    _MAX_CONSECUTIVE_FAILURES: int = 3
    _DEFAULT_MAX_IMAGE_DIM: int = 1536
    _DEFAULT_MAX_TOKENS: int = 4096

    def __init__(
        self,
        file_path: str | Path,
        *,
        timeout: float | None = None,
        max_retries: int | None = None,
        max_image_dimension: int | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """Open the PDF and prepare the vision client.

        Args:
            file_path (str | Path): Path to the PDF being processed.
            timeout (float | None): Per-request timeout; falls back to the endpoint config.
            max_retries (int | None): SDK-level retries per request.
            max_image_dimension (int | None): Max pixel dimension of the rendered region.
            max_tokens (int | None): Max tokens the model may generate per table.
        """
        self._file_path = Path(file_path)
        self._pipeline = OpenAIPipeline()
        self._prompt = self._pipeline.load_prompt(kw="table_structure")
        openai_cfg = load_openai_env()
        self._config = load_table_vlm_env(
            default_api_base=openai_cfg.api_base or "",
            default_api_key=openai_cfg.api_key,
            default_model=self._pipeline.vision_model_id,
            default_timeout=timeout if timeout is not None else openai_cfg.timeout,
        )
        self._max_image_dim = max_image_dimension or self._DEFAULT_MAX_IMAGE_DIM
        self._max_tokens = max_tokens or self._DEFAULT_MAX_TOKENS
        self._pdf = pypdfium2.PdfDocument(str(self._file_path))
        self._client = _OpenAI(
            api_key=self._config.api_key,
            base_url=self._config.api_base,
            timeout=self._config.timeout,
            max_retries=max_retries if max_retries is not None else 1,
        )
        self.stats = TableVlmStats()
        self._consecutive_failures = 0
        self.disabled = False

    def structure_for(self, page_index: int, bbox: BBox) -> list[list[str]] | None:
        """Return the grid the model reads out of the region at *bbox*.

        Args:
            page_index (int): Zero-based page number.
            bbox (BBox): The table's region, in bottom-left-origin points.

        Returns:
            list[list[str]] | None: The recovered grid, or ``None`` when the
                model could not be reached, answered with an error, or returned
                nothing usable. Callers keep their existing grid in that case.
        """
        if self.disabled:
            self.stats.tables_skipped += 1
            return None
        try:
            img_b64 = self._render_region(page_index, bbox)
        except Exception as exc:
            logger.warning("Could not render table region on page {}: {}", page_index, exc)
            self.stats.tables_failed += 1
            return None

        answer, reachable = self._ask(img_b64, page_index)
        if answer is None and reachable:
            # The endpoint answered badly: worth one more try after a pause, so
            # the retry does not land inside the same burst.
            time.sleep(self._RETRY_BACKOFF_SECONDS)
            answer, reachable = self._ask(img_b64, page_index)

        if answer is None:
            self._note_failure(page_index, reachable=reachable)
            return None

        self._consecutive_failures = 0
        grid = parse_html_table(answer)
        if grid is None:
            logger.debug("Table structure model returned no usable table for page {}", page_index)
            self.stats.tables_failed += 1
            return None
        self.stats.tables_recovered += 1
        return grid

    def close(self) -> None:
        """Release the underlying document handle."""
        try:
            self._pdf.close()
        except Exception as exc:  # pragma: no cover - best effort
            logger.debug("Closing table-structure document failed: {}", exc)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _render_region(self, page_index: int, bbox: BBox) -> str:
        """Render just the table's region as base64 JPEG.

        Args:
            page_index (int): Zero-based page number.
            bbox (BBox): Region in bottom-left-origin points.

        Returns:
            str: Base64-encoded JPEG of the region.
        """
        page = self._pdf[page_index]
        width = float(page.get_width())
        height = float(page.get_height())
        pad = self._REGION_PADDING
        # pypdfium2's crop is how much to cut off each side.
        crop = (
            max(0.0, bbox.x0 - pad),
            max(0.0, bbox.y0 - pad),
            max(0.0, width - bbox.x1 - pad),
            max(0.0, height - bbox.y1 - pad),
        )
        bitmap = page.render(scale=self._RENDER_DPI / 72, crop=crop)
        image = cap_image(bitmap.to_pil(), self._max_image_dim, context=f"table on page {page_index}")
        return encode_jpeg(image)

    def _ask(self, img_b64: str, page_index: int) -> tuple[str | None, bool]:
        """Send the region to the model.

        Args:
            img_b64 (str): Base64 JPEG of the table region.
            page_index (int): Zero-based page number (for logging).

        Returns:
            tuple[str | None, bool]: The answer (``None`` on failure) and whether
                the endpoint was reachable at all.
        """
        content_parts: list[ChatCompletionContentPartParam] = [
            {"type": "text", "text": self._prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        ]
        messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": content_parts}]
        request_kwargs: dict[str, object] = {}
        if self._pipeline.reasoning_effort is not None:
            request_kwargs["reasoning_effort"] = self._pipeline.reasoning_effort
        try:
            response = self._client.chat.completions.create(  # type: ignore[call-overload]
                model=self._config.model,
                messages=messages,
                max_tokens=self._max_tokens,
                seed=self._pipeline.seed,
                temperature=self._pipeline.temperature,
                top_p=self._pipeline.top_p,
                **request_kwargs,
            )
        except APIStatusError as exc:
            logger.warning("Table structure endpoint rejected page {}: {}", page_index, exc)
            return None, True
        except (APITimeoutError, APIConnectionError) as exc:
            logger.warning("Table structure endpoint unreachable for page {}: {}", page_index, exc)
            return None, False
        except Exception as exc:
            # Treated as unreachable so the per-document budget still protects
            # against a failure mode we have not seen yet.
            logger.warning("Table structure call failed for page {}: {}", page_index, exc)
            return None, False

        raw = response.choices[0].message.content or ""
        cleaned, _reasoning = strip_reasoning(raw)
        return cleaned.strip() or None, True

    def _note_failure(self, page_index: int, *, reachable: bool) -> None:
        """Record a failed table and disable the lane once the endpoint looks dead.

        Args:
            page_index (int): Zero-based page number.
            reachable (bool): Whether the endpoint answered at all.
        """
        self.stats.tables_failed += 1
        if reachable:
            return
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._MAX_CONSECUTIVE_FAILURES:
            self.disabled = True
            logger.warning(
                "Table structure endpoint unreachable {} times in a row (page {}); "
                "disabling the lane for the rest of this document",
                self._consecutive_failures,
                page_index,
            )
