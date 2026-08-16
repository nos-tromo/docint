"""Tests for the one OCR engine: model families, coordinates, and failure policy."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai import APIConnectionError, APITimeoutError, InternalServerError
from PIL import Image as PILImage

from docint.core.ocr.engine import DocumentOcrEngine, build_engine
from docint.core.ocr.families import (
    DotsFamily,
    GenericFamily,
    OcrBlock,
    OcrBox,
    OcrCategory,
    OcrFrame,
    OcrLimits,
    OcrTask,
    aligned_size,
    clean_json_array,
    family_for,
)

_ENDPOINT = "http://ocr:8000/v1/chat/completions"

# A dots answer for an 800x1000 px image, top-left pixel coordinates.
DOTS_PAGE = json.dumps(
    [
        {"bbox": [100, 30, 700, 60], "category": "Page-header", "text": "Quarterly Review 2031"},
        {"bbox": [100, 100, 700, 150], "category": "Title", "text": "Annual Report"},
        {"bbox": [100, 180, 700, 210], "category": "Section-header", "text": "1 Overview"},
        {"bbox": [100, 230, 700, 400], "category": "Text", "text": "Revenue rose in every quarter."},
        {"bbox": [100, 420, 700, 440], "category": "Caption", "text": "Table 1: Results"},
        {
            "bbox": [100, 450, 700, 600],
            "category": "Table",
            "text": (
                "<table><tr><td>Model</td><td colspan='2'>Score</td></tr>"
                "<tr><td>Alpha</td><td>1</td><td>2</td></tr></table>"
            ),
        },
        {"bbox": [100, 620, 400, 800], "category": "Picture"},
        {"bbox": [100, 820, 700, 850], "category": "Formula", "text": "E = mc^2"},
        {"bbox": [100, 960, 700, 990], "category": "Page-footer", "text": "7"},
    ]
)


def _http_500() -> InternalServerError:
    request = httpx.Request("POST", _ENDPOINT)
    return InternalServerError("Error code: 500", response=httpx.Response(500, request=request), body=None)


def _timeout() -> APITimeoutError:
    return APITimeoutError(request=httpx.Request("POST", _ENDPOINT))


def _connection_error() -> APIConnectionError:
    return APIConnectionError(request=httpx.Request("POST", _ENDPOINT))


def _response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


def _pipeline() -> MagicMock:
    pipeline = MagicMock()
    pipeline.load_prompt.side_effect = lambda kw: {"ocr": "Read the text.", "table_structure": "Return HTML."}[kw]
    pipeline.seed = 42
    pipeline.temperature = 0.0
    pipeline.top_p = 0.1
    pipeline.reasoning_effort = None
    return pipeline


class TestFamilySelection:
    """Which contract a configured model speaks."""

    def test_dots_ids(self) -> None:
        """Every dots spelling in the wild lands on the layout family."""
        for model in ("dots-studio/dots.mocr", "rednote-hilab/dots.ocr", "local/DotsOCR"):
            assert family_for(model, _pipeline()).name == "dots"

    def test_everything_else_is_generic(self) -> None:
        """A model we know nothing about is asked for plain text."""
        for model in ("zai-org/GLM-OCR", "Qwen/Qwen3.5-2B", "gpt-4o", ""):
            assert family_for(model, _pipeline()).name == "generic"


class TestAlignedSize:
    """The dots family renders at the size the model would resize to."""

    def test_multiples_of_the_patch_factor_within_budget(self) -> None:
        """Sides land on the 28-px grid so the model's own resize is a no-op."""
        width, height = aligned_size(612.0, 792.0, max_pixels=2_000_000)
        assert width % 28 == 0 and height % 28 == 0
        assert width * height <= 2_000_000
        assert abs(width / height - 612.0 / 792.0) < 0.03

    def test_the_default_budget_is_the_servers_own(self) -> None:
        """28*28*2560 is what vllm-service's `ocr` backend caps at.

        Rendering above the cap is the one setting that breaks a layout model
        silently: it resizes the image itself and answers in the frame of the
        image it saw, not the one that was sent.
        """
        assert DocumentOcrEngine._DEFAULT_MAX_PIXELS == 28 * 28 * 2560
        assert OcrLimits().max_pixels == DocumentOcrEngine._DEFAULT_MAX_PIXELS

    def test_small_budget_still_yields_a_usable_image(self) -> None:
        """A tight budget shrinks the image rather than failing."""
        width, height = aligned_size(612.0, 792.0, max_pixels=200_000)
        assert width * height <= 200_000
        assert width >= 28 and height >= 28


class TestCleanJsonArray:
    """Models fence their answers, and long pages get cut off mid-element."""

    def test_plain_and_fenced(self) -> None:
        """A bare array and a fenced one both parse."""
        assert clean_json_array(DOTS_PAGE) == DOTS_PAGE
        assert json.loads(clean_json_array("```json\n" + DOTS_PAGE + "\n```"))

    def test_prose_around_the_array(self) -> None:
        """Chatty models wrap the answer in a sentence."""
        assert json.loads(clean_json_array("Here you go:\n" + DOTS_PAGE + "\nHope that helps"))

    def test_truncated_tail_keeps_complete_elements(self) -> None:
        """A page that exhausted the token budget still yields what arrived."""
        truncated = DOTS_PAGE[: DOTS_PAGE.index('{"bbox": [100, 420')] + '{"bbox": [1'
        elements = json.loads(clean_json_array(truncated))
        assert [e["category"] for e in elements] == ["Page-header", "Title", "Section-header", "Text"]

    def test_junk(self) -> None:
        """Nothing array-shaped in, nothing out."""
        assert clean_json_array("I cannot read this.") == ""
        assert clean_json_array("") == ""


class TestDotsFamily:
    """The layout family's answer becomes blocks in the caller's frame."""

    @staticmethod
    def _parse(frame: OcrFrame, task: OcrTask = OcrTask.PAGE) -> list[OcrBlock]:
        return DotsFamily().parse(DOTS_PAGE, task, image_size=(800, 1000), frame=frame)

    def test_categories(self) -> None:
        """Every dots category maps onto the package's own vocabulary."""
        blocks = self._parse(OcrFrame(width=400.0, height=500.0))
        assert [b.category for b in blocks] == [
            OcrCategory.PAGE_HEADER,
            OcrCategory.TITLE,
            OcrCategory.SECTION_HEADER,
            OcrCategory.TEXT,
            OcrCategory.CAPTION,
            OcrCategory.TABLE,
            OcrCategory.PICTURE,
            OcrCategory.FORMULA,
            OcrCategory.PAGE_FOOTER,
        ]

    def test_page_coordinates_are_scaled_and_flipped(self) -> None:
        """Pixels with a top-left origin become frame units with a bottom-left one."""
        blocks = self._parse(OcrFrame(width=400.0, height=500.0))
        title = next(b for b in blocks if b.category is OcrCategory.TITLE)
        assert (title.bbox.x0, title.bbox.x1) == pytest.approx((50.0, 350.0))
        assert (title.bbox.y0, title.bbox.y1) == pytest.approx((425.0, 450.0))

    def test_region_coordinates_are_offset_into_the_page(self) -> None:
        """A crop's boxes come back in page space, not crop space."""
        frame = OcrFrame(width=400.0, height=500.0, offset_x=100.0, offset_y=200.0)
        title = next(b for b in self._parse(frame) if b.category is OcrCategory.TITLE)
        assert (title.bbox.x0, title.bbox.x1) == pytest.approx((150.0, 450.0))
        assert (title.bbox.y0, title.bbox.y1) == pytest.approx((625.0, 650.0))

    def test_table_html_becomes_cells_and_row_major_text(self) -> None:
        """A spanning header is expanded, and the text reads row by row."""
        table = next(b for b in self._parse(OcrFrame(width=400.0, height=500.0)) if b.category is OcrCategory.TABLE)
        assert table.cells == [["Model", "Score", "Score"], ["Alpha", "1", "2"]]
        assert table.text.splitlines()[0] == "Model | Score | Score"

    def test_picture_carries_no_text(self) -> None:
        """A picture is a region, not words."""
        picture = next(b for b in self._parse(OcrFrame(width=400.0, height=500.0)) if b.category is OcrCategory.PICTURE)
        assert picture.text == ""

    def test_table_task_keeps_only_the_table(self) -> None:
        """Asked about a table crop, the caption and stray marks are noise."""
        blocks = self._parse(OcrFrame(width=400.0, height=500.0), OcrTask.TABLE)
        assert [b.category for b in blocks] == [OcrCategory.TABLE]

    def test_junk_answer(self) -> None:
        """An answer that is not a layout yields nothing."""
        family = DotsFamily()
        assert family.parse("nope", OcrTask.PAGE, image_size=(800, 1000), frame=OcrFrame(400.0, 500.0)) == []


class TestGenericFamily:
    """A model without layout reads text, and that is one block."""

    def test_page_answer_is_one_block_over_the_frame(self) -> None:
        """A model without layout gives one block covering the page."""
        family = GenericFamily(_pipeline())
        blocks = family.parse("Some read text.", OcrTask.PAGE, image_size=(100, 100), frame=OcrFrame(612.0, 792.0))
        assert len(blocks) == 1
        assert blocks[0].category is OcrCategory.TEXT
        assert blocks[0].text == "Some read text."
        assert (blocks[0].bbox.x1, blocks[0].bbox.y1) == pytest.approx((612.0, 792.0))

    def test_table_answer_is_parsed_as_html(self) -> None:
        """A table crop's HTML answer becomes cells."""
        family = GenericFamily(_pipeline())
        blocks = family.parse(
            "<table><tr><td>A</td><td>B</td></tr></table>",
            OcrTask.TABLE,
            image_size=(10, 10),
            frame=OcrFrame(10.0, 10.0),
        )
        assert blocks[0].category is OcrCategory.TABLE
        assert blocks[0].cells == [["A", "B"]]

    def test_empty_answer_is_no_block(self) -> None:
        """An empty answer is not a block of empty text."""
        family = GenericFamily(_pipeline())
        assert family.parse("   ", OcrTask.PAGE, image_size=(10, 10), frame=OcrFrame(10.0, 10.0)) == []

    def test_prompts_differ_by_task(self) -> None:
        """Reading a page and reading a table are different instructions."""
        family = GenericFamily(_pipeline())
        assert family.prompt(OcrTask.PAGE) == "Read the text."
        assert family.prompt(OcrTask.TABLE) == "Return HTML."


def _engine(model: str, **kwargs: object) -> tuple[DocumentOcrEngine, MagicMock]:
    """Build an engine over a faked PDF and client."""
    page = MagicMock()
    page.get_width.return_value = 612.0
    page.get_height.return_value = 792.0
    bitmap = MagicMock()
    bitmap.to_pil.return_value = PILImage.new("RGB", (1232, 1596), color="white")
    page.render.return_value = bitmap
    pdf = MagicMock()
    pdf.__getitem__ = MagicMock(return_value=page)

    with (
        patch("docint.core.ocr.engine.pypdfium2") as mock_pdfium,
        patch("docint.core.ocr.engine.OpenAIPipeline", return_value=_pipeline()),
        patch("docint.core.ocr.engine._OpenAI"),
        patch("docint.core.ocr.engine.load_openai_env") as mock_openai_env,
        patch("docint.core.ocr.engine.load_ocr_client_env") as mock_ocr_env,
        patch("docint.core.ocr.engine.load_model_env") as mock_model_env,
    ):
        mock_pdfium.PdfDocument.return_value = pdf
        mock_openai_env.return_value.api_base = "http://vision:8000/v1"
        mock_openai_env.return_value.api_key = "sk-test"
        mock_openai_env.return_value.timeout = 60.0
        mock_ocr_env.return_value.model = model
        mock_ocr_env.return_value.api_base = "http://ocr:8000/v1"
        mock_ocr_env.return_value.api_key = "sk-test"
        mock_ocr_env.return_value.timeout = 120.0
        mock_model_env.return_value.vision_model = "vision/model"
        engine = DocumentOcrEngine("/fake/doc.pdf", max_retries=0, **kwargs)  # type: ignore[arg-type]
    return engine, page


class TestEngineReads:
    """The three entry points, and what reaches the endpoint."""

    def test_read_page_with_a_layout_model(self) -> None:
        """A dots page comes back as blocks in page points."""
        engine, page = _engine("dots-studio/dots.mocr", max_pixels=2_000_000)
        with (
            patch("docint.core.ocr.engine.time.sleep"),
            patch.object(engine._client.chat.completions, "create", return_value=_response(DOTS_PAGE)) as create,
        ):
            blocks = engine.read_page(0)

        assert engine.reads_layout is True
        assert len(blocks) == 9
        assert blocks[1].category is OcrCategory.TITLE
        # Scaled by the size actually sent (1232 px wide), which is the point of
        # rendering on the model's own grid — not by the fixture's nominal size.
        assert blocks[1].bbox.x0 == pytest.approx(612.0 * 100 / 1232, abs=1.0)
        kwargs = create.call_args[1]
        assert kwargs["model"] == "dots-studio/dots.mocr"
        parts = kwargs["messages"][0]["content"]
        assert parts[0]["type"] == "image_url"  # image first, as the model's own client sends it
        assert "layout" in next(p["text"] for p in parts if p["type"] == "text").lower()
        # Rendered on the model's pixel grid, not at a fixed DPI.
        assert page.render.call_args[1]["scale"] == pytest.approx(1232 / 612.0, rel=0.02)
        assert engine.stats.pages_read == 1

    def test_read_page_with_a_text_only_model(self) -> None:
        """A generic model's page is one text block spanning the page."""
        engine, page = _engine("zai-org/GLM-OCR", max_image_dimension=256)
        with (
            patch("docint.core.ocr.engine.time.sleep"),
            patch.object(engine._client.chat.completions, "create", return_value=_response("Read text")) as create,
        ):
            blocks = engine.read_page(0)

        assert engine.reads_layout is False
        assert [b.category for b in blocks] == [OcrCategory.TEXT]
        assert (blocks[0].bbox.x1, blocks[0].bbox.y1) == pytest.approx((612.0, 792.0))
        assert (
            next(p["text"] for p in create.call_args[1]["messages"][0]["content"] if p["type"] == "text")
            == "Read the text."
        )
        assert page.render.call_args[1]["scale"] == pytest.approx(120 / 72, rel=0.01)

    def test_read_region_crops_and_maps_back(self) -> None:
        """A region is cropped before sending, and its boxes come back in page space."""
        engine, page = _engine("dots-studio/dots.mocr")
        with (
            patch("docint.core.ocr.engine.time.sleep"),
            patch.object(engine._client.chat.completions, "create", return_value=_response(DOTS_PAGE)),
        ):
            blocks = engine.read_region(0, OcrBox(x0=100.0, y0=400.0, x1=500.0, y1=700.0))

        crop = page.render.call_args[1]["crop"]
        assert crop[0] == pytest.approx(94.0)  # left, padded
        assert crop[1] == pytest.approx(394.0)  # bottom
        assert crop[2] == pytest.approx(106.0)  # right = 612 - 500 - 6
        assert crop[3] == pytest.approx(86.0)  # top = 792 - 700 - 6
        # Every box sits inside the padded region, not at the page origin.
        assert all(b.bbox.x0 >= 94.0 - 0.01 and b.bbox.y0 >= 394.0 - 0.01 for b in blocks)

    def test_read_image_without_a_document(self) -> None:
        """An image needs no PDF; its boxes are in pixels."""
        with (
            patch("docint.core.ocr.engine.pypdfium2") as mock_pdfium,
            patch("docint.core.ocr.engine.OpenAIPipeline", return_value=_pipeline()),
            patch("docint.core.ocr.engine._OpenAI"),
            patch("docint.core.ocr.engine.load_openai_env"),
            patch("docint.core.ocr.engine.load_ocr_client_env") as mock_ocr_env,
            patch("docint.core.ocr.engine.load_model_env"),
        ):
            mock_ocr_env.return_value.model = "dots-studio/dots.mocr"
            mock_ocr_env.return_value.api_base = "http://ocr:8000/v1"
            mock_ocr_env.return_value.api_key = "k"
            mock_ocr_env.return_value.timeout = 30.0
            engine = DocumentOcrEngine(max_retries=0)
            mock_pdfium.PdfDocument.assert_not_called()

        image = PILImage.new("RGB", (800, 1000), color="white")
        with (
            patch("docint.core.ocr.engine.time.sleep"),
            patch.object(engine._client.chat.completions, "create", return_value=_response(DOTS_PAGE)),
        ):
            blocks = engine.read_image(image)
        assert len(blocks) == 9
        engine.close()  # no document: must not raise

    def test_reading_a_page_without_a_document_is_a_clear_error(self) -> None:
        """An engine built for images cannot be asked for page 3 of nothing."""
        with (
            patch("docint.core.ocr.engine.OpenAIPipeline", return_value=_pipeline()),
            patch("docint.core.ocr.engine._OpenAI"),
            patch("docint.core.ocr.engine.load_openai_env"),
            patch("docint.core.ocr.engine.load_ocr_client_env") as mock_ocr_env,
            patch("docint.core.ocr.engine.load_model_env"),
        ):
            mock_ocr_env.return_value.model = "dots.mocr"
            mock_ocr_env.return_value.api_base = "b"
            mock_ocr_env.return_value.api_key = "k"
            mock_ocr_env.return_value.timeout = 30.0
            engine = DocumentOcrEngine()
        with pytest.raises(RuntimeError, match="without a document"):
            engine.read_page(0)


class TestEngineFailurePolicy:
    """A degraded read must cost a page, and a dead endpoint must cost a document."""

    def test_error_status_costs_one_page_and_keeps_the_lane(self) -> None:
        """The endpoint answered — badly. The next page is usually fine."""
        engine, _ = _engine("dots-studio/dots.mocr")
        with (
            patch("docint.core.ocr.engine.time.sleep"),
            patch.object(engine._client.chat.completions, "create", side_effect=_http_500()),
        ):
            assert engine.read_page(0) == []
        assert engine.stats.pages_failed == 1
        assert engine.disabled is False

    def test_retry_waits_before_trying_again(self) -> None:
        """The retry must not land inside the same burst of rejections."""
        engine, _ = _engine("dots-studio/dots.mocr")
        with (
            patch("docint.core.ocr.engine.time.sleep") as sleep,
            patch.object(engine._client.chat.completions, "create", side_effect=[_http_500(), _response(DOTS_PAGE)]),
        ):
            assert len(engine.read_page(0)) == 9
        assert sleep.call_count == 1
        assert sleep.call_args[0][0] > 0

    def test_generic_family_retries_at_half_resolution(self) -> None:
        """The historical behaviour: an endpoint that choked gets a smaller image."""
        engine, _ = _engine("zai-org/GLM-OCR", max_image_dimension=1024)
        sizes: list[int] = []

        def _capture(**kwargs: object) -> MagicMock:
            url = kwargs["messages"][0]["content"][0]["image_url"]["url"]  # type: ignore[index]
            sizes.append(len(url))
            if len(sizes) == 1:
                raise _timeout()
            return _response("recovered")

        with (
            patch("docint.core.ocr.engine.time.sleep"),
            patch.object(engine._client.chat.completions, "create", side_effect=_capture),
        ):
            blocks = engine.read_page(0)
        assert [b.text for b in blocks] == ["recovered"]
        assert sizes[1] < sizes[0]  # the retry payload is smaller

    def test_three_unanswered_calls_disable_the_document(self) -> None:
        """Each unanswered call costs a full timeout; three is enough to stop."""
        engine, _ = _engine("dots-studio/dots.mocr")
        with (
            patch("docint.core.ocr.engine.time.sleep"),
            patch.object(engine._client.chat.completions, "create", side_effect=_timeout()),
        ):
            for _ in range(3):
                assert engine.read_page(0) == []
        assert engine.disabled is True
        with patch.object(engine._client.chat.completions, "create", side_effect=_connection_error()) as create:
            assert engine.read_page(0) == []
            create.assert_not_called()
        assert engine.stats.pages_skipped == 1

    def test_rejections_do_not_reset_the_budget(self) -> None:
        """An unreachable endpoint interleaved with rejections is still unreachable."""
        engine, _ = _engine("dots-studio/dots.mocr")
        with patch("docint.core.ocr.engine.time.sleep"):
            for error in (_timeout(), _http_500(), _timeout(), _timeout()):
                with patch.object(engine._client.chat.completions, "create", side_effect=error):
                    engine.read_page(0)
        assert engine.disabled is True

    def test_refusal_is_treated_as_nothing_read(self) -> None:
        """A safety refusal is not the page's text."""
        engine, _ = _engine("zai-org/GLM-OCR")
        with (
            patch("docint.core.ocr.engine.time.sleep"),
            patch.object(
                engine._client.chat.completions,
                "create",
                return_value=_response("I'm sorry, I can't assist with that."),
            ),
        ):
            assert engine.read_page(0) == []
        assert engine.stats.pages_failed == 1

    def test_repeated_filler_is_squeezed(self) -> None:
        """A dotted form line can lock a model into filler until max_tokens."""
        engine, _ = _engine("zai-org/GLM-OCR")
        filler = "Name" + "." * 5000 + "date"
        with (
            patch("docint.core.ocr.engine.time.sleep"),
            patch.object(engine._client.chat.completions, "create", return_value=_response(filler)),
        ):
            blocks = engine.read_page(0)
        assert len(blocks[0].text) < 200

    def test_empty_answer_is_retried_with_more_detail(self) -> None:
        """Non-Latin scripts needed this: same page, larger image, explicit instruction."""
        engine, _ = _engine("zai-org/GLM-OCR", max_image_dimension=256)
        prompts: list[str] = []

        def _capture(**kwargs: object) -> MagicMock:
            parts = kwargs["messages"][0]["content"]  # type: ignore[index]
            prompts.append(next(p["text"] for p in parts if p["type"] == "text"))
            return _response("" if len(prompts) == 1 else "نص عربي")

        with (
            patch("docint.core.ocr.engine.time.sleep"),
            patch.object(engine._client.chat.completions, "create", side_effect=_capture),
        ):
            blocks = engine.read_page(0)
        assert [b.text for b in blocks] == ["نص عربي"]
        assert len(prompts) == 2
        assert "non-Latin" in prompts[1]


def test_build_engine_returns_none_when_it_cannot_be_built() -> None:
    """Callers treat OCR as optional; an unbuildable engine is not an error there."""
    with patch("docint.core.ocr.engine.OpenAIPipeline", side_effect=RuntimeError("no prompts")):
        assert build_engine("/fake/doc.pdf") is None
