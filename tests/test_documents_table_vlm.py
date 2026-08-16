"""Tests for the table-structure vision lane: HTML parsing, gating, engine behaviour."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai import APIConnectionError, APITimeoutError, InternalServerError
from PIL import Image as PILImage

from docint.core.readers.documents.models import BBox
from docint.core.readers.documents.table_vlm import (
    TableStructureEngine,
    parse_html_table,
    table_grid_from_dots,
)
from docint.core.readers.documents.tables import needs_structure

_ENDPOINT = "http://vision:8000/v1/chat/completions"


def _http_500() -> InternalServerError:
    """An endpoint that answered with an error status."""
    request = httpx.Request("POST", _ENDPOINT)
    response = httpx.Response(500, request=request, json={"error": "Internal Server Error"})
    return InternalServerError("Error code: 500", response=response, body=None)


def _timeout() -> APITimeoutError:
    """An endpoint that did not answer in time."""
    return APITimeoutError(request=httpx.Request("POST", _ENDPOINT))


def _connection_error() -> APIConnectionError:
    """An endpoint that could not be reached."""
    return APIConnectionError(request=httpx.Request("POST", _ENDPOINT))


def _response(content: str) -> MagicMock:
    """A chat completion carrying ``content``."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


class TestParseHtmlTable:
    """Turning the model's HTML into a flat grid."""

    def test_simple_table(self) -> None:
        """Rows and cells come back in order."""
        html = "<table><tr><th>Model</th><th>BLEU</th></tr><tr><td>Alpha</td><td>23.8</td></tr></table>"
        assert parse_html_table(html) == [["Model", "BLEU"], ["Alpha", "23.8"]]

    def test_colspan_is_expanded_into_every_column_it_covers(self) -> None:
        """A spanning header labels each of its columns — the case geometry cannot see."""
        html = (
            "<table>"
            "<tr><td></td><td colspan='2'>BLEU</td></tr>"
            "<tr><td>Model</td><td>EN-DE</td><td>EN-FR</td></tr>"
            "<tr><td>Alpha</td><td>23.8</td><td>39.2</td></tr>"
            "</table>"
        )
        assert parse_html_table(html) == [
            ["", "BLEU", "BLEU"],
            ["Model", "EN-DE", "EN-FR"],
            ["Alpha", "23.8", "39.2"],
        ]

    def test_rowspan_is_expanded_down_the_rows(self) -> None:
        """A cell spanning rows repeats into the rows below it."""
        html = "<table><tr><td rowspan='2'>Group</td><td>a</td></tr><tr><td>b</td></tr></table>"
        assert parse_html_table(html) == [["Group", "a"], ["Group", "b"]]

    def test_fenced_output_is_accepted(self) -> None:
        """Models like to wrap answers in a code fence; the fence is not the table."""
        html = "```html\n<table><tr><td>A</td><td>B</td></tr></table>\n```"
        assert parse_html_table(html) == [["A", "B"]]

    def test_prose_around_the_table_is_ignored(self) -> None:
        """Only the first table matters, whatever the model says around it."""
        html = "Here is the table:\n<table><tr><td>A</td></tr></table>\nHope that helps!"
        assert parse_html_table(html) == [["A"]]

    def test_entities_and_nested_markup_are_unescaped(self) -> None:
        """Cell text is plain text: entities resolved, inline markup dropped."""
        html = "<table><tr><td>R&amp;D</td><td><b>2.3</b> &middot; 10<sup>19</sup></td></tr></table>"
        assert parse_html_table(html) == [["R&D", "2.3 · 1019"]]

    def test_ragged_rows_are_padded(self) -> None:
        """A short row keeps the table rectangular."""
        html = "<table><tr><td>A</td><td>B</td></tr><tr><td>C</td></tr></table>"
        assert parse_html_table(html) == [["A", "B"], ["C", ""]]

    def test_no_table_returns_none(self) -> None:
        """Nothing usable in, nothing out."""
        assert parse_html_table("I could not read the image.") is None
        assert parse_html_table("") is None
        assert parse_html_table("<table></table>") is None


class TestNeedsStructure:
    """Which tables are worth a vision call."""

    def test_missing_grid_needs_help(self) -> None:
        """No structure at all is the clearest case."""
        assert needs_structure(None) is True
        assert needs_structure([]) is True

    def test_single_column_needs_help(self) -> None:
        """One column is not a recovered table."""
        assert needs_structure([["a"], ["b"], ["c"]]) is True

    def test_sparse_grid_needs_help(self) -> None:
        """Empty cells are where spanning headers were flattened away."""
        grid = [["", "BLEU", ""], ["Model", "", ""], ["Alpha", "23.8", "39.2"]]
        assert needs_structure(grid) is True

    def test_dense_grid_is_left_alone(self) -> None:
        """A table geometry recovered cleanly costs no vision call."""
        grid = [["Parser", "Training", "F1"], ["Alpha", "WSJ only", "88.3"], ["Beta", "semi-sup", "91.3"]]
        assert needs_structure(grid) is False


class TestTableStructureEngine:
    """Rendering the region, calling the endpoint, and failing soft."""

    @staticmethod
    def _engine(tmp_path: Path) -> tuple[TableStructureEngine, MagicMock]:
        """Build an engine over a faked pypdfium2 document."""
        page = MagicMock()
        page.get_width.return_value = 612.0
        page.get_height.return_value = 792.0
        bitmap = MagicMock()
        bitmap.to_pil.return_value = PILImage.new("RGB", (400, 200), color="white")
        page.render.return_value = bitmap
        pdf = MagicMock()
        pdf.__getitem__ = MagicMock(return_value=page)

        with (
            patch("docint.core.readers.documents.table_vlm.pypdfium2") as mock_pdfium,
            patch("docint.core.readers.documents.table_vlm.OpenAIPipeline") as mock_pipeline_cls,
            patch("docint.core.readers.documents.table_vlm._OpenAI"),
            patch("docint.core.readers.documents.table_vlm.load_openai_env"),
            patch("docint.core.readers.documents.table_vlm.load_table_vlm_env") as mock_table_env,
        ):
            mock_pdfium.PdfDocument.return_value = pdf
            pipeline = MagicMock()
            pipeline.load_prompt.return_value = "Return the table as HTML"
            pipeline.seed = 42
            pipeline.temperature = 0.0
            pipeline.top_p = 0.1
            pipeline.reasoning_effort = None
            mock_pipeline_cls.return_value = pipeline
            mock_table_env.return_value.model = "vision-model"
            mock_table_env.return_value.api_base = "http://vision:8000/v1"
            mock_table_env.return_value.api_key = "sk-test"
            mock_table_env.return_value.timeout = 30.0
            engine = TableStructureEngine(
                tmp_path / "doc.pdf",
                timeout=30.0,
                max_retries=0,
                max_image_dimension=512,
                max_tokens=1024,
            )
        return engine, page

    def test_renders_only_the_table_region(self, tmp_path: Path) -> None:
        """The page is cropped to the block's bbox before it is sent."""
        engine, page = self._engine(tmp_path)
        bbox = BBox(x0=100.0, y0=400.0, x1=500.0, y1=700.0)
        with (
            patch("docint.core.readers.documents.table_vlm.time.sleep"),
            patch.object(
                engine._client.chat.completions,
                "create",
                return_value=_response("<table><tr><td>A</td><td>B</td></tr></table>"),
            ),
        ):
            engine.structure_for(0, bbox)

        crop = page.render.call_args[1]["crop"]
        # pypdfium2 crop = amount cut off (left, bottom, right, top), padded outward.
        assert crop[0] == pytest.approx(100.0, abs=12.0)
        assert crop[1] == pytest.approx(400.0, abs=12.0)
        assert crop[2] == pytest.approx(112.0, abs=12.0)
        assert crop[3] == pytest.approx(92.0, abs=12.0)

    def test_returns_the_parsed_grid(self, tmp_path: Path) -> None:
        """A well-formed answer becomes the table's grid."""
        engine, _ = self._engine(tmp_path)
        html = "<table><tr><td></td><td colspan='2'>BLEU</td></tr><tr><td>Model</td><td>DE</td><td>FR</td></tr></table>"
        with (
            patch("docint.core.readers.documents.table_vlm.time.sleep"),
            patch.object(engine._client.chat.completions, "create", return_value=_response(html)) as create,
        ):
            grid = engine.structure_for(0, BBox(x0=0, y0=0, x1=612, y1=792))

        assert grid == [["", "BLEU", "BLEU"], ["Model", "DE", "FR"]]
        kwargs = create.call_args[1]
        assert kwargs["model"] == "vision-model"
        assert kwargs["max_tokens"] == 1024
        assert engine.stats.tables_recovered == 1

    def test_unusable_answer_is_no_answer(self, tmp_path: Path) -> None:
        """Junk back from the model leaves the geometric grid in place."""
        engine, _ = self._engine(tmp_path)
        with (
            patch("docint.core.readers.documents.table_vlm.time.sleep"),
            patch.object(engine._client.chat.completions, "create", return_value=_response("no table here")),
        ):
            assert engine.structure_for(0, BBox(x0=0, y0=0, x1=612, y1=792)) is None
        assert engine.stats.tables_failed == 1

    def test_http_error_costs_one_table_only(self, tmp_path: Path) -> None:
        """An error status is this table's problem, not the document's."""
        engine, _ = self._engine(tmp_path)
        with (
            patch("docint.core.readers.documents.table_vlm.time.sleep"),
            patch.object(engine._client.chat.completions, "create", side_effect=_http_500()),
        ):
            assert engine.structure_for(0, BBox(x0=0, y0=0, x1=612, y1=792)) is None
        assert engine.stats.tables_failed == 1
        assert engine.disabled is False

    def test_retry_waits_before_trying_again(self, tmp_path: Path) -> None:
        """The retry does not land inside the same burst of failures."""
        engine, _ = self._engine(tmp_path)
        html = "<table><tr><td>A</td><td>B</td></tr></table>"
        with (
            patch("docint.core.readers.documents.table_vlm.time.sleep") as sleep,
            patch.object(
                engine._client.chat.completions,
                "create",
                side_effect=[_http_500(), _response(html)],
            ),
        ):
            assert engine.structure_for(0, BBox(x0=0, y0=0, x1=612, y1=792)) == [["A", "B"]]
        assert sleep.call_count == 1
        assert sleep.call_args[0][0] > 0

    def test_unreachable_endpoint_disables_the_lane(self, tmp_path: Path) -> None:
        """Three consecutive dead calls give up on the document instead of burning timeouts."""
        engine, _ = self._engine(tmp_path)
        with (
            patch("docint.core.readers.documents.table_vlm.time.sleep"),
            patch.object(engine._client.chat.completions, "create", side_effect=_timeout()),
        ):
            for _ in range(3):
                assert engine.structure_for(0, BBox(x0=0, y0=0, x1=612, y1=792)) is None

        assert engine.disabled is True
        with patch.object(engine._client.chat.completions, "create", side_effect=_connection_error()) as create:
            assert engine.structure_for(0, BBox(x0=0, y0=0, x1=612, y1=792)) is None
            create.assert_not_called()
        assert engine.stats.tables_skipped == 1


class TestDotsTableAnswers:
    """A dots-family model answers with layout JSON; the table's HTML is inside it."""

    def test_table_element_html_becomes_the_grid(self) -> None:
        """The first Table element's HTML is expanded like a plain HTML answer."""
        answer = (
            '[{"bbox": [0, 0, 500, 30], "category": "Caption", "text": "Table 2: Scores"}, '
            '{"bbox": [0, 40, 500, 300], "category": "Table", '
            '"text": "<table><tr><td>Model</td><td colspan=\'2\'>BLEU</td></tr>'
            '<tr><td>Alpha</td><td>1</td><td>2</td></tr></table>"}]'
        )
        assert table_grid_from_dots(answer) == [["Model", "BLEU", "BLEU"], ["Alpha", "1", "2"]]

    def test_no_table_element_is_no_answer(self) -> None:
        """A layout with no Table element gives nothing (the geometric grid stays)."""
        assert table_grid_from_dots('[{"bbox": [0, 0, 10, 10], "category": "Text", "text": "hi"}]') is None
        assert table_grid_from_dots("junk") is None


class TestTableEngineWithDotsModel:
    """When the configured table model is dots, the engine uses its layout prompt and JSON."""

    @staticmethod
    def _engine(tmp_path: Path, model: str) -> tuple[TableStructureEngine, MagicMock]:
        page = MagicMock()
        page.get_width.return_value = 612.0
        page.get_height.return_value = 792.0
        bitmap = MagicMock()
        bitmap.to_pil.return_value = PILImage.new("RGB", (400, 200), color="white")
        page.render.return_value = bitmap
        pdf = MagicMock()
        pdf.__getitem__ = MagicMock(return_value=page)
        with (
            patch("docint.core.readers.documents.table_vlm.pypdfium2") as mock_pdfium,
            patch("docint.core.readers.documents.table_vlm.OpenAIPipeline") as mock_pipeline_cls,
            patch("docint.core.readers.documents.table_vlm._OpenAI"),
            patch("docint.core.readers.documents.table_vlm.load_openai_env"),
            patch("docint.core.readers.documents.table_vlm.load_table_vlm_env") as mock_table_env,
        ):
            mock_pdfium.PdfDocument.return_value = pdf
            pipeline = MagicMock()
            pipeline.load_prompt.return_value = "Return the table as HTML"
            pipeline.seed = 42
            pipeline.temperature = 0.0
            pipeline.top_p = 0.1
            pipeline.reasoning_effort = None
            mock_pipeline_cls.return_value = pipeline
            mock_table_env.return_value.model = model
            mock_table_env.return_value.api_base = "http://ocr:8000/v1"
            mock_table_env.return_value.api_key = "sk-test"
            mock_table_env.return_value.timeout = 30.0
            engine = TableStructureEngine(tmp_path / "doc.pdf", timeout=30.0, max_retries=0)
        return engine, page

    def test_dots_model_gets_the_layout_prompt_and_json_is_parsed(self, tmp_path: Path) -> None:
        """Prompt is dots' own layout task; the Table element's HTML becomes the grid."""
        engine, _ = self._engine(tmp_path, "dots-studio/dots.mocr")
        answer = (
            '[{"bbox": [0, 0, 500, 300], "category": "Table", "text": "<table><tr><td>A</td><td>B</td></tr></table>"}]'
        )
        with (
            patch("docint.core.readers.documents.table_vlm.time.sleep"),
            patch.object(engine._client.chat.completions, "create", return_value=_response(answer)) as create,
        ):
            grid = engine.structure_for(0, BBox(x0=0, y0=0, x1=612, y1=792))
        assert grid == [["A", "B"]]
        parts = create.call_args[1]["messages"][0]["content"]
        prompt = next(part["text"] for part in parts if part["type"] == "text")
        assert "layout" in prompt.lower() and "bbox" in prompt.lower()

    def test_generic_model_keeps_the_html_prompt(self, tmp_path: Path) -> None:
        """Any other model is asked for HTML directly, as before."""
        engine, _ = self._engine(tmp_path, "Qwen/Qwen3.5-2B")
        with (
            patch("docint.core.readers.documents.table_vlm.time.sleep"),
            patch.object(
                engine._client.chat.completions,
                "create",
                return_value=_response("<table><tr><td>A</td><td>B</td></tr></table>"),
            ) as create,
        ):
            assert engine.structure_for(0, BBox(x0=0, y0=0, x1=612, y1=792)) == [["A", "B"]]
        parts = create.call_args[1]["messages"][0]["content"]
        assert next(part["text"] for part in parts if part["type"] == "text") == "Return the table as HTML"


def test_table_lane_defaults_to_the_ocr_model_when_one_is_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """With OCR_MODEL set, tables go to the document model unless TABLE_VLM_MODEL says otherwise."""
    from docint.utils.env_cfg import resolve_table_vlm_default_model

    monkeypatch.delenv("TABLE_VLM_MODEL", raising=False)
    monkeypatch.setenv("OCR_MODEL", "dots-studio/dots.mocr")
    assert resolve_table_vlm_default_model("Qwen/Qwen3.5-2B") == "dots-studio/dots.mocr"
    monkeypatch.delenv("OCR_MODEL", raising=False)
    assert resolve_table_vlm_default_model("Qwen/Qwen3.5-2B") == "Qwen/Qwen3.5-2B"
