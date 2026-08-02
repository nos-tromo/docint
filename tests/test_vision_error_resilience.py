"""Tests that a burst of upstream vision errors costs pages, not the document.

Measured against the dev stack: a successful vision OCR page takes 68-117s, a
successful image tagging call 3.5-17s, and *every* failure came back in
0.5-1.0s carrying an upstream HTTP 500. Those fast rejects arrive in bursts of
a few seconds with immediate recovery either side.

Two behaviours turned that blip into lost work. The OCR failure budget --
built to stop bleeding a full timeout per page against an *unreachable*
endpoint -- counted answered-with-an-error the same as no-answer-at-all, so
three fast 500s disabled vision OCR for the whole document (19 of 30 pages
went unread, reported as ``pages_failed=0``). And both retry paths fire
immediately, landing the retry inside the same burst.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai import APIConnectionError, APITimeoutError, InternalServerError
from PIL import Image as PILImage

from docint.core.ingest.images_service import ImageIngestionService
from docint.core.readers.documents.ocr import VisionOCREngine


def _http_500() -> InternalServerError:
    """Build the error the OpenAI SDK raises for an upstream HTTP 500.

    Returns:
        InternalServerError: The SDK's wrapper around a 500 response.
    """
    request = httpx.Request("POST", "http://ollama:11434/v1/chat/completions")
    response = httpx.Response(500, request=request, json={"error": "Internal Server Error (ref: test)"})
    return InternalServerError("Error code: 500", response=response, body=None)


def _timeout() -> APITimeoutError:
    """Build the error the SDK raises when nothing came back in time.

    Returns:
        APITimeoutError: The SDK's timeout error.
    """
    return APITimeoutError(request=httpx.Request("POST", "http://ollama:11434/v1/chat/completions"))


def _connection_error() -> APIConnectionError:
    """Build the error the SDK raises when the endpoint is unreachable.

    Returns:
        APIConnectionError: The SDK's connection error.
    """
    return APIConnectionError(request=httpx.Request("POST", "http://ollama:11434/v1/chat/completions"))


def _build_engine() -> VisionOCREngine:
    """Build a VisionOCREngine over a stubbed PDF and client.

    Returns:
        VisionOCREngine: An engine whose pages render to a small blank image.
    """
    page = MagicMock()
    bitmap = MagicMock()
    bitmap.to_pil.return_value = PILImage.new("RGB", (200, 200), color="white")
    page.render.return_value = bitmap
    pdf = MagicMock()
    pdf.__getitem__ = MagicMock(return_value=page)

    with (
        patch("docint.core.readers.documents.ocr.pypdfium2") as mock_pdfium,
        patch("docint.core.readers.documents.ocr.OpenAIPipeline") as MockPipeline,
        patch("docint.core.readers.documents.ocr._OpenAI"),
        patch("docint.core.readers.documents.ocr.load_openai_env"),
    ):
        mock_pdfium.PdfDocument.return_value = pdf
        pipeline_instance = MagicMock()
        pipeline_instance.load_prompt.return_value = "Extract text"
        pipeline_instance.seed = 42
        pipeline_instance.temperature = 0.0
        pipeline_instance.top_p = 0.0
        pipeline_instance.reasoning_effort = None
        MockPipeline.return_value = pipeline_instance
        return VisionOCREngine("/fake/doc.pdf", timeout=30.0, max_retries=0, max_image_dimension=256, max_tokens=512)


def _run_pages(engine: VisionOCREngine, error: Exception, pages: int) -> MagicMock:
    """Fail ``pages`` consecutive pages with ``error``.

    Args:
        engine (VisionOCREngine): The engine under test.
        error (Exception): The error every vision call raises.
        pages (int): How many pages to attempt.

    Returns:
        MagicMock: The patched ``create`` call, for call-count assertions.
    """
    with (
        patch("docint.core.readers.documents.ocr.load_model_env"),
        patch("docint.core.readers.documents.ocr.time.sleep"),
        patch.object(engine._vision_client.chat.completions, "create", side_effect=error) as create,
    ):
        for index in range(pages):
            engine.ocr_page(index)
        return create


def test_a_burst_of_upstream_errors_does_not_disable_the_document() -> None:
    """An endpoint that answers -- even with a 500 -- is not an absent endpoint.

    The failure budget exists to stop spending a full timeout per page on an
    endpoint that never answers. A fast 500 costs ~1s and recovers within
    seconds, so it must cost its own page and nothing more.
    """
    engine = _build_engine()

    _run_pages(engine, _http_500(), pages=4)

    assert engine._disabled is False


def test_the_endpoint_is_still_called_after_a_burst_of_upstream_errors() -> None:
    """The page after a burst gets its chance; the burst is usually over."""
    engine = _build_engine()
    _run_pages(engine, _http_500(), pages=3)

    with (
        patch("docint.core.readers.documents.ocr.load_model_env"),
        patch("docint.core.readers.documents.ocr.time.sleep"),
        patch.object(engine._vision_client.chat.completions, "create") as create,
    ):
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "Recovered text"
        create.return_value = response
        spans = engine.ocr_page(3)

    assert [span.text for span in spans] == ["Recovered text"]


@pytest.mark.parametrize("error_factory", [_timeout, _connection_error])
def test_an_endpoint_that_never_answers_still_disables_the_document(error_factory: Any) -> None:
    """The original guarantee holds: don't bleed a timeout per page on a dead endpoint."""
    engine = _build_engine()

    _run_pages(engine, error_factory(), pages=3)

    assert engine._disabled is True


def test_a_page_the_endpoint_rejected_yields_no_text() -> None:
    """Not disabling the document does not mean pretending the page worked."""
    engine = _build_engine()

    with (
        patch("docint.core.readers.documents.ocr.load_model_env"),
        patch("docint.core.readers.documents.ocr.time.sleep"),
        patch.object(engine._vision_client.chat.completions, "create", side_effect=_http_500()),
    ):
        spans = engine.ocr_page(0)

    assert spans == []


def test_pages_without_ocr_text_are_counted() -> None:
    """The pipeline summary reported pages_failed=0 while 22 pages went unread."""
    engine = _build_engine()

    _run_pages(engine, _http_500(), pages=3)

    assert engine.ocr_stats.pages_failed == 3
    assert engine.ocr_stats.pages_skipped == 0


def test_pages_skipped_by_the_failure_budget_are_counted_separately() -> None:
    """Skipping because we gave up is a different fact from the page failing."""
    engine = _build_engine()
    _run_pages(engine, _timeout(), pages=3)

    with patch("docint.core.readers.documents.ocr.load_model_env"):
        engine.ocr_page(3)
        engine.ocr_page(4)

    assert engine._disabled is True
    assert engine.ocr_stats.pages_skipped == 2


def test_the_half_resolution_retry_waits_for_the_burst_to_pass() -> None:
    """Retrying inside the same second lands inside the same burst.

    Both attempts for a page were observed ~0.9s apart, so the retry was still
    inside the ~4s cluster of upstream 500s that had just rejected the first.
    """
    engine = _build_engine()

    with (
        patch("docint.core.readers.documents.ocr.load_model_env"),
        patch("docint.core.readers.documents.ocr.time.sleep") as sleep,
        patch.object(engine._vision_client.chat.completions, "create", side_effect=_http_500()),
    ):
        engine.ocr_page(0)

    assert sleep.call_count == 1
    assert sleep.call_args[0][0] > 0


def test_image_tagging_waits_between_attempts() -> None:
    """The image lane retried with no delay at all; it only survived by luck."""
    calls: list[int] = []

    def _always_fails() -> str:
        """Fail every time.

        Raises:
            RuntimeError: Always.
        """
        calls.append(1)
        raise RuntimeError("Vision inference failed: Error code: 500")

    with patch("docint.core.ingest.images_service.time.sleep") as sleep:
        with pytest.raises(RuntimeError):
            ImageIngestionService._run_with_retries(_always_fails, attempts=2)

    assert len(calls) == 2
    assert sleep.call_count == 1
    assert sleep.call_args[0][0] > 0


def test_the_manifest_reports_unread_ocr_pages() -> None:
    """``pages_ocr`` counts pages that needed OCR, not pages that got it.

    The observed run logged ``pages_total=30 pages_ocr=25 pages_failed=0``
    while only 3 pages had produced any text, so nothing in the summary
    revealed that 22 pages went unread.
    """
    from docint.core.readers.documents.models import DocumentManifest

    manifest = DocumentManifest(doc_id="d", file_path="/f.pdf", file_name="f.pdf", pipeline_version="1")

    assert manifest.pages_ocr_failed == 0
    assert manifest.pages_ocr_skipped == 0


def test_image_tagging_does_not_wait_after_the_last_attempt() -> None:
    """A delay after the final attempt buys nothing but latency."""
    with patch("docint.core.ingest.images_service.time.sleep") as sleep:
        result = ImageIngestionService._run_with_retries(lambda: "ok", attempts=2)

    assert result == "ok"
    assert sleep.call_count == 0
