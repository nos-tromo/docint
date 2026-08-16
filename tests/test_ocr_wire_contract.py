"""Tests the OCR engine against a real HTTP server, not a mocked client.

Every other OCR test patches ``_OpenAI`` away. A mocked client can still be
asked what arguments it received, so the message shape is covered there — but
it never builds a URL, never sends a header and never serializes anything, so
the path the SDK appends to the configured base, the Authorization header and
the JSON round trip are invisible to it. That is where a live-only fault
hides: a base URL missing its ``/v1`` passes every mocked test and 404s on the
first real call, the way ``EMBED_API_BASE`` once did.

The server here speaks the OpenAI chat-completions shape over localhost and
records what it was sent. No real endpoint, no real document, no network
beyond the loopback interface.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, ClassVar
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image as PILImage
from typing_extensions import override

from docint.core.ocr.engine import DocumentOcrEngine
from docint.core.ocr.families import OcrCategory

DOTS_ANSWER = json.dumps(
    [
        {"bbox": [10, 10, 380, 40], "category": "Title", "text": "Prüfbericht"},
        {"bbox": [10, 60, 380, 180], "category": "Text", "text": "Die Messung ergab keine Abweichung."},
    ]
)


class _Recorder(BaseHTTPRequestHandler):
    """Answers completions with a fixed layout and records the request."""

    requests: ClassVar[list[dict[str, Any]]] = []
    answer: ClassVar[str] = DOTS_ANSWER

    def do_POST(self) -> None:
        """Record the request body and answer with the canned layout."""
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        _Recorder.requests.append(
            {
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
                "body": body,
            }
        )
        raw = json.dumps({"choices": [{"message": {"content": _Recorder.answer}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    @override
    def log_message(self, format: str, *args: Any) -> None:
        """Keep the test output quiet."""


@pytest.fixture
def endpoint() -> Iterator[str]:
    """Serve the recorder on a loopback port for one test."""
    _Recorder.requests = []
    _Recorder.answer = DOTS_ANSWER
    server = HTTPServer(("127.0.0.1", 0), _Recorder)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _pipeline() -> MagicMock:
    """A stand-in OpenAIPipeline with fixed sampling settings."""
    pipeline = MagicMock()
    pipeline.load_prompt.side_effect = lambda kw: {"ocr": "Read the text.", "table_structure": "Return HTML."}[kw]
    pipeline.seed = 42
    pipeline.temperature = 0.0
    pipeline.top_p = 0.1
    pipeline.reasoning_effort = None
    return pipeline


def _engine(endpoint: str, model: str = "dots-studio/dots.mocr") -> DocumentOcrEngine:
    """Build an engine whose client really talks to *endpoint*."""
    with (
        patch("docint.core.ocr.engine.OpenAIPipeline", return_value=_pipeline()),
        patch("docint.core.ocr.engine.load_openai_env") as openai_env,
        patch("docint.core.ocr.engine.load_ocr_client_env") as ocr_env,
        patch("docint.core.ocr.engine.load_model_env") as model_env,
    ):
        openai_env.return_value.api_base = endpoint
        openai_env.return_value.api_key = "sk-test"
        openai_env.return_value.timeout = 30.0
        ocr_env.return_value.model = model
        ocr_env.return_value.api_base = endpoint
        ocr_env.return_value.api_key = "sk-test"
        ocr_env.return_value.timeout = 30.0
        model_env.return_value.vision_model = "vision/model"
        return DocumentOcrEngine(timeout=30.0, max_retries=0, max_tokens=4096)


def test_the_request_reaches_the_completions_path_with_its_key(endpoint: str) -> None:
    """The SDK appends /chat/completions to the configured base, key attached."""
    _engine(endpoint).read_image(PILImage.new("RGB", (400, 200), color="white"))

    assert len(_Recorder.requests) == 1
    assert _Recorder.requests[0]["path"] == "/v1/chat/completions"
    assert _Recorder.requests[0]["authorization"] == "Bearer sk-test"


def test_the_image_is_sent_before_the_instruction(endpoint: str) -> None:
    """The order the document models' own clients use, and some require."""
    _engine(endpoint).read_image(PILImage.new("RGB", (400, 200), color="white"))

    content = _Recorder.requests[0]["body"]["messages"][0]["content"]
    assert [part["type"] for part in content] == ["image_url", "text"]
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[1]["text"]


def test_the_configured_model_and_budget_go_out(endpoint: str) -> None:
    """A model id typo would otherwise surface only as an upstream 404."""
    _engine(endpoint).read_image(PILImage.new("RGB", (400, 200), color="white"))

    body = _Recorder.requests[0]["body"]
    assert body["model"] == "dots-studio/dots.mocr"
    assert body["max_tokens"] == 4096
    assert body["temperature"] == 0.0


def test_a_real_response_parses_into_blocks(endpoint: str) -> None:
    """The whole round trip: HTTP in, layout blocks out."""
    blocks = _engine(endpoint).read_image(PILImage.new("RGB", (400, 200), color="white"))

    assert [block.category for block in blocks] == [OcrCategory.TITLE, OcrCategory.TEXT]
    assert blocks[0].text == "Prüfbericht"


def test_a_text_only_model_gets_the_plain_reading_prompt(endpoint: str) -> None:
    """The family decides the instruction, and it goes out on the wire."""
    _Recorder.answer = "Prüfbericht\nDie Messung ergab keine Abweichung."
    blocks = _engine(endpoint, model="zai-org/GLM-OCR").read_image(PILImage.new("RGB", (400, 200), color="white"))

    content = _Recorder.requests[0]["body"]["messages"][0]["content"]
    assert content[1]["text"] == "Read the text."
    assert len(blocks) == 1
    assert blocks[0].text.startswith("Prüfbericht")
