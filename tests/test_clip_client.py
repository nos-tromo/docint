"""Tests for the remote CLIP HTTP client."""

from __future__ import annotations

import base64
import json
from collections.abc import Iterator

import httpx
import pytest

from docint.utils.clip_client import RemoteCLIPBackend


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Clear CLIP_* env vars so each test starts from a known baseline."""
    for key in ("CLIP_API_BASE", "CLIP_API_KEY", "CLIP_TIMEOUT"):
        monkeypatch.delenv(key, raising=False)
    yield


def _install_mock_transport(
    monkeypatch: pytest.MonkeyPatch,
    handler: httpx.MockTransport,
) -> None:
    """Force ``clip_client._build_client`` to use ``handler`` as the transport."""
    original_client = httpx.Client

    def _patched_client(*args: object, **kwargs: object) -> httpx.Client:
        kwargs["transport"] = handler
        return original_client(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("docint.utils.clip_client.httpx.Client", _patched_client)


def test_remote_clip_backend_probes_dimension_and_embeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy path: dimension probe at construction, embed_image + embed_text round-trip."""
    captured: list[dict[str, object]] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        record: dict[str, object] = {
            "method": request.method,
            "url": str(request.url),
            "auth": request.headers.get("authorization"),
        }
        if request.method == "POST":
            record["body"] = json.loads(request.content)
        captured.append(record)
        if request.url.path == "/clip/dimension":
            return httpx.Response(200, json={"dimension": 512})
        if request.url.path == "/clip/embed_image":
            return httpx.Response(200, json={"embedding": [0.1, 0.2, 0.3], "dimension": 512})
        if request.url.path == "/clip/embed_text":
            return httpx.Response(200, json={"embedding": [0.4, 0.5, 0.6], "dimension": 512})
        return httpx.Response(404)

    _install_mock_transport(monkeypatch, httpx.MockTransport(_handler))
    monkeypatch.setenv("CLIP_API_BASE", "http://clip-embed:8000")

    backend = RemoteCLIPBackend()

    assert backend.dimension == 512
    assert backend.embed(b"\x89PNG-fake-bytes") == [0.1, 0.2, 0.3]
    assert backend.embed_text("a photo of a cat") == [0.4, 0.5, 0.6]

    paths = [str(r["url"]) for r in captured]
    assert paths == [
        "http://clip-embed:8000/clip/dimension",
        "http://clip-embed:8000/clip/embed_image",
        "http://clip-embed:8000/clip/embed_text",
    ]
    # No CLIP_API_KEY -> no Authorization header (clip-only-shape posture).
    assert all(r["auth"] is None for r in captured)

    image_post = next(r for r in captured if str(r["url"]).endswith("/clip/embed_image"))
    image_body = image_post["body"]
    assert isinstance(image_body, dict)
    assert base64.b64decode(image_body["image_b64"]) == b"\x89PNG-fake-bytes"

    text_post = next(r for r in captured if str(r["url"]).endswith("/clip/embed_text"))
    assert text_post["body"] == {"text": "a photo of a cat"}


def test_remote_clip_backend_sends_bearer_when_api_key_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full vllm-service shape: CLIP_API_KEY produces a Bearer header."""
    auth_headers: list[str | None] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        auth_headers.append(request.headers.get("authorization"))
        if request.url.path == "/clip/dimension":
            return httpx.Response(200, json={"dimension": 512})
        return httpx.Response(200, json={"embedding": [0.0], "dimension": 512})

    _install_mock_transport(monkeypatch, httpx.MockTransport(_handler))
    monkeypatch.setenv("CLIP_API_BASE", "http://vllm-router:4000")
    monkeypatch.setenv("CLIP_API_KEY", "sk-token")

    backend = RemoteCLIPBackend()
    backend.embed(b"\x00")
    backend.embed_text("hi")

    assert all(header == "Bearer sk-token" for header in auth_headers)


def test_remote_clip_backend_raises_when_dimension_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construction-time probe failures surface so the caller can decide."""

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "starting up"})

    _install_mock_transport(monkeypatch, httpx.MockTransport(_handler))
    monkeypatch.setenv("CLIP_API_BASE", "http://clip-embed:8000")

    with pytest.raises(httpx.HTTPStatusError):
        RemoteCLIPBackend()


def test_remote_clip_backend_raises_on_malformed_embed_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Server-side payload corruption raises ValueError, not silent zeros."""

    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/clip/dimension":
            return httpx.Response(200, json={"dimension": 4})
        return httpx.Response(200, json={"oops": "no embedding key"})

    _install_mock_transport(monkeypatch, httpx.MockTransport(_handler))
    monkeypatch.setenv("CLIP_API_BASE", "http://clip-embed:8000")

    backend = RemoteCLIPBackend()
    with pytest.raises(ValueError, match="returned no embedding"):
        backend.embed(b"x")
    with pytest.raises(ValueError, match="returned no embedding"):
        backend.embed_text("hi")


def test_remote_clip_backend_respects_explicit_cfg_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Passing a ``CLIPClientConfig`` skips env loading."""
    from docint.utils.env_cfg import CLIPClientConfig

    captured_urls: list[str] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        captured_urls.append(str(request.url))
        if request.url.path == "/clip/dimension":
            return httpx.Response(200, json={"dimension": 7})
        return httpx.Response(200, json={"embedding": [0.0] * 7, "dimension": 7})

    _install_mock_transport(monkeypatch, httpx.MockTransport(_handler))
    monkeypatch.setenv("CLIP_API_BASE", "http://env-should-be-ignored:9999")

    cfg = CLIPClientConfig(api_base="http://explicit:7000", api_key=None, timeout=5.0)
    backend = RemoteCLIPBackend(cfg=cfg)

    assert backend.dimension == 7
    assert all(url.startswith("http://explicit:7000") for url in captured_urls)
