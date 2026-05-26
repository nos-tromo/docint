"""Tests for the remote GLiNER NER HTTP client."""

import json
from collections.abc import Iterator

import httpx
import pytest

from docint.utils.env_cfg import NERClientConfig
from docint.utils.ner_client import (
    DEFAULT_NER_LABELS,
    build_remote_ner_extractor,
)


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Clear NER_* env vars so each test starts from a known baseline."""
    for key in ("NER_API_BASE", "NER_API_KEY", "NER_THRESHOLD", "NER_TIMEOUT"):
        monkeypatch.delenv(key, raising=False)
    yield


def _install_mock_transport(
    monkeypatch: pytest.MonkeyPatch,
    handler: httpx.MockTransport,
) -> None:
    """Force ner_client._build_client to use ``handler`` as the transport."""
    original_client = httpx.Client

    def _patched_client(*args: object, **kwargs: object) -> httpx.Client:
        kwargs["transport"] = handler
        return original_client(*args, **kwargs)  # type: ignore[arg-type]

    # Patch via the attribute path string so mypy doesn't require the
    # ner_client module to re-export ``httpx`` explicitly.
    monkeypatch.setattr("docint.utils.ner_client.httpx.Client", _patched_client)


def test_remote_ner_extractor_returns_mapped_entities(monkeypatch: pytest.MonkeyPatch) -> None:
    """Success path: server entities map to docint's {text,type,score} shape."""
    captured: dict[str, object] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json={
                "entities": [
                    {"start": 0, "end": 5, "text": "Alice", "label": "person", "score": 0.97},
                    {"start": 15, "end": 24, "text": "Acme", "label": "org", "score": 0.92},
                ]
            },
        )

    _install_mock_transport(monkeypatch, httpx.MockTransport(_handler))
    monkeypatch.setenv("NER_API_BASE", "http://gliner-ner:8000")

    extract = build_remote_ner_extractor()
    entities, relations = extract("Alice works at Acme.")

    assert entities == [
        {"text": "Alice", "type": "person", "score": 0.97},
        {"text": "Acme", "type": "org", "score": 0.92},
    ]
    assert relations == []
    assert captured["url"] == "http://gliner-ner:8000/gliner"
    assert captured["body"] == {
        "text": "Alice works at Acme.",
        "labels": DEFAULT_NER_LABELS,
        "threshold": 0.3,
    }
    # No NER_API_KEY set -> no Authorization header (ner-only-shape posture).
    assert captured["auth"] is None


def test_remote_ner_extractor_sends_bearer_when_api_key_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Full vllm-service shape: NER_API_KEY produces a Bearer header."""
    captured: dict[str, object] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"entities": []})

    _install_mock_transport(monkeypatch, httpx.MockTransport(_handler))
    monkeypatch.setenv("NER_API_BASE", "http://vllm-router:4000")
    monkeypatch.setenv("NER_API_KEY", "sk-test-key")

    extract = build_remote_ner_extractor()
    extract("any text")

    assert captured["auth"] == "Bearer sk-test-key"


def test_remote_ner_extractor_empty_text_short_circuits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whitespace-only input returns ([], []) without calling the service."""
    call_count = 0

    def _handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(200, json={"entities": []})

    _install_mock_transport(monkeypatch, httpx.MockTransport(_handler))

    extract = build_remote_ner_extractor()
    entities, relations = extract("   \n\t  ")

    assert entities == []
    assert relations == []
    assert call_count == 0


def test_remote_ner_extractor_network_error_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any httpx error during the request fails closed with ([], [])."""

    def _handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    _install_mock_transport(monkeypatch, httpx.MockTransport(_handler))

    extract = build_remote_ner_extractor()
    entities, relations = extract("Alice met Bob.")

    assert entities == []
    assert relations == []


def test_remote_ner_extractor_http_5xx_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 5xx from the upstream also fails closed."""

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="upstream unavailable")

    _install_mock_transport(monkeypatch, httpx.MockTransport(_handler))

    extract = build_remote_ner_extractor()
    entities, relations = extract("Alice met Bob.")

    assert entities == []
    assert relations == []


def test_remote_ner_extractor_malformed_payload_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing or non-list ``entities`` key produces ([], [])."""

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    _install_mock_transport(monkeypatch, httpx.MockTransport(_handler))

    extract = build_remote_ner_extractor()
    entities, relations = extract("Alice met Bob.")

    assert entities == []
    assert relations == []


def test_remote_ner_extractor_drops_entries_missing_text_or_label(monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-entity validation skips items without both ``text`` and ``label``."""

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "entities": [
                    {"text": "Alice", "label": "person", "score": 0.9},
                    {"text": "", "label": "person", "score": 0.5},
                    {"text": "Berlin", "label": ""},
                    "not-a-dict",
                ]
            },
        )

    _install_mock_transport(monkeypatch, httpx.MockTransport(_handler))

    extract = build_remote_ner_extractor()
    entities, _ = extract("Alice in Berlin")

    assert entities == [{"text": "Alice", "type": "person", "score": 0.9}]


def test_remote_ner_extractor_uses_explicit_cfg(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit NERClientConfig overrides the environment-derived one."""
    captured: dict[str, object] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"entities": []})

    _install_mock_transport(monkeypatch, httpx.MockTransport(_handler))
    monkeypatch.setenv("NER_API_BASE", "http://wrong-host:1234")

    cfg = NERClientConfig(
        api_base="http://explicit-host:9000",
        api_key=None,
        threshold=0.7,
        timeout=5.0,
    )
    extract = build_remote_ner_extractor(labels=["person"], cfg=cfg)
    extract("test")

    assert captured["url"] == "http://explicit-host:9000/gliner"
    assert captured["body"] == {"text": "test", "labels": ["person"], "threshold": 0.7}
