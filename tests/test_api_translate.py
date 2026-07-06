"""Tests for the ``POST /translate`` endpoint.

Mirrors the ``TestClient(api.app)`` + default-identity header pattern used by
``tests/test_version.py`` -- the route touches neither ``rag`` nor Qdrant, so
no additional fixtures are needed beyond patching the module-level
``translate`` seam.
"""

import pytest
from fastapi.testclient import TestClient

import docint.core.api as api
from docint.utils.translate_client import TranslateResult

client = TestClient(api.app)
AUTH = {"X-Auth-User": "alice"}


def test_translate_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful translate call returns 200 with the translated text."""
    monkeypatch.setattr(
        api,
        "translate",
        lambda text, **kw: TranslateResult(True, "Hallo", "m", "de", None),
    )
    r = client.post("/translate", json={"text": "Hello"}, headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["translation"] == "Hallo"
    assert body["target_lang"] == "de"
    assert body["model"] == "m"
    assert body["error"] is None


def test_translate_failsoft_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed translate call still returns 200, with ok=False and an error token."""
    monkeypatch.setattr(
        api,
        "translate",
        lambda text, **kw: TranslateResult(False, None, "m", "de", "unavailable"),
    )
    r = client.post("/translate", json={"text": "Hello"}, headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["error"] == "unavailable"
    assert body["translation"] is None
    assert body["model"] == "m"


def test_translate_requires_principal(monkeypatch: pytest.MonkeyPatch) -> None:
    """POST /translate 401s with no trusted header and no configured default identity."""
    monkeypatch.delenv("DOCINT_AUTH_HEADER", raising=False)
    monkeypatch.delenv("DOCINT_DEFAULT_IDENTITY", raising=False)

    r = client.post("/translate", json={"text": "Hello"})
    assert r.status_code == 401
