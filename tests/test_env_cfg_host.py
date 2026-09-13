"""``load_host_env()`` — the ``QDRANT_API_KEY`` contract.

Unset and empty mean "no key" so existing deployments (data-plane without
``QDRANT__SERVICE__API_KEY``) are unaffected; a set value is passed through
verbatim apart from surrounding whitespace.
"""

from __future__ import annotations

import pytest

from docint.utils.env_cfg import load_host_env


def test_qdrant_api_key_defaults_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that qdrant_api_key defaults to None when unset."""
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    assert load_host_env().qdrant_api_key is None


def test_qdrant_api_key_empty_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that qdrant_api_key is None when set to whitespace only."""
    monkeypatch.setenv("QDRANT_API_KEY", "   ")
    assert load_host_env().qdrant_api_key is None


def test_qdrant_api_key_is_read_and_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that qdrant_api_key is read and whitespace is stripped."""
    monkeypatch.setenv("QDRANT_API_KEY", " change-me-qdrant-key ")
    assert load_host_env().qdrant_api_key == "change-me-qdrant-key"


def test_qdrant_host_unaffected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that qdrant_host is unaffected by qdrant_api_key changes."""
    monkeypatch.setenv("QDRANT_HOST", "http://qdrant:6333")
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    cfg = load_host_env()
    assert cfg.qdrant_host == "http://qdrant:6333"
    assert cfg.qdrant_api_key is None
