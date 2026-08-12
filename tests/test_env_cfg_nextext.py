"""Tests for NextextConfig and load_nextext_env loader."""

from __future__ import annotations

import pytest

from docint.utils.env_cfg import load_nextext_env


def test_nextext_disabled_when_base_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nextext client is disabled and config falls back to built-in defaults.

    Clears every env var whose default is asserted below so an ambient value
    (e.g. a developer ``.env`` loaded by ``load_dotenv`` at import) cannot make
    the test flap.
    """
    for var in (
        "NEXTEXT_API_BASE",
        "NEXTEXT_MAX_CONCURRENCY",
        "KEYFRAMES_PER_MINUTE",
        "KEYFRAMES_MAX",
        "KEYFRAME_DEDUP_COSINE",
    ):
        monkeypatch.delenv(var, raising=False)
    cfg = load_nextext_env()
    assert cfg.enabled is False
    assert cfg.keyframes_per_minute == 4
    assert cfg.keyframes_max == 20
    assert cfg.keyframe_dedup_cosine == 0.95
    assert cfg.nextext_max_concurrency == 4


def test_nextext_enabled_and_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nextext client is enabled and overrides work correctly."""
    monkeypatch.setenv("NEXTEXT_API_BASE", "http://nextext:8000/")
    monkeypatch.setenv("KEYFRAMES_PER_MINUTE", "6")
    monkeypatch.setenv("NEXTEXT_MAX_CONCURRENCY", "8")
    cfg = load_nextext_env()
    assert cfg.enabled is True
    assert cfg.api_base == "http://nextext:8000"
    assert cfg.keyframes_per_minute == 6
    assert cfg.nextext_max_concurrency == 8


def test_nextext_rejects_out_of_range_cosine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """KEYFRAME_DEDUP_COSINE outside [0, 1] raises ValueError."""
    monkeypatch.setenv("KEYFRAME_DEDUP_COSINE", "1.5")
    with pytest.raises(ValueError):
        load_nextext_env()


def test_nextext_max_concurrency_floors_at_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero/negative NEXTEXT_MAX_CONCURRENCY is clamped up to 1.

    Args:
        monkeypatch: Fixture to override environment variables.
    """
    monkeypatch.setenv("NEXTEXT_MAX_CONCURRENCY", "0")
    cfg = load_nextext_env()
    assert cfg.nextext_max_concurrency == 1


def test_nextext_identity_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """The trusted-header identity defaults to X-Auth-User / docint."""
    for var in ("NEXTEXT_AUTH_HEADER", "NEXTEXT_IDENTITY"):
        monkeypatch.delenv(var, raising=False)
    cfg = load_nextext_env()
    assert cfg.auth_header == "X-Auth-User"
    assert cfg.identity == "docint"


def test_nextext_identity_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """NEXTEXT_AUTH_HEADER / NEXTEXT_IDENTITY override the identity docint sends."""
    monkeypatch.setenv("NEXTEXT_AUTH_HEADER", "X-Custom-User")
    monkeypatch.setenv("NEXTEXT_IDENTITY", "svc-docint")
    cfg = load_nextext_env()
    assert cfg.auth_header == "X-Custom-User"
    assert cfg.identity == "svc-docint"


def test_nextext_identity_empty_suppresses(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty NEXTEXT_IDENTITY resolves to '' (client sends no identity header)."""
    monkeypatch.setenv("NEXTEXT_IDENTITY", "  ")
    cfg = load_nextext_env()
    assert cfg.identity == ""
