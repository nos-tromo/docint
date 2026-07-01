"""Tests for NextextConfig and load_nextext_env loader."""

from __future__ import annotations

import pytest

from docint.utils.env_cfg import load_nextext_env


def test_nextext_disabled_when_base_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nextext client is disabled when NEXTEXT_API_BASE is unset."""
    monkeypatch.delenv("NEXTEXT_API_BASE", raising=False)
    cfg = load_nextext_env()
    assert cfg.enabled is False
    assert cfg.keyframes_per_minute == 4
    assert cfg.keyframes_max == 20
    assert cfg.keyframe_dedup_cosine == 0.95


def test_nextext_enabled_and_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nextext client is enabled and overrides work correctly."""
    monkeypatch.setenv("NEXTEXT_API_BASE", "http://nextext:8000/")
    monkeypatch.setenv("KEYFRAMES_PER_MINUTE", "6")
    cfg = load_nextext_env()
    assert cfg.enabled is True
    assert cfg.api_base == "http://nextext:8000"
    assert cfg.keyframes_per_minute == 6


def test_nextext_rejects_out_of_range_cosine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """KEYFRAME_DEDUP_COSINE outside [0, 1] raises ValueError."""
    monkeypatch.setenv("KEYFRAME_DEDUP_COSINE", "1.5")
    with pytest.raises(ValueError):
        load_nextext_env()
