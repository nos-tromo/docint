"""Tests for IngestionConfig.media_filetypes and its DEFAULT_MEDIA_FILETYPES default."""

from __future__ import annotations

import pytest

from docint.utils.env_cfg import DEFAULT_MEDIA_FILETYPES, load_ingestion_env


def test_default_media_filetypes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default media_filetypes covers common A/V extensions and excludes supported_filetypes."""
    monkeypatch.delenv("MEDIA_FILETYPES", raising=False)
    cfg = load_ingestion_env()
    assert ".mp4" in cfg.media_filetypes
    assert ".mp3" in cfg.media_filetypes
    assert ".wav" in cfg.media_filetypes
    assert cfg.media_filetypes == DEFAULT_MEDIA_FILETYPES
    # A/V must NOT be in the generic reader whitelist (they route via the pre-pass).
    assert ".mp4" not in cfg.supported_filetypes


def test_media_filetypes_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """MEDIA_FILETYPES env var overrides the default with a comma-separated list."""
    monkeypatch.setenv("MEDIA_FILETYPES", ".mp4, .mov")
    cfg = load_ingestion_env()
    assert cfg.media_filetypes == [".mp4", ".mov"]


def test_media_filetypes_override_normalizes_case_dot_and_empties(monkeypatch: pytest.MonkeyPatch) -> None:
    """Override entries are lowercased, empty segments dropped, and a leading dot ensured.

    A dotless entry like ``mp4`` would otherwise never match ``Path.suffix`` (which
    always carries the dot), silently skipping the operator's media — so normalize it.
    """
    monkeypatch.setenv("MEDIA_FILETYPES", "MP4, .mov, ,mkv")
    cfg = load_ingestion_env()
    assert cfg.media_filetypes == [".mp4", ".mov", ".mkv"]
