"""Tests for ingestion-related environment configuration."""

from __future__ import annotations

import pytest

from docint.utils.env_cfg import load_ingestion_env


def test_pipeline_overlap_defaults_to_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enrichment overlaps persistence unless an operator turns it off."""
    monkeypatch.delenv("INGEST_PIPELINE_OVERLAP_ENABLED", raising=False)

    assert load_ingestion_env().ingest_pipeline_overlap_enabled is True


def test_pipeline_overlap_switches_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """The inline path is still reachable for a deployment that needs it."""
    monkeypatch.setenv("INGEST_PIPELINE_OVERLAP_ENABLED", "false")

    assert load_ingestion_env().ingest_pipeline_overlap_enabled is False


def test_preprocess_workers_defaults_and_floors(monkeypatch: pytest.MonkeyPatch) -> None:
    """The per-file pool is bounded, defaults to 4, and never drops below one worker."""
    monkeypatch.delenv("INGEST_PREPROCESS_WORKERS", raising=False)
    assert load_ingestion_env().ingest_preprocess_workers == 4

    monkeypatch.setenv("INGEST_PREPROCESS_WORKERS", "8")
    assert load_ingestion_env().ingest_preprocess_workers == 8

    monkeypatch.setenv("INGEST_PREPROCESS_WORKERS", "0")
    assert load_ingestion_env().ingest_preprocess_workers == 1


def test_preprocess_on_upload_defaults_to_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Uploaded files start their heavy stages on arrival unless switched off."""
    monkeypatch.delenv("INGEST_PREPROCESS_ON_UPLOAD", raising=False)
    assert load_ingestion_env().ingest_preprocess_on_upload is True

    monkeypatch.setenv("INGEST_PREPROCESS_ON_UPLOAD", "false")
    assert load_ingestion_env().ingest_preprocess_on_upload is False
