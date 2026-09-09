"""Tests for ingestion-related environment configuration."""

from __future__ import annotations

import pytest

from docint.utils.env_cfg import load_ingestion_env


def test_timestamp_link_defaults_to_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """The timestamp fallback is on unless an operator turns it off."""
    monkeypatch.delenv("SOCIAL_TIMESTAMP_LINK_ENABLED", raising=False)

    assert load_ingestion_env().social_timestamp_link_enabled is True


@pytest.mark.parametrize("value", ["false", "0", "no", "off", "FALSE"])
def test_timestamp_link_switches_off(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """Anything that is not an affirmative reading disables the fallback."""
    monkeypatch.setenv("SOCIAL_TIMESTAMP_LINK_ENABLED", value)

    assert load_ingestion_env().social_timestamp_link_enabled is False


@pytest.mark.parametrize("value", ["true", "1", "yes", "TRUE"])
def test_timestamp_link_accepts_the_usual_affirmatives(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """The loader reads the same affirmatives as its sibling social switches."""
    monkeypatch.setenv("SOCIAL_TIMESTAMP_LINK_ENABLED", value)

    assert load_ingestion_env().social_timestamp_link_enabled is True


def test_text_link_defaults_to_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """The text match is on unless an operator turns it off."""
    monkeypatch.delenv("SOCIAL_TEXT_LINK_ENABLED", raising=False)

    assert load_ingestion_env().social_text_link_enabled is True


@pytest.mark.parametrize("value", ["false", "0", "no", "off", "FALSE"])
def test_text_link_switches_off(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """Anything that is not an affirmative reading disables the text match."""
    monkeypatch.setenv("SOCIAL_TEXT_LINK_ENABLED", value)

    assert load_ingestion_env().social_text_link_enabled is False


@pytest.mark.parametrize("value", ["true", "1", "yes", "TRUE"])
def test_text_link_accepts_the_usual_affirmatives(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """The loader reads the same affirmatives as its sibling social switches."""
    monkeypatch.setenv("SOCIAL_TEXT_LINK_ENABLED", value)

    assert load_ingestion_env().social_text_link_enabled is True


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
