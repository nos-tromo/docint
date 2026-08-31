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
