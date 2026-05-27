"""Tests for the locale-aware ``ui_string`` helper."""

from __future__ import annotations

import pytest

from docint.utils.ui_strings import UI_STRINGS, ui_string


def test_ui_string_returns_english_when_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ``RESPONSE_LANGUAGE`` set, ``ui_string`` returns the English entry."""
    monkeypatch.delenv("RESPONSE_LANGUAGE", raising=False)

    assert ui_string("clarify_generic") == "Could you clarify what you need?"


def test_ui_string_returns_german_when_de(monkeypatch: pytest.MonkeyPatch) -> None:
    """With ``RESPONSE_LANGUAGE=de`` the German variant is returned."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "de")

    assert ui_string("clarify_generic") == "Können Sie präzisieren, was Sie benötigen?"


def test_ui_string_table_is_complete_across_languages() -> None:
    """Every language must define the exact same key set as English."""
    english_keys = set(UI_STRINGS["en"])

    for lang, strings in UI_STRINGS.items():
        assert set(strings) == english_keys, f"language '{lang}' has divergent UI keys"


def test_ui_string_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown key surfaces as ``KeyError`` rather than silently returning empty."""
    monkeypatch.delenv("RESPONSE_LANGUAGE", raising=False)

    with pytest.raises(KeyError):
        ui_string("not_a_real_key")
