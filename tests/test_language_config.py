"""Tests for the ``RESPONSE_LANGUAGE`` env var and ``LanguageConfig``."""

from __future__ import annotations

import pytest

from docint.utils.env_cfg import LanguageConfig, load_language_env


def test_load_language_env_defaults_to_english(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``RESPONSE_LANGUAGE`` is unset, the default is ``'en'``.

    English is the back-compat default so existing deployments that don't
    update their ``.env`` keep behaving as before.
    """
    monkeypatch.delenv("RESPONSE_LANGUAGE", raising=False)

    assert load_language_env() == LanguageConfig(code="en")


def test_load_language_env_recognises_de(monkeypatch: pytest.MonkeyPatch) -> None:
    """``RESPONSE_LANGUAGE=de`` switches the config to German."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "de")

    assert load_language_env().code == "de"


def test_load_language_env_is_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    """``DE``/``De``/``de`` must all resolve to ``'de'``."""
    for raw in ("DE", "De", "de", " de "):
        monkeypatch.setenv("RESPONSE_LANGUAGE", raw)
        assert load_language_env().code == "de", f"failed for raw value {raw!r}"


def test_load_language_env_unknown_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown values fall back silently so a typo cannot break startup."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "klingon")

    assert load_language_env().code == "en"
    assert load_language_env(default="de").code == "de"


def test_load_language_env_default_arg_used_only_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit env value wins over the ``default`` argument."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "de")

    assert load_language_env(default="en").code == "de"
