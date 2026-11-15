"""Tests for WorkbenchClientConfig and the load_workbench_env loader."""

from __future__ import annotations

import pytest

from docint.utils.env_cfg import load_workbench_env

KEY = "k" * 32
_VARS = ("WORKBENCH_API_KEY", "WORKBENCH_API_BASE", "WORKBENCH_TIMEOUT")


@pytest.fixture(autouse=True)
def _clean_workbench_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear every workbench variable so an ambient ``.env`` cannot leak in."""
    for name in _VARS:
        monkeypatch.delenv(name, raising=False)


def test_workbench_is_off_without_a_key() -> None:
    """Nothing set means no key, no base, the default timeout and a disabled tool."""
    cfg = load_workbench_env()
    assert (cfg.api_key, cfg.api_base, cfg.timeout, cfg.enabled) == (None, "", 75.0, False)


def test_a_key_shorter_than_32_characters_stops_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    """A short key raises, naming the variable and never its value."""
    monkeypatch.setenv("WORKBENCH_API_KEY", "k" * 31)
    with pytest.raises(ValueError, match="WORKBENCH_API_KEY") as caught:
        load_workbench_env()
    assert "k" * 31 not in str(caught.value)


def test_a_key_and_a_base_switch_the_tool_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """A key plus a base enables the tool, and the base loses its trailing slash."""
    monkeypatch.setenv("WORKBENCH_API_KEY", KEY)
    monkeypatch.setenv("WORKBENCH_API_BASE", "http://workbench-api:8000/")
    cfg = load_workbench_env()
    assert cfg.api_key == KEY
    assert cfg.api_base == "http://workbench-api:8000"
    assert cfg.enabled is True


def test_a_blank_key_counts_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """A whitespace-only key is no key, and does not trip the length check."""
    monkeypatch.setenv("WORKBENCH_API_KEY", "   ")
    monkeypatch.setenv("WORKBENCH_API_BASE", "http://workbench-api:8000")
    cfg = load_workbench_env()
    assert cfg.api_key is None
    assert cfg.enabled is False


def test_a_key_without_a_base_offers_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The key alone is not enough: with no base the tool stays off."""
    monkeypatch.setenv("WORKBENCH_API_KEY", KEY)
    cfg = load_workbench_env()
    assert cfg.api_key == KEY
    assert cfg.api_base == ""
    assert cfg.enabled is False


def test_a_base_without_a_key_offers_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A base alone is not enough either."""
    monkeypatch.setenv("WORKBENCH_API_BASE", "http://workbench-api:8000")
    assert load_workbench_env().enabled is False


def test_the_default_base_is_used_when_the_variable_is_unset() -> None:
    """The caller's default base applies only while the variable is unset."""
    cfg = load_workbench_env(default_api_base="http://workbench-api:8000/", default_timeout=30.0)
    assert (cfg.api_base, cfg.timeout) == ("http://workbench-api:8000", 30.0)


def test_a_blank_base_counts_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """A blank base falls back to the default, like a blank key counts as unset."""
    monkeypatch.setenv("WORKBENCH_API_BASE", "  ")
    assert load_workbench_env(default_api_base="http://workbench-api:8000").api_base == "http://workbench-api:8000"


def test_the_key_stays_out_of_the_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    """Logging the config can never print the key."""
    monkeypatch.setenv("WORKBENCH_API_KEY", KEY)
    assert KEY not in repr(load_workbench_env())


def test_a_timeout_is_read_as_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    """A valid timeout is returned as a float."""
    monkeypatch.setenv("WORKBENCH_TIMEOUT", "45")
    assert load_workbench_env().timeout == 45.0


def test_a_blank_timeout_counts_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """A blank timeout falls back to the default instead of failing."""
    monkeypatch.setenv("WORKBENCH_TIMEOUT", "  ")
    assert load_workbench_env().timeout == 75.0


@pytest.mark.parametrize("raw", ["0", "-5", "nan", "inf", "soon"])
def test_the_timeout_must_be_a_positive_finite_number(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    """A zero, negative, non-finite or non-numeric timeout stops startup."""
    monkeypatch.setenv("WORKBENCH_TIMEOUT", raw)
    with pytest.raises(ValueError, match="WORKBENCH_TIMEOUT") as caught:
        load_workbench_env()
    assert raw not in str(caught.value)
