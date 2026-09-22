"""Tests for RetentionConfig: the ``COLLECTION_RETENTION`` window."""

from __future__ import annotations

import pytest
from _pytest.logging import LogCaptureFixture

from docint.utils.env_cfg import load_retention_env


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear the variable so an ambient ``.env`` cannot skew the default."""
    monkeypatch.delenv("COLLECTION_RETENTION", raising=False)


def test_retention_is_off_by_default() -> None:
    """Nothing is ever deleted unless an operator opts in."""
    cfg = load_retention_env()
    assert cfg.window == "off"
    assert cfg.months == 0
    assert cfg.enabled is False


@pytest.mark.parametrize(("value", "months"), [("6m", 6), ("12m", 12), ("18m", 18), ("24m", 24)])
def test_each_window_maps_to_its_months(monkeypatch: pytest.MonkeyPatch, value: str, months: int) -> None:
    """The four supported windows are calendar months."""
    monkeypatch.setenv("COLLECTION_RETENTION", value)
    cfg = load_retention_env()
    assert cfg.window == value
    assert cfg.months == months
    assert cfg.enabled is True


def test_case_and_whitespace_are_forgiven(monkeypatch: pytest.MonkeyPatch) -> None:
    """``' 12M '`` is still the 12-month window."""
    monkeypatch.setenv("COLLECTION_RETENTION", " 12M ")
    assert load_retention_env().window == "12m"


def test_an_empty_value_is_off_without_a_warning(
    monkeypatch: pytest.MonkeyPatch, loguru_caplog: LogCaptureFixture
) -> None:
    """``COLLECTION_RETENTION=`` reads as unset, not as a typo."""
    monkeypatch.setenv("COLLECTION_RETENTION", "")
    assert load_retention_env().window == "off"
    assert loguru_caplog.records == []


@pytest.mark.parametrize("value", ["6", "6 months", "1y", "on", "true", "36m"])
def test_an_unknown_value_is_off_and_warns(
    monkeypatch: pytest.MonkeyPatch, loguru_caplog: LogCaptureFixture, value: str
) -> None:
    """A typo must never delete on a period nobody chose."""
    monkeypatch.setenv("COLLECTION_RETENTION", value)
    cfg = load_retention_env()
    assert cfg.window == "off"
    assert cfg.months == 0
    message = "\n".join(str(r.msg) for r in loguru_caplog.records)
    assert "COLLECTION_RETENTION" in message
    assert "off" in message
