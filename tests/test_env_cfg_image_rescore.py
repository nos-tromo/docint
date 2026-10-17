"""Tests for the image re-score settings in ``load_image_ingestion_config``.

A sub-floor image inside the cut is re-scored against a caption-style
description of the question. The operator can switch that off, which costs one
chat call per such turn, retune the score that rescues an image, and bound how
long the description call may hold up the answer.
"""

from __future__ import annotations

import pytest
from _pytest.logging import LogCaptureFixture

from docint.utils.env_cfg import load_image_ingestion_config


@pytest.fixture(autouse=True)
def _clear_rescore_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear the settings so a developer ``.env`` cannot skew the defaults.

    Args:
        monkeypatch: Pytest environment patcher.
    """
    for name in ("IMAGE_RESCORE_ENABLED", "IMAGE_RESCORE_MIN_SCORE", "IMAGE_RESCORE_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)


def test_the_rescore_is_on_by_default_with_its_measured_floor() -> None:
    """With nothing set, sub-floor images inside the cut get their second look."""
    cfg = load_image_ingestion_config()

    assert cfg.rescore_enabled is True
    assert cfg.rescore_min_score == 0.02


@pytest.mark.parametrize("value", ["false", "False", "0", "no", "off", ""])
def test_the_rescore_can_be_switched_off(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anything outside the truthy set switches the re-score off.

    Args:
        value: The environment-variable spelling under test.
        monkeypatch: Pytest environment patcher.
    """
    monkeypatch.setenv("IMAGE_RESCORE_ENABLED", value)

    assert load_image_ingestion_config().rescore_enabled is False


@pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes"])
def test_truthy_values_switch_the_rescore_on_against_a_false_default(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Truthy spellings win over a caller-supplied ``False`` default.

    Args:
        value: The environment-variable spelling under test.
        monkeypatch: Pytest environment patcher.
    """
    monkeypatch.setenv("IMAGE_RESCORE_ENABLED", value)

    assert load_image_ingestion_config(default_rescore_enabled=False).rescore_enabled is True


def test_the_rescore_floor_is_retunable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Operators can move the description score that rescues an image.

    Args:
        monkeypatch: Pytest environment patcher.
    """
    monkeypatch.setenv("IMAGE_RESCORE_MIN_SCORE", "0.1")

    assert load_image_ingestion_config().rescore_min_score == 0.1


def test_the_caller_default_applies_when_the_floor_is_unset() -> None:
    """The caller-supplied floor stands in for an unset variable."""
    assert load_image_ingestion_config(default_rescore_min_score=0.04).rescore_min_score == 0.04


def test_the_description_call_has_a_ten_second_deadline_by_default() -> None:
    """The call runs before the answer's first token, so it gets a short deadline of its own."""
    assert load_image_ingestion_config().rescore_timeout == 10.0


def test_the_deadline_is_retunable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A slow chat model can be given longer.

    Args:
        monkeypatch: Pytest environment patcher.
    """
    monkeypatch.setenv("IMAGE_RESCORE_TIMEOUT", "2.5")

    assert load_image_ingestion_config().rescore_timeout == 2.5


@pytest.mark.parametrize("value", ["0", "-1", "soon", "nan", "inf"])
def test_an_unusable_deadline_keeps_the_default_with_a_warning(
    value: str, monkeypatch: pytest.MonkeyPatch, loguru_caplog: LogCaptureFixture
) -> None:
    """A deadline that is not a positive number of seconds must neither stall turns nor break bring-up.

    Args:
        value: The unusable spelling under test.
        monkeypatch: Pytest environment patcher.
        loguru_caplog: Captures loguru WARNING records.
    """
    monkeypatch.setenv("IMAGE_RESCORE_TIMEOUT", value)

    assert load_image_ingestion_config().rescore_timeout == 10.0
    assert any("IMAGE_RESCORE_TIMEOUT" in record.getMessage() for record in loguru_caplog.records)
