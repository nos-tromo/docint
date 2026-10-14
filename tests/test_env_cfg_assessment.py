"""Assessment-turn configuration and the deployment gate. All values are invented."""

from __future__ import annotations

import pytest
from _pytest.logging import LogCaptureFixture

from docint.agents.assessment import gate
from docint.agents.assessment.gate import MIN_CTX_WINDOW, assessment_available
from docint.utils.env_cfg import load_assessment_env


def test_defaults_enable_six_research_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset variables mean assessments on, with six research rounds.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.delenv("ASSESSMENT_ENABLED", raising=False)
    monkeypatch.delenv("ASSESSMENT_MAX_STEPS", raising=False)

    cfg = load_assessment_env()

    assert cfg.enabled is True
    assert cfg.max_steps == 6


@pytest.mark.parametrize(("raw", "expected"), [("false", False), ("0", False), ("TRUE", True), ("yes", True)])
def test_enabled_parses_truthy_strings(monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool) -> None:
    """The switch accepts the same truthy spellings as every other docint flag.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        raw (str): The environment value.
        expected (bool): The parsed flag.
    """
    monkeypatch.setenv("ASSESSMENT_ENABLED", raw)

    assert load_assessment_env().enabled is expected


@pytest.mark.parametrize(("raw", "expected"), [("3", 3), ("0", 0), ("-2", 0)])
def test_max_steps_is_clamped_at_zero(monkeypatch: pytest.MonkeyPatch, raw: str, expected: int) -> None:
    """A negative step count means no research, never an error.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        raw (str): The environment value.
        expected (int): The parsed step count.
    """
    monkeypatch.setenv("ASSESSMENT_MAX_STEPS", raw)

    assert load_assessment_env().max_steps == expected


def test_gate_needs_the_minimum_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """Below 8192 tokens the answer step cannot hold its evidence, so the gate closes.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setenv("ASSESSMENT_ENABLED", "true")

    assert assessment_available(MIN_CTX_WINDOW) is True
    assert assessment_available(MIN_CTX_WINDOW - 1) is False


def test_gate_is_closed_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """The operator switch wins over any window size.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setenv("ASSESSMENT_ENABLED", "false")

    assert assessment_available(32768) is False


def test_small_window_warns_once_and_names_the_knob(
    monkeypatch: pytest.MonkeyPatch, loguru_caplog: LogCaptureFixture
) -> None:
    """A too-small window is reported once per process, naming what to change.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    monkeypatch.setenv("ASSESSMENT_ENABLED", "true")
    gate._warn_small_window.cache_clear()

    assessment_available(4096)
    assessment_available(4096)

    warnings = [message for message in loguru_caplog.messages if "Assessment turns are off" in message]
    assert len(warnings) == 1
    assert "OPENAI_CTX_WINDOW" in warnings[0]
