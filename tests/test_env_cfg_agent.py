"""Agent-turn configuration. All values are invented."""

from __future__ import annotations

import pytest
from _pytest.logging import LogCaptureFixture

from docint.utils import env_cfg
from docint.utils.env_cfg import AgentConfig, load_agent_env

_VARS = (
    "AGENT_TURNS_ENABLED",
    "AGENT_MAX_ROUNDS",
    "AGENT_TIME_BUDGET_S",
    "AGENT_MAX_CALLS_PER_ROUND",
    "ASSESSMENT_MAX_STEPS",
)

DEFAULTS = AgentConfig(enabled=True, max_rounds=6, time_budget_s=120.0, max_calls_per_round=4)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test with no agent variable set and no warning remembered.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    for var in _VARS:
        monkeypatch.delenv(var, raising=False)
    env_cfg._warn_agent_setting.cache_clear()


def test_defaults() -> None:
    """Unset variables mean agent turns on: six rounds, two minutes, four calls a round."""
    assert load_agent_env() == DEFAULTS


def test_every_knob_reads_its_own_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each setting follows its own variable and no other.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setenv("AGENT_TURNS_ENABLED", "false")
    monkeypatch.setenv("AGENT_MAX_ROUNDS", "3")
    monkeypatch.setenv("AGENT_TIME_BUDGET_S", "45.5")
    monkeypatch.setenv("AGENT_MAX_CALLS_PER_ROUND", "2")

    assert load_agent_env() == AgentConfig(enabled=False, max_rounds=3, time_budget_s=45.5, max_calls_per_round=2)


def test_the_callers_defaults_apply_when_nothing_is_set() -> None:
    """Each ``default_*`` parameter is the value its setting really falls back to."""
    cfg = load_agent_env(
        default_enabled=False,
        default_max_rounds=2,
        default_time_budget_s=30.0,
        default_max_calls_per_round=1,
    )

    assert cfg == AgentConfig(enabled=False, max_rounds=2, time_budget_s=30.0, max_calls_per_round=1)


def test_the_environment_beats_the_callers_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """The operator's value wins over the caller's default, in both directions.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setenv("AGENT_TURNS_ENABLED", "true")
    monkeypatch.setenv("AGENT_MAX_ROUNDS", "5")
    monkeypatch.setenv("AGENT_TIME_BUDGET_S", "90")
    monkeypatch.setenv("AGENT_MAX_CALLS_PER_ROUND", "3")

    cfg = load_agent_env(
        default_enabled=False,
        default_max_rounds=2,
        default_time_budget_s=30.0,
        default_max_calls_per_round=1,
    )

    assert cfg == AgentConfig(enabled=True, max_rounds=5, time_budget_s=90.0, max_calls_per_round=3)

    monkeypatch.setenv("AGENT_TURNS_ENABLED", "false")

    assert load_agent_env(default_enabled=True).enabled is False


@pytest.mark.parametrize("agent_value", [None, "", "   "])
def test_agent_max_rounds_falls_back_to_assessment_max_steps(
    monkeypatch: pytest.MonkeyPatch, agent_value: str | None
) -> None:
    """A deployment that tuned ``ASSESSMENT_MAX_STEPS`` keeps its budget when ``AGENT_MAX_ROUNDS`` is unset or blank.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        agent_value (str | None): The ``AGENT_MAX_ROUNDS`` value; ``None`` leaves it unset.
    """
    monkeypatch.setenv("ASSESSMENT_MAX_STEPS", "3")
    if agent_value is not None:
        monkeypatch.setenv("AGENT_MAX_ROUNDS", agent_value)

    assert load_agent_env().max_rounds == 3


def test_agent_max_rounds_wins_over_assessment_max_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Once the new variable is set, the older one no longer matters.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setenv("ASSESSMENT_MAX_STEPS", "3")
    monkeypatch.setenv("AGENT_MAX_ROUNDS", "2")

    assert load_agent_env().max_rounds == 2


def test_agent_turns_can_be_switched_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """The operator switch turns the loop off.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setenv("AGENT_TURNS_ENABLED", "false")

    assert load_agent_env().enabled is False


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("true", True),
        ("TRUE", True),
        ("1", True),
        ("yes", True),
        ("on", True),
        ("false", False),
        ("FALSE", False),
        ("0", False),
        ("no", False),
        ("off", False),
        (" off ", False),
    ],
)
def test_the_switch_accepts_the_usual_spellings(monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool) -> None:
    """Every spelling of on and off other docint flags accept is understood.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        raw (str): The environment value.
        expected (bool): The parsed flag.
    """
    monkeypatch.setenv("AGENT_TURNS_ENABLED", raw)

    assert load_agent_env().enabled is expected


def test_a_blank_value_counts_as_unset(monkeypatch: pytest.MonkeyPatch, loguru_caplog: LogCaptureFixture) -> None:
    """An empty ``NAME=`` line in ``.env`` or compose is the default: not "off", and not a typo.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    for var in _VARS:
        monkeypatch.setenv(var, "")

    assert load_agent_env() == DEFAULTS
    assert loguru_caplog.messages == []


@pytest.mark.parametrize(
    ("var", "raw", "used"),
    [
        ("AGENT_TURNS_ENABLED", "maybe", "true"),
        ("AGENT_MAX_ROUNDS", "many", "6"),
        ("AGENT_MAX_ROUNDS", "2.5", "6"),
        ("AGENT_TIME_BUDGET_S", "soon", "120.0"),
        ("AGENT_TIME_BUDGET_S", "nan", "120.0"),
        ("AGENT_TIME_BUDGET_S", "inf", "120.0"),
        ("AGENT_TIME_BUDGET_S", "0", "120.0"),
        ("AGENT_TIME_BUDGET_S", "-30", "120.0"),
        ("AGENT_MAX_CALLS_PER_ROUND", "several", "4"),
    ],
)
def test_an_unusable_value_falls_back_to_the_default_and_warns(
    monkeypatch: pytest.MonkeyPatch, loguru_caplog: LogCaptureFixture, var: str, raw: str, used: str
) -> None:
    """A typo keeps the default, and the warning names the variable, the value and what is used instead.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
        var (str): The variable under test.
        raw (str): The unusable value.
        used (str): The default the warning says is used.
    """
    monkeypatch.setenv(var, raw)

    assert load_agent_env() == DEFAULTS
    assert loguru_caplog.messages == [f"Invalid {var} {raw!r}; using {used}."]


def test_an_unusable_agent_max_rounds_counts_as_unset(
    monkeypatch: pytest.MonkeyPatch, loguru_caplog: LogCaptureFixture
) -> None:
    """A typo in the new variable does not discard the budget the older one set.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    monkeypatch.setenv("ASSESSMENT_MAX_STEPS", "3")
    monkeypatch.setenv("AGENT_MAX_ROUNDS", "many")

    assert load_agent_env().max_rounds == 3
    assert loguru_caplog.messages == ["Invalid AGENT_MAX_ROUNDS 'many'; using 3."]


def test_an_unusable_assessment_max_steps_leaves_the_default(
    monkeypatch: pytest.MonkeyPatch, loguru_caplog: LogCaptureFixture
) -> None:
    """The inherited variable is read as carefully as the new ones.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    monkeypatch.setenv("ASSESSMENT_MAX_STEPS", "many")

    assert load_agent_env().max_rounds == 6
    assert loguru_caplog.messages == ["Invalid ASSESSMENT_MAX_STEPS 'many'; using 6."]


@pytest.mark.parametrize("var", ["AGENT_MAX_ROUNDS", "ASSESSMENT_MAX_STEPS"])
@pytest.mark.parametrize(("raw", "expected"), [("3", 3), ("1", 1), ("0", 0), ("-2", 0)])
def test_max_rounds_is_clamped_at_zero(monkeypatch: pytest.MonkeyPatch, var: str, raw: str, expected: int) -> None:
    """A negative budget means no research rounds, never an error; zero stays a valid budget.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        var (str): The variable carrying the round budget.
        raw (str): The environment value.
        expected (int): The parsed round budget.
    """
    monkeypatch.setenv(var, raw)

    assert load_agent_env().max_rounds == expected


@pytest.mark.parametrize(("raw", "expected"), [("8", 8), ("1", 1), ("0", 1), ("-3", 1)])
def test_max_calls_per_round_is_at_least_one(monkeypatch: pytest.MonkeyPatch, raw: str, expected: int) -> None:
    """A round that may make no call could never research, so the floor is one.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        raw (str): The environment value.
        expected (int): The parsed per-round call limit.
    """
    monkeypatch.setenv("AGENT_MAX_CALLS_PER_ROUND", raw)

    assert load_agent_env().max_calls_per_round == expected


def test_a_clamped_value_warns(monkeypatch: pytest.MonkeyPatch, loguru_caplog: LogCaptureFixture) -> None:
    """Raising a value to its floor is reported, so the operator learns the setting was not taken as written.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    monkeypatch.setenv("AGENT_MAX_CALLS_PER_ROUND", "0")

    load_agent_env()

    assert loguru_caplog.messages == ["AGENT_MAX_CALLS_PER_ROUND 0 is below 1; using 1."]


def test_a_bad_value_is_reported_once_per_process(
    monkeypatch: pytest.MonkeyPatch, loguru_caplog: LogCaptureFixture
) -> None:
    """The loader runs on every chat turn; a typo must not repeat itself in the log.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    monkeypatch.setenv("AGENT_MAX_ROUNDS", "many")

    for _ in range(3):
        load_agent_env()

    assert loguru_caplog.messages == ["Invalid AGENT_MAX_ROUNDS 'many'; using 6."]
