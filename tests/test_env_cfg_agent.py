"""Agent-turn configuration. All values are invented."""

from __future__ import annotations

import pytest
from _pytest.logging import LogCaptureFixture

from docint.utils.env_cfg import AgentConfig, load_agent_env, load_assessment_env

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
    """Start every test with no agent variable set, so an ambient ``.env`` cannot skew it.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    for var in _VARS:
        monkeypatch.delenv(var, raising=False)


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
        (" Yes ", True),
        ("false", False),
        ("FALSE", False),
        ("0", False),
        ("no", False),
        ("off", False),
        ("on", False),
        ("maybe", False),
        ("disabled", False),
    ],
)
def test_the_switch_reads_the_spellings_every_docint_flag_does(
    monkeypatch: pytest.MonkeyPatch, loguru_caplog: LogCaptureFixture, raw: str, expected: bool
) -> None:
    """Only ``true``, ``1`` and ``yes`` switch it on; any other value is false, ``on`` included, and none warns.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
        raw (str): The environment value.
        expected (bool): The parsed flag.
    """
    monkeypatch.setenv("AGENT_TURNS_ENABLED", raw)

    assert load_agent_env().enabled is expected
    assert loguru_caplog.messages == []


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_value_counts_as_unset(
    monkeypatch: pytest.MonkeyPatch, loguru_caplog: LogCaptureFixture, blank: str
) -> None:
    """An empty ``NAME=`` line in ``.env`` or compose is the default: not "off", and not a typo.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
        blank (str): The blank environment value.
    """
    for var in _VARS:
        monkeypatch.setenv(var, blank)

    assert load_agent_env() == DEFAULTS
    assert loguru_caplog.messages == []


@pytest.mark.parametrize(
    ("var", "raw", "used"),
    [
        ("AGENT_MAX_ROUNDS", "many", "6"),
        ("AGENT_MAX_ROUNDS", "2.5", "6"),
        ("AGENT_MAX_CALLS_PER_ROUND", "several", "4"),
    ],
)
def test_an_unusable_integer_falls_back_to_the_default_and_warns(
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


@pytest.mark.parametrize(
    ("raw", "problem"),
    [
        ("soon", "is not a number"),
        ("nan", "must be a positive, finite number of seconds"),
        ("inf", "must be a positive, finite number of seconds"),
        ("0", "must be a positive, finite number of seconds"),
        ("-30", "must be a positive, finite number of seconds"),
    ],
)
def test_an_unusable_time_budget_falls_back_to_the_default_and_says_why(
    monkeypatch: pytest.MonkeyPatch, loguru_caplog: LogCaptureFixture, raw: str, problem: str
) -> None:
    """A budget that is not a number, or not positive and finite, keeps the default and the warning says which.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
        raw (str): The unusable value.
        problem (str): What the warning says is wrong with it.
    """
    monkeypatch.setenv("AGENT_TIME_BUDGET_S", raw)

    assert load_agent_env() == DEFAULTS
    assert loguru_caplog.messages == [f"AGENT_TIME_BUDGET_S={raw!r} {problem}; using default 120.0."]


def test_an_unusable_agent_max_rounds_gets_the_default_not_the_older_value(
    monkeypatch: pytest.MonkeyPatch, loguru_caplog: LogCaptureFixture
) -> None:
    """Once ``AGENT_MAX_ROUNDS`` is set it is the variable in force, so a typo in it earns the default.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    monkeypatch.setenv("ASSESSMENT_MAX_STEPS", "3")
    monkeypatch.setenv("AGENT_MAX_ROUNDS", "many")

    assert load_agent_env().max_rounds == 6
    assert loguru_caplog.messages == ["Invalid AGENT_MAX_ROUNDS 'many'; using 6."]


def test_an_unusable_assessment_max_steps_leaves_the_default(
    monkeypatch: pytest.MonkeyPatch, loguru_caplog: LogCaptureFixture
) -> None:
    """While it stands in for ``AGENT_MAX_ROUNDS``, the older variable is read as carefully as the new ones.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    monkeypatch.setenv("ASSESSMENT_MAX_STEPS", "many")

    assert load_agent_env().max_rounds == 6
    assert loguru_caplog.messages == ["Invalid ASSESSMENT_MAX_STEPS 'many'; using 6."]


def test_assessment_max_steps_is_not_read_while_agent_max_rounds_is_set(
    monkeypatch: pytest.MonkeyPatch, loguru_caplog: LogCaptureFixture
) -> None:
    """A malformed older variable is not reported, and no misleading "using 6" appears, when it is not consulted.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    monkeypatch.setenv("AGENT_MAX_ROUNDS", "2")
    monkeypatch.setenv("ASSESSMENT_MAX_STEPS", "many")

    assert load_agent_env().max_rounds == 2
    assert loguru_caplog.messages == []


@pytest.mark.parametrize("var", ["AGENT_MAX_ROUNDS", "ASSESSMENT_MAX_STEPS"])
@pytest.mark.parametrize(("raw", "expected"), [("3", 3), ("1", 1), ("0", 1), ("-2", 1)])
def test_max_rounds_is_at_least_one(monkeypatch: pytest.MonkeyPatch, var: str, raw: str, expected: int) -> None:
    """A turn may always search once: with no tool round, every turn would answer from no evidence.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        var (str): The variable carrying the round budget.
        raw (str): The environment value.
        expected (int): The parsed round budget.
    """
    monkeypatch.setenv(var, raw)

    assert load_agent_env().max_rounds == expected


@pytest.mark.parametrize("var", ["AGENT_MAX_ROUNDS", "ASSESSMENT_MAX_STEPS"])
def test_a_round_budget_below_one_warns_naming_its_variable(
    monkeypatch: pytest.MonkeyPatch, loguru_caplog: LogCaptureFixture, var: str
) -> None:
    """``0`` once meant "answer from the first search alone"; raised to one round, it says so, naming the variable read.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
        var (str): The variable carrying the round budget.
    """
    monkeypatch.setenv(var, "0")

    assert load_agent_env().max_rounds == 1
    assert loguru_caplog.messages == [f"{var} 0 is < 1; clamping."]


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

    assert loguru_caplog.messages == ["AGENT_MAX_CALLS_PER_ROUND 0 is < 1; clamping."]


def test_assessments_are_on_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset, ``ASSESSMENT_ENABLED`` lets an agent turn answer a judgement as a labelled assessment.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.delenv("ASSESSMENT_ENABLED", raising=False)

    assert load_assessment_env().enabled is True


@pytest.mark.parametrize(("raw", "expected"), [("false", False), ("0", False), ("TRUE", True), ("yes", True)])
def test_the_assessment_switch_parses_truthy_strings(monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool) -> None:
    """The switch accepts the same truthy spellings as every other docint flag.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        raw (str): The environment value.
        expected (bool): The parsed flag.
    """
    monkeypatch.setenv("ASSESSMENT_ENABLED", raw)

    assert load_assessment_env().enabled is expected


def test_a_malformed_older_round_count_cannot_fail_the_assessment_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    """``ASSESSMENT_MAX_STEPS`` is only ``AGENT_MAX_ROUNDS``' fallback; every agent turn reads the switch beside it.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.delenv("ASSESSMENT_ENABLED", raising=False)
    monkeypatch.setenv("ASSESSMENT_MAX_STEPS", "many")

    assert load_assessment_env().enabled is True
