"""Turn budget and the agent gate. Every value here is invented."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, fields, replace
from typing import Any

import pytest
from _pytest.logging import LogCaptureFixture
from agent_fakes import PLANNED_SCHEMA_TOKENS

from docint.agents.assessment import gate
from docint.agents.assessment.gate import MIN_CTX_WINDOW, agent_available
from docint.agents.turn.budget import TurnBudget, history_allowance_tokens, schema_tokens

RATIO = 3.5


def _budget(
    *,
    ctx_window: int = 8192,
    reasoning: bool = False,
    system: int = 900,
    schema: int = PLANNED_SCHEMA_TOKENS,
    history: int | None = None,
    message: int = 300,
) -> TurnBudget:
    """Build the budget of one turn, its history at the allowance unless given.

    Args:
        ctx_window (int): The chat model's context window in tokens.
        reasoning (bool): Whether the model thinks before it answers.
        system (int): Tokens of the system prompt.
        schema (int): Tokens of the offered tool schemas.
        history (int | None): Tokens of the history; the allowance for ``reasoning`` when ``None``.
        message (int): Tokens of the user's message.

    Returns:
        TurnBudget: The budget.
    """
    return TurnBudget(
        ctx_window=ctx_window,
        reasoning=reasoning,
        char_token_ratio=RATIO,
        system_tokens=system,
        tool_schema_tokens=schema,
        history_tokens=history_allowance_tokens(reasoning) if history is None else history,
        message_tokens=message,
    )


def _tool(name: str, description: str, argument: str = "query") -> dict[str, Any]:
    """Build the schema of a tool that takes one text argument, as the endpoint receives it.

    Args:
        name (str): The tool's name.
        description (str): What the tool is for.
        argument (str): The one argument's name.

    Returns:
        dict[str, Any]: The function-tool schema.
    """
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {argument: {"type": "string", "description": f"The {argument}."}},
                "required": [argument],
            },
        },
    }


SEARCH = _tool("search", "Search the collection and return numbered passages.")
READ = _tool("read", "Return the full text of one numbered passage.", argument="n")


@pytest.fixture(autouse=True)
def _fresh_warnings() -> None:
    """Let every test see the agent gate's once-per-window warning afresh."""
    gate._warn_small_agent_window.cache_clear()


def test_the_budget_is_a_frozen_value_built_in_the_pinned_field_order() -> None:
    """The fields keep the order later tasks build a budget in, and none can change once a turn has started."""
    budget = TurnBudget(8192, False, RATIO, 900, PLANNED_SCHEMA_TOKENS, 1300, 300)

    assert budget == _budget(history=1300)
    for name in (field.name for field in fields(TurnBudget)):
        with pytest.raises(FrozenInstanceError):
            setattr(budget, name, 1)


def test_an_8k_window_leaves_room_for_three_full_results() -> None:
    """A plain model at 8192 tokens can still read three full tool results before it answers."""
    budget = _budget()

    assert budget.available
    assert budget.round_room >= 3 * budget.result_chars / RATIO


@pytest.mark.parametrize(("reasoning", "reserve"), [(False, 1024), (True, 3072)])
def test_the_room_is_what_the_reply_and_the_prompt_leave_of_the_window(reasoning: bool, reserve: int) -> None:
    """Room is 95% of the window less the reply reserve, the four parts of the prompt and two message envelopes.

    Args:
        reasoning (bool): Whether the model thinks before it answers.
        reserve (int): The reply room it keeps free.
    """
    budget = _budget(reasoning=reasoning, history=1300)

    assert budget.round_room == 7782 - reserve - (900 + PLANNED_SCHEMA_TOKENS + 1300 + 300) - 2 * 40


def test_reasoning_at_8k_shrinks_history_before_refusing() -> None:
    """A thinking model's larger reply reserve is paid for by the history first; only then would a turn be refused."""
    assert history_allowance_tokens(False) == 1300
    assert history_allowance_tokens(True) == 750

    shrunk = _budget(reasoning=True, system=1200)
    unshrunk = replace(shrunk, history_tokens=history_allowance_tokens(False))

    assert shrunk.available
    assert not unshrunk.available
    assert _budget(system=1200).available


@pytest.mark.parametrize("reasoning", [False, True])
def test_too_small_a_window_is_unavailable(reasoning: bool) -> None:
    """At 4096 tokens the reply and the prompt leave nothing for tool results: no room, and never a negative one.

    Args:
        reasoning (bool): Whether the model thinks before it answers.
    """
    budget = _budget(ctx_window=4096, reasoning=reasoning)

    assert budget.round_room == 0
    assert not budget.available


def test_a_turn_needs_at_least_1500_tokens_of_room() -> None:
    """Room of exactly 1500 tokens is enough for a turn; one token less is not."""
    base = _budget()
    enough = replace(base, history_tokens=base.history_tokens + base.round_room - 1500)
    short = replace(enough, history_tokens=enough.history_tokens + 1)

    assert enough.round_room == 1500
    assert enough.available
    assert short.round_room == 1499
    assert not short.available


@pytest.mark.parametrize(
    ("reasoning", "result_chars", "passage_chars"),
    [(False, 3000, 900), (True, 2000, 700)],
    ids=["plain", "thinking"],
)
def test_results_and_passages_are_cut_shorter_for_a_thinking_model(
    reasoning: bool, result_chars: int, passage_chars: int
) -> None:
    """A thinking model has less room to spare, so a tool result and each passage in it are cut shorter.

    Args:
        reasoning (bool): Whether the model thinks before it answers.
        result_chars (int): The longest tool result in characters.
        passage_chars (int): The longest passage in a result in characters.
    """
    budget = _budget(reasoning=reasoning)

    assert (budget.result_chars, budget.passage_chars) == (result_chars, passage_chars)


def test_schema_tokens_are_measured_from_the_offered_tools() -> None:
    """The cost follows the tools a turn offers: none costs nothing, and each tool adds its own JSON."""
    assert schema_tokens([], RATIO) == 0

    one = schema_tokens([SEARCH], RATIO)
    both = schema_tokens([SEARCH, READ], RATIO)
    assert 0 < one < both
    assert schema_tokens([SEARCH, READ], 1.0) == len(json.dumps([SEARCH, READ], ensure_ascii=False))
    assert schema_tokens([SEARCH, READ], 2 * RATIO) < both


def test_schema_tokens_count_characters_not_json_escapes() -> None:
    """A description's accented letters cost what they cost the model, not six characters of escape each."""
    accented = _tool("search", "ü" * 100)
    plain = _tool("search", "u" * 100)

    assert schema_tokens([accented], RATIO) == schema_tokens([plain], RATIO)


@pytest.mark.parametrize(
    ("switch", "ctx_window", "expected"),
    [
        ("true", MIN_CTX_WINDOW, True),
        ("true", MIN_CTX_WINDOW - 1, False),
        ("true", 32768, True),
        ("false", 32768, False),
        (None, MIN_CTX_WINDOW, True),
    ],
    ids=["at-the-minimum", "below-the-minimum", "large-window", "switched-off", "on-by-default"],
)
def test_agent_available_respects_the_switch_and_the_window(
    monkeypatch: pytest.MonkeyPatch, switch: str | None, ctx_window: int, expected: bool
) -> None:
    """Agent turns need the switch on (it is by default) and a window of at least 8192 tokens.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        switch (str | None): ``AGENT_TURNS_ENABLED``; unset when ``None``.
        ctx_window (int): The chat model's context window in tokens.
        expected (bool): Whether agent turns may run.
    """
    monkeypatch.setenv("ASSESSMENT_ENABLED", "false")
    if switch is None:
        monkeypatch.delenv("AGENT_TURNS_ENABLED", raising=False)
    else:
        monkeypatch.setenv("AGENT_TURNS_ENABLED", switch)

    assert agent_available(ctx_window) is expected


def test_a_small_window_warns_once_per_window_and_names_the_knob(
    monkeypatch: pytest.MonkeyPatch, loguru_caplog: LogCaptureFixture
) -> None:
    """A too-small window is reported once per size, naming what to change.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    monkeypatch.setenv("AGENT_TURNS_ENABLED", "true")

    agent_available(4096)
    agent_available(4096)
    agent_available(2048)

    warnings = [message for message in loguru_caplog.messages if "Agent turns are off" in message]
    assert warnings == [
        "Agent turns are off: the chat context window is 4096 tokens, below the 8192 they need. "
        "Raise OPENAI_CTX_WINDOW (and the context window the chat endpoint serves) to enable them.",
        "Agent turns are off: the chat context window is 2048 tokens, below the 8192 they need. "
        "Raise OPENAI_CTX_WINDOW (and the context window the chat endpoint serves) to enable them.",
    ]


def test_a_switched_off_gate_does_not_warn_about_the_window(
    monkeypatch: pytest.MonkeyPatch, loguru_caplog: LogCaptureFixture
) -> None:
    """An operator who turned agent turns off is not told their window is too small for them.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    monkeypatch.setenv("AGENT_TURNS_ENABLED", "false")

    assert agent_available(4096) is False
    assert [message for message in loguru_caplog.messages if "Agent turns are off" in message] == []
