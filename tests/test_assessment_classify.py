"""Turn classification. All questions and answers are invented."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from _pytest.logging import LogCaptureFixture

from docint.agents.assessment.classify import DEFAULT_ASSESSMENT_CLASSIFY_PROMPT, classify_turn
from docint.agents.types import PriorTurn

QUESTION = "zzsecretzz What could be improved in this study?"


def _llm(reply: str) -> MagicMock:
    """Build a model double whose completion returns ``reply``.

    Args:
        reply (str): The raw completion text.

    Returns:
        MagicMock: The model double.
    """
    llm = MagicMock()
    llm.complete.return_value = SimpleNamespace(text=reply)
    return llm


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ('{"mode": "assessment"}', "assessment"),
        ('```json\n{"mode": "ASSESSMENT"}\n```', "assessment"),
        ('<think>asks for critique</think>{"mode": "assessment"}', "assessment"),
        ('Sure: {"mode": "lookup"}', "lookup"),
        ('{"mode": "maybe"}', "lookup"),
        ("no json at all", "lookup"),
        ("{not json}", "lookup"),
    ],
)
def test_a_reply_maps_to_a_mode(reply: str, expected: str) -> None:
    """Anything but an explicit assessment verdict is a lookup.

    Args:
        reply (str): The classifier's raw reply.
        expected (str): The mode it must map to.
    """
    assert classify_turn(cast(Any, _llm(reply)), question=QUESTION, prior_turn=None) == expected


def test_no_model_or_a_blank_question_is_a_lookup() -> None:
    """Nothing to classify means today's behaviour, without a model call."""
    llm = _llm('{"mode": "assessment"}')

    assert classify_turn(None, question=QUESTION, prior_turn=None) == "lookup"
    assert classify_turn(cast(Any, llm), question="   ", prior_turn=None) == "lookup"
    llm.complete.assert_not_called()


def test_a_failed_call_is_a_lookup_and_never_logs_the_question(loguru_caplog: LogCaptureFixture) -> None:
    """The warning names the failure's type; the question never reaches the log.

    Args:
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    llm = MagicMock()
    llm.complete.side_effect = RuntimeError(QUESTION)

    assert classify_turn(cast(Any, llm), question=QUESTION, prior_turn=None) == "lookup"
    assert "RuntimeError" in loguru_caplog.text
    assert "zzsecretzz" not in loguru_caplog.text


def test_the_prompt_carries_the_question_and_the_prior_exchange(monkeypatch: pytest.MonkeyPatch) -> None:
    """Follow-ups like "and what do you think of that?" need the previous answer.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    llm = _llm('{"mode": "lookup"}')
    prior = PriorTurn(user_text="Summarise it.", assistant_text="It reports one experiment [1].")

    classify_turn(cast(Any, llm), question=QUESTION, prior_turn=prior)

    prompt = llm.complete.call_args.args[0]
    assert QUESTION in prompt
    assert "It reports one experiment [1]." in prompt
    assert '{"mode": "assessment"}' in prompt


def _assert_leans_to_lookup(prompt: str, tie_break: str) -> None:
    """Check that lookup is listed first, wins a tie, and leads the format line.

    Args:
        prompt (str): The rendered classifier prompt.
        tie_break (str): The locale's "when unsure" sentence.
    """
    assert prompt.index('- "lookup"') < prompt.index('- "assessment"')
    assert tie_break in prompt.split('- "assessment"', 1)[1]
    last_line = prompt.strip().splitlines()[-1]
    assert last_line.index('{"mode": "lookup"}') < last_line.index('{"mode": "assessment"}')


@pytest.mark.parametrize(
    ("code", "tie_break"), [("en", "When unsure, answer lookup."), ("de", "Im Zweifel antworte mit lookup.")]
)
def test_the_prompt_leans_to_lookup(monkeypatch: pytest.MonkeyPatch, code: str, tie_break: str) -> None:
    """A model that echoes the format line, or cannot decide, picks the cheap mode.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        code (str): The locale.
        tie_break (str): The locale's "when unsure" sentence.
    """
    monkeypatch.setenv("RESPONSE_LANGUAGE", code)
    llm = _llm('{"mode": "lookup"}')

    classify_turn(cast(Any, llm), question=QUESTION, prior_turn=None)

    _assert_leans_to_lookup(llm.complete.call_args.args[0], tie_break)


def test_the_bundled_fallback_prompt_leans_to_lookup() -> None:
    """The default used when the prompt file is missing keeps the same order."""
    prompt = DEFAULT_ASSESSMENT_CLASSIFY_PROMPT.format(question=QUESTION, prior_turn="(none)")

    _assert_leans_to_lookup(prompt, "When unsure, answer lookup.")
