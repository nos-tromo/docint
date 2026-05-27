"""Tests for :class:`ContextualUnderstandingAgent` elaboration handling."""

from __future__ import annotations

import json
from typing import Any

import pytest

from docint.agents import ContextualUnderstandingAgent, Turn
from docint.agents.context import TurnContext


@pytest.fixture(autouse=True)
def _pin_response_language_to_english(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin ``RESPONSE_LANGUAGE`` to ``en`` so prompt-text assertions are stable.

    The intent-analyst template is locale-aware (see
    ``prompts/{en,de}/intent_analyst.txt``); these tests assert on the English
    "rewritten_query MUST" phrasing, so they would flake under
    ``RESPONSE_LANGUAGE=de`` deployments without this pin.
    """
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")


class _FakeLLM:
    """Minimal LLM stub that records the prompt and returns a canned response."""

    def __init__(self, response_text: str) -> None:
        """Capture the canned text returned from ``complete``."""
        self._response_text = response_text
        self.last_prompt: str | None = None

    def complete(self, prompt: str) -> Any:
        """Record the prompt and return an object with a ``text`` attribute."""
        self.last_prompt = prompt

        class _Resp:
            def __init__(self, text: str) -> None:
                self.text = text

        return _Resp(self._response_text)


def test_prompt_documents_elaborate_intent_and_rule() -> None:
    """The understanding prompt must name 'elaborate' and the inlining rule.

    Guards against regressions where someone edits the prompt without
    preserving the elaboration follow-up handling.
    """
    llm = _FakeLLM(json.dumps({"intent": "qa", "rewritten_query": "hello"}))
    agent = ContextualUnderstandingAgent(llm=llm)  # type: ignore[arg-type]
    agent.analyze(Turn(user_input="hello"), context=TurnContext())
    assert llm.last_prompt is not None
    assert "elaborate" in llm.last_prompt.lower()
    assert "rewritten_query MUST" in llm.last_prompt or "MUST inline" in llm.last_prompt


def test_analyze_accepts_elaborate_intent() -> None:
    """``IntentAnalysis.intent`` should accept the new 'elaborate' value verbatim."""
    llm = _FakeLLM(
        json.dumps(
            {
                "intent": "elaborate",
                "rewritten_query": "UN Security Council references",
                "reason": "follow-up",
            }
        )
    )
    agent = ContextualUnderstandingAgent(llm=llm)  # type: ignore[arg-type]
    result = agent.analyze(
        Turn(user_input="I did not understand the UN references. Please elaborate."),
        context=TurnContext(
            history=[
                {"role": "user", "content": "Which interpretation is correct?"},
                {"role": "assistant", "content": "... UN Security Council ..."},
            ]
        ),
    )
    assert result.intent == "elaborate"
    assert result.rewritten_query == "UN Security Council references"


def test_analyze_forwards_history_into_prompt() -> None:
    """The recent assistant turn must appear in the prompt so the LLM can inline it."""
    llm = _FakeLLM(json.dumps({"intent": "qa", "rewritten_query": "noop"}))
    agent = ContextualUnderstandingAgent(llm=llm)  # type: ignore[arg-type]
    history = [
        {"role": "user", "content": "Which interpretation is correct?"},
        {"role": "assistant", "content": "The text mentions the UN Security Council."},
    ]
    agent.analyze(
        Turn(user_input="Please elaborate."),
        context=TurnContext(history=history),
    )
    assert llm.last_prompt is not None
    assert "UN Security Council" in llm.last_prompt
