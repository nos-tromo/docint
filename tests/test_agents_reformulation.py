"""Tests for :class:`QueryReformulationAgent` in the reformulation module."""

from typing import Any, cast

import pytest

from docint.agents.reformulation import QueryReformulationAgent


@pytest.fixture(autouse=True)
def _pin_response_language_to_english(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin ``RESPONSE_LANGUAGE`` to ``en`` so prompt-text assertions are stable.

    The reformulation template is locale-aware (see
    ``prompts/{en,de}/reformulate_retrieval.txt``); these tests assert on the
    English text, so they would flake under ``RESPONSE_LANGUAGE=de``.
    """
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")


class _FakeLLMResponse:
    """Minimal stand-in for an LLM completion response."""

    def __init__(self, text: str) -> None:
        """Initialise with a canned response text.

        Args:
            text: The response text to return.
        """
        self.text = text


class _FakeLLM:
    """Controllable fake LLM that returns a pre-configured response."""

    def __init__(self, response_text: str) -> None:
        """Initialise with the text the fake LLM should return.

        Args:
            response_text: The canned completion text.
        """
        self.response_text = response_text
        self.calls = 0
        self.last_prompt: str | None = None

    def complete(self, prompt: str) -> _FakeLLMResponse:
        """Record the prompt and return the pre-configured response.

        Args:
            prompt: The prompt string sent to the LLM.

        Returns:
            A ``_FakeLLMResponse`` with the canned text.
        """
        self.calls += 1
        self.last_prompt = prompt
        return _FakeLLMResponse(self.response_text)


class _RaisingLLM:
    """Fake LLM whose completion call always fails."""

    def __init__(self) -> None:
        """Initialise the call counter."""
        self.calls = 0

    def complete(self, prompt: str) -> _FakeLLMResponse:
        """Raise to simulate a transport or model failure.

        Args:
            prompt: The prompt string (ignored).

        Raises:
            RuntimeError: Always.
        """
        _ = prompt
        self.calls += 1
        raise RuntimeError("reformulation endpoint unreachable")


def _agent(llm: Any) -> QueryReformulationAgent:
    """Build an agent bound to a fake LLM.

    Args:
        llm: The fake LLM stand-in.

    Returns:
        QueryReformulationAgent: Agent under test.
    """
    return QueryReformulationAgent(llm=cast("Any", llm))


def test_reformulate_returns_query_and_prompts_with_all_inputs() -> None:
    """The prompt carries the question, the failed query, and the reason."""
    llm = _FakeLLM("Security Council resolutions on sanctions")
    agent = _agent(llm)

    result = agent.reformulate(
        user_query="What did the UN say?",
        failed_query="UN say",
        validation_reason="no UN content in sources",
    )

    assert result == "Security Council resolutions on sanctions"
    assert llm.calls == 1
    prompt = llm.last_prompt or ""
    assert "What did the UN say?" in prompt
    assert "UN say" in prompt
    assert "no UN content in sources" in prompt


def test_reformulate_collapses_whitespace_in_the_model_output() -> None:
    """Multi-line model output becomes a single-line query."""
    agent = _agent(_FakeLLM("  Security Council\n  sanctions regime \n"))

    assert agent.reformulate(user_query="What did the UN say?") == "Security Council sanctions regime"


def test_reformulate_returns_none_without_an_llm() -> None:
    """No bound LLM means no retry rather than an error."""
    agent = QueryReformulationAgent(llm=None)

    assert agent.reformulate(user_query="What did the UN say?") is None


def test_reformulate_returns_none_on_empty_output() -> None:
    """A blank completion yields no reformulation."""
    agent = _agent(_FakeLLM("   \n  "))

    assert agent.reformulate(user_query="What did the UN say?") is None


def test_reformulate_returns_none_when_llm_raises() -> None:
    """A failing model degrades to no retry instead of propagating."""
    llm = _RaisingLLM()
    agent = _agent(llm)

    assert agent.reformulate(user_query="What did the UN say?") is None
    assert llm.calls == 1


@pytest.mark.parametrize("echoed", ["UN sanctions", "un SANCTIONS", "What did the UN say?"])
def test_reformulate_rejects_already_tried_queries(echoed: str) -> None:
    """Echoing the failed query or the question cannot retrieve anything new."""
    agent = _agent(_FakeLLM(echoed))

    result = agent.reformulate(
        user_query="What did the UN say?",
        failed_query="UN sanctions",
        validation_reason="off-topic sources",
    )

    assert result is None


def test_reformulate_returns_none_for_blank_user_query() -> None:
    """An empty question has nothing to reformulate against."""
    llm = _FakeLLM("something")
    agent = _agent(llm)

    assert agent.reformulate(user_query="   ") is None
    assert llm.calls == 0


def test_reformulate_defaults_missing_inputs_in_the_prompt() -> None:
    """Absent failed query / reason fall back to safe placeholders."""
    llm = _FakeLLM("broader terms")
    agent = _agent(llm)

    assert agent.reformulate(user_query="What did the UN say?") == "broader terms"
    prompt = llm.last_prompt or ""
    # The user query stands in for the failed query, and the reason is named.
    assert "No reason recorded." in prompt
    assert prompt.count("What did the UN say?") >= 2
