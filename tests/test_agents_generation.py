"""Tests for :class:`ResultValidationResponseAgent` in the generation module."""

from docint.agents.generation import ResultValidationResponseAgent
from docint.agents.types import RetrievalResult, Turn


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
        return _FakeLLMResponse(text=self.response_text)


def test_validation_agent_sets_alert_on_mismatch() -> None:
    """Grounding mismatch from the LLM should set the validation alert flag."""
    llm = _FakeLLM(
        '{"summary_grounded": false, "sources_relevant": true, "reason":"hallucinated fact"}'
    )
    agent = ResultValidationResponseAgent(enabled=True, llm=llm)
    result = RetrievalResult(answer="answer", sources=[{"text": "source evidence"}])

    finalized = agent.finalize(result, Turn(user_input="question"))

    assert llm.calls == 1
    assert llm.last_prompt is not None
    assert "User query:\nquestion" in llm.last_prompt
    assert "Answer:\nanswer" in llm.last_prompt
    assert finalized.validation_checked is True
    assert finalized.validation_mismatch is True
    assert finalized.validation_reason == "hallucinated fact"


def test_validation_agent_disabled_is_noop() -> None:
    """Disabled validator should not invoke the LLM or set any flags."""
    llm = _FakeLLM(
        '{"summary_grounded": false, "sources_relevant": false, "reason":"bad"}'
    )
    agent = ResultValidationResponseAgent(enabled=False, llm=llm)
    result = RetrievalResult(answer="answer", sources=[{"text": "source evidence"}])

    finalized = agent.finalize(result, Turn(user_input="question"))

    assert llm.calls == 0
    assert finalized.validation_checked is None
    assert finalized.validation_mismatch is None
    assert finalized.validation_reason is None


def test_validation_agent_parses_markdown_wrapped_json() -> None:
    """Markdown-fenced JSON from the LLM should be unwrapped and parsed."""
    llm = _FakeLLM(
        '```json\n{"summary_grounded": true, "sources_relevant": true, "reason":"ok"}\n```'
    )
    agent = ResultValidationResponseAgent(enabled=True, llm=llm)
    result = RetrievalResult(answer="answer", sources=[{"id": 1, "content": "source"}])

    finalized = agent.finalize(result, Turn(user_input="question"))

    assert finalized.validation_checked is True
    assert finalized.validation_mismatch is False
    assert finalized.validation_reason is None


def test_validation_agent_handles_invalid_schema() -> None:
    """Invalid JSON schema should mark validation as unavailable."""
    llm = _FakeLLM('{"reason":"missing booleans"}')
    agent = ResultValidationResponseAgent(enabled=True, llm=llm)
    result = RetrievalResult(answer="answer", sources=[{"content": "source"}])

    finalized = agent.finalize(result, Turn(user_input="question"))

    assert finalized.validation_checked is False
    assert finalized.validation_mismatch is None
    assert finalized.validation_reason == "Validation model returned invalid schema."


def test_validation_agent_document_coverage_does_not_override_relevance() -> None:
    """Document-level coverage should not suppress a relevance mismatch."""
    llm = _FakeLLM(
        '{"summary_grounded": true, "sources_relevant": false, "reason":"partial source fit"}'
    )
    agent = ResultValidationResponseAgent(enabled=True, llm=llm)
    result = RetrievalResult(
        answer="answer",
        sources=[{"text": "source"}],
        summary_diagnostics={
            "total_documents": 10,
            "covered_documents": 8,
            "coverage_ratio": 0.8,
            "coverage_target": 0.7,
            "uncovered_documents": ["doc9.pdf", "doc10.pdf"],
        },
    )

    finalized = agent.finalize(result, Turn(user_input="summarize collection"))

    assert finalized.validation_checked is True
    assert finalized.validation_mismatch is True
    assert finalized.validation_reason == "partial source fit"


def test_validation_agent_post_coverage_can_override_relevance() -> None:
    """Post-level coverage may suppress overly strict relevance mismatches."""
    llm = _FakeLLM(
        '{"summary_grounded": true, "sources_relevant": false, "reason":"partial source fit"}'
    )
    agent = ResultValidationResponseAgent(enabled=True, llm=llm)
    result = RetrievalResult(
        answer="answer",
        sources=[{"text": "source"}],
        summary_diagnostics={
            "total_documents": 10,
            "covered_documents": 8,
            "coverage_ratio": 0.8,
            "coverage_target": 0.7,
            "coverage_unit": "posts",
            "uncovered_documents": [],
        },
    )

    finalized = agent.finalize(result, Turn(user_input="summarize collection"))

    assert finalized.validation_checked is True
    assert finalized.validation_mismatch is False
    assert finalized.validation_reason is None


def test_validation_agent_summary_coverage_does_not_override_grounding() -> None:
    """Test that when summary coverage is below the target, but the LLM indicates that
    the summary is not grounded, it sets validation_mismatch to True and uses the LLM's
    reason. This ensures that grounding issues are still prioritized over coverage when
    both are present.
    """
    llm = _FakeLLM(
        '{"summary_grounded": false, "sources_relevant": true, "reason":"unsupported claim"}'
    )
    agent = ResultValidationResponseAgent(enabled=True, llm=llm)
    result = RetrievalResult(
        answer="answer",
        sources=[{"text": "source"}],
        summary_diagnostics={
            "total_documents": 10,
            "covered_documents": 8,
            "coverage_ratio": 0.8,
            "coverage_target": 0.7,
            "uncovered_documents": [],
        },
    )

    finalized = agent.finalize(result, Turn(user_input="summarize collection"))

    assert finalized.validation_checked is True
    assert finalized.validation_mismatch is True
    assert finalized.validation_reason == "unsupported claim"


def test_validation_agent_empty_response_skips_validation() -> None:
    """Empty validator output should mark validation as not checked without error."""
    llm = _FakeLLM("")
    agent = ResultValidationResponseAgent(enabled=True, llm=llm)
    result = RetrievalResult(answer="answer", sources=[{"text": "source evidence"}])

    finalized = agent.finalize(result, Turn(user_input="question"))

    assert finalized.validation_checked is False
    assert finalized.validation_mismatch is None
    assert finalized.validation_reason == "Validation model returned empty output."


def test_validation_agent_non_json_response_skips_validation() -> None:
    """Non-JSON validator output should be treated as validation unavailable."""
    llm = _FakeLLM("not-json")
    agent = ResultValidationResponseAgent(enabled=True, llm=llm)
    result = RetrievalResult(answer="answer", sources=[{"text": "source evidence"}])

    finalized = agent.finalize(result, Turn(user_input="question"))

    assert finalized.validation_checked is False
    assert finalized.validation_mismatch is None
    assert finalized.validation_reason == "Validation model returned non-JSON output."


def test_validation_agent_without_model_reports_unavailable_reason() -> None:
    """Missing validator model should yield an explicit unavailable reason."""
    agent = ResultValidationResponseAgent(enabled=True, llm=None)
    result = RetrievalResult(answer="answer", sources=[{"text": "source evidence"}])

    finalized = agent.finalize(result, Turn(user_input="question"))

    assert finalized.validation_checked is False
    assert finalized.validation_mismatch is None
    assert finalized.validation_reason == "Validation model unavailable."


# ---------------------------------------------------------------------------
# New tests for refactor/response-validation-context changes
# ---------------------------------------------------------------------------


def test_retrieval_query_rendered_when_differs_from_user_input() -> None:
    """Retrieval query should appear in prompt when it differs from user input.

    Arranges a RetrievalResult where retrieval_query differs from the Turn's
    user_input, builds the prompt via a fake LLM call, and asserts both the
    "User query" and "Retrieval query (after rewrite)" lines are present.
    """
    llm = _FakeLLM(
        '{"summary_grounded": true, "sources_relevant": true, "reason":"ok"}'
    )
    agent = ResultValidationResponseAgent(enabled=True, llm=llm)
    result = RetrievalResult(
        answer="some answer",
        sources=[{"text": "source"}],
        retrieval_query="rewritten thing",
    )

    agent.finalize(result, Turn(user_input="original thing"))

    assert llm.last_prompt is not None
    assert "User query:\noriginal thing" in llm.last_prompt
    assert "Retrieval query (after rewrite):\nrewritten thing" in llm.last_prompt


def test_retrieval_query_not_rendered_when_equal_to_user_input() -> None:
    """Retrieval query line must be suppressed when it equals the user input.

    When retrieval_query and Turn.user_input are the same string (whitespace-
    insensitive), the "Retrieval query (after rewrite)" line must not appear in
    the prompt.
    """
    llm = _FakeLLM(
        '{"summary_grounded": true, "sources_relevant": true, "reason":"ok"}'
    )
    agent = ResultValidationResponseAgent(enabled=True, llm=llm)
    result = RetrievalResult(
        answer="some answer",
        sources=[{"text": "source"}],
        retrieval_query="same thing",
    )

    agent.finalize(result, Turn(user_input="same thing"))

    assert llm.last_prompt is not None
    assert "Retrieval query (after rewrite)" not in llm.last_prompt


def test_retrieval_query_not_rendered_when_case_only_difference() -> None:
    """Case-only differences between user input and retrieval query must not render.

    A rewrite that only changes casing (e.g. ``"EU Policy"`` → ``"eu policy"``)
    is not a meaningful rewrite, so the validator prompt must not include a
    "Retrieval query (after rewrite)" line.
    """
    llm = _FakeLLM(
        '{"summary_grounded": true, "sources_relevant": true, "reason":"ok"}'
    )
    agent = ResultValidationResponseAgent(enabled=True, llm=llm)
    result = RetrievalResult(
        answer="some answer",
        sources=[{"text": "source"}],
        retrieval_query="eu policy",
    )

    agent.finalize(result, Turn(user_input="EU Policy"))

    assert llm.last_prompt is not None
    assert "Retrieval query (after rewrite)" not in llm.last_prompt


def test_rewritten_query_used_as_fallback_when_retrieval_query_is_none() -> None:
    """rewritten_query is used when retrieval_query is None and both differ from user input.

    Arranges a RetrievalResult with retrieval_query=None but rewritten_query set,
    then asserts the prompt renders the rewrite line using the rewritten_query value.
    """
    llm = _FakeLLM(
        '{"summary_grounded": true, "sources_relevant": true, "reason":"ok"}'
    )
    agent = ResultValidationResponseAgent(enabled=True, llm=llm)
    result = RetrievalResult(
        answer="some answer",
        sources=[{"text": "source"}],
        retrieval_query=None,
        rewritten_query="rewrite",
    )

    agent.finalize(result, Turn(user_input="original"))

    assert llm.last_prompt is not None
    assert "Retrieval query (after rewrite):\nrewrite" in llm.last_prompt


def test_intent_and_tool_rendered_in_routing_line() -> None:
    """Intent and tool_used should appear together on a "Routing:" line in the prompt.

    Sets intent="rag_chat" and tool_used="rag_chat" on a RetrievalResult and asserts
    the rendered prompt contains the expected routing metadata.
    """
    llm = _FakeLLM(
        '{"summary_grounded": true, "sources_relevant": true, "reason":"ok"}'
    )
    agent = ResultValidationResponseAgent(enabled=True, llm=llm)
    result = RetrievalResult(
        answer="some answer",
        sources=[{"text": "source"}],
        intent="rag_chat",
        tool_used="rag_chat",
    )

    agent.finalize(result, Turn(user_input="question"))

    assert llm.last_prompt is not None
    assert "Routing: intent=rag_chat, tool=rag_chat" in llm.last_prompt


def test_empty_sources_short_circuits_without_calling_llm() -> None:
    """An answer with no sources must be flagged as mismatched without invoking the LLM.

    Verifies the short-circuit path: validation_checked=True, validation_mismatch=True,
    the canonical reason string, and that the LLM receives zero calls.
    """
    llm = _FakeLLM(
        '{"summary_grounded": true, "sources_relevant": true, "reason":"ok"}'
    )
    agent = ResultValidationResponseAgent(enabled=True, llm=llm)
    result = RetrievalResult(answer="hi", sources=[])

    finalized = agent.finalize(result, Turn(user_input="question"))

    assert llm.calls == 0
    assert finalized.validation_checked is True
    assert finalized.validation_mismatch is True
    assert finalized.validation_reason == "Answer produced without retrieved sources."


def test_all_new_fields_none_produces_original_prompt_structure() -> None:
    """When new context fields are all None the prompt retains its original structure.

    Regression guard: a RetrievalResult with retrieval_query, rewritten_query, intent,
    and tool_used all None must still produce a prompt that contains the user query and
    answer blocks and must NOT contain the new "Retrieval query (after rewrite)" or
    "Routing:" lines.
    """
    llm = _FakeLLM(
        '{"summary_grounded": true, "sources_relevant": true, "reason":"ok"}'
    )
    agent = ResultValidationResponseAgent(enabled=True, llm=llm)
    result = RetrievalResult(
        answer="answer",
        sources=[{"text": "source"}],
        retrieval_query=None,
        rewritten_query=None,
        intent=None,
        tool_used=None,
    )

    agent.finalize(result, Turn(user_input="question"))

    assert llm.last_prompt is not None
    assert "User query:\nquestion" in llm.last_prompt
    assert "Answer:\nanswer" in llm.last_prompt
    assert "Retrieval query (after rewrite)" not in llm.last_prompt
    assert "Routing:" not in llm.last_prompt
