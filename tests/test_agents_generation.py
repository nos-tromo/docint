from docint.agents.generation import ResultValidationResponseAgent
from docint.agents.types import RetrievalResult, Turn


class _FakeLLMResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeLLM:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls = 0
        self.last_prompt: str | None = None

    def complete(self, prompt: str) -> _FakeLLMResponse:
        self.calls += 1
        self.last_prompt = prompt
        return _FakeLLMResponse(text=self.response_text)


def test_validation_agent_sets_alert_on_mismatch() -> None:
    llm = _FakeLLM(
        '{"summary_grounded": false, "sources_relevant": true, "reason":"hallucinated fact"}'
    )
    agent = ResultValidationResponseAgent(enabled=True, llm=llm)
    result = RetrievalResult(answer="answer", sources=[{"text": "source evidence"}])

    finalized = agent.finalize(result, Turn(user_input="question"))

    assert llm.calls == 1
    assert llm.last_prompt is not None
    assert "Query:\nquestion" in llm.last_prompt
    assert "Answer:\nanswer" in llm.last_prompt
    assert finalized.validation_checked is True
    assert finalized.validation_mismatch is True
    assert finalized.validation_reason == "hallucinated fact"


def test_validation_agent_disabled_is_noop() -> None:
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
    llm = _FakeLLM('{"reason":"missing booleans"}')
    agent = ResultValidationResponseAgent(enabled=True, llm=llm)
    result = RetrievalResult(answer="answer", sources=[{"content": "source"}])

    finalized = agent.finalize(result, Turn(user_input="question"))

    assert finalized.validation_checked is False
    assert finalized.validation_mismatch is True
    assert finalized.validation_reason == "Validation model returned invalid schema"


def test_validation_agent_summary_coverage_threshold_overrides_relevance() -> None:
    """Test that when summary coverage is below the target, it sets validation_mismatch
    to True regardless of the grounding and relevance flags returned by the LLM. This
    ensures that coverage issues are prioritized in the validation logic.
    """
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
