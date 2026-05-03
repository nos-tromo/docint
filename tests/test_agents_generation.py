"""Tests for :class:`ResultValidationResponseAgent` in the generation module."""

from docint.agents.generation import MAX_SOURCE_CHARS, ResultValidationResponseAgent
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


def test_validation_agent_prompt_includes_reference_metadata() -> None:
    """Reference-metadata fields must reach the validator LLM prompt.

    Regression test for false-positive ungrounded verdicts on answers that cite
    social-post metadata (Network, UUID, Timestamp, Author, ...) which lives in
    ``source["reference_metadata"]`` rather than ``source["text"]``.
    """
    llm = _FakeLLM(
        '{"summary_grounded": true, "sources_relevant": true, "reason":"ok"}'
    )
    agent = ResultValidationResponseAgent(enabled=True, llm=llm)
    source = {
        "text": "post body",
        "filename": "table_socials.csv",
        "row": 26,
        "reference_metadata": {
            "network": "Facebook",
            "type": "posting",
            "uuid": "2d2425aeadfd4ca5a2cddd2d3b8e27cb",
            "timestamp": "2025-09-09 17:47:50.000000",
            "author": "Wolfgang Krieger",
            "author_id": "100007940942252",
            "vanity": "krieger.advokat",
            "text_id": "b9613b34-d488-565d-a4bf-af7b9d1de212",
        },
    }
    result = RetrievalResult(answer="answer", sources=[source])

    agent.finalize(result, Turn(user_input="who posted this?"))

    assert llm.last_prompt is not None
    prompt = llm.last_prompt
    for expected in (
        "Facebook",
        "2d2425aeadfd4ca5a2cddd2d3b8e27cb",
        "2025-09-09 17:47:50.000000",
        "Wolfgang Krieger",
        "100007940942252",
        "krieger.advokat",
        "b9613b34-d488-565d-a4bf-af7b9d1de212",
        "table_socials.csv",
        "row=26",
        "post body",
    ):
        assert expected in prompt, f"missing {expected!r} in validator prompt"


def test_validation_agent_prompt_metadata_block_not_truncated() -> None:
    """Metadata fields must survive even when the text body exceeds the per-source cap."""
    llm = _FakeLLM(
        '{"summary_grounded": true, "sources_relevant": true, "reason":"ok"}'
    )
    agent = ResultValidationResponseAgent(enabled=True, llm=llm)
    long_body = "x" * 5000
    source = {
        "text": long_body,
        "filename": "big.csv",
        "row": 1,
        "reference_metadata": {
            "network": "Facebook",
            "uuid": "deadbeef",
            "author": "Alice",
        },
    }
    result = RetrievalResult(answer="a", sources=[source])

    agent.finalize(result, Turn(user_input="q"))

    assert llm.last_prompt is not None
    prompt = llm.last_prompt
    assert "Facebook" in prompt
    assert "deadbeef" in prompt
    assert "Alice" in prompt
    # Body sliced to MAX_SOURCE_CHARS — exactly that many consecutive x's, no more.
    assert ("x" * MAX_SOURCE_CHARS) in prompt
    assert ("x" * (MAX_SOURCE_CHARS + 1)) not in prompt


def test_validation_agent_prompt_handles_text_only_source() -> None:
    """Sources without reference_metadata must still produce a well-formed prompt."""
    llm = _FakeLLM(
        '{"summary_grounded": true, "sources_relevant": true, "reason":"ok"}'
    )
    agent = ResultValidationResponseAgent(enabled=True, llm=llm)
    result = RetrievalResult(
        answer="a", sources=[{"text": "plain body without metadata"}]
    )

    agent.finalize(result, Turn(user_input="q"))

    assert llm.last_prompt is not None
    prompt = llm.last_prompt
    assert "Source 1" in prompt
    assert "plain body without metadata" in prompt
    assert "- Network:" not in prompt
    assert "- UUID:" not in prompt


def test_validation_agent_prompt_includes_metadata_text_when_no_top_level_body() -> (
    None
):
    """Sources whose body lives inside ``reference_metadata['text']`` must still
    surface the body to the validator — covers transcript-style payloads where
    the text may not be propagated as a top-level key."""
    llm = _FakeLLM(
        '{"summary_grounded": true, "sources_relevant": true, "reason":"ok"}'
    )
    agent = ResultValidationResponseAgent(enabled=True, llm=llm)
    result = RetrievalResult(
        answer="a",
        sources=[
            {
                "filename": "seg.jsonl",
                "row": 3,
                "reference_metadata": {
                    "speaker": "Alice",
                    "language": "en",
                    "text": "spoken segment body",
                },
            }
        ],
    )

    agent.finalize(result, Turn(user_input="q"))

    assert llm.last_prompt is not None
    prompt = llm.last_prompt
    assert "Alice" in prompt
    assert "en" in prompt
    assert "spoken segment body" in prompt
