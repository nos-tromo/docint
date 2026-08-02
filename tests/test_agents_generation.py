"""Tests for :class:`ResultValidationResponseAgent` in the generation module."""

import re
from typing import Any, cast

import pytest
from loguru import logger

from docint.agents.generation import ResultValidationResponseAgent
from docint.agents.types import RetrievalResult, Turn

MARKER = "MARKER-SECRET-1234"


@pytest.fixture(autouse=True)
def _pin_response_language_to_english(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin ``RESPONSE_LANGUAGE`` to ``en`` so prompt-text assertions are stable.

    The validator template is locale-aware (see ``prompts/{en,de}/response_validator.txt``);
    these tests assert on the English text of that template, so they would
    flake under ``RESPONSE_LANGUAGE=de`` deployments without this pin.
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
        return _FakeLLMResponse(text=self.response_text)


def test_validation_agent_sets_alert_on_mismatch() -> None:
    """Grounding mismatch from the LLM should set the validation alert flag."""
    llm = _FakeLLM('{"summary_grounded": false, "sources_relevant": true, "reason":"hallucinated fact"}')
    agent = ResultValidationResponseAgent(enabled=True, llm=cast(Any, llm))
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
    llm = _FakeLLM('{"summary_grounded": false, "sources_relevant": false, "reason":"bad"}')
    agent = ResultValidationResponseAgent(enabled=False, llm=cast(Any, llm))
    result = RetrievalResult(answer="answer", sources=[{"text": "source evidence"}])

    finalized = agent.finalize(result, Turn(user_input="question"))

    assert llm.calls == 0
    assert finalized.validation_checked is None
    assert finalized.validation_mismatch is None
    assert finalized.validation_reason is None


def test_validation_agent_parses_markdown_wrapped_json() -> None:
    """Markdown-fenced JSON from the LLM should be unwrapped and parsed."""
    llm = _FakeLLM('```json\n{"summary_grounded": true, "sources_relevant": true, "reason":"ok"}\n```')
    agent = ResultValidationResponseAgent(enabled=True, llm=cast(Any, llm))
    result = RetrievalResult(answer="answer", sources=[{"id": 1, "content": "source"}])

    finalized = agent.finalize(result, Turn(user_input="question"))

    assert finalized.validation_checked is True
    assert finalized.validation_mismatch is False
    assert finalized.validation_reason is None


def test_validation_agent_handles_invalid_schema() -> None:
    """Invalid JSON schema should mark validation as unavailable."""
    llm = _FakeLLM('{"reason":"missing booleans"}')
    agent = ResultValidationResponseAgent(enabled=True, llm=cast(Any, llm))
    result = RetrievalResult(answer="answer", sources=[{"content": "source"}])

    finalized = agent.finalize(result, Turn(user_input="question"))

    assert finalized.validation_checked is False
    assert finalized.validation_mismatch is None
    assert finalized.validation_reason == "Validation model returned invalid schema."


def test_validation_agent_document_coverage_does_not_override_relevance() -> None:
    """Document-level coverage should not suppress a relevance mismatch."""
    llm = _FakeLLM('{"summary_grounded": true, "sources_relevant": false, "reason":"partial source fit"}')
    agent = ResultValidationResponseAgent(enabled=True, llm=cast(Any, llm))
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
    llm = _FakeLLM('{"summary_grounded": true, "sources_relevant": false, "reason":"partial source fit"}')
    agent = ResultValidationResponseAgent(enabled=True, llm=cast(Any, llm))
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
    """Low summary coverage must not override an LLM grounding failure.

    When the LLM marks the summary as not grounded, validation_mismatch must be True with the
    LLM's reason — grounding issues are prioritized over coverage shortfalls.
    """
    llm = _FakeLLM('{"summary_grounded": false, "sources_relevant": true, "reason":"unsupported claim"}')
    agent = ResultValidationResponseAgent(enabled=True, llm=cast(Any, llm))
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
    agent = ResultValidationResponseAgent(enabled=True, llm=cast(Any, llm))
    result = RetrievalResult(answer="answer", sources=[{"text": "source evidence"}])

    finalized = agent.finalize(result, Turn(user_input="question"))

    assert finalized.validation_checked is False
    assert finalized.validation_mismatch is None
    assert finalized.validation_reason == "Validation model returned empty output."


def test_validation_agent_non_json_response_skips_validation() -> None:
    """Non-JSON validator output should be treated as validation unavailable."""
    llm = _FakeLLM("not-json")
    agent = ResultValidationResponseAgent(enabled=True, llm=cast(Any, llm))
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


def test_validation_agent_request_exception_reason_is_generic_and_logged() -> None:
    """An exception from the validator LLM must not leak into ``validation_reason``.

    ``validation_reason`` persists to the session DB and is returned via the
    API, so it must be static text; the real exception detail belongs in logs
    only.
    """

    class _RaisingLLM:
        """Fake LLM whose ``complete`` always raises."""

        def complete(self, prompt: str) -> Any:
            """Raise a marked exception instead of returning a response.

            Args:
                prompt: The prompt string (ignored).
            """
            raise RuntimeError(MARKER)

    agent = ResultValidationResponseAgent(enabled=True, llm=cast(Any, _RaisingLLM()))
    result = RetrievalResult(answer="answer", sources=[{"text": "source evidence"}])

    records: list[str] = []
    sink_id = logger.add(lambda m: records.append(str(m)), level="DEBUG")
    try:
        finalized = agent.finalize(result, Turn(user_input="question"))
    finally:
        logger.remove(sink_id)

    assert finalized.validation_checked is False
    assert finalized.validation_mismatch is None
    assert finalized.validation_reason == "Validation request failed."
    assert MARKER not in (finalized.validation_reason or "")
    assert any(MARKER in r for r in records)


def test_validation_agent_prompt_includes_reference_metadata() -> None:
    """Reference-metadata fields must reach the validator LLM prompt.

    Regression test for false-positive ungrounded verdicts on answers that cite
    social-post metadata (Network, UUID, Timestamp, Author, ...) which lives in
    ``source["reference_metadata"]`` rather than ``source["text"]``.
    """
    llm = _FakeLLM('{"summary_grounded": true, "sources_relevant": true, "reason":"ok"}')
    agent = ResultValidationResponseAgent(enabled=True, llm=cast(Any, llm))
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


def test_validation_agent_prompt_metadata_block_not_truncated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Metadata fields must survive even when the text body eats the budget.

    Args:
        monkeypatch (pytest.MonkeyPatch): Environment patcher.
    """
    monkeypatch.setenv("RESPONSE_VALIDATION_SOURCE_BUDGET_CHARS", "1200")
    llm = _FakeLLM('{"summary_grounded": true, "sources_relevant": true, "reason":"ok"}')
    agent = ResultValidationResponseAgent(enabled=True, llm=cast(Any, llm))
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
    # Body sliced to the budget — exactly that many consecutive x's, no more.
    assert ("x" * 1200) in prompt
    assert ("x" * 1201) not in prompt


def test_validation_agent_prompt_handles_text_only_source() -> None:
    """Sources without reference_metadata must still produce a well-formed prompt."""
    llm = _FakeLLM('{"summary_grounded": true, "sources_relevant": true, "reason":"ok"}')
    agent = ResultValidationResponseAgent(enabled=True, llm=cast(Any, llm))
    result = RetrievalResult(answer="a", sources=[{"text": "plain body without metadata"}])

    agent.finalize(result, Turn(user_input="q"))

    assert llm.last_prompt is not None
    prompt = llm.last_prompt
    assert "Source 1" in prompt
    assert "plain body without metadata" in prompt
    assert "- Network:" not in prompt
    assert "- UUID:" not in prompt


def test_validation_agent_prompt_includes_metadata_text_when_no_top_level_body() -> None:
    """Sources whose body lives inside ``reference_metadata['text']`` must still validate.

    Covers transcript-style payloads where the text may not be propagated as a top-level key —
    the validator should still see the body.
    """
    llm = _FakeLLM('{"summary_grounded": true, "sources_relevant": true, "reason":"ok"}')
    agent = ResultValidationResponseAgent(enabled=True, llm=cast(Any, llm))
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


# ---------------------------------------------------------------------------
# Source budget
#
# The generator sees whole chunks; the validator used to see a fixed 1200
# characters of each. Anything the answer drew from further in was reported as
# a hallucination — a false mismatch on a correct, grounded answer.
# ---------------------------------------------------------------------------


def _body_runs(prompt: str, filler: str) -> list[int]:
    """Return the length of each rendered filler-body run in the prompt.

    Counting bare occurrences would also pick up the letter as it appears in
    the prompt's own English text; a run of the synthetic filler is
    unambiguously body text.

    Args:
        prompt (str): The full validator prompt.
        filler (str): The single filler character the body is built from.

    Returns:
        list[int]: Length of every run longer than one character, in order.
    """
    return [len(run) for run in re.findall(f"{filler}+", prompt) if len(run) > 1]


def _ok_llm() -> _FakeLLM:
    """Build a fake LLM that validates everything as grounded.

    Returns:
        _FakeLLM: The canned validator LLM.
    """
    return _FakeLLM('{"summary_grounded": true, "sources_relevant": true, "reason":"ok"}')


def test_validation_prompt_shows_source_text_past_the_legacy_cap() -> None:
    """Evidence deep inside a chunk reaches the validator."""
    llm = _ok_llm()
    agent = ResultValidationResponseAgent(enabled=True, llm=cast(Any, llm))
    body = ("x" * 4000) + MARKER + ("x" * 4000)
    result = RetrievalResult(answer="a", sources=[{"filename": "doc.pdf", "text": body}])

    agent.finalize(result, Turn(user_input="q"))

    assert llm.last_prompt is not None
    assert MARKER in llm.last_prompt


def test_validation_prompt_source_text_stays_within_the_total_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Many long sources share one bounded budget rather than each getting a cap.

    Args:
        monkeypatch (pytest.MonkeyPatch): Environment patcher.
    """
    monkeypatch.setenv("RESPONSE_VALIDATION_SOURCE_BUDGET_CHARS", "4000")
    llm = _ok_llm()
    agent = ResultValidationResponseAgent(enabled=True, llm=cast(Any, llm))
    sources = [{"filename": f"doc{i}.pdf", "text": "x" * 10_000} for i in range(4)]
    result = RetrievalResult(answer="a", sources=cast(Any, sources))

    agent.finalize(result, Turn(user_input="q"))

    assert llm.last_prompt is not None
    shown = _body_runs(llm.last_prompt, "x")
    assert shown == [1000, 1000, 1000, 1000]


def test_validation_prompt_redistributes_unused_budget_to_long_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A short source leaves its unused share to the long one.

    Args:
        monkeypatch (pytest.MonkeyPatch): Environment patcher.
    """
    monkeypatch.setenv("RESPONSE_VALIDATION_SOURCE_BUDGET_CHARS", "4000")
    llm = _ok_llm()
    agent = ResultValidationResponseAgent(enabled=True, llm=cast(Any, llm))
    result = RetrievalResult(
        answer="a",
        sources=[
            {"filename": "short.pdf", "text": "y" * 100},
            {"filename": "long.pdf", "text": "x" * 10_000},
        ],
    )

    agent.finalize(result, Turn(user_input="q"))

    assert llm.last_prompt is not None
    assert _body_runs(llm.last_prompt, "y") == [100]
    assert _body_runs(llm.last_prompt, "x") == [3900]


def test_validation_prompt_marks_truncated_sources_and_warns_the_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A trimmed source is labelled, and the prompt says unseen text is not a hallucination.

    Args:
        monkeypatch (pytest.MonkeyPatch): Environment patcher.
    """
    monkeypatch.setenv("RESPONSE_VALIDATION_SOURCE_BUDGET_CHARS", "100")
    llm = _ok_llm()
    agent = ResultValidationResponseAgent(enabled=True, llm=cast(Any, llm))
    result = RetrievalResult(answer="a", sources=[{"filename": "doc.pdf", "text": "x" * 500}])

    agent.finalize(result, Turn(user_input="q"))

    assert llm.last_prompt is not None
    prompt = llm.last_prompt
    assert "400" in prompt, "the hidden character count must be stated"
    assert "truncat" in prompt.lower()


def test_validation_prompt_has_no_truncation_warning_when_everything_fits() -> None:
    """A fully-shown source set carries no truncation caveat."""
    llm = _ok_llm()
    agent = ResultValidationResponseAgent(enabled=True, llm=cast(Any, llm))
    result = RetrievalResult(answer="a", sources=[{"filename": "doc.pdf", "text": "short body"}])

    agent.finalize(result, Turn(user_input="q"))

    assert llm.last_prompt is not None
    assert "truncat" not in llm.last_prompt.lower()
