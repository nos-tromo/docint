"""Tests for :class:`AgentOrchestrator` turn-handling logic."""

from dataclasses import dataclass
from typing import Any, cast

import pytest
from typing_extensions import override

from docint.agents import (
    AgentOrchestrator,
    ClarificationConfig,
    ClarificationPolicy,
    ClarificationRequest,
    OrchestratorResult,
    RetrievalRequest,
    RetrievalResult,
    SimpleClarificationAgent,
    SimpleUnderstandingAgent,
    Turn,
    is_weak_answer,
)
from docint.agents.orchestrator import WEAK_ANSWER_MIN_CHARS
from docint.agents.types import ResponseAgent, RetrievalAgent
from docint.agents.types import Turn as TurnType


@dataclass
class _DummyRetrievalResult:
    """Minimal retrieval result dataclass for testing."""

    answer: str
    sources: list[dict[str, Any]]


class _StubRetrievalAgent(RetrievalAgent):
    """Retrieval agent that records the request and returns a canned result."""

    def __init__(self) -> None:
        """Initialise with no recorded request."""
        self.called_with: RetrievalRequest | None = None

    @override
    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """Record the request and return a fixed retrieval result.

        Args:
            request: The retrieval request from the orchestrator.

        Returns:
            A canned ``RetrievalResult`` long enough to clear weak-answer detection.
        """
        self.called_with = request
        return RetrievalResult(
            answer=(
                "This is a substantive stub answer for the retrieval agent — "
                "long enough to clear the weak-answer threshold."
            ),
            sources=[{"id": 1}],
            session_id="s1",
        )


class _StubResponseAgent(ResponseAgent):
    """Response agent that marks validation fields on the result."""

    def __init__(self) -> None:
        """Initialise with the call flag unset."""
        self.called = False

    @override
    def finalize(self, result: RetrievalResult, turn: TurnType) -> RetrievalResult:
        """Set validation flags on the result and record the call.

        Args:
            result: The retrieval result to finalize.
            turn: The current conversation turn.

        Returns:
            The modified ``RetrievalResult`` with validation fields set.
        """
        _ = turn
        self.called = True
        result.validation_checked = True
        result.validation_mismatch = True
        result.validation_reason = "mismatch"
        return result


class _NoopClarifier(SimpleClarificationAgent):
    """Clarifier that always requests clarification."""

    @override
    def build(self, turn: TurnType, analysis: Any) -> ClarificationRequest:
        """Return a clarification request regardless of input.

        Args:
            turn: The current conversation turn.
            analysis: The intent analysis (unused).

        Returns:
            A ``ClarificationRequest`` with ``needed=True``.
        """
        _ = turn, analysis
        return ClarificationRequest(needed=True, message="clarify")


class _AlwaysClarifyPolicy(ClarificationPolicy):
    """Clarification policy that always triggers clarification."""

    def __init__(self) -> None:
        """Initialise with a strict confidence threshold."""
        super().__init__(ClarificationConfig(confidence_threshold=1.0, require_entities=True))

    @override
    def evaluate(self, analysis: Any, clarifications_so_far: int = 0) -> ClarificationRequest:
        """Always request clarification.

        Args:
            analysis: The intent analysis (unused).
            clarifications_so_far: Number of prior clarifications (unused).

        Returns:
            A ``ClarificationRequest`` with ``needed=True``.
        """
        _ = analysis, clarifications_so_far
        return ClarificationRequest(needed=True, message="clarify", reason="force")


class _NeverClarifyPolicy(ClarificationPolicy):
    """Clarification policy that never triggers clarification."""

    def __init__(self) -> None:
        """Initialise with a permissive confidence threshold."""
        super().__init__(ClarificationConfig(confidence_threshold=0.0, require_entities=False))


@pytest.fixture
def turn() -> Turn:
    """Create a simple conversation turn fixture.

    Returns:
        A ``Turn`` with a basic user input and session ID.
    """
    return Turn(user_input="hello", session_id="s1")


def test_orchestrator_requests_clarification(turn: Turn) -> None:
    """Low confidence should trigger a clarification request."""
    orchestrator = AgentOrchestrator(
        understanding=SimpleUnderstandingAgent(default_confidence=0.1),
        clarifier=_NoopClarifier(),
        retriever=_StubRetrievalAgent(),
        policy=_AlwaysClarifyPolicy(),
    )

    result: OrchestratorResult = orchestrator.handle_turn(turn)

    assert result.clarification is not None
    assert result.clarification.needed is True
    assert result.retrieval is None


def test_orchestrator_retrieves_when_confident(turn: Turn) -> None:
    """High confidence should bypass clarification and perform retrieval."""
    retriever = _StubRetrievalAgent()
    orchestrator = AgentOrchestrator(
        understanding=SimpleUnderstandingAgent(default_confidence=0.9),
        clarifier=_NoopClarifier(),
        retriever=retriever,
        policy=_NeverClarifyPolicy(),
    )

    result: OrchestratorResult = orchestrator.handle_turn(turn)

    assert result.clarification is None
    assert result.retrieval is not None
    assert "substantive stub answer" in (result.retrieval.answer or "")
    assert retriever.called_with is not None
    assert retriever.called_with.turn is turn


def test_orchestrator_runs_response_agent(turn: Turn) -> None:
    """Response agent should be invoked after retrieval to finalize the result."""
    retriever = _StubRetrievalAgent()
    responder = _StubResponseAgent()
    orchestrator = AgentOrchestrator(
        understanding=SimpleUnderstandingAgent(default_confidence=0.9),
        clarifier=_NoopClarifier(),
        retriever=retriever,
        responder=responder,
        policy=_NeverClarifyPolicy(),
    )

    result = orchestrator.handle_turn(turn)

    assert result.retrieval is not None
    assert responder.called is True
    assert result.retrieval.validation_checked is True
    assert result.retrieval.validation_mismatch is True
    assert result.retrieval.validation_reason == "mismatch"


class _WeakAnswerRetrievalAgent(RetrievalAgent):
    """Retrieval agent that returns a degenerate ``"Evidence insufficient."`` answer."""

    @override
    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """Return a refusal-shaped answer to exercise the fallback path.

        Args:
            request: The retrieval request from the orchestrator.

        Returns:
            A canned weak ``RetrievalResult``.
        """
        _ = request
        return RetrievalResult(
            answer="Evidence insufficient.",
            sources=[],
            session_id="s1",
        )


class _MismatchResponseAgent(ResponseAgent):
    """Response agent that flags ``validation_mismatch=True``."""

    @override
    def finalize(self, result: RetrievalResult, turn: TurnType) -> RetrievalResult:
        """Set mismatch metadata on the result.

        Args:
            result: The retrieval result to annotate.
            turn: The current conversation turn (unused).

        Returns:
            The annotated retrieval result.
        """
        _ = turn
        result.validation_checked = True
        result.validation_mismatch = True
        result.validation_reason = "no UN content in sources"
        return result


def test_orchestrator_falls_back_to_clarification_on_weak_mismatched_answer(
    turn: Turn,
) -> None:
    """A weak answer with validation_mismatch=True should convert to a clarification."""
    orchestrator = AgentOrchestrator(
        understanding=SimpleUnderstandingAgent(default_confidence=0.9),
        clarifier=_NoopClarifier(),
        retriever=_WeakAnswerRetrievalAgent(),
        responder=_MismatchResponseAgent(),
        policy=_NeverClarifyPolicy(),
    )

    result = orchestrator.handle_turn(turn)

    assert result.clarification is not None
    assert result.clarification.needed is True
    assert result.clarification.reason == "weak_answer_after_validation_mismatch"
    assert result.retrieval is None


def test_orchestrator_keeps_strong_answer_even_with_mismatch(turn: Turn) -> None:
    """A long, substantive answer should be returned even when validator flags mismatch.

    The fallback is gated on BOTH ``validation_mismatch=True`` AND a weak
    answer signal (short text or refusal phrase). The default
    ``_StubRetrievalAgent`` returns ``answer="ok"`` (3 chars) which IS weak;
    use a richer stub here to assert the strong-answer path is preserved.
    """

    class _StrongRetriever(RetrievalAgent):
        @override
        def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
            _ = request
            return RetrievalResult(
                answer=(
                    "Hamas's stance is described as oppositional toward Jews "
                    "and Christians, with multiple cited passages spanning "
                    "pages 13 and 67 of the source corpus."
                ),
                sources=[{"id": 1}],
                session_id="s1",
            )

    orchestrator = AgentOrchestrator(
        understanding=SimpleUnderstandingAgent(default_confidence=0.9),
        clarifier=_NoopClarifier(),
        retriever=_StrongRetriever(),
        responder=_MismatchResponseAgent(),
        policy=_NeverClarifyPolicy(),
    )

    result = orchestrator.handle_turn(turn)

    assert result.clarification is None
    assert result.retrieval is not None
    assert result.retrieval.validation_mismatch is True


def test_orchestrator_respects_max_clarifications_for_fallback(turn: Turn) -> None:
    """Once max_clarifications is reached, the weak result is returned as-is."""
    from docint.agents.context import TurnContext

    orchestrator = AgentOrchestrator(
        understanding=SimpleUnderstandingAgent(default_confidence=0.9),
        clarifier=_NoopClarifier(),
        retriever=_WeakAnswerRetrievalAgent(),
        responder=_MismatchResponseAgent(),
        policy=_NeverClarifyPolicy(),
    )

    # The default ClarificationConfig.max_clarifications is 2; saturate it.
    ctx = TurnContext(session_id=turn.session_id, clarifications=2)
    result = orchestrator.handle_turn(turn, context=ctx)

    assert result.clarification is None
    assert result.retrieval is not None
    assert result.retrieval.answer == "Evidence insufficient."


class _StubReformulator:
    """Reformulator returning a canned query and recording its inputs."""

    def __init__(self, query: str | None = "reformulated query") -> None:
        """Initialise with the query to hand back.

        Args:
            query: The reformulation to return, or ``None`` to decline.
        """
        self.query = query
        self.calls: list[dict[str, Any]] = []

    def reformulate(
        self,
        *,
        user_query: str,
        failed_query: str | None = None,
        validation_reason: str | None = None,
    ) -> str | None:
        """Record the call and return the canned query.

        Args:
            user_query: The user's original question.
            failed_query: The query that produced the rejected answer.
            validation_reason: The validator's rejection reason.

        Returns:
            The canned reformulation, or ``None``.
        """
        self.calls.append(
            {
                "user_query": user_query,
                "failed_query": failed_query,
                "validation_reason": validation_reason,
            }
        )
        return self.query


class _CountingWeakRetriever(RetrievalAgent):
    """Retrieval agent that records every request and answers weakly."""

    def __init__(self, answers: list[str] | None = None) -> None:
        """Initialise with the sequence of answers to return.

        Args:
            answers: Answers to return per call; the last one repeats.
        """
        self.requests: list[RetrievalRequest] = []
        self.answers = answers or ["Evidence insufficient."]

    @override
    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """Record the request and return the next canned answer.

        Args:
            request: The retrieval request from the orchestrator.

        Returns:
            A ``RetrievalResult`` carrying the next canned answer.
        """
        self.requests.append(request)
        answer = self.answers[min(len(self.requests) - 1, len(self.answers) - 1)]
        return RetrievalResult(
            answer=answer,
            sources=[],
            session_id="s1",
            retrieval_query="failed query",
            turn_idx=3,
        )


def _retry_orchestrator(
    retriever: RetrievalAgent,
    reformulator: _StubReformulator | None,
    responder: ResponseAgent | None = None,
) -> AgentOrchestrator:
    """Build an orchestrator wired for the corrective-retry path.

    Args:
        retriever: The retrieval agent under test.
        reformulator: The reformulator to inject, or ``None`` to disable retry.
        responder: Optional responder; defaults to a mismatch-flagging one.

    Returns:
        AgentOrchestrator: Orchestrator under test.
    """
    return AgentOrchestrator(
        understanding=SimpleUnderstandingAgent(default_confidence=0.9),
        clarifier=_NoopClarifier(),
        retriever=retriever,
        responder=responder if responder is not None else _MismatchResponseAgent(),
        policy=_NeverClarifyPolicy(),
        reformulator=cast(Any, reformulator),
    )


def test_orchestrator_retries_weak_mismatched_answer_with_reformulated_query(
    turn: Turn,
) -> None:
    """A weak, mismatched answer triggers one re-retrieval with a new query."""
    retriever = _CountingWeakRetriever(
        answers=[
            "Evidence insufficient.",
            "The Security Council adopted three resolutions on the matter in 2019.",
        ]
    )
    reformulator = _StubReformulator()

    result = _retry_orchestrator(retriever, reformulator).handle_turn(turn)

    assert len(retriever.requests) == 2
    assert retriever.requests[1].analysis.rewritten_query == "reformulated query"
    assert result.retrieval is not None
    assert result.retrieval.retried is True
    assert result.retrieval.retry_query == "reformulated query"
    assert result.retrieval.answer is not None
    assert "Security Council" in result.retrieval.answer


def test_orchestrator_retry_replaces_the_first_attempts_turn(turn: Turn) -> None:
    """The retry overwrites the persisted turn instead of appending one."""
    retriever = _CountingWeakRetriever(answers=["Evidence insufficient.", "A much better grounded answer here."])

    _retry_orchestrator(retriever, _StubReformulator()).handle_turn(turn)

    assert retriever.requests[0].replace_turn_idx is None
    assert retriever.requests[1].replace_turn_idx == 3


def test_orchestrator_retry_feeds_the_validation_reason_to_the_reformulator(
    turn: Turn,
) -> None:
    """The reformulator sees the question, the failed query, and the reason."""
    reformulator = _StubReformulator()

    _retry_orchestrator(_CountingWeakRetriever(), reformulator).handle_turn(turn)

    assert reformulator.calls == [
        {
            "user_query": "hello",
            "failed_query": "failed query",
            "validation_reason": "no UN content in sources",
        }
    ]


def test_orchestrator_revalidates_the_retry_answer(turn: Turn) -> None:
    """The responder runs again so the fallback judges what the user will see."""

    class _CountingResponder(ResponseAgent):
        """Responder that counts calls and flags a mismatch each time."""

        def __init__(self) -> None:
            """Initialise the call counter."""
            self.calls = 0

        @override
        def finalize(self, result: RetrievalResult, turn: TurnType) -> RetrievalResult:
            """Flag a mismatch and count the call.

            Args:
                result: The retrieval result to annotate.
                turn: The current conversation turn (unused).

            Returns:
                The annotated retrieval result.
            """
            _ = turn
            self.calls += 1
            result.validation_checked = True
            result.validation_mismatch = True
            result.validation_reason = "still off-topic"
            return result

    responder = _CountingResponder()
    retriever = _CountingWeakRetriever(answers=["Evidence insufficient.", "A much better grounded answer here."])

    _retry_orchestrator(retriever, _StubReformulator(), responder=responder).handle_turn(turn)

    assert responder.calls == 2


def test_orchestrator_falls_back_to_clarification_when_the_retry_also_fails(
    turn: Turn,
) -> None:
    """Two weak attempts still end in the clarification nudge."""
    retriever = _CountingWeakRetriever(answers=["Evidence insufficient."])

    result = _retry_orchestrator(retriever, _StubReformulator()).handle_turn(turn)

    assert len(retriever.requests) == 2
    assert result.retrieval is None
    assert result.clarification is not None
    assert result.clarification.reason == "weak_answer_after_validation_mismatch"


def test_orchestrator_skips_retry_when_the_reformulator_declines(turn: Turn) -> None:
    """No usable reformulation means the original fallback, not a second call."""
    retriever = _CountingWeakRetriever()

    result = _retry_orchestrator(retriever, _StubReformulator(query=None)).handle_turn(turn)

    assert len(retriever.requests) == 1
    assert result.clarification is not None
    assert result.clarification.reason == "weak_answer_after_validation_mismatch"


def test_orchestrator_does_not_retry_a_strong_mismatched_answer(turn: Turn) -> None:
    """A substantive answer is kept even when the validator flags it."""
    retriever = _CountingWeakRetriever(
        answers=["Hamas's stance is described across multiple cited passages spanning pages 13 and 67."]
    )
    reformulator = _StubReformulator()

    result = _retry_orchestrator(retriever, reformulator).handle_turn(turn)

    assert len(retriever.requests) == 1
    assert reformulator.calls == []
    assert result.retrieval is not None
    assert result.retrieval.retried is None


_CITED_CAVEAT_EN = (
    "The ferry left the north quay at nine [1] and reached the island before noon [2]. "
    "The log does not mention the captain's name."
)
_UNCITED_CAVEAT_DE = (
    "Wie zuvor beschrieben, fuhr die Fähre von Mai bis September täglich. "
    "Sie beförderte auf jeder Überfahrt Passagiere und Fracht. "
    "Zum Fahrpreis finden sich keine Angaben."
)


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param(_CITED_CAVEAT_EN, id="en-cited"),
        pytest.param(_UNCITED_CAVEAT_DE, id="de-uncited"),
    ],
)
def test_orchestrator_keeps_an_answer_that_flags_a_missing_detail(turn: Turn, answer: str) -> None:
    """A substantive answer naming one missing detail is neither retried nor replaced."""
    retriever = _CountingWeakRetriever(answers=[answer])
    reformulator = _StubReformulator()

    result = _retry_orchestrator(retriever, reformulator).handle_turn(turn)

    assert len(retriever.requests) == 1
    assert reformulator.calls == []
    assert result.clarification is None
    assert result.retrieval is not None
    assert result.retrieval.answer == answer
    assert result.retrieval.retried is None


def test_orchestrator_without_a_reformulator_never_retries(turn: Turn) -> None:
    """The retry is opt-in: no reformulator wired, no second retrieval."""
    retriever = _CountingWeakRetriever()

    result = _retry_orchestrator(retriever, None).handle_turn(turn)

    assert len(retriever.requests) == 1
    assert result.clarification is not None


def test_orchestrator_forwards_history_into_retrieval_request(turn: Turn) -> None:
    """``TurnContext.history`` must reach the retrieval agent via ``RetrievalRequest.history``."""
    from docint.agents.context import TurnContext

    retriever = _StubRetrievalAgent()
    orchestrator = AgentOrchestrator(
        understanding=SimpleUnderstandingAgent(default_confidence=0.9),
        clarifier=_NoopClarifier(),
        retriever=retriever,
        policy=_NeverClarifyPolicy(),
    )

    history = [
        {"role": "user", "content": "prior question"},
        {"role": "assistant", "content": "prior answer"},
    ]
    ctx = TurnContext(session_id=turn.session_id, history=history)
    orchestrator.handle_turn(turn, context=ctx)

    assert retriever.called_with is not None
    assert retriever.called_with.history == history


# ---------------------------------------------------------------------------
# Predicate: is_weak_answer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param(
            "Aus den bereitgestellten Notizen geht nicht hervor, welches Schiff zuerst anlegte.",
            id="de-geht-nicht-hervor",
        ),
        pytest.param(
            "Zu der Frage nach dem Fahrplan finden sich in den Unterlagen keine Angaben dazu.",
            id="de-keine-angaben",
        ),
        pytest.param(
            "Welcher Kapitän das Boot am Morgen führte, ist dem Protokoll nicht zu entnehmen.",
            id="de-nicht-zu-entnehmen",
        ),
        pytest.param(
            "Die Zahl der Passagiere auf der Fähre wird im Bericht des Hafenamts nicht erwähnt.",
            id="de-nicht-erwaehnt",
        ),
        pytest.param(
            "The harbour survey is detailed, but the excerpt does not mention the channel depth.",
            id="en-does-not-mention",
        ),
        pytest.param(
            "The two reports describe the storm, yet the wind speed is not mentioned in either.",
            id="en-not-mentioned",
        ),
        pytest.param(
            "Although the logbook lists every departure, it does not say which crew was aboard.",
            id="en-does-not-say",
        ),
        pytest.param(
            "The excerpts cover the festival week, but there are no details about the catering budget.",
            id="en-no-details",
        ),
        pytest.param(
            "From the two invented accounts, the cause of the delay cannot be determined with certainty.",
            id="en-cannot-be-determined",
        ),
    ],
)
def test_is_weak_answer_flags_refusal_phrasings(answer: str) -> None:
    """A refusal naming a missing detail is weak, in German or English."""
    assert len(answer.strip()) > WEAK_ANSWER_MIN_CHARS
    assert is_weak_answer(answer) is True


def test_is_weak_answer_keeps_a_substantive_answer() -> None:
    """A grounded answer that matches no refusal phrase is not weak."""
    answer = "Laut dem Bericht öffnete das Hafenbecken im Frühjahr und bediente zunächst zwei Fähren am Tag."
    assert is_weak_answer(answer) is False


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param(_CITED_CAVEAT_EN, id="en-cited-caveat-sentence"),
        pytest.param(
            "Die Fähre legte um neun Uhr am Nordkai ab [1] und erreichte die Insel vor Mittag [2]. "
            "Der Name des Kapitäns wird im Protokoll nicht erwähnt.",
            id="de-cited-caveat-sentence",
        ),
        pytest.param(
            "The ferry left the north quay at nine and reached the island before noon [1], "
            "although the log does not mention the captain's name.",
            id="en-cited-caveat-clause",
        ),
        pytest.param(
            "Die Fähre legte um neun Uhr am Nordkai ab und erreichte die Insel vor Mittag [1]; "
            "der Name des Kapitäns ist dem Protokoll nicht zu entnehmen.",
            id="de-cited-caveat-clause",
        ),
        pytest.param(
            "No, the harbour report does not mention a strike; it covers only the ferry wages [1].",
            id="en-cited-negative-answer",
        ),
        pytest.param(
            "As noted before, the ferry ran daily from May to September. "
            "It carried passengers and freight on every crossing. The fare is not mentioned.",
            id="en-uncited-elaboration",
        ),
        pytest.param(_UNCITED_CAVEAT_DE, id="de-uncited-elaboration"),
        pytest.param("Both ferries ran daily [1, 2]. The fare is not mentioned.", id="citation-list"),
        pytest.param("Both ferries ran daily [1][2]. The fare is not mentioned.", id="citation-adjacent"),
        pytest.param("Both ferries ran daily [1-3]. The fare is not mentioned.", id="citation-range"),
        pytest.param("Both ferries ran daily [1, S. 4]. The fare is not mentioned.", id="citation-with-page"),
        pytest.param(
            "Das Protokoll hält fest: „Die Fähre legte um acht Uhr am Nordkai an.“ "
            "Wer sie steuerte, wird nicht erwähnt.",
            id="de-quoted-sentence",
        ),
        pytest.param(
            'The log records: "The ferry docked at the north quay at eight." Who steered it is not mentioned.',
            id="en-quoted-sentence",
        ),
        pytest.param(
            "The log records: **The ferry docked at the north quay at eight.** Who steered it is not mentioned.",
            id="en-bold-sentence",
        ),
        pytest.param(
            "| Fähre | Kapitän |\n|---|---|\n"
            "| Möwe, ab Nordkai 08:00 | nicht erwähnt |\n"
            "| Kormoran, ab Südkai 09:30 | nicht erwähnt |",
            id="de-table-with-missing-cells",
        ),
        pytest.param(
            "Die im Bericht nicht erwähnten Fähren fuhren ebenfalls täglich zwischen beiden Häfen.",
            id="de-inflected-adjective",
        ),
        pytest.param(
            "There is no information about the captain, but the log records three departures [1][2] "
            "and two arrivals [3].",
            id="en-no-information-caveat",
        ),
    ],
)
def test_is_weak_answer_keeps_answers_with_missing_detail_caveats(answer: str) -> None:
    """An answer that flags a missing detail beside cited or substantive content is not weak."""
    assert is_weak_answer(answer) is False


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param("The notes cover only the harbour. The ferry schedule is not mentioned.", id="en-lead-in"),
        pytest.param("Die Notizen behandeln nur den Hafen. Zum Fahrplan gibt es keine Angaben.", id="de-lead-in"),
        pytest.param(
            "### Antwort\nAus den Notizen geht nicht hervor, welches Schiff zuerst anlegte.",
            id="de-heading",
        ),
        pytest.param(
            "Aus den Notizen geht nicht hervor, welches Schiff am 3. Mai zuerst anlegte.",
            id="de-date-abbreviation",
        ),
        pytest.param(
            "- Abfahrt: nicht erwähnt\n- Kapitän: nicht erwähnt\n- Ziel: nicht erwähnt",
            id="de-bulleted",
        ),
        pytest.param(
            "There is no information in the provided notes about who captained the ferry.",
            id="en-no-information",
        ),
    ],
)
def test_is_weak_answer_flags_bare_refusals(answer: str) -> None:
    """An uncited answer that is nothing but missing-detail statements is weak."""
    assert len(answer.strip()) > WEAK_ANSWER_MIN_CHARS
    assert is_weak_answer(answer) is True


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param(
            "The ferry docked at the north quay at eight [1]. Evidence is insufficient to say who steered it.",
            id="en",
        ),
        pytest.param(
            "Die Fähre legte um acht Uhr am Nordkai an [1]. Die Belege sind unzureichend, um den Kapitän zu nennen.",
            id="de",
        ),
    ],
)
def test_is_weak_answer_flags_evidence_phrases_anywhere(answer: str) -> None:
    """A statement that the evidence is insufficient marks even a cited answer weak."""
    assert is_weak_answer(answer) is True
