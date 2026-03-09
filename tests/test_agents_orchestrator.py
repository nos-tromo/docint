"""Tests for :class:`AgentOrchestrator` turn-handling logic."""

from dataclasses import dataclass

import pytest

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
)
from docint.agents.types import ResponseAgent, RetrievalAgent, Turn as TurnType


@dataclass
class _DummyRetrievalResult:
    """Minimal retrieval result dataclass for testing."""

    answer: str
    sources: list[dict]


class _StubRetrievalAgent(RetrievalAgent):
    """Retrieval agent that records the request and returns a canned result."""

    def __init__(self) -> None:
        """Initialise with no recorded request."""
        self.called_with: RetrievalRequest | None = None

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """Record the request and return a fixed retrieval result.

        Args:
            request: The retrieval request from the orchestrator.

        Returns:
            A canned ``RetrievalResult``.
        """
        self.called_with = request
        return RetrievalResult(answer="ok", sources=[{"id": 1}], session_id="s1")


class _StubResponseAgent(ResponseAgent):
    """Response agent that marks validation fields on the result."""

    def __init__(self) -> None:
        """Initialise with the call flag unset."""
        self.called = False

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

    def build(self, turn: TurnType, analysis) -> ClarificationRequest:  # type: ignore[override]
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
        super().__init__(
            ClarificationConfig(confidence_threshold=1.0, require_entities=True)
        )

    def evaluate(self, analysis, clarifications_so_far: int = 0):  # type: ignore[override]
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
        super().__init__(
            ClarificationConfig(confidence_threshold=0.0, require_entities=False)
        )


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
    assert result.retrieval.answer == "ok"
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
