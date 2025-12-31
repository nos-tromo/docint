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
from docint.agents.types import RetrievalAgent, Turn as TurnType


@dataclass
class _DummyRetrievalResult:
    answer: str
    sources: list[dict]


class _StubRetrievalAgent(RetrievalAgent):
    def __init__(self):
        self.called_with: RetrievalRequest | None = None

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        self.called_with = request
        return RetrievalResult(answer="ok", sources=[{"id": 1}], session_id="s1")


class _NoopClarifier(SimpleClarificationAgent):
    def build(self, turn: TurnType, analysis) -> ClarificationRequest:  # type: ignore[override]
        _ = turn, analysis
        return ClarificationRequest(needed=True, message="clarify")


class _AlwaysClarifyPolicy(ClarificationPolicy):
    def __init__(self):
        super().__init__(
            ClarificationConfig(confidence_threshold=1.0, require_entities=True)
        )

    def evaluate(self, analysis, clarifications_so_far: int = 0):  # type: ignore[override]
        _ = analysis, clarifications_so_far
        return ClarificationRequest(needed=True, message="clarify", reason="force")


class _NeverClarifyPolicy(ClarificationPolicy):
    def __init__(self):
        super().__init__(
            ClarificationConfig(confidence_threshold=0.0, require_entities=False)
        )


@pytest.fixture
def turn() -> Turn:
    return Turn(user_input="hello", session_id="s1")


def test_orchestrator_requests_clarification(turn: Turn) -> None:
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
