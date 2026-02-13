"""Shared types for agent orchestration."""

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class Turn:
    """
    Represents a single user turn.
    """

    user_input: str
    session_id: str | None = None


@dataclass
class IntentAnalysis:
    """
    Intent and entity analysis result.
    """

    intent: str
    confidence: float
    entities: dict[str, Any]
    tool: str | None = None
    reason: str | None = None
    rewritten_query: str | None = None


@dataclass
class ClarificationRequest:
    """
    Clarification request returned by the policy or clarifier.
    """

    needed: bool
    message: str | None = None
    reason: str | None = None


@dataclass
class RetrievalRequest:
    """
    Input to the retrieval agent.
    """

    turn: Turn
    analysis: IntentAnalysis


@dataclass
class RetrievalResult:
    """
    Output from retrieval or generation step.
    """

    answer: str | None
    sources: list[dict[str, Any]]
    session_id: str | None = None
    intent: str | None = None
    confidence: float | None = None
    tool_used: str | None = None
    latency_ms: float | None = None


@dataclass
class OrchestratorResult:
    """
    Top-level result for a single orchestrated turn.
    """

    clarification: ClarificationRequest | None
    retrieval: RetrievalResult | None
    analysis: IntentAnalysis | None


class UnderstandingAgent(Protocol):
    """
    Interface for understanding user input.
    """

    def analyze(
        self, turn: Turn, context: Any | None = None
    ) -> IntentAnalysis:  # pragma: no cover - interface
        """
        Analyze a turn and return intent/entities/confidence.

        Args:
            turn (Turn): The turn to analyze.
            context (Any | None, optional): The conversation context.

        Returns:
            IntentAnalysis: The result of the analysis.
        """
        ...


class ClarificationAgent(Protocol):
    """
    Interface for clarification generation.
    """

    def build(
        self, turn: Turn, analysis: IntentAnalysis
    ) -> ClarificationRequest:  # pragma: no cover - interface
        """
        Return a clarification request for the user.

        Args:
            turn (Turn): The current turn in the conversation.
            analysis (IntentAnalysis): The result of intent analysis.

        Returns:
            ClarificationRequest: The constructed clarification request.
        """
        ...


class RetrievalAgent(Protocol):
    """
    Interface for retrieval.
    """

    def retrieve(
        self, request: RetrievalRequest
    ) -> RetrievalResult:  # pragma: no cover - interface
        """
        Return retrieval results for the turn and analysis.

        Args:
            request (RetrievalRequest): The retrieval request containing the turn and analysis.

        Returns:
            RetrievalResult: The result of the retrieval or generation step.
        """
        ...


class ResponseAgent(Protocol):
    """
    Interface for response post-processing.
    """

    def finalize(
        self, result: RetrievalResult, turn: Turn
    ) -> RetrievalResult:  # pragma: no cover - interface
        """
        Optionally post-process retrieval output before returning.

        Args:
            result (RetrievalResult): The retrieval result to finalize.
            turn (Turn): The current turn in the conversation.

        Returns:
            RetrievalResult: The finalized retrieval result.
        """
        ...
