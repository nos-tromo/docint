"""Shared types for agent orchestration."""

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

AnswerMode = Literal["lookup", "assessment"]
"""How a chat turn is answered: a grounded lookup, or with a labelled model assessment."""

AgentAction = Literal["seed", "carried", "search", "answer", "note", "search_images", "keyframes", "keyword", "read"]
"""What one step of an agent turn did: a tool call by the tool's name, or a ``note``, the text a
round wrote before its calls. ``seed`` opens a scoped turn's trail with its pinned passages, and
``carried`` a follow-up's with the passages its earlier answers cite; ``answer`` is found only in
trails stored by the earlier assessment pipeline."""


@dataclass(frozen=True, slots=True)
class AgentProgress:
    """One step of an agent turn, streamed to the client as an ``{"agent": ...}`` frame.

    A dataclass rather than a dict on purpose: the API treats the one dict a
    chat stream yields as its final payload, and ``None`` as its end.

    Attributes:
        step (int): The round the step belongs to, from ``1``; a scoped turn's
            ``seed`` and a follow-up's ``carried`` are ``0``.
        action (AgentAction): What the step did.
        detail (str): The call's main argument, such as a model-written search
            query, or a note's text; empty when there is none. Shown to the user,
            never logged.
        found (int): Passages the step added to the evidence or extended there.
        field (str | None): The field a keyword check matched; ``None`` on any other step.
        matches (int | None): Indexed chunks a keyword check counted, 0 included; ``None`` on any
            other step, or when the count failed.
        unchecked (int | None): Chunks a keyword check could not reach because the keyword index
            does not cover them; ``None`` on any other step, or when it covers all.
    """

    step: int
    action: AgentAction
    detail: str
    found: int
    field: str | None = None
    matches: int | None = None
    unchecked: int | None = None


@dataclass
class Turn:
    """Represents a single user turn.

    Attributes:
        user_input (str): The user's message, verbatim.
        session_id (str | None): The conversation it belongs to, if any.
        previous_user_input (str | None): The user's message before this one, for a validator
            judging a follow-up; ``None`` when it is not shown one.
    """

    user_input: str
    session_id: str | None = None
    previous_user_input: str | None = None


@dataclass
class IntentAnalysis:
    """Intent and entity analysis result."""

    intent: str
    confidence: float
    entities: dict[str, Any]
    tool: str | None = None
    reason: str | None = None
    rewritten_query: str | None = None


@dataclass
class ClarificationRequest:
    """Clarification request returned by the policy or clarifier."""

    needed: bool
    message: str | None = None
    reason: str | None = None


@dataclass
class PriorTurn:
    """Captures the immediately preceding user/assistant exchange.

    Used by the orchestrator path to (a) skip the
    ``SessionManager.chat`` second rewrite (the orchestrator's
    understanding agent already produced a context-aware query) and
    (b) bind ``{prior_turn_context}`` into the grounded-QA template so
    elaboration follow-ups can quote and expand on the prior assistant
    answer.
    """

    user_text: str
    assistant_text: str
    evidence_chars: int | None = None
    """Length of the source-backed part when the answer was an assessment.

    ``None`` means the whole answer is source-backed, as on an ordinary lookup.
    """

    def evidence_only(self) -> "PriorTurn | None":
        """Return this exchange without the model's own assessment.

        A lookup turn may restate its prior answer, so it must never be handed
        the general-knowledge part of an assessment as if it were sourced.

        Returns:
            PriorTurn | None: The exchange cut to its source-backed part, or
            ``None`` when nothing source-backed remains.
        """
        if self.evidence_chars is None:
            return self
        evidence = self.assistant_text[: self.evidence_chars].strip()
        if not evidence:
            return None
        return PriorTurn(user_text=self.user_text, assistant_text=evidence)


@dataclass
class RetrievalRequest:
    """Input to the retrieval agent."""

    turn: Turn
    analysis: IntentAnalysis
    history: list[dict[str, str]] = field(default_factory=list)
    replace_turn_idx: int | None = None
    """Overwrite this conversation turn instead of appending a new one.

    Set by the corrective retry so the second attempt replaces the rejected
    answer in place: one user message must persist as exactly one turn, or the
    session would read back as though the user had asked twice.
    """


@dataclass
class RetrievalResult:
    """Output from retrieval or generation step."""

    answer: str | None
    sources: list[dict[str, Any]]
    session_id: str | None = None
    intent: str | None = None
    confidence: float | None = None
    tool_used: str | None = None
    latency_ms: float | None = None
    validation_checked: bool | None = None
    validation_mismatch: bool | None = None
    validation_reason: str | None = None
    summary_diagnostics: dict[str, Any] | None = None
    retrieval_query: str | None = None
    rewritten_query: str | None = None
    retried: bool | None = None
    retry_query: str | None = None
    keyword_checks: str | None = None
    """An agent turn's keyword checks with a known count, one line each.

    Facts about the collection the validator reads beside the sources, so an
    answer may state a count no passage shows. ``None`` or empty for none.
    """
    turn_idx: int | None = None
    """Index of the persisted conversation turn; an internal join key.

    Carried so a corrective retry can overwrite the turn its first attempt
    wrote. Never serialized to clients.
    """


@dataclass
class OrchestratorResult:
    """Top-level result for a single orchestrated turn."""

    clarification: ClarificationRequest | None
    retrieval: RetrievalResult | None
    analysis: IntentAnalysis | None


class UnderstandingAgent(Protocol):
    """Interface for understanding user input."""

    def analyze(self, turn: Turn, context: Any | None = None) -> IntentAnalysis:  # pragma: no cover - interface
        """Analyze a turn and return intent/entities/confidence.

        Args:
            turn (Turn): The turn to analyze.
            context (Any | None, optional): The conversation context.

        Returns:
            IntentAnalysis: The result of the analysis.
        """
        ...


class ClarificationAgent(Protocol):
    """Interface for clarification generation."""

    def build(self, turn: Turn, analysis: IntentAnalysis) -> ClarificationRequest:  # pragma: no cover - interface
        """Return a clarification request for the user.

        Args:
            turn (Turn): The current turn in the conversation.
            analysis (IntentAnalysis): The result of intent analysis.

        Returns:
            ClarificationRequest: The constructed clarification request.
        """
        ...


class RetrievalAgent(Protocol):
    """Interface for retrieval."""

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:  # pragma: no cover - interface
        """Return retrieval results for the turn and analysis.

        Args:
            request (RetrievalRequest): The retrieval request containing the turn and analysis.

        Returns:
            RetrievalResult: The result of the retrieval or generation step.
        """
        ...


class ResponseAgent(Protocol):
    """Interface for response post-processing."""

    def finalize(self, result: RetrievalResult, turn: Turn) -> RetrievalResult:  # pragma: no cover - interface
        """Optionally post-process retrieval output before returning.

        Args:
            result (RetrievalResult): The retrieval result to finalize.
            turn (Turn): The current turn in the conversation.

        Returns:
            RetrievalResult: The finalized retrieval result.
        """
        ...
