"""Agent orchestrator that routes understanding, clarification, and retrieval."""

from docint.agents.context import TurnContext
from docint.agents.policies import ClarificationPolicy
from docint.agents.types import (
    ClarificationAgent,
    ClarificationRequest,
    OrchestratorResult,
    ResponseAgent,
    RetrievalAgent,
    RetrievalRequest,
    RetrievalResult,
    Turn,
    UnderstandingAgent,
)

WEAK_ANSWER_MIN_CHARS = 40
WEAK_ANSWER_PHRASES: tuple[str, ...] = (
    "evidence insufficient",
    "evidence is insufficient",
    "i couldn't generate",
    "cannot answer based on the provided",
    "the retrieved context does not",
    "no information",
)
WEAK_ANSWER_FALLBACK_MESSAGE = (
    "I couldn't find enough specific evidence to elaborate. Could you tell me "
    "which part of my previous answer you'd like me to expand on — for example, "
    "a specific name, organization, or quote I mentioned?"
)


def _is_weak_answer(answer: str | None) -> bool:
    """Return True when an answer is short or matches a known refusal phrase.

    Multi-signal so that we avoid both over-triggering (validation mismatch
    can fire on perfectly grounded answers when the retrieval drifted from
    the user's intent) and under-triggering (the LLM invents new refusal
    phrasings the validator already caught).

    Args:
        answer: The generated answer text, or ``None``.

    Returns:
        True when the answer is empty/very short or contains a refusal phrase.
    """
    if not answer or len(answer.strip()) < WEAK_ANSWER_MIN_CHARS:
        return True
    lowered = answer.lower()
    return any(phrase in lowered for phrase in WEAK_ANSWER_PHRASES)


class AgentOrchestrator:
    """Coordinate agents for a single conversational turn.

    Handles understanding, clarification, and retrieval in sequence.
    """

    def __init__(
        self,
        understanding: UnderstandingAgent,
        clarifier: ClarificationAgent,
        retriever: RetrievalAgent,
        responder: ResponseAgent | None = None,
        policy: ClarificationPolicy | None = None,
    ) -> None:
        """Initialize the AgentOrchestrator.

        Args:
            understanding (UnderstandingAgent): The agent responsible for understanding user input.
            clarifier (ClarificationAgent): The agent responsible for handling clarifications.
            retriever (RetrievalAgent): The agent responsible for retrieving information.
            responder (ResponseAgent | None, optional): The agent responsible for response validation/post-processing.
            policy (ClarificationPolicy | None, optional): Policy deciding when clarification
                is needed. Defaults to None.
        """
        self.understanding = understanding
        self.clarifier = clarifier
        self.retriever = retriever
        self.responder = responder
        self.policy = policy or ClarificationPolicy()

    def handle_turn(self, turn: Turn, context: TurnContext | None = None) -> OrchestratorResult:
        """Process a turn: understand, possibly clarify, otherwise retrieve/respond.

        Args:
            turn (Turn): The user turn to process.
            context (TurnContext | None): Per-turn context (session id, clarification count,
                ...). Defaults to a fresh context bound to the turn's session id.

        Returns:
            OrchestratorResult: Clarification or retrieval result for the turn.
        """
        ctx = context or TurnContext(session_id=turn.session_id)
        analysis = self.understanding.analyze(turn, context=ctx)
        clarification_decision: ClarificationRequest = self.policy.evaluate(
            analysis, clarifications_so_far=ctx.clarifications
        )

        if clarification_decision.needed:
            clarification = self.clarifier.build(turn, analysis)
            # Prefer clarifier message if provided; fall back to policy message.
            message = clarification.message or clarification_decision.message
            return OrchestratorResult(
                clarification=ClarificationRequest(
                    needed=True,
                    message=message,
                    reason=clarification.reason or clarification_decision.reason,
                ),
                retrieval=None,
                analysis=analysis,
            )

        retrieval_request = RetrievalRequest(
            turn=turn,
            analysis=analysis,
            history=list(ctx.history),
        )
        retrieval: RetrievalResult = self.retriever.retrieve(retrieval_request)
        if self.responder is not None:
            retrieval = self.responder.finalize(retrieval, turn)

        # Validation-driven clarification fallback: if the responder flagged
        # the answer as mismatched AND it is also weak (empty, very short, or
        # contains a refusal phrase), convert the turn into a clarification
        # request so the user gets a useful nudge instead of a bare
        # "Evidence insufficient." Respects the per-session clarification cap.
        if (
            retrieval.validation_mismatch is True
            and _is_weak_answer(retrieval.answer)
            and ctx.clarifications < self.policy.config.max_clarifications
        ):
            return OrchestratorResult(
                clarification=ClarificationRequest(
                    needed=True,
                    message=WEAK_ANSWER_FALLBACK_MESSAGE,
                    reason="weak_answer_after_validation_mismatch",
                ),
                retrieval=None,
                analysis=analysis,
            )
        return OrchestratorResult(clarification=None, retrieval=retrieval, analysis=analysis)
