"""Agent orchestrator that routes understanding, clarification, and retrieval."""

import dataclasses

from docint.agents.context import TurnContext
from docint.agents.policies import ClarificationPolicy
from docint.agents.reformulation import QueryReformulationAgent
from docint.agents.types import (
    ClarificationAgent,
    ClarificationRequest,
    IntentAnalysis,
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
    # German refusals, for RESPONSE_LANGUAGE=de deployments. The phrase list
    # can only ever cover the wordings we have seen; the length check below is
    # the locale-agnostic half of the signal and does the heavier lifting.
    "keine informationen",
    "keine belege",
    "unzureichende belege",
    "belege sind unzureichend",
    "kann auf grundlage der bereitgestellten",
    "konnte keine antwort",
)
WEAK_ANSWER_FALLBACK_MESSAGE = (
    "I couldn't find enough specific evidence to elaborate. Could you tell me "
    "which part of my previous answer you'd like me to expand on — for example, "
    "a specific name, organization, or quote I mentioned?"
)


def is_weak_answer(answer: str | None) -> bool:
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


# Retained for callers that predate the public name.
_is_weak_answer = is_weak_answer


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
        reformulator: QueryReformulationAgent | None = None,
    ) -> None:
        """Initialize the AgentOrchestrator.

        Args:
            understanding (UnderstandingAgent): The agent responsible for understanding user input.
            clarifier (ClarificationAgent): The agent responsible for handling clarifications.
            retriever (RetrievalAgent): The agent responsible for retrieving information.
            responder (ResponseAgent | None, optional): The agent responsible for response validation/post-processing.
            policy (ClarificationPolicy | None, optional): Policy deciding when clarification
                is needed. Defaults to None.
            reformulator (QueryReformulationAgent | None, optional): Enables the
                corrective retry when supplied. Left as ``None`` the orchestrator
                behaves exactly as before, so the config knob lives entirely in
                the wiring rather than in here.
        """
        self.understanding = understanding
        self.clarifier = clarifier
        self.retriever = retriever
        self.responder = responder
        self.policy = policy or ClarificationPolicy()
        self.reformulator = reformulator

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

        retrieval = self._maybe_retry(turn, ctx, analysis, retrieval)

        # Validation-driven clarification fallback: if the responder flagged
        # the answer as mismatched AND it is also weak (empty, very short, or
        # contains a refusal phrase), convert the turn into a clarification
        # request so the user gets a useful nudge instead of a bare
        # "Evidence insufficient." Respects the per-session clarification cap.
        if (
            retrieval.validation_mismatch is True
            and is_weak_answer(retrieval.answer)
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

    def _maybe_retry(
        self,
        turn: Turn,
        ctx: TurnContext,
        analysis: IntentAnalysis,
        retrieval: RetrievalResult,
    ) -> RetrievalResult:
        """Re-answer once with a reformulated query when the first attempt failed.

        Runs only when the responder flagged a mismatch *and* the answer is weak
        — a mismatched but substantive answer is still worth showing, and
        discarding it to chase a better one would trade a real answer for a
        coin flip. The retry is capped at one attempt structurally (there is no
        loop), and it re-validates so the caller's fallback decision is made
        against the answer the user will actually see.

        Args:
            turn (Turn): The user turn being answered.
            ctx (TurnContext): Per-turn context, read for conversation history.
            analysis (IntentAnalysis): The understanding agent's analysis.
            retrieval (RetrievalResult): The rejected first attempt.

        Returns:
            RetrievalResult: The retry's validated result, or the original one
            when no retry ran or no reformulation was available.
        """
        if self.reformulator is None:
            return retrieval
        if retrieval.validation_mismatch is not True or not is_weak_answer(retrieval.answer):
            return retrieval

        new_query = self.reformulator.reformulate(
            user_query=turn.user_input,
            failed_query=retrieval.retrieval_query or analysis.rewritten_query,
            validation_reason=retrieval.validation_reason,
        )
        if not new_query:
            return retrieval

        retry_analysis = dataclasses.replace(analysis, rewritten_query=new_query)
        second = self.retriever.retrieve(
            RetrievalRequest(
                turn=turn,
                analysis=retry_analysis,
                history=list(ctx.history),
                # One user message, one persisted turn: the retry overwrites the
                # row its first attempt wrote.
                replace_turn_idx=retrieval.turn_idx,
            )
        )
        if self.responder is not None:
            second = self.responder.finalize(second, turn)
        second.retried = True
        second.retry_query = new_query
        return second
