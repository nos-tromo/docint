"""Agent orchestrator that routes understanding, clarification, and retrieval."""

import dataclasses
import re

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
# Statements about the evidence itself. The QA prompt reserves these for a
# refusal, so they mark an answer weak wherever they appear.
WEAK_ANSWER_PHRASES: tuple[str, ...] = (
    "evidence insufficient",
    "evidence is insufficient",
    "i couldn't generate",
    "cannot answer based on the provided",
    "the retrieved context does not",
    # German, for RESPONSE_LANGUAGE=de deployments.
    "keine belege",
    "unzureichende belege",
    "belege sind unzureichend",
    "kann auf grundlage der bereitgestellten",
    "konnte keine antwort",
)
# Statements that a detail is missing. A substantive answer flags a gap the
# same way, so these count only in an answer that is nothing but the refusal.
MISSING_DETAIL_PHRASES: tuple[str, ...] = (
    "no information",
    "does not mention",
    "not mentioned",
    "does not say",
    "no details",
    "cannot be determined",
    "keine informationen",
    "geht nicht hervor",
    "keine angaben",
    "nicht zu entnehmen",
    "nicht erwähnt",
)
_MISSING_DETAIL_RE = re.compile(r"\b(?:" + "|".join(map(re.escape, MISSING_DETAIL_PHRASES)) + r")\b")
# Loose on purpose: any bracketed marker holding a digit ([3], [1, 2], [1, S. 4]).
_CITATION_RE = re.compile(r"\[[^\]\n]{0,40}\d[^\]\n]{0,40}\]")
# Sentence ends (with any closing quote or emphasis after them), lines and table cells.
_ANSWER_UNIT_BREAK_RE = re.compile(r"[.!?]\S*\s+|\n+|\|")
WEAK_ANSWER_FALLBACK_MESSAGE = (
    "I couldn't find enough specific evidence to elaborate. Could you tell me "
    "which part of my previous answer you'd like me to expand on — for example, "
    "a specific name, organization, or quote I mentioned?"
)


def _is_missing_detail_refusal(lowered: str) -> bool:
    """Return True when an answer holds nothing but missing-detail statements.

    The answer is cut into sentences, lines and table cells. The units naming
    a missing detail are set aside, and what remains must be too short to be
    an answer of its own. A cited answer has delivered evidence, so it is
    never such a refusal. Both checks err towards keeping the answer:
    splitting too often or matching a citation too loosely can only keep one.

    Args:
        lowered: The lowercased answer text.

    Returns:
        True when the answer cites nothing and, without its missing-detail
        units, is shorter than ``WEAK_ANSWER_MIN_CHARS``.
    """
    if _CITATION_RE.search(lowered):
        return False
    units = [unit.strip() for unit in _ANSWER_UNIT_BREAK_RE.split(lowered)]
    refusals = [bool(_MISSING_DETAIL_RE.search(unit)) for unit in units]
    if not any(refusals):
        return False
    rest = sum(len(unit) for unit, refusal in zip(units, refusals, strict=True) if not refusal)
    return rest < WEAK_ANSWER_MIN_CHARS


def is_weak_answer(answer: str | None) -> bool:
    """Return True when an answer is short or a refusal.

    Multi-signal so that we avoid both over-triggering (validation mismatch
    can fire on perfectly grounded answers when the retrieval drifted from
    the user's intent) and under-triggering (the LLM invents new refusal
    phrasings the validator already caught).

    A statement about the evidence itself ("evidence is insufficient") is a
    refusal wherever it appears. A missing-detail statement ("does not
    mention", "geht nicht hervor") is how a refusal names its gap, but also
    how a substantive answer flags the one detail it lacks, so it counts only
    when the answer is nothing but such statements. Both callers discard a
    weak answer, so a doubtful case is kept.

    Args:
        answer: The generated answer text, or ``None``.

    Returns:
        True when the answer is empty/very short or a refusal.
    """
    if not answer or len(answer.strip()) < WEAK_ANSWER_MIN_CHARS:
        return True
    lowered = answer.lower()
    if any(phrase in lowered for phrase in WEAK_ANSWER_PHRASES):
        return True
    return _is_missing_detail_refusal(lowered)


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
