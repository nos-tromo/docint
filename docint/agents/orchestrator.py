"""Agent orchestrator that routes understanding, clarification, and retrieval."""

from docint.agents.context import TurnContext
from docint.agents.policies import ClarificationPolicy
from docint.agents.types import (
    ClarificationAgent,
    ClarificationRequest,
    OrchestratorResult,
    RetrievalAgent,
    RetrievalRequest,
    RetrievalResult,
    Turn,
    UnderstandingAgent,
)


class AgentOrchestrator:
    """
    Coordinate agents for a single conversational turn.
    Handles understanding, clarification, and retrieval in sequence.
    """

    def __init__(
        self,
        understanding: UnderstandingAgent,
        clarifier: ClarificationAgent,
        retriever: RetrievalAgent,
        policy: ClarificationPolicy | None = None,
    ) -> None:
        """
        Initialize the AgentOrchestrator.

        Args:
            understanding (UnderstandingAgent): The agent responsible for understanding user input.
            clarifier (ClarificationAgent): The agent responsible for handling clarifications.
            retriever (RetrievalAgent): The agent responsible for retrieving information.
            policy (ClarificationPolicy | None, optional): The policy to decide when clarification is needed. Defaults to None.
        """        
        self.understanding = understanding
        self.clarifier = clarifier
        self.retriever = retriever
        self.policy = policy or ClarificationPolicy()

    def handle_turn(
        self, turn: Turn, context: TurnContext | None = None
    ) -> OrchestratorResult:
        """
        Process a turn: understand, possibly clarify, otherwise retrieve/respond.

        Args:
            turn (Turn): The user turn to process.

        Returns:
            OrchestratorResult: Clarification or retrieval result for the turn.
        """
        ctx = context or TurnContext(session_id=turn.session_id)
        analysis = self.understanding.analyze(turn)
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

        retrieval_request = RetrievalRequest(turn=turn, analysis=analysis)
        retrieval: RetrievalResult = self.retriever.retrieve(retrieval_request)
        return OrchestratorResult(
            clarification=None, retrieval=retrieval, analysis=analysis
        )
