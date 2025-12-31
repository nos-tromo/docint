"""Routing and clarification policies."""

from dataclasses import dataclass

from docint.agents.types import ClarificationRequest, IntentAnalysis


@dataclass
class ClarificationConfig:
    """
    Configuration for clarification routing.
    """

    confidence_threshold: float = 0.5
    require_entities: bool = True
    max_clarifications: int = 2


@dataclass
class ClarificationPolicy:
    """
    Threshold-based clarification policy with limits.
    """

    def __init__(self, config: ClarificationConfig | None = None) -> None:
        """
        Initialize the ClarificationPolicy.

        Args:
            config (ClarificationConfig | None, optional): Configuration for clarification. Defaults to None.
        """
        self.config = config or ClarificationConfig()

    def evaluate(
        self, analysis: IntentAnalysis, clarifications_so_far: int = 0
    ) -> ClarificationRequest:
        """
        Decide whether to ask for clarification based on the analysis.

        Args:
            analysis (IntentAnalysis): The intent analysis result.

        Returns:
            ClarificationRequest: Clarification decision and message.
        """
        cfg = self.config
        if clarifications_so_far >= cfg.max_clarifications:
            return ClarificationRequest(needed=False, reason="clarification limit")

        needs_entities = cfg.require_entities and not bool(analysis.entities)
        low_confidence = analysis.confidence < cfg.confidence_threshold
        if low_confidence or needs_entities:
            reason = []
            if low_confidence:
                reason.append("low confidence")
            if needs_entities:
                reason.append("missing entities")
            return ClarificationRequest(
                needed=True,
                message="Could you clarify what you need?",
                reason=", ".join(reason) or None,
            )

        return ClarificationRequest(needed=False)
