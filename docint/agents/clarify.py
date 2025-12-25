"""Clarification agents."""

from docint.agents.types import (
    ClarificationAgent,
    ClarificationRequest,
    IntentAnalysis,
    Turn,
)


class SimpleClarificationAgent(ClarificationAgent):
    """
    Simple clarifier that returns a generic prompt.
    It can also list missing required entities from the analysis.
    """

    def __init__(
        self, prompt: str | None = None, required_entities: list[str] | None = None
    ) -> None:
        """
            Initialize the SimpleClarificationAgent.

        Args:
            prompt (str | None, optional): The clarification prompt to use. Defaults to None.
            required_entities (list[str] | None, optional): List of required entities to check for. Defaults to None.
        """        
        self.prompt = prompt or "Could you clarify what you need?"
        self.required_entities = required_entities or ["query"]

    def build(self, turn: Turn, analysis: IntentAnalysis) -> ClarificationRequest:
        """
        Return a clarification request, listing missing entities if any.

        Args:
            turn (Turn): The current turn in the conversation.
            analysis (IntentAnalysis): The result of intent analysis.

        Returns:
            ClarificationRequest: The constructed clarification request.
        """        
        missing = [k for k in self.required_entities if k not in analysis.entities]
        detail = (
            f"Missing details: {', '.join(missing)}" if missing else analysis.reason
        )
        message = self.prompt
        if missing:
            message = f"{self.prompt} Please provide: {', '.join(missing)}."
        return ClarificationRequest(needed=True, message=message, reason=detail)
