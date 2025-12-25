"""Response agent stubs."""

from docint.agents.types import ResponseAgent, RetrievalResult, Turn


class PassthroughResponseAgent(ResponseAgent):
    """
    No-op response post-processor.
    Returns retrieval results unchanged.
    """

    def finalize(self, result: RetrievalResult, turn: Turn) -> RetrievalResult:
        """
        Return retrieval results unchanged.
        
        Args:
            result (RetrievalResult): The retrieval result to finalize.
            turn (Turn): The current turn in the conversation.

        Returns:
            RetrievalResult: The unchanged retrieval result.
        """
        _ = turn  # placeholder until custom formatting is needed
        return result
