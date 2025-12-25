"""Response agent stubs."""

from docint.agents.types import ResponseAgent, RetrievalResult, Turn


class PassthroughResponseAgent(ResponseAgent):
    """No-op response post-processor."""

    def finalize(self, result: RetrievalResult, turn: Turn) -> RetrievalResult:
        """Return retrieval results unchanged."""
        _ = turn  # placeholder until custom formatting is needed
        return result
