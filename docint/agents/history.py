"""Conversation-history helpers shared across agent code paths."""

from docint.agents.types import PriorTurn


def build_prior_turn(history: list[dict[str, str]]) -> PriorTurn | None:
    """Extract the immediately preceding user/assistant exchange from history.

    Scans the history tail for the most recent assistant message and the
    user message that triggered it. Returns ``None`` when the conversation
    has no prior exchange yet (first turn) or when the history is malformed
    (e.g., only system messages).

    Args:
        history: Ordered list of ``{"role": ..., "content": ...}`` messages.

    Returns:
        The prior exchange, or ``None`` when not derivable.
    """
    if not history:
        return None
    last_assistant_idx: int | None = None
    for idx in range(len(history) - 1, -1, -1):
        if history[idx].get("role") == "assistant":
            last_assistant_idx = idx
            break
    if last_assistant_idx is None:
        return None
    assistant_text = (history[last_assistant_idx].get("content") or "").strip()
    if not assistant_text:
        return None
    user_text = ""
    for idx in range(last_assistant_idx - 1, -1, -1):
        if history[idx].get("role") == "user":
            user_text = (history[idx].get("content") or "").strip()
            break
    return PriorTurn(user_text=user_text, assistant_text=assistant_text)
