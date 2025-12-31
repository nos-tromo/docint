"""Conversation context helpers."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TurnContext:
    """
    Carries session-level context for orchestrated turns.
    """

    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    clarifications: int = 0
