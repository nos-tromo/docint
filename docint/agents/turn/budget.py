"""How much of the chat model's context window a turn's tool replies may fill."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from docint.agents.assessment.ledger import ANSWER_OUTPUT_TOKENS, ANSWER_OUTPUT_TOKENS_REASONING, SAFETY_MARGIN
from docint.utils.embed_chunking import estimate_tokens

#: What a message costs beyond its text: the role markers and framing a chat template wraps around it.
MESSAGE_OVERHEAD_TOKENS = 40
#: Least room for tool replies that a turn can do useful work with.
MIN_ROUND_ROOM = 1500
#: Longest one tool reply in characters; shorter for a thinking model, whose larger reply reserve leaves less room.
RESULT_CHARS = 3000
RESULT_CHARS_REASONING = 2000
#: Longest text of one passage within a search reply in characters, its header excluded.
PASSAGE_CHARS = 900
PASSAGE_CHARS_REASONING = 700
#: Tokens of earlier conversation a turn sends: its summary block in the system prompt and its recent exchanges
#: as messages, each with its envelope; a thinking model's reply reserve takes the difference.
HISTORY_ALLOWANCE_TOKENS = 1300
HISTORY_ALLOWANCE_TOKENS_REASONING = 750


def message_cost(text: str, char_token_ratio: float) -> int:
    """Estimate what one message of a turn's prompt costs: its text and its envelope.

    Args:
        text (str): The message's text.
        char_token_ratio (float): Characters per token.

    Returns:
        int: Estimated tokens.

    Raises:
        ValueError: If ``char_token_ratio`` is not positive.
    """
    return estimate_tokens(text, char_token_ratio) + MESSAGE_OVERHEAD_TOKENS


def history_allowance_tokens(reasoning: bool) -> int:
    """Return how many tokens of earlier conversation a turn may send.

    The allowance covers the whole conversation window: its summary block
    (``ConversationWindow.summary_block()``), which the system prompt carries, and its
    recent exchanges, sent as messages each with its envelope. A thinking model
    reserves more of the window for its reply, so its allowance is smaller: the
    history gives way before the turn runs out of room for tool replies.

    Args:
        reasoning (bool): Whether the answer model thinks before it answers.

    Returns:
        int: The allowance in tokens.
    """
    return HISTORY_ALLOWANCE_TOKENS_REASONING if reasoning else HISTORY_ALLOWANCE_TOKENS


def schema_tokens(schemas: Sequence[dict[str, Any]], ratio: float) -> int:
    """Estimate what the offered tool schemas cost in the prompt.

    A chat template renders each schema as JSON with accented letters unescaped, so
    the estimate is that text's length at ``ratio`` characters per token. A turn that
    offers fewer tools pays for fewer.

    Args:
        schemas (Sequence[dict[str, Any]]): The tool schemas as the endpoint receives them.
        ratio (float): Characters per token.

    Returns:
        int: Estimated tokens; ``0`` when no tool is offered.
    """
    if not schemas:
        return 0
    return estimate_tokens(json.dumps(list(schemas), ensure_ascii=False), ratio)


@dataclass(frozen=True)
class TurnBudget:
    """How one turn spends the chat model's context window.

    The window holds the system prompt, the tool schemas, the history, the user's
    message, every tool reply the loop adds and the answer. The first four are known
    when the turn starts and the answer's room is reserved, so what is left is the
    room for tool replies. All token counts are estimates.

    Attributes:
        ctx_window (int): The chat model's context window in tokens.
        reasoning (bool): Whether the model thinks before it answers.
        char_token_ratio (float): Characters per token the counts were measured with;
            the tools use it to size a reply in characters.
        system_tokens (int): The system prompt as sent.
        tool_schema_tokens (int): The tool schemas the turn offers; see :func:`schema_tokens`.
        history_tokens (int): The earlier exchanges sent as messages, each counted with
            its :data:`MESSAGE_OVERHEAD_TOKENS`.
        message_tokens (int): The user's message.
    """

    ctx_window: int
    reasoning: bool
    char_token_ratio: float
    system_tokens: int
    tool_schema_tokens: int
    history_tokens: int
    message_tokens: int

    @property
    def round_room(self) -> int:
        """Tokens left for tool replies across the turn.

        The usable part of the window less the answer's reserve, the four parts of the
        prompt and the envelopes of the system and the user message.

        Returns:
            int: The room, never negative.
        """
        usable = int(self.ctx_window * SAFETY_MARGIN)
        reserve = ANSWER_OUTPUT_TOKENS_REASONING if self.reasoning else ANSWER_OUTPUT_TOKENS
        prompt = self.system_tokens + self.tool_schema_tokens + self.history_tokens + self.message_tokens
        # The system and the user message; each history message carries its own envelope in history_tokens.
        envelopes = 2 * MESSAGE_OVERHEAD_TOKENS
        return max(0, usable - reserve - prompt - envelopes)

    @property
    def available(self) -> bool:
        """Whether the turn has room for tool replies worth running a loop for.

        Returns:
            bool: ``True`` once :attr:`round_room` reaches :data:`MIN_ROUND_ROOM`.
        """
        return self.round_room >= MIN_ROUND_ROOM

    @property
    def result_chars(self) -> int:
        """Longest one tool reply in characters.

        Returns:
            int: Fewer for a thinking model, whose larger reply reserve leaves less room.
        """
        return RESULT_CHARS_REASONING if self.reasoning else RESULT_CHARS

    @property
    def passage_chars(self) -> int:
        """Longest text of one passage within a search reply in characters, its header excluded.

        Returns:
            int: Fewer for a thinking model, whose larger reply reserve leaves less room.
        """
        return PASSAGE_CHARS_REASONING if self.reasoning else PASSAGE_CHARS
