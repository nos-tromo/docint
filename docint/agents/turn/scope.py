"""A pinned scope: the passages the user hand-picked, shown to the model as the turn's whole evidence.

A scoped turn answers only from what the user pinned, so it runs no search: the pinned
passages enter the ledger, numbered in the order given, and are rendered into the one user
message the round sends, between the scope note and the user's message. Pinned evidence is
never dropped: a scope the ledger cannot hold, or cannot show legibly in the room it has,
raises :class:`EvidenceDoesNotFit` before the turn shows anything.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from llama_index.core.schema import BaseNode, NodeWithScore

from docint.agents.assessment.ledger import EvidenceLedger, render_context

#: The line above and below the pinned passages, as the grounded prompts fence their context.
PASSAGES_FENCE = "---------------------"


class EvidenceDoesNotFit(ValueError):
    """The evidence cannot be shown legibly in one answer, so the caller answers the turn another way.

    Raised before anything is shown: a pinned scope then takes the caller's existing scoped
    path. A ``ValueError`` naming the context window, so that if it ever escapes its callers
    the API still reports ``context_overflow``.

    Attributes:
        passages (int): How many passages the evidence held.
    """

    def __init__(self, passages: int) -> None:
        """Record how many passages did not fit.

        Args:
            passages (int): How many passages the evidence held.
        """
        super().__init__(f"{passages} passages cannot be shown legibly in the model's context window.")
        self.passages = passages


def scope_message(
    pinned: Sequence[NodeWithScore],
    *,
    ledger: EvidenceLedger,
    note: str,
    message: str,
    llm_view: Callable[[BaseNode], BaseNode],
    budget_tokens: int,
    char_token_ratio: float,
) -> tuple[str, int]:
    """Admit the pinned passages and write the one user message that shows them.

    Args:
        pinned (Sequence[NodeWithScore]): The passages the user pinned, in the order to number them.
        ledger (EvidenceLedger): The turn's evidence, empty; the passages are admitted into it.
        note (str): What the model is told about the scope; left out when empty.
        message (str): The user's message, sent verbatim after the passages.
        llm_view (Callable[[BaseNode], BaseNode]): Hides the metadata the model must not see.
        budget_tokens (int): Tokens the rendered passages may use.
        char_token_ratio (float): Characters per token.

    Returns:
        tuple[str, int]: The message: the note, the passages between fences, then the user's
        message; and how many passages were admitted or extended, which the seed step reports.

    Raises:
        EvidenceDoesNotFit: When the ledger cannot hold every pinned passage, or the passages
            cannot all be shown legibly in ``budget_tokens``.
    """
    admitted = ledger.add(pinned)
    ids = {scored.node.node_id for scored in pinned}
    if any(ledger.number_of(node_id) is None for node_id in ids):
        raise EvidenceDoesNotFit(len(ids))
    rendered = render_context(
        ledger.numbered(), llm_view=llm_view, budget_tokens=budget_tokens, char_token_ratio=char_token_ratio
    )
    if not rendered.legible:
        raise EvidenceDoesNotFit(len(ledger))
    fenced = f"{PASSAGES_FENCE}\n{rendered.context}\n{PASSAGES_FENCE}"
    return "\n\n".join(part for part in (note, fenced, message) if part), len(admitted)
