"""Share a character budget across texts and mark what had to be cut."""

from collections.abc import Sequence


def allocate_shortest_first(lengths: Sequence[int], budget: int) -> list[int]:
    """Split a shared character budget across texts, shortest first.

    Each text is offered an equal share of what is left; a text needing less
    than its share returns the remainder to those still unallocated, so a short
    text never wastes budget a long one could use and none starves the others.

    Args:
        lengths (Sequence[int]): Length of each text, in order.
        budget (int): Total characters available across all texts.

    Returns:
        list[int]: Characters granted to each text, aligned with ``lengths``.
    """
    allocation = [0] * len(lengths)
    remaining_budget = budget
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    for position, index in enumerate(order):
        share = remaining_budget // (len(order) - position)
        granted = min(lengths[index], share)
        allocation[index] = granted
        remaining_budget -= granted
    return allocation


def fit_with_marker(text: str, allowance: int) -> tuple[str, bool]:
    """Trim a text to its allowance, marking how much was withheld.

    The marker names the hidden character count so a reader can tell "this
    says nothing about X" apart from "I was not shown the part that does".

    Args:
        text (str): The full text.
        allowance (int): Characters it may show.

    Returns:
        tuple[str, bool]: The rendered text and whether it was trimmed.
    """
    if len(text) <= allowance:
        return (text, False)
    hidden = len(text) - allowance
    return (f"{text[:allowance]}[... {hidden} of {len(text)} characters not shown ...]", True)
