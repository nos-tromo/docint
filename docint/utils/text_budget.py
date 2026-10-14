"""Share a character budget across texts and mark what had to be cut."""

from collections.abc import Sequence

#: Node metadata naming where a passage's retrieved match sits: ``[start, end]`` spans in its own text.
MATCH_SPANS_KEY = "docint_match_spans"
#: Worst-case length of the cut marker, reserved for every entry.
MARKER_RESERVE_CHARS = 50
#: Fewest characters a region kept around a match may show; a passage cut to 300 characters still keeps one.
MIN_REGION_CHARS = 200


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


def _marker(hidden: int, total: int) -> str:
    """Name a gap in a shortened text.

    Args:
        hidden (int): Characters the gap withholds.
        total (int): Length of the whole text.

    Returns:
        str: The marker.
    """
    return f"[... {hidden} of {total} characters not shown ...]"


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
    return (f"{text[:allowance]}{_marker(len(text) - allowance, len(text))}", True)


def merge_spans(spans: Sequence[Sequence[int]], length: int) -> list[tuple[int, int]]:
    """Clip spans to a text, drop the empty ones, and merge those that overlap or touch.

    Args:
        spans (Sequence[Sequence[int]]): ``(start, end)`` offsets, in any order.
        length (int): Length of the text they point into.

    Returns:
        list[tuple[int, int]]: Disjoint spans inside ``[0, length]``, in text order.
    """
    clipped = sorted((max(0, start), min(length, end)) for start, end in spans if min(length, end) > max(0, start))
    merged: list[tuple[int, int]] = []
    for start, end in clipped:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def place(start: int, end: int, size: int, length: int) -> tuple[int, int]:
    """Place a region of ``size`` characters on a span of a text.

    The region is centred on the span, or starts where the span does when the
    span is longer than the region, and is moved back inside the text when it
    would run past either end.

    Args:
        start (int): Where the span starts.
        end (int): Where the span ends.
        size (int): Characters the region holds.
        length (int): Length of the text.

    Returns:
        tuple[int, int]: The region's ``(start, end)``; shorter than ``size`` only when the text is.
    """
    first = start if end - start >= size else start - (size - (end - start)) // 2
    first = max(0, min(first, length - size))
    return first, min(length, first + size)


def excerpt(text: str, anchors: Sequence[Sequence[int]], allowance: int) -> tuple[str, bool]:
    """Trim a text to its allowance, keeping the text around each anchor.

    A text with no anchor past the prefix it may show is cut exactly as
    :func:`fit_with_marker` cuts it. Otherwise each anchor gets a region of an
    equal share of the allowance, less one marker each; while those shares
    would fall below ``MIN_REGION_CHARS``, the anchors latest in the text are
    dropped. Every gap between what is shown carries its own marker, and the
    result stays within the allowance plus one ``MARKER_RESERVE_CHARS``, as a
    prefix cut does.

    Args:
        text (str): The full text.
        anchors (Sequence[Sequence[int]]): ``(start, end)`` spans worth keeping.
        allowance (int): Characters it may show.

    Returns:
        tuple[str, bool]: The rendered text and whether it was trimmed.
    """
    spans = merge_spans(anchors, len(text))
    while spans and allowance < len(spans) * (MIN_REGION_CHARS + MARKER_RESERVE_CHARS):
        spans.pop()
    if len(text) <= allowance or all(end <= allowance for _, end in spans):
        return fit_with_marker(text, allowance)
    size = (allowance - len(spans) * MARKER_RESERVE_CHARS) // len(spans)
    regions = merge_spans([place(start, end, size, len(text)) for start, end in spans], len(text))
    parts: list[str] = []
    cursor = 0
    for start, end in regions:
        if start > cursor:
            parts.append(_marker(start - cursor, len(text)))
        parts.append(text[start:end])
        cursor = end
    if cursor < len(text):
        parts.append(_marker(len(text) - cursor, len(text)))
    return "".join(parts), True
