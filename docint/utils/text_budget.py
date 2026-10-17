"""Share a character budget across texts and mark what had to be cut."""

from collections.abc import Sequence

#: Node metadata naming where a passage's retrieved match sits: ``[start, end]`` spans in its own text.
MATCH_SPANS_KEY = "docint_match_spans"
#: Seam between two windows of one passage; no brackets, so it never reads as a citation.
WINDOW_JOIN = "\n\n…\n\n"
#: Worst-case length of the cut marker, reserved for every entry.
MARKER_RESERVE_CHARS = 50
#: Fewest characters a region kept around a match may show; a passage cut to 300 characters still keeps one.
MIN_REGION_CHARS = 200
#: Least allowance one kept region costs: its text and its marker.
REGION_CHARS = MIN_REGION_CHARS + MARKER_RESERVE_CHARS


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


def allocate_with_floors(lengths: Sequence[int], floors: Sequence[int], budget: int) -> list[int]:
    """Split a shared character budget so that each text first gets its floor.

    When the budget covers every floor, each counted up to its text's length,
    each text is granted its floor and the rest is shared shortest-first over
    what each still lacks. A budget that cannot cover them is shared as
    :func:`allocate_shortest_first` shares it, so floors of ``min(len, 300)``
    reproduce that allocation exactly.

    Args:
        lengths (Sequence[int]): Length of each text, in order.
        floors (Sequence[int]): The least each text should get, aligned with ``lengths``.
        budget (int): Total characters available across all texts.

    Returns:
        list[int]: Characters granted to each text, aligned with ``lengths``.
    """
    granted = [min(length, floor) for length, floor in zip(lengths, floors, strict=True)]
    if budget < sum(granted):
        return allocate_shortest_first(lengths, budget)
    lacking = [length - grant for length, grant in zip(lengths, granted, strict=True)]
    extra = allocate_shortest_first(lacking, budget - sum(granted))
    return [grant + more for grant, more in zip(granted, extra, strict=True)]


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


def _word_tail(text: str, allowance: int) -> str:
    """Take the last characters of a text, leaving out a word they would open in the middle of.

    Args:
        text (str): The full text, longer than ``allowance``.
        allowance (int): Characters it may show.

    Returns:
        str: At most ``allowance`` characters from the end, none of them leading whitespace.
    """
    if allowance <= 0:
        return ""
    start = len(text) - allowance
    tail = text[start:]
    if not text[start - 1].isspace() and not text[start].isspace():
        _, *rest = tail.split(None, 1)
        tail = rest[0] if rest else tail
    return tail.lstrip()


def fit_tail_with_marker(text: str, allowance: int) -> tuple[str, bool]:
    """Trim a text to its allowance, keeping its end and marking how much was withheld.

    For a text that grows by appending, the end is the newest part. Whole lines are
    kept from the end while they fit; when not even the last one does, its end is
    shown from the first full word in it. The marker leads, on a line of its own, and
    like :func:`fit_with_marker`'s comes on top of the allowance.

    Args:
        text (str): The full text.
        allowance (int): Characters it may show, the marker excluded.

    Returns:
        tuple[str, bool]: The rendered text and whether it was trimmed.
    """
    if len(text) <= allowance:
        return (text, False)
    kept = ""
    for line in reversed(text.splitlines(keepends=True)):
        if len(kept) + len(line) > allowance:
            break
        kept = line + kept
    shown = kept.lstrip() or _word_tail(text, allowance)
    marker = _marker(len(text) - len(shown), len(text))
    return (f"{marker}\n{shown}" if shown else marker, True)


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


def distinct_spans(spans: Sequence[Sequence[int]], length: int) -> list[tuple[int, int]]:
    """Clip spans to a text and drop the empty ones and exact repeats, keeping each match apart.

    Unlike :func:`merge_spans`, spans that touch, overlap or nest stay separate, so each
    one is still counted as its own match.

    Args:
        spans (Sequence[Sequence[int]]): ``(start, end)`` offsets, in any order.
        length (int): Length of the text they point into.

    Returns:
        list[tuple[int, int]]: The distinct non-empty spans inside ``[0, length]``, sorted by start, then end.
    """
    return sorted({(max(0, start), min(length, end)) for start, end in spans if min(length, end) > max(0, start)})


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
    :func:`fit_with_marker` cuts it. Otherwise each distinct anchor gets a
    region of an equal share of the allowance, less one marker each, even where
    anchors touch, overlap or nest (see :func:`distinct_spans`); only the
    regions are merged. While those shares would fall below
    ``MIN_REGION_CHARS``, the anchors latest in the text are dropped. Every gap
    between what is shown carries its own marker, and the result stays within
    the allowance plus one ``MARKER_RESERVE_CHARS``, as a prefix cut does.

    Args:
        text (str): The full text.
        anchors (Sequence[Sequence[int]]): ``(start, end)`` spans worth keeping.
        allowance (int): Characters it may show.

    Returns:
        tuple[str, bool]: The rendered text and whether it was trimmed.
    """
    spans = distinct_spans(anchors, len(text))
    while spans and allowance < len(spans) * REGION_CHARS:
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
