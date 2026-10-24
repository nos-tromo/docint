"""The numbered evidence a turn gathers, and how its passages are packed into a budget."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import NamedTuple

from llama_index.core.schema import BaseNode, MetadataMode, NodeWithScore, TextNode

from docint.agents.assessment.split import neutralize_sentinel
from docint.utils.text_budget import (
    MARKER_RESERVE_CHARS,
    MATCH_SPANS_KEY,
    MIN_REGION_CHARS,
    REGION_CHARS,
    WINDOW_JOIN,
    allocate_with_floors,
    distinct_spans,
    excerpt,
    fit_with_marker,
)

#: What a passage pays for each window past its first when it is cut: the join and that window's marker.
SEAM_CHARS = len(WINDOW_JOIN) + MARKER_RESERVE_CHARS
#: Fewest characters a shortened passage may keep and still be read.
MIN_TEXT_CHARS = 300
#: Metadata a compact header keeps after its citation line.
COMPACT_HEADER_CHARS = 240
#: Tokens a turn keeps free in the context window for the answer.
ANSWER_OUTPUT_TOKENS = 1024
#: A thinking model spends its reply budget on reasoning before the answer.
ANSWER_OUTPUT_TOKENS_REASONING = 3072
#: Share of the context window a turn plans to use; the rest absorbs estimation error.
SAFETY_MARGIN = 0.95
#: How OpenAI-compatible servers word a prompt longer than the model's context.
CONTEXT_ERROR_MARKERS = (
    "context length",
    "maximum model length",
    "context size",
    "context window",
    "too long",
    "context_length_exceeded",
    "input length exceeds",
)


def is_context_overflow(exc: BaseException) -> bool:
    """Return whether a failed model call says the prompt did not fit the context.

    Args:
        exc (BaseException): The failure.

    Returns:
        bool: ``True`` when its message carries an overflow marker, in any case.
    """
    message = str(exc).lower()
    return any(marker in message for marker in CONTEXT_ERROR_MARKERS)


def _anchors(node: BaseNode) -> list[tuple[int, int]]:
    """Return where a passage's retrieved matches sit in its text, skipping malformed spans.

    Args:
        node (BaseNode): The passage.

    Returns:
        list[tuple[int, int]]: Its ``MATCH_SPANS_KEY`` spans as ``(start, end)``; none when unstamped.
    """
    raw = (node.metadata or {}).get(MATCH_SPANS_KEY)
    if not isinstance(raw, list):
        return []
    return [
        (item[0], item[1])
        for item in raw
        if isinstance(item, (list, tuple)) and len(item) == 2 and all(type(value) is int for value in item)
    ]


def _shifted(spans: Iterable[tuple[int, int]], by: int) -> list[tuple[int, int]]:
    """Move spans to where their text sits in another, longer or shorter one.

    Args:
        spans (Iterable[tuple[int, int]]): The spans.
        by (int): Characters to add to each offset; negative moves them back.

    Returns:
        list[tuple[int, int]]: The moved spans.
    """
    return [(start + by, end + by) for start, end in spans]


def _match_ranges(body: str, span: tuple[int, int]) -> list[tuple[int, int]]:
    """Return what every packing at or above the floor keeps of one match: its opening in each window.

    Args:
        body (str): The passage text.
        span (tuple[int, int]): The match, in ``body``.

    Returns:
        list[tuple[int, int]]: For each window of ``body`` the match has a part in,
        that part's first ``MIN_REGION_CHARS``, in body coordinates; none when it has
        no part in any window.
    """
    start, end = span
    ranges: list[tuple[int, int]] = []
    offset = 0
    for window in body.split(WINDOW_JOIN):
        lo, hi = max(start, offset), min(end, offset + len(window))
        if hi > lo:
            ranges.append((lo, lo + min(MIN_REGION_CHARS, hi - lo)))
        offset += len(window) + len(WINDOW_JOIN)
    return ranges


def _has_match(body: str, spans: Sequence[tuple[int, int]]) -> bool:
    """Return whether a passage holds a match with a part in some window of its text.

    Args:
        body (str): The passage text.
        spans (Sequence[tuple[int, int]]): Its match spans.

    Returns:
        bool: ``True`` when packing keeps a region for at least one of them.
    """
    return any(_match_ranges(body, span) for span in spans)


def _covered(needed: Sequence[tuple[int, int]], others: Sequence[Sequence[tuple[int, int]]]) -> bool:
    """Return whether a single one of other matches keeps everything packing keeps of a match.

    Args:
        needed (Sequence[tuple[int, int]]): What packing keeps of the match (see :func:`_match_ranges`).
        others (Sequence[Sequence[tuple[int, int]]]): What it keeps of each match that may keep it, in the same text.

    Returns:
        bool: ``True`` when one of ``others`` contains each of ``needed``, or
        ``needed`` is empty; two matches that touch never count together.
    """
    return not needed or any(
        all(any(lo <= start and end <= hi for lo, hi in kept) for start, end in needed) for kept in others
    )


def head_spans(text: str) -> list[tuple[int, int]]:
    """Return matches whose kept text is all that packing keeps of a passage without any.

    An unstamped passage keeps its opening (see :func:`kept_ranges`), which can
    run past ``MIN_REGION_CHARS`` and, in a short passage, across a window join,
    while a match keeps only its first ``MIN_REGION_CHARS`` in each window. So
    each window's share of the opening becomes a match to the window's end, and
    a share longer than one match keeps gets a second, whose kept text ends
    where the share does. Merged with a stamped emission of the same text, the
    passage then still keeps everything the model was shown of it.

    Args:
        text (str): The passage text.

    Returns:
        list[tuple[int, int]]: The matches; none when packing keeps nothing of the passage.
    """
    spans: list[tuple[int, int]] = []
    kept = kept_ranges(text, [])
    offset = 0
    for window in text.split(WINDOW_JOIN):
        end = offset + len(window)
        for lo, hi in kept:
            first, last = max(lo, offset), min(hi, end)
            if last > first:
                spans.append((first, end))
                if last - first > MIN_REGION_CHARS:
                    spans.append((last - MIN_REGION_CHARS, end))
        offset = end + len(WINDOW_JOIN)
    return spans


def _merge_windows(held: NodeWithScore, arriving: NodeWithScore) -> NodeWithScore | None:
    """Fold another emission of a held passage into its entry, with its match.

    The node supplying the text is the one the entry is copied from: the held
    node when the arriving window extends it or adds only a match, the arriving
    one when it already holds the held text. Each side's match spans are moved
    to where its text sits in the result. Matches count one by one, as packing
    counts them, by what packing keeps of each (see :func:`_match_ranges`): an
    arriving match is news unless a single held match keeps all of it, and a held
    match a single arriving one keeps is dropped when the arriving window
    replaces the entry. A re-cut of a child that the child's own opening holds
    therefore adds no floor, while one that starts inside a long child keeps its
    own. A stamped and an unstamped emission of different texts never merge: one
    keeps its opening, the other its match. Of one text they do, as when a
    keyword check anchors a passage held unstamped: the unstamped side counts
    its opening as matches (see :func:`head_spans`), so the merged entry keeps
    all that the model was shown of either. Anything but a ``TextNode`` keeps the
    held entry as is.

    Args:
        held (NodeWithScore): The ledger's entry.
        arriving (NodeWithScore): Another emission under the same node id.

    Returns:
        NodeWithScore | None: A new entry with both excerpts, every match and the
        higher score; ``None`` when the entry already keeps the arriving text and
        every match it carries, or when exactly one of two different texts holds
        a match. Neither input is written to.
    """
    held_text = held.node.get_content(metadata_mode=MetadataMode.NONE)
    new_text = arriving.node.get_content(metadata_mode=MetadataMode.NONE)
    held_spans, new_spans = _anchors(held.node), _anchors(arriving.node)
    held_match = _has_match(held_text, held_spans)
    if held_match != _has_match(new_text, new_spans):
        if new_text != held_text:
            return None
        # One text with and without a match: the unmatched side keeps its opening as matches.
        if held_match:
            new_spans = head_spans(new_text)
        else:
            held_spans = head_spans(held_text)
    if new_text in held_text:
        held_matches = distinct_spans(held_spans, len(held_text))
        arriving_matches = distinct_spans(_shifted(new_spans, held_text.find(new_text)), len(held_text))
        held_kept = [_match_ranges(held_text, span) for span in held_matches]
        fresh = [span for span in arriving_matches if not _covered(_match_ranges(held_text, span), held_kept)]
        if not fresh:
            return None
        base, text, spans = held.node, held_text, held_spans + fresh
    elif held_text in new_text:
        arriving_matches = distinct_spans(new_spans, len(new_text))
        held_matches = distinct_spans(_shifted(held_spans, new_text.find(held_text)), len(new_text))
        arriving_kept = [_match_ranges(new_text, span) for span in arriving_matches]
        kept = [span for span in held_matches if not _covered(_match_ranges(new_text, span), arriving_kept)]
        base, text, spans = arriving.node, new_text, new_spans + kept
    else:
        shift = len(held_text) + len(WINDOW_JOIN)
        base, text, spans = held.node, f"{held_text}{WINDOW_JOIN}{new_text}", held_spans + _shifted(new_spans, shift)
    if not isinstance(base, TextNode):
        return None
    node = base.model_copy(deep=True)
    node.text = text
    if "window_chars" in node.metadata:
        node.metadata["window_chars"] = len(text)
    if spans:
        node.metadata[MATCH_SPANS_KEY] = [[start, end] for start, end in spans]
    else:
        node.metadata.pop(MATCH_SPANS_KEY, None)
    scores = [score for score in (held.score, arriving.score) if score is not None]
    return NodeWithScore(node=node, score=max(scores, default=None))


def stamp(scored: NodeWithScore, number: int) -> NodeWithScore:
    """Clone a passage stamped with the ledger number it is cited by.

    Args:
        scored (NodeWithScore): The passage; a cached node is never written to.
        number (int): Its ledger number.

    Returns:
        NodeWithScore: The clone, its ``citation_index`` visible to the prompt.
    """
    clone = scored.node.model_copy(deep=True)
    clone.metadata["citation_index"] = number
    clone.excluded_llm_metadata_keys = [k for k in clone.excluded_llm_metadata_keys if k != "citation_index"]
    return NodeWithScore(node=clone, score=scored.score)


class EvidenceLedger:
    """Evidence numbered in the order it was found; an entry keeps its number and can only grow."""

    def __init__(self, *, cap: int) -> None:
        """Start empty.

        Args:
            cap (int): The most entries the ledger admits.
        """
        self._cap = max(1, cap)
        self._entries: list[NodeWithScore] = []
        self._positions: dict[str, int] = {}

    def __len__(self) -> int:
        """Return how many entries the ledger holds.

        Returns:
            int: The entry count.
        """
        return len(self._entries)

    @property
    def full(self) -> bool:
        """Whether the ledger admits no more entries.

        Returns:
            bool: ``True`` at the cap.
        """
        return len(self._entries) >= self._cap

    def add(self, nodes: Iterable[NodeWithScore]) -> list[tuple[int, NodeWithScore]]:
        """Admit passages not seen before, up to the cap, and fold in new windows of passages held.

        Parent-context windows of one section share the section's node id, so a
        later window of a held id is new evidence: it extends that entry, which
        keeps its number. So is a window the entry already shows when it carries
        a match whose kept text the entry lacks (see :func:`_match_ranges`). A
        stamped and an unstamped emission of one id merge only when their text is
        the same (see :func:`_merge_windows`); otherwise whichever comes second is
        nothing new. A full ledger admits no new entry but still extends one.

        Args:
            nodes (Iterable[NodeWithScore]): Passages in ranked order.

        Returns:
            list[tuple[int, NodeWithScore]]: An entry for each passage admitted or
            extended: its ledger number and the arriving passage, whose text is
            the new excerpt rather than the merged entry.
        """
        changed: list[tuple[int, NodeWithScore]] = []
        for scored in nodes:
            node_id = scored.node.node_id
            position = self._positions.get(node_id)
            if position is not None:
                merged = _merge_windows(self._entries[position], scored)
                if merged is None:
                    continue
                self._entries[position] = merged
                changed.append((position + 1, scored))
            elif not self.full:
                self._positions[node_id] = len(self._entries)
                self._entries.append(scored)
                changed.append((len(self._entries), scored))
        return changed

    def entries(self) -> list[NodeWithScore]:
        """Return the entries in ledger order, unnumbered.

        Returns:
            list[NodeWithScore]: The admitted passages.
        """
        return list(self._entries)

    def numbered(self) -> list[NodeWithScore]:
        """Return clones stamped with their ledger number, in ledger order.

        Returns:
            list[NodeWithScore]: Clones carrying ``citation_index``; the cached
            originals are never written to.
        """
        return [stamp(scored, number) for number, scored in enumerate(self._entries, start=1)]

    def number_of(self, node_id: str) -> int | None:
        """Return the number a held passage is cited by.

        Args:
            node_id (str): The passage's node id, the ledger's dedupe key.

        Returns:
            int | None: Its ledger number, which a later window never moves; ``None``
            when the ledger holds no passage of that id.
        """
        position = self._positions.get(node_id)
        return None if position is None else position + 1


def kept_ranges(body: str, anchors: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return the ranges of a passage that every packing at or above its :func:`text_floor` keeps.

    A passage with a match in some window keeps, for each match in stored (rank)
    order, its first ``MIN_REGION_CHARS`` in each window it has a part in; an exact
    repeat counts once. There is no whole-text rule: a short window merged into an
    entry is floored per match, not shown whole. A passage with no such match keeps
    its whole text when that is no longer than its floor; otherwise its first
    window's share at its floor, which is its first ``MIN_TEXT_CHARS`` when it has
    one window.

    Args:
        body (str): The passage text, as packing cuts it.
        anchors (Sequence[tuple[int, int]]): Where its retrieved matches sit in ``body``, in stored order.

    Returns:
        list[tuple[int, int]]: The kept ranges in ``body``, matches in stored order;
        empty only when the floor leaves a passage without matches nothing of its
        first window.
    """
    ranges: list[tuple[int, int]] = []
    for anchor in anchors:
        ranges += [kept for kept in _match_ranges(body, anchor) if kept not in ranges]
    if ranges:
        return ranges
    floor = text_floor(body, anchors)
    if len(body) <= floor:
        return [(0, len(body))]
    windows = body.split(WINDOW_JOIN)
    if len(windows) == 1:
        return [(0, floor)]
    lengths = [len(window) for window in windows]
    share = allocate_with_floors(lengths, [0] * len(windows), floor - (len(windows) - 1) * SEAM_CHARS)[0]
    return [(0, share)] if share else []


def effective_ratio(
    texts: Sequence[str],
    *,
    char_token_ratio: float,
    count_tokens: Callable[[str], int] | None,
    sample_chars: int = 12_000,
) -> float:
    """Return the characters per token to pack with: measured when possible, never above the configured ratio.

    Args:
        texts (Sequence[str]): The parts to be rendered.
        char_token_ratio (float): The configured characters per token.
        count_tokens (Callable[[str], int] | None): A tokenizer, if one is loaded.
        sample_chars (int): The most characters measured, taken evenly across ``texts``.

    Returns:
        float: The denser of the measured and the configured ratio; the configured
        one without a tokenizer, without text, or when the tokenizer fails.
    """
    if count_tokens is None or not any(texts):
        return char_token_ratio
    sample = "\n\n".join(text[: sample_chars // max(1, len(texts))] for text in texts)
    try:
        tokens = count_tokens(sample)
    except Exception:
        return char_token_ratio
    if tokens <= 0:
        return char_token_ratio
    return min(char_token_ratio, len(sample) / tokens)


def _compact_header(view: BaseNode) -> str:
    """Render a header whose citation line comes first, whole, and whose other keys are cut short.

    Args:
        view (BaseNode): A numbered passage's LLM view.

    Returns:
        str: The ``citation_index`` line, then the remaining visible metadata cut
        to ``COMPACT_HEADER_CHARS`` with a marker.
    """
    number = view.metadata.get("citation_index")
    first = "" if number is None else view.metadata_template.format(key="citation_index", value=str(number))
    rest = view.model_copy()
    rest.excluded_llm_metadata_keys = [*view.excluded_llm_metadata_keys, "citation_index"]
    tail, _ = fit_with_marker(rest.get_metadata_str(mode=MetadataMode.LLM), COMPACT_HEADER_CHARS)
    return view.metadata_separator.join(part for part in (first, tail) if part)


def _windows(body: str, anchors: Sequence[tuple[int, int]]) -> list[tuple[str, list[tuple[int, int]]]]:
    """Split a passage at its window joins, each window with the matches inside it, as :func:`excerpt` sees them.

    A match crossing a ``WINDOW_JOIN`` that the source text itself holds is clipped
    into the windows on both sides, so it counts in both. Matches that touch,
    overlap or nest each count on their own; only an exact repeat counts once.

    Args:
        body (str): The passage text.
        anchors (Sequence[tuple[int, int]]): Where its retrieved matches sit in ``body``.

    Returns:
        list[tuple[str, list[tuple[int, int]]]]: Each window and its distinct
        matches, moved into the window and clipped to it (see :func:`distinct_spans`).
    """
    windows: list[tuple[str, list[tuple[int, int]]]] = []
    offset = 0
    for window in body.split(WINDOW_JOIN):
        windows.append((window, distinct_spans(_shifted(anchors, -offset), len(window))))
        offset += len(window) + len(WINDOW_JOIN)
    return windows


def _window_floor(window: str, spans: Sequence[tuple[int, int]]) -> int:
    """Return the least a window must keep so that :func:`excerpt` holds on to every match in it.

    Args:
        window (str): The window's text.
        spans (Sequence[tuple[int, int]]): Its distinct matches.

    Returns:
        int: A region and its marker per match, or the whole window when that is less.
    """
    return min(len(window), REGION_CHARS * len(spans))


def text_floor(body: str, anchors: Sequence[tuple[int, int]]) -> int:
    """Return the least text a passage must keep so that every retrieved match it holds stays in it.

    ``min(L, max(MIN_TEXT_CHARS, (W - 1) * SEAM_CHARS + max(W, sum(window floors))))``
    over its ``W`` windows, which for one window is ``min(L, max(300, 250 * k))``. Each
    region's 250 includes its marker, and the first marker is in the entry's reserve.

    Args:
        body (str): The passage text, as packing cuts it.
        anchors (Sequence[tuple[int, int]]): Where its retrieved matches sit in ``body``.

    Returns:
        int: The floor; never more than the passage, never less than the legible
        minimum a passage without matches keeps.
    """
    windows = _windows(body, anchors)
    needed = sum(_window_floor(window, spans) for window, spans in windows)
    return min(len(body), max(MIN_TEXT_CHARS, (len(windows) - 1) * SEAM_CHARS + max(len(windows), needed)))


def passage_parts(
    nodes: list[NodeWithScore], llm_view: Callable[[BaseNode], BaseNode], *, compact: bool = False
) -> tuple[list[str], list[str], list[list[tuple[int, int]]]]:
    """Split each passage's LLM view into its metadata header, its text and where its matches sit.

    Args:
        nodes (list[NodeWithScore]): The numbered ledger.
        llm_view (Callable[[BaseNode], BaseNode]): Hides metadata the prompt must not see.
        compact (bool): Whether headers are compacted (see :func:`_compact_header`).

    Returns:
        tuple[list[str], list[str], list[list[tuple[int, int]]]]: The headers, sentinels
        defused; the texts as stored, which their spans index (a text is defused once
        it is cut, see :func:`render_context`); and each text's match spans.
    """
    headers: list[str] = []
    bodies: list[str] = []
    anchors: list[list[tuple[int, int]]] = []
    for scored in nodes:
        view = llm_view(scored.node)
        header = _compact_header(view) if compact else view.get_metadata_str(mode=MetadataMode.LLM)
        headers.append(neutralize_sentinel(header))
        bodies.append(view.get_content(metadata_mode=MetadataMode.NONE))
        anchors.append(_anchors(scored.node))
    return headers, bodies, anchors


def passage_costs(header: str, body: str, anchors: Sequence[tuple[int, int]]) -> tuple[int, int]:
    """Price one passage as packing does.

    Args:
        header (str): Its rendered header.
        body (str): Its text.
        anchors (Sequence[tuple[int, int]]): Where its retrieved matches sit in ``body``.

    Returns:
        tuple[int, int]: What it reserves (header, joins, cut marker) and the
        least text it must keep, every match it holds included (see :func:`text_floor`).
    """
    return len(header) + 4 + MARKER_RESERVE_CHARS, text_floor(body, anchors)


def _allowances(
    headers: list[str],
    bodies: list[str],
    anchors: list[list[tuple[int, int]]],
    *,
    budget_tokens: int,
    char_token_ratio: float,
    count_tokens: Callable[[str], int] | None,
) -> tuple[int, list[int], list[int]]:
    """Share the characters left once every header, join and cut marker is reserved, floors first.

    Args:
        headers (list[str]): The metadata headers, kept whole.
        bodies (list[str]): The texts that share what is left.
        anchors (list[list[tuple[int, int]]]): Where each text's retrieved matches sit.
        budget_tokens (int): Tokens the rendered evidence may use.
        char_token_ratio (float): The configured characters per token.
        count_tokens (Callable[[str], int] | None): A tokenizer, if one is loaded.

    Returns:
        tuple[int, list[int], list[int]]: The text budget, zero or less when the
        headers use it all; each text's floor (see :func:`text_floor`); and each
        text's allowance in input order, every floor granted first when the budget
        covers them all (see :func:`allocate_with_floors`).
    """
    ratio = effective_ratio(headers + bodies, char_token_ratio=char_token_ratio, count_tokens=count_tokens)
    reserved = sum(len(header) + 4 for header in headers) + MARKER_RESERVE_CHARS * len(headers)
    char_budget = int(budget_tokens * ratio) - reserved
    floors = [text_floor(body, spans) for body, spans in zip(bodies, anchors, strict=True)]
    return char_budget, floors, allocate_with_floors([len(body) for body in bodies], floors, max(0, char_budget))


def _legible(char_budget: int, floors: Sequence[int], allowances: Sequence[int]) -> bool:
    """Judge a packing legible: text room remains and every passage keeps its floor.

    Args:
        char_budget (int): The text budget left after headers, joins and markers.
        floors (Sequence[int]): Each passage's :func:`text_floor`.
        allowances (Sequence[int]): Each passage's allowance, aligned with ``floors``.

    Returns:
        bool: Whether the packing keeps every passage legible.
    """
    return char_budget > 0 and all(allowance >= floor for floor, allowance in zip(floors, allowances, strict=True))


def _fit_windows(body: str, allowance: int, anchors: Sequence[tuple[int, int]] = ()) -> tuple[str, bool]:
    """Fit one passage to its allowance, window by window when it joins several, around its matches.

    Cut from the tail, a passage extended by :meth:`EvidenceLedger.add` would lose
    its newest window first, so each window is shortened on its own. The joins and
    one cut marker per window beyond the first come out of the allowance (the
    caller's reserve covers the first marker); the rest is shared with each
    window's floor granted first, a region per match it holds, then shortest-first.
    Each window, or the passage as a whole, is cut by :func:`excerpt` around the
    match spans that fall inside it, so a match deep in a section is not cut away;
    an allowance of at least :func:`text_floor` keeps every one. The result stays
    within the allowance plus one marker reserve, as a whole-passage cut does.

    Args:
        body (str): The passage text.
        allowance (int): Characters it may show before its cut marker.
        anchors (Sequence[tuple[int, int]]): Where its retrieved matches sit in ``body``.

    Returns:
        tuple[str, bool]: The rendered text and whether it was shortened. A passage
        that fits, joins no windows, or leaves less than a character per window is
        cut as a whole.
    """
    windows = _windows(body, anchors)
    remainder = allowance - (len(windows) - 1) * SEAM_CHARS
    if len(body) <= allowance or len(windows) == 1 or remainder < len(windows):
        return excerpt(body, anchors, allowance)
    floors = [_window_floor(window, spans) for window, spans in windows]
    shares = allocate_with_floors([len(window) for window, _ in windows], floors, remainder)
    fitted = [excerpt(window, spans, share) for (window, spans), share in zip(windows, shares, strict=True)]
    return WINDOW_JOIN.join(text for text, _ in fitted), any(cut for _, cut in fitted)


class RenderedContext(NamedTuple):
    """Evidence packed into a budget.

    Attributes:
        context (str): The context block.
        shortened (int): How many entries were shortened.
        legible (bool): Whether text room remained after the headers and every
            passage kept its :func:`text_floor`.
    """

    context: str
    shortened: int
    legible: bool


def render_context(
    nodes: list[NodeWithScore],
    *,
    llm_view: Callable[[BaseNode], BaseNode],
    budget_tokens: int,
    char_token_ratio: float,
    count_tokens: Callable[[str], int] | None = None,
    compact: bool = False,
) -> RenderedContext:
    """Render numbered nodes like the grounded prompt's context, shortened to fit.

    Headers are always kept; the text budget is shared with each passage's
    :func:`text_floor` granted first, then shortest-first, and every cut is marked,
    so the model can tell a short excerpt from a short source. A passage that joins
    several windows is cut window by window, so each keeps text, and a passage is
    cut around its retrieved matches rather than to its opening. Texts are cut as
    stored, where their match spans point, and defused afterwards with the context.

    Args:
        nodes (list[NodeWithScore]): The numbered ledger.
        llm_view (Callable[[BaseNode], BaseNode]): Hides metadata the prompt
            must not see.
        budget_tokens (int): Tokens the rendered evidence may use.
        char_token_ratio (float): Characters per token for the budget.
        count_tokens (Callable[[str], int] | None): A tokenizer; when given, the
            evidence is packed at its measured density (see :func:`effective_ratio`).
        compact (bool): Whether headers are compacted (see :func:`_compact_header`).

    Returns:
        RenderedContext: The context block, how many entries were shortened, and
        whether the packing kept every passage legible.
    """
    headers, bodies, anchors = passage_parts(nodes, llm_view, compact=compact)
    char_budget, floors, allowances = _allowances(
        headers,
        bodies,
        anchors,
        budget_tokens=budget_tokens,
        char_token_ratio=char_token_ratio,
        count_tokens=count_tokens,
    )
    rendered: list[str] = []
    shortened = 0
    for header, body, spans, allowance in zip(headers, bodies, anchors, allowances, strict=True):
        fitted, cut = _fit_windows(body, allowance, spans)
        shortened += int(cut)
        rendered.append(f"{header}\n\n{fitted}" if header else fitted)
    # Texts are defused once cut, which only shortens them, and a join can complete a sentinel no part holds.
    context = neutralize_sentinel("\n\n".join(rendered))
    return RenderedContext(context, shortened, _legible(char_budget, floors, allowances))
