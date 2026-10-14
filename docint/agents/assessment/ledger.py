"""The numbered evidence an assessment turn gathers, and the room it may use."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

from llama_index.core.schema import BaseNode, MetadataMode, NodeWithScore, TextNode

from docint.agents.assessment.split import neutralize_sentinel
from docint.utils.embed_chunking import estimate_tokens
from docint.utils.text_budget import (
    MARKER_RESERVE_CHARS,
    MATCH_SPANS_KEY,
    REGION_CHARS,
    WINDOW_JOIN,
    allocate_with_floors,
    distinct_spans,
    excerpt,
    fit_with_marker,
    place,
)

MAX_LEDGER_ENTRIES = 24
#: Smallest share of the answer budget worth giving one entry (~200 text + header).
MIN_ENTRY_TOKENS = 240
DIGEST_CHARS = 400
#: Longest file name a digest label carries.
DIGEST_LABEL_CHARS = 120
#: What a passage pays for each window past its first when it is cut: the join and that window's marker.
SEAM_CHARS = len(WINDOW_JOIN) + MARKER_RESERVE_CHARS
#: Fewest characters a shortened passage may keep and still be read.
MIN_TEXT_CHARS = 300
#: Metadata a compact header keeps after its citation line; compaction is the overflow retry's fallback only.
COMPACT_HEADER_CHARS = 240
ANSWER_OUTPUT_TOKENS = 1024
#: A thinking model spends its reply budget on reasoning before the answer.
ANSWER_OUTPUT_TOKENS_REASONING = 3072
PRIOR_TURN_TOKENS = 400
QUESTION_TOKENS = 150
SLACK_TOKENS = 64
SAFETY_MARGIN = 0.95
#: Share of the evidence budget an overflowing answer prompt is rebuilt with, once.
OVERFLOW_RETRY_SHARE = 0.6
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


@dataclass(frozen=True)
class AnswerBudget:
    """How much the answer step may spend.

    Attributes:
        evidence_tokens (int): Room for the rendered ledger.
        output_tokens (int): Reply room kept free inside the window; no longer
            sent as ``max_tokens``, so the answer call is uncapped like a lookup's.
        max_entries (int): Ledger cap that keeps every entry legible.
    """

    evidence_tokens: int
    output_tokens: int
    max_entries: int


def answer_budget(*, ctx_window: int, template: str, reasoning: bool, char_token_ratio: float) -> AnswerBudget:
    """Size the answer step from the context window.

    Args:
        ctx_window (int): The chat model's context window in tokens.
        template (str): The answer prompt template, measured for its overhead.
        reasoning (bool): Whether the answer model thinks before answering.
        char_token_ratio (float): Characters per token for estimation.

    Returns:
        AnswerBudget: Evidence room, output reserve and ledger cap.
    """
    output = ANSWER_OUTPUT_TOKENS_REASONING if reasoning else ANSWER_OUTPUT_TOKENS
    evidence = max(
        0,
        int(ctx_window * SAFETY_MARGIN)
        - estimate_tokens(template, char_token_ratio)
        - PRIOR_TURN_TOKENS
        - QUESTION_TOKENS
        - SLACK_TOKENS
        - output,
    )
    return AnswerBudget(
        evidence_tokens=evidence,
        output_tokens=output,
        max_entries=max(1, min(MAX_LEDGER_ENTRIES, evidence // MIN_ENTRY_TOKENS)),
    )


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


def _inside_one(span: tuple[int, int], spans: Sequence[tuple[int, int]]) -> bool:
    """Return whether a match lies wholly inside a single one of other matches.

    Args:
        span (tuple[int, int]): The match.
        spans (Sequence[tuple[int, int]]): The matches that may hold it.

    Returns:
        bool: ``True`` when one of ``spans`` holds all of it; lying across two that touch does not count.
    """
    start, end = span
    return any(lo <= start and end <= hi for lo, hi in spans)


def _merge_windows(held: NodeWithScore, arriving: NodeWithScore) -> NodeWithScore | None:
    """Fold another emission of a held passage into its entry, with its match.

    The node supplying the text is the one the entry is copied from: the held
    node when the arriving window extends it or adds only a match, the arriving
    one when it already holds the held text. Each side's match spans are moved
    to where its text sits in the result. Matches count one by one, as packing
    counts them: an arriving match is news unless a single held match holds it,
    and a held match a single arriving one holds is dropped when the arriving
    window replaces the entry, so a re-cut of the same child never raises its
    floor. Anything but a ``TextNode`` keeps the held entry as is.

    Args:
        held (NodeWithScore): The ledger's entry.
        arriving (NodeWithScore): Another emission under the same node id.

    Returns:
        NodeWithScore | None: A new entry with both excerpts, every match and the
        higher score, or ``None`` when the entry already shows the arriving text
        and every match it carries. Neither input is written to.
    """
    held_text = held.node.get_content(metadata_mode=MetadataMode.NONE)
    new_text = arriving.node.get_content(metadata_mode=MetadataMode.NONE)
    held_spans, new_spans = _anchors(held.node), _anchors(arriving.node)
    if new_text in held_text:
        held_matches = distinct_spans(held_spans, len(held_text))
        arriving_matches = distinct_spans(_shifted(new_spans, held_text.find(new_text)), len(held_text))
        fresh = [span for span in arriving_matches if not _inside_one(span, held_matches)]
        if not fresh:
            return None
        base, text, spans = held.node, held_text, held_spans + fresh
    elif held_text in new_text:
        arriving_matches = distinct_spans(new_spans, len(new_text))
        held_matches = distinct_spans(_shifted(held_spans, new_text.find(held_text)), len(new_text))
        kept = [span for span in held_matches if not _inside_one(span, arriving_matches)]
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


def _stamp(scored: NodeWithScore, number: int) -> NodeWithScore:
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
        self._room: Room | None = None
        self._refusals = 0

    def __len__(self) -> int:
        """Return how many entries the ledger holds.

        Returns:
            int: The entry count.
        """
        return len(self._entries)

    @property
    def refusals(self) -> int:
        """How many passages the room has turned away.

        Returns:
            int: Extensions and new passages the room refused, and new passages
            skipped while it was out of room; none skipped at the entry cap, and
            none before a room is installed.
        """
        return self._refusals

    @property
    def out_of_room(self) -> bool:
        """Whether the room, and not the entry cap, has closed the ledger.

        Returns:
            bool: ``True`` while the ledger is below its cap but its room cannot
            hold one more legible passage; never without a room installed.
        """
        return len(self._entries) < self._cap and self._room is not None and self._room.closed

    @property
    def full(self) -> bool:
        """Whether the ledger admits no more entries.

        Returns:
            bool: ``True`` at the cap, or once its room cannot hold even one more legible passage.
        """
        return len(self._entries) >= self._cap or (self._room is not None and self._room.closed)

    def restrict(self, admits: Room) -> None:
        """Admit from now on only what keeps the evidence legible.

        Installed once the seed is in: research may then add an entry, or extend
        one, only while ``admits`` says every passage still fits legibly.

        Args:
            admits (Room): The room the evidence must stay legible in, holding the entries so far.
        """
        self._room = admits

    def _admitted(self, number: int, candidate: NodeWithScore) -> bool:
        """Ask the room, if one is installed, whether a new or extended entry keeps the evidence legible.

        Args:
            number (int): The ledger number the entry carries.
            candidate (NodeWithScore): The new entry, or the merged one replacing it.

        Returns:
            bool: ``True`` when it may be taken; the room then counts it.
        """
        return self._room is None or self._room(number, candidate)

    def add(self, nodes: Iterable[NodeWithScore]) -> list[tuple[int, NodeWithScore]]:
        """Admit passages not seen before, up to the cap, and fold in new windows of passages held.

        Parent-context windows of one section share the section's node id, so a
        later window of a held id is new evidence: it extends that entry, which
        keeps its number. So is a window the entry already shows when it carries
        a match the entry lacks. A full ledger admits no new entry but still
        extends one. Once :meth:`restrict` is installed, a passage or extension
        that would leave the evidence illegible is refused and the rest of the
        batch is still considered; :attr:`refusals` counts what the room turns away.

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
                if self._admitted(position + 1, merged):
                    self._entries[position] = merged
                    changed.append((position + 1, scored))
                else:
                    self._refusals += 1
            elif self.out_of_room:
                self._refusals += 1
            elif not self.full and self._admitted(len(self._entries) + 1, scored):
                self._positions[node_id] = len(self._entries)
                self._entries.append(scored)
                changed.append((len(self._entries), scored))
            elif not self.full:
                self._refusals += 1
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
        return [_stamp(scored, number) for number, scored in enumerate(self._entries, start=1)]


def digest(number: int, scored: NodeWithScore) -> str:
    """Render one bounded line the research model reads instead of the passage.

    A passage that says where its retrieved matches sit is read from a view of
    ``DIGEST_CHARS`` placed on the first one it names that lies in its text,
    which is its top-ranked child, led by ``…`` when the view skips the
    passage's opening; any other passage is read from its start.

    Args:
        number (int): The passage's ledger number.
        scored (NodeWithScore): The passage.

    Returns:
        str: ``[n] (file, p. x) text…`` on a single line.
    """
    meta = scored.node.metadata or {}
    name = " ".join(str(meta.get("filename") or meta.get("file_name") or "").split())
    # Brackets would read as an entry marker, so the label keeps only parentheses.
    name = name[:DIGEST_LABEL_CHARS].replace("[", "(").replace("]", ")")
    page = meta.get("page") if meta.get("page") is not None else meta.get("page_number")
    where = f"{name}, p. {page}" if name and page is not None else name
    raw = scored.node.get_content(metadata_mode=MetadataMode.NONE)
    # Stored order, not text order: the first stamp is the top-ranked child.
    spans = [clipped for anchor in _anchors(scored.node) for clipped in distinct_spans([anchor], len(raw))]
    if spans:
        start, end = place(*spans[0], DIGEST_CHARS, len(raw))
        snippet = " ".join(raw[start:end].split())
        snippet = f"… {snippet}" if start > 0 else snippet
    else:
        snippet = " ".join(raw.split())[:DIGEST_CHARS]
    line = f"[{number}] ({where}) {snippet}" if where else f"[{number}] {snippet}"
    return neutralize_sentinel(" ".join(line.split()))


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


def _parts(
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


def _costs(header: str, body: str, anchors: Sequence[tuple[int, int]]) -> tuple[int, int]:
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


class Room:
    """Legibility of growing evidence, kept as running sums.

    With ``room = int(budget * ratio) - sum(header + 4 + MARKER_RESERVE_CHARS)``, the
    evidence is legible exactly when ``room > 0`` and ``room >= sum(text_floor)``,
    which is what :func:`evidence_fits` finds by packing it. The ratio is measured
    once, on the evidence the room starts from, and then held, so each candidate
    costs one header render.
    """

    def __init__(
        self,
        nodes: list[NodeWithScore],
        *,
        llm_view: Callable[[BaseNode], BaseNode],
        budget_tokens: int,
        char_token_ratio: float,
        count_tokens: Callable[[str], int] | None = None,
        compact: bool = False,
    ) -> None:
        """Measure the evidence so far.

        Args:
            nodes (list[NodeWithScore]): The numbered entries held so far.
            llm_view (Callable[[BaseNode], BaseNode]): Hides metadata the prompt must not see.
            budget_tokens (int): Tokens the rendered evidence may use.
            char_token_ratio (float): The configured characters per token.
            count_tokens (Callable[[str], int] | None): A tokenizer, if one is loaded.
            compact (bool): Whether headers are priced compacted.
        """
        headers, bodies, anchors = _parts(nodes, llm_view, compact=compact)
        ratio = effective_ratio(headers + bodies, char_token_ratio=char_token_ratio, count_tokens=count_tokens)
        self._llm_view = llm_view
        self._compact = compact
        self._chars = int(budget_tokens * ratio)
        self._costs = [_costs(*part) for part in zip(headers, bodies, anchors, strict=True)]
        self._reserved = sum(reserve for reserve, _ in self._costs)
        self._floor = sum(floor for _, floor in self._costs)

    def _fits(self, reserved: int, floor: int) -> bool:
        """Apply the closed form.

        Args:
            reserved (int): Characters every header, join and marker reserves.
            floor (int): Text every passage must keep.

        Returns:
            bool: Whether text room remains and covers the floor.
        """
        room = self._chars - reserved
        return room > 0 and room >= floor

    @property
    def legible(self) -> bool:
        """Whether every passage held can be shown legibly.

        Returns:
            bool: The closed form over the entries so far.
        """
        return self._fits(self._reserved, self._floor)

    @property
    def headers_fit(self) -> bool:
        """Whether any text room is left once the headers, joins and markers are reserved.

        Returns:
            bool: ``True`` while the headers alone do not exhaust the budget.
        """
        return self._chars - self._reserved > 0

    @property
    def closed(self) -> bool:
        """Whether not even a header-less passage of ``MIN_TEXT_CHARS`` would still fit.

        Returns:
            bool: ``True`` once research can add nothing worth reading.
        """
        return self._chars - self._reserved - self._floor < 4 + MARKER_RESERVE_CHARS + MIN_TEXT_CHARS

    def __call__(self, number: int, candidate: NodeWithScore) -> bool:
        """Admit a new entry, or the merged form of a held one, when the evidence stays legible.

        Args:
            number (int): The ledger number it carries; one past the last for a new entry.
            candidate (NodeWithScore): The entry as it would be held.

        Returns:
            bool: ``True`` when it fits; its cost is then counted in place of the entry it replaces.
        """
        headers, bodies, anchors = _parts([_stamp(candidate, number)], self._llm_view, compact=self._compact)
        reserve, floor = _costs(headers[0], bodies[0], anchors[0])
        held_reserve, held_floor = self._costs[number - 1] if number <= len(self._costs) else (0, 0)
        reserved = self._reserved - held_reserve + reserve
        floors = self._floor - held_floor + floor
        if not self._fits(reserved, floors):
            return False
        if number <= len(self._costs):
            self._costs[number - 1] = (reserve, floor)
        else:
            self._costs.append((reserve, floor))
        self._reserved, self._floor = reserved, floors
        return True


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


def evidence_fits(
    nodes: list[NodeWithScore],
    *,
    llm_view: Callable[[BaseNode], BaseNode],
    budget_tokens: int,
    char_token_ratio: float,
    count_tokens: Callable[[str], int] | None = None,
    compact: bool = False,
) -> bool:
    """Return whether the evidence can be shown legibly in its budget, packed as :func:`render_context` packs it.

    Args:
        nodes (list[NodeWithScore]): The numbered ledger.
        llm_view (Callable[[BaseNode], BaseNode]): Hides metadata the prompt must not see.
        budget_tokens (int): Tokens the rendered evidence may use.
        char_token_ratio (float): The configured characters per token.
        count_tokens (Callable[[str], int] | None): A tokenizer, if one is loaded.
        compact (bool): Whether headers are compacted, as the overflow retry renders them.

    Returns:
        bool: ``True`` when text room remains after the headers and every passage
        keeps its :func:`text_floor`: at least ``MIN_TEXT_CHARS``, and a region for
        every retrieved match it holds; a passage shown whole fits however short it is.
    """
    headers, bodies, anchors = _parts(nodes, llm_view, compact=compact)
    return _legible(
        *_allowances(
            headers,
            bodies,
            anchors,
            budget_tokens=budget_tokens,
            char_token_ratio=char_token_ratio,
            count_tokens=count_tokens,
        )
    )


def fit_ledger(
    nodes: list[NodeWithScore],
    *,
    keep: int,
    llm_view: Callable[[BaseNode], BaseNode],
    budget_tokens: int,
    char_token_ratio: float,
    count_tokens: Callable[[str], int] | None = None,
    compact: bool = False,
) -> list[NodeWithScore]:
    """Leave out the latest research additions until every passage can be shown legibly.

    Density is measured afresh on each candidate set, so evidence denser than the
    seed it was admitted against still fits. Entries go from the tail, which keeps
    the numbering contiguous; the first ``keep`` (the seed) are never left out, even
    when they alone are not legible.

    Args:
        nodes (list[NodeWithScore]): The numbered ledger.
        keep (int): How many leading entries must stay.
        llm_view (Callable[[BaseNode], BaseNode]): Hides metadata the prompt must not see.
        budget_tokens (int): Tokens the rendered evidence may use.
        char_token_ratio (float): The configured characters per token.
        count_tokens (Callable[[str], int] | None): A tokenizer, if one is loaded.
        compact (bool): Whether headers are compacted.

    Returns:
        list[NodeWithScore]: The leading entries that will be rendered.
    """
    kept = list(nodes)
    while len(kept) > keep and not evidence_fits(
        kept,
        llm_view=llm_view,
        budget_tokens=budget_tokens,
        char_token_ratio=char_token_ratio,
        count_tokens=count_tokens,
        compact=compact,
    ):
        kept.pop()
    return kept


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
    """Evidence packed into the answer prompt's context.

    Attributes:
        context (str): The context block.
        shortened (int): How many entries were shortened.
        legible (bool): Whether every passage kept its :func:`text_floor`, as
            :func:`evidence_fits` judges it for the same evidence and budget.
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
        compact (bool): Whether headers are compacted (see :func:`_compact_header`);
            only the overflow retry compacts them.

    Returns:
        RenderedContext: The context block, how many entries were shortened, and
        whether the packing kept every passage legible.
    """
    headers, bodies, anchors = _parts(nodes, llm_view, compact=compact)
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
