"""The numbered evidence an assessment turn gathers, and the room it may use."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from llama_index.core.schema import BaseNode, MetadataMode, NodeWithScore, TextNode

from docint.agents.assessment.split import neutralize_sentinel
from docint.utils.embed_chunking import estimate_tokens
from docint.utils.text_budget import (
    MARKER_RESERVE_CHARS,
    MATCH_SPANS_KEY,
    allocate_shortest_first,
    excerpt,
    merge_spans,
    place,
)

MAX_LEDGER_ENTRIES = 24
#: Smallest share of the answer budget worth giving one entry (~200 text + header).
MIN_ENTRY_TOKENS = 240
DIGEST_CHARS = 400
#: Longest file name a digest label carries.
DIGEST_LABEL_CHARS = 120
#: Seam between two windows of one passage; no brackets, so it never reads as a citation.
WINDOW_JOIN = "\n\n…\n\n"
#: Fewest characters a shortened passage may keep and still be read.
MIN_TEXT_CHARS = 300
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


def _merge_windows(held: NodeWithScore, arriving: NodeWithScore) -> NodeWithScore | None:
    """Fold another emission of a held passage into its entry, with its match.

    The node supplying the text is the one the entry is copied from: the held
    node when the arriving window extends it or adds only a match, the arriving
    one when it already holds the held text. Each side's match spans are moved
    to where its text sits in the result. Anything but a ``TextNode`` keeps the
    held entry as is.

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
        covered = merge_spans(held_spans, len(held_text))
        fresh = [
            (start, end)
            for start, end in _shifted(new_spans, held_text.find(new_text))
            if not any(lo <= start and end <= hi for lo, hi in covered)
        ]
        if not fresh:
            return None
        base, text, spans = held.node, held_text, held_spans + fresh
    elif held_text in new_text:
        base, text, spans = arriving.node, new_text, new_spans + _shifted(held_spans, new_text.find(held_text))
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
        a match the entry lacks. A full ledger admits no new entry but still
        extends one.

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
                if merged is not None:
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
        numbered: list[NodeWithScore] = []
        for number, scored in enumerate(self._entries, start=1):
            clone = scored.node.model_copy(deep=True)
            clone.metadata["citation_index"] = number
            clone.excluded_llm_metadata_keys = [k for k in clone.excluded_llm_metadata_keys if k != "citation_index"]
            numbered.append(NodeWithScore(node=clone, score=scored.score))
        return numbered


def digest(number: int, scored: NodeWithScore) -> str:
    """Render one bounded line the research model reads instead of the passage.

    A passage that says where its retrieved match sits is read from a view of
    ``DIGEST_CHARS`` placed on that match, led by ``…`` when the view skips the
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
    spans = merge_spans(_anchors(scored.node), len(raw))
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


def _parts(nodes: list[NodeWithScore], llm_view: Callable[[BaseNode], BaseNode]) -> tuple[list[str], list[str]]:
    """Split each passage's LLM view into its metadata header and its text.

    Args:
        nodes (list[NodeWithScore]): The numbered ledger.
        llm_view (Callable[[BaseNode], BaseNode]): Hides metadata the prompt must not see.

    Returns:
        tuple[list[str], list[str]]: The headers and the texts, sentinels defused.
    """
    headers: list[str] = []
    bodies: list[str] = []
    for scored in nodes:
        view = llm_view(scored.node)
        headers.append(neutralize_sentinel(view.get_metadata_str(mode=MetadataMode.LLM)))
        bodies.append(neutralize_sentinel(view.get_content(metadata_mode=MetadataMode.NONE)))
    return headers, bodies


def _allowances(
    headers: list[str],
    bodies: list[str],
    *,
    budget_tokens: int,
    char_token_ratio: float,
    count_tokens: Callable[[str], int] | None,
) -> tuple[int, list[int]]:
    """Share the characters left once every header, join and cut marker is reserved.

    Args:
        headers (list[str]): The metadata headers, kept whole.
        bodies (list[str]): The texts that share what is left.
        budget_tokens (int): Tokens the rendered evidence may use.
        char_token_ratio (float): The configured characters per token.
        count_tokens (Callable[[str], int] | None): A tokenizer, if one is loaded.

    Returns:
        tuple[int, list[int]]: The text budget, zero or less when the headers use
        it all, and each text's allowance in input order, shared shortest-first.
    """
    ratio = effective_ratio(headers + bodies, char_token_ratio=char_token_ratio, count_tokens=count_tokens)
    reserved = sum(len(header) + 4 for header in headers) + MARKER_RESERVE_CHARS * len(headers)
    char_budget = int(budget_tokens * ratio) - reserved
    return char_budget, allocate_shortest_first([len(body) for body in bodies], max(0, char_budget))


def evidence_fits(
    nodes: list[NodeWithScore],
    *,
    llm_view: Callable[[BaseNode], BaseNode],
    budget_tokens: int,
    char_token_ratio: float,
    count_tokens: Callable[[str], int] | None = None,
) -> bool:
    """Return whether the evidence can be shown legibly in its budget, packed as :func:`render_context` packs it.

    Args:
        nodes (list[NodeWithScore]): The numbered ledger.
        llm_view (Callable[[BaseNode], BaseNode]): Hides metadata the prompt must not see.
        budget_tokens (int): Tokens the rendered evidence may use.
        char_token_ratio (float): The configured characters per token.
        count_tokens (Callable[[str], int] | None): A tokenizer, if one is loaded.

    Returns:
        bool: ``True`` when text room remains after the headers and every passage
        that must be cut keeps at least ``MIN_TEXT_CHARS``; a passage shown whole
        fits however short it is.
    """
    headers, bodies = _parts(nodes, llm_view)
    char_budget, allowances = _allowances(
        headers, bodies, budget_tokens=budget_tokens, char_token_ratio=char_token_ratio, count_tokens=count_tokens
    )
    if char_budget <= 0:
        return False
    return all(
        allowance >= len(body) or allowance >= MIN_TEXT_CHARS
        for body, allowance in zip(bodies, allowances, strict=True)
    )


def _fit_windows(body: str, allowance: int, anchors: Sequence[tuple[int, int]] = ()) -> tuple[str, bool]:
    """Fit one passage to its allowance, window by window when it joins several, around its matches.

    Cut from the tail, a passage extended by :meth:`EvidenceLedger.add` would lose
    its newest window first, so each window is shortened on its own. The joins and
    one cut marker per window beyond the first come out of the allowance (the
    caller's reserve covers the first marker); the rest is shared shortest-first.
    Each window, or the passage as a whole, is cut by :func:`excerpt` around the
    match spans that fall inside it, so a match deep in a section is not cut away.
    The result stays within the allowance plus one marker reserve, as a
    whole-passage cut does.

    Args:
        body (str): The passage text.
        allowance (int): Characters it may show before its cut marker.
        anchors (Sequence[tuple[int, int]]): Where its retrieved matches sit in ``body``.

    Returns:
        tuple[str, bool]: The rendered text and whether it was shortened. A passage
        that fits, joins no windows, or leaves less than a character per window is
        cut as a whole.
    """
    windows = body.split(WINDOW_JOIN)
    seams = len(windows) - 1
    remainder = allowance - seams * (len(WINDOW_JOIN) + MARKER_RESERVE_CHARS)
    if len(body) <= allowance or seams == 0 or remainder < len(windows):
        return excerpt(body, anchors, allowance)
    shares = allocate_shortest_first([len(window) for window in windows], remainder)
    fitted: list[tuple[str, bool]] = []
    offset = 0
    for window, share in zip(windows, shares, strict=True):
        # Spans are clipped to the window by ``excerpt``.
        fitted.append(excerpt(window, _shifted(anchors, -offset), share))
        offset += len(window) + len(WINDOW_JOIN)
    return WINDOW_JOIN.join(text for text, _ in fitted), any(cut for _, cut in fitted)


def render_context(
    nodes: list[NodeWithScore],
    *,
    llm_view: Callable[[BaseNode], BaseNode],
    budget_tokens: int,
    char_token_ratio: float,
    count_tokens: Callable[[str], int] | None = None,
) -> tuple[str, int]:
    """Render numbered nodes like the grounded prompt's context, shortened to fit.

    Headers are always kept; the text budget is shared shortest-first and every
    cut is marked, so the model can tell a short excerpt from a short source. A
    passage that joins several windows is cut window by window, so each keeps text,
    and a passage is cut around its retrieved matches rather than to its opening.

    Args:
        nodes (list[NodeWithScore]): The numbered ledger.
        llm_view (Callable[[BaseNode], BaseNode]): Hides metadata the prompt
            must not see.
        budget_tokens (int): Tokens the rendered evidence may use.
        char_token_ratio (float): Characters per token for the budget.
        count_tokens (Callable[[str], int] | None): A tokenizer; when given, the
            evidence is packed at its measured density (see :func:`effective_ratio`).

    Returns:
        tuple[str, int]: The context block and how many entries were shortened.
    """
    headers, bodies = _parts(nodes, llm_view)
    _, allowances = _allowances(
        headers, bodies, budget_tokens=budget_tokens, char_token_ratio=char_token_ratio, count_tokens=count_tokens
    )
    rendered: list[str] = []
    shortened = 0
    for scored, header, body, allowance in zip(nodes, headers, bodies, allowances, strict=True):
        fitted, cut = _fit_windows(body, allowance, _anchors(scored.node))
        shortened += int(cut)
        rendered.append(f"{header}\n\n{fitted}" if header else fitted)
    # The join can complete a sentinel that neither part holds alone.
    return neutralize_sentinel("\n\n".join(rendered)), shortened
