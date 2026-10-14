"""The numbered evidence an assessment turn gathers, and the room it may use."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from llama_index.core.schema import BaseNode, MetadataMode, NodeWithScore

from docint.agents.assessment.split import neutralize_sentinel
from docint.utils.embed_chunking import estimate_tokens
from docint.utils.text_budget import allocate_shortest_first, fit_with_marker

MAX_LEDGER_ENTRIES = 24
#: Smallest share of the answer budget worth giving one entry (~200 text + header).
MIN_ENTRY_TOKENS = 240
DIGEST_CHARS = 400
#: Longest file name a digest label carries.
DIGEST_LABEL_CHARS = 120
#: Worst-case length of the cut marker, reserved for every entry.
MARKER_RESERVE_CHARS = 50
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


class EvidenceLedger:
    """Append-only evidence, numbered in the order it was found."""

    def __init__(self, *, cap: int) -> None:
        """Start empty.

        Args:
            cap (int): The most entries the ledger admits.
        """
        self._cap = max(1, cap)
        self._entries: list[NodeWithScore] = []
        self._ids: set[str] = set()

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
        """Admit passages not seen before, up to the cap.

        Args:
            nodes (Iterable[NodeWithScore]): Passages in ranked order.

        Returns:
            list[tuple[int, NodeWithScore]]: The newly admitted passages with
            their ledger numbers.
        """
        added: list[tuple[int, NodeWithScore]] = []
        for scored in nodes:
            if self.full:
                break
            node_id = scored.node.node_id
            if node_id in self._ids:
                continue
            self._ids.add(node_id)
            self._entries.append(scored)
            added.append((len(self._entries), scored))
        return added

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
    text = " ".join(scored.node.get_content(metadata_mode=MetadataMode.NONE).split())
    snippet = text[:DIGEST_CHARS]
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
        it all, and each text's allowance, shortest first.
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
    cut is marked, so the model can tell a short excerpt from a short source.

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
    for header, body, allowance in zip(headers, bodies, allowances, strict=True):
        fitted, cut = fit_with_marker(body, allowance)
        shortened += int(cut)
        rendered.append(f"{header}\n\n{fitted}" if header else fitted)
    # The join can complete a sentinel that neither part holds alone.
    return neutralize_sentinel("\n\n".join(rendered)), shortened
