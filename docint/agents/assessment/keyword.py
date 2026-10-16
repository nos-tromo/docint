"""The keyword research tool: an exact, indexed word match over the turn's collection.

``search`` finds passages by meaning; ``keyword`` finds every indexed chunk that
contains a set of words and says how many there are, which a ranked search
cannot promise. The count is exact over what the keyword index covers, so a
zero is an answer rather than a miss. The match itself runs in RAG, which owns
the index, the filters and the retrieval chain; this module holds what research
and the answer step do with a match, and imports nothing from
:mod:`docint.core.rag`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from llama_index.core.schema import MetadataMode, NodeWithScore

from docint.agents.assessment.split import CITATION_NUMBERS_RE, neutralize_sentinel
from docint.core.search.fields import search_field_spec
from docint.core.search.fulltext import index_tokens, parse_keywords
from docint.utils.text_budget import MATCH_SPANS_KEY, MIN_REGION_CHARS

#: The field a keyword call matches when it names none: the passage text.
DEFAULT_KEYWORD_FIELD = "text"
#: Longest keyword input in characters. Longer input is refused, never cut: a cut last word would widen the match.
MAX_KEYWORD_CHARS = 100
#: Characters before an anchored term that its region keeps, so a negation in front of it stays visible.
TERM_CONTEXT_CHARS = 80

#: What the answer prompt's checks block holds when research made no check with a known count.
NO_CHECKS = "(none)"
#: Why a keyword call cannot run: none of its words is long enough for the index.
NO_USABLE_WORD = "no_usable_word"
#: Why a keyword call cannot run: the field matches one exact value, and several words were given.
ONE_VALUE_ONLY = "one_value_only"
#: Why a keyword call cannot run: the index skips a word or a piece of one, as ``D`` in ``vitamin D`` or ``2.5``.
UNINDEXABLE_WORD = "unindexable_word"
#: Why a keyword call cannot run: a word is longer than the index reads, which would match more than it names.
LONG_WORD = "long_word"

_EDGE_PUNCTUATION_RE = re.compile(r"^\W+|\W+$")


@dataclass(frozen=True)
class KeywordResult:
    """What one keyword call found.

    Attributes:
        nodes (list[NodeWithScore]): The best matching passages, ranked and
            expanded as a search's are.
        matches (int | None): Indexed chunks in the searched material that
            contain every word; ``None`` when the count failed, never 0.
        unchecked (int): Chunks in the searched material the keyword index does
            not cover, so a count can be that much too low.
        keywords (tuple[str, ...]): The words the match used, as the index reads them.
    """

    nodes: list[NodeWithScore]
    matches: int | None
    unchecked: int
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class KeywordTool:
    """The keyword match a turn offers research, bound to its collection, filters and target.

    Attributes:
        fields (tuple[str, ...]): The fields whose indexes are ready, ``text``
            first when it is.
        check (Callable[[str, str], str | None]): Why ``(words, field)`` cannot
            run, :data:`NO_USABLE_WORD`, :data:`ONE_VALUE_ONLY`,
            :data:`UNINDEXABLE_WORD` or :data:`LONG_WORD`, or ``None``; it does no I/O.
        run (Callable[[str, str], KeywordResult]): Counts and retrieves the
            matches of ``(words, field)``; research runs it on a search slot.
    """

    fields: tuple[str, ...]
    check: Callable[[str, str], str | None]
    run: Callable[[str, str], KeywordResult]


def check_key(words: str, field: str) -> tuple[str, frozenset[str], frozenset[Any]]:
    """Return what a keyword check matches by, so the same check asked again is known.

    Text keys match the tokens the index reads, so case, order and punctuation it
    ignores make no new check. Value keys match exact values, in the forms the
    field tries them in, so two ids that only tokenize alike, or one id in two
    cases, are two checks, while a uuid pasted with or without its dashes is one.

    Args:
        words (str): The words of the check.
        field (str): A field the check may use.

    Returns:
        tuple[str, frozenset[str], frozenset[Any]]: The field, the index tokens of
        its text match, and the exact values of its value match.
    """
    spec = search_field_spec(field)
    tokens: frozenset[str] = frozenset()
    if spec.text_keys:
        tokens = frozenset(token for word in parse_keywords(words) for token in index_tokens(word))
    values: frozenset[Any] = frozenset(spec.value_forms(words)) if spec.value_keys else frozenset()
    return field, tokens, values


def _term_pattern(keywords: Sequence[str]) -> re.Pattern[str] | None:
    """Compile the index's word-prefix rule for a set of keywords.

    Args:
        keywords (Sequence[str]): The words of the match.

    Returns:
        re.Pattern[str] | None: A case-insensitive pattern matching any keyword
        at the start of a word, the longest first; ``None`` when no keyword
        holds a word character.
    """
    terms = {_EDGE_PUNCTUATION_RE.sub("", word) for word in keywords}
    ordered = sorted((term for term in terms if term), key=len, reverse=True)
    if not ordered:
        return None
    return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(term) for term in ordered) + ")", re.IGNORECASE)


def _context_start(text: str, start: int, found: int) -> int:
    """Return where an anchored region begins: a word start shortly before the term, never before its span.

    Args:
        text (str): The passage text.
        start (int): Where the span starts.
        found (int): Where the term starts, inside the span.

    Returns:
        int: The span's own start when the term sits within ``TERM_CONTEXT_CHARS``
        of it; otherwise the first word start at most that far before the term,
        or the term itself when no word starts there.
    """
    lead = found - TERM_CONTEXT_CHARS
    if lead <= start:
        return start
    if text[lead - 1].isspace():
        return lead
    gap = re.search(r"\s", text[lead:found])
    return lead + gap.end() if gap else found


def anchor_terms(scored: NodeWithScore, keywords: Sequence[str]) -> NodeWithScore:
    """Move each match of a keyword hit to where its words occur.

    A match span marks a retrieved child chunk, and packing keeps a region from
    its start, which shows the chunk's opening rather than the word the chunk
    was found by. Each span whose text holds a keyword at the start of a word
    past that opening starts again just before the first one, with some left
    context; its end stays put, and a span never starts before it did, so it
    cannot cross into the window before it. A span whose opening already shows
    the word stays as it is: moved, it would merge into a held entry as a fresh
    match and charge the room again for text the entry already shows. A passage
    without spans keeps its shape: stamping it would change how it merges with
    the ledger's entries.

    Args:
        scored (NodeWithScore): A passage a keyword call retrieved.
        keywords (Sequence[str]): The words the match used.

    Returns:
        NodeWithScore: A copy with the moved spans, or ``scored`` itself when no
        span moves; the input is never written to.
    """
    raw = (scored.node.metadata or {}).get(MATCH_SPANS_KEY)
    pattern = _term_pattern(keywords)
    if not isinstance(raw, list) or pattern is None:
        return scored
    text = scored.node.get_content(metadata_mode=MetadataMode.NONE)
    moved: list[Any] = []
    changed = False
    for item in raw:
        valid = isinstance(item, (list, tuple)) and len(item) == 2 and all(type(value) is int for value in item)
        if not valid or not 0 <= item[0] < item[1] <= len(text):
            moved.append(item)
            continue
        start, end = item
        found = pattern.search(text, start, end)
        shown = found is None or found.end() <= start + MIN_REGION_CHARS
        new_start = start if shown else _context_start(text, start, found.start())
        changed = changed or new_start != start
        moved.append([new_start, end])
    if not changed:
        return scored
    node = scored.node.model_copy(deep=True)
    node.metadata[MATCH_SPANS_KEY] = moved
    return NodeWithScore(node=node, score=scored.score)


def keyword_reply(*, matches: int | None, unchecked: int, digests: Sequence[str]) -> str:
    """Render the tool reply research reads after a keyword call.

    Args:
        matches (int | None): The exact count, or ``None`` when it failed.
        unchecked (int): Chunks the keyword index does not cover.
        digests (Sequence[str]): One line per new passage.

    Returns:
        str: The count, a coverage note when the index is partial, then the
        new passages or ``No new passages.``, one per line.
    """
    if matches is None:
        lines = ["The number of matching chunks is unavailable."]
    elif matches == 0:
        lines = ["No indexed chunk in the searched material contains all of these words."]
    elif matches == 1:
        lines = ["1 indexed chunk in the searched material contains all of these words."]
    else:
        lines = [f"{matches} indexed chunks in the searched material contain all of these words."]
    if unchecked == 1:
        lines.append("1 chunk is not in the keyword index yet, so it was not checked.")
    elif unchecked > 1:
        lines.append(f"{unchecked} chunks are not in the keyword index yet, so they were not checked.")
    lines.extend(digests or ["No new passages."])
    return "\n".join(lines)


def _check_words(detail: Any) -> str:
    """Defuse the model-written words of a check before the answer reads them.

    Args:
        detail (Any): The words as research recorded them.

    Returns:
        str: The words on one line, without citation numbers, sentinels or
        double quotes, at most ``MAX_KEYWORD_CHARS`` long.
    """
    text = CITATION_NUMBERS_RE.sub("", str(detail or "").replace('"', "'"))
    return " ".join(neutralize_sentinel(text).split())[:MAX_KEYWORD_CHARS]


def render_keyword_checks(actions: Sequence[Mapping[str, Any]], *, max_chars: int) -> str:
    """Render the keyword checks of a research run for the answer prompt.

    Only checks with a known count are listed: a failed count says nothing about
    the collection. Lines are kept whole, in research order, while they fit.

    Args:
        actions (Sequence[Mapping[str, Any]]): The research actions, in order.
        max_chars (int): Characters the block may take.

    Returns:
        str: One ``- "words" (field): count`` line per check, with
        ``; N chunks not checked`` when the index does not cover every chunk;
        :data:`NO_CHECKS` when there is none.
    """
    lines: list[str] = []
    used = 0
    for action in actions:
        matches = action.get("matches")
        if action.get("action") != "keyword" or type(matches) is not int:
            continue
        line = f'- "{_check_words(action.get("detail"))}" ({action.get("field") or "text"}): {matches}'
        unchecked = action.get("unchecked")
        if type(unchecked) is int and unchecked > 0:
            line += f"; {unchecked} chunk not checked" if unchecked == 1 else f"; {unchecked} chunks not checked"
        cost = len(line) + (1 if lines else 0)
        if used + cost > max_chars:
            break
        lines.append(line)
        used += cost
    return "\n".join(lines) or NO_CHECKS
