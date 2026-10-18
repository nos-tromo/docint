"""The keyword check of an agent turn: an exact, indexed word match over the turn's collection.

``search`` finds passages by meaning; ``keyword`` finds every indexed chunk that
contains a set of words and says how many there are, which a ranked search
cannot promise. The count is exact over what the keyword index covers, so a
zero is an answer rather than a miss. The match itself runs in RAG, which owns
the index, the filters and the retrieval chain; this module holds what the turn
does with a match (it anchors each hit on its words, writes the count note and
keys a check so the same one asked again is known) and imports nothing from
:mod:`docint.core.rag` or :mod:`docint.core.state`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import regex
from llama_index.core.schema import MetadataMode, NodeWithScore

from docint.agents.assessment.ledger import kept_ranges
from docint.core.search.fields import search_field_spec
from docint.core.search.fulltext import INDEX_WORD_CLASS, index_pieces, index_tokens, parse_keywords
from docint.utils.text_budget import MATCH_SPANS_KEY, MIN_REGION_CHARS, WINDOW_JOIN

#: The field a keyword call matches when it names none: the passage text.
DEFAULT_KEYWORD_FIELD = "text"
#: Longest keyword input in characters. Longer input is refused, never cut: a cut last word would widen the match.
MAX_KEYWORD_CHARS = 100
#: Characters before an anchored term that its region keeps, so a negation in front of it stays visible.
TERM_CONTEXT_CHARS = 80

#: Why a keyword call cannot run: none of its words is long enough for the index.
NO_USABLE_WORD = "no_usable_word"
#: Why a keyword call cannot run: the field matches one exact value, and several words were given.
ONE_VALUE_ONLY = "one_value_only"
#: Why a keyword call cannot run: the index skips a word or a piece of one, as ``D`` in ``vitamin D`` or ``2.5``.
UNINDEXABLE_WORD = "unindexable_word"
#: Why a keyword call cannot run: a word is longer than the index reads, which would match more than it names.
LONG_WORD = "long_word"

#: A word starts where the index starts one: no letter or digit right before it.
_WORD_START = rf"(?<![{INDEX_WORD_CLASS}])"
#: Between two pieces of one keyword: the rest of the first word, then anything the index splits on.
_PIECE_GAP = rf"[{INDEX_WORD_CLASS}]*[^{INDEX_WORD_CLASS}]+"


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
class KeywordMatcher:
    """The keyword match an agent turn offers its model, bound to its collection, filters and target.

    Attributes:
        fields (tuple[str, ...]): The fields whose indexes are ready, ``text``
            first when it is.
        check (Callable[[str, str], str | None]): Why ``(words, field)`` cannot
            run, :data:`NO_USABLE_WORD`, :data:`ONE_VALUE_ONLY`,
            :data:`UNINDEXABLE_WORD` or :data:`LONG_WORD`, or ``None``; it does no I/O.
        run (Callable[[str, str], KeywordResult]): Counts and retrieves the
            matches of ``(words, field)``; the turn runs it on a search slot.
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


def _term_pattern(keywords: Sequence[str]) -> regex.Pattern[str] | None:
    """Compile the index's word-prefix rule for a set of keywords.

    Each keyword matches as the index reads it: its pieces (see
    :func:`~docint.core.search.fulltext.index_pieces`) in order, each at the
    start of a word, whatever separates them in the text, so
    ``placebo-controlled`` finds ``placebo controlled`` and a dash of any kind,
    as the count does. A keyword's pieces stand together: anchored on any one of
    them, the ``19`` of ``COVID-19`` would match an unrelated ``1990`` first.

    Args:
        keywords (Sequence[str]): The words of the match.

    Returns:
        regex.Pattern[str] | None: A case-insensitive pattern matching any
        keyword, the longest first; ``None`` when no keyword holds a letter or digit.
    """
    alternatives = {
        _PIECE_GAP.join(regex.escape(piece) for piece in pieces) for pieces in map(index_pieces, keywords) if pieces
    }
    if not alternatives:
        return None
    ordered = sorted(alternatives, key=len, reverse=True)
    return regex.compile(_WORD_START + "(?:" + "|".join(ordered) + ")", regex.IGNORECASE)


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
    without spans is one match over its whole text, which packing shows from
    its opening (see :func:`_anchor_unstamped`).

    Args:
        scored (NodeWithScore): A passage a keyword call retrieved.
        keywords (Sequence[str]): The words the match used.

    Returns:
        NodeWithScore: A copy with the moved spans, or ``scored`` itself when no
        span moves; the input is never written to.
    """
    pattern = _term_pattern(keywords)
    if pattern is None:
        return scored
    text = scored.node.get_content(metadata_mode=MetadataMode.NONE)
    raw = (scored.node.metadata or {}).get(MATCH_SPANS_KEY)
    if not isinstance(raw, list):
        return _anchor_unstamped(scored, text, pattern)
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


def _anchor_unstamped(scored: NodeWithScore, text: str, pattern: regex.Pattern[str]) -> NodeWithScore:
    """Give a hit without spans a match where its word lies past the opening packing keeps of it.

    Packing shows an unstamped passage from its opening, so a word past that
    opening could be cut away while its count still reaches the answer. Such a
    hit gains one match, from just before the word to the end of the window
    holding it, so no region is charged in another window. The ledger merges it
    into the same text held unstamped, whose opening then counts as matches of
    its own.

    Args:
        scored (NodeWithScore): A passage without match spans.
        text (str): Its text.
        pattern (regex.Pattern[str]): The keywords' word-prefix rule (see :func:`_term_pattern`).

    Returns:
        NodeWithScore: A stamped copy, or ``scored`` itself when the text holds
        no keyword or its opening already shows the first one.
    """
    found = pattern.search(text)
    if found is None or any(lo <= found.start() and found.end() <= hi for lo, hi in kept_ranges(text, [])):
        return scored
    join = text.rfind(WINDOW_JOIN, 0, found.start())
    start = 0 if join < 0 else join + len(WINDOW_JOIN)
    # From the term's start: a keyword's pieces may straddle a join, and the span must not.
    end = text.find(WINDOW_JOIN, found.start())
    node = scored.node.model_copy(deep=True)
    node.metadata[MATCH_SPANS_KEY] = [[_context_start(text, start, found.start()), len(text) if end < 0 else end]]
    return NodeWithScore(node=node, score=scored.score)


def keyword_note(*, matches: int | None, unchecked: int) -> str:
    """Write the note that ends a keyword reply: the exact count, then what the index does not cover.

    The passages, or ``No new passages.``, are rendered by
    :func:`~docint.agents.turn.passages.answer_passages`; this note only says what the
    match counted and what it could not check.

    Args:
        matches (int | None): The exact count, or ``None`` when it failed.
        unchecked (int): Chunks the keyword index does not cover.

    Returns:
        str: The count line, then a coverage line when the index is partial, one per line.
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
    return "\n".join(lines)
