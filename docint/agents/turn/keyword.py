"""The keyword check of an agent turn: an exact, indexed word match over the turn's collection.

``search`` finds passages by meaning; ``keyword`` finds every indexed chunk that
contains a set of words and says how many there are, which a ranked search
cannot promise. The count is exact over what the keyword index covers, so a
zero is an answer rather than a miss. The match itself runs in RAG, which owns
the index, the filters and the retrieval chain; this module holds what the turn
does with a match (it anchors each hit on its words, writes the count note and
keys a check so the same one asked again is known), the :class:`KeywordTool`
the model calls it by, and the checks block the turn's validator reads its
counts from. It imports nothing from :mod:`docint.core.rag` or
:mod:`docint.core.state`.
"""

from __future__ import annotations

import functools
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import regex
from llama_index.core.schema import MetadataMode, NodeWithScore

from docint.agents.assessment.ledger import kept_ranges
from docint.agents.assessment.split import CITATION_NUMBERS_RE, neutralize_sentinel
from docint.agents.turn.tools import (
    ErrorKind,
    ToolContext,
    ToolReply,
    ToolWork,
    error_reply,
    passage_reply,
    trail_detail,
)
from docint.core.search.fields import SEARCH_FIELDS, search_field_spec
from docint.core.search.fulltext import INDEX_WORD_CLASS, index_pieces, index_tokens, parse_keywords
from docint.core.search.index import SEARCH_MAX_TOKEN_LEN
from docint.utils.text_budget import MATCH_SPANS_KEY, MIN_REGION_CHARS, WINDOW_JOIN

#: The field a keyword call matches when it names none: the passage text.
DEFAULT_KEYWORD_FIELD = "text"
#: Longest keyword input in characters. Longer input is refused, never cut: a cut last word would widen the match.
MAX_KEYWORD_CHARS = 100
#: Characters before an anchored term that its region keeps, so a negation in front of it stays visible.
TERM_CONTEXT_CHARS = 80
#: Characters the keyword checks may take in the validator's prompt; whole lines only.
KEYWORD_CHECKS_CHARS = 800

# Each of the four problems below is also the error kind of the reply that refuses the call.
#: Why a keyword call cannot run: none of its words is long enough for the index.
NO_USABLE_WORD = "no_usable_word"
#: Why a keyword call cannot run: the field matches one exact value, and several words were given.
ONE_VALUE_ONLY = "one_value_only"
#: Why a keyword call cannot run: the index skips a word or a piece of one, as ``D`` in ``vitamin D`` or ``2.5``.
UNINDEXABLE_WORD = "unindexable_word"
#: Why a keyword call cannot run: a word is longer than the index reads, which would match more than it names.
LONG_WORD = "long_word"

KEYWORD_DESCRIPTION = (
    "Count indexed chunks containing all these words, in any order (a word also matches longer words it begins), "
    "and return the best new ones under [n]. For exact names, numbers and ids, or whether something is mentioned "
    "at all: a count of 0 is exact."
)
KEYWORD_WORDS_DESCRIPTION = (
    "One to three words every match must contain; no single letter or digit, alone or where dots or signs split "
    f"a word (2.5, U.S.), nor words over {SEARCH_MAX_TOKEN_LEN} characters."
)
#: What each field matches, for the schema of a tool that offers more than the text.
KEYWORD_FIELD_NOTES = {
    "text": "the passage text (the default)",
    "author": "who wrote a posting",
    "network": "the platform",
    "uuid": "one posting's id",
}

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
        fields (tuple[str, ...]): The fields whose indexes are ready: the text, always first,
            then any others, each a key of :data:`~docint.core.search.fields.SEARCH_FIELDS`.
        check (Callable[[str, str], str | None]): Why ``(words, field)`` cannot
            run, :data:`NO_USABLE_WORD`, :data:`ONE_VALUE_ONLY`,
            :data:`UNINDEXABLE_WORD` or :data:`LONG_WORD`, or ``None``. It must stay free of
            I/O and of side effects beyond a log line: one call runs it up to twice, once to key
            the call and once to check it.
        run (Callable[[str, str], KeywordResult]): Counts and retrieves the
            matches of ``(words, field)``; the turn runs it on a search slot.
    """

    fields: tuple[str, ...]
    check: Callable[[str, str], str | None]
    run: Callable[[str, str], KeywordResult]

    def __post_init__(self) -> None:
        """Check the contract the tool's schema and repeat keys rely on.

        Raises:
            ValueError: When ``fields`` is empty or does not start with the text, or names a
                field the search does not know.
        """
        if not self.fields or self.fields[0] != DEFAULT_KEYWORD_FIELD:
            raise ValueError(f"A keyword matcher offers the {DEFAULT_KEYWORD_FIELD} first; got {self.fields!r}.")
        unknown = ", ".join(repr(name) for name in self.fields if name not in SEARCH_FIELDS)
        if unknown:
            raise ValueError(f"A keyword matcher offers only fields the search knows; unknown: {unknown}.")


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
    """Write the note that leads a keyword reply: the exact count, then what the index does not cover.

    The passages, or ``No new passages.``, follow it, rendered by
    :func:`~docint.agents.turn.passages.answer_passages`; this note only says what the
    match counted and what it could not check. Leading, it survives the overflow retry,
    which keeps the head of every tool reply. The chunks it could not check may still
    contain the words, and the coverage line says so: a zero rules out only what was checked.

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
        lines.append(
            "1 chunk is not in the keyword index yet, so it was not checked and may still contain these words."
        )
    elif unchecked > 1:
        lines.append(
            f"{unchecked} chunks are not in the keyword index yet, so they were not checked and may still contain "
            "these words."
        )
    return "\n".join(lines)


def _check_words(detail: Any) -> str:
    """Defuse the model-written words of a check before the validator reads them.

    Args:
        detail (Any): The words as the turn's trail recorded them.

    Returns:
        str: The words on one line, without citation numbers, sentinels or
        double quotes, at most ``MAX_KEYWORD_CHARS`` long.
    """
    text = CITATION_NUMBERS_RE.sub("", str(detail or "").replace('"', "'"))
    return " ".join(neutralize_sentinel(text).split())[:MAX_KEYWORD_CHARS]


def render_keyword_checks(actions: Sequence[Mapping[str, Any]], *, max_chars: int) -> str:
    """Render a turn's keyword checks for its validator, which reads the counts as facts beside the sources.

    Only keyword steps whose count is known are listed: a failed count says nothing
    about the collection. Lines are kept whole, in trail order, while they fit.

    Args:
        actions (Sequence[Mapping[str, Any]]): The turn's trail (``agent.actions``), in order.
        max_chars (int): Characters the block may take.

    Returns:
        str: One ``- "words" (field): count`` line per check, with ``; N chunks not checked``
        when the index does not cover every chunk; empty when there is none.
    """
    lines: list[str] = []
    used = 0
    for action in actions:
        matches = action.get("matches")
        if action.get("action") != "keyword" or type(matches) is not int:
            continue
        line = f'- "{_check_words(action.get("detail"))}" ({action.get("field") or DEFAULT_KEYWORD_FIELD}): {matches}'
        unchecked = action.get("unchecked")
        if type(unchecked) is int and unchecked > 0:
            line += f"; {unchecked} chunk not checked" if unchecked == 1 else f"; {unchecked} chunks not checked"
        cost = len(line) + (1 if lines else 0)
        if used + cost > max_chars:
            break
        lines.append(line)
        used += cost
    return "\n".join(lines)


def _keyword_parameters(fields: Sequence[str]) -> dict[str, Any]:
    """Build the JSON schema of a keyword call's arguments, its field limited to the offered ones.

    A tool that offers only the text takes no field at all: naming fields the
    collection does not hold would only invite calls that fail.

    Args:
        fields (Sequence[str]): The fields whose indexes are ready, the text first
            (see :class:`KeywordMatcher`).

    Returns:
        dict[str, Any]: The schema of one object: ``words``, and ``field`` when anything
        besides the text is offered.
    """
    properties: dict[str, Any] = {"words": {"type": "string", "description": KEYWORD_WORDS_DESCRIPTION}}
    if tuple(fields) != (DEFAULT_KEYWORD_FIELD,):
        notes = "; ".join(f"{name}: {KEYWORD_FIELD_NOTES[name]}" for name in fields if name in KEYWORD_FIELD_NOTES)
        properties["field"] = {
            "type": "string",
            "enum": list(fields),
            "default": DEFAULT_KEYWORD_FIELD,
            "description": f"{notes}.",
        }
    return {"type": "object", "properties": properties, "required": ["words"]}


def _request(arguments: Mapping[str, Any]) -> tuple[str, str]:
    """Read a keyword call's words and field, whatever types the model sent.

    Args:
        arguments (Mapping[str, Any]): The call's arguments.

    Returns:
        tuple[str, str]: The words on one line, and the field, the text when none is named.
    """
    words = " ".join(str(arguments.get("words") or "").split())
    return words, str(arguments.get("field") or DEFAULT_KEYWORD_FIELD)


class KeywordTool:
    """``keyword(words[, field])``: count the indexed chunks that hold every word and show the best of them."""

    def __init__(self, matcher: KeywordMatcher) -> None:
        """Bind the match and describe the tool by the fields it offers.

        Args:
            matcher (KeywordMatcher): The turn's keyword match, with its collection, filters and
                target; its fields are the ones the tool offers, the text first.
        """
        self.name = "keyword"
        self.description = KEYWORD_DESCRIPTION
        self.parameters = _keyword_parameters(matcher.fields)
        self._matcher = matcher

    def prepare(self, arguments: Mapping[str, Any], ctx: ToolContext) -> ToolReply | ToolWork:
        """Check the words and the field; the match is the call's fetch.

        Args:
            arguments (Mapping[str, Any]): ``words``, and optionally ``field``, what to match them in.
            ctx (ToolContext): Unused; a match needs nothing of the evidence to start.

        Returns:
            ToolReply | ToolWork: An error reply saying why the call cannot run, which is never
            remembered; otherwise the match to run and the answer to give with what it found.
        """
        words, field = _request(arguments)
        detail = trail_detail(words)
        refusal = self._refusal(words, field)
        if refusal is not None:
            message, kind = refusal
            return error_reply(self.name, message, kind=kind, detail=detail)
        return ToolWork(
            detail=detail,
            fetch=functools.partial(self._matcher.run, words, field),
            answer=functools.partial(self._answer, field, detail),
        )

    def recall_key(self, arguments: Mapping[str, Any]) -> str | None:
        """Name a call by what it matches, so the same check asked in other words is known.

        Asked before the arguments are checked, so it takes any arguments and never raises on
        them. A call the tool refuses has no key of its own: the index drops a lone ``D`` from
        ``vitamin D``, which would otherwise be answered as a repeat of ``vitamin``. A call whose
        value the search layer cannot read has none either.

        Args:
            arguments (Mapping[str, Any]): The call's arguments.

        Returns:
            str | None: The field, the index tokens of the words and their exact values, each
            sorted (see :func:`check_key`); ``None`` for a call this tool refuses or whose key
            the search layer cannot make, which is left to the default key.
        """
        words, field = _request(arguments)
        if self._refusal(words, field) is not None:
            return None
        try:
            key_field, tokens, values = check_key(words, field)
        except ValueError:
            return None
        return json.dumps([key_field, sorted(tokens), sorted(str(value) for value in values)], ensure_ascii=False)

    def _refusal(self, words: str, field: str) -> tuple[str, ErrorKind] | None:
        """Say why a call cannot run, in the order its problems are found.

        Args:
            words (str): The words on one line.
            field (str): The field.

        Returns:
            tuple[str, ErrorKind] | None: The message for an error reply and its kind, which for a
            problem the match's check names is that problem; ``None`` when the call can run.
        """
        if not words:
            return "call keyword with the words to match.", "missing_argument"
        if len(words) > MAX_KEYWORD_CHARS:
            return f"keep the words within {MAX_KEYWORD_CHARS} characters; use the most telling ones.", "too_long"
        if field not in self._matcher.fields:
            return f"field must be one of: {', '.join(self._matcher.fields)}.", "bad_field"
        problem = self._matcher.check(words, field)
        if problem == NO_USABLE_WORD:
            return "give at least one word of two or more characters.", NO_USABLE_WORD
        if problem == ONE_VALUE_ONLY:
            return f"{field} matches one exact value; give a single one.", ONE_VALUE_ONLY
        if problem == UNINDEXABLE_WORD:
            message = (
                "the index cannot match a single letter or digit, alone, such as the D in vitamin D, or where dots "
                "or signs split a word, such as 2.5, U.S. or p<0.05, nor a word with no letter or digit at all; "
                "check the other words only, or use search."
            )
            return message, UNINDEXABLE_WORD
        if problem == LONG_WORD:
            message = (
                f"the index reads a word only up to its first {SEARCH_MAX_TOKEN_LEN} characters, so a longer one "
                "cannot be checked exactly; check a shorter beginning of it, which also matches the longer words "
                "it begins."
            )
            return message, LONG_WORD
        return None

    def _answer(self, field: str, detail: str, fetched: KeywordResult, ctx: ToolContext) -> ToolReply:
        """Show the best new matches after the exact count.

        Args:
            field (str): The field the call matched.
            detail (str): The call's main argument, for the trail.
            fetched (KeywordResult): What the match found.
            ctx (ToolContext): The turn's evidence, budget and LLM view.

        Returns:
            ToolReply: The count note, then the passages, as a :class:`PassageReply` that is news even
            when it admits nothing, so the same call can be answered by number. A plain reply when
            the count failed, which is never remembered, so a retry runs again.
        """
        nodes = fetched.nodes
        if field == DEFAULT_KEYWORD_FIELD:
            nodes = [anchor_terms(scored, fetched.keywords) for scored in nodes]
        note = keyword_note(matches=fetched.matches, unchecked=fetched.unchecked)
        shown = passage_reply(nodes, ctx, action=self.name, detail=detail, note=note, note_first=True)
        trail: dict[str, Any] = {"field": field, "matches": fetched.matches}
        if fetched.unchecked > 0:
            trail["unchecked"] = fetched.unchecked
        if fetched.matches is None:
            return ToolReply(text=shown.text, found=shown.found, action=shown.action, detail=shown.detail, trail=trail)
        return replace(shown, informative=True, trail=trail)
