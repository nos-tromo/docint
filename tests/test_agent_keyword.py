"""Unit tests for the keyword check: anchoring, the repeat key, the count note and the tool. All invented."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest
from agent_fakes import KEYWORD_STEPS, keyword_match, programme_passage, shown_numbers, tool_call, tool_context
from assessment_fakes import llm_view, scored_node
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.schema import MetadataMode, NodeWithScore

from docint.agents.assessment.ledger import EvidenceLedger, passage_costs, passage_parts, render_context, text_floor
from docint.agents.turn import keyword as keyword_module
from docint.agents.turn.keyword import (
    KEYWORD_CHECKS_CHARS,
    KEYWORD_DESCRIPTION,
    KEYWORD_FIELD_NOTES,
    LONG_WORD,
    MAX_KEYWORD_CHARS,
    NO_USABLE_WORD,
    ONE_VALUE_ONLY,
    TERM_CONTEXT_CHARS,
    UNINDEXABLE_WORD,
    WRITING_LINE,
    KeywordMatcher,
    KeywordResult,
    KeywordTool,
    anchor_terms,
    check_key,
    keyword_note,
    render_keyword_checks,
)
from docint.agents.turn.loop import _shortened
from docint.agents.turn.passages import NO_NEW_PASSAGES
from docint.agents.turn.stream import ToolCall
from docint.agents.turn.tools import (
    PassageReply,
    ToolContext,
    ToolRegistry,
    ToolReply,
    ToolWork,
    cleared_text,
    passage_reply,
    trail_detail,
)
from docint.core.search.fields import SEARCH_FIELDS, UnknownSearchFieldError
from docint.core.search.index import SEARCH_MAX_TOKEN_LEN
from docint.utils.text_budget import MATCH_SPANS_KEY, MIN_REGION_CHARS, WINDOW_JOIN

LEAD = "The trial enrolled two hundred patients across four sites over two years. " * 3
TERM_TEXT = f"{LEAD}There was keine Kontrollgruppe in the design, which the authors acknowledge."


def _spans(scored: NodeWithScore) -> list[list[int]]:
    """Return a passage's stored match spans.

    Args:
        scored (NodeWithScore): The passage.

    Returns:
        list[list[int]]: Its ``MATCH_SPANS_KEY`` value.
    """
    return list(scored.node.metadata[MATCH_SPANS_KEY])


def _shown_at_floor(ledger: EvidenceLedger) -> str:
    """Render a ledger's first entry with exactly the text room its floor grants, as the tightest legible reply does.

    Args:
        ledger (EvidenceLedger): The ledger.

    Returns:
        str: The rendered passage, header included.
    """
    numbered = ledger.numbered()[:1]
    headers, bodies, anchors = passage_parts(numbered, llm_view)
    reserve, floor = passage_costs(headers[0], bodies[0], anchors[0])
    return render_context(numbered, llm_view=llm_view, budget_tokens=reserve + floor, char_token_ratio=1.0).context


def test_a_span_moves_to_just_before_the_term_and_keeps_its_end() -> None:
    """The kept region then shows the term instead of the chunk's opening."""
    scored = scored_node("s", TERM_TEXT, **{MATCH_SPANS_KEY: [[0, len(TERM_TEXT)]]})

    anchored = anchor_terms(scored, ["kontrollgruppe"])

    start, end = _spans(anchored)[0]
    term = TERM_TEXT.index("Kontrollgruppe")
    assert end == len(TERM_TEXT)
    assert term - TERM_CONTEXT_CHARS <= start < term


def test_the_left_context_starts_at_a_word_and_keeps_a_negation() -> None:
    """A negation right before the term must stay inside what is shown."""
    scored = scored_node("s", TERM_TEXT, **{MATCH_SPANS_KEY: [[0, len(TERM_TEXT)]]})

    start, _ = _spans(anchor_terms(scored, ["Kontrollgruppe"]))[0]

    assert TERM_TEXT[start - 1].isspace()
    assert "keine Kontrollgruppe" in TERM_TEXT[start : start + 200]


def test_a_span_never_moves_before_its_own_start() -> None:
    """A term near the span's start leaves the span where it was."""
    text = "Kontrollgruppe fehlt. " + LEAD
    scored = scored_node("s", text, **{MATCH_SPANS_KEY: [[0, len(text)]]})

    assert _spans(anchor_terms(scored, ["Kontrollgruppe"])) == [[0, len(text)]]


#: A word past the left context but inside the opening packing keeps of its span.
NEAR_TEXT = "Patients from two sites took part in it. " * 3 + "There was keine Kontrollgruppe here." + " More." * 40


def test_a_term_inside_the_kept_opening_leaves_the_span_alone() -> None:
    """The opening already shows the word, so moving the span would only charge the room twice."""
    term = NEAR_TEXT.index("Kontrollgruppe")
    assert TERM_CONTEXT_CHARS < term and term + len("Kontrollgruppe") <= MIN_REGION_CHARS
    scored = scored_node("s", NEAR_TEXT, **{MATCH_SPANS_KEY: [[0, len(NEAR_TEXT)]]})

    assert anchor_terms(scored, ["Kontrollgruppe"]) is scored


def test_a_held_section_showing_the_term_gains_nothing_from_its_check() -> None:
    """Re-found by a check, a section whose kept opening holds the word is no new evidence."""
    held = scored_node("s", NEAR_TEXT, **{MATCH_SPANS_KEY: [[0, len(NEAR_TEXT)]]})
    ledger = EvidenceLedger(cap=5)
    ledger.add([held])

    assert ledger.add([anchor_terms(held, ["Kontrollgruppe"])]) == []


def test_a_term_outside_the_span_does_not_move_it() -> None:
    """Only an occurrence starting inside the span anchors it."""
    span = [0, len(LEAD)]
    scored = scored_node("s", TERM_TEXT, **{MATCH_SPANS_KEY: [span]})

    assert anchor_terms(scored, ["Kontrollgruppe"]) is scored


def test_a_passage_without_spans_is_returned_as_is() -> None:
    """An unstamped passage short enough to be shown whole already shows its term, so it keeps its shape."""
    scored = scored_node("s", TERM_TEXT)

    assert anchor_terms(scored, ["Kontrollgruppe"]) is scored


#: An invented unstamped chunk, as a collection without parent sections stores it, its telling sentence deep inside.
UNSTAMPED_FACT = "No randomization was used."
UNSTAMPED_TEXT = (
    "The cohort was followed for twelve months at three clinics. " * 12
    + UNSTAMPED_FACT
    + " The authors call it a limitation."
)


def test_an_unstamped_hit_is_anchored_on_its_term_past_the_opening() -> None:
    """Packing shows an unstamped passage from its opening, so a term past it gets a match of its own."""
    scored = scored_node("u", UNSTAMPED_TEXT, filename="trial.pdf")

    anchored = anchor_terms(scored, ["randomization"])

    term = UNSTAMPED_TEXT.index("randomization")
    [[start, end]] = _spans(anchored)
    assert term - TERM_CONTEXT_CHARS <= start < term
    assert end == len(UNSTAMPED_TEXT)
    assert MATCH_SPANS_KEY not in scored.node.metadata


@pytest.mark.parametrize(("lead", "stamped"), [(291, False), (292, True)], ids=["ends-at-300", "ends-at-301"])
def test_an_unstamped_hit_is_anchored_only_past_the_opening_packing_keeps(lead: int, stamped: bool) -> None:
    """Packing keeps an unstamped passage's first 300 characters, so a term ending inside them needs no match.

    Args:
        lead (int): Characters before the space that precedes the term.
        stamped (bool): Whether the hit gains a match.
    """
    scored = scored_node("u", "w" * lead + " Zielwert" + " filler" * 60)

    assert (anchor_terms(scored, ["Zielwert"]) is not scored) is stamped


def test_an_unstamped_hit_is_anchored_inside_the_window_holding_its_term() -> None:
    """A match reaching back across a window join would be charged a region in the window before it."""
    first = "Opening filler words. " * 40
    text = f"{first}{WINDOW_JOIN}Short lead. Zielwert appears here.{' Trailing filler.' * 20}"

    [[start, end]] = _spans(anchor_terms(scored_node("u", text), ["Zielwert"]))

    assert len(first) + len(WINDOW_JOIN) <= start < text.index("Zielwert")
    assert end == len(text)


def test_the_input_passage_is_never_written_to() -> None:
    """Retrieved nodes can be cached and shared, so anchoring works on a copy."""
    scored = scored_node("s", TERM_TEXT, **{MATCH_SPANS_KEY: [[0, len(TERM_TEXT)]]})

    anchored = anchor_terms(scored, ["Kontrollgruppe"])

    assert anchored is not scored
    assert _spans(scored) == [[0, len(TERM_TEXT)]]


def test_matching_follows_the_index_word_prefix_rule() -> None:
    """A keyword matches the start of a word, case-insensitively, never its middle."""
    text = f"{LEAD}Die Bundespartei tagte. {LEAD}Der Parteitag begann."
    scored = scored_node("s", text, **{MATCH_SPANS_KEY: [[0, len(text)]]})

    start, _ = _spans(anchor_terms(scored, ["partei"]))[0]

    assert text.index("Parteitag") - TERM_CONTEXT_CHARS <= start
    assert "Parteitag" in text[start : start + 200]


def test_the_earliest_of_several_words_anchors() -> None:
    """All words occur in a match, so the first one found leads the region."""
    text = f"{LEAD}alpha comes first. {LEAD}beta comes later."
    scored = scored_node("s", text, **{MATCH_SPANS_KEY: [[0, len(text)]]})

    start, _ = _spans(anchor_terms(scored, ["beta", "alpha"]))[0]

    assert "alpha" in text[start : start + 200]


def test_punctuation_around_a_keyword_does_not_hide_the_term() -> None:
    """The index splits on punctuation, so a keyword typed with a bracket still anchors."""
    scored = scored_node("s", TERM_TEXT, **{MATCH_SPANS_KEY: [[0, len(TERM_TEXT)]]})

    start, _ = _spans(anchor_terms(scored, ["(Kontrollgruppe,"]))[0]

    assert start > 0


@pytest.mark.parametrize(
    "gap",
    [" ", "-", chr(0x2011), f" {chr(0x2013)} "],
    ids=["space", "hyphen", "non-breaking-hyphen", "spaced-en-dash"],
)
def test_a_hyphenated_check_anchors_on_its_words_however_the_text_joins_them(gap: str) -> None:
    """The index reads placebo-controlled as two words, and so does the anchor, whatever stands between them.

    Args:
        gap (str): What separates the two words in the passage.
    """
    text = f"{LEAD}The trial had no placebo{gap}controlled arm at any site."
    scored = scored_node("s", text, **{MATCH_SPANS_KEY: [[0, len(text)]]})

    start, end = _spans(anchor_terms(scored, ["placebo-controlled"]))[0]

    assert text.index("placebo") - TERM_CONTEXT_CHARS <= start < text.index("placebo")
    assert end == len(text)


def test_a_short_piece_of_a_hyphenated_check_never_anchors_alone() -> None:
    """Its words stand together, or the 19 of COVID-19 would anchor on an earlier, unrelated 1990."""
    text = f"{LEAD}The clinic opened in 1990 and grew. {'Filler about staffing. ' * 6}Then COVID-19 closed it."
    scored = scored_node("s", text, **{MATCH_SPANS_KEY: [[0, len(text)]]})

    start, _ = _spans(anchor_terms(scored, ["COVID-19"]))[0]

    assert "COVID-19" in text[start : start + MIN_REGION_CHARS]
    assert "1990" not in text[start : start + MIN_REGION_CHARS]


def test_a_word_after_an_underscore_starts_where_the_index_starts_one() -> None:
    """The index splits on underscores, so a word after one is a word start for the anchor as well."""
    text = f"{LEAD}The export was named site_report_final by the board."
    scored = scored_node("s", text, **{MATCH_SPANS_KEY: [[0, len(text)]]})

    start, _ = _spans(anchor_terms(scored, ["report"]))[0]

    assert start > 0
    assert "site_report" in text[start : start + MIN_REGION_CHARS]


def test_an_unstamped_hit_is_anchored_on_a_hyphenated_check_its_text_spaces() -> None:
    """A chunk without parent context gains its match where the index found the words, not where they are typed."""
    text = "The cohort was followed for twelve months at three clinics. " * 12 + "There was no placebo controlled arm."

    [[start, end]] = _spans(anchor_terms(scored_node("u", text), ["placebo-controlled"]))

    assert text.index("placebo") - TERM_CONTEXT_CHARS <= start < text.index("placebo")
    assert end == len(text)


def test_each_span_anchors_on_its_own_occurrence() -> None:
    """Every retrieved child of a section is moved within itself, in stored order."""
    first = f"{LEAD}Kontrollgruppe one."
    second = f"{LEAD}Kontrollgruppe two."
    text = f"{first} {second}"
    spans = [[len(first) + 1, len(text)], [0, len(first)]]
    scored = scored_node("s", text, **{MATCH_SPANS_KEY: spans})

    moved = _spans(anchor_terms(scored, ["Kontrollgruppe"]))

    assert moved[0][0] > len(first) + 1 and moved[0][1] == len(text)
    assert 0 < moved[1][0] < len(first) and moved[1][1] == len(first)


@pytest.mark.parametrize("keywords", [[], ["--"]], ids=["no-words", "no-letter-or-digit"])
@pytest.mark.parametrize("stamped", [True, False], ids=["stamped", "unstamped"])
def test_keywords_without_a_word_leave_a_hit_as_it_is(keywords: list[str], stamped: bool) -> None:
    """With no letter or digit to find there is nothing to anchor on: the very hit comes back, however deep its text.

    Args:
        keywords (list[str]): Words the index reads as nothing.
        stamped (bool): Whether the hit carries match spans.
    """
    if stamped:
        scored, word = scored_node("s", DEEP_TEXT, **{MATCH_SPANS_KEY: [[0, len(DEEP_TEXT)]]}), "Kontrollgruppe"
    else:
        scored, word = scored_node("u", UNSTAMPED_TEXT), "randomization"

    assert anchor_terms(scored, keywords) is scored
    # The same hit does move for a word it holds, so the identity above is the keywords' doing.
    assert anchor_terms(scored, [word]) is not scored


@pytest.mark.parametrize(
    "malformed",
    [[5, 3], "oops", [0, 10_000], [1.5, 9]],
    ids=["inverted", "a-string", "past-the-text", "not-whole-numbers"],
)
def test_a_malformed_stored_span_is_kept_as_stored_while_a_valid_one_beside_it_moves(malformed: Any) -> None:
    """Spans come from stored metadata: one that is no span is left alone and does not stop the others.

    Args:
        malformed (Any): A stored value that is no span of the passage.
    """
    scored = scored_node("s", TERM_TEXT, **{MATCH_SPANS_KEY: [malformed, [0, len(TERM_TEXT)]]})

    anchored = anchor_terms(scored, ["Kontrollgruppe"])

    kept, moved = anchored.node.metadata[MATCH_SPANS_KEY]
    term = TERM_TEXT.index("Kontrollgruppe")
    assert kept == malformed
    assert term - TERM_CONTEXT_CHARS <= moved[0] < term and moved[1] == len(TERM_TEXT)
    assert scored.node.metadata[MATCH_SPANS_KEY][0] == malformed


@pytest.mark.parametrize("per_window", [True, False], ids=["span-per-window", "one-span-across-the-join"])
def test_an_anchored_span_starts_in_the_window_holding_its_term(per_window: bool) -> None:
    """With a span per window, as RAG stamps them, an anchored span starts in the window holding its term.

    That is after the seam between two windows, never inside it. One span across the join, a shape RAG does not
    stamp, does the same while its term lies further past the seam than the left context reaches.

    Args:
        per_window (bool): Whether each window carries its own span, as a folded section's do, or one span
            reaches across the join.
    """
    first = "Opening filler words. " * 40
    text = f"{first}{WINDOW_JOIN}{TERM_TEXT}"
    window = len(first) + len(WINDOW_JOIN)
    spans = [[0, len(first)], [window, len(text)]] if per_window else [[0, len(text)]]
    scored = scored_node("s", text, **{MATCH_SPANS_KEY: spans})

    moved = _spans(anchor_terms(scored, ["Kontrollgruppe"]))

    term = text.index("Kontrollgruppe")
    assert moved[-1][1] == len(text)
    assert window <= term - TERM_CONTEXT_CHARS <= moved[-1][0] < term
    assert moved[:-1] == ([[0, len(first)]] if per_window else [])


def test_a_term_just_after_a_window_join_keeps_its_span_where_the_window_starts() -> None:
    """The 80 characters of left context would reach back over the join; a span never starts before its own start."""
    first = "Opening filler words. " * 40
    second = f"Short lead. Zielwert appears here.{' Trailing filler.' * 20}"
    text = f"{first}{WINDOW_JOIN}{second}"
    window = len(first) + len(WINDOW_JOIN)
    scored = scored_node("s", text, **{MATCH_SPANS_KEY: [[0, len(first)], [window, len(text)]]})

    assert text.index("Zielwert") - TERM_CONTEXT_CHARS < len(first)
    assert _spans(anchor_terms(scored, ["Zielwert"])) == [[0, len(first)], [window, len(text)]]


#: An invented chunk whose telling sentence lies well past the 300 characters its floor shows of its opening.
DEEP_TEXT = f"{LEAD * 3}There was keine Kontrollgruppe in the design, which the authors acknowledge. {LEAD}"


def test_an_anchored_passage_shown_at_its_floor_shows_the_term() -> None:
    """The least text the model is shown of a hit is the term in its context, not the chunk's first sentences."""
    scored = scored_node("s", DEEP_TEXT, filename="trial.pdf", **{MATCH_SPANS_KEY: [[0, len(DEEP_TEXT)]]})
    anchored = anchor_terms(scored, ["Kontrollgruppe"])
    unmoved, moved = EvidenceLedger(cap=5), EvidenceLedger(cap=5)
    unmoved.add([scored])
    moved.add([anchored])

    # Moving a match changes what the floor shows, never what it costs.
    assert text_floor(DEEP_TEXT, [(lo, hi) for lo, hi in _spans(scored)]) == text_floor(
        DEEP_TEXT, [(lo, hi) for lo, hi in _spans(anchored)]
    )
    assert "Kontrollgruppe" not in _shown_at_floor(unmoved)
    assert "keine Kontrollgruppe" in _shown_at_floor(moved)


def test_an_anchored_reemission_of_a_held_section_extends_it() -> None:
    """A section the turn already holds gains the term's region as a match of its own."""
    held = scored_node("s", TERM_TEXT, **{MATCH_SPANS_KEY: [[0, len(TERM_TEXT)]]})
    ledger = EvidenceLedger(cap=5)
    ledger.add([held])

    added = ledger.add([anchor_terms(held, ["Kontrollgruppe"])])

    assert [number for number, _ in added] == [1]
    entry = ledger.entries()[0]
    assert len(entry.node.metadata[MATCH_SPANS_KEY]) == 2
    assert entry.node.get_content(metadata_mode=MetadataMode.NONE) == TERM_TEXT


def test_an_anchored_copy_of_a_chunk_held_unstamped_merges_and_shows_the_term() -> None:
    """A chunk shown from its opening, found again by a check, is extended with the term's region, not turned away."""
    held = scored_node("u", UNSTAMPED_TEXT, filename="trial.pdf")
    ledger = EvidenceLedger(cap=5)
    ledger.add([held])
    assert UNSTAMPED_FACT not in _shown_at_floor(ledger)

    anchored = anchor_terms(held, ["randomization"])

    assert anchored is not held
    assert ledger.add([anchored]) == [(1, anchored)]
    assert UNSTAMPED_FACT in _shown_at_floor(ledger)


@pytest.mark.parametrize(
    ("matches", "line"),
    [
        (None, "The number of matching chunks is unavailable."),
        (0, "No indexed chunk in the searched material contains all of these words."),
        (1, "1 indexed chunk in the searched material contains all of these words."),
        (2, "2 indexed chunks in the searched material contain all of these words."),
        (37, "37 indexed chunks in the searched material contain all of these words."),
    ],
    ids=["unknown", "zero", "one", "two", "many"],
)
def test_the_note_names_unknown_zero_one_and_many_counts_apart(matches: int | None, line: str) -> None:
    """A zero is an exact answer, and an unknown count must never read as one; the note is the count alone.

    Args:
        matches (int | None): The count a keyword call made.
        line (str): The note the model reads.
    """
    assert keyword_note(matches=matches, unchecked=0) == line


@pytest.mark.parametrize(
    ("unchecked", "line"),
    [
        (1, "1 chunk is not in the keyword index yet, so it was not checked and may still contain these words."),
        (2, "2 chunks are not in the keyword index yet, so they were not checked and may still contain these words."),
        (12, "12 chunks are not in the keyword index yet, so they were not checked and may still contain these words."),
    ],
    ids=["one", "two", "many"],
)
def test_the_note_says_what_the_index_does_not_cover(unchecked: int, line: str) -> None:
    """On a partial index a zero rules out only what was checked: the coverage line says the rest may hold the words.

    Args:
        unchecked (int): Chunks the keyword index does not cover.
        line (str): The coverage line the model reads.
    """
    assert keyword_note(matches=0, unchecked=unchecked).splitlines() == [
        "No indexed chunk in the searched material contains all of these words.",
        line,
    ]


def test_the_note_joins_its_count_line_and_its_coverage_line_with_a_newline() -> None:
    """The lines of one note are a count and a coverage line; a carriage return in the joiner would pass splitlines."""
    assert keyword_note(matches=0, unchecked=2) == (
        "No indexed chunk in the searched material contains all of these words.\n"
        "2 chunks are not in the keyword index yet, so they were not checked and may still contain these words."
    )


def test_the_note_has_a_coverage_line_even_when_the_count_failed() -> None:
    """An unknown count does not hide that part of the collection was never checked."""
    assert keyword_note(matches=None, unchecked=3).splitlines() == [
        "The number of matching chunks is unavailable.",
        "3 chunks are not in the keyword index yet, so they were not checked and may still contain these words.",
    ]


def test_the_note_has_no_coverage_line_when_the_index_covers_every_chunk() -> None:
    """A complete index has nothing to add to the count."""
    assert keyword_note(matches=5, unchecked=0).splitlines() == [
        "5 indexed chunks in the searched material contain all of these words."
    ]


@pytest.mark.parametrize(
    ("matches", "line"),
    [
        (None, "The number of matching chunks is unavailable."),
        (0, "No indexed chunk in the searched material contains all of these words."),
        (1, "1 indexed chunk in the searched material contains all of these words."),
        (37, "37 indexed chunks in the searched material contain all of these words."),
    ],
    ids=["unknown", "zero", "one", "many"],
)
def test_the_note_says_a_check_ran_while_its_collection_was_written(matches: int | None, line: str) -> None:
    """A count made while the collection was written covers only what was stored, so even a zero is hedged.

    Args:
        matches (int | None): The count a keyword call made.
        line (str): The count line the model reads first.
    """
    assert keyword_note(matches=matches, unchecked=0, writing=True).splitlines() == [line, WRITING_LINE]


def test_the_writing_line_comes_between_the_count_and_the_coverage_line() -> None:
    """The writing line is the second of the note, ahead of the coverage line, so the overflow retry's head keeps it."""
    assert keyword_note(matches=0, unchecked=2, writing=True) == (
        "No indexed chunk in the searched material contains all of these words.\n"
        "The collection was being written during this check, so it covers only the chunks stored so far; "
        "chunks still arriving were not checked and may still contain these words.\n"
        "2 chunks are not in the keyword index yet, so they were not checked and may still contain these words."
    )


def test_the_checks_list_every_keyword_step_with_a_known_count() -> None:
    """Other steps and a failed count say nothing about the collection; a zero does, and is listed."""
    actions = [
        {"action": "seed", "detail": "", "found": 3},
        {"action": "search", "detail": "Hafenfest", "found": 1},
        *KEYWORD_STEPS,
        {"action": "keyword", "detail": "Kai", "found": 0, "field": "text", "matches": "4"},
        {"action": "note", "detail": "Ich prüfe noch die Regatta.", "found": 0},
    ]

    assert render_keyword_checks(actions, max_chars=KEYWORD_CHECKS_CHARS).splitlines() == [
        '- "Hafenfest" (author): 3; 2 chunks not checked',
        '- "Regatta" (text): 0',
    ]


def test_a_single_unchecked_chunk_reads_in_the_singular() -> None:
    """The validator reads the line as prose, so it should read right."""
    actions = [{"action": "keyword", "detail": "Regatta", "found": 0, "field": "text", "matches": 0, "unchecked": 1}]

    assert (
        render_keyword_checks(actions, max_chars=KEYWORD_CHECKS_CHARS) == '- "Regatta" (text): 0; 1 chunk not checked'
    )


def test_a_check_made_while_its_collection_was_written_is_marked_for_the_validator() -> None:
    """The mark ends a check's line, after any coverage suffix, and only that check's: the count is a count so far."""
    actions = [
        {**KEYWORD_STEPS[1], "writing": True},
        {**KEYWORD_STEPS[0], "writing": True},
        {"action": "keyword", "detail": "Sturmwarnung", "found": 0, "field": "text", "matches": 5},
    ]

    assert render_keyword_checks(actions, max_chars=KEYWORD_CHECKS_CHARS).splitlines() == [
        '- "Regatta" (text): 0; collection being written',
        '- "Hafenfest" (author): 3; 2 chunks not checked; collection being written',
        '- "Sturmwarnung" (text): 5',
    ]


def test_the_mark_counts_toward_the_room_of_the_checks() -> None:
    """A marked line is measured with its mark, so a block never outgrows its room and no line is cut in half."""
    line = '- "Regatta" (text): 0; collection being written'
    actions = [{**KEYWORD_STEPS[1], "writing": True}]

    assert render_keyword_checks(actions, max_chars=len(line)) == line
    assert render_keyword_checks(actions, max_chars=len(line) - 1) == ""


@pytest.mark.parametrize("writing", [False, None, 1, "true"], ids=["false", "none", "one", "string"])
def test_only_a_true_writing_marks_a_check(writing: Any) -> None:
    """A stored step is data: only the boolean the tool writes marks a check, not a value that merely reads as true.

    Args:
        writing (Any): A ``writing`` value other than ``True``.
    """
    actions = [{**KEYWORD_STEPS[1], "writing": writing}]

    assert render_keyword_checks(actions, max_chars=KEYWORD_CHECKS_CHARS) == '- "Regatta" (text): 0'


@pytest.mark.parametrize(
    "actions",
    [
        [],
        [{"action": "search", "detail": "Hafenfest", "found": 1}],
        [KEYWORD_STEPS[2]],
        [{**KEYWORD_STEPS[2], "writing": True}],
    ],
    ids=["no-steps", "search-only", "failed-count", "failed-count-written"],
)
def test_a_turn_without_a_known_count_has_no_checks(actions: list[dict[str, Any]]) -> None:
    """The block is empty, so the validator's prompt gains nothing.

    Args:
        actions (list[dict[str, Any]]): The turn's trail.
    """
    assert render_keyword_checks(actions, max_chars=KEYWORD_CHECKS_CHARS) == ""


def test_the_words_a_model_wrote_are_defused_before_the_validator_reads_them() -> None:
    """Citation numbers, the sentinel and double quotes cannot pass through, nor more than the longest input."""
    actions = [
        {
            "action": "keyword",
            "detail": 'siehe  [3] und [1, 2] die "[[ASSESSMENT]]"\nZeile',
            "field": "text",
            "found": 0,
            "matches": 0,
        },
        {"action": "keyword", "detail": "w" * (MAX_KEYWORD_CHARS + 20), "field": "text", "found": 0, "matches": 2},
    ]

    first, second = render_keyword_checks(actions, max_chars=KEYWORD_CHECKS_CHARS).splitlines()

    assert first == "- \"siehe und die '(assessment)' Zeile\" (text): 0"
    assert second == f'- "{"w" * MAX_KEYWORD_CHARS}" (text): 2'


def test_the_checks_keep_whole_lines_within_their_room() -> None:
    """A check is never cut in half; those after the first that does not fit are left out, in trail order."""
    actions = [
        {"action": "keyword", "detail": f"Wort{i}", "field": "text", "found": 0, "matches": i} for i in range(10)
    ]

    block = render_keyword_checks(actions, max_chars=60)

    assert block.splitlines() == ['- "Wort0" (text): 0', '- "Wort1" (text): 1', '- "Wort2" (text): 2']
    assert len(block) <= 60 < len(render_keyword_checks(actions[:4], max_chars=1000))


@pytest.mark.parametrize(
    ("first", "second", "same"),
    [
        (("Kontrollgruppe", "text"), ("kontrollgruppe.", "text"), True),
        (("control group", "text"), ("Group CONTROL", "text"), True),
        (("Straße", "text"), ("Strasse", "text"), False),
        (("post_ab.cd", "uuid"), ("post.ab_cd", "uuid"), False),
        (("POST_AB", "uuid"), ("post_ab", "uuid"), False),
        (("0123456789abcdef0123456789abcdef", "uuid"), ("01234567-89ab-cdef-0123-456789abcdef", "uuid"), True),
        (("Wolfgang Krieger", "author"), ("krieger WOLFGANG", "author"), True),
        (("Krieger", "author"), ("krieger", "author"), False),
        (("Krieger", "text"), ("Krieger", "author"), False),
    ],
)
def test_a_check_is_keyed_by_what_it_matches(first: tuple[str, str], second: tuple[str, str], same: bool) -> None:
    """Text matches the index's tokens, so case, order and punctuation make no new check; an id matches as written.

    A uuid pasted with and without its dashes is tried in both forms either way,
    so it is one check, while two ids that only tokenize alike are two.

    Args:
        first (tuple[str, str]): The words and field of one check.
        second (tuple[str, str]): The words and field of another.
        same (bool): Whether the two are one check.
    """
    assert (check_key(*first) == check_key(*second)) is same


# --- The keyword tool -----------------------------------------------------------------------------

#: The fields a collection of social postings offers, the text first.
ALL_FIELDS = ("text", "author", "network", "uuid")
#: The most words the description may take: every round of every turn is sent all of them.
MAX_DESCRIPTION_WORDS = 45
WEATHER = "Für Sonntag meldet der Wetterdienst bedeckten Himmel, Wind aus West und kühle Temperaturen am Hafen."


def _weather() -> NodeWithScore:
    """Build the invented weather passage.

    Returns:
        NodeWithScore: The passage.
    """
    return scored_node("weather", WEATHER, filename="wetterbericht_hafen.pdf", page=2)


def _result(
    nodes: Sequence[NodeWithScore] = (),
    *,
    matches: int | None,
    unchecked: int = 0,
    keywords: Sequence[str] = ("hafenfest",),
    writing: bool = False,
) -> KeywordResult:
    """Build what a keyword match found.

    Args:
        nodes (Sequence[NodeWithScore]): The best matching passages.
        matches (int | None): The exact count; ``None`` for a count that failed.
        unchecked (int): Chunks the keyword index does not cover.
        keywords (Sequence[str]): The words the match used.
        writing (bool): Whether the collection was being written while the check ran.

    Returns:
        KeywordResult: The result.
    """
    return KeywordResult(
        nodes=list(nodes), matches=matches, unchecked=unchecked, keywords=tuple(keywords), writing=writing
    )


class _Matcher:
    """A keyword match double: refuses what it is told to, answers each run with its next result, records both."""

    def __init__(
        self,
        *results: KeywordResult,
        fields: tuple[str, ...] = ("text",),
        refused: dict[str, str] | None = None,
    ) -> None:
        """Describe the match.

        Args:
            *results (KeywordResult): What the runs return in turn, the last one for every run after it; a
                count of zero with no passage when none is given.
            fields (tuple[str, ...]): The fields it offers.
            refused (dict[str, str] | None): By words, the reason a check gives for them; it accepts the rest.
        """
        self._results = list(results) or [_result(matches=0)]
        self._refused = refused or {}
        self.fields = fields
        self.checks: list[tuple[str, str]] = []
        self.runs: list[tuple[str, str]] = []

    @property
    def tool(self) -> KeywordTool:
        """The tool over this match.

        Returns:
            KeywordTool: A new tool.
        """
        return KeywordTool(KeywordMatcher(fields=self.fields, check=self._check, run=self._run))

    def _check(self, words: str, field: str) -> str | None:
        """Say why a check cannot run, and note that it was asked.

        Args:
            words (str): The words.
            field (str): The field.

        Returns:
            str | None: The reason given for these words, or ``None``.
        """
        self.checks.append((words, field))
        return self._refused.get(words)

    def _run(self, words: str, field: str) -> KeywordResult:
        """Answer a run with the next result, and note that it was asked.

        Args:
            words (str): The words.
            field (str): The field.

        Returns:
            KeywordResult: The next result, or the last one once the others are used.
        """
        self.runs.append((words, field))
        return self._results.pop(0) if len(self._results) > 1 else self._results[0]


def _ask(tool: KeywordTool, ctx: ToolContext, **arguments: Any) -> ToolReply:
    """Answer one call without the registry: check it, run its match and answer it.

    Args:
        tool (KeywordTool): The tool.
        ctx (ToolContext): The turn's context.
        **arguments (Any): The call's arguments.

    Returns:
        ToolReply: The tool's answer.
    """
    work = tool.prepare(arguments, ctx)
    assert isinstance(work, ToolWork) and work.fetch is not None
    return work.answer(work.fetch(), ctx)


def test_the_tool_is_named_keyword_and_describes_itself_briefly() -> None:
    """The description is the module's, and short: the schemas of every tool are sent in every round."""
    tool = _Matcher().tool

    assert (tool.name, tool.description) == ("keyword", KEYWORD_DESCRIPTION)
    assert len(KEYWORD_DESCRIPTION.split()) <= MAX_DESCRIPTION_WORDS


def test_a_tool_offering_only_the_text_takes_words_and_nothing_else() -> None:
    """The words come with the limits the index puts on them, built from the index's own constant."""
    assert _Matcher().tool.parameters == {
        "type": "object",
        "properties": {
            "words": {
                "type": "string",
                "description": (
                    "One to three words every match must contain; no single letter or digit, alone or where dots "
                    f"or signs split a word (2.5, U.S.), nor words over {SEARCH_MAX_TOKEN_LEN} characters."
                ),
            }
        },
        "required": ["words"],
    }


def test_a_tool_offering_more_fields_takes_one_of_exactly_those_with_their_notes() -> None:
    """The enum is the offered fields in order, defaulting to the text, and says what each one matches."""
    schema = _Matcher(fields=ALL_FIELDS).tool.parameters

    assert list(schema["properties"]) == ["words", "field"]
    assert schema["required"] == ["words"]
    assert schema["properties"]["field"] == {
        "type": "string",
        "enum": ["text", "author", "network", "uuid"],
        "default": "text",
        "description": (
            "text: the passage text (the default); author: who wrote a posting; network: the platform; "
            "uuid: one posting's id."
        ),
    }


def test_the_field_notes_cover_only_the_fields_offered() -> None:
    """A field the collection cannot match is not described, so the model is not invited to ask for it."""
    field = _Matcher(fields=("text", "uuid")).tool.parameters["properties"]["field"]

    assert field["enum"] == ["text", "uuid"]
    assert field["description"] == "text: the passage text (the default); uuid: one posting's id."


def test_every_field_the_search_knows_has_a_note() -> None:
    """A field added to the search without a note would be offered with no word on what it matches."""
    assert set(SEARCH_FIELDS) <= set(KEYWORD_FIELD_NOTES)


def _accepts(words: str, field: str) -> str | None:
    """Accept every call.

    Args:
        words (str): The words.
        field (str): The field.

    Returns:
        str | None: No reason to refuse.
    """
    return None


def _finds_nothing(words: str, field: str) -> KeywordResult:
    """Count nothing.

    Args:
        words (str): The words.
        field (str): The field.

    Returns:
        KeywordResult: A count of zero.
    """
    return _result(matches=0)


@pytest.mark.parametrize("fields", [("text",), ("text", "author"), ALL_FIELDS], ids=["text", "two", "all"])
def test_a_matcher_offering_the_text_first_and_fields_the_search_knows_is_accepted(fields: tuple[str, ...]) -> None:
    """The text leads, as the schema's default and the tool's refusals assume; the rest are the search's own names.

    Args:
        fields (tuple[str, ...]): The fields offered.
    """
    assert KeywordMatcher(fields=fields, check=_accepts, run=_finds_nothing).fields == fields


@pytest.mark.parametrize(
    ("fields", "complaint"),
    [
        ((), "text first"),
        (("author",), "text first"),
        (("author", "text"), "text first"),
        (("text", "gibtsnicht"), "unknown: 'gibtsnicht'"),
        (("text", "author", "gibtsnicht", "nirgends"), "unknown: 'gibtsnicht', 'nirgends'"),
    ],
    ids=["no-fields", "text-missing", "text-not-first", "unknown-field", "unknown-fields"],
)
def test_a_matcher_must_offer_the_text_first_and_only_fields_the_search_knows(
    fields: tuple[str, ...], complaint: str
) -> None:
    """A matcher that breaks the contract is refused when it is built, not when a model's call reaches the tool.

    Args:
        fields (tuple[str, ...]): The fields offered.
        complaint (str): What the error says is wrong.
    """
    with pytest.raises(ValueError, match=complaint):
        KeywordMatcher(fields=fields, check=_accepts, run=_finds_nothing)


#: Calls the tool refuses: the arguments, the reason the match's check gives, the message, the checks asked, the kind.
REFUSALS = [
    pytest.param({}, None, "call keyword with the words to match.", [], "missing_argument", id="no-words"),
    pytest.param(
        {"words": " \n "}, None, "call keyword with the words to match.", [], "missing_argument", id="blank-words"
    ),
    pytest.param(
        {"words": "w" * (MAX_KEYWORD_CHARS + 1)},
        None,
        f"keep the words within {MAX_KEYWORD_CHARS} characters; use the most telling ones.",
        [],
        "too_long",
        id="too-long",
    ),
    pytest.param(
        {"words": "Hafenfest", "field": "network"},
        None,
        "field must be one of: text, author, uuid.",
        [],
        "bad_field",
        id="field-not-offered",
    ),
    pytest.param(
        {"words": "a b"},
        NO_USABLE_WORD,
        "give at least one word of two or more characters.",
        [("a b", "text")],
        "no_usable_word",
        id="no-usable-word",
    ),
    pytest.param(
        {"words": "abc def", "field": "uuid"},
        ONE_VALUE_ONLY,
        "uuid matches one exact value; give a single one.",
        [("abc def", "uuid")],
        "one_value_only",
        id="one-value-only",
    ),
    pytest.param(
        {"words": "vitamin D"},
        UNINDEXABLE_WORD,
        "the index cannot match a single letter or digit, alone, such as the D in vitamin D, or where dots or signs "
        "split a word, such as 2.5, U.S. or p<0.05, nor a word with no letter or digit at all; check the other words "
        "only, or use search.",
        [("vitamin D", "text")],
        "unindexable_word",
        id="unindexable-word",
    ),
    pytest.param(
        {"words": "x" * (SEARCH_MAX_TOKEN_LEN + 1)},
        LONG_WORD,
        f"the index reads a word only up to its first {SEARCH_MAX_TOKEN_LEN} characters, so a longer one cannot be "
        "checked exactly; check a shorter beginning of it, which also matches the longer words it begins.",
        [("x" * (SEARCH_MAX_TOKEN_LEN + 1), "text")],
        "long_word",
        id="long-word",
    ),
    pytest.param(
        {"words": "w" * (MAX_KEYWORD_CHARS + 1), "field": "network"},
        None,
        f"keep the words within {MAX_KEYWORD_CHARS} characters; use the most telling ones.",
        [],
        "too_long",
        id="too-long-before-field",
    ),
    pytest.param(
        {"field": "network"},
        None,
        "call keyword with the words to match.",
        [],
        "missing_argument",
        id="no-words-before-field",
    ),
    pytest.param(
        {"words": "a b", "field": "network"},
        NO_USABLE_WORD,
        "field must be one of: text, author, uuid.",
        [],
        "bad_field",
        id="field-before-check",
    ),
]


@pytest.mark.parametrize(("arguments", "problem", "message", "checks", "kind"), REFUSALS)
def test_a_call_that_cannot_run_is_refused_with_its_reason_and_fetches_nothing(
    arguments: dict[str, Any], problem: str | None, message: str, checks: list[tuple[str, str]], kind: str
) -> None:
    """Each reason has its own message and kind, found in a fixed order, and the match is never run for a refused call.

    Args:
        arguments (dict[str, Any]): The call's arguments.
        problem (str | None): What the match's check says about every words it is asked.
        message (str): The message the reply carries after ``Error: ``.
        checks (list[tuple[str, str]]): The checks the refusal needed, in order.
        kind (str): The kind token the reply carries.
    """
    words = " ".join(str(arguments.get("words", "")).split())
    double = _Matcher(fields=("text", "author", "uuid"), refused={} if problem is None else {words: problem})

    reply = double.tool.prepare(arguments, tool_context())

    assert type(reply) is ToolReply
    assert (reply.text, reply.error, reply.action, reply.found) == (f"Error: {message}", True, "keyword", 0)
    assert reply.kind == kind
    assert (reply.informative, reply.trail) == (False, None)
    assert reply.detail == trail_detail(words)
    assert (double.checks, double.runs) == (checks, [])


@pytest.mark.parametrize(
    ("arguments", "asked"),
    [
        ({"words": "Hafenfest"}, ("Hafenfest", "text")),
        ({"words": "  Hafenfest \n Sturmwarnung\t"}, ("Hafenfest Sturmwarnung", "text")),
        ({"words": "Hafenfest", "field": None}, ("Hafenfest", "text")),
        ({"words": "Hafenfest", "field": ""}, ("Hafenfest", "text")),
        ({"words": "Wolfgang Krieger", "field": "author"}, ("Wolfgang Krieger", "author")),
        ({"words": 2024}, ("2024", "text")),
    ],
    ids=["plain", "whitespace", "null-field", "empty-field", "author", "number"],
)
def test_a_call_checks_then_fetches_its_words_on_one_line_in_the_field_it_names(
    arguments: dict[str, Any], asked: tuple[str, str]
) -> None:
    """The check sees what the match will run, and the match runs only when the fetch does, off the loop thread.

    Args:
        arguments (dict[str, Any]): The call's arguments.
        asked (tuple[str, str]): The words and field the check and the match are given.
    """
    double = _Matcher(fields=("text", "author"))

    work = double.tool.prepare(arguments, tool_context())

    assert isinstance(work, ToolWork) and work.fetch is not None
    assert (work.detail, double.checks, double.runs) == (asked[0], [asked], [])
    work.fetch()
    assert double.runs == [asked]


def test_words_of_exactly_the_longest_length_are_accepted() -> None:
    """The limit is inclusive: words of exactly that length run, and one character more is refused."""
    double = _Matcher()

    work = double.tool.prepare({"words": "w" * MAX_KEYWORD_CHARS}, tool_context())

    assert isinstance(work, ToolWork)


def test_the_repeat_key_is_the_field_the_sorted_tokens_and_the_sorted_values_as_json() -> None:
    """A key is one JSON list, whatever the words' order: the field, the tokens, then the exact values."""
    tool = _Matcher(fields=("text", "uuid")).tool

    words = tool.recall_key({"words": "zebra Mango apfel Kirsche birne Dattel"})
    uuid = tool.recall_key({"words": "0123456789abcdef0123456789abcdef", "field": "uuid"})

    assert words == '["text", ["apfel", "birne", "dattel", "kirsche", "mango", "zebra"], []]'
    assert uuid == '["uuid", [], ["01234567-89ab-cdef-0123-456789abcdef", "0123456789abcdef0123456789abcdef"]]'


def _backwards(words: str, field: str) -> tuple[str, tuple[str, ...], tuple[Any, ...]]:
    """Make a check's key as the search layer does, but list its tokens and values in reverse order.

    Equal sets can iterate in different orders. This stands in for the worst one, so only a key that sorts
    what it lists comes out the same.

    Args:
        words (str): The words of the check.
        field (str): A field the check may use.

    Returns:
        tuple[str, tuple[str, ...], tuple[Any, ...]]: The field, then the tokens and the values, each reversed.
    """
    name, tokens, values = check_key(words, field)
    return name, tuple(sorted(tokens, reverse=True)), tuple(sorted(values, key=str, reverse=True))


def test_the_repeat_key_sorts_the_tokens_and_the_values_whatever_order_they_are_listed_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calls that match the same thing are one call, so the key cannot depend on how a set happens to iterate.

    Args:
        monkeypatch (pytest.MonkeyPatch): Makes the check list its tokens and values in reverse order.
    """
    monkeypatch.setattr(keyword_module, "check_key", _backwards)
    tool = _Matcher(fields=("text", "uuid")).tool

    words = tool.recall_key({"words": "zebra mango apfel"})
    uuid = tool.recall_key({"words": "0123456789abcdef0123456789abcdef", "field": "uuid"})

    assert words == '["text", ["apfel", "mango", "zebra"], []]'
    assert uuid == '["uuid", [], ["01234567-89ab-cdef-0123-456789abcdef", "0123456789abcdef0123456789abcdef"]]'


@pytest.mark.parametrize(
    ("field", "anchored"),
    [("text", True), ("author", False), ("network", False), ("uuid", False)],
)
def test_only_the_text_hits_are_anchored_on_the_words(field: str, anchored: bool) -> None:
    """A text hit is moved to the words it was found by; the other fields match whole values, not words in text.

    Args:
        field (str): The field the call matched.
        anchored (bool): Whether the hit's match is moved to the term.
    """
    hit = scored_node("trial", DEEP_TEXT, filename="trial.pdf", **{MATCH_SPANS_KEY: [[0, len(DEEP_TEXT)]]})
    double = _Matcher(_result([hit], matches=1, keywords=("kontrollgruppe",)), fields=ALL_FIELDS)
    ctx = tool_context()

    _ask(double.tool, ctx, words="Kontrollgruppe", field=field)

    [[start, end]] = _spans(ctx.ledger.entries()[0])
    assert end == len(DEEP_TEXT)
    assert (start > 0) is anchored


def test_the_count_note_leads_the_passages_it_counts() -> None:
    """The note is the reply's first paragraph, so the overflow retry, which keeps each reply's head, keeps it."""
    double = _Matcher(_result([programme_passage()], matches=3, unchecked=2))
    note = keyword_note(matches=3, unchecked=2)

    reply = _ask(double.tool, tool_context(), words="Hafenfest")

    assert shown_numbers(reply.text) == [1]
    assert reply.text.startswith(f"{note}\n\ncitation_index: 1\n")


def test_a_count_that_shows_no_new_passage_says_so_after_the_count() -> None:
    """A zero is an answer, and so is a count of passages the turn holds: the reply names no passage and says why."""
    reply = _ask(_Matcher(_result(matches=0)).tool, tool_context(), words="Hafenfest")

    assert reply.text == f"{keyword_note(matches=0, unchecked=0)}\n\n{NO_NEW_PASSAGES}"
    assert (reply.found, reply.error) == (0, False)


@pytest.mark.parametrize("reasoning", [False, True], ids=["plain", "reasoning"])
def test_a_count_leading_a_full_reply_keeps_it_within_result_chars(reasoning: bool) -> None:
    """The note's room is reserved before any passage takes it, so a full reply never outgrows its cap.

    Args:
        reasoning (bool): Whether the model thinks, which shortens replies.
    """
    found = [scored_node(f"lang-{number}", DEEP_TEXT, filename=f"studie_{number}.pdf") for number in range(1, 9)]
    double = _Matcher(_result(found, matches=40, unchecked=12))
    ctx = tool_context(reasoning=reasoning)

    reply = _ask(double.tool, ctx, words="Kontrollgruppe")

    assert reply.text.startswith(keyword_note(matches=40, unchecked=12))
    assert "did not fit in this reply" in reply.text
    assert len(reply.text) <= ctx.budget.result_chars


def test_the_overflow_retry_keeps_the_count_of_a_long_keyword_reply() -> None:
    """Cut to its head on an overflow, a reply still says how many chunks matched; at its tail the count was lost."""
    found = [scored_node(f"lang-{number}", DEEP_TEXT, filename=f"studie_{number}.pdf") for number in range(1, 4)]
    note = keyword_note(matches=17, unchecked=2)
    reply = _ask(_Matcher(_result(found, matches=17, unchecked=2)).tool, tool_context(), words="Kontrollgruppe")
    trailing = passage_reply(found, tool_context(), action="keyword", detail="Kontrollgruppe", note=note)

    def _cut(text: str) -> str:
        """Cut one tool reply as the overflow retry does.

        Args:
            text (str): The reply.

        Returns:
            str: What the retry keeps of it.
        """
        shortened = _shortened([ChatMessage(role=MessageRole.TOOL, content=text)])
        assert shortened is not None
        return shortened[0].content or ""

    assert len(reply.text) > 2 * len(note)
    assert note in _cut(reply.text)
    assert note not in _cut(trailing.text)


def test_the_overflow_cut_of_a_zero_reply_keeps_that_its_collection_was_written() -> None:
    """Cut to its head, a zero with unchecked chunks still says its collection was written; the coverage line goes."""
    double = _Matcher(_result(matches=0, unchecked=12, writing=True))
    reply = _ask(double.tool, tool_context(), words="Hafenfest")

    shortened = _shortened([ChatMessage(role=MessageRole.TOOL, content=reply.text)])

    assert shortened is not None
    cut = shortened[0].content or ""
    assert "The collection was being written" in cut
    assert "12 chunks are not in the keyword index" not in cut


def test_found_counts_the_passages_admitted_not_the_chunks_matched() -> None:
    """Forty chunks match, two are new to the turn, and the same two found again are admitted no more."""
    both = [programme_passage(), _weather()]
    double = _Matcher(_result(both, matches=40), _result(both, matches=40))
    tool, ctx = double.tool, tool_context()

    first = _ask(tool, ctx, words="Hafenfest")
    again = _ask(tool, ctx, words="Wetterdienst")

    assert (first.found, again.found) == (2, 0)
    assert "Also matched, shown before: [1], [2]." in again.text


@pytest.mark.parametrize(
    ("field", "matches", "unchecked", "trail"),
    [
        ("text", 3, 0, {"field": "text", "matches": 3}),
        ("text", 0, 0, {"field": "text", "matches": 0}),
        ("author", 3, 2, {"field": "author", "matches": 3, "unchecked": 2}),
        ("text", None, 0, {"field": "text", "matches": None}),
        ("text", None, 4, {"field": "text", "matches": None, "unchecked": 4}),
    ],
    ids=["count", "zero", "unchecked", "failed-count", "failed-count-unchecked"],
)
def test_the_trail_names_the_field_and_the_count_and_the_unchecked_only_when_there_are_some(
    field: str, matches: int | None, unchecked: int, trail: dict[str, Any]
) -> None:
    """A failed count keeps ``matches`` as ``None``, which says it failed; ``unchecked`` is left out at zero.

    Args:
        field (str): The field the call matched.
        matches (int | None): The count the match made.
        unchecked (int): Chunks the keyword index does not cover.
        trail (dict[str, Any]): The extra keys of the call's step, exactly.
    """
    double = _Matcher(_result([programme_passage()], matches=matches, unchecked=unchecked), fields=ALL_FIELDS)

    reply = _ask(double.tool, tool_context(), words="Hafenfest", field=field)

    assert reply.trail == trail
    assert json.loads(json.dumps(reply.trail)) == trail


@pytest.mark.parametrize(
    ("field", "matches", "unchecked", "writing", "trail"),
    [
        ("text", 3, 0, True, {"field": "text", "matches": 3, "writing": True}),
        ("text", 0, 0, True, {"field": "text", "matches": 0, "writing": True}),
        ("author", 3, 2, True, {"field": "author", "matches": 3, "unchecked": 2, "writing": True}),
        ("text", None, 0, True, {"field": "text", "matches": None, "writing": True}),
        ("text", None, 4, True, {"field": "text", "matches": None, "unchecked": 4, "writing": True}),
        ("author", 3, 2, False, {"field": "author", "matches": 3, "unchecked": 2}),
    ],
    ids=["count", "zero", "unchecked", "failed-count", "failed-count-unchecked", "not-written"],
)
def test_the_trail_says_the_collection_was_written_only_when_it_was(
    field: str, matches: int | None, unchecked: int, writing: bool, trail: dict[str, Any]
) -> None:
    """The step carries ``writing`` as a true flag, even beside a failed count, and has no such key otherwise.

    Args:
        field (str): The field the call matched.
        matches (int | None): The count the match made.
        unchecked (int): Chunks the keyword index does not cover.
        writing (bool): Whether the collection was being written while the check ran.
        trail (dict[str, Any]): The extra keys of the call's step, exactly.
    """
    double = _Matcher(
        _result([programme_passage()], matches=matches, unchecked=unchecked, writing=writing), fields=ALL_FIELDS
    )

    reply = _ask(double.tool, tool_context(), words="Hafenfest", field=field)

    assert reply.trail == trail
    assert json.loads(json.dumps(reply.trail)) == trail


@pytest.mark.parametrize("matches", [0, 1, 40])
def test_a_known_count_is_news_even_when_it_is_zero_and_the_call_is_remembered(matches: int) -> None:
    """The model learned something from the count, so the round is not idle, and the call can be recalled.

    Args:
        matches (int): The count the match made.
    """
    reply = _ask(_Matcher(_result(matches=matches)).tool, tool_context(), words="Hafenfest")

    assert type(reply) is PassageReply
    assert (reply.informative, reply.error) == (True, False)


def test_a_failed_count_is_a_plain_reply_that_is_no_news() -> None:
    """The number is unavailable, so nothing was learned about the collection; passages it did find still count."""
    passages = [programme_passage()]
    reply = _ask(_Matcher(_result(passages, matches=None, unchecked=1)).tool, tool_context(), words="Hafenfest")

    note = keyword_note(matches=None, unchecked=1)
    shown = passage_reply(passages, tool_context(), action="keyword", detail="Hafenfest", note=note, note_first=True)
    assert type(reply) is ToolReply
    assert (reply.text, reply.found, reply.action, reply.detail) == (shown.text, shown.found, "keyword", "Hafenfest")
    assert (reply.informative, reply.error) == (False, False)
    assert reply.trail == {"field": "text", "matches": None, "unchecked": 1}
    assert reply.text.startswith(note)
    assert reply.keep == note


def test_a_cleared_keyword_reply_keeps_its_count() -> None:
    """Cut down to make room, a keyword reply keeps its count ahead of the passages it named: no read can restore it."""
    reply = ToolRegistry([KeywordTool(keyword_match())]).call(
        tool_call("keyword", words="Hafenfest", field="author"), tool_context()
    )

    note = keyword_note(matches=3, unchecked=2)
    assert isinstance(reply, PassageReply)
    assert reply.keep == note
    assert cleared_text(reply).startswith(note)
    assert (
        cleared_text(reply) == f"{note}\n\nCleared to make room: this reply showed [1]; read(n) shows a passage again."
    )


def test_reordered_or_recased_words_on_the_text_are_answered_from_memory() -> None:
    """The same check asked again in other words is not run, admits nothing, and is no news."""
    double = _Matcher(_result([programme_passage()], matches=2, unchecked=1))
    registry = ToolRegistry([double.tool])
    ctx = tool_context()

    first = registry.call(tool_call("keyword", words="Hafenfest Sturmwarnung"), ctx)
    again = registry.call(tool_call("keyword", words="  STURMWARNUNG   hafenfest"), ctx)

    assert double.runs == [("Hafenfest Sturmwarnung", "text")]
    assert again.text == "Already called keyword with these arguments; see [1]."
    assert (first.found, first.informative) == (1, True)
    assert first.trail == {"field": "text", "matches": 2, "unchecked": 1}
    assert (again.found, again.informative, again.trail) == (0, False, first.trail)


def test_a_count_made_while_written_is_news_and_remembered_with_its_flag() -> None:
    """A zero made while the collection was written is news, and asked again it replays its step with the flag."""
    double = _Matcher(_result(matches=0, unchecked=1, writing=True))
    registry = ToolRegistry([double.tool])
    ctx = tool_context()

    first = registry.call(tool_call("keyword", words="Hafenfest Sturmwarnung"), ctx)
    again = registry.call(tool_call("keyword", words="  STURMWARNUNG   hafenfest"), ctx)

    assert double.runs == [("Hafenfest Sturmwarnung", "text")]
    assert (first.found, first.informative) == (0, True)
    assert first.trail == {"field": "text", "matches": 0, "unchecked": 1, "writing": True}
    assert again.text == "Already called keyword with these arguments; it showed no passages."
    assert (again.found, again.informative) == (0, False)
    assert again.trail == first.trail


def test_two_fields_are_two_calls_even_for_the_same_words() -> None:
    """The same words match different things in the text and in a network, so neither answers for the other."""
    double = _Matcher(_result([programme_passage()], matches=2), fields=("text", "network"))
    registry = ToolRegistry([double.tool])
    ctx = tool_context()

    registry.call(tool_call("keyword", words="Hafenfest"), ctx)
    other = registry.call(tool_call("keyword", words="Hafenfest", field="network"), ctx)
    again = registry.call(tool_call("keyword", words="hafenfest"), ctx)

    assert double.runs == [("Hafenfest", "text"), ("Hafenfest", "network")]
    assert other.trail == {"field": "network", "matches": 2}
    assert again.text == "Already called keyword with these arguments; see [1]."


@pytest.mark.parametrize(
    ("first", "second", "same"),
    [
        ({"words": "Hans Müller"}, {"words": "müller   HANS"}, True),
        ({"words": "Kontrollgruppe"}, {"words": "kontrollgruppe."}, True),
        ({"words": "Straße"}, {"words": "Strasse"}, False),
        ({"words": "Hafenfest"}, {"words": "Hafenfest", "field": "network"}, False),
        ({"words": "Krieger"}, {"words": "Krieger", "field": "author"}, False),
        ({"words": "Krieger", "field": "author"}, {"words": "krieger", "field": "author"}, False),
        ({"words": "Wolfgang Krieger", "field": "author"}, {"words": "krieger WOLFGANG", "field": "author"}, True),
        (
            {"words": "0123456789abcdef0123456789abcdef", "field": "uuid"},
            {"words": "01234567-89ab-cdef-0123-456789abcdef", "field": "uuid"},
            True,
        ),
        ({"words": "post_ab.cd", "field": "uuid"}, {"words": "post.ab_cd", "field": "uuid"}, False),
        ({"words": "POST_AB", "field": "uuid"}, {"words": "post_ab", "field": "uuid"}, False),
    ],
    ids=[
        "reordered",
        "punctuation",
        "umlaut-is-not-folded",
        "same-tokens-two-fields",
        "two-fields",
        "an-id-is-exact",
        "names-in-any-order",
        "uuid-dashes",
        "uuid-pieces",
        "uuid-case",
    ],
)
def test_calls_are_one_call_when_they_match_the_same_thing(
    first: dict[str, Any], second: dict[str, Any], same: bool
) -> None:
    """Words match by the tokens the index reads, an id by its exact value; the field is always part of the call.

    Args:
        first (dict[str, Any]): One call's arguments.
        second (dict[str, Any]): Another call's.
        same (bool): Whether the registry remembers them as one call.
    """
    registry = ToolRegistry([_Matcher(fields=ALL_FIELDS).tool])

    keys = [registry.prepare(tool_call("keyword", **arguments), tool_context()).key for arguments in (first, second)]

    assert (keys[0] == keys[1]) is same


def test_a_failed_count_is_never_remembered_so_the_same_call_runs_again() -> None:
    """A retry can succeed where the count failed; the call that then succeeds is the one that is remembered."""
    double = _Matcher(_result([programme_passage()], matches=None), _result([programme_passage()], matches=2))
    registry = ToolRegistry([double.tool])
    ctx = tool_context()

    failed = registry.call(tool_call("keyword", words="Hafenfest"), ctx)
    retried = registry.call(tool_call("keyword", words="Hafenfest"), ctx)
    again = registry.call(tool_call("keyword", words="hafenfest"), ctx)

    assert double.runs == [("Hafenfest", "text")] * 2
    assert (type(failed), failed.informative, failed.trail) == (ToolReply, False, {"field": "text", "matches": None})
    assert (type(retried), retried.informative, retried.trail) == (PassageReply, True, {"field": "text", "matches": 2})
    assert again.text == "Already called keyword with these arguments; see [1]."


def test_a_call_the_tool_refuses_is_never_answered_as_a_repeat_of_one_it_ran() -> None:
    """The index drops the lone D of ``vitamin D``, so it keys like ``vitamin``; its refusal must still be given."""
    double = _Matcher(_result([programme_passage()], matches=2), refused={"vitamin D": UNINDEXABLE_WORD})
    registry = ToolRegistry([double.tool])
    ctx = tool_context()
    assert check_key("vitamin D", "text") == check_key("vitamin", "text")

    ran = registry.call(tool_call("keyword", words="vitamin"), ctx)
    refused = registry.call(tool_call("keyword", words="vitamin D"), ctx)

    assert (ran.error, refused.error) == (False, True)
    assert refused.text.startswith("Error: the index cannot match a single letter or digit")
    assert double.runs == [("vitamin", "text")]


#: Arguments a model may send, and whether the tool names such a call by a repeat key of its own.
ODD_ARGUMENTS = [
    pytest.param({}, False, id="no-arguments"),
    pytest.param({"words": None}, False, id="null-words"),
    pytest.param({"words": ""}, False, id="empty-words"),
    pytest.param({"words": " \n "}, False, id="blank-words"),
    pytest.param({"words": 0}, False, id="zero-words"),
    pytest.param({"words": 2024}, True, id="number-words"),
    pytest.param({"words": ["Hafenfest", "Sturm"]}, True, id="list-words"),
    pytest.param({"words": {"w": "Hafenfest"}}, True, id="object-words"),
    pytest.param({"words": "Hafenfest", "field": None}, True, id="null-field"),
    pytest.param({"words": "Hafenfest", "field": "gibtsnicht"}, False, id="unknown-field"),
    pytest.param({"words": "Hafenfest", "field": "network"}, False, id="field-not-offered"),
    pytest.param({"words": "Hafenfest", "field": 7}, False, id="number-field"),
    pytest.param({"words": "Hafenfest", "field": ["text"]}, False, id="list-field"),
    pytest.param({"words": "Hafenfest", "field": {"text": 1}}, False, id="object-field"),
    pytest.param({"field": "author"}, False, id="field-without-words"),
    pytest.param({"words": "w" * (MAX_KEYWORD_CHARS + 1)}, False, id="over-long-words"),
    pytest.param(
        {"words": "\N{SUPERSCRIPT TWO}\N{SUPERSCRIPT TWO}", "field": "author"}, True, id="digits-that-are-no-decimals"
    ),
]


@pytest.mark.parametrize(
    "failure",
    [ValueError("invented value problem"), UnknownSearchFieldError("invented")],
    ids=["value-error", "unknown-field"],
)
def test_a_key_the_search_layer_refuses_to_make_is_no_key_rather_than_an_error(
    monkeypatch: pytest.MonkeyPatch, failure: ValueError
) -> None:
    """Asked before the arguments are checked, a repeat key the search layer cannot make is left to the default key.

    The call itself still runs: only the way it is remembered changes.

    Args:
        monkeypatch (pytest.MonkeyPatch): Makes the check's key fail.
        failure (ValueError): What it fails with.
    """

    def _fails(words: str, field: str) -> Any:
        """Fail as the search layer does for a value it cannot read.

        Args:
            words (str): The words of the check.
            field (str): The field.

        Raises:
            ValueError: Always.
        """
        raise failure

    monkeypatch.setattr(keyword_module, "check_key", _fails)
    tool = _Matcher(fields=("text", "author")).tool

    prepared = ToolRegistry([tool]).prepare(tool_call("keyword", words="Hafenfest"), tool_context())

    assert tool.recall_key({"words": "Hafenfest"}) is None
    assert prepared.key == ("keyword", '{"words": "hafenfest"}')
    assert prepared.work is not None


def test_a_failure_of_the_key_that_is_no_value_problem_is_not_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only a value the search layer cannot read is forgiven; any other fault in the key surfaces as a failed call.

    Args:
        monkeypatch (pytest.MonkeyPatch): Makes the check's key fail.
    """

    def _fails(words: str, field: str) -> Any:
        """Fail as a bug would.

        Args:
            words (str): The words of the check.
            field (str): The field.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("invented bug")

    monkeypatch.setattr(keyword_module, "check_key", _fails)

    with pytest.raises(RuntimeError, match="invented bug"):
        _Matcher().tool.recall_key({"words": "Hafenfest"})


@pytest.mark.parametrize(("arguments", "keyed"), ODD_ARGUMENTS)
def test_the_repeat_key_copes_with_any_arguments_a_model_may_send(arguments: dict[str, Any], keyed: bool) -> None:
    """The key is asked for before the arguments are checked, so a call the tool will refuse must not raise.

    A call it has no key for falls to the registry's default key, and the refusal is given as ever.

    Args:
        arguments (dict[str, Any]): The call's arguments.
        keyed (bool): Whether the tool names the call itself.
    """
    double = _Matcher(fields=("text", "author"))
    tool = double.tool
    registry = ToolRegistry([tool])
    call = ToolCall(id="call_0", name="keyword", arguments=json.dumps(arguments))

    key = tool.recall_key(arguments)
    prepared = registry.prepare(call, tool_context())

    assert isinstance(key, str) is keyed
    assert prepared.key is not None and prepared.key[0] == "keyword"
    assert (prepared.key[1] == key) is keyed
    # A call the tool can key is one it can run; one it cannot is refused at once.
    assert (prepared.work is not None) is keyed
