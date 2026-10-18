"""Unit tests for the keyword check's pure parts: anchoring, the repeat key and the count note. All invented."""

from __future__ import annotations

import pytest
from assessment_fakes import llm_view, scored_node
from llama_index.core.schema import MetadataMode, NodeWithScore

from docint.agents.assessment.ledger import EvidenceLedger, passage_costs, passage_parts, render_context, text_floor
from docint.agents.turn.keyword import TERM_CONTEXT_CHARS, anchor_terms, check_key, keyword_note
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
        (1, "1 chunk is not in the keyword index yet, so it was not checked."),
        (2, "2 chunks are not in the keyword index yet, so they were not checked."),
        (12, "12 chunks are not in the keyword index yet, so they were not checked."),
    ],
    ids=["one", "two", "many"],
)
def test_the_note_says_what_the_index_does_not_cover(unchecked: int, line: str) -> None:
    """On a partial index a zero is only as good as what was checked, so the count line is followed by the coverage.

    Args:
        unchecked (int): Chunks the keyword index does not cover.
        line (str): The coverage line the model reads.
    """
    assert keyword_note(matches=0, unchecked=unchecked).splitlines() == [
        "No indexed chunk in the searched material contains all of these words.",
        line,
    ]


def test_the_note_has_a_coverage_line_even_when_the_count_failed() -> None:
    """An unknown count does not hide that part of the collection was never checked."""
    assert keyword_note(matches=None, unchecked=3).splitlines() == [
        "The number of matching chunks is unavailable.",
        "3 chunks are not in the keyword index yet, so they were not checked.",
    ]


def test_the_note_has_no_coverage_line_when_the_index_covers_every_chunk() -> None:
    """A complete index has nothing to add to the count."""
    assert keyword_note(matches=5, unchecked=0).splitlines() == [
        "5 indexed chunks in the searched material contain all of these words."
    ]


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
