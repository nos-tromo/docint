"""Unit tests for the keyword tool's pure parts. Every passage and word here is invented."""

from __future__ import annotations

from assessment_fakes import scored_node
from llama_index.core.schema import MetadataMode

from docint.agents.assessment.keyword import (
    TERM_CONTEXT_CHARS,
    anchor_terms,
    keyword_reply,
    render_keyword_checks,
)
from docint.agents.assessment.ledger import EvidenceLedger, digest
from docint.utils.text_budget import MATCH_SPANS_KEY

LEAD = "The trial enrolled two hundred patients across four sites over two years. " * 3
TERM_TEXT = f"{LEAD}There was keine Kontrollgruppe in the design, which the authors acknowledge."


def _spans(scored: object) -> list[list[int]]:
    """Return a passage's stored match spans.

    Args:
        scored (object): The passage.

    Returns:
        list[list[int]]: Its ``MATCH_SPANS_KEY`` value.
    """
    return list(scored.node.metadata[MATCH_SPANS_KEY])  # type: ignore[attr-defined]


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


def test_a_term_outside_the_span_does_not_move_it() -> None:
    """Only an occurrence starting inside the span anchors it."""
    span = [0, len(LEAD)]
    scored = scored_node("s", TERM_TEXT, **{MATCH_SPANS_KEY: [span]})

    assert anchor_terms(scored, ["Kontrollgruppe"]) is scored


def test_a_passage_without_spans_is_returned_as_is() -> None:
    """An unstamped passage keeps its shape: stamping it would change how it merges."""
    scored = scored_node("s", TERM_TEXT)

    assert anchor_terms(scored, ["Kontrollgruppe"]) is scored


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


def test_the_digest_of_an_anchored_passage_shows_the_term() -> None:
    """What research reads is the term in its context, not the chunk's first sentence."""
    scored = scored_node("s", TERM_TEXT, filename="trial.pdf", **{MATCH_SPANS_KEY: [[0, len(TERM_TEXT)]]})

    assert "Kontrollgruppe" not in digest(1, scored)
    assert "keine Kontrollgruppe" in digest(1, anchor_terms(scored, ["Kontrollgruppe"]))


def test_an_anchored_reemission_of_a_held_section_extends_it() -> None:
    """A section research already holds gains the term's region as a match of its own."""
    held = scored_node("s", TERM_TEXT, **{MATCH_SPANS_KEY: [[0, len(TERM_TEXT)]]})
    ledger = EvidenceLedger(cap=5)
    ledger.add([held])

    added = ledger.add([anchor_terms(held, ["Kontrollgruppe"])])

    assert [number for number, _ in added] == [1]
    entry = ledger.entries()[0]
    assert len(entry.node.metadata[MATCH_SPANS_KEY]) == 2
    assert entry.node.get_content(metadata_mode=MetadataMode.NONE) == TERM_TEXT


def test_the_reply_states_the_count_then_the_new_passages() -> None:
    """The count leads, so research reads how much matched before what is new."""
    reply = keyword_reply(matches=37, unchecked=0, digests=["[4] (a.pdf) one", "[5] (b.pdf) two"])

    assert reply.splitlines() == [
        "37 indexed chunks in the searched material contain all of these words.",
        "[4] (a.pdf) one",
        "[5] (b.pdf) two",
    ]


def test_the_reply_names_zero_one_and_unknown_counts_apart() -> None:
    """A zero is an exact answer; an unknown count must never read as one."""
    assert keyword_reply(matches=0, unchecked=0, digests=[]).splitlines() == [
        "No indexed chunk in the searched material contains all of these words.",
        "No new passages.",
    ]
    assert keyword_reply(matches=1, unchecked=0, digests=[]).splitlines()[0] == (
        "1 indexed chunk in the searched material contains all of these words."
    )
    assert keyword_reply(matches=None, unchecked=0, digests=[]).splitlines()[0] == (
        "The number of matching chunks is unavailable."
    )


def test_the_reply_says_what_the_index_does_not_cover() -> None:
    """On a partial index a zero is only as good as what was checked."""
    reply = keyword_reply(matches=0, unchecked=12, digests=[])

    assert "12 chunks are not in the keyword index yet, so they were not checked." in reply.splitlines()


def test_the_checks_list_every_keyword_call_with_a_known_count() -> None:
    """Searches and checks whose count failed say nothing about the collection."""
    actions = [
        {"action": "seed", "detail": "", "found": 3},
        {"action": "search", "detail": "method", "found": 1},
        {"action": "keyword", "detail": "Kontrollgruppe", "field": "text", "found": 0, "matches": 0},
        {"action": "keyword", "detail": "Krieger", "field": "author", "found": 2, "matches": 5, "unchecked": 12},
        {"action": "keyword", "detail": "lost", "field": "text", "found": 0, "matches": None},
    ]

    assert render_keyword_checks(actions, max_chars=1000).splitlines() == [
        '- "Kontrollgruppe" (text): 0',
        '- "Krieger" (author): 5, unchecked: 12',
    ]


def test_no_checks_render_as_none() -> None:
    """The answer prompt always carries the block, so an empty one says so."""
    assert render_keyword_checks([{"action": "search", "detail": "x", "found": 1}], max_chars=1000) == "(none)"


def test_model_written_words_are_defused_before_the_answer_reads_them() -> None:
    """Research wrote the words, so the sentinel, citation numbers and quotes cannot pass through."""
    actions = [
        {
            "action": "keyword",
            "detail": 'see  [3] the "[[ASSESSMENT]]"\nline',
            "field": "text",
            "found": 0,
            "matches": 0,
        }
    ]

    block = render_keyword_checks(actions, max_chars=1000)

    assert "[[ASSESSMENT]]" not in block
    assert "[3]" not in block
    assert block.count('"') == 2
    assert "\n" not in block


def test_the_checks_keep_whole_lines_within_their_room() -> None:
    """A check is never cut in half; the ones that do not fit are left out."""
    actions = [
        {"action": "keyword", "detail": f"word{i}", "field": "text", "found": 0, "matches": i} for i in range(10)
    ]

    block = render_keyword_checks(actions, max_chars=60)

    assert len(block) <= 60
    assert all(line.startswith('- "word') for line in block.splitlines())
    assert block.splitlines()[0] == '- "word0" (text): 0'
