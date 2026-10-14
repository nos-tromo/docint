"""Evidence ledger, answer budget and packing. All passages are invented."""

from __future__ import annotations

import random
from typing import Any

import pytest
from assessment_fakes import llm_view, scored_node, social_post
from llama_index.core.schema import Document, MetadataMode, NodeWithScore

from docint.agents.assessment.ledger import (
    COMPACT_HEADER_CHARS,
    CONTEXT_ERROR_MARKERS,
    DIGEST_CHARS,
    DIGEST_LABEL_CHARS,
    MARKER_RESERVE_CHARS,
    MIN_TEXT_CHARS,
    WINDOW_JOIN,
    EvidenceLedger,
    Room,
    _allowances,
    _parts,
    answer_budget,
    digest,
    effective_ratio,
    evidence_fits,
    fit_ledger,
    is_context_overflow,
    render_context,
    text_floor,
)
from docint.agents.assessment.runner import DEFAULT_ASSESSMENT_ANSWER_PROMPT
from docint.agents.assessment.split import SENTINEL, compose_labelled_answer
from docint.core.rag import ParentContextPostprocessor
from docint.utils.embed_chunking import estimate_tokens
from docint.utils.prompt_loader import load_localized_prompt
from docint.utils.text_budget import (
    MATCH_SPANS_KEY,
    MIN_REGION_CHARS,
    allocate_shortest_first,
    distinct_spans,
    fit_with_marker,
)


def test_ledger_numbers_in_insertion_order_and_dedupes() -> None:
    """A passage found twice keeps its first number; numbers never skip."""
    ledger = EvidenceLedger(cap=10)
    first, second, third = scored_node("a", "A"), scored_node("b", "B"), scored_node("c", "C")

    assert ledger.add([first, second]) == [(1, first), (2, second)]
    assert ledger.add([second, third]) == [(3, third)]
    assert len(ledger) == 3


def test_ledger_stops_at_its_cap() -> None:
    """The cap bounds what the answer must fit; nothing past it is admitted."""
    ledger = EvidenceLedger(cap=2)

    added = ledger.add([scored_node("a", "A"), scored_node("b", "B"), scored_node("c", "C")])

    assert [number for number, _ in added] == [1, 2]
    assert ledger.full is True


def _window(text: str, *, score: float = 1.0) -> NodeWithScore:
    """Build one parent-context window of an invented section, as the postprocessor emits it.

    Every window of a section carries the section's own node id.

    Args:
        text (str): The windowed text.
        score (float): The hit's score.

    Returns:
        NodeWithScore: The window.
    """
    scored = scored_node(
        "section-1",
        text,
        filename="paper.pdf",
        page=2,
        parent_context_windowed=True,
        parent_full_chars=9000,
        window_chars=len(text),
    )
    scored.score = score
    return scored


def _text(scored: NodeWithScore) -> str:
    """Return a passage's text without metadata.

    Args:
        scored (NodeWithScore): The passage.

    Returns:
        str: Its text.
    """
    return scored.node.get_content(metadata_mode=MetadataMode.NONE)


def test_another_window_of_a_held_section_extends_its_entry() -> None:
    """Windows of one section share its id, so a later window is new evidence and must not be dropped."""
    ledger = EvidenceLedger(cap=5)
    opening, closing = _window("The study opens with a pilot."), _window("It closes with a limitations note.")
    ledger.add([opening])

    changed = ledger.add([closing])

    assert len(changed) == 1
    number, arriving = changed[0]
    assert (number, arriving) == (1, closing)
    assert len(ledger) == 1
    numbered = ledger.numbered()
    assert _text(numbered[0]) == f"The study opens with a pilot.{WINDOW_JOIN}It closes with a limitations note."
    assert numbered[0].node.metadata["citation_index"] == 1


def test_the_join_between_windows_cannot_read_as_a_citation_marker() -> None:
    """The model cites by square brackets, so the seam between two windows carries none."""
    assert "[" not in WINDOW_JOIN
    assert "]" not in WINDOW_JOIN


@pytest.mark.parametrize(
    ("held", "arriving"),
    [
        ("The study opens with a pilot.", "The study opens with a pilot."),
        ("The study opens with a pilot, then compares.", "opens with a pilot"),
    ],
    ids=["identical", "contained"],
)
def test_a_window_the_entry_already_shows_is_not_new(held: str, arriving: str) -> None:
    """Repeating or narrowing what an entry shows adds nothing, so it is neither returned nor counted.

    Args:
        held (str): The text the ledger holds.
        arriving (str): The text of the window that arrives.
    """
    ledger = _restricted([_window(held)], 100_000)
    first = ledger.entries()[0]

    assert ledger.add([_window(arriving)]) == []
    assert ledger.entries()[0] is first
    assert _text(ledger.numbered()[0]) == held
    # Nothing new is not something the room turned away.
    assert ledger.refusals == 0


def test_a_fuller_window_replaces_the_one_it_contains() -> None:
    """A whole section arriving after a slice of it supersedes the slice instead of repeating it."""
    ledger = EvidenceLedger(cap=5)
    ledger.add([_window("opens with a pilot")])
    whole = scored_node("section-1", "The study opens with a pilot, then compares.", filename="paper.pdf", page=2)

    changed = ledger.add([whole])

    assert changed == [(1, whole)]
    assert len(ledger) == 1
    held = ledger.numbered()[0]
    assert _text(held) == "The study opens with a pilot, then compares."
    assert "parent_context_windowed" not in held.node.metadata


def test_a_merge_leaves_the_cached_nodes_alone() -> None:
    """Passages come out of cached stores, so folding one in must never write to it."""
    ledger = EvidenceLedger(cap=5)
    opening, closing = _window("The study opens with a pilot."), _window("It closes with a limitations note.")
    ledger.add([opening])
    ledger.add([closing])

    assert _text(opening) == "The study opens with a pilot."
    assert _text(closing) == "It closes with a limitations note."
    assert opening.node.metadata["window_chars"] == len("The study opens with a pilot.")
    assert ledger.entries()[0].node is not opening.node


def test_a_merge_keeps_the_entry_number_the_higher_score_and_an_honest_window_size() -> None:
    """The entry stays where citations number it; its window marker follows its text, and its score its best hit."""
    ledger = EvidenceLedger(cap=5)
    ledger.add([scored_node("other", "Elsewhere."), _window("Opening.", score=0.4)])

    ledger.add([_window("Closing.", score=0.9)])
    ledger.add([_window("Middle.", score=0.1)])

    entry = ledger.entries()[1]
    assert entry.score == pytest.approx(0.9)
    assert entry.node.metadata["window_chars"] == len(_text(entry))
    assert entry.node.metadata["parent_full_chars"] == 9000
    assert [scored.node.metadata["citation_index"] for scored in ledger.numbered()] == [1, 2]
    assert _text(entry) == f"Opening.{WINDOW_JOIN}Closing.{WINDOW_JOIN}Middle."


def test_a_merge_never_fills_the_ledger() -> None:
    """The cap counts entries, and extending one adds none."""
    ledger = EvidenceLedger(cap=2)
    ledger.add([_window("Opening.")])

    ledger.add([_window("Closing.")])

    assert len(ledger) == 1
    assert ledger.full is False
    assert [number for number, _ in ledger.add([scored_node("other", "Elsewhere.")])] == [2]


def test_a_full_ledger_still_extends_what_it_holds_but_admits_nothing_new() -> None:
    """A section already cited keeps gaining windows at the cap; only new sections are refused."""
    ledger = EvidenceLedger(cap=2)
    ledger.add([scored_node("other", "Elsewhere."), _window("Opening.")])
    assert ledger.full is True

    changed = ledger.add([scored_node("late", "Too late."), _window("Closing.")])

    assert [number for number, _ in changed] == [2]
    assert len(ledger) == 2
    assert _text(ledger.entries()[1]) == f"Opening.{WINDOW_JOIN}Closing."


def test_a_node_that_is_not_a_text_node_keeps_its_first_emission() -> None:
    """Only a ``TextNode`` can be rewritten, so any other kind stays as it was admitted."""
    ledger = EvidenceLedger(cap=5)
    first = NodeWithScore(node=Document(id_="doc-1", text="Opening."), score=1.0)
    ledger.add([first])

    assert ledger.add([NodeWithScore(node=Document(id_="doc-1", text="Closing."), score=1.0)]) == []
    assert ledger.entries()[0] is first


def test_numbered_returns_stamped_clones_and_leaves_originals_alone() -> None:
    """Numbering a cached docstore node in place would leak into later turns."""
    original = scored_node("a", "A", filename="paper.pdf")
    ledger = EvidenceLedger(cap=5)
    ledger.add([original])

    numbered = ledger.numbered()

    assert numbered[0].node.metadata["citation_index"] == 1
    assert "citation_index" not in original.node.metadata


def test_render_context_keeps_every_entry_and_marks_cuts() -> None:
    """Entries are shortened to fit, never dropped, so numbering stays contiguous."""
    ledger = EvidenceLedger(cap=5)
    ledger.add([scored_node(f"n{i}", "word " * 100, filename=f"doc{i}.pdf") for i in range(3)])

    context, shortened, _ = render_context(ledger.numbered(), llm_view=llm_view, budget_tokens=60, char_token_ratio=3.5)

    assert context.count("citation_index: ") == 3
    assert "characters not shown" in context
    assert shortened == 3


def test_render_context_fits_its_budget_when_it_packs() -> None:
    """Packed entries, their headers and their cut markers stay inside the evidence budget."""
    ledger = EvidenceLedger(cap=22)
    ledger.add([scored_node(f"n{i}", "word " * 1080, filename=f"doc{i}.pdf") for i in range(22)])

    context, _, _ = render_context(ledger.numbered(), llm_view=llm_view, budget_tokens=5429, char_token_ratio=3.5)

    assert estimate_tokens(context, 3.5) <= 5429
    assert context.count("citation_index: ") == 22


def test_render_context_defuses_a_sentinel_inside_a_passage() -> None:
    """Document text quoting the sentinel must not move the answer's boundary."""
    ledger = EvidenceLedger(cap=5)
    ledger.add([scored_node("a", f"The appendix prints {SENTINEL} verbatim.", filename="paper.pdf")])

    context, _, _ = render_context(ledger.numbered(), llm_view=llm_view, budget_tokens=4000, char_token_ratio=3.5)

    assert compose_labelled_answer(context).outcome == "unsplit"


def test_render_context_defuses_a_sentinel_in_metadata() -> None:
    """A forged sentinel in a filename must not move the answer's boundary."""
    ledger = EvidenceLedger(cap=5)
    ledger.add([scored_node("a", "Plain body text.", filename="[[ASSESSMENT]].pdf")])

    context, _, _ = render_context(ledger.numbered(), llm_view=llm_view, budget_tokens=4000, char_token_ratio=3.5)

    assert compose_labelled_answer(context).outcome == "unsplit"
    assert "[[ASSESSMENT]]" not in context


def test_render_context_defuses_a_sentinel_across_a_seam() -> None:
    """A sentinel that only the join of a header and its passage completes is still defused."""
    ledger = EvidenceLedger(cap=5)
    # Bites only because citation_index=0 seeds the first key, leaving the filename last in the header.
    ledger.add([scored_node("a", "ASSESSMENT]] text.", citation_index=0, filename="notes [[")])

    context, _, _ = render_context(ledger.numbered(), llm_view=llm_view, budget_tokens=4000, char_token_ratio=3.5)

    assert compose_labelled_answer(context).outcome == "unsplit"
    assert "text." in context


def test_digest_is_one_short_numbered_line() -> None:
    """The research model reads a bounded digest, never a whole section."""
    line = digest(4, scored_node("a", "Method.\n" + "x" * 2000, filename="paper.pdf", page=3))

    assert line.startswith("[4] (paper.pdf, p. 3) Method.")
    assert "\n" not in line
    assert len(line) <= DIGEST_CHARS + 40


def test_digest_defuses_a_sentinel() -> None:
    """The research transcript is model input too."""
    assert SENTINEL not in digest(1, scored_node("a", f"quote {SENTINEL}"))


def test_digest_defuses_a_sentinel_in_its_file_label() -> None:
    """A forged sentinel in a filename must not reach the research transcript."""
    line = digest(1, scored_node("a", "Text.", filename="[[ASSESSMENT]].pdf"))

    assert SENTINEL not in line
    assert compose_labelled_answer(line).outcome == "unsplit"


def test_digest_flattens_and_bounds_its_file_label() -> None:
    """A file name cannot split the digest, forge a numbered entry or grow without bound."""
    forged = digest(1, scored_node("a", "Text.", filename="a.pdf\n[2] (b.pdf, p. 9) forged passage"))
    oversized = digest(1, scored_node("b", "Text.", filename="x" * 5000))

    for line in (forged, oversized):
        assert "\n" not in line
        assert line.count("[2]") == 0
        assert len(line) <= DIGEST_CHARS + DIGEST_LABEL_CHARS + 40


@pytest.mark.parametrize(
    ("window", "reasoning", "evidence", "output", "entries"),
    [(8192, False, 5429, 1024, 22), (8192, True, 3381, 3072, 14), (32768, False, 28776, 1024, 24)],
)
def test_answer_budget_matches_the_planned_table(
    window: int, reasoning: bool, evidence: int, output: int, entries: int
) -> None:
    """The budget arithmetic of the spec, for a ~715-token template.

    Args:
        window (int): Context window in tokens.
        reasoning (bool): Whether the answer model thinks first.
        evidence (int): Expected evidence tokens.
        output (int): Expected output reserve.
        entries (int): Expected ledger cap.
    """
    budget = answer_budget(ctx_window=window, template="x" * 2500, reasoning=reasoning, char_token_ratio=3.5)

    assert (budget.evidence_tokens, budget.output_tokens, budget.max_entries) == (evidence, output, entries)


def test_the_overflow_markers_are_the_ruled_set() -> None:
    """The phrasings OpenAI, vLLM and llama.cpp use for a prompt longer than the model's context."""
    assert CONTEXT_ERROR_MARKERS == (
        "context length",
        "maximum model length",
        "context size",
        "context window",
        "too long",
        "context_length_exceeded",
        "input length exceeds",
    )


@pytest.mark.parametrize("marker", CONTEXT_ERROR_MARKERS)
def test_every_overflow_marker_matches_in_any_case(marker: str) -> None:
    """Servers capitalize these phrases however they like.

    Args:
        marker (str): One overflow marker.
    """
    assert is_context_overflow(RuntimeError(f"Upstream error: {marker.upper()}."))


def test_an_unrelated_failure_is_not_an_overflow() -> None:
    """A transport fault must never be retried as if the prompt were too long."""
    assert not is_context_overflow(RuntimeError("Connection reset by peer"))


def _two_chars_a_token(text: str) -> int:
    """Count tokens like a tokenizer meeting dense text.

    Args:
        text (str): The text to count.

    Returns:
        int: One token per two characters.
    """
    return len(text) // 2


def _raises(text: str) -> int:
    """Stand in for a tokenizer that fails.

    Args:
        text (str): The text to count, unused.

    Raises:
        RuntimeError: Always.
    """
    raise RuntimeError("tokenizer unavailable")


@pytest.mark.parametrize(
    ("texts", "count_tokens", "expected"),
    [
        (["dense passage " * 50], None, 3.5),
        ([], _two_chars_a_token, 3.5),
        (["", ""], _two_chars_a_token, 3.5),
        (["dense passage " * 50], _two_chars_a_token, 2.0),
        (["sparse passage " * 50], lambda text: max(1, len(text) // 6), 3.5),
        (["dense passage " * 50], _raises, 3.5),
        (["dense passage " * 50], lambda text: 0, 3.5),
    ],
    ids=["no-counter", "no-texts", "empty-texts", "dense", "sparse-is-capped", "counter-raises", "counter-zero"],
)
def test_effective_ratio_never_exceeds_the_configured_one(texts: list[str], count_tokens: Any, expected: float) -> None:
    """Measured density only ever tightens the budget, and a broken counter changes nothing.

    Args:
        texts (list[str]): The rendered parts.
        count_tokens (Any): The tokenizer double, or ``None``.
        expected (float): The ratio to pack with.
    """
    assert effective_ratio(texts, char_token_ratio=3.5, count_tokens=count_tokens) == pytest.approx(expected, abs=0.01)


def test_effective_ratio_samples_every_entry_evenly() -> None:
    """A bounded sample takes the same prefix from every entry, so one long passage cannot dominate."""
    seen: list[str] = []

    def _record(text: str) -> int:
        """Record the sample and count it densely.

        Args:
            text (str): The sample.

        Returns:
            int: One token per two characters.
        """
        seen.append(text)
        return max(1, len(text) // 2)

    effective_ratio(["a" * 100, "b" * 10, "c" * 100], char_token_ratio=3.5, count_tokens=_record, sample_chars=60)

    assert seen == ["a" * 20 + "\n\n" + "b" * 10 + "\n\n" + "c" * 20]


def test_render_context_packs_tighter_for_dense_text() -> None:
    """With a tokenizer, packing follows the evidence's measured density instead of the configured ratio."""
    ledger = EvidenceLedger(cap=10)
    ledger.add([scored_node(f"n{i}", "word " * 1000, filename=f"doc{i}.pdf") for i in range(10)])

    plain, _, _ = render_context(ledger.numbered(), llm_view=llm_view, budget_tokens=4000, char_token_ratio=3.5)
    dense, _, _ = render_context(
        ledger.numbered(),
        llm_view=llm_view,
        budget_tokens=4000,
        char_token_ratio=3.5,
        count_tokens=_two_chars_a_token,
    )

    assert len(dense) < len(plain)
    assert _two_chars_a_token(dense) <= 4000


#: The LLM view the grounded synthesizer uses, which keeps ``table`` and ``reference_metadata``.
PRODUCTION_VIEW = ParentContextPostprocessor._emit_with_llm_exclusion


def _numbered(nodes: list[NodeWithScore]) -> list[NodeWithScore]:
    """Number passages the way the answer step sees them.

    Args:
        nodes (list[NodeWithScore]): The passages.

    Returns:
        list[NodeWithScore]: Clones carrying ``citation_index``.
    """
    ledger = EvidenceLedger(cap=len(nodes))
    ledger.add(nodes)
    return ledger.numbered()


def test_a_pinned_social_post_carries_a_header_of_about_335_characters() -> None:
    """The header the legibility check is sized against, measured through the production view."""
    header = PRODUCTION_VIEW(_numbered([social_post(12)])[0].node).get_metadata_str(mode=MetadataMode.LLM)

    assert 325 <= len(header) <= 345


@pytest.mark.parametrize(("posts", "fits"), [(50, False), (30, True)])
def test_pinned_social_posts_fit_an_8k_window_only_while_every_one_stays_legible(posts: int, fits: bool) -> None:
    """Fifty headers leave no room for the posts themselves; thirty leave every post whole.

    Args:
        posts (int): How many posts are pinned.
        fits (bool): Whether the evidence can be shown legibly.
    """
    template = load_localized_prompt("assessment_answer", default=DEFAULT_ASSESSMENT_ANSWER_PROMPT, lang="en")
    budget = answer_budget(ctx_window=8192, template=template, reasoning=False, char_token_ratio=3.5)

    result = evidence_fits(
        _numbered([social_post(number) for number in range(1, posts + 1)]),
        llm_view=PRODUCTION_VIEW,
        budget_tokens=budget.evidence_tokens,
        char_token_ratio=3.5,
    )

    assert result is fits


def test_every_cut_passage_must_keep_the_legible_minimum() -> None:
    """A cut passage needs ``MIN_TEXT_CHARS`` of text; one character less and the evidence does not fit."""
    nodes = _numbered([scored_node(f"n{number}", "x" * 1000, page=number) for number in range(1, 5)])
    reserved = sum(
        len(llm_view(scored.node).get_metadata_str(mode=MetadataMode.LLM)) + 4 + MARKER_RESERVE_CHARS
        for scored in nodes
    )

    assert evidence_fits(nodes, llm_view=llm_view, budget_tokens=reserved + 4 * MIN_TEXT_CHARS, char_token_ratio=1.0)
    assert not evidence_fits(
        nodes, llm_view=llm_view, budget_tokens=reserved + 4 * MIN_TEXT_CHARS - 1, char_token_ratio=1.0
    )


def test_passages_shown_whole_fit_however_short_they_are() -> None:
    """The minimum applies to what was cut; a short passage shown in full is legible as it is."""
    nodes = _numbered([scored_node(f"n{number}", "Short.", page=number) for number in range(1, 4)])

    assert evidence_fits(nodes, llm_view=llm_view, budget_tokens=200, char_token_ratio=3.5)


def test_evidence_with_no_room_left_for_text_does_not_fit() -> None:
    """When the headers alone use the budget there is nothing to read, even where no text was cut."""
    nodes = _numbered([scored_node(f"n{number}", "", page=number) for number in range(1, 4)])

    assert not evidence_fits(nodes, llm_view=llm_view, budget_tokens=10, char_token_ratio=3.5)
    assert evidence_fits(nodes, llm_view=llm_view, budget_tokens=200, char_token_ratio=3.5)


def test_shared_budget_helpers_keep_the_validator_semantics() -> None:
    """The lifted helpers behave exactly like the validator's originals."""
    assert allocate_shortest_first([10, 100, 1000], 300) == [10, 100, 190]
    assert fit_with_marker("abcdef", 3) == ("abc[... 3 of 6 characters not shown ...]", True)
    assert fit_with_marker("abc", 3) == ("abc", False)


#: Two windows of one invented section, each about a thousand characters.
FIRST_WINDOW = "Alpha: " + "The first survey wave lost two respondents. " * 23
LAST_WINDOW = "Omega: " + "The final wave was rerun with a larger sample. " * 21


def _render_passage(body: str, allowance: int) -> tuple[str, int]:
    """Render one passage whose text may show exactly ``allowance`` characters before its cut marker.

    Args:
        body (str): The passage text.
        allowance (int): Characters its text may show.

    Returns:
        tuple[str, int]: The rendered text without its header, and how many entries were shortened.
    """
    nodes = _numbered([scored_node("sec", body, filename="paper.pdf", page=2)])
    header = llm_view(nodes[0].node).get_metadata_str(mode=MetadataMode.LLM)
    budget = len(header) + 4 + MARKER_RESERVE_CHARS + allowance
    context, shortened, _ = render_context(nodes, llm_view=llm_view, budget_tokens=budget, char_token_ratio=1.0)
    return context.removeprefix(f"{header}\n\n"), shortened


def test_a_shortened_merged_passage_keeps_text_from_every_window() -> None:
    """Cut from the tail, the newest window would be the first to go, so each window is cut on its own."""
    body = f"{FIRST_WINDOW}{WINDOW_JOIN}{LAST_WINDOW}"

    rendered, shortened = _render_passage(body, 1000)

    assert FIRST_WINDOW[:300] in rendered
    assert LAST_WINDOW[:300] in rendered
    assert rendered.count("characters not shown") == 2
    assert WINDOW_JOIN in rendered
    assert shortened == 1
    assert len(rendered) <= 1000 + MARKER_RESERVE_CHARS


def test_a_short_window_stays_whole_and_hands_its_unused_share_to_the_long_one() -> None:
    """The windows share what is left shortest first, as whole passages do."""
    short = "Beta: only a short note."
    long = "Gamma: " + "A long window of invented text. " * 60

    rendered, shortened = _render_passage(f"{short}{WINDOW_JOIN}{long}", 1000)

    share = 1000 - len(WINDOW_JOIN) - MARKER_RESERVE_CHARS - len(short)
    assert rendered == f"{short}{WINDOW_JOIN}{fit_with_marker(long, share)[0]}"
    assert shortened == 1


def test_a_long_passage_without_the_join_renders_exactly_as_before() -> None:
    """Only a passage that holds several windows is cut window by window."""
    body = "The harbour survey repeated its findings. " * 80

    rendered, shortened = _render_passage(body, 1000)

    assert rendered == fit_with_marker(body, 1000)[0]
    assert shortened == 1


@pytest.mark.parametrize("slack", [0, 1, 54, 500])
def test_a_merged_passage_that_fits_is_rendered_untouched(slack: int) -> None:
    """Fitting is judged on the whole passage: its windows would be cut on their own shares, so the guard matters.

    Args:
        slack (int): Characters of allowance beyond the passage's length.
    """
    body = f"{FIRST_WINDOW[:500]}{WINDOW_JOIN}{LAST_WINDOW[:500]}"

    rendered, shortened = _render_passage(body, len(body) + slack)

    assert rendered == body
    assert shortened == 0


def test_a_merged_passage_one_character_over_its_allowance_is_cut_window_by_window() -> None:
    """The first character that does not fit starts the window logic; both windows keep text and a marker."""
    body = f"{FIRST_WINDOW[:500]}{WINDOW_JOIN}{LAST_WINDOW[:500]}"

    rendered, shortened = _render_passage(body, len(body) - 1)

    assert FIRST_WINDOW[:200] in rendered
    assert LAST_WINDOW[:200] in rendered
    assert rendered.count("characters not shown") == 2
    assert shortened == 1


@pytest.mark.parametrize("allowance", [0, 20, 56])
def test_an_allowance_too_small_for_a_character_per_window_cuts_the_whole_passage(allowance: int) -> None:
    """The joins and the second marker take 55 characters; with a character per window unaffordable, the cut is today's.

    Args:
        allowance (int): Characters the passage may show.
    """
    body = f"{FIRST_WINDOW}{WINDOW_JOIN}{LAST_WINDOW}"

    rendered, shortened = _render_passage(body, allowance)

    assert rendered == fit_with_marker(body, allowance)[0]
    assert shortened == 1


def test_a_character_per_window_is_enough_to_cut_window_by_window() -> None:
    """One character more than the fallback's last allowance gives each window one."""
    body = f"{FIRST_WINDOW}{WINDOW_JOIN}{LAST_WINDOW}"

    rendered, _ = _render_passage(body, 57)

    assert rendered == f"{fit_with_marker(FIRST_WINDOW, 1)[0]}{WINDOW_JOIN}{fit_with_marker(LAST_WINDOW, 1)[0]}"


@pytest.mark.parametrize("windows", [2, 3, 5])
@pytest.mark.parametrize("allowance", [60, 200, 600, 1500, 4000])
def test_a_merged_passage_never_renders_past_its_allowance_and_one_marker(windows: int, allowance: int) -> None:
    """The bound is the one a whole-passage cut keeps, however many windows share the allowance.

    Args:
        windows (int): How many windows the passage joins.
        allowance (int): Characters the passage may show.
    """
    parts = [f"W{index}: " + f"invented sentence {index}. " * (10 * (index + 1)) for index in range(windows)]
    body = WINDOW_JOIN.join(parts)

    rendered, shortened = _render_passage(body, allowance)

    assert len(rendered) <= allowance + MARKER_RESERVE_CHARS
    assert shortened == int(len(body) > allowance)
    reserve = (windows - 1) * (len(WINDOW_JOIN) + MARKER_RESERVE_CHARS)
    if len(body) > allowance and allowance - reserve >= windows:
        assert rendered.count(WINDOW_JOIN) == windows - 1


def test_several_merged_passages_pack_within_the_evidence_budget_with_text_from_every_window() -> None:
    """The context bound is unchanged: each entry pays for its joins and markers out of its own allowance."""
    ledger = EvidenceLedger(cap=10)
    for entry in range(5):
        for window in range(3):
            text = f"E{entry}W{window}: " + "Invented window text. " * 28
            ledger.add([scored_node(f"sec-{entry}", text, filename=f"doc{entry}.pdf")])
    assert len(ledger) == 5

    context, shortened, _ = render_context(
        ledger.numbered(), llm_view=llm_view, budget_tokens=3000, char_token_ratio=1.0
    )

    assert len(context) <= 3000
    assert shortened == 5
    assert all(f"E{entry}W{window}:" in context for entry in range(5) for window in range(3))


def _anchored(text: str, match: str, *, node_id: str = "section-1", score: float = 1.0) -> NodeWithScore:
    """Build a passage of an invented section that says where its retrieved match sits.

    Args:
        text (str): The passage text.
        match (str): The retrieved text inside it.
        node_id (str): The section's node id.
        score (float): The hit's score.

    Returns:
        NodeWithScore: The passage, stamped like the parent-context postprocessor stamps it.
    """
    start = text.index(match)
    scored = scored_node(
        node_id, text, filename="paper.pdf", page=2, **{MATCH_SPANS_KEY: [[start, start + len(match)]]}
    )
    scored.score = score
    return scored


def _matches(scored: NodeWithScore) -> list[str]:
    """Return the text each of a passage's match spans points at.

    Args:
        scored (NodeWithScore): The passage.

    Returns:
        list[str]: The matched texts, in span order.
    """
    return [_text(scored)[start:end] for start, end in scored.node.metadata.get(MATCH_SPANS_KEY, [])]


def test_an_extending_window_brings_its_match_along() -> None:
    """The arriving window's match is shifted past the held text and the join."""
    ledger = EvidenceLedger(cap=5)
    ledger.add([_anchored("The study opens with a pilot.", "pilot")])

    ledger.add([_anchored("It closes with a limitations note.", "limitations")])

    assert _matches(ledger.entries()[0]) == ["pilot", "limitations"]


def test_a_fuller_window_keeps_the_match_of_the_slice_it_replaces() -> None:
    """A section that supersedes a slice of itself also keeps the slice's match, shifted to where the slice sits."""
    ledger = EvidenceLedger(cap=5)
    ledger.add([_anchored("opens with a pilot", "pilot")])

    ledger.add([_anchored("The study opens with a pilot, then compares.", "compares")])

    assert _matches(ledger.entries()[0]) == ["compares", "pilot"]


def test_a_contained_window_with_a_new_match_is_news() -> None:
    """A window the entry already shows still adds a match the entry did not have, so research hears of it."""
    ledger = EvidenceLedger(cap=5)
    whole = "The study opens with a pilot, then compares."
    ledger.add([_anchored(whole, "pilot")])
    arriving = _anchored("then compares.", "compares")

    changed = ledger.add([arriving])

    assert changed == [(1, arriving)]
    entry = ledger.entries()[0]
    assert _text(entry) == whole
    assert _matches(entry) == ["pilot", "compares"]


def test_a_contained_window_whose_match_is_covered_adds_nothing() -> None:
    """A window repeating a match the entry already carries is not news."""
    ledger = EvidenceLedger(cap=5)
    first = _anchored("The study opens with a pilot, then compares.", "opens with a pilot")
    ledger.add([first])

    assert ledger.add([_anchored("with a pilot", "pilot")]) == []
    assert ledger.entries()[0] is first


def test_a_match_across_two_touching_held_matches_is_news() -> None:
    """Held matches count one by one, so a match straddling the point where two of them touch is not covered."""
    whole = "The study opens with a pilot, then compares two invented cohorts."
    held = scored_node("section-1", whole, filename="paper.pdf", page=2, **{MATCH_SPANS_KEY: [[0, 15], [15, 34]]})
    ledger = EvidenceLedger(cap=5)
    ledger.add([held])
    arriving = _anchored("opens with a pilot", "opens with")

    changed = ledger.add([arriving])

    assert changed == [(1, arriving)]
    assert _matches(ledger.entries()[0]) == ["The study opens", " with a pilot, then", "opens with"]


@pytest.mark.parametrize(
    ("arriving_matches", "matches"),
    [
        (["opens with a pilot"], ["opens with a pilot"]),
        (["opens wi", "th a pilot"], ["opens wi", "th a pilot", "with a"]),
    ],
    ids=["inside-one-arriving-match", "across-two-touching-arriving-matches"],
)
def test_a_fuller_window_keeps_a_held_match_only_when_no_single_match_of_its_own_holds_it(
    arriving_matches: list[str], matches: list[str]
) -> None:
    """A re-cut of a child the arriving section already stamps whole adds no floor; one across two of its matches does.

    Args:
        arriving_matches (list[str]): The arriving section's matched texts.
        matches (list[str]): The matched texts the merged entry must carry, in span order.
    """
    whole = "The study opens with a pilot, then compares."
    ledger = EvidenceLedger(cap=5)
    ledger.add([_anchored("opens with a pilot", "with a")])
    spans = [[whole.index(text), whole.index(text) + len(text)] for text in arriving_matches]
    arriving = scored_node("section-1", whole, filename="paper.pdf", page=2, **{MATCH_SPANS_KEY: spans})

    changed = ledger.add([arriving])

    assert changed == [(1, arriving)]
    assert _text(ledger.entries()[0]) == whole
    assert _matches(ledger.entries()[0]) == matches


#: An invented section whose decisive sentence sits deep inside it.
DEEP_FACT = "The pilot excluded every respondent recruited after March."
DEEP_SECTION = ("Background filler sentence. " * 190)[:5214] + DEEP_FACT + " Closing filler sentence." * 40


def _render_numbered(nodes: list[NodeWithScore], allowance: int) -> str:
    """Render one passage whose text may show exactly ``allowance`` characters before its cut marker.

    Args:
        nodes (list[NodeWithScore]): The passage, alone in its ledger.
        allowance (int): Characters its text may show.

    Returns:
        str: The rendered context.
    """
    numbered = _numbered(nodes)
    header = llm_view(numbered[0].node).get_metadata_str(mode=MetadataMode.LLM)
    budget = len(header) + 4 + MARKER_RESERVE_CHARS + allowance
    return render_context(numbered, llm_view=llm_view, budget_tokens=budget, char_token_ratio=1.0)[0]


def test_a_shortened_passage_keeps_the_text_around_its_match() -> None:
    """Cut to 700 characters, a promoted section shows its retrieved sentence, not its opening."""
    context = _render_numbered([_anchored(DEEP_SECTION, DEEP_FACT)], 700)

    assert DEEP_FACT in context
    # The body opens on the gap before the match, not on the section's first characters.
    assert context.split("\n\n", 1)[1].startswith("[... ")


def test_each_window_of_a_shortened_merged_passage_keeps_its_own_match() -> None:
    """The match spans are clipped into each window, so every window is cut around its own match."""
    first_fact, last_fact = "The first wave lost two respondents.", "The final wave was rerun with more."
    # The facts sit at different offsets of their windows, so a span left unshifted points at the wrong text.
    first = ("Opening filler words. " * 40)[:800] + first_fact + " Trailing filler words." * 8
    last = ("Middle filler words. " * 40)[:300] + last_fact + " Closing filler words." * 30
    ledger = EvidenceLedger(cap=5)
    ledger.add([_anchored(first, first_fact)])
    ledger.add([_anchored(last, last_fact)])

    context = _render_numbered(ledger.entries(), 700)

    assert first_fact in context
    assert last_fact in context
    assert WINDOW_JOIN in context


def test_digest_reads_around_the_match() -> None:
    """The research model's digest of a promoted section shows the retrieved sentence, and says it skipped ahead."""
    line = digest(3, _anchored(DEEP_SECTION, DEEP_FACT))

    assert line.startswith("[3] (paper.pdf, p. 2) … ")
    assert DEEP_FACT in line
    assert len(line) <= DIGEST_CHARS + 40


def test_digest_of_a_match_near_the_start_reads_from_the_start() -> None:
    """A match inside the opening needs no skip, so the digest starts where the passage does."""
    line = digest(1, _anchored(DEEP_SECTION, "Background filler sentence. Background"))

    assert line.startswith("[1] (paper.pdf, p. 2) Background filler sentence.")
    assert "…" not in line


@pytest.mark.parametrize(
    "lead",
    [[], [[-50, -10]], [[len(DEEP_SECTION) + 5, len(DEEP_SECTION) + 90]]],
    ids=["first-stamped", "before-the-text-skipped", "past-the-text-skipped"],
)
def test_digest_reads_around_the_first_stamped_match_not_the_earliest_in_the_text(lead: list[list[int]]) -> None:
    """The top-ranked child is stamped first, so research reads it even where a lower-ranked one sits earlier.

    Args:
        lead (list[list[int]]): Spans stamped ahead of the first real one, empty once clipped to the text.
    """
    fact_at = DEEP_SECTION.index(DEEP_FACT)
    early = DEEP_SECTION.index("Background", 600)
    spans = [*lead, [fact_at, fact_at + len(DEEP_FACT)], [early, early + 40]]

    line = digest(2, scored_node("section-1", DEEP_SECTION, filename="paper.pdf", page=2, **{MATCH_SPANS_KEY: spans}))

    assert line.startswith("[2] (paper.pdf, p. 2) … ")
    assert DEEP_FACT in line


def _sized(count: int, body: int, *, label: int = 0, start: int = 1) -> list[NodeWithScore]:
    """Build invented passages of a given text length whose visible header grows with ``label``.

    Args:
        count (int): How many passages.
        body (int): Characters of text each carries.
        label (int): Characters of file name each header shows.
        start (int): The first passage's page number, which also varies its id.

    Returns:
        list[NodeWithScore]: The passages, unnumbered.
    """
    return [
        scored_node(f"p{page}", "w" * body, filename="f" * label, page=page) for page in range(start, start + count)
    ]


@pytest.mark.parametrize("bodies", [[], [0], [10, 299, 300, 301], [1000] * 6, [5, 2000, 40, 300, 700]])
@pytest.mark.parametrize("label", [0, 40, 400])
@pytest.mark.parametrize("budget", [0, 20, 100, 300, 600, 1200, 3000])
@pytest.mark.parametrize("count_tokens", [None, _two_chars_a_token], ids=["ratio", "tokenizer"])
def test_the_closed_form_agrees_with_packing(bodies: list[int], label: int, budget: int, count_tokens: Any) -> None:
    """``Room`` says legible exactly when the shortest-first packing keeps every cut passage legible.

    Args:
        bodies (list[int]): Each passage's text length.
        label (int): Characters of file name in each header.
        budget (int): The evidence budget in tokens.
        count_tokens (Any): The tokenizer double, or ``None``.
    """
    nodes = _numbered(
        [scored_node(f"p{i}", "w" * length, page=i, filename="f" * label) for i, length in enumerate(bodies)]
    )
    kwargs: dict[str, Any] = {"llm_view": llm_view, "budget_tokens": budget, "char_token_ratio": 3.5}

    assert Room(nodes, count_tokens=count_tokens, **kwargs).legible is evidence_fits(
        nodes, count_tokens=count_tokens, **kwargs
    )


@pytest.mark.parametrize(
    ("body", "slack", "legible"),
    [(1000, -1, False), (1000, 0, True), (1000, 1, True), (0, 0, False), (0, 1, True)],
    ids=["one-short-of-the-minimum", "exactly-the-minimum", "one-over", "no-text-room", "one-character-of-room"],
)
def test_the_closed_form_agrees_with_packing_at_its_boundaries(body: int, slack: int, legible: bool) -> None:
    """At exactly the legible minimum, and at exactly no room for text, both answers match.

    Args:
        body (int): Each passage's text length.
        slack (int): Characters beyond the boundary the budget allows.
        legible (bool): The expected answer.
    """
    nodes = _numbered(_sized(4, body))
    reserved = sum(
        len(llm_view(scored.node).get_metadata_str(mode=MetadataMode.LLM)) + 4 + MARKER_RESERVE_CHARS
        for scored in nodes
    )
    kwargs: dict[str, Any] = {
        "llm_view": llm_view,
        "budget_tokens": reserved + 4 * min(body, MIN_TEXT_CHARS) + slack,
        "char_token_ratio": 1.0,
    }

    assert Room(nodes, **kwargs).legible is legible
    assert evidence_fits(nodes, **kwargs) is legible


def _restricted(seed: list[NodeWithScore], budget: int, *, cap: int = 50) -> EvidenceLedger:
    """Build a ledger holding ``seed`` that admits nothing which would leave the evidence illegible.

    Args:
        seed (list[NodeWithScore]): The seed passages.
        budget (int): The evidence budget, in characters (one per token).
        cap (int): The ledger cap.

    Returns:
        EvidenceLedger: The ledger, restricted.
    """
    ledger = EvidenceLedger(cap=cap)
    ledger.add(seed)
    ledger.restrict(Room(ledger.numbered(), llm_view=llm_view, budget_tokens=budget, char_token_ratio=1.0))
    return ledger


def test_a_candidate_whose_header_would_break_legibility_is_refused_and_the_batch_goes_on() -> None:
    """A metadata-heavy passage is refused; the lighter one after it in the same batch is admitted."""
    ledger = _restricted([scored_node("seed", "Seed passage text. " * 20, page=1)], 2000)
    heavy = scored_node("heavy", "Heavy passage.", page=2, filename="h" * 1800)
    light = scored_node("light", "Light passage.", page=3, filename="light.pdf")

    changed = ledger.add([heavy, light])

    assert changed == [(2, light)]
    assert [scored.node.node_id for scored in ledger.entries()] == ["seed", "light"]
    assert evidence_fits(ledger.numbered(), llm_view=llm_view, budget_tokens=2000, char_token_ratio=1.0)


def test_an_extension_that_would_break_legibility_is_refused() -> None:
    """Admission prices a merged entry too; a window it refuses leaves the entry as it was."""
    # The header leaves about 210 characters: the 64-character window fits, a merge needing 300 does not.
    held = scored_node("sec", "Opening window. " * 4, page=1, filename="f" * 1700)
    ledger = _restricted([held], 2000)

    assert ledger.add([scored_node("sec", "Closing window. " * 40, page=1, filename="f" * 1700)]) == []
    assert ledger.entries()[0] is held


def test_an_extension_is_priced_in_place_of_the_entry_it_replaces() -> None:
    """A later window of a passage near the limit still fits: the merged entry's cost replaces the held one's."""
    ledger = EvidenceLedger(cap=5)
    ledger.add([scored_node("sec", "Opening window. " * 20, page=1)])
    header = len(llm_view(ledger.numbered()[0].node).get_metadata_str(mode=MetadataMode.LLM))
    budget = header + 4 + MARKER_RESERVE_CHARS + MIN_TEXT_CHARS + 100
    ledger.restrict(Room(ledger.numbered(), llm_view=llm_view, budget_tokens=budget, char_token_ratio=1.0))
    arriving = scored_node("sec", "Closing window. " * 20, page=1)

    assert ledger.add([arriving]) == [(1, arriving)]
    assert _text(ledger.entries()[0]) == f"{'Opening window. ' * 20}{WINDOW_JOIN}{'Closing window. ' * 20}"


def test_admission_keeps_every_state_legible() -> None:
    """Whatever arrives, in whatever order, what the ledger holds stays legible; some arrivals are refused."""
    rng = random.Random(7)
    ledger = _restricted([scored_node("s0", "Seed passage. " * 30, page=0)], 9000)
    refused = 0

    for step in range(300):
        body = "w" * rng.choice([5, 120, 299, 300, 900, 2500])
        label = "f" * rng.choice([0, 60, 600, 2500])
        node_id = f"c{rng.randrange(30)}"
        known = node_id in {scored.node.node_id for scored in ledger.entries()}
        if not ledger.add([scored_node(node_id, body, page=step, filename=label)]) and not known and not ledger.full:
            refused += 1

        assert evidence_fits(ledger.numbered(), llm_view=llm_view, budget_tokens=9000, char_token_ratio=1.0)
    assert len(ledger) > 4
    assert refused > 0


def test_the_ledger_closes_when_not_even_a_bare_legible_passage_would_fit() -> None:
    """The room left must hold a header-less 300-character text, its marker and its joins, or research stops."""
    seed = _sized(1, 300, label=10)
    header = len(llm_view(_numbered(seed)[0].node).get_metadata_str(mode=MetadataMode.LLM))
    spent = header + 4 + MARKER_RESERVE_CHARS + MIN_TEXT_CHARS
    bare = 4 + MARKER_RESERVE_CHARS + MIN_TEXT_CHARS

    assert _restricted(seed, spent + bare).full is False
    assert _restricted(seed, spent + bare - 1).full is True


def test_fit_ledger_leaves_out_the_latest_additions_until_every_passage_is_legible() -> None:
    """The newest passages go first, so numbering stays contiguous and the earliest evidence stays."""
    nodes = _numbered(_sized(8, 1000))
    header = len(llm_view(nodes[0].node).get_metadata_str(mode=MetadataMode.LLM))
    budget = 5 * (header + 4 + MARKER_RESERVE_CHARS + MIN_TEXT_CHARS)

    kept = fit_ledger(nodes, keep=2, llm_view=llm_view, budget_tokens=budget, char_token_ratio=1.0)

    assert kept == nodes[:5]


def test_fit_ledger_never_leaves_out_the_seed() -> None:
    """A seed too large to show legibly is kept whole; research additions are all that may go."""
    nodes = _numbered(_sized(6, 1000))

    assert fit_ledger(nodes, keep=6, llm_view=llm_view, budget_tokens=100, char_token_ratio=1.0) == nodes
    assert fit_ledger(nodes, keep=4, llm_view=llm_view, budget_tokens=100, char_token_ratio=1.0) == nodes[:4]


def test_a_compact_header_keeps_its_citation_line_first_and_caps_the_rest() -> None:
    """Under compaction the number always survives; the rest of the metadata is cut with a marker."""
    nodes = _numbered(_sized(1, 40, label=2000))

    context, _, _ = render_context(nodes, llm_view=llm_view, budget_tokens=10_000, char_token_ratio=1.0, compact=True)

    header, body = context.split("\n\n", 1)
    assert header.startswith("citation_index: 1\nfilename: fff")
    assert "characters not shown" in header
    assert len(header) <= len("citation_index: 1\n") + COMPACT_HEADER_CHARS + MARKER_RESERVE_CHARS
    assert body == "w" * 40


def test_a_compact_header_short_enough_is_only_reordered() -> None:
    """Compaction cuts nothing it does not have to."""
    nodes = _numbered(_sized(1, 40, label=10))

    context, _, _ = render_context(nodes, llm_view=llm_view, budget_tokens=10_000, char_token_ratio=1.0, compact=True)

    assert context.split("\n\n", 1)[0] == f"citation_index: 1\nfilename: {'f' * 10}\npage: 1"


def test_compact_legibility_is_judged_on_the_headers_it_renders() -> None:
    """The check and the render use the same compact headers, so a budget exactly legible renders within itself."""
    nodes = _numbered(_sized(4, 1000, label=2000))
    context, _, _ = render_context(nodes, llm_view=llm_view, budget_tokens=100_000, char_token_ratio=1.0, compact=True)
    headers = [block for block in context.split("\n\n") if block.startswith("citation_index: ")]
    budget = sum(len(header) + 4 + MARKER_RESERVE_CHARS + MIN_TEXT_CHARS for header in headers)

    assert len(headers) == 4
    assert evidence_fits(nodes, llm_view=llm_view, budget_tokens=budget, char_token_ratio=1.0, compact=True)
    assert not evidence_fits(nodes, llm_view=llm_view, budget_tokens=budget - 1, char_token_ratio=1.0, compact=True)
    packed, _, _ = render_context(nodes, llm_view=llm_view, budget_tokens=budget, char_token_ratio=1.0, compact=True)
    assert len(packed) <= budget


def _join(*windows: str) -> str:
    """Join invented windows the way the ledger joins the windows of one passage.

    Args:
        *windows (str): The windows' texts.

    Returns:
        str: The passage text.
    """
    return WINDOW_JOIN.join(windows)


#: Where the second window starts in a passage joining two windows of a thousand characters.
SECOND = 1000 + len(WINDOW_JOIN)


def _matched_windows(count: int, length: int = 1270) -> tuple[str, list[tuple[int, int]]]:
    """Join ``count`` invented windows, each holding one match in its middle.

    Args:
        count (int): How many windows.
        length (int): Characters per window.

    Returns:
        tuple[str, list[tuple[int, int]]]: The passage text and its match spans.
    """
    step = length + len(WINDOW_JOIN)
    return _join(*["w" * length] * count), [(index * step + 600, index * step + 680) for index in range(count)]


CODEX_ENTRY = _matched_windows(7)


@pytest.mark.parametrize(
    ("body", "anchors", "floor"),
    [
        ("x" * 1000, [], 300),
        ("x" * 1000, [(800, 850)], 300),
        ("x" * 120, [(10, 20)], 120),
        ("x" * 1000, [(100, 120), (800, 820)], 500),
        ("x" * 1000, [(100, 160), (150, 220)], 500),
        ("x" * 1000, [(100, 150), (150, 220)], 500),
        ("x" * 1000, [(100, 400), (150, 220)], 500),
        ("x" * 1000, [(100, 150), (100, 150)], 300),
        ("x" * 700, [(10, 20), (300, 310), (600, 610)], 700),
        ("x" * 1000, [(-40, -10), (1005, 1050)], 300),
        (_join("a" * 1000, "b" * 1000), [(800, 820), (SECOND + 300, SECOND + 320)], 555),
        (_join("a" * 120, "b" * 2000), [(10, 20), (125 + 900, 125 + 920)], 425),
        (_join("a" * 1000, "b" * 1000), [], 300),
        (_join("a" * 1000, "b" * 1000), [(800, 820)], 305),
        (_join("a" * 1000, "b" * 1000), [(990, SECOND + 10)], 555),
        (_join("a" * 1000, "b" * 1000), [(990, SECOND - 2)], 305),
        (_join("a" * 1000, "b" * 1000), [(100, 120), (800, 820), (SECOND + 300, SECOND + 320)], 805),
        (_join("a" * 100, "b" * 100), [(10, 20), (115, 125)], 205),
        (_join(*["w" * 100] * 6), [], 300),
        (_join(*["w" * 100] * 7), [], 337),
        (*CODEX_ENTRY, 2080),
    ],
    ids=[
        "no-match",
        "one-match",
        "shorter-than-the-minimum",
        "two-matches",
        "overlapping-matches",
        "touching-matches",
        "nested-matches",
        "a-repeated-match",
        "three-matches-in-a-shorter-text",
        "matches-outside-the-text",
        "two-windows-a-match-each",
        "a-short-matched-window-shown-whole",
        "windows-without-matches",
        "one-of-two-windows-matched",
        "a-match-across-a-seam-in-the-text",
        "a-match-ending-inside-the-seam",
        "two-matches-in-one-of-two-windows",
        "windows-shorter-than-their-floor",
        "six-windows-without-matches",
        "seven-windows-without-matches",
        "seven-windows-a-match-each",
    ],
)
def test_text_floor_reserves_a_region_for_every_match_and_a_seam_for_every_window(
    body: str, anchors: list[tuple[int, int]], floor: int
) -> None:
    """Each window needs ``min(len, 250 * matches)``, each seam 55, and the passage never less than today's minimum.

    Matches that touch, overlap or nest count one by one; only an exact repeat counts once.

    Args:
        body (str): The passage text.
        anchors (list[tuple[int, int]]): Its match spans.
        floor (int): The least text it must keep.
    """
    assert text_floor(body, anchors) == floor


@pytest.mark.parametrize("length", [0, 1, 299, 300, 301, 5000])
@pytest.mark.parametrize("span", [None, (0.0, 0.01), (0.5, 0.6), (0.9, 1.0)], ids=["none", "start", "middle", "end"])
def test_a_seed_passage_keeps_todays_floor(length: int, span: tuple[float, float] | None) -> None:
    """One window holding at most one match, as every seed passage does, is floored exactly as before.

    Args:
        length (int): The passage's length.
        span (tuple[float, float] | None): Its match, as fractions of the text.
    """
    anchors: list[tuple[int, int]] = [] if span is None else [(int(span[0] * length), int(span[1] * length))]

    assert text_floor("s" * length, anchors) == min(length, MIN_TEXT_CHARS)


def _header_chars(nodes: list[NodeWithScore]) -> int:
    """Return what the numbered passages' headers, joins and cut markers reserve.

    Args:
        nodes (list[NodeWithScore]): The numbered passages.

    Returns:
        int: The reserved characters.
    """
    return sum(
        len(llm_view(scored.node).get_metadata_str(mode=MetadataMode.LLM)) + 4 + MARKER_RESERVE_CHARS
        for scored in nodes
    )


def _facts_section(node_id: str, windows: int) -> tuple[list[NodeWithScore], list[str]]:
    """Build the windows of one invented section, each holding its own decisive sentence in its middle.

    Args:
        node_id (str): The section's node id, shared by every window.
        windows (int): How many windows.

    Returns:
        tuple[list[NodeWithScore], list[str]]: The windows and their sentences.
    """
    nodes: list[NodeWithScore] = []
    facts: list[str] = []
    for index in range(windows):
        fact = f"Invented finding {index} of {node_id}: the cohort shrank in week {index + 2}."
        before = (f"Filler {index} before the finding. " * 30)[:600]
        after = (f" Filler {index} after the finding." * 30)[:600]
        nodes.append(_anchored(before + fact + after, fact, node_id=node_id))
        facts.append(fact)
    return nodes, facts


def test_a_merged_passage_keeps_every_match_beside_a_long_plain_one() -> None:
    """Floored for all seven of its matches, a merged passage is not cut to an equal share with a plain one."""
    windows, facts = _facts_section("sec", 7)
    ledger = EvidenceLedger(cap=5)
    for window in windows:
        ledger.add([window])
    ledger.add([scored_node("plain", "Plain invented passage text. " * 120, filename="notes.pdf", page=1)])
    nodes = ledger.numbered()
    merged = nodes[0]
    floor = text_floor(_text(merged), [(start, end) for start, end in merged.node.metadata[MATCH_SPANS_KEY]])
    budget = _header_chars(nodes) + floor + MIN_TEXT_CHARS
    assert evidence_fits(nodes, llm_view=llm_view, budget_tokens=budget, char_token_ratio=1.0)

    context, shortened, _ = render_context(nodes, llm_view=llm_view, budget_tokens=budget, char_token_ratio=1.0)

    assert shortened == 2
    assert [fact for fact in facts if fact not in context] == []
    assert len(context) <= budget


@pytest.mark.parametrize(("slack", "admitted"), [(-1, False), (0, True)], ids=["one-short", "exactly-enough"])
def test_an_extension_is_admitted_only_with_room_for_every_match_it_brings(slack: int, admitted: bool) -> None:
    """A second matched window raises its passage's floor from 300 to 555; one character short, it is refused.

    Args:
        slack (int): Characters beyond the merged floor the budget allows.
        admitted (bool): Whether the window is taken.
    """
    windows, facts = _facts_section("sec", 2)
    ledger = EvidenceLedger(cap=5)
    ledger.add(windows[:1])
    budget = _header_chars(ledger.numbered()) + 555 + slack
    ledger.restrict(Room(ledger.numbered(), llm_view=llm_view, budget_tokens=budget, char_token_ratio=1.0))

    changed = ledger.add(windows[1:])

    assert bool(changed) is admitted
    context, _, _ = render_context(ledger.numbered(), llm_view=llm_view, budget_tokens=budget, char_token_ratio=1.0)
    assert facts[0] in context
    assert (facts[1] in context) is admitted


def test_defused_sentinels_before_a_match_do_not_move_the_text_kept_around_it() -> None:
    """Spans index the stored text, so the cut is made there and the sentinels are defused in what it keeps."""
    quoted = "The appendix quotes [[ ASSESSMENT ]] verbatim. " * 60
    section = quoted + ("Filler sentence. " * 80) + DEEP_FACT + " Closing filler." * 40

    context = _render_numbered([_anchored(section, DEEP_FACT)], MIN_TEXT_CHARS)

    assert DEEP_FACT in context
    assert compose_labelled_answer(context).outcome == "unsplit"


def test_the_room_counts_an_extension_it_refuses() -> None:
    """A window whose matches would no longer fit is a refusal, and the entry stays as it was."""
    windows, _ = _facts_section("sec", 2)
    ledger = EvidenceLedger(cap=5)
    ledger.add(windows[:1])
    budget = _header_chars(ledger.numbered()) + 555 - 1
    ledger.restrict(Room(ledger.numbered(), llm_view=llm_view, budget_tokens=budget, char_token_ratio=1.0))

    assert (ledger.add(windows[1:]), ledger.refusals) == ([], 1)


def test_the_room_counts_a_new_passage_it_refuses() -> None:
    """A passage whose header would crowd the others out is a refusal; the one admitted after it is not."""
    ledger = _restricted([scored_node("seed", "Seed passage text. " * 20, page=1)], 2000)
    heavy = scored_node("heavy", "Heavy passage.", page=2, filename="h" * 1800)
    light = scored_node("light", "Light passage.", page=3, filename="light.pdf")

    assert ([number for number, _ in ledger.add([heavy, light])], ledger.refusals) == ([2], 1)


def test_the_room_counts_a_passage_skipped_while_it_is_out_of_room() -> None:
    """Once the room has closed the ledger, every new passage it skips is a refusal."""
    seed = _sized(1, 300, label=10)
    ledger = _restricted(seed, _header_chars(_numbered(seed)) + MIN_TEXT_CHARS + 353)
    assert ledger.out_of_room is True

    assert (ledger.add(_sized(2, 40, start=5)), ledger.refusals) == ([], 2)


@pytest.mark.parametrize("slack", [353, 100_000], ids=["closed-room", "open-room"])
def test_a_passage_skipped_at_the_entry_cap_is_no_refusal(slack: int) -> None:
    """The cap, not the room, stops the passage, whatever the room says.

    Args:
        slack (int): Characters of room beyond what the seed keeps.
    """
    seed = _sized(1, 300, label=10)
    ledger = _restricted(seed, _header_chars(_numbered(seed)) + MIN_TEXT_CHARS + slack, cap=1)

    assert (ledger.add(_sized(2, 40, start=5)), ledger.refusals) == ([], 0)


def test_without_a_room_nothing_is_refused() -> None:
    """The seed is admitted before any room is installed, and the entry cap alone limits it."""
    ledger = EvidenceLedger(cap=2)

    ledger.add([scored_node("heavy", "Heavy passage.", page=1, filename="h" * 50_000), *_sized(3, 40, start=5)])

    assert (len(ledger), ledger.refusals) == (2, 0)


def test_out_of_room_means_the_room_and_not_the_cap_closed_the_ledger() -> None:
    """Research can only report running out of room when the room, not the entry cap, stopped it."""
    seed = _sized(1, 300, label=10)
    spent = _header_chars(_numbered(seed)) + MIN_TEXT_CHARS
    bare = 4 + MARKER_RESERVE_CHARS + MIN_TEXT_CHARS
    unrestricted = EvidenceLedger(cap=5)
    unrestricted.add(seed)

    assert _restricted(seed, spent + bare).out_of_room is False
    assert _restricted(seed, spent + bare - 1).out_of_room is True
    assert _restricted(seed, spent + bare - 1, cap=1).out_of_room is False
    assert unrestricted.out_of_room is False


@pytest.mark.parametrize("windows", [1, 2, 3, 7])
@pytest.mark.parametrize("matches", [0, 1, 2])
@pytest.mark.parametrize("budget", [0, 300, 900, 1800, 2600, 4000, 9000])
@pytest.mark.parametrize("count_tokens", [None, _two_chars_a_token], ids=["ratio", "tokenizer"])
def test_the_closed_form_agrees_with_packing_for_merged_matched_passages(
    windows: int, matches: int, budget: int, count_tokens: Any
) -> None:
    """With a floor per retained match, ``Room`` still says legible exactly when packing keeps every floor.

    Args:
        windows (int): Windows per passage.
        matches (int): Matches per window.
        budget (int): The evidence budget in tokens.
        count_tokens (Any): The tokenizer double, or ``None``.
    """
    ledger = EvidenceLedger(cap=5)
    for entry in range(3):
        for index in range(windows):
            spans = [[40 + 200 * match, 60 + 200 * match] for match in range(matches)]
            extra: dict[str, Any] = {MATCH_SPANS_KEY: spans} if spans else {}
            ledger.add([scored_node(f"sec-{entry}", f"E{entry}W{index}: " + "invented window text " * 30, **extra)])
    nodes = ledger.numbered()
    kwargs: dict[str, Any] = {"llm_view": llm_view, "budget_tokens": budget, "char_token_ratio": 3.5}

    assert Room(nodes, count_tokens=count_tokens, **kwargs).legible is evidence_fits(
        nodes, count_tokens=count_tokens, **kwargs
    )


@pytest.mark.parametrize("budget", [0, 300, 520, 900, 2600, 9000])
@pytest.mark.parametrize("compact", [False, True], ids=["full", "compact"])
@pytest.mark.parametrize("count_tokens", [None, _two_chars_a_token], ids=["ratio", "tokenizer"])
def test_a_render_is_legible_exactly_when_the_evidence_fits(budget: int, compact: bool, count_tokens: Any) -> None:
    """The render reports the legibility it packed for, judged as ``evidence_fits`` judges it.

    Args:
        budget (int): The evidence budget in tokens.
        compact (bool): Whether headers are compacted.
        count_tokens (Any): The tokenizer double, or ``None``.
    """
    ledger = EvidenceLedger(cap=5)
    for entry in range(3):
        for index in range(2):
            text = f"E{entry}W{index}: " + "invented window text " * 30
            ledger.add([scored_node(f"sec-{entry}", text, page=entry, **{MATCH_SPANS_KEY: [[40, 60]]})])
    nodes = ledger.numbered()
    kwargs: dict[str, Any] = {
        "llm_view": llm_view,
        "budget_tokens": budget,
        "char_token_ratio": 3.5,
        "count_tokens": count_tokens,
        "compact": compact,
    }

    assert render_context(nodes, **kwargs).legible is evidence_fits(nodes, **kwargs)


#: Random text from this alphabet never spells a sentinel, so every shown match reads verbatim.
_ALPHABET = "abcdefgh "


def _random_window(rng: random.Random, node_id: str, *, seam: bool) -> NodeWithScore:
    """Build one invented window of random text carrying up to three match spans.

    Args:
        rng (random.Random): The seeded source.
        node_id (str): The section's node id.
        seam (bool): Whether the window's own text holds a ``WINDOW_JOIN`` with a match across it.

    Returns:
        NodeWithScore: The window.
    """
    text = "".join(rng.choice(_ALPHABET) for _ in range(rng.choice([30, 180, 260, 640, 1500])))
    spans: list[list[int]] = []
    if seam and len(text) > 60:
        cut = rng.randrange(20, len(text) - 20)
        text = f"{text[:cut]}{WINDOW_JOIN}{text[cut:]}"
        spans.append([cut - rng.randrange(1, 20), cut + len(WINDOW_JOIN) + rng.randrange(1, 20)])
    for _ in range(rng.randrange(0, 4)):
        start = rng.randrange(0, len(text))
        spans.append([start, min(len(text), start + rng.choice([1, 25, 120, 260]))])
    metadata: dict[str, Any] = {"filename": "f" * rng.choice([0, 30, 300]), "page": rng.randrange(1, 9)}
    if spans:
        metadata[MATCH_SPANS_KEY] = spans
    return scored_node(node_id, text, **metadata)


def _held_prefixes(scored: NodeWithScore) -> list[str]:
    """Return the opening of every match a passage holds, window by window, as packing must keep it.

    Args:
        scored (NodeWithScore): The passage.

    Returns:
        list[str]: The first ``MIN_REGION_CHARS`` characters, or fewer, of each distinct match in each window.
    """
    spans = [(start, end) for start, end in scored.node.metadata.get(MATCH_SPANS_KEY, [])]
    prefixes: list[str] = []
    offset = 0
    for window in _text(scored).split(WINDOW_JOIN):
        local = distinct_spans([(start - offset, end - offset) for start, end in spans], len(window))
        prefixes += [window[start : start + min(MIN_REGION_CHARS, end - start)] for start, end in local]
        offset += len(window) + len(WINDOW_JOIN)
    return prefixes


def _render_legibly(nodes: list[NodeWithScore], budget: int) -> int:
    """Render legible evidence and check what legibility promises: every match, the minimum share and the budget.

    Args:
        nodes (list[NodeWithScore]): The numbered evidence, legible in ``budget``.
        budget (int): The evidence budget, in characters (one per token).

    Returns:
        int: How many passages were shortened.
    """
    context, shortened, _ = render_context(nodes, llm_view=llm_view, budget_tokens=budget, char_token_ratio=1.0)
    assert len(context) <= budget
    assert [prefix for scored in nodes for prefix in _held_prefixes(scored) if prefix not in context] == []
    headers, bodies, anchors = _parts(nodes, llm_view)
    _, _, allowances = _allowances(
        headers, bodies, anchors, budget_tokens=budget, char_token_ratio=1.0, count_tokens=None
    )
    kept = zip(bodies, allowances, strict=True)
    assert all(allowance >= min(len(body), MIN_TEXT_CHARS) for body, allowance in kept)
    return shortened


def test_legible_evidence_renders_every_match_it_holds() -> None:
    """Seeded fuzz: whenever the evidence is legible, packing keeps every match, every minimum share and the budget."""
    rng = random.Random(41)
    legible = shortened = 0

    for _ in range(1500):
        ledger = EvidenceLedger(cap=50)
        for entry in range(rng.randrange(1, 6)):
            for _ in range(rng.choice([1, 1, 2, 3, 5, 8])):
                ledger.add([_random_window(rng, f"e{entry}", seam=rng.random() < 0.15)])
        nodes = ledger.numbered()
        budget = rng.randrange(200, 9000)
        if evidence_fits(nodes, llm_view=llm_view, budget_tokens=budget, char_token_ratio=1.0):
            legible += 1
            shortened += _render_legibly(nodes, budget) > 0

    assert legible > 300
    assert shortened > 150


def test_admission_never_holds_a_match_the_answer_would_lose() -> None:
    """Seeded walk: whatever research offers, what admission keeps renders with every match it holds."""
    rng = random.Random(53)
    refused = extended = 0

    for _ in range(40):
        budget = rng.randrange(1500, 9000)
        ledger = _restricted([_random_window(rng, "seed", seam=False)], budget)
        for _ in range(60):
            node_id = f"e{rng.randrange(6)}"
            known = node_id in {scored.node.node_id for scored in ledger.entries()}
            changed = ledger.add([_random_window(rng, node_id, seam=rng.random() < 0.15)])
            refused += not changed and (known or not ledger.full)
            extended += bool(changed) and known
            nodes = ledger.numbered()
            assert evidence_fits(nodes, llm_view=llm_view, budget_tokens=budget, char_token_ratio=1.0)
            _render_legibly(nodes, budget)

    assert refused > 50
    assert extended > 100
