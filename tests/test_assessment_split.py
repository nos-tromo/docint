"""The two-part answer protocol. All text is invented."""

from __future__ import annotations

import pytest

from docint.agents.assessment.split import (
    SENTINEL,
    SentinelStreamSplitter,
    assessment_label,
    cited_sources,
    compose_labelled_answer,
    neutralize_sentinel,
)

EVIDENCE = "The study reports an F1 of 0.71 on one dataset [1]."
JUDGEMENT = "A second dataset would show whether the result generalises."
STREAM = f"{EVIDENCE}\n{SENTINEL}\n{JUDGEMENT}"


@pytest.fixture(autouse=True)
def _english(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the label language; the developer ``.env`` is German.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")


def test_compose_inserts_the_label_between_the_parts() -> None:
    """The label replaces the sentinel and the evidence part ends where it starts."""
    labelled = compose_labelled_answer(f"{EVIDENCE}\n\n{SENTINEL}\n\n{JUDGEMENT}")

    assert labelled.outcome == "split"
    assert labelled.text == f"{EVIDENCE}{assessment_label(with_rule=True)}{JUDGEMENT}"
    assert labelled.evidence_chars == len(EVIDENCE)
    assert labelled.text[: labelled.evidence_chars] == EVIDENCE


def test_the_label_is_a_rule_and_a_bold_heading() -> None:
    """Bold, not ``#``: reports nest answers under their own ``###`` headings."""
    label = assessment_label(with_rule=True)

    assert label.startswith("\n\n---\n\n**Model assessment**\n\n*")
    assert label.endswith("*\n\n")
    assert "#" not in label


@pytest.mark.parametrize(
    "marker",
    ["**[[ASSESSMENT]]**", "[[assessment]]", "[ASSESSMENT]", "## [[ASSESSMENT]]", "## **[[Assessment]]**"],
)
def test_compose_tolerates_decorated_sentinels(marker: str) -> None:
    """Models restyle markers; the boundary must survive bold, heading marks, case and brackets.

    Args:
        marker (str): A decorated sentinel variant.
    """
    labelled = compose_labelled_answer(f"{EVIDENCE}\n{marker}\n{JUDGEMENT}")

    assert labelled.outcome == "split"
    assert labelled.text == f"{EVIDENCE}{assessment_label(with_rule=True)}{JUDGEMENT}"


def test_compose_keeps_only_the_first_sentinel() -> None:
    """A repeated sentinel is noise, never a second boundary."""
    labelled = compose_labelled_answer(f"{EVIDENCE}\n{SENTINEL}\n{JUDGEMENT}\n{SENTINEL}")

    assert labelled.text.endswith(JUDGEMENT)
    assert labelled.text.count("Model assessment") == 1


def test_an_answer_without_a_sentinel_is_labelled_as_a_whole() -> None:
    """Unseparated text cannot pass as source-backed, so all of it is the model's."""
    labelled = compose_labelled_answer(EVIDENCE)

    assert labelled.outcome == "unsplit"
    assert labelled.evidence_chars == 0
    assert labelled.text == f"{assessment_label(with_rule=False)}{EVIDENCE}"
    assert "---" not in labelled.text


def test_an_empty_evidence_part_has_no_rule() -> None:
    """A rule with nothing above it would print literally in a report's answer line."""
    labelled = compose_labelled_answer(f"{SENTINEL}\n{JUDGEMENT}")

    assert labelled.outcome == "split"
    assert labelled.evidence_chars == 0
    assert labelled.text == f"{assessment_label(with_rule=False)}{JUDGEMENT}"


def test_an_empty_assessment_part_leaves_plain_evidence() -> None:
    """No judgement means no label; the whole text is source-backed."""
    labelled = compose_labelled_answer(f"{EVIDENCE}\n{SENTINEL}\n")

    assert labelled.outcome == "split"
    assert labelled.text == EVIDENCE
    assert labelled.evidence_chars is None


def test_blank_text_is_empty() -> None:
    """There is nothing to label in a blank answer."""
    labelled = compose_labelled_answer("   ")

    assert labelled.outcome == "empty"
    assert labelled.text == ""
    assert labelled.evidence_chars is None


def test_a_sentinel_only_answer_is_empty() -> None:
    """A bare sentinel carries no text on either side, so there is nothing to label."""
    labelled = compose_labelled_answer(SENTINEL)

    assert labelled.outcome == "empty"
    assert (labelled.text, labelled.evidence_chars) == ("", None)


def test_the_label_follows_the_response_language(monkeypatch: pytest.MonkeyPatch) -> None:
    """The label is localized like the rest of the answer.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setenv("RESPONSE_LANGUAGE", "de")

    assert "**Einschätzung des Modells**" in assessment_label(with_rule=True)


def test_neutralize_sentinel_defuses_document_text() -> None:
    """A passage quoting the sentinel must not be able to split the answer."""
    defused = neutralize_sentinel(f"see {SENTINEL} and **[ASSESSMENT]** here")

    assert compose_labelled_answer(defused).outcome == "unsplit"


def test_cited_sources_picks_cited_sources_in_citation_order() -> None:
    """Only the sources the evidence cites are validated, capped at the validator's six."""
    sources = [{"citation_index": index, "id": f"s{index}"} for index in range(1, 9)]

    picked = cited_sources("a [3] b [1, 2] c [7]", sources)

    assert [source["id"] for source in picked] == ["s1", "s2", "s3", "s7"]
    assert [source["id"] for source in cited_sources("a [3] b [1, 2]", sources, limit=2)] == ["s1", "s2"]


def test_cited_sources_falls_back_to_the_first_sources() -> None:
    """Evidence that cites nothing is checked against what the validator always saw."""
    sources = [{"citation_index": index} for index in range(1, 9)]

    assert len(cited_sources("no citations here", sources)) == 6


@pytest.mark.parametrize("cut", range(1, len(STREAM)))
def test_splitter_swaps_a_sentinel_split_at_any_position(cut: int) -> None:
    """A sentinel arriving across two tokens is still caught.

    Args:
        cut (int): Where the stream is split into two tokens.
    """
    splitter = SentinelStreamSplitter()

    out = splitter.feed(STREAM[:cut]) + splitter.feed(STREAM[cut:]) + splitter.flush()

    assert out == STREAM.replace(SENTINEL, assessment_label(with_rule=True))


def test_splitter_ignores_a_sentinel_inside_think() -> None:
    """Reasoning may mention the sentinel; only the answer's own one is the boundary."""
    stream = f"<think>maybe {SENTINEL} later</think>{EVIDENCE}\n{SENTINEL}\n{JUDGEMENT}"
    splitter = SentinelStreamSplitter()

    out = "".join(splitter.feed(char) for char in stream) + splitter.flush()

    expected_head = f"<think>maybe {SENTINEL} later</think>{EVIDENCE}\n"
    assert out == f"{expected_head}{assessment_label(with_rule=True)}\n{JUDGEMENT}"


def test_splitter_opens_without_a_rule_when_nothing_preceded() -> None:
    """The streamed label matches the composed one when the evidence part is empty."""
    splitter = SentinelStreamSplitter()

    out = splitter.feed(f"{SENTINEL}\n{JUDGEMENT}") + splitter.flush()

    assert out == f"{assessment_label(with_rule=False)}\n{JUDGEMENT}"


def test_splitter_holds_back_only_a_possible_sentinel_prefix() -> None:
    """Text reaches the client as soon as it cannot become a sentinel."""
    splitter = SentinelStreamSplitter()

    assert splitter.feed("Fact [[ASSESS") == "Fact "
    assert splitter.flush() == "[[ASSESS"


def test_splitter_keeps_its_indices_aligned_past_a_dotted_capital_i() -> None:
    """A dotted capital I lower-cases to two characters; the cut must still land on the sentinel."""
    text = "Istanbul \u0130 fact [1].\n[[ASSESSMENT]]\nView."
    splitter = SentinelStreamSplitter()

    out = splitter.feed(text) + splitter.flush()

    assert out == text.replace(SENTINEL, assessment_label(with_rule=True))
