"""Unit tests for the shared build_prior_turn helper."""

from docint.agents.history import build_prior_turn
from docint.agents.types import PriorTurn


def test_build_prior_turn_pairs_last_assistant_with_preceding_user() -> None:
    """A trailing assistant message is paired with the user message that triggered it."""
    history = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1 referencing UN Security Council."},
    ]
    prior = build_prior_turn(history)
    assert isinstance(prior, PriorTurn)
    assert prior.user_text == "Q1"
    assert prior.assistant_text == "A1 referencing UN Security Council."


def test_build_prior_turn_returns_none_for_empty_history() -> None:
    """Empty history yields no prior turn (first turn of the session)."""
    assert build_prior_turn([]) is None


def test_build_prior_turn_returns_none_without_assistant() -> None:
    """History with only a user message yields no prior turn."""
    assert build_prior_turn([{"role": "user", "content": "first question"}]) is None


def test_build_prior_turn_returns_none_for_blank_assistant() -> None:
    """A whitespace-only assistant message is treated as no prior turn."""
    history = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "   "},
    ]
    assert build_prior_turn(history) is None


def test_build_prior_turn_finds_last_assistant_when_newer_user_follows() -> None:
    """The tail scan pairs the most recent assistant with its preceding user, ignoring a later user message."""
    history = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
    ]
    prior = build_prior_turn(history)
    assert isinstance(prior, PriorTurn)
    assert prior.user_text == "Q1"
    assert prior.assistant_text == "A1"


def test_build_prior_turn_handles_assistant_only_history() -> None:
    """History with only an assistant message yields a prior turn with empty user text."""
    prior = build_prior_turn([{"role": "assistant", "content": "Standalone assistant note."}])
    assert isinstance(prior, PriorTurn)
    assert prior.user_text == ""
    assert prior.assistant_text == "Standalone assistant note."


def test_build_prior_turn_reads_the_evidence_boundary() -> None:
    """An assessment's history entry says where its source-backed part ends."""
    history = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "Fact [1]. View.", "agent": {"evidence_chars": 9}},
    ]

    prior = build_prior_turn(history)

    assert prior is not None
    assert prior.evidence_chars == 9


def test_evidence_only_cuts_the_assessment_part() -> None:
    """A lookup may restate its prior answer, so it sees only the sourced part."""
    prior = PriorTurn(
        user_text="Q", assistant_text="Fact [1].\n\n---\n\n**Model assessment**\n\nView.", evidence_chars=9
    )

    assert prior.evidence_only() == PriorTurn(user_text="Q", assistant_text="Fact [1].")


def test_evidence_only_keeps_a_lookup_answer_whole() -> None:
    """A lookup answer is source-backed throughout."""
    prior = PriorTurn(user_text="Q", assistant_text="Alice [1].")

    assert prior.evidence_only() is prior


def test_evidence_only_drops_an_answer_with_no_source_backed_part() -> None:
    """Nothing sourced remains, so there is no prior turn to show a lookup."""
    prior = PriorTurn(user_text="Q", assistant_text="**Model assessment**\n\nView.", evidence_chars=0)

    assert prior.evidence_only() is None
