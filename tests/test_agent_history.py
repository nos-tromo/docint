"""The conversation window an agent turn is sent, and the session read behind it. All text is invented."""

from __future__ import annotations

import random
import re
from collections.abc import Generator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from llama_index.core.base.llms.types import MessageRole
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from docint.agents.assessment.split import SENTINEL, assessment_label, compose_labelled_answer
from docint.agents.turn.budget import (
    HISTORY_ALLOWANCE_TOKENS,
    HISTORY_ALLOWANCE_TOKENS_REASONING,
    MESSAGE_OVERHEAD_TOKENS,
)
from docint.agents.turn.history import (
    ANSWER_ASSESSMENT_LABEL,
    ANSWER_SOURCED_LABEL,
    CARRIED_HEADER,
    CARRIED_MAX_PASSAGES,
    CARRIED_TOKENS,
    CITED_RANGE_MAX,
    EARLIER_QUESTIONS_TOKENS,
    MIN_EXCHANGE_TOKENS,
    NO_HISTORY,
    QUESTIONS_HEADER,
    RESOLVED_QUESTION_CHARS,
    RESOLVED_QUESTION_LABEL,
    SUMMARY_HEADER,
    SUMMARY_TOKENS,
    CarriedPassage,
    ConversationTurn,
    ConversationWindow,
    _chars_within,
    build_window,
    cited_node_ids,
    last_answered,
    question_block,
    recent_positions,
)
from docint.core.state.base import Base
from docint.core.state.conversation import Conversation
from docint.core.state.session_manager import SessionManager
from docint.utils.embed_chunking import estimate_tokens
from docint.utils.text_budget import fit_with_marker

RATIO = 3.5
PLAIN = HISTORY_ALLOWANCE_TOKENS
THINKING = HISTORY_ALLOWANCE_TOKENS_REASONING
FOLD = "Person A asked about boat counts at piers one to five; the answers gave counts only."
OPEN_QUESTION = "Which pier had the lowest footfall, and was that count adjusted for the ferry days?"


@pytest.fixture(autouse=True)
def _english(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the language of the assessment label an earlier answer carries.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")


@pytest.fixture
def session_manager() -> Generator[SessionManager, None, None]:
    """Build a SessionManager on an in-memory SQLite database with a mocked RAG.

    Returns:
        Generator[SessionManager, None, None]: The session manager.
    """
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    rag_mock = MagicMock()
    rag_mock.qdrant_collection = "test_collection"
    rag_mock.conversation_summary_prompt = "Summarize turns:\n"
    rag_mock.text_model.complete.return_value = SimpleNamespace(text=FOLD)
    cast(Any, rag_mock.get_source_by_node_id).return_value = None
    manager = SessionManager(rag=rag_mock)
    manager._SessionMaker = sessionmaker(bind=engine)
    yield manager
    engine.dispose()


def _persist(
    manager: SessionManager,
    session_id: str,
    question: str,
    answer: str,
    *,
    owner: str | None = None,
    trail: dict[str, Any] | None = None,
) -> None:
    """Store one exchange the way a finished turn does.

    Args:
        manager (SessionManager): The session manager.
        session_id (str): The session.
        question (str): The user's message.
        answer (str): The stored answer text.
        owner (str | None): The principal that owns the session.
        trail (dict[str, Any] | None): The assessment trail, if the answer was one.
    """
    resp = MagicMock()
    resp.metadata = cast(dict[str, Any], {})
    resp.source_nodes = cast(list[Any], [])
    manager._persist_turn(session_id, question, resp, {"response": answer}, owner=owner, agent_trail=trail)


def _question(number: int) -> str:
    """Return an invented question about one pier.

    Args:
        number (int): The pier.

    Returns:
        str: The question.
    """
    return f"What did the harbour survey count at pier {number}?"


def _answer(number: int) -> str:
    """Return an invented answer about one pier, with no citation marker.

    Args:
        number (int): The pier.

    Returns:
        str: The answer.
    """
    return f"The invented survey counted {number * 12} boats at pier {number}."


def _turn(number: int) -> ConversationTurn:
    """Return an exchange about one pier.

    Args:
        number (int): The pier.

    Returns:
        ConversationTurn: The exchange.
    """
    return ConversationTurn(_question(number), _answer(number), None)


def _contents(window: ConversationWindow) -> list[str]:
    """Return the text of every message the window sends.

    Args:
        window (ConversationWindow): The window.

    Returns:
        list[str]: The messages' text, oldest first.
    """
    return [str(message.content) for message in window.messages()]


def _spent(window: ConversationWindow, ratio: float = RATIO) -> int:
    """Count what the window costs a prompt: its block, its messages and their envelopes.

    Args:
        window (ConversationWindow): The window.
        ratio (float): Characters per token.

    Returns:
        int: Estimated tokens.
    """
    messages = sum(estimate_tokens(text, ratio) + MESSAGE_OVERHEAD_TOKENS for text in _contents(window))
    return estimate_tokens(window.summary_block(), ratio) + messages


def test_the_last_two_exchanges_stay_raw_right_after_a_fold(session_manager: SessionManager) -> None:
    """The fifth turn folds the summary; the turns just before it must still reach the model word for word.

    ``_get_session_context`` hands over the summary alone at that point, so the model would lose the
    exchange it is being asked to follow up on.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    for number in range(1, 6):
        _persist(session_manager, "s1", _question(number), _answer(number))
    session_manager._maybe_update_summary("s1")
    turns, summary = session_manager.get_conversation("s1", owner=None)
    assert summary == FOLD

    window = build_window(turns, summary, allowance_tokens=PLAIN, char_token_ratio=RATIO)

    assert [message.role for message in window.messages()] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.USER,
        MessageRole.ASSISTANT,
    ]
    assert _contents(window) == [_question(4), _answer(4), _question(5), _answer(5)]
    assert FOLD in window.summary_block()
    assert window.earlier_questions == (_question(3), _question(2), _question(1))


@pytest.mark.parametrize("allowance", [PLAIN, THINKING], ids=["plain", "thinking"])
def test_an_open_question_five_turns_back_reaches_the_model(allowance: int) -> None:
    """A question the answers left open is still in front of the model when the user comes back to it.

    Args:
        allowance (int): The history allowance, plain or a thinking model's.
    """
    turns = [_turn(1), ConversationTurn(OPEN_QUESTION, "The counts are in the annex.", None), *map(_turn, (3, 4, 5, 6))]

    window = build_window(turns, "", allowance_tokens=allowance, char_token_ratio=RATIO)

    assert OPEN_QUESTION in window.earlier_questions
    assert OPEN_QUESTION in window.summary_block()
    assert OPEN_QUESTION not in _contents(window)


def test_long_recent_answers_do_not_crowd_out_an_open_question() -> None:
    """The earlier questions claim their small budget first, so a long answer is what gets cut."""
    long_answer = "The invented survey counted boats at the piers and listed every figure. " * 80
    turns = [
        ConversationTurn(OPEN_QUESTION, "The counts are in the annex.", None),
        ConversationTurn(_question(2), long_answer, None),
        ConversationTurn(_question(3), long_answer, None),
        ConversationTurn(_question(4), long_answer, None),
    ]

    window = build_window(turns, "", allowance_tokens=PLAIN, char_token_ratio=RATIO)

    assert OPEN_QUESTION in window.earlier_questions
    assert any("characters not shown" in text for text in _contents(window))
    assert _spent(window) <= PLAIN


@pytest.mark.parametrize(
    "marker",
    ["[3]", "[1, 2]", "[2\N{EN DASH}4]", "[1;2]", "[3-5]", "[ 4 ]"],
    ids=["single", "list", "en-dash-range", "semicolon", "hyphen-range", "padded"],
)
def test_citation_markers_are_stripped_from_earlier_answers(marker: str) -> None:
    """Markers number an earlier turn's sources; kept, they would be cited as this turn's.

    Everything else survives: other brackets, the user's own words and the line structure.

    Args:
        marker (str): A bracketed citation the earlier answer carries.
    """
    answer = f"The survey counted 48 boats at pier 4 {marker}.\n- A second point keeps its [draft] tag."
    turns = [ConversationTurn("Tell me more about [2]", answer, None)]

    window = build_window(turns, "", allowance_tokens=PLAIN, char_token_ratio=RATIO)

    assert _contents(window) == [
        "Tell me more about [2]",
        "The survey counted 48 boats at pier 4.\n- A second point keeps its [draft] tag.",
    ]


def test_citation_markers_are_stripped_from_the_summary_too() -> None:
    """A summary restates earlier answers, markers included, and they are as stale there."""
    window = build_window(
        [], "Person A asked about pier 4 [1]; the count was 48 [2, 3].", allowance_tokens=PLAIN, char_token_ratio=RATIO
    )

    assert window.summary == "Person A asked about pier 4; the count was 48."


def _fold(number: int) -> str:
    """Return one invented fold of a rolling summary, a line of about 470 characters.

    Args:
        number (int): Which fold.

    Returns:
        str: The fold.
    """
    return f"Fold {number}: " + "Person A asked about boat counts and the answers gave the figures. " * 6


def test_the_summary_keeps_its_newest_part_within_budget() -> None:
    """Every fold appends to the summary, so the end is the newest part; the start is what gives way."""
    summary = "\n".join(_fold(number) for number in (1, 2, 3))

    window = build_window([_turn(1)], summary, allowance_tokens=PLAIN, char_token_ratio=RATIO)

    assert _fold(3).strip() in window.summary
    assert _fold(1)[:20] not in window.summary
    assert _fold(2)[:20] not in window.summary
    first_line = window.summary.splitlines()[0]
    assert first_line.startswith("[... ") and first_line.endswith("characters not shown ...]")
    section = f"{SUMMARY_HEADER}:\n{window.summary}"
    assert 0 < estimate_tokens(section, RATIO) <= SUMMARY_TOKENS


def test_a_summary_that_fits_is_kept_whole_and_unmarked() -> None:
    """Nothing is cut, so nothing is marked."""
    window = build_window([_turn(1)], f"  {FOLD}\n", allowance_tokens=PLAIN, char_token_ratio=RATIO)

    assert window.summary == FOLD


def test_a_foreign_session_yields_no_history(session_manager: SessionManager) -> None:
    """Another principal's session, and an unknown one, read as no conversation at all.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    _persist(session_manager, "mine", _question(1), _answer(1), owner="alice")
    _persist(session_manager, "legacy", _question(2), _answer(2))
    with session_manager._session_scope() as session:
        conversation = session.get(Conversation, "mine")
        assert conversation is not None
        conversation.rolling_summary = FOLD
        session.commit()

    assert session_manager.get_conversation("mine", owner="mallory") == ([], "")
    assert session_manager.get_conversation("mine", owner=None) == ([], "")
    assert session_manager.get_conversation("missing", owner="alice") == ([], "")
    assert session_manager.get_conversation("legacy", owner="alice") == ([], "")
    assert session_manager.get_conversation("mine", owner="alice") == ([_turn(1)], FOLD)
    assert session_manager.get_conversation("legacy", owner=None) == ([_turn(2)], "")


def test_get_conversation_reads_the_turns_oldest_first_with_their_evidence_boundary(
    session_manager: SessionManager,
) -> None:
    """The window needs every turn's text and where an assessment's source-backed part ends, and no citations.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    labelled = compose_labelled_answer(f"Fact one [1].\n{SENTINEL}\nMy view.")
    trail: dict[str, Any] = {"steps": 1, "actions": [], "reason": None, "answer": "split"}
    trail["evidence_chars"] = labelled.evidence_chars
    _persist(session_manager, "s1", f"  {_question(1)}\n", f"{_answer(1)}  ")
    _persist(session_manager, "s1", _question(2), labelled.text, trail=trail)

    turns, summary = session_manager.get_conversation("s1", owner=None)

    assert turns == [_turn(1), ConversationTurn(_question(2), labelled.text, labelled.evidence_chars)]
    assert labelled.evidence_chars == len("Fact one [1].")
    assert summary == ""
    cast(Any, session_manager.rag.get_source_by_node_id).assert_not_called()


FOLLOW_UP = "Should we proceed?"
#: An invented rewrite that recovered a budget set two turns earlier.
RESOLVED = "Should we proceed with the invented pilot deployment within the €10,000 budget?"
#: An invented message longer than the resolved line's cap.
LONG_QUESTION = ("Assess the invented harbour survey's sampling plan, its staffing and its timeline. " * 20).strip()


@pytest.mark.parametrize(
    "resolved",
    [None, "", " \n\t ", "should we PROCEED", "Should   we\nproceed?!", "<think>the user means it</think>"],
    ids=["none", "empty", "blank", "same-words-other-case", "same-words-other-spacing", "only-reasoning"],
)
def test_a_rewrite_that_adds_no_words_leaves_the_question_alone(resolved: str | None) -> None:
    """Nothing resolved, or the same words up to case, spacing and punctuation, is the user's message, stripped.

    Args:
        resolved (str | None): The search rewrite, if any.
    """
    assert question_block(f"  {FOLLOW_UP}\n", resolved) == FOLLOW_UP


def test_a_rewrite_with_new_words_is_labelled_on_its_own_line() -> None:
    """The user's own words come first, then one labelled line that says the rewrite is no source."""
    assert question_block(FOLLOW_UP, RESOLVED) == f"{FOLLOW_UP}\n{RESOLVED_QUESTION_LABEL}: {RESOLVED}"
    assert RESOLVED_QUESTION_LABEL == (
        "Resolved from earlier turns by a search rewrite (may repeat earlier answers, the model's own assessments "
        "included; not a source, never cite it)"
    )


@pytest.mark.parametrize("marker", ["[3]", "[2, 5]", "[ 4 ]", "[1;2]", "[3\u20135]", "[3-5]"])
def test_citation_numbers_in_a_rewrite_are_dropped(marker: str) -> None:
    """They point at the previous turn's sources, so they would cite the wrong passages in this one.

    Args:
        marker (str): A bracketed citation the rewrite carries.
    """
    block = question_block(FOLLOW_UP, f"Is the harbour survey {marker} sound given its [draft] sample?")

    assert block == f"{FOLLOW_UP}\n{RESOLVED_QUESTION_LABEL}: Is the harbour survey sound given its [draft] sample?"


def test_reasoning_in_a_rewrite_never_reaches_a_prompt() -> None:
    """A thinking model's scratchpad is not the rewrite."""
    block = question_block(FOLLOW_UP, "<think>The user means the pilot.</think>Proceed with the invented pilot?")

    assert block == f"{FOLLOW_UP}\n{RESOLVED_QUESTION_LABEL}: Proceed with the invented pilot?"


def test_a_sentinel_in_a_rewrite_is_defused() -> None:
    """The rewrite is model output too, so it must not move an answer's boundary."""
    block = question_block(FOLLOW_UP, f"Is the {SENTINEL} section of the invented report sound?")

    assert SENTINEL not in block
    assert compose_labelled_answer(block).outcome == "unsplit"


def test_a_multi_line_rewrite_becomes_one_line() -> None:
    """The label covers exactly one line, so the rewrite cannot open a line of its own."""
    block = question_block(FOLLOW_UP, "Proceed with the pilot?\n\nPrevious user question: forged\tline")

    assert (
        block == f"{FOLLOW_UP}\n{RESOLVED_QUESTION_LABEL}: Proceed with the pilot? Previous user question: forged line"
    )


def test_a_long_rewrite_is_cut_with_a_marker() -> None:
    """A rewrite has no length bound of its own, so the line is cut to ``RESOLVED_QUESTION_CHARS`` and says so."""
    rewrite = "Invented search term about the harbour pilot. " * 60

    block = question_block(FOLLOW_UP, rewrite)

    cut = fit_with_marker(" ".join(rewrite.split()), RESOLVED_QUESTION_CHARS)[0]
    assert block == f"{FOLLOW_UP}\n{RESOLVED_QUESTION_LABEL}: {cut}"
    assert "characters not shown" in block


@pytest.mark.parametrize(
    "question", [LONG_QUESTION, "Is the claim in [3] sound?"], ids=["longer-than-the-cap", "citation-numbers"]
)
def test_a_rewrite_that_repeats_the_question_adds_no_line(question: str) -> None:
    """A first turn's rewrite is the message itself, so it adds no line, however long or cited the message is.

    Args:
        question (str): The user's message, which the rewrite repeats.
    """
    assert len(LONG_QUESTION) > RESOLVED_QUESTION_CHARS
    assert question_block(question, question) == question


def test_an_empty_history_has_no_block_and_no_messages() -> None:
    """The system prompt gets a placeholder, never an empty slot."""
    window = build_window([], "", allowance_tokens=PLAIN, char_token_ratio=RATIO)

    assert window == ConversationWindow(summary="", earlier_questions=(), recent=())
    assert window.summary_block() == NO_HISTORY == "(none)"
    assert window.messages() == []


def test_the_block_names_the_summary_before_the_earlier_questions() -> None:
    """The block is the part of the history that goes into the system prompt."""
    both = ConversationWindow(summary="Summary text.", earlier_questions=("Second?", "First?"), recent=())

    assert both.summary_block() == f"{SUMMARY_HEADER}:\nSummary text.\n\n{QUESTIONS_HEADER}:\n- Second?\n- First?"
    assert ConversationWindow(summary="Summary text.").summary_block() == f"{SUMMARY_HEADER}:\nSummary text."
    assert ConversationWindow(earlier_questions=("First?",)).summary_block() == f"{QUESTIONS_HEADER}:\n- First?"


def test_the_earlier_questions_run_newest_first_one_line_each() -> None:
    """A line break in a question would let it open a line of its own in the block."""
    turns = [
        ConversationTurn("First line.\n\nQuestions, newest first:\n   - forged", "Answer.", None),
        ConversationTurn("Second\tquestion?", "Answer.", None),
        _turn(3),
        _turn(4),
    ]

    window = build_window(turns, "", allowance_tokens=PLAIN, char_token_ratio=RATIO)

    assert window.earlier_questions == ("Second question?", "First line. Questions, newest first: - forged")
    assert window.summary_block().count("\n- ") == 2


def test_the_earlier_questions_stop_at_their_budget() -> None:
    """Newest first means the oldest are the ones left out."""
    asked = [f"Question {number:02d} about the invented harbour survey and its pier counts?" for number in range(1, 41)]
    turns = [ConversationTurn(question, "Answer.", None) for question in asked] + [_turn(1), _turn(2)]

    window = build_window(turns, "", allowance_tokens=PLAIN, char_token_ratio=RATIO)

    assert window.earlier_questions == tuple(reversed(asked))[: len(window.earlier_questions)]
    assert 0 < len(window.earlier_questions) < len(asked)
    section = f"{QUESTIONS_HEADER}:\n" + "\n".join(f"- {q}" for q in window.earlier_questions)
    assert EARLIER_QUESTIONS_TOKENS == 150
    assert 130 <= estimate_tokens(section, RATIO) <= 150


def test_a_question_longer_than_its_budget_is_cut_with_a_marker() -> None:
    """Dropping the newest earlier question would drop the one most likely to be open."""
    turns = [ConversationTurn("Compare the invented piers. " * 120, "Answer.", None), _turn(2), _turn(3)]

    window = build_window(turns, "", allowance_tokens=PLAIN, char_token_ratio=RATIO)

    assert len(window.earlier_questions) == 1
    assert window.earlier_questions[0].startswith("Compare the invented piers.")
    assert "characters not shown" in window.earlier_questions[0]
    assert estimate_tokens(f"{QUESTIONS_HEADER}:\n- {window.earlier_questions[0]}", RATIO) <= EARLIER_QUESTIONS_TOKENS


@pytest.mark.parametrize(
    ("allowance", "kept"),
    [(PLAIN, 2), (PLAIN - 1, 1), (THINKING, 1), (2 * PLAIN, 2)],
    ids=["plain", "just-under-plain", "thinking", "double"],
)
def test_a_smaller_allowance_keeps_one_exchange_raw(allowance: int, kept: int) -> None:
    """The exchange that gives way is still asked about, as an earlier question.

    Args:
        allowance (int): The history allowance.
        kept (int): How many exchanges stay as messages.
    """
    turns = [_turn(number) for number in range(1, 6)]

    window = build_window(turns, "", allowance_tokens=allowance, char_token_ratio=RATIO)

    assert len(window.recent) == kept
    assert _contents(window)[-2:] == [_question(5), _answer(5)]
    assert window.earlier_questions[0] == _question(5 - kept)


def test_a_lookup_answer_is_sent_as_it_was_written() -> None:
    """Only an assessment is labelled; an ordinary answer carries nothing the model did not write."""
    window = build_window([_turn(1)], "", allowance_tokens=PLAIN, char_token_ratio=RATIO)

    assert _contents(window) == [_question(1), _answer(1)]
    assert window.recent == (_turn(1),)


SOURCED = "The invented survey counted 48 boats at pier 4 [1] and 52 at pier 5 [2, 3]."
JUDGEMENT = "Rerun the count with a second cohort [2] before relying on it."


def _assessment(sourced: str = SOURCED, judgement: str = JUDGEMENT) -> ConversationTurn:
    """Store an invented assessment the way a finished turn persists it.

    Args:
        sourced (str): The source-backed part.
        judgement (str): The model's own assessment.

    Returns:
        ConversationTurn: The exchange, with the boundary between its parts.
    """
    labelled = compose_labelled_answer(f"{sourced}\n{SENTINEL}\n{judgement}")
    return ConversationTurn("Is the count sound?", labelled.text, labelled.evidence_chars)


def test_an_earlier_assessment_shows_its_two_parts_labelled() -> None:
    """The model must tell what the sources said from what an earlier turn only judged."""
    window = build_window([_assessment(), _turn(2)], "", allowance_tokens=PLAIN, char_token_ratio=RATIO)

    sourced = "The invented survey counted 48 boats at pier 4 and 52 at pier 5."
    judgement = f"{assessment_label(with_rule=False)}Rerun the count with a second cohort before relying on it."
    assert _contents(window)[1] == f"{ANSWER_SOURCED_LABEL}: {sourced}\n{ANSWER_ASSESSMENT_LABEL}: {judgement}"
    assert "[n]" not in ANSWER_SOURCED_LABEL
    assert [turn.evidence_chars for turn in window.recent] == [None, None]


def test_a_long_source_backed_part_cannot_push_the_judgement_out() -> None:
    """The two parts share the message's room shortest first, so the judgement keeps its share."""
    sourced = "The invented survey interviewed respondents at three piers. " * 120

    window = build_window([_assessment(sourced), _turn(2)], "", allowance_tokens=PLAIN, char_token_ratio=RATIO)

    reply = _contents(window)[1]
    assert "characters not shown" in reply
    assert reply.index(ANSWER_SOURCED_LABEL) < reply.index(ANSWER_ASSESSMENT_LABEL)
    assert "Rerun the count with a second cohort before relying on it." in reply
    assert len(reply) > 1800
    assert _spent(window) <= PLAIN


@pytest.mark.parametrize(
    "cut", [-5, 10_000, 3, None], ids=["negative", "past-the-end", "off-the-rule", "wraps-to-the-rule"]
)
def test_a_boundary_that_does_not_sit_on_the_rule_is_not_trusted(cut: int | None) -> None:
    """An answer whose stored boundary is wrong goes in whole, not split at a guess.

    Args:
        cut (int | None): A stored boundary that does not fall on the assessment rule; ``None`` stands
            for the negative one that counts back from the end onto the rule.
    """
    stored = _assessment()
    boundary = (stored.evidence_chars or 0) - len(stored.assistant_text) if cut is None else cut
    turn = ConversationTurn(stored.user_text, stored.assistant_text, boundary)

    window = build_window([turn], "", allowance_tokens=PLAIN, char_token_ratio=RATIO)

    assert ANSWER_SOURCED_LABEL not in _contents(window)[1]
    assert "Rerun the count with a second cohort before relying on it." in _contents(window)[1]


def test_an_exchange_without_an_answer_is_an_open_question_not_a_message() -> None:
    """A message pair needs both sides; a question nobody answered is exactly what must not be lost."""
    turns = [_turn(1), _turn(2), ConversationTurn("Which ferry days were left out?", "", None)]

    window = build_window(turns, "", allowance_tokens=PLAIN, char_token_ratio=RATIO)

    assert _contents(window) == [_question(1), _answer(1), _question(2), _answer(2)]
    assert window.earlier_questions == ("Which ferry days were left out?",)


def test_identical_exchanges_are_kept_by_position_not_by_value() -> None:
    """A user who asks the same thing again gets the same answer; neither turn may hide the other."""
    window = build_window([_turn(1), _turn(1), _turn(1)], "", allowance_tokens=PLAIN, char_token_ratio=RATIO)

    assert len(window.recent) == 2
    assert window.earlier_questions == (_question(1),)


def _heavy(count: int) -> list[ConversationTurn]:
    """Return exchanges whose questions and answers are all long, with citation markers.

    Args:
        count (int): How many exchanges.

    Returns:
        list[ConversationTurn]: The exchanges, oldest first.
    """
    return [
        ConversationTurn(
            f"Question {number}. " + "Which pier was counted, and how? " * 12,
            f"Answer {number}. " + "The invented survey counted boats at the piers [1]. " * 60,
            None,
        )
        for number in range(1, count + 1)
    ]


HEAVY_SUMMARY = "\n".join(_fold(number) for number in range(1, 8))


@pytest.mark.parametrize("ratio", [3.5, 2.0, 3.3, 2.3], ids=["english", "dense", "fractional", "rounding"])
@pytest.mark.parametrize("allowance", [100, 300, 500, THINKING, PLAIN, 2000])
def test_the_window_fits_its_allowance_text_and_envelopes_together(allowance: int, ratio: float) -> None:
    """The allowance counts the block, every message's text and the envelope each message costs.

    It also spends it: a window far under its allowance would pass the bound by saying nothing.

    Args:
        allowance (int): The history allowance.
        ratio (float): Characters per token.
    """
    window = build_window(_heavy(12), HEAVY_SUMMARY, allowance_tokens=allowance, char_token_ratio=ratio)

    assert _spent(window, ratio) <= allowance
    assert _spent(window, ratio) >= 0.9 * allowance


@pytest.mark.parametrize("allowance", [100, 300, THINKING, PLAIN])
def test_a_short_history_fits_its_allowance_without_a_cut(allowance: int) -> None:
    """Room to spare means nothing is cut and nothing is marked.

    Args:
        allowance (int): The history allowance.
    """
    window = build_window([_turn(1), _turn(2), _turn(3)], FOLD, allowance_tokens=allowance, char_token_ratio=RATIO)

    assert _spent(window) <= allowance
    assert "characters not shown" not in window.summary_block() + "".join(_contents(window))


def test_each_message_pays_its_envelope_inside_the_allowance() -> None:
    """Four messages at 40 tokens each are 160 tokens the text may not use."""
    turns = [ConversationTurn("Q" * 400, "A" * 4000, None), ConversationTurn("Q" * 400, "A" * 4000, None)]

    window = build_window(turns, "", allowance_tokens=PLAIN, char_token_ratio=RATIO)

    text_tokens = sum(estimate_tokens(text, RATIO) for text in _contents(window))
    assert text_tokens <= PLAIN - 4 * MESSAGE_OVERHEAD_TOKENS - estimate_tokens(NO_HISTORY, RATIO)
    assert text_tokens >= 0.9 * (PLAIN - 4 * MESSAGE_OVERHEAD_TOKENS)


def test_older_context_cannot_take_the_room_an_exchange_is_worth_sending_for() -> None:
    """At 500 tokens the questions and the summary could take it all; the exchange keeps its floor first."""
    window = build_window(_heavy(12), HEAVY_SUMMARY, allowance_tokens=500, char_token_ratio=RATIO)

    assert len(window.recent) == 1
    assert sum(estimate_tokens(text, RATIO) for text in _contents(window)) >= MIN_EXCHANGE_TOKENS - 5


def test_messages_that_each_round_up_still_fit_the_allowance() -> None:
    """Every message is estimated on its own, so the window leaves a token per message for the rounding.

    These lengths made the window one token over its allowance when the rounding went unpaid.
    """
    turns = [
        ConversationTurn("q" * 119, "a", None),
        ConversationTurn("q" * 163, "a", None),
        ConversationTurn("q" * 255, "a" * 1525, None),
        ConversationTurn("q" * 27, "a" * 1594, None),
    ]

    window = build_window(turns, "", allowance_tokens=PLAIN, char_token_ratio=3.3)

    assert len(window.recent) == 2
    assert _spent(window, 3.3) <= PLAIN


def _prose(rng: random.Random, words: int) -> str:
    """Return invented prose of a given number of words.

    Args:
        rng (random.Random): The source of choices.
        words (int): How many words.

    Returns:
        str: Space-separated words.
    """
    vocabulary = ["pier", "harbour", "survey", "boats", "count", "ferry", "annex", "Person", "A", "asked", "the"]
    return " ".join(rng.choice(vocabulary) for _ in range(words))


def test_random_histories_never_overspend_their_allowance() -> None:
    """Whatever the lengths, ratio and allowance, the window costs no more than it was given."""
    rng = random.Random(3)
    for _ in range(800):
        ratio = rng.choice([3.5, 2.0, 3.3, 2.3, 4.0, 1.7])
        allowance = rng.choice([150, 300, 500, THINKING, 1000, PLAIN, 2000])
        turns = [
            ConversationTurn(_prose(rng, rng.randrange(1, 60)) + "?", _prose(rng, rng.randrange(1, 400)), None)
            for _ in range(rng.randrange(0, 9))
        ]
        summary = "\n".join(_prose(rng, rng.randrange(5, 120)) for _ in range(rng.randrange(0, 8)))

        window = build_window(turns, summary, allowance_tokens=allowance, char_token_ratio=ratio)

        assert _spent(window, ratio) <= allowance
        assert estimate_tokens(window.summary_block(), ratio) <= EARLIER_QUESTIONS_TOKENS + SUMMARY_TOKENS


def test_a_summary_without_line_breaks_is_cut_at_a_word_to_fill_its_budget() -> None:
    """A single long paragraph has no line to start from, so the budget is filled from its last full word."""
    window = build_window([_turn(1)], "word " * 600, allowance_tokens=PLAIN, char_token_ratio=RATIO)

    marker, _, tail = window.summary.partition("\n")
    assert marker.startswith("[... ") and marker.endswith("characters not shown ...]")
    assert set(tail.split()) == {"word"} and tail.startswith("word word") and tail.endswith("word")
    assert SUMMARY_TOKENS == 250
    assert 235 <= estimate_tokens(f"{SUMMARY_HEADER}:\n{window.summary}", RATIO) <= 250


def test_a_zero_allowance_yields_an_empty_window() -> None:
    """History gives way completely before the window raises or overspends."""
    window = build_window(_heavy(6), HEAVY_SUMMARY, allowance_tokens=0, char_token_ratio=RATIO)

    assert window == ConversationWindow()


def test_a_non_positive_ratio_is_refused() -> None:
    """The estimate has no meaning without a positive characters-per-token ratio."""
    with pytest.raises(ValueError, match="char_token_ratio"):
        build_window([_turn(1)], FOLD, allowance_tokens=PLAIN, char_token_ratio=0.0)


def test_a_text_at_the_character_budget_never_estimates_over_the_token_budget() -> None:
    """At a ratio of 2.3, ``int(30 * 2.3)`` characters divide to 30.000000000000004 tokens, which round up to 31.

    The budget must also not be wasted: it is within a character of the longest text that fits.
    """
    rng = random.Random(7)
    pairs = [(30, 2.3), (250, 2.3), (460, 2.3)]
    pairs += [(rng.randrange(1, 3000), round(rng.uniform(1.0, 6.0), 1)) for _ in range(3000)]
    for tokens, ratio in pairs:
        chars = _chars_within(tokens, ratio)

        assert estimate_tokens("x" * chars, ratio) <= tokens
        assert estimate_tokens("x" * (chars + 2), ratio) > tokens


def test_a_budget_that_is_not_positive_allows_no_characters() -> None:
    """A negative room, as a crowded allowance leaves, must read as none rather than slice from the end."""
    assert _chars_within(0, RATIO) == _chars_within(-5, RATIO) == 0


def _passage(number: int) -> CarriedPassage:
    """Return an invented carried passage with a label of 110 characters.

    Args:
        number (int): Which passage.

    Returns:
        CarriedPassage: The passage, its node named ``n<number>``.
    """
    return CarriedPassage(f"n{number}", f"Bericht {number}: " + " ".join(["Pegelstände der Anleger."] * 4))


def _cited(answer: str, *sources: str | None) -> ConversationTurn:
    """Return an exchange whose answer cites stored passages.

    Args:
        answer (str): The answer, markers included.
        *sources (str | None): The node each citation number names, in order.

    Returns:
        ConversationTurn: The exchange.
    """
    return ConversationTurn("Was zeigen die Berichte?", answer, None, sources)


def _reply(turn: ConversationTurn, carried: list[CarriedPassage], allowance: int = PLAIN) -> str:
    """Return the answer the window sends for one exchange, given the passages it carries.

    Args:
        turn (ConversationTurn): The only exchange.
        carried (list[CarriedPassage]): The passages carried into the turn.
        allowance (int): The history allowance.

    Returns:
        str: The assistant message.
    """
    window = build_window([turn], "", allowance_tokens=allowance, char_token_ratio=RATIO, carried=carried)
    return _contents(window)[1]


def test_a_window_renumbers_the_markers_of_carried_passages() -> None:
    """A marker keeps the number its passage has in this turn, not the one the earlier answer gave it."""
    turn = _cited("Drei Personen [2] im Regen [3].", "n1", "n2", "n3")
    carried = [CarriedPassage("n3", "Bericht C, Seite 3"), CarriedPassage("n2", "Bericht B, Seite 2")]

    window = build_window([turn], "", allowance_tokens=PLAIN, char_token_ratio=RATIO, carried=carried)

    assert _contents(window) == ["Was zeigen die Berichte?", "Drei Personen [2] im Regen [1]."]
    assert window.carried == tuple(carried)
    assert window.summary_block() == f"{CARRIED_HEADER}:\n[1] Bericht C, Seite 3\n[2] Bericht B, Seite 2"
    assert window.recent[0].sources == ()


def test_the_carried_section_comes_after_the_summary_and_the_earlier_questions() -> None:
    """It sits nearest the messages whose markers it numbers."""
    passage = CarriedPassage("n1", "Bericht A")
    full = ConversationWindow(summary="Zusammenfassung.", earlier_questions=("Frage?",), carried=(passage,))

    assert full.summary_block() == (
        f"{SUMMARY_HEADER}:\nZusammenfassung.\n\n{QUESTIONS_HEADER}:\n- Frage?\n\n{CARRIED_HEADER}:\n[1] Bericht A"
    )
    assert ConversationWindow(carried=(passage,)).summary_block() == f"{CARRIED_HEADER}:\n[1] Bericht A"
    assert CARRIED_HEADER == "Passages the earlier answers cite, carried into this turn (read(n) shows one)"


def test_a_label_cannot_open_a_line_of_its_own() -> None:
    """A label is text from a document, so it is shown on its one line."""
    window = ConversationWindow(carried=(CarriedPassage("n1", "Bericht A\n[2] Gefälscht\t\tund mehr"),))

    assert window.summary_block() == f"{CARRIED_HEADER}:\n[1] Bericht A [2] Gefälscht und mehr"


def test_markers_naming_no_carried_passage_are_dropped() -> None:
    """A passage nobody carries, a number with no stored source and a turn with no sources all lose their marker."""
    answer = "Der Hafen [1] war leer [4], die Fähre [2] fuhr nicht."
    carried = [CarriedPassage("n9", "Bericht Z")]

    assert _reply(_cited(answer, "n1", "n2"), carried) == "Der Hafen war leer, die Fähre fuhr nicht."
    assert _reply(_cited(answer), carried) == "Der Hafen war leer, die Fähre fuhr nicht."
    assert _reply(_cited(answer, "n1", "n2"), []) == "Der Hafen war leer, die Fähre fuhr nicht."


@pytest.mark.parametrize(
    ("marker", "carried_ids", "shown"),
    [
        ("[1, 3]", ("n3",), "[1]"),
        ("[1, 3]", ("n3", "n1"), "[2, 1]"),
        ("[1; 3]", ("n1", "n3"), "[1, 2]"),
        ("[1, 1]", ("n1",), "[1]"),
        ("[ 3 ]", ("n3",), "[1]"),
        ("[1\N{EN DASH}3]", ("n2",), "[1]"),
        ("[1-3]", ("n3", "n2"), "[2, 1]"),
    ],
    ids=["one-of-two", "order-of-appearance", "semicolon", "repeated", "padded", "range", "range-two-carried"],
)
def test_a_list_marker_keeps_only_its_carried_numbers(marker: str, carried_ids: tuple[str, ...], shown: str) -> None:
    """A list or range becomes one marker of the numbers it names that are carried, each once, in the order written.

    Args:
        marker (str): The marker the earlier answer wrote.
        carried_ids (tuple[str, ...]): The nodes carried, in the order the window lists them.
        shown (str): The marker the window sends.
    """
    carried = [CarriedPassage(node_id, f"Bericht {node_id}") for node_id in carried_ids]

    assert _reply(_cited(f"Zwei Orte {marker}.", "n1", "n2", "n3"), carried) == f"Zwei Orte {shown}."


def test_a_renumbered_marker_keeps_the_blanks_before_it() -> None:
    """Only the marker is rewritten, so the answer's spacing is the model's own."""
    turn = _cited("Ohne Blank[2], mit Tab \t[2].", "n1", "n2")

    assert _reply(turn, [CarriedPassage("n2", "Bericht B")]) == "Ohne Blank[1], mit Tab \t[1]."


@pytest.mark.parametrize(
    ("last", "carried_id", "shown"),
    [
        (CITED_RANGE_MAX, "n5", "Alle [1] zusammen."),
        (CITED_RANGE_MAX + 1, "n5", "Alle zusammen."),
        (CITED_RANGE_MAX + 1, "n1", "Alle [1] zusammen."),
        (CITED_RANGE_MAX + 1, f"n{CITED_RANGE_MAX + 1}", "Alle [1] zusammen."),
    ],
    ids=["widest-read-in-full", "wider-keeps-only-its-ends", "first-end", "last-end"],
)
def test_a_range_marker_covers_its_numbers_up_to_the_cap(last: int, carried_id: str, shown: str) -> None:
    """A range of ``CITED_RANGE_MAX`` numbers names each of them; a wider one names only its two ends.

    Args:
        last (int): The range's upper end, its lower end being 1.
        carried_id (str): The one node carried.
        shown (str): The answer the window sends.
    """
    sources = tuple(f"n{number}" for number in range(1, 20))

    assert _reply(_cited(f"Alle [1\N{EN DASH}{last}] zusammen.", *sources), [CarriedPassage(carried_id, "B")]) == shown


def test_a_range_written_backwards_names_the_same_numbers() -> None:
    """The model wrote ``[5-3]``; it cites 3 to 5 whichever way round it spelled it."""
    sources = tuple(f"n{number}" for number in range(1, 8))

    assert _reply(_cited("Reihe [5-3].", *sources), [CarriedPassage("n4", "B")]) == "Reihe [1]."


def test_both_parts_of_an_earlier_assessment_are_renumbered() -> None:
    """The source-backed part and the model's own assessment cite the same turn's sources."""
    labelled = compose_labelled_answer(f"Fakt [1] und Fakt [2].\n{SENTINEL}\nMeine Einschätzung [2], nicht [1].")
    turn = ConversationTurn("Stimmt das?", labelled.text, labelled.evidence_chars, ("n1", "n2"))

    window = build_window(
        [turn], "", allowance_tokens=PLAIN, char_token_ratio=RATIO, carried=[CarriedPassage("n2", "Bericht B")]
    )

    judgement = f"{assessment_label(with_rule=False)}Meine Einschätzung [1], nicht."
    assert _contents(window)[1] == f"{ANSWER_SOURCED_LABEL}: Fakt und Fakt [1].\n{ANSWER_ASSESSMENT_LABEL}: {judgement}"


def test_two_answers_naming_one_passage_by_different_numbers_show_the_same_number() -> None:
    """The number is the passage's in this turn, whichever exchange cited it and however it numbered it."""
    first = _cited("Ort A [1].", "n7", "n8")
    second = _cited("Ort A [2].", "n8", "n7")

    window = build_window(
        [first, second],
        "",
        allowance_tokens=PLAIN,
        char_token_ratio=RATIO,
        carried=[CarriedPassage("n7", "Bericht G")],
    )

    assert [turn.assistant_text for turn in window.recent] == ["Ort A [1].", "Ort A [1]."]


def test_the_summary_and_the_earlier_questions_never_carry_numbers() -> None:
    """Only the recent answers are renumbered; the rest of the history stays as plain as before."""
    turns = [ConversationTurn("Was steht in [2]?", "Nichts.", None, ("n1", "n2")), _turn(2), _turn(3)]

    window = build_window(
        turns,
        "Person A fragte nach dem Pier [1].",
        allowance_tokens=PLAIN,
        char_token_ratio=RATIO,
        carried=[CarriedPassage("n1", "Bericht A")],
    )

    assert window.summary == "Person A fragte nach dem Pier."
    assert window.earlier_questions == ("Was steht in [2]?",)


def test_carried_lines_that_do_not_fit_are_not_carried() -> None:
    """The oldest give way: the list is the newest run that fits, and the markers of the others go."""
    passages = [_passage(number) for number in range(1, 7)]
    answer = "Erstens [1], zweitens [2], drittens [3], viertens [4], fünftens [5], sechstens [6]."
    turn = _cited(answer, *(passage.node_id for passage in passages))

    window = build_window([turn], "", allowance_tokens=380, char_token_ratio=RATIO, carried=passages)

    spare = 380 - 2 * MESSAGE_OVERHEAD_TOKENS - MIN_EXCHANGE_TOKENS
    assert window.carried == tuple(passages[-2:])
    assert window.summary_block().splitlines()[1:] == [f"[1] {passages[4].label}", f"[2] {passages[5].label}"]
    assert _contents(window)[1] == "Erstens, zweitens, drittens, viertens, fünftens [1], sechstens [2]."
    assert estimate_tokens(ConversationWindow(carried=window.carried).summary_block(), RATIO) <= spare
    assert estimate_tokens(ConversationWindow(carried=tuple(passages[-3:])).summary_block(), RATIO) > spare


def test_a_line_that_does_not_fit_ends_the_run_even_when_an_older_one_would() -> None:
    """The run stops at the first line from the newest that does not fit; a short passage behind it is left out."""
    passages = [CarriedPassage("n1", "Kurz"), CarriedPassage("n2", "x" * 400), CarriedPassage("n3", "Kurz")]
    turn = _cited("Eins [1], zwei [2], drei [3].", "n1", "n2", "n3")

    window = build_window([turn], "", allowance_tokens=380, char_token_ratio=RATIO, carried=passages)

    assert window.carried == (passages[2],)
    assert _contents(window)[1] == "Eins, zwei, drei [1]."


def _long_passage(node_id: str) -> CarriedPassage:
    """Return an invented carried passage with a label of 128 characters.

    Args:
        node_id (str): The stored node, which also opens the label.

    Returns:
        CarriedPassage: The passage.
    """
    return CarriedPassage(node_id, f"{node_id}: " + " ".join(["Pegelstände der Anleger."] * 5))


@pytest.mark.parametrize(
    ("answer", "sources", "kept", "shown"),
    [
        (
            "Neu: A [1], B [2], C [3], D [4].",
            ("b1", "b2", "b3", "b4"),
            ["b1", "b2", "b3", "b4"],
            "Neu: A [1], B [2], C [3], D [4].",
        ),
        (
            "Neu: A [1], B [2], C [3], D [4], E [5].",
            ("b1", "b2", "b3", "b4", "b5"),
            ["b2", "b3", "b4", "b5"],
            "Neu: A, B [1], C [2], D [3], E [4].",
        ),
    ],
    ids=["newest-answer-fits-whole", "newest-answer-loses-its-oldest"],
)
def test_the_passages_of_an_older_answer_give_way_to_the_newest_answers(
    answer: str, sources: tuple[str, ...], kept: list[str], shown: str
) -> None:
    """A follow-up most likely concerns the latest answer, so the oldest citations are the ones the list loses.

    At the plain allowance four of these 128-character lines fit in the 200 tokens; the older answer's three
    passages are cited first, so they are the first to give way, and its markers go with them.

    Args:
        answer (str): The newest answer.
        sources (tuple[str, ...]): The nodes its citation numbers name.
        kept (list[str]): The nodes the window carries, in the order it numbers them.
        shown (str): The newest answer as the window sends it.
    """
    turns = [_cited("Alt: X [1], Y [2], Z [3].", "a1", "a2", "a3"), _cited(answer, *sources)]
    ids = cited_node_ids(turns, recent_positions(turns, PLAIN))
    carried = [_long_passage(node_id) for node_id in ids]

    window = build_window(turns, "", allowance_tokens=PLAIN, char_token_ratio=RATIO, carried=carried)

    assert ids == ("a1", "a2", "a3", *sources)
    assert [passage.node_id for passage in window.carried] == kept
    assert window.summary_block().splitlines()[1] == f"[1] {_long_passage(kept[0]).label}"
    assert _contents(window)[1] == "Alt: X, Y, Z."
    assert _contents(window)[3] == shown


@pytest.mark.parametrize(("label_chars", "carried"), [(265, True), (266, False)], ids=["fits", "one-over"])
def test_the_carried_section_is_priced_with_the_separator_ahead_of_it(label_chars: int, carried: bool) -> None:
    """The block joins its sections with a blank line, so the list may use its share less those two characters.

    At 380 tokens one exchange is sent and 100 tokens are spare, which are 350 characters at 3.5 per token.

    Args:
        label_chars (int): The length of the one passage's label.
        carried (bool): Whether the passage is carried.
    """
    passage = CarriedPassage("n1", "x" * label_chars)

    window = build_window([_turn(1)], "", allowance_tokens=380, char_token_ratio=RATIO, carried=[passage])

    assert window.carried == ((passage,) if carried else ())


def test_carried_passages_take_at_most_their_share_of_the_allowance() -> None:
    """At the plain allowance the room is ample, so ``CARRIED_TOKENS`` is what ends the list."""
    passages = [_passage(number) for number in range(1, 9)]

    window = build_window([_turn(1)], "", allowance_tokens=PLAIN, char_token_ratio=RATIO, carried=passages)

    assert CARRIED_TOKENS == 200
    assert window.carried == tuple(passages[-5:])
    assert 180 <= estimate_tokens(window.summary_block(), RATIO) <= CARRIED_TOKENS
    assert _spent(window) <= PLAIN


def test_carried_passages_claim_their_room_before_the_earlier_questions_and_the_summary() -> None:
    """With room for little else, the carried list is what the history spends it on; the exchanges are unaffected."""
    turns = [ConversationTurn("Was lag am Anleger 1?", "Nichts.", None), _cited("Zwei [1] [2] [3].", "n1", "n2", "n3")]
    passages = [_passage(number) for number in range(1, 4)]

    baseline = build_window(turns, FOLD, allowance_tokens=380, char_token_ratio=RATIO)
    window = build_window(turns, FOLD, allowance_tokens=380, char_token_ratio=RATIO, carried=passages)

    assert baseline.earlier_questions and baseline.summary
    assert window.carried == tuple(passages[-2:])
    assert window.earlier_questions == () and window.summary == ""
    assert len(window.recent) == len(baseline.recent) == 1
    assert _spent(window) <= 380


def test_a_zero_allowance_carries_nothing() -> None:
    """History gives way completely, the carried list included."""
    window = build_window(
        _heavy(6),
        HEAVY_SUMMARY,
        allowance_tokens=0,
        char_token_ratio=RATIO,
        carried=[_passage(number) for number in range(1, 4)],
    )

    assert window == ConversationWindow()


def test_random_histories_with_carried_passages_never_overspend_their_allowance() -> None:
    """Whatever the lengths, ratio and allowance, the carried list is a newest run in its share and the window fits."""
    rng = random.Random(5)
    for _ in range(800):
        ratio = rng.choice([3.5, 2.0, 3.3, 2.3, 4.0, 1.7])
        allowance = rng.choice([40, 150, 300, 380, 500, THINKING, 1000, PLAIN, 2000])
        passages = [CarriedPassage(f"n{k}", _prose(rng, rng.randrange(1, 25))) for k in range(1, rng.randrange(1, 10))]
        turns = [
            ConversationTurn(
                _prose(rng, rng.randrange(1, 60)) + "?",
                " ".join(f"{_prose(rng, rng.randrange(1, 40))} [{rng.randrange(1, 12)}]" for _ in range(6)),
                None,
                tuple(rng.choice([None, *(p.node_id for p in passages)]) for _ in range(10)),
            )
            for _ in range(rng.randrange(0, 5))
        ]
        summary = "\n".join(_prose(rng, rng.randrange(5, 120)) for _ in range(rng.randrange(0, 8)))

        window = build_window(turns, summary, allowance_tokens=allowance, char_token_ratio=ratio, carried=passages)

        assert _spent(window, ratio) <= allowance
        if window.summary_block() != NO_HISTORY:
            exchanges = len(window.recent)
            spare = max(0, allowance - 2 * exchanges * MESSAGE_OVERHEAD_TOKENS - exchanges * MIN_EXCHANGE_TOKENS)
            assert estimate_tokens(window.summary_block(), ratio) <= spare
        assert window.carried == tuple(passages[len(passages) - len(window.carried) :])
        if window.carried:
            assert estimate_tokens(ConversationWindow(carried=window.carried).summary_block(), ratio) <= CARRIED_TOKENS
        markers = [group for text in _contents(window) for group in re.findall(r"\[([\d, ]+)\]", text)]
        shown = [int(number) for group in markers for number in group.split(",")]
        assert all(1 <= number <= len(window.carried) for number in shown)


def test_cited_node_ids_keeps_the_newest_eight() -> None:
    """Each id keeps its latest citation; the eight cited last win, ordered by that citation, oldest first."""
    older = _cited("[1] [2] [3] [4] [5]", "s1", "s2", "s3", "s4", "s5")
    newer = _cited("[1] [2] [3] [4] [5]", "t1", "t2", "t3", "s2", "t4")

    assert CARRIED_MAX_PASSAGES == 8
    assert cited_node_ids([older, newer], [0, 1]) == ("s3", "s4", "s5", "t1", "t2", "t3", "s2", "t4")
    assert cited_node_ids([older, newer], [0]) == ("s1", "s2", "s3", "s4", "s5")
    assert cited_node_ids([older, newer], [1]) == ("t1", "t2", "t3", "s2", "t4")
    assert cited_node_ids([older, newer], []) == ()


def test_cited_node_ids_counts_a_passage_cited_twice_once() -> None:
    """The cap is on distinct ids, and the repeat moves the passage to its latest citation."""
    turn = _cited("[1] [2] [3] [4] [5] [6] [7] [8] [9] [10] [1]", *(f"n{number}" for number in range(1, 11)))

    assert cited_node_ids([turn], [0]) == ("n4", "n5", "n6", "n7", "n8", "n9", "n10", "n1")


def test_cited_node_ids_reads_lists_and_ranges() -> None:
    """A semicolon list and a dashed range name their numbers; a 40-wide range names only its ends."""
    sources = tuple(f"n{number}" for number in range(1, 51))
    turn = _cited("Erstens [2; 4], zweitens [3\N{EN DASH}5], drittens [1\N{EN DASH}40].", *sources)

    assert cited_node_ids([turn], [0]) == ("n2", "n3", "n4", "n5", "n1", "n40")


@pytest.mark.parametrize(
    ("marker", "ids"),
    [
        ("[2]", ("n2",)),
        ("[ 2 ]", ("n2",)),
        ("[2, 4]", ("n2", "n4")),
        ("[2;4]", ("n2", "n4")),
        ("[3\N{EN DASH}5]", ("n3", "n4", "n5")),
        ("[3-5]", ("n3", "n4", "n5")),
        ("[3 - 5]", ("n3", "n4", "n5")),
        ("[1, 3\N{EN DASH}4, 7]", ("n1", "n3", "n4", "n7")),
        ("[5\N{EN DASH}3]", ("n3", "n4", "n5")),
        ("[4\N{EN DASH}4]", ("n4",)),
        ("[1\N{EN DASH}40]", ("n1", "n40")),
    ],
    ids=[
        "single",
        "padded",
        "comma-list",
        "semicolon-list",
        "en-dash-range",
        "hyphen-range",
        "spaced-range",
        "list-with-a-range",
        "backwards-range",
        "one-number-range",
        "wide-range",
    ],
)
def test_cited_node_ids_reads_every_marker_shape(marker: str, ids: tuple[str, ...]) -> None:
    """The markers the answer model writes are the ones the dropping already recognizes.

    Args:
        marker (str): The marker in the answer.
        ids (tuple[str, ...]): The nodes it cites.
    """
    turn = _cited(f"Das steht {marker}.", *(f"n{number}" for number in range(1, 51)))

    assert cited_node_ids([turn], [0]) == ids


def test_cited_node_ids_skips_numbers_without_a_source() -> None:
    """Zero, a number past the sources, a citation the store has no id for and a huge number name nothing."""
    turn = _cited("[0] [1] [2] [3] [4] [6] [99999999999999999999]", "n1", None, "", "n4", "n5")

    assert cited_node_ids([turn], [0]) == ("n1", "n4")
    assert cited_node_ids([_cited("Kein Beleg [1].")], [0]) == ()


def test_cited_node_ids_reads_both_parts_of_an_assessment() -> None:
    """The model's own assessment may cite too."""
    labelled = compose_labelled_answer(f"Fakt [1].\n{SENTINEL}\nEinschätzung [2].")
    turn = ConversationTurn("Stimmt das?", labelled.text, labelled.evidence_chars, ("n1", "n2"))

    assert cited_node_ids([turn], [0]) == ("n1", "n2")


def test_the_ids_the_answers_cite_are_the_ones_the_window_numbers() -> None:
    """Carrying what ``cited_node_ids`` names leaves no marker of the recent answers unnumbered."""
    turns = [
        _cited("Zuerst [1] und [2\N{EN DASH}3].", "n1", "n2", "n3"),
        _cited("Dann [2, 1] und [3].", "n3", "n2", "n4"),
    ]
    carried = [
        CarriedPassage(node_id, f"Bericht {node_id}")
        for node_id in cited_node_ids(turns, recent_positions(turns, PLAIN))
    ]

    window = build_window(turns, "", allowance_tokens=PLAIN, char_token_ratio=RATIO, carried=carried)

    assert [passage.node_id for passage in window.carried] == ["n1", "n2", "n3", "n4"]
    assert [turn.assistant_text for turn in window.recent] == ["Zuerst [1] und [2, 3].", "Dann [2, 3] und [4]."]


@pytest.mark.parametrize("allowance", [PLAIN, THINKING, 300, 100], ids=["plain", "thinking", "small", "none"])
def test_recent_positions_is_the_window_s_own_choice(allowance: int) -> None:
    """The exchanges the carried ids are read from are exactly the ones the window sends.

    Args:
        allowance (int): The history allowance.
    """
    turns = [
        _turn(1),
        _turn(2),
        ConversationTurn("Welche Fähren?", "", None),
        _turn(4),
        _turn(5),
        ConversationTurn("Und jetzt?", "", None),
    ]

    positions = recent_positions(turns, allowance)
    window = build_window(turns, "", allowance_tokens=allowance, char_token_ratio=RATIO)

    assert [turn.user_text for turn in window.recent] == [turns[position].user_text for position in positions]
    assert positions == {PLAIN: [3, 4], THINKING: [4], 300: [4], 100: []}[allowance]


def test_last_answered_skips_a_turn_with_no_question_and_one_with_no_answer_left() -> None:
    """A turn whose answer is nothing but markers is no exchange, carried passages or not."""
    turns = [_turn(1), ConversationTurn("", "Antwort.", None), ConversationTurn("Frage?", "[1]", None), _turn(4)]

    assert last_answered(turns, 2) == [0, 3]
    assert last_answered(turns, 1) == [3]
    assert last_answered([], 2) == []
