"""The conversation window an agent turn is sent, and the session read behind it. All text is invented."""

from __future__ import annotations

import random
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
    EARLIER_QUESTIONS_TOKENS,
    MIN_EXCHANGE_TOKENS,
    NO_HISTORY,
    QUESTIONS_HEADER,
    RESOLVED_QUESTION_CHARS,
    RESOLVED_QUESTION_LABEL,
    SUMMARY_HEADER,
    SUMMARY_TOKENS,
    ConversationTurn,
    ConversationWindow,
    _chars_within,
    build_window,
    question_block,
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
