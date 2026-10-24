"""The conversation behind an agent turn: a bounded window the model reads before the new message.

The window has four parts, all drawn from a session's stored turns and rolling summary. The last
exchanges are sent as real user and assistant messages, so a follow-up can quote them. The user's
earlier questions and the newest part of the summary go into the system prompt, so a question left
open several turns back still reaches the model. So do the carried passages, the ones those last
answers cite, each listed under the number the turn will know it by.

Citation numbers are dropped from everything an earlier turn wrote, because they number that turn's
sources, not this one's. The exception is a number naming a carried passage: it stays, renumbered
to the passage's number in the list.

The session's own context string cannot serve: it drops the raw turns right after a summary fold and
its summary grows without bound.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

from llama_index.core.base.llms.types import ChatMessage, MessageRole

from docint.agents.assessment.split import ASSESSMENT_RULE, CITATION_NUMBERS_RE, neutralize_sentinel
from docint.agents.turn.budget import HISTORY_ALLOWANCE_TOKENS, MESSAGE_OVERHEAD_TOKENS
from docint.utils.embed_chunking import estimate_tokens
from docint.utils.llm_sanitize import strip_reasoning
from docint.utils.text_budget import (
    MARKER_RESERVE_CHARS,
    allocate_shortest_first,
    fit_tail_with_marker,
    fit_with_marker,
)

#: Exchanges the window sends as messages: two, or one when the allowance is below the plain model's.
RECENT_EXCHANGES = 2
#: Tokens the earlier questions may take, their heading included.
EARLIER_QUESTIONS_TOKENS = 150
#: Tokens the newest part of the summary may take, its heading included.
SUMMARY_TOKENS = 250
#: Tokens the carried passages may take, their heading included.
CARRIED_TOKENS = 200
#: Passages carried into a turn at most: the ones cited last by the exchanges the window sends.
CARRIED_MAX_PASSAGES = 8
#: Most numbers a range in a citation marker names in full; a wider one names only its two ends.
CITED_RANGE_MAX = 10
#: Least text an exchange sent as messages is worth its envelopes for; below it the oldest one gives way.
MIN_EXCHANGE_TOKENS = 200

#: What the system prompt shows when there is no summary, no earlier question and no carried passage.
NO_HISTORY = "(none)"
#: Headings of the three sections of :meth:`ConversationWindow.summary_block`.
SUMMARY_HEADER = "Summary of the earlier conversation"
QUESTIONS_HEADER = "Earlier questions, newest first (their answers are not shown)"
CARRIED_HEADER = "Passages the earlier answers cite, carried into this turn (read(n) shows one)"
#: What joins the sections of the block.
_SECTION_SEPARATOR = "\n\n"

#: How an earlier assessment's two parts are labelled in the window's messages.
ANSWER_SOURCED_LABEL = "Previous assistant answer, source-backed part"
ANSWER_ASSESSMENT_LABEL = "Previous assistant answer, the model's own assessment"

#: Longest history-resolved rewrite, in characters, that a plain lookup's answer is shown beside the question.
RESOLVED_QUESTION_CHARS = 1000
#: How the question block labels that rewrite.
RESOLVED_QUESTION_LABEL = (
    "Resolved from earlier turns by a search rewrite (may repeat earlier answers, the model's own assessments "
    "included; not a source, never cite it)"
)
#: A citation with the blanks before it, so removing it leaves no gap ahead of the punctuation that follows.
_CITATION_MARKER_RE = re.compile(rf"[ \t]*{CITATION_NUMBERS_RE.pattern}")


@dataclass(frozen=True)
class ConversationTurn:
    """One stored exchange of a session.

    Attributes:
        user_text (str): The user's message.
        assistant_text (str): The stored answer, an assessment's label and rule included.
        evidence_chars (int | None): Length of the source-backed part when the answer was an
            assessment, as the turn's trail recorded it; ``None`` when the whole answer is
            source-backed. A turn in a built window carries ``None``: its parts are already
            rendered, labelled, into ``assistant_text``.
        sources (tuple[str | None, ...]): The stored node each citation number of the answer
            names, ``sources[n - 1]`` being ``[n]``; ``None`` where the store keeps no id for
            it. Empty when the turn's citations were not read. A turn in a built window carries
            ``()``: its markers are already renumbered.
    """

    user_text: str
    assistant_text: str
    evidence_chars: int | None = None
    sources: tuple[str | None, ...] = ()


@dataclass(frozen=True)
class CarriedPassage:
    """A passage an earlier answer cited, which the window lists and the turn starts out holding.

    Attributes:
        node_id (str): The stored node the earlier answer cited.
        label (str): What the window lists for it: a description of the passage on one line.
            Whitespace is collapsed when it is listed, so a label cannot open a line of its own.
    """

    node_id: str
    label: str


@dataclass(frozen=True)
class ConversationWindow:
    """The conversation as an agent turn is sent it.

    Attributes:
        summary (str): The newest part of the rolling summary, a leading marker saying so when
            the start was cut.
        earlier_questions (tuple[str, ...]): The user's questions from before the recent
            exchanges, newest first, each on one line.
        recent (tuple[ConversationTurn, ...]): The last exchanges, oldest first, cut to fit
            their share of the allowance. A citation number survives in an answer only where it
            names a carried passage, renumbered to that passage's number in ``carried``.
        carried (tuple[CarriedPassage, ...]): The passages the block lists, the first being
            ``[1]``: the newest run of those asked for that fits their share of the allowance,
            in the order asked for.
    """

    summary: str = ""
    earlier_questions: tuple[str, ...] = ()
    recent: tuple[ConversationTurn, ...] = ()
    carried: tuple[CarriedPassage, ...] = ()

    def summary_block(self) -> str:
        """Render the summary, the earlier questions and the carried passages for the system prompt.

        The carried passages come last, nearest the messages whose citation numbers they number.

        Returns:
            str: The summary section, then the earlier questions section, then the carried
            passages section, whichever exist; ``NO_HISTORY`` when none does.
        """
        sections = []
        if self.summary:
            sections.append(_summary_section(self.summary))
        if self.earlier_questions:
            sections.append(_questions_section(self.earlier_questions))
        if self.carried:
            sections.append(_carried_section(self.carried))
        return _SECTION_SEPARATOR.join(sections) or NO_HISTORY

    def messages(self) -> list[ChatMessage]:
        """Render the recent exchanges as the messages that precede the new one.

        Returns:
            list[ChatMessage]: A user message, then an assistant message, per exchange, oldest first.
        """
        messages: list[ChatMessage] = []
        for turn in self.recent:
            messages.append(ChatMessage(role=MessageRole.USER, content=turn.user_text))
            messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=turn.assistant_text))
        return messages


def split_answer(text: str, evidence_chars: int | None) -> tuple[str, str]:
    """Split a stored answer into its source-backed part and the model's own assessment.

    The stored boundary is trusted only where it is ``0`` or sits on ``ASSESSMENT_RULE``;
    one out of range or off the rule splits nothing. The assessment part keeps its
    localized heading and disclaimer.

    Args:
        text (str): The stored answer.
        evidence_chars (int | None): Length of its source-backed part, if it was an assessment.

    Returns:
        tuple[str, str]: ``(sourced, assessment)``, the rule between them dropped; both empty
        when the answer is not an assessment, or has nothing on either side.
    """
    cut = evidence_chars
    aligned = cut is not None and 0 <= cut <= len(text) and (cut == 0 or text.startswith(ASSESSMENT_RULE, cut))
    return (text[:cut], text[cut:].removeprefix(ASSESSMENT_RULE)) if aligned else ("", "")


def labelled_lines(labels: Sequence[str], parts: Sequence[str]) -> str:
    """Put each part behind its label, one line per part, and leave out an empty one.

    Args:
        labels (Sequence[str]): One label per part.
        parts (Sequence[str]): The parts.

    Returns:
        str: The labelled lines.
    """
    return "\n".join(f"{label}: {part}" for label, part in zip(labels, parts, strict=True) if part)


def question_block(question: str, resolved: str | None) -> str:
    """Render the question a plain lookup's answer is asked: the user's words, then the rewrite.

    A plain lookup retrieves with the history-resolved search rewrite, which can
    carry what the conversation settled before the last exchange, such as a
    constraint set two turns earlier, so its answer reads the rewrite on a
    labelled line under the user's own words. The rewrite is model output, so it
    is shown sanitized: reasoning blocks and bracketed citation numbers, which
    point at an earlier turn's sources, are dropped, its whitespace collapses onto
    one line, sentinels are defused, and it is cut to ``RESOLVED_QUESTION_CHARS``
    with a marker.

    Args:
        question (str): The user's message, verbatim.
        resolved (str | None): The history-resolved search rewrite, if any.

    Returns:
        str: The stripped question, then one line labelled ``RESOLVED_QUESTION_LABEL``
        carrying the sanitized rewrite; the question alone when there is no rewrite,
        nothing of it is left, or it holds the question's words up to case, spacing,
        punctuation and citation numbers, compared before the cut.
    """
    asked = question.strip()
    if resolved is None:
        return asked
    visible, _ = strip_reasoning(resolved)
    plain = neutralize_sentinel(" ".join(CITATION_NUMBERS_RE.sub("", visible).split()))
    if not plain or _words(plain) == _words(CITATION_NUMBERS_RE.sub("", asked)):
        return asked
    sanitized, _ = fit_with_marker(plain, RESOLVED_QUESTION_CHARS)
    return f"{asked}\n{RESOLVED_QUESTION_LABEL}: {sanitized}"


def _words(text: str) -> list[str]:
    """Return *text*'s words, case-folded, for comparing a rewrite with the question.

    Args:
        text (str): A question or its rewrite.

    Returns:
        list[str]: The runs of word characters in *text*, in order.
    """
    return re.findall(r"\w+", text.casefold())


def _plain(text: str) -> str:
    """Drop the citation numbers from a text an earlier turn wrote.

    Args:
        text (str): An earlier answer, or the summary of some.

    Returns:
        str: The text without its markers, line breaks kept, stripped.
    """
    return _CITATION_MARKER_RE.sub("", text).strip()


def _summary_section(summary: str) -> str:
    """Render the summary section of the block.

    Args:
        summary (str): The newest part of the rolling summary.

    Returns:
        str: The heading and the summary.
    """
    return f"{SUMMARY_HEADER}:\n{summary}"


def _questions_section(questions: Sequence[str]) -> str:
    """Render the earlier-questions section of the block.

    Args:
        questions (Sequence[str]): The questions, newest first.

    Returns:
        str: The heading and one bullet per question.
    """
    return f"{QUESTIONS_HEADER}:\n" + "\n".join(f"- {question}" for question in questions)


def _carried_section(carried: Sequence[CarriedPassage]) -> str:
    """Render the carried-passages section of the block.

    Args:
        carried (Sequence[CarriedPassage]): The passages, the first being ``[1]``.

    Returns:
        str: The heading and one numbered line per passage, its label on one line.
    """
    lines = (f"[{number}] {' '.join(passage.label.split())}" for number, passage in enumerate(carried, 1))
    return f"{CARRIED_HEADER}:\n" + "\n".join(lines)


def _envelopes(exchanges: int) -> int:
    """Count the tokens the messages of some exchanges cost beyond their text.

    Args:
        exchanges (int): How many exchanges are sent as messages.

    Returns:
        int: Tokens for their user and assistant messages.
    """
    return 2 * exchanges * MESSAGE_OVERHEAD_TOKENS


def _chars_within(tokens: int, char_token_ratio: float) -> int:
    """Return the longest text whose token estimate stays within a budget.

    Args:
        tokens (int): The budget.
        char_token_ratio (float): Characters per token.

    Returns:
        int: Characters; ``0`` for a budget that is not positive.
    """
    chars = max(0, int(tokens * char_token_ratio))
    # The estimate divides, so a boundary length can round one token over.
    while chars and chars / char_token_ratio > tokens:
        chars -= 1
    return chars


def _within(text: str, chars: int) -> str:
    """Cut a text to ``chars`` in all, the cut marker included.

    Args:
        text (str): The full text.
        chars (int): The most characters it may use.

    Returns:
        str: The text; cut with a marker when it is longer, bare when even a marker would not fit.
    """
    chars = max(0, chars)
    if len(text) <= chars:
        return text
    if chars <= MARKER_RESERVE_CHARS:
        return text[:chars]
    return fit_with_marker(text, chars - MARKER_RESERVE_CHARS)[0]


def _newest_questions(asked: Iterable[str], chars: int) -> tuple[str, ...]:
    """Take the newest questions that fit a section of ``chars`` characters.

    Args:
        asked (Iterable[str]): The questions, newest first, each on one line.
        chars (int): The most characters the rendered section may use.

    Returns:
        tuple[str, ...]: The questions kept, newest first. The newest one is cut with a marker
        rather than dropped when it alone is too long, since it is the likeliest to be open.
    """
    kept: list[str] = []
    for question in asked:
        if len(_questions_section([*kept, question])) <= chars:
            kept.append(question)
            continue
        if not kept:
            cut = _within(question, chars - len(_questions_section([""])))
            kept.extend([cut] if cut else [])
        break
    return tuple(kept)


def _newest_summary(summary: str, chars: int) -> str:
    """Take the newest part of the rolling summary that fits a section of ``chars`` characters.

    Args:
        summary (str): The whole rolling summary, newest at the end.
        chars (int): The most characters the rendered section may use.

    Returns:
        str: The summary without citation numbers, whole when it fits; else its newest whole
        lines behind a marker. Empty when there is nothing, or no room for more than the marker.
    """
    text = _plain(summary)
    room = chars - len(_summary_section(""))
    if len(text) <= room:
        return text
    allowance = room - MARKER_RESERVE_CHARS - 1
    return fit_tail_with_marker(text, allowance)[0] if allowance > 0 else ""


def _carried_within(carried: Sequence[CarriedPassage], chars: int) -> tuple[CarriedPassage, ...]:
    """Take the newest carried passages whose lines fit a section of ``chars`` characters.

    Args:
        carried (Sequence[CarriedPassage]): The passages, oldest cited first, as
            :func:`cited_node_ids` returns them.
        chars (int): The most characters the rendered section may use.

    Returns:
        tuple[CarriedPassage, ...]: The longest newest run of ``carried`` that fits, in the order
        given, so the oldest passage kept is ``[1]``. The lines are walked from the newest back
        and kept whole; the first that does not fit ends the run, and every older passage is left out.
    """
    kept: list[CarriedPassage] = []
    for passage in reversed(carried):
        if len(_carried_section([passage, *kept])) > chars:
            break
        kept.insert(0, passage)
    return tuple(kept)


def _span(first: int, last: int) -> list[int]:
    """List the numbers a range in a citation marker names.

    Args:
        first (int): The number the range starts at.
        last (int): The number it ends at; a range may be written backwards.

    Returns:
        list[int]: Every number from the lower end to the upper, or only those two ends when the
        range is wider than ``CITED_RANGE_MAX`` numbers, which cites everything rather than each.
    """
    low, high = sorted((first, last))
    return list(range(low, high + 1)) if high - low < CITED_RANGE_MAX else [low, high]


def _marker_numbers(marker: str) -> list[int]:
    """Read the citation numbers a marker names.

    Args:
        marker (str): A marker as ``_CITATION_MARKER_RE`` matches it, such as ``[2, 4]``,
            ``[2; 4]`` or ``[3-5]``; a range may be written with an en dash as well.

    Returns:
        list[int]: The numbers in the order written; a range gives the numbers it spans.
    """
    numbers: list[int] = []
    for item in re.split(r"[,;]", marker):
        ends = [int(digits) for digits in re.findall(r"\d+", item)]
        numbers.extend(_span(ends[0], ends[-1]) if len(ends) > 1 else ends)
    return numbers


def _marker_ids(marker: str, sources: Sequence[str | None]) -> list[str]:
    """Name the stored nodes a marker cites.

    Args:
        marker (str): A marker as ``_CITATION_MARKER_RE`` matches it.
        sources (Sequence[str | None]): The answer's stored nodes, ``sources[n - 1]`` being ``[n]``.

    Returns:
        list[str]: The node of each number the marker names, in the order written; a number with
        no stored node names none.
    """
    ids: list[str] = []
    for number in _marker_numbers(marker):
        node_id = sources[number - 1] if 1 <= number <= len(sources) else None
        if node_id:
            ids.append(node_id)
    return ids


def _renumbered(text: str, sources: Sequence[str | None], numbering: Mapping[str, int]) -> str:
    """Rewrite an earlier answer's citation markers to number the carried passages, and drop the rest.

    Args:
        text (str): The answer, or one part of it.
        sources (Sequence[str | None]): The answer's stored nodes, ``sources[n - 1]`` being ``[n]``.
        numbering (Mapping[str, int]): Each carried node's number in the window's list.

    Returns:
        str: The text, stripped. A marker naming carried passages becomes one naming their numbers,
        each once, in the order written; one naming none is removed with the blanks before it.
    """

    def renumber(match: re.Match[str]) -> str:
        found = match.group()
        shown = dict.fromkeys(numbering[node_id] for node_id in _marker_ids(found, sources) if node_id in numbering)
        return f"{found[: found.index('[')]}[{', '.join(map(str, shown))}]" if shown else ""

    return _CITATION_MARKER_RE.sub(renumber, text).strip()


def _reply_parts(turn: ConversationTurn, numbering: Mapping[str, int]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split an earlier answer into the parts the window shows and the labels they carry.

    Args:
        turn (ConversationTurn): The exchange.
        numbering (Mapping[str, int]): Each carried node's number in the window's list.

    Returns:
        tuple[tuple[str, ...], tuple[str, ...]]: The labels and the parts, citation numbers
        dropped except those naming a carried passage, which are renumbered: an assessment's
        two parts, each labelled where :func:`split_answer` trusts its boundary, or the whole
        answer with no label.
    """
    sourced, assessment = split_answer(turn.assistant_text, turn.evidence_chars)
    if sourced or assessment:
        parts = (_renumbered(sourced, turn.sources, numbering), _renumbered(assessment, turn.sources, numbering))
        return (ANSWER_SOURCED_LABEL, ANSWER_ASSESSMENT_LABEL), parts
    return (), (_renumbered(turn.assistant_text, turn.sources, numbering),)


def _show_reply(labels: Sequence[str], parts: Sequence[str]) -> str:
    """Join an earlier answer's parts as its assistant message reads.

    Args:
        labels (Sequence[str]): The parts' labels, none for a plain answer.
        parts (Sequence[str]): The parts.

    Returns:
        str: The labelled lines of an assessment, or the one part of a plain answer.
    """
    return labelled_lines(labels, parts) if labels else parts[0]


def last_answered(turns: Sequence[ConversationTurn], count: int) -> list[int]:
    """Find the last turns that are complete exchanges, scanning from the end.

    Args:
        turns (Sequence[ConversationTurn]): The session's turns, oldest first.
        count (int): How many to find at most.

    Returns:
        list[int]: Their positions, oldest first; a turn with no question, or no answer left
        once its citation numbers are dropped, is not one.
    """
    found: list[int] = []
    for position in range(len(turns) - 1, -1, -1):
        if turns[position].user_text.strip() and _plain(turns[position].assistant_text):
            found.append(position)
            if len(found) == count:
                break
    return found[::-1]


def recent_positions(turns: Sequence[ConversationTurn], allowance_tokens: int) -> list[int]:
    """Choose the exchanges a window of this allowance sends as messages.

    The window's own choice, which :func:`build_window` makes through this function, so a
    caller that reads the answers the window will show reads exactly those.

    Args:
        turns (Sequence[ConversationTurn]): The session's turns, oldest first.
        allowance_tokens (int): Tokens the window may cost; see ``history_allowance_tokens``.

    Returns:
        list[int]: The positions of the last two complete exchanges, oldest first; the last one
        when the allowance is below the plain model's; fewer, down to none, when the allowance
        cannot pay ``MIN_EXCHANGE_TOKENS`` and an envelope for each.
    """
    answered = last_answered(turns, RECENT_EXCHANGES)
    exchanges = min(RECENT_EXCHANGES if allowance_tokens >= HISTORY_ALLOWANCE_TOKENS else 1, len(answered))
    while exchanges and allowance_tokens - _envelopes(exchanges) < exchanges * MIN_EXCHANGE_TOKENS:
        exchanges -= 1
    return answered[len(answered) - exchanges :]


def cited_node_ids(turns: Sequence[ConversationTurn], positions: Sequence[int]) -> tuple[str, ...]:
    """Name the stored passages the answers at some positions cite.

    Args:
        turns (Sequence[ConversationTurn]): The session's turns, oldest first.
        positions (Sequence[int]): The exchanges to read, oldest first; see :func:`recent_positions`.

    Returns:
        tuple[str, ...]: At most ``CARRIED_MAX_PASSAGES`` distinct node ids: the ones cited last,
        each placed by its latest citation, oldest cited first. That is the order the window
        numbers them in, the first being ``[1]`` when all of them are listed.
    """
    cited = [
        node_id
        for position in positions
        for marker in _CITATION_MARKER_RE.finditer(turns[position].assistant_text)
        for node_id in _marker_ids(marker.group(), turns[position].sources)
    ]
    newest_first = list(dict.fromkeys(reversed(cited)))[:CARRIED_MAX_PASSAGES]
    return tuple(reversed(newest_first))


def _fit_recent(
    turns: Sequence[ConversationTurn], *, room_tokens: int, char_token_ratio: float, numbering: Mapping[str, int]
) -> tuple[ConversationTurn, ...]:
    """Cut the recent exchanges to share a room, every text shortest-first.

    A text needing less than its share returns the rest to the longer ones, so a short
    question is never cut for a long answer's sake, and an assessment's judgement is
    not pushed out by its source-backed part. The answers are renumbered before they are
    cut, so the cut is measured on the text that is sent.

    Args:
        turns (Sequence[ConversationTurn]): The exchanges, oldest first, each answered.
        room_tokens (int): Tokens for their text; the envelopes are already paid.
        char_token_ratio (float): Characters per token.
        numbering (Mapping[str, int]): Each carried node's number in the window's list.

    Returns:
        tuple[ConversationTurn, ...]: The exchanges as the window sends them.
    """
    asked = [turn.user_text.strip() for turn in turns]
    replies = [_reply_parts(turn, numbering) for turn in turns]
    texts = [text for question, (_, parts) in zip(asked, replies, strict=True) for text in (question, *parts)]
    labelling = sum(len(_show_reply(labels, parts)) - sum(map(len, parts)) for labels, parts in replies)
    # Each message is estimated on its own, so each may round up by one token.
    chars = max(0, _chars_within(room_tokens - 2 * len(turns), char_token_ratio) - labelling)
    grants = iter(allocate_shortest_first([len(text) for text in texts], chars))
    fitted: list[ConversationTurn] = []
    for question, (labels, parts) in zip(asked, replies, strict=True):
        question_shown, *parts_shown = (_within(text, next(grants)) for text in (question, *parts))
        fitted.append(ConversationTurn(question_shown, _show_reply(labels, parts_shown), None))
    return tuple(fitted)


def build_window(
    turns: Sequence[ConversationTurn],
    summary: str,
    *,
    allowance_tokens: int,
    char_token_ratio: float,
    carried: Sequence[CarriedPassage] = (),
) -> ConversationWindow:
    """Bound a session's history to what an agent turn may send.

    The last two exchanges are sent as messages in their own words, cut only to fit; one
    when the allowance is below the plain model's (a thinking model's), since two would
    leave nothing for the rest. The passages in ``carried`` are listed in the block, up to
    ``CARRIED_TOKENS``; the user's earlier questions, newest first, take up to
    ``EARLIER_QUESTIONS_TOKENS``, and the newest part of the rolling summary up to
    ``SUMMARY_TOKENS``. All three claim their share before the exchanges are cut, the
    carried passages first, so long answers cannot push an open question out. The block
    shows the carried passages last, nearest the messages whose markers they number.

    The carried list is the newest run of ``carried`` whose lines fit, whole lines only,
    numbered in the order given: when they do not all fit, the oldest give way and are not
    carried. Citation numbers are dropped from the summary and every answer, except that a
    marker naming a carried passage stays, renumbered to the passage's number in the list;
    an earlier assessment shows its two parts labelled where :func:`split_answer` trusts its
    stored boundary. With nothing carried, every marker is dropped.

    The allowance covers the whole window: the block, every message's text and the
    ``MESSAGE_OVERHEAD_TOKENS`` envelope each message costs, so a caller that counts the block
    in the system prompt counts the messages and their envelopes as the history. An
    allowance too small for an exchange worth sending (``MIN_EXCHANGE_TOKENS``) gives up
    the oldest exchange as a message; the question is still asked about.

    A turn with no answer is not an exchange to send, and its question is an earlier one.
    A cut text says how much it lost, unless its room is smaller than that note.

    Args:
        turns (Sequence[ConversationTurn]): The session's turns, oldest first.
        summary (str): The rolling summary, which grows by appending, so its end is the newest.
        allowance_tokens (int): Tokens the window may cost; see ``history_allowance_tokens``.
        char_token_ratio (float): Characters per token for the estimate.
        carried (Sequence[CarriedPassage]): The passages to list, distinct, oldest cited first,
            which is the order to number them in; see :func:`cited_node_ids` for which the recent
            answers cite.

    Returns:
        ConversationWindow: The window, never over the allowance.

    Raises:
        ValueError: If ``char_token_ratio`` is not positive.
    """
    recent_at = recent_positions(turns, allowance_tokens)
    exchanges = len(recent_at)
    spare = max(0, allowance_tokens - _envelopes(exchanges) - exchanges * MIN_EXCHANGE_TOKENS)
    # Claimed first, and priced with the separator ahead of it, since the sections before it are not known yet.
    room_chars = _chars_within(min(CARRIED_TOKENS, spare), char_token_ratio) - len(_SECTION_SEPARATOR)
    listed = _carried_within(carried, room_chars)
    if listed:
        spare -= estimate_tokens(_SECTION_SEPARATOR + _carried_section(listed), char_token_ratio)
    asked = [" ".join(turns[i].user_text.split()) for i in reversed(range(len(turns))) if i not in recent_at]
    question_chars = _chars_within(min(EARLIER_QUESTIONS_TOKENS, spare), char_token_ratio)
    questions = _newest_questions(filter(None, asked), question_chars)
    if questions:
        spare -= estimate_tokens(_questions_section(questions), char_token_ratio)
    separator = len(_SECTION_SEPARATOR) if questions else 0
    newest = _newest_summary(summary, _chars_within(min(SUMMARY_TOKENS, spare), char_token_ratio) - separator)
    older = ConversationWindow(summary=newest, earlier_questions=questions, carried=listed)
    room = allowance_tokens - _envelopes(exchanges) - estimate_tokens(older.summary_block(), char_token_ratio)
    numbering = {passage.node_id: number for number, passage in enumerate(listed, 1)}
    recent = _fit_recent(
        [turns[i] for i in recent_at], room_tokens=room, char_token_ratio=char_token_ratio, numbering=numbering
    )
    return replace(older, recent=recent)
