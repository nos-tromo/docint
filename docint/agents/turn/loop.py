"""One agent turn: the model reads the conversation, calls tools, reads every reply, and answers.

The loop follows an SDK agent loop. Each round is one model call, offered the turn's tools with
``tool_choice="auto"``: a round that calls tools has every call answered under its call's id and
goes again, and a round that calls none is the answer, streamed as it comes. A round's text is
provisional (see :func:`~docint.agents.turn.stream.stream_round`); text written before calls
becomes a ``note`` step, which tells the client to withdraw whatever of it was streamed.

Limits make the model answer in one more round, its last: the same transcript plus the final note
as a user message, the tools still declared but ``tool_choice="none"``. ``agent.reason`` names the
first cause: the rounds spent (``round_budget``), the time spent (``time_budget``), no room left
for one more full tool reply (``context_budget``), or two rounds in a row that added no passage
(``idle``). Limits are checked between rounds, and the time also bounds every call's I/O; a
round is never cut off while it streams, so an answer already streaming is never abandoned.

Calls. The calls of one round are prepared, fetched and finished as the tool registry requires:
prepared in call order on the loop thread, their I/O started on the process-wide search slots so
they run at once, then finished in call order on the loop thread, which numbers what they admit in
call order. Every call id gets a reply: calls past ``max_calls_per_round`` are skipped, a call
without room for a full reply never runs, and a call that cannot start, fails or outlives the time
left gets an error reply. Work abandoned at the deadline keeps its search slot until it ends.

Room. The turn's room for tool replies is the budget's ``round_room`` less the final note. A round
first reserves its assistant message and a short reply for every call; a call then runs only
while a full reply, ``result_chars`` at the turn's ratio plus its envelope, still fits in place of
its short one. No reply enters the transcript longer than ``result_chars``.

Failures. Model calls run on the loop thread, one model slot each. Before the turn has shown
anything, a failure goes to the caller, who can still answer another way: no model slot within
``MODEL_SLOT_WAIT_S`` raises :class:`~docint.agents.assessment.bounded.NoSlot`, the first call
refused as sent (HTTP 400 or 422, not an overflow) raises :class:`ToolsUnavailable`, and anything
else is raised as it came. Once the turn has shown a step or text, it ends with an
:class:`AgentResult` unless the answer round itself fails:

* a round offered tools that finds no model slot within ``MODEL_SLOT_WAIT_S`` (never past the
  time left), or whose call fails, makes the model answer (``research_failed``); text it showed
  before breaking off is withdrawn by a note, as a round ending in calls withdraws its text;
* the answer round waits for a model slot up to the time left, and at least ``MODEL_SLOT_WAIT_S``;
  without one the turn ends with no answer text (``research_failed``);
* a round the endpoint finds too long is retried once as the answer, every tool reply cut to
  ``OVERFLOW_REPLY_SHARE`` (``context_budget``); with nothing to cut, or on a second overflow, it
  raises ``ValueError`` naming the context window;
* any other failure of the answer round raises the error as it came: there is no answer to keep,
  and a partial one must not pass as whole.

Nothing here logs a query, an answer, a tool argument or document text: only counts, tool names,
exception types and reason tokens.
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import time
from collections.abc import Callable, Generator, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import openai
from llama_index.core.base.llms.types import ChatMessage, MessageRole, TextBlock, ToolCallBlock
from llama_index.core.schema import BaseNode, NodeWithScore
from loguru import logger

from docint.agents.assessment.bounded import (
    MODEL_SLOT_POOL,
    MODEL_SLOT_WAIT_S,
    SEARCH_SLOT_POOL,
    SEARCH_SLOT_WAIT_S,
    NoSlot,
    start_bounded,
)
from docint.agents.assessment.ledger import EvidenceLedger, is_context_overflow
from docint.agents.assessment.split import SentinelStreamSplitter
from docint.agents.turn.budget import MESSAGE_OVERHEAD_TOKENS, TurnBudget
from docint.agents.turn.stream import Provisional, RoundDone, stream_round
from docint.agents.turn.tools import PreparedCall, ToolContext, ToolRegistry, ToolReply, error_reply
from docint.agents.types import AgentAction, AgentProgress
from docint.utils.embed_chunking import estimate_tokens
from docint.utils.text_budget import MARKER_RESERVE_CHARS, fit_with_marker

if TYPE_CHECKING:
    from llama_index.core.llms import LLM

    from docint.agents.turn.history import ConversationWindow

#: Longest note, in characters, that a round's text before its calls leaves in the trail and the transcript.
NOTE_CHARS = 500
#: Name of the thread a tool call's I/O runs on, so an abandoned one can be found.
SEARCH_THREAD_NAME = "agent-turn-search"
#: Share of each tool reply the overflow retry keeps.
OVERFLOW_REPLY_SHARE = 0.5
#: Rounds in a row that add no passage before the model must answer.
IDLE_ROUNDS = 2
#: HTTP statuses with which an endpoint refuses a request as sent, rather than failing to serve it.
_REFUSED_STATUSES = frozenset({400, 422})
#: What a second overflow raises; the API reports a ``ValueError`` naming the context window as ``context_overflow``.
_OVERFLOW_MESSAGE = "The agent turn's prompt exceeds the model's context window."

# The replies the loop writes itself, for calls it did not run or that did not answer in time;
# model-facing protocol like the tools' own, each prefixed ``Error: `` by ``error_reply``.
SKIPPED_REPLY = (
    "not run: at most {limit} calls run in one round; call it again in your next round if you still need it."
)
NO_ROOM_REPLY = "not run: the context has no room left for its reply; answer from the passages shown."
NO_TIME_REPLY = "not run: the time for this turn is up; answer from the passages shown."
LATE_REPLY = "no reply in time: the time for this turn ran out first; answer from the passages shown."
FAILED_REPLY = "the call failed; try other arguments, or answer from the passages shown."
NO_CAPACITY_REPLY = (
    "not run: no search capacity was free; call it again in your next round, or answer from the passages shown."
)


class ToolsUnavailable(Exception):
    """The endpoint refused the turn's first call, tools declared, before any token.

    Nothing was shown, so the caller answers without tools; the refusal is the ``__cause__``.
    """


@dataclass(frozen=True)
class AgentResult:
    """What an agent turn produced, for normalization and persistence.

    Attributes:
        raw_text (str): The answer exactly as the model wrote it, sentinel included, reasoning
            left out; empty when no answer could be written.
        nodes (list[NodeWithScore]): The turn's evidence numbered in ledger order: every passage
            a tool reply showed, stamped with its ``citation_index``.
        steps (int): Rounds that called tools.
        actions (list[dict[str, Any]]): The trail: one ``{action, detail, found}`` per note and
            per call to an offered tool, in order.
        reason (str | None): The first cause that made the model answer before it chose to, or
            ``None``.
        images_attached (int): Pictures sent to the model with the evidence.
        reasoning (str | None): Every round's reasoning in order, joined by blank lines; ``None``
            when there was none. Never logged.
    """

    raw_text: str
    nodes: list[NodeWithScore]
    steps: int
    actions: list[dict[str, Any]]
    reason: str | None
    images_attached: int = 0
    reasoning: str | None = None


@dataclass
class _Turn:
    """The state of one running turn.

    Attributes:
        ctx (ToolContext): The tools' evidence, budget, LLM view and memory.
        base (list[ChatMessage]): The system prompt, the history and the user's message.
        schemas (list[dict[str, Any]]): The tools as every round declares them.
        started (float): The clock's reading when the turn began.
        transcript (list[ChatMessage]): The rounds so far: assistant messages and tool replies.
        spent (int): Tokens the transcript costs.
        steps (int): Rounds that called tools.
        idle (int): Rounds in a row that added no passage.
        model_calls (int): Model calls begun.
        shown (bool): Whether anything has been yielded.
        reason (str | None): The first cause that made the model answer.
        shortened (bool): Whether the tool replies were cut after an overflow.
        actions (list[dict[str, Any]]): The trail so far.
        reasoning (list[str]): Each round's reasoning.
    """

    ctx: ToolContext
    base: list[ChatMessage]
    schemas: list[dict[str, Any]]
    started: float
    transcript: list[ChatMessage] = field(default_factory=list)
    spent: int = 0
    steps: int = 0
    idle: int = 0
    model_calls: int = 0
    shown: bool = False
    reason: str | None = None
    shortened: bool = False
    actions: list[dict[str, Any]] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)

    def stop(self, reason: str) -> None:
        """Record why the model must answer now, keeping the first cause.

        Args:
            reason (str): The reason token.
        """
        self.reason = self.reason or reason


@dataclass
class _Streamed:
    """How one round's stream went.

    Attributes:
        splitter (SentinelStreamSplitter): Swaps the sentinel for the label in what the round shows.
        done (RoundDone | None): How the round ended; ``None`` when it failed.
        error (Exception | None): Why it failed.
        sent (str): The round's text that went out as provisional text.
        shown (bool): Whether any of its text was yielded.
    """

    splitter: SentinelStreamSplitter = field(default_factory=SentinelStreamSplitter)
    done: RoundDone | None = None
    error: Exception | None = None
    sent: str = ""
    shown: bool = False


def _refused(exc: BaseException) -> bool:
    """Return whether the endpoint refused a request as sent.

    Args:
        exc (BaseException): The failure.

    Returns:
        bool: ``True`` for an HTTP 400 or 422 answer.
    """
    return isinstance(exc, openai.APIStatusError) and exc.status_code in _REFUSED_STATUSES


def _echoed(arguments: str) -> str:
    """Return a call's arguments as the transcript echoes them back.

    Chat templates parse the arguments of an earlier call as JSON, so arguments that are not a
    JSON object would fail the next round; they are echoed as an empty object, and the call's
    error reply says what was wrong.

    Args:
        arguments (str): The arguments as the model sent them.

    Returns:
        str: The arguments when they are a JSON object, else ``{}``.
    """
    try:
        value = json.loads(arguments)
    except (ValueError, RecursionError):
        return "{}"
    return arguments if isinstance(value, dict) else "{}"


def _shortened(transcript: Sequence[ChatMessage]) -> list[ChatMessage] | None:
    """Cut every tool reply to ``OVERFLOW_REPLY_SHARE`` of its length, marking what it lost.

    Args:
        transcript (Sequence[ChatMessage]): The rounds so far.

    Returns:
        list[ChatMessage] | None: The transcript with its replies cut; ``None`` when no reply is
        long enough for the cut to shorten it.
    """
    cut = False
    kept: list[ChatMessage] = []
    for message in transcript:
        text = str(message.content or "")
        if message.role == MessageRole.TOOL:
            shorter, trimmed = fit_with_marker(text, int(len(text) * OVERFLOW_REPLY_SHARE))
            if trimmed and len(shorter) < len(text):
                message = ChatMessage(
                    role=MessageRole.TOOL, content=shorter, additional_kwargs=dict(message.additional_kwargs)
                )
                cut = True
        kept.append(message)
    return kept if cut else None


class AgentLoop:
    """Run one chat turn as an agent loop over the turn's tools."""

    def __init__(
        self,
        *,
        llm: LLM,
        registry: ToolRegistry,
        budget: TurnBudget,
        system_prompt: str,
        final_note: str,
        max_rounds: int,
        time_budget_s: float,
        max_calls_per_round: int,
        char_token_ratio: float,
        llm_view: Callable[[BaseNode], BaseNode],
        clock: Callable[[], float] = time.monotonic,
        model_slots: Any = MODEL_SLOT_POOL,
        search_slots: Any = SEARCH_SLOT_POOL,
        reasoning_open: bool = False,
    ) -> None:
        """Bind the model, the tools and the limits.

        Args:
            llm (LLM): The chat model every round calls; its ``stream_chat`` must pass the raw
                chunks on ``raw``.
            registry (ToolRegistry): The tools the turn offers; one per turn.
            budget (TurnBudget): The turn's budget, measured on this system prompt, these tools'
                schemas, the window and the message.
            system_prompt (str): The formatted system prompt, the window's summary block in it.
            final_note (str): What the last round tells the model: answer now, no more tools.
            max_rounds (int): Rounds that may call tools; ``0`` allows none.
            time_budget_s (float): Seconds the turn may take before the model must answer.
            max_calls_per_round (int): Calls one round may run; the rest are skipped.
            char_token_ratio (float): Characters per token, the ratio the budget was measured with.
            llm_view (Callable[[BaseNode], BaseNode]): Hides the metadata the model must not see.
            clock (Callable[[], float]): Monotonic clock, injectable for tests.
            model_slots (Any): The model-call slots, a semaphore shared by the process
                (``MODEL_SLOT_POOL``); injectable for tests.
            search_slots (Any): The slots tool I/O runs in, a semaphore shared by the process
                (``SEARCH_SLOT_POOL``); injectable for tests.
            reasoning_open (bool): Whether thinking was asked for, so a reply may open in reasoning
                the chat template began; pass it only when the request enables thinking.

        Raises:
            ValueError: If ``char_token_ratio`` is not positive.
        """
        self._llm = llm
        self._registry = registry
        self._budget = budget
        self._system_prompt = system_prompt
        self._final_note = final_note
        self._max_rounds = max_rounds
        self._time_budget_s = time_budget_s
        self._max_calls = max_calls_per_round
        self._ratio = char_token_ratio
        self._llm_view = llm_view
        self._clock = clock
        self._model_slots = model_slots
        self._search_slots = search_slots
        self._reasoning_open = reasoning_open
        self._room = budget.round_room - self._message_tokens(final_note)
        self._full_reply = math.ceil(budget.result_chars / char_token_ratio) + MESSAGE_OVERHEAD_TOKENS
        short = (
            SKIPPED_REPLY.format(limit=max_calls_per_round),
            NO_ROOM_REPLY,
            NO_TIME_REPLY,
            LATE_REPLY,
            FAILED_REPLY,
            NO_CAPACITY_REPLY,
        )
        self._short_reply = max(self._message_tokens(error_reply("", text).text) for text in short)

    def _message_tokens(self, text: str) -> int:
        """Estimate what one message costs: its text and its envelope.

        Args:
            text (str): The message's text.

        Returns:
            int: Estimated tokens.
        """
        return estimate_tokens(text, self._ratio) + MESSAGE_OVERHEAD_TOKENS

    def _time_left(self, turn: _Turn) -> float:
        """Return the seconds the turn has left.

        Args:
            turn (_Turn): The turn.

        Returns:
            float: The time left; negative once the budget is spent.
        """
        return self._time_budget_s - (self._clock() - turn.started)

    def run(
        self, *, window: ConversationWindow, message: str, ledger: EvidenceLedger
    ) -> Iterator[AgentProgress | str | AgentResult]:
        """Run the turn.

        Args:
            window (ConversationWindow): The conversation before the message; its summary block
                is already in the system prompt, and its recent exchanges are sent as messages.
            message (str): The user's message, verbatim.
            ledger (EvidenceLedger): The turn's evidence, empty, its cap set by the caller.

        Yields:
            AgentProgress | str | AgentResult: Steps and answer text as they come, the label
            swapped in for the sentinel, then exactly one result.

        Raises:
            NoSlot: Before anything is yielded, when the first round finds no model slot.
            ToolsUnavailable: Before anything is yielded, when the first call is refused as sent.
            ValueError: Naming the context window, when the answer does not fit even after the
                tool replies were cut.
            Exception: The endpoint's error, raised as it came, when it fails before anything is
                yielded or fails the answer.
        """
        turn = _Turn(
            ctx=ToolContext(ledger=ledger, budget=self._budget, llm_view=self._llm_view),
            base=[
                ChatMessage(role=MessageRole.SYSTEM, content=self._system_prompt),
                *window.messages(),
                ChatMessage(role=MessageRole.USER, content=message),
            ],
            schemas=self._registry.schemas(),
            started=self._clock(),
        )
        while True:
            limit = self._limit(turn)
            if limit is not None:
                turn.stop(limit)
                break
            wait = min(MODEL_SLOT_WAIT_S, max(0.0, self._time_left(turn)))
            streamed = yield from self._stream(turn, [*turn.base, *turn.transcript], tool_choice="auto", wait=wait)
            if streamed is None:
                if not turn.shown:
                    raise NoSlot
                logger.warning("No model slot came free for an agent round; the model answers from what it found")
                turn.stop("research_failed")
                break
            if streamed.done is None:
                withdrawn = self._failed_round(turn, streamed)
                if withdrawn is not None:
                    yield withdrawn
                break
            if streamed.done.reasoning:
                turn.reasoning.append(streamed.done.reasoning)
            if not streamed.done.calls:
                yield from self._answer_tail(turn, streamed)
                yield self._result(turn, streamed.done.text)
                return
            yield from self._tool_round(turn, streamed.done)
        yield from self._final_round(turn)

    def _limit(self, turn: _Turn) -> str | None:
        """Say which limit, if any, makes the model answer before the next round.

        Args:
            turn (_Turn): The turn.

        Returns:
            str | None: ``round_budget``, ``time_budget``, ``context_budget`` or ``idle``, checked
            in that order; ``None`` while the turn may go on.
        """
        if turn.steps >= self._max_rounds:
            return "round_budget"
        if self._time_left(turn) <= 0:
            return "time_budget"
        # One full reply, and the envelope of the message that asks for it.
        if self._room - turn.spent < self._full_reply + MESSAGE_OVERHEAD_TOKENS:
            return "context_budget"
        if turn.idle >= IDLE_ROUNDS:
            return "idle"
        return None

    def _stream(
        self, turn: _Turn, messages: list[ChatMessage], *, tool_choice: str, wait: float
    ) -> Generator[str, None, _Streamed | None]:
        """Stream one round in a model slot, yielding what it releases with the label swapped in.

        The slot is held while the round streams and given back when it ends, fails, or the
        caller stops reading. A failure is returned, not raised, for the caller to judge.

        Args:
            turn (_Turn): The turn.
            messages (list[ChatMessage]): What the round is sent.
            tool_choice (str): ``auto`` for a tool round, ``none`` for the answer round.
            wait (float): The longest the round waits for a model slot.

        Yields:
            str: Answer text the round releases.

        Returns:
            _Streamed | None: How the round went; ``None`` when no model slot came free.
        """
        if not self._model_slots.acquire(timeout=wait):
            return None
        try:
            turn.model_calls += 1
            streamed = _Streamed()
            try:
                for event in stream_round(
                    self._llm,
                    messages,
                    tools=turn.schemas,
                    tool_choice=tool_choice,
                    reasoning_open=self._reasoning_open,
                ):
                    if isinstance(event, Provisional):
                        streamed.sent += event.text
                        text = streamed.splitter.feed(event.text)
                        if text:
                            streamed.shown = turn.shown = True
                            yield text
                    else:
                        streamed.done = event
            except Exception as exc:
                streamed.done, streamed.error = None, exc
            return streamed
        finally:
            self._model_slots.release()

    def _answer_tail(self, turn: _Turn, streamed: _Streamed) -> Iterator[str]:
        """Send what an answer round wrote and has not released yet, then what the splitter holds.

        Args:
            turn (_Turn): The turn.
            streamed (_Streamed): The answer round, ended.

        Yields:
            str: The rest of the answer.
        """
        if streamed.done is None:
            return
        # What went out provisionally is the start of the round's text, so the rest is its tail.
        unsent = streamed.done.text[len(streamed.sent) :]
        for piece in (streamed.splitter.feed(unsent) if unsent else "", streamed.splitter.flush()):
            if piece:
                turn.shown = True
                yield piece

    def _result(self, turn: _Turn, text: str) -> AgentResult:
        """Build the turn's result.

        Args:
            turn (_Turn): The turn.
            text (str): The answer as the model wrote it.

        Returns:
            AgentResult: The result.
        """
        return AgentResult(
            raw_text=text,
            nodes=turn.ctx.ledger.numbered(),
            steps=turn.steps,
            actions=list(turn.actions),
            reason=turn.reason,
            reasoning="\n\n".join(turn.reasoning) or None,
        )

    def _overflowed(self, turn: _Turn, cause: Exception) -> None:
        """Cut the tool replies so the answer round can be retried, once.

        Args:
            turn (_Turn): The turn.
            cause (Exception): The endpoint's overflow error.

        Raises:
            ValueError: Naming the context window, when the replies were cut before or no reply
                is long enough to cut.
        """
        shorter = None if turn.shortened else _shortened(turn.transcript)
        if shorter is None:
            raise ValueError(_OVERFLOW_MESSAGE) from cause
        logger.warning("An agent round overflowed the context window; the model answers from shortened tool replies")
        turn.transcript, turn.shortened = shorter, True
        turn.stop("context_budget")

    def _failed_round(self, turn: _Turn, streamed: _Streamed) -> AgentProgress | None:
        """Judge a round offered tools that failed: retry an overflow, raise before anything was shown, else answer.

        Args:
            turn (_Turn): The turn.
            streamed (_Streamed): The failed round.

        Returns:
            AgentProgress | None: A note withdrawing the text the round showed before it broke
            off, for the caller to yield; ``None`` when it showed none.

        Raises:
            ToolsUnavailable: When the turn's first call was refused as sent.
            ValueError: Naming the context window, when an overflow leaves nothing to cut.
            Exception: The round's error, when nothing was shown yet.
        """
        exc = cast(Exception, streamed.error)
        if streamed.shown:
            # Its text is provisional: a note withdraws it, as for a round that ends in calls.
            logger.warning("An agent round broke off ({}); the model answers anew", type(exc).__name__)
            turn.stop("research_failed")
            return self._step(turn, "note", " ".join(streamed.sent.split())[:NOTE_CHARS], 0, step=turn.steps + 1)
        if is_context_overflow(exc):
            self._overflowed(turn, exc)
            return None
        if not turn.shown:
            if turn.model_calls == 1 and _refused(exc):
                raise ToolsUnavailable("The chat endpoint refused a request offering tools.") from exc
            raise exc
        logger.warning("An agent round failed ({}); the model answers from what it found", type(exc).__name__)
        turn.stop("research_failed")
        return None

    def _final_round(self, turn: _Turn) -> Iterator[str | AgentResult]:
        """Make the model answer with no tool to choose, and end the turn.

        Args:
            turn (_Turn): The turn, its reason recorded.

        Yields:
            str | AgentResult: The answer text, then the result.

        Raises:
            NoSlot: When no model slot comes free before anything was shown.
            ToolsUnavailable: When this is the turn's first call and it is refused as sent.
            ValueError: Naming the context window, on an overflow with nothing left to cut.
            Exception: The endpoint's error, when the answer fails.
        """
        logger.info("An agent turn answers after {} tool rounds ({})", turn.steps, turn.reason)
        while True:
            time_left = max(0.0, self._time_left(turn))
            # A started turn owes its answer: it waits the time left, and at least the usual wait.
            wait = max(MODEL_SLOT_WAIT_S, time_left) if turn.shown else min(MODEL_SLOT_WAIT_S, time_left)
            note = ChatMessage(role=MessageRole.USER, content=self._final_note)
            streamed = yield from self._stream(
                turn, [*turn.base, *turn.transcript, note], tool_choice="none", wait=wait
            )
            if streamed is None:
                if not turn.shown:
                    raise NoSlot
                logger.warning("No model slot came free for an agent turn's answer; it ends without one")
                turn.stop("research_failed")
                yield self._result(turn, "")
                return
            if streamed.done is not None:
                break
            exc = cast(Exception, streamed.error)
            if streamed.shown:
                raise exc
            if is_context_overflow(exc):
                self._overflowed(turn, exc)
                continue
            if not turn.shown and turn.model_calls == 1 and _refused(exc):
                raise ToolsUnavailable("The chat endpoint refused a request offering tools.") from exc
            raise exc
        done = streamed.done
        if done.reasoning:
            turn.reasoning.append(done.reasoning)
        if done.calls:
            logger.warning("An agent turn's answer round asked for {} tool calls anyway; none ran", len(done.calls))
        yield from self._answer_tail(turn, streamed)
        yield self._result(turn, done.text)

    def _step(self, turn: _Turn, action: str, detail: str, found: int, *, step: int | None = None) -> AgentProgress:
        """Record one step in the trail.

        Args:
            turn (_Turn): The turn.
            action (str): A note, or the name of an offered tool.
            detail (str): The note's text, or the call's main argument.
            found (int): Passages the call admitted or extended.
            step (int | None): The round it belongs to; the last tool round by default.

        Returns:
            AgentProgress: The step, numbered by its round.
        """
        turn.actions.append({"action": action, "detail": detail, "found": found})
        turn.shown = True
        number = turn.steps if step is None else step
        return AgentProgress(step=number, action=cast(AgentAction, action), detail=detail, found=found)

    def _tool_round(self, turn: _Turn, done: RoundDone) -> Iterator[AgentProgress]:
        """Answer a round's calls and add the round to the transcript.

        Args:
            turn (_Turn): The turn.
            done (RoundDone): The round, ended in calls.

        Yields:
            AgentProgress: The round's note, if it wrote text, then one step per call to an offered tool.
        """
        turn.steps += 1
        note = " ".join(done.text.split())[:NOTE_CHARS]
        if note:
            yield self._step(turn, "note", note, 0)
        preface = fit_with_marker(done.text, NOTE_CHARS)[0] if done.text else ""
        echoed = [_echoed(call.arguments) for call in done.calls]
        blocks: list[Any] = [TextBlock(text=preface)] if preface else []
        blocks.extend(
            ToolCallBlock(tool_call_id=call.id, tool_name=call.name, tool_kwargs=arguments)
            for call, arguments in zip(done.calls, echoed, strict=True)
        )
        written = preface + "".join(
            json.dumps({"name": call.name, "arguments": arguments}, ensure_ascii=False)
            for call, arguments in zip(done.calls, echoed, strict=True)
        )
        asked = self._message_tokens(written)
        prepared = [self._registry.prepare(call, turn.ctx) for call in done.calls]
        replies = self._replies(turn, prepared, self._admitted(turn, prepared, asked))
        turn.transcript.append(ChatMessage(role=MessageRole.ASSISTANT, blocks=blocks))
        turn.spent += asked
        found = 0
        for call, reply in zip(done.calls, replies, strict=True):
            text = self._bounded(reply)
            turn.transcript.append(
                ChatMessage(role=MessageRole.TOOL, content=text, additional_kwargs={"tool_call_id": call.id})
            )
            turn.spent += self._message_tokens(text)
            found += reply.found
            if reply.action in self._registry.names:
                yield self._step(turn, reply.action, reply.detail, reply.found)
        turn.idle = 0 if found else turn.idle + 1

    def _admitted(self, turn: _Turn, prepared: Sequence[PreparedCall], asked: int) -> list[bool | None]:
        """Decide which calls may run: within the round's limit, and while a full reply still fits.

        The round first reserves its assistant message and the least each call's reply costs: a
        call answered when prepared costs its reply, any other the shortest reply this loop
        writes. A call with work then runs, in call order, only while the rest of the room holds
        its full reply in place of that shortest one.

        Args:
            turn (_Turn): The turn.
            prepared (Sequence[PreparedCall]): The round's calls, prepared.
            asked (int): Tokens the round's assistant message costs.

        Returns:
            list[bool | None]: Per call: ``None`` when it is past the round's limit, ``True`` when
            it runs (or was answered when prepared), ``False`` when its reply has no room.
        """
        room = self._room - turn.spent - asked
        for position, ready in enumerate(prepared):
            within = position < self._max_calls
            room -= self._message_tokens(ready.reply.text) if within and ready.reply else self._short_reply
        admitted: list[bool | None] = []
        for position, ready in enumerate(prepared):
            if position >= self._max_calls:
                admitted.append(None)
            elif ready.reply is not None:
                admitted.append(True)
            elif room >= self._full_reply - self._short_reply:
                room -= self._full_reply - self._short_reply
                admitted.append(True)
            else:
                admitted.append(False)
        return admitted

    def _replies(
        self, turn: _Turn, prepared: Sequence[PreparedCall], admitted: Sequence[bool | None]
    ) -> list[ToolReply]:
        """Run the admitted calls' I/O at once on the search slots, then finish every call in call order.

        Args:
            turn (_Turn): The turn.
            prepared (Sequence[PreparedCall]): The round's calls, prepared.
            admitted (Sequence[bool | None]): Which may run (see :meth:`_admitted`).

        Returns:
            list[ToolReply]: One reply per call, in call order.
        """
        replies: list[ToolReply | None] = [None] * len(prepared)
        started: dict[int, concurrent.futures.Future[Any]] = {}
        no_slot = 0
        for position, (ready, runs) in enumerate(zip(prepared, admitted, strict=True)):
            if runs is None:
                replies[position] = error_reply(
                    ready.action, SKIPPED_REPLY.format(limit=self._max_calls), detail=ready.detail
                )
            elif not runs:
                replies[position] = error_reply(ready.action, NO_ROOM_REPLY, detail=ready.detail)
            elif ready.fetch is not None:
                left = self._time_left(turn)
                if left <= 0:
                    replies[position] = error_reply(ready.action, NO_TIME_REPLY, detail=ready.detail)
                    continue
                try:
                    started[position] = start_bounded(
                        ready.fetch,
                        slots=self._search_slots,
                        name=SEARCH_THREAD_NAME,
                        slot_wait=min(SEARCH_SLOT_WAIT_S, left),
                    )
                except NoSlot:
                    no_slot += 1
                    replies[position] = error_reply(ready.action, NO_CAPACITY_REPLY, detail=ready.detail)
                except Exception as exc:
                    logger.warning("An agent's '{}' call could not start: {}", ready.action, type(exc).__name__)
                    replies[position] = error_reply(ready.action, FAILED_REPLY, detail=ready.detail)
        if no_slot:
            logger.warning("No search slot came free for {} of an agent round's tool calls", no_slot)
        if started:
            # Waited on, never asked for a result with a timeout: a search's own TimeoutError is no deadline.
            concurrent.futures.wait(started.values(), timeout=max(0.0, self._time_left(turn)))
        late = 0
        for position, ready in enumerate(prepared):
            if replies[position] is not None:
                continue
            fetched: Any = None
            future = started.get(position)
            if future is not None:
                if not future.done():
                    late += 1
                    replies[position] = error_reply(ready.action, LATE_REPLY, detail=ready.detail)
                    continue
                failure = future.exception()
                if failure is not None:
                    logger.warning("An agent's '{}' call failed: {}", ready.action, type(failure).__name__)
                    replies[position] = error_reply(ready.action, FAILED_REPLY, detail=ready.detail)
                    continue
                fetched = future.result()
            try:
                replies[position] = self._registry.finish(ready, fetched, turn.ctx)
            except Exception as exc:
                logger.warning("An agent's '{}' call failed: {}", ready.action, type(exc).__name__)
                replies[position] = error_reply(ready.action, FAILED_REPLY, detail=ready.detail)
        if late:
            logger.warning("{} of an agent round's tool calls outlived the turn's time budget", late)
        return [cast(ToolReply, reply) for reply in replies]

    def _bounded(self, reply: ToolReply) -> str:
        """Return a reply's text within ``result_chars``, cut with a marker when longer.

        Args:
            reply (ToolReply): The reply.

        Returns:
            str: The text the transcript holds.
        """
        limit = self._budget.result_chars
        if len(reply.text) <= limit:
            return reply.text
        logger.warning("A '{}' tool reply ran past {} characters and was cut", reply.action, limit)
        return fit_with_marker(reply.text, limit - MARKER_RESERVE_CHARS)[0]
