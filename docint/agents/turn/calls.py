"""Answer the tool calls of one agent round: which may run, their I/O at once, every reply in call order.

The calls of one round are prepared, fetched and finished as the tool registry requires (see
:mod:`docint.agents.turn.tools`): prepared in call order on the loop thread, their I/O started on
the process-wide search slots so it runs at once, then finished in call order on the loop thread,
which numbers what they admit in call order. Every call gets a reply: calls past the round's limit
are skipped, a call without room for a full reply never runs, and a call that cannot start, fails
or outlives the time left gets an error reply. Work abandoned at the deadline keeps its search slot
until it ends. A call that could not be prepared, whose I/O failed, could not start or found no
search slot, or whose answer raised, is counted as failed: the evidence it would have found is
missing. One that ran out of time is not; the turn's time limit says so.

Room. A round first reserves the least reply every call needs (:meth:`RoundCalls.least`), which
the loop has already checked fits; a call then runs only while a full reply, ``result_chars`` at
the turn's ratio plus its envelope, still fits in place of its short one. No reply leaves here
longer than ``result_chars``.

Nothing here logs a query, a tool argument or document text: only counts, tool names and
exception types.
"""

from __future__ import annotations

import concurrent.futures
import math
import threading
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any, NamedTuple, cast

from loguru import logger

from docint.agents.assessment.bounded import SEARCH_SLOT_WAIT_S, NoSlot, start_bounded
from docint.agents.turn.budget import MESSAGE_OVERHEAD_TOKENS, message_cost
from docint.agents.turn.stream import ToolCall
from docint.agents.turn.tools import NAME_CHARS, PreparedCall, ToolContext, ToolRegistry, ToolReply, error_reply
from docint.utils.text_budget import MARKER_RESERVE_CHARS, fit_with_marker

#: Name of the thread a tool call's I/O runs on, so an abandoned one can be found.
SEARCH_THREAD_NAME = "agent-turn-search"

# The replies written here, for calls that did not run or did not answer in time; model-facing
# protocol like the tools' own, each prefixed ``Error: `` by ``error_reply``.
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


class PreparedRound(NamedTuple):
    """One round's calls, prepared in call order.

    Attributes:
        calls (list[PreparedCall]): One per call; a call whose ``prepare`` raised carries the
            failed-call reply.
        failed (int): Calls whose ``prepare`` raised.
    """

    calls: list[PreparedCall]
    failed: int


class RoundReplies(NamedTuple):
    """The replies to one round's calls.

    Attributes:
        replies (list[ToolReply]): One per call, in call order, each within ``result_chars``.
        failed (int): Calls that could not be prepared, whose I/O failed, could not start or
            found no search slot, or whose answer raised.
    """

    replies: list[ToolReply]
    failed: int


class RoundCalls:
    """Answer each round's tool calls within the room and the time the turn has left."""

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        result_chars: int,
        max_calls_per_round: int,
        char_token_ratio: float,
        search_slots: threading.Semaphore,
    ) -> None:
        """Bind the tools, the reply size and the per-round limit.

        Args:
            registry (ToolRegistry): The tools the turn offers.
            result_chars (int): Longest one reply, in characters.
            max_calls_per_round (int): Calls one round may run; the rest are skipped.
            char_token_ratio (float): Characters per token, the ratio the budget was measured with.
            search_slots (threading.Semaphore): The slots tool I/O runs in, shared by the process.

        Raises:
            ValueError: If ``char_token_ratio`` is not positive.
        """
        if char_token_ratio <= 0:
            raise ValueError("char_token_ratio must be positive")
        self._registry = registry
        self._result_chars = result_chars
        self._max_calls = max_calls_per_round
        self._ratio = char_token_ratio
        self._search_slots = search_slots
        #: What a full reply costs: ``result_chars`` at the ratio, and its envelope.
        self.full_reply = math.ceil(result_chars / char_token_ratio) + MESSAGE_OVERHEAD_TOKENS
        short = (
            SKIPPED_REPLY.format(limit=max_calls_per_round),
            NO_ROOM_REPLY,
            NO_TIME_REPLY,
            LATE_REPLY,
            FAILED_REPLY,
            NO_CAPACITY_REPLY,
        )
        #: What the longest reply written here costs.
        self.short_reply = max(message_cost(error_reply("", text).text, char_token_ratio) for text in short)

    def prepare(self, calls: Sequence[ToolCall], ctx: ToolContext) -> PreparedRound:
        """Prepare a round's calls on the loop thread, in call order.

        A call past the round's limit is not prepared at all: it is answered with the skipped
        reply and never runs. A call whose ``prepare`` raises is answered with the failed-call
        reply, as one whose ``finish`` raises is, and counted as failed: the round's other calls
        still run.

        Args:
            calls (Sequence[ToolCall]): The round's calls, as the model made them.
            ctx (ToolContext): The turn's context, read and never written.

        Returns:
            PreparedRound: The prepared calls, and how many could not be prepared.
        """
        prepared: list[PreparedCall] = []
        failed = 0
        for position, call in enumerate(calls):
            action = call.name.strip()[:NAME_CHARS]
            if position >= self._max_calls:
                prepared.append(PreparedCall(action=action, detail="", reply=self._skipped(action, "")))
                continue
            try:
                prepared.append(self._registry.prepare(call, ctx))
            except Exception as exc:
                logger.warning("An agent tool call could not be prepared: {}", type(exc).__name__)
                failed += 1
                prepared.append(PreparedCall(action=action, detail="", reply=error_reply(action, FAILED_REPLY)))
        return PreparedRound(calls=prepared, failed=failed)

    def _skipped(self, action: str, detail: str) -> ToolReply:
        """Build the reply to a call past the round's limit.

        Args:
            action (str): The call's tool name, as the model wrote it.
            detail (str): The call's main argument, for the trail.

        Returns:
            ToolReply: The skipped reply, an error reply.
        """
        return error_reply(action, SKIPPED_REPLY.format(limit=self._max_calls), detail=detail)

    def answer(
        self, prepared: PreparedRound, ctx: ToolContext, *, room: int, time_left: Callable[[], float]
    ) -> RoundReplies:
        """Answer a round's calls, one reply per call in call order.

        Args:
            prepared (PreparedRound): The round's calls, prepared in call order (see :meth:`prepare`).
            ctx (ToolContext): The turn's evidence, budget, LLM view and memory.
            room (int): Tokens the round's replies may take: the turn's room less the transcript
                so far and the round's assistant message. Never less than :meth:`least`, which
                every call's reply needs: the loop withdraws a round that does not fit.
            time_left (Callable[[], float]): Reads the seconds the turn has left.

        Returns:
            RoundReplies: The replies, each within ``result_chars``, and how many calls failed.
        """
        assert room >= self.least(prepared), "a round that does not fit is withdrawn before it is answered"
        calls = prepared.calls
        replies, failed = self._replies(calls, self._admitted(calls, room), ctx, time_left)
        return RoundReplies(replies=[self._bounded(reply) for reply in replies], failed=prepared.failed + failed)

    def least(self, prepared: PreparedRound) -> int:
        """Return the fewest tokens a round's replies can take: one reply per call, each as short as it can be.

        A call answered when prepared costs its reply; any other, past the round's limit or not,
        at least the longest reply written here, which is how it is answered when it cannot run.

        Args:
            prepared (PreparedRound): The round's calls, prepared.

        Returns:
            int: The tokens.
        """
        return sum(self._least(position, ready) for position, ready in enumerate(prepared.calls))

    def _least(self, position: int, ready: PreparedCall) -> int:
        """Return the fewest tokens one call's reply can take.

        Args:
            position (int): The call's place in its round, from zero.
            ready (PreparedCall): The call, prepared.

        Returns:
            int: Its reply's cost when it was answered at prepare within the round's limit, else
            the cost of the longest reply written here.
        """
        if position < self._max_calls and ready.reply is not None:
            return message_cost(ready.reply.text, self._ratio)
        return self.short_reply

    def _admitted(self, prepared: Sequence[PreparedCall], room: int) -> list[bool | None]:
        """Decide which calls may run: within the round's limit, and while a full reply still fits.

        The round first reserves the least each call's reply costs (see :meth:`least`). A call
        with work then runs, in call order, only while the rest of the room holds its full reply
        in place of the short one reserved for it.

        Args:
            prepared (Sequence[PreparedCall]): The round's calls, prepared.
            room (int): Tokens the round's replies may take.

        Returns:
            list[bool | None]: Per call: ``None`` when it is past the round's limit, ``True`` when
            it runs (or was answered when prepared), ``False`` when its reply has no room.
        """
        room -= sum(self._least(position, ready) for position, ready in enumerate(prepared))
        admitted: list[bool | None] = []
        for position, ready in enumerate(prepared):
            if position >= self._max_calls:
                admitted.append(None)
            elif ready.reply is not None:
                admitted.append(True)
            elif room >= self.full_reply - self.short_reply:
                room -= self.full_reply - self.short_reply
                admitted.append(True)
            else:
                admitted.append(False)
        return admitted

    def _replies(
        self,
        prepared: Sequence[PreparedCall],
        admitted: Sequence[bool | None],
        ctx: ToolContext,
        time_left: Callable[[], float],
    ) -> tuple[list[ToolReply], int]:
        """Run the admitted calls' I/O at once on the search slots, then finish every call in call order.

        Args:
            prepared (Sequence[PreparedCall]): The round's calls, prepared.
            admitted (Sequence[bool | None]): Which may run (see :meth:`_admitted`).
            ctx (ToolContext): The turn's evidence, budget, LLM view and memory.
            time_left (Callable[[], float]): Reads the seconds the turn has left.

        Returns:
            tuple[list[ToolReply], int]: One reply per call, in call order, and how many calls
            failed: their I/O raised or could not start, no search slot came free for it, or
            their answer raised.
        """
        replies: list[ToolReply | None] = [None] * len(prepared)
        started: dict[int, concurrent.futures.Future[Any]] = {}
        no_slot, failed = 0, 0
        for position, (ready, runs) in enumerate(zip(prepared, admitted, strict=True)):
            if runs is None:
                replies[position] = self._skipped(ready.action, ready.detail)
            elif not runs:
                replies[position] = error_reply(ready.action, NO_ROOM_REPLY, detail=ready.detail)
            elif ready.fetch is not None:
                left = time_left()
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
                    failed += 1
                    replies[position] = error_reply(ready.action, FAILED_REPLY, detail=ready.detail)
        if no_slot:
            logger.warning("No search slot came free for {} of an agent round's tool calls", no_slot)
        if started:
            # Waited on, never asked for a result with a timeout: a search's own TimeoutError is no deadline.
            concurrent.futures.wait(started.values(), timeout=max(0.0, time_left()))
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
                    failed += 1
                    replies[position] = error_reply(ready.action, FAILED_REPLY, detail=ready.detail)
                    continue
                fetched = future.result()
            try:
                replies[position] = self._registry.finish(ready, fetched, ctx)
            except Exception as exc:
                logger.warning("An agent's '{}' call failed: {}", ready.action, type(exc).__name__)
                failed += 1
                replies[position] = error_reply(ready.action, FAILED_REPLY, detail=ready.detail)
        if late:
            logger.warning("{} of an agent round's tool calls outlived the turn's time budget", late)
        return [cast(ToolReply, reply) for reply in replies], no_slot + failed

    def _bounded(self, reply: ToolReply) -> ToolReply:
        """Return a reply within ``result_chars``, its text cut with a marker when longer.

        Args:
            reply (ToolReply): The reply.

        Returns:
            ToolReply: The reply the transcript holds.
        """
        if len(reply.text) <= self._result_chars:
            return reply
        logger.warning("A '{}' tool reply ran past {} characters and was cut", reply.action, self._result_chars)
        return replace(reply, text=fit_with_marker(reply.text, self._result_chars - MARKER_RESERVE_CHARS)[0])
