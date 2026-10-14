"""The research step: the plain model searches the collection until it has enough.

Calls tools with ``tool_choice="required"``; a model that answers in prose
anyway (an endpoint that serves ``required`` as ``auto``) ends research. The
turn reports ``tools_unavailable`` when the endpoint rejects the first call, or
when the first round returns no tool call. The time budget bounds every model
call and every search; a search still running when it runs out is abandoned.
Searches share ``SEARCH_SLOTS`` slots per process, and an abandoned one keeps
its slot until it ends, so no outage can pile up more running searches than
that.
"""

from __future__ import annotations

import concurrent.futures
import contextvars
import threading
import time
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

import openai
from llama_index.core.base.llms.types import ChatMessage, MessageRole, ToolCallBlock
from llama_index.core.tools import FunctionTool
from loguru import logger

from docint.agents.assessment.ledger import EvidenceLedger, digest, is_context_overflow
from docint.agents.assessment.runner import ResearchOutcome, SearchFn, prior_turn_block
from docint.agents.types import AgentProgress, PriorTurn
from docint.utils.embed_chunking import estimate_tokens
from docint.utils.prompt_loader import load_localized_prompt

if TYPE_CHECKING:
    from llama_index.core.llms.function_calling import FunctionCallingLLM
    from llama_index.core.schema import NodeWithScore

RESEARCH_REPLY_TOKENS = 512
TOOL_SCHEMA_TOKENS = 350
#: One round: the call plus a tool reply of five ~400-character digests.
STEP_TOKENS = 700
MESSAGE_OVERHEAD_TOKENS = 40
RESEARCH_WALL_CLOCK_S = 120.0
MAX_QUERY_CHARS = 300
#: Name of the thread a search runs on, so an abandoned one can be found.
SEARCH_THREAD_NAME = "assessment-research-search"
#: Searches that may run at once in this process, abandoned ones included.
SEARCH_SLOTS = 8
#: Longest a search waits for a free slot; the wait comes out of the research budget.
SEARCH_SLOT_WAIT_S = 5.0
#: The process-wide slots every research search takes; a resource cap, not session state.
SEARCH_SLOT_POOL = threading.BoundedSemaphore(SEARCH_SLOTS)

DEFAULT_ASSESSMENT_RESEARCH_PROMPT = (
    "Gather the evidence a careful judgement of the material needs; do not answer yet.\n\n"
    "The user asked:\n{question}\n\n"
    "Previous exchange:\n{prior_turn}\n\n"
    "Passages already found, numbered:\n{digest}\n\n"
    "Tools: search(query) returns new passages; finish() ends research. Call exactly one tool per step.\n"
)


SEARCH_DESCRIPTION = "Search the collection and return new numbered passages. Use a short, specific query."
FINISH_DESCRIPTION = "Stop researching: the passages cover what the judgement needs, or searching finds nothing new."


def _search(query: str) -> str:
    """Schema source for the ``search`` tool; the loop dispatches calls itself.

    Args:
        query (str): What to search for.

    Returns:
        str: The query, unused.
    """
    return query


def _finish() -> str:
    """Schema source for the ``finish`` tool; the loop dispatches calls itself.

    Returns:
        str: Nothing, unused.
    """
    return ""


def _failure_reason(exc: Exception, *, first: bool) -> str:
    """Name why a research call failed, as a static token.

    Args:
        exc (Exception): The failure.
        first (bool): Whether it was the first call of the turn.

    Returns:
        str: ``context_budget``, ``tools_unavailable`` or ``research_failed``.
    """
    if is_context_overflow(exc):
        return "context_budget"
    if first and isinstance(exc, openai.BadRequestError):
        return "tools_unavailable"
    return "research_failed"


class _NoSearchSlot(Exception):
    """Every search slot stayed taken for as long as the search could wait."""


def _bounded(
    search: SearchFn,
    query: str,
    timeout: float,
    *,
    slots: threading.Semaphore,
    clock: Callable[[], float],
) -> concurrent.futures.Future[list[NodeWithScore]] | None:
    """Run one search on its own thread, in a free search slot, and wait for it no longer than ``timeout``.

    The search first waits for a slot, at most ``min(timeout, SEARCH_SLOT_WAIT_S)``,
    and that wait comes out of ``timeout``. The worker gives the slot back when
    retrieval actually ends, before it resolves the future, so an abandoned search
    holds its slot until it finishes and the caller never races its own slot on its
    next search; abandoned work is thus capped at the slot count. The remaining
    deadline is deliberately not passed into retrieval: its clients are the shared
    lookup clients, with timeouts of their own that free the slot when they fire.

    The search runs in a copy of the caller's context, since retrieval reads the
    request's collection and reasoning scope from context variables. Its outcome,
    result or exception, lands on a future that is only waited on, never asked
    for a result with a timeout: a search's own ``TimeoutError`` must not read as
    the budget running out.

    Args:
        search (SearchFn): Runs one search through the turn's engine.
        query (str): What to search for.
        timeout (float): Seconds the slot wait and the search may take together.
        slots (threading.Semaphore): The search slots to take one from.
        clock (Callable[[], float]): Monotonic clock the slot wait is measured with.

    Returns:
        concurrent.futures.Future[list[NodeWithScore]] | None: The finished
        search, holding its passages or its exception; ``None`` when the slot wait
        left no time (the search never starts) or the search outlasts the timeout,
        in which case it finishes in the background, unread, still in its slot.

    Raises:
        _NoSearchSlot: When no slot frees up in time; the search is never called
            and no thread is started.
        Exception: Whatever stops the search thread from starting, once its slot
            has been given back.
    """
    began = clock()
    if not slots.acquire(timeout=min(timeout, SEARCH_SLOT_WAIT_S)):
        raise _NoSearchSlot
    left = timeout - (clock() - began)
    if left <= 0:
        slots.release()
        return None
    future: concurrent.futures.Future[list[NodeWithScore]] = concurrent.futures.Future()
    context = contextvars.copy_context()

    def _run() -> None:
        """Run the search, give its slot back, then hand its outcome to the future."""
        try:
            try:
                found = context.run(search, query)
            finally:
                slots.release()
        except BaseException as exc:
            future.set_exception(exc)
        else:
            future.set_result(found)

    thread = threading.Thread(target=_run, name=SEARCH_THREAD_NAME, daemon=True)
    try:
        thread.start()
    except BaseException:
        slots.release()
        raise
    done, _ = concurrent.futures.wait([future], timeout=left)
    return future if done else None


def _tool_reply(call_id: str, text: str) -> ChatMessage:
    """Build the tool message answering one call.

    Args:
        call_id (str): The id of the call being answered.
        text (str): The reply.

    Returns:
        ChatMessage: The tool message.
    """
    return ChatMessage(role=MessageRole.TOOL, content=text, additional_kwargs={"tool_call_id": call_id})


class ResearchLoop:
    """Let the plain model search the collection until it has enough evidence."""

    def __init__(
        self,
        *,
        llm: FunctionCallingLLM,
        search: SearchFn,
        max_steps: int,
        ctx_window: int,
        char_token_ratio: float,
        prompt: str | None = None,
        wall_clock_s: float = RESEARCH_WALL_CLOCK_S,
        clock: Callable[[], float] = time.monotonic,
        slots: threading.Semaphore = SEARCH_SLOT_POOL,
    ) -> None:
        """Bind the model, the search and the limits.

        Args:
            llm (FunctionCallingLLM): The plain (non-reasoning) chat model; one
                that makes a single attempt per call, so ``wall_clock_s`` is absolute.
            search (SearchFn): Runs one search through the turn's engine.
            max_steps (int): The most model rounds research may take.
            ctx_window (int): The chat model's context window in tokens.
            char_token_ratio (float): Characters per token for budgeting.
            prompt (str | None): Template override; the localized
                ``assessment_research`` prompt by default.
            wall_clock_s (float): Seconds research may take in total, its model
                calls and its searches included.
            clock (Callable[[], float]): Monotonic clock, injectable for tests.
            slots (threading.Semaphore): The search slots shared by every research
                in the process (``SEARCH_SLOT_POOL``); injectable for tests.
        """
        self._llm = llm
        self._search = search
        self._slots = slots
        self._max_steps = max_steps
        self._ratio = char_token_ratio
        self._token_limit = int(ctx_window * 0.95) - RESEARCH_REPLY_TOKENS - TOOL_SCHEMA_TOKENS
        self._prompt = prompt or load_localized_prompt(
            "assessment_research", default=DEFAULT_ASSESSMENT_RESEARCH_PROMPT
        )
        self._wall_clock_s = wall_clock_s
        self._clock = clock
        self._tools = [
            FunctionTool.from_defaults(fn=_search, name="search", description=SEARCH_DESCRIPTION),
            FunctionTool.from_defaults(fn=_finish, name="finish", description=FINISH_DESCRIPTION),
        ]

    def _transcript_tokens(self, transcript: list[ChatMessage]) -> int:
        """Estimate what the transcript costs to resend.

        Args:
            transcript (list[ChatMessage]): The research conversation so far.

        Returns:
            int: Estimated prompt tokens.
        """
        return sum(estimate_tokens(str(m.content or ""), self._ratio) + MESSAGE_OVERHEAD_TOKENS for m in transcript)

    def run(
        self, *, question: str, prior_turn: PriorTurn | None, ledger: EvidenceLedger
    ) -> Iterator[AgentProgress | ResearchOutcome]:
        """Search until the evidence suffices, yielding each search, then one outcome.

        Research reports ``context_budget`` when it ends with the ledger out of room
        before its step cap, or when its last search lost a passage to the room,
        whatever then ended it; a refusal that a later search follows is not reported.

        Args:
            question (str): The user's message, verbatim.
            prior_turn (PriorTurn | None): The previous exchange, if any.
            ledger (EvidenceLedger): The evidence so far; extended in place.

        Yields:
            AgentProgress | ResearchOutcome: One step per search, then the outcome.
        """
        digests = "\n".join(digest(number, scored) for number, scored in enumerate(ledger.entries(), start=1))
        prompt = self._prompt.format(
            question=question.strip(), prior_turn=prior_turn_block(prior_turn), digest=digests or "(none)"
        )
        transcript: list[ChatMessage] = [ChatMessage(role=MessageRole.USER, content=prompt)]
        actions: list[dict[str, Any]] = []
        seen: set[str] = set()
        started = self._clock()
        steps, idle = 0, 0
        refused_last = False
        reason: str | None = None
        while steps < self._max_steps and not ledger.full:
            if self._clock() - started > self._wall_clock_s:
                reason = "time_budget"
                break
            if self._transcript_tokens(transcript) + STEP_TOKENS > self._token_limit:
                reason = "context_budget"
                break
            try:
                response = self._llm.chat_with_tools(
                    self._tools,
                    user_msg=None,
                    chat_history=list(transcript),
                    tool_choice="required",
                    strict=False,
                    allow_parallel_tool_calls=False,
                    max_tokens=RESEARCH_REPLY_TOKENS,
                    # One hung call must not outlast the research budget.
                    timeout=max(1.0, self._wall_clock_s - (self._clock() - started)),
                )
                calls = self._llm.get_tool_calls_from_response(response, error_on_no_tool_call=False)
            except Exception as exc:
                # The call's timeout is the time left, so failing once it is spent is the budget running out.
                spent = self._clock() - started > self._wall_clock_s
                reason = "time_budget" if spent else _failure_reason(exc, first=steps == 0)
                logger.warning("Assessment research stopped ({}): {}", reason, type(exc).__name__)
                break
            if not calls:
                if steps == 0:
                    # Prose in the first round: the endpoint served ``required`` as ``auto``.
                    reason = "tools_unavailable"
                    logger.warning(
                        "Assessment research stopped (tools_unavailable): the model replied without a tool call"
                    )
                break
            steps += 1
            call = calls[0]
            call_id = call.tool_id or f"call_{steps}"
            # Arguments that parse to a list or a string count as no arguments.
            kwargs = call.tool_kwargs if isinstance(call.tool_kwargs, dict) else {}
            transcript.append(
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    blocks=[ToolCallBlock(tool_call_id=call_id, tool_name=call.tool_name, tool_kwargs=dict(kwargs))],
                )
            )
            if call.tool_name == "finish":
                break
            query = str(kwargs.get("query") or "").strip()[:MAX_QUERY_CHARS] if call.tool_name == "search" else ""
            if not query:
                transcript.append(_tool_reply(call_id, "Error: call search with a non-empty query, or call finish."))
                continue
            if query.casefold() in seen:
                break
            seen.add(query.casefold())
            remaining = self._wall_clock_s - (self._clock() - started)
            if remaining <= 0:
                reason = "time_budget"
                break
            try:
                finished = _bounded(self._search, query, remaining, slots=self._slots, clock=self._clock)
            except _NoSearchSlot:
                reason = "research_failed"
                logger.warning("Assessment research stopped (research_failed): no search capacity")
                break
            except Exception as exc:
                # The search thread could not start; its slot is already back, and what research found stays.
                reason = "research_failed"
                logger.warning("Assessment research search could not start: {}", type(exc).__name__)
                break
            if finished is None:
                reason = "time_budget"
                logger.warning("Assessment research stopped (time_budget): a search outlasted the research budget")
                break
            try:
                found = finished.result()
            except Exception as exc:
                reason = "research_failed"
                logger.warning("Assessment research search failed: {}", type(exc).__name__)
                break
            refused_before = ledger.refusals
            added = ledger.add(found)
            refused_last = ledger.refusals > refused_before
            actions.append({"action": "search", "detail": query, "found": len(added)})
            yield AgentProgress(step=steps, action="search", detail=query, found=len(added))
            reply = "\n".join(digest(number, scored) for number, scored in added) or "No new passages."
            transcript.append(_tool_reply(call_id, reply))
            # Checked after every search, the last one a step cap allows included.
            if self._clock() - started > self._wall_clock_s:
                reason = "time_budget"
                break
            idle = 0 if added else idle + 1
            if idle >= 2:
                break
        if reason is None and (refused_last or (steps < self._max_steps and ledger.out_of_room)):
            # The room turned away part of the last search, or ended research before its step cap.
            reason = "context_budget"
        yield ResearchOutcome(steps=steps, actions=actions, reason=reason)
