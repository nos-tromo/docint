"""The research step: the plain model searches the collection until it has enough.

Calls tools with ``tool_choice="required"``; a model that answers in prose
anyway (an endpoint that serves ``required`` as ``auto``) ends research. The
turn reports ``tools_unavailable`` when the endpoint rejects the first call, or
when the first round returns no tool call. The time budget bounds every model
call and every search, and model calls run like searches: each on its own
thread, abandoned when the budget runs out, and a reply that lands after it is
not acted on. Searches share ``SEARCH_SLOTS`` slots per process and model calls
``MODEL_SLOTS``; an abandoned call keeps its slot until it ends, so no outage
can pile up more running calls than that.

Besides ``search`` and ``finish``, a turn whose collection has a ready keyword
index offers ``keyword``: an exact word match that reports how many indexed
chunks contain the words (:mod:`~docint.agents.assessment.keyword`). It runs on
the search slots, and its count reaches the answer as a fact.
"""

from __future__ import annotations

import concurrent.futures
import functools
import json
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import openai
from llama_index.core.base.llms.types import ChatMessage, MessageRole, ToolCallBlock
from llama_index.core.bridge.pydantic import BaseModel, Field, create_model
from llama_index.core.tools import FunctionTool
from loguru import logger

from docint.agents.assessment.bounded import NoSlot, run_bounded
from docint.agents.assessment.keyword import (
    DEFAULT_KEYWORD_FIELD,
    LONG_WORD,
    MAX_KEYWORD_CHARS,
    NO_USABLE_WORD,
    ONE_VALUE_ONLY,
    UNINDEXABLE_WORD,
    KeywordResult,
    KeywordTool,
    anchor_terms,
    check_key,
    keyword_reply,
)
from docint.agents.assessment.ledger import EvidenceLedger, digest, is_context_overflow
from docint.agents.assessment.runner import ResearchOutcome, SearchFn, prior_turn_block
from docint.agents.types import AgentProgress, PriorTurn
from docint.core.search.index import SEARCH_MAX_TOKEN_LEN
from docint.utils.embed_chunking import estimate_tokens
from docint.utils.prompt_loader import load_localized_prompt

if TYPE_CHECKING:
    from llama_index.core.llms.function_calling import FunctionCallingLLM
    from llama_index.core.schema import NodeWithScore

T = TypeVar("T")

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
#: Name of the thread a model call runs on, so an abandoned one can be found.
MODEL_THREAD_NAME = "assessment-research-model"
#: Research model calls that may run at once in this process, abandoned ones included.
MODEL_SLOTS = 8
#: Longest a model call waits for a free slot; the wait comes out of the research budget.
MODEL_SLOT_WAIT_S = 5.0
#: The process-wide slots every research model call takes; a resource cap, not session state.
MODEL_SLOT_POOL = threading.BoundedSemaphore(MODEL_SLOTS)

DEFAULT_ASSESSMENT_RESEARCH_PROMPT = (
    "Gather the evidence a careful judgement of the material needs; do not answer yet.\n\n"
    "The user asked:\n{question}\n\n"
    "Previous exchange:\n{prior_turn}\n\n"
    "Passages already found, numbered:\n{digest}\n\n"
    "Tools:\n- search(query) returns new passages.\n{keyword_tool}- finish() ends research.\n"
    "Call exactly one tool per step.\n"
)
DEFAULT_ASSESSMENT_RESEARCH_KEYWORD_PROMPT = (
    "- keyword(words) counts the indexed chunks that contain every word, in any order, and returns "
    "the most relevant new ones; a count of 0 is exact."
)
DEFAULT_ASSESSMENT_RESEARCH_KEYWORD_FIELDS_PROMPT = "It also takes field, one of {fields}; text is the default."

#: What each field matches, for the schema of a tool that offers more than the text.
_FIELD_NOTES = {
    "text": "the passage text (the default)",
    "author": "who wrote a posting",
    "network": "the platform",
    "uuid": "one posting's id",
}


SEARCH_DESCRIPTION = "Search the collection and return new numbered passages. Use a short, specific query."
KEYWORD_DESCRIPTION = (
    "Count the indexed chunks that contain every one of these words, in any order (a word also matches longer "
    "words it begins), and return the most relevant new ones. For exact names, numbers, ids and terms, and to "
    "check whether the material mentions something at all: a count of 0 is exact."
)
FINISH_DESCRIPTION = "Stop researching: the passages cover what the judgement needs, or searching finds nothing new."


@dataclass(frozen=True)
class _Stopped:
    """Why research stops instead of reading a search-slot call's result.

    Attributes:
        reason (str): The static stop token.
    """

    reason: str


def _search(query: str) -> str:
    """Schema source for the ``search`` tool; the loop dispatches calls itself.

    Args:
        query (str): What to search for.

    Returns:
        str: The query, unused.
    """
    return query


def _keyword(words: str, field: str = DEFAULT_KEYWORD_FIELD) -> str:
    """Schema source for the ``keyword`` tool; the loop dispatches calls itself.

    Args:
        words (str): The words every match must contain.
        field (str): What to match them in.

    Returns:
        str: The words, unused.
    """
    return words


def _keyword_schema(fields: Sequence[str]) -> type[BaseModel]:
    """Build the ``keyword`` tool's arguments, its field limited to the ready ones.

    A tool that offers only the text takes no field at all: naming fields the
    collection does not hold would only invite calls that fail.

    Args:
        fields (Sequence[str]): The fields whose indexes are ready, ``text`` first.

    Returns:
        type[BaseModel]: The argument model the tool schema is rendered from.
    """
    words = (str, Field(description="The words every match must contain, in any order."))
    if tuple(fields) == (DEFAULT_KEYWORD_FIELD,):
        return create_model("keyword", words=words)
    choices: dict[str, Any] = {"enum": list(fields)}
    notes = "; ".join(f"{name}: {_FIELD_NOTES[name]}" for name in fields if name in _FIELD_NOTES)
    return create_model(
        "keyword",
        words=words,
        field=(str, Field(default=DEFAULT_KEYWORD_FIELD, description=f"{notes}.", json_schema_extra=choices)),
    )


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


def _bounded(
    search: SearchFn,
    query: str,
    timeout: float,
    *,
    slots: threading.Semaphore,
    clock: Callable[[], float],
) -> concurrent.futures.Future[list[NodeWithScore]] | None:
    """Run one search as :func:`~docint.agents.assessment.bounded.run_bounded` runs work, on a search slot.

    The remaining deadline is not passed into retrieval: its clients are the shared
    lookup clients, with timeouts of their own that free the slot when they fire.

    Args:
        search (SearchFn): Runs one search through the turn's engine.
        query (str): What to search for.
        timeout (float): Seconds the slot wait and the search may take together.
        slots (threading.Semaphore): The search slots to take one from.
        clock (Callable[[], float]): Monotonic clock the slot wait is measured with.

    Returns:
        concurrent.futures.Future[list[NodeWithScore]] | None: The finished search,
        holding its passages or its exception; ``None`` when it never started or
        outlasted the timeout, in which case it finishes unread, still in its slot.

    Raises:
        NoSlot: When no search slot frees up in time; the search is never called.
        Exception: Whatever stops the search from starting, once its slot is back.
    """
    return run_bounded(
        lambda: search(query), timeout, slots=slots, clock=clock, name=SEARCH_THREAD_NAME, slot_wait=SEARCH_SLOT_WAIT_S
    )


def _bounded_check(
    keyword: KeywordTool,
    words: str,
    field: str,
    timeout: float,
    *,
    slots: threading.Semaphore,
    clock: Callable[[], float],
) -> concurrent.futures.Future[KeywordResult] | None:
    """Run one keyword check as :func:`_bounded` runs a search, on a search slot.

    Args:
        keyword (KeywordTool): The turn's keyword tool.
        words (str): The words to match.
        field (str): The field to match them in.
        timeout (float): Seconds the slot wait and the check may take together.
        slots (threading.Semaphore): The search slots to take one from.
        clock (Callable[[], float]): Monotonic clock the slot wait is measured with.

    Returns:
        concurrent.futures.Future[KeywordResult] | None: As :func:`_bounded`.

    Raises:
        NoSlot: When no search slot frees up in time; the check is never run.
        Exception: Whatever stops the check from starting, once its slot is back.
    """
    return run_bounded(
        functools.partial(keyword.run, words, field),
        timeout,
        slots=slots,
        clock=clock,
        name=SEARCH_THREAD_NAME,
        slot_wait=SEARCH_SLOT_WAIT_S,
    )


def _keyword_request(keyword: KeywordTool, kwargs: Mapping[str, Any]) -> tuple[str, str, str | None]:
    """Read a keyword call's words and field, and say why it cannot run if it cannot.

    Args:
        keyword (KeywordTool): The turn's keyword tool.
        kwargs (Mapping[str, Any]): The call's arguments.

    Returns:
        tuple[str, str, str | None]: The words on one line, the field, and the
        error reply when the call cannot run, else ``None``.
    """
    words = " ".join(str(kwargs.get("words") or "").split())
    field = str(kwargs.get("field") or DEFAULT_KEYWORD_FIELD)
    if not words:
        return words, field, "Error: call keyword with the words to match, or call search or finish."
    if len(words) > MAX_KEYWORD_CHARS:
        return words, field, f"Error: keep the words within {MAX_KEYWORD_CHARS} characters; use the most telling ones."
    if field not in keyword.fields:
        return words, field, f"Error: field must be one of: {', '.join(keyword.fields)}."
    problem = keyword.check(words, field)
    if problem == NO_USABLE_WORD:
        return words, field, "Error: give at least one word of two or more characters."
    if problem == ONE_VALUE_ONLY:
        return words, field, f"Error: {field} matches one exact value; give a single one."
    if problem == UNINDEXABLE_WORD:
        return (
            words,
            field,
            "Error: the index cannot match a single letter or digit, such as the D in vitamin D, nor a word it "
            "splits on dots and signs, such as 2.5, U.S. or p<0.05; check the other words only, or use search.",
        )
    if problem == LONG_WORD:
        return (
            words,
            field,
            f"Error: the index reads a word only up to its first {SEARCH_MAX_TOKEN_LEN} characters, so a longer "
            "one cannot be checked exactly; check a shorter beginning of it, which also matches the longer words "
            "it begins.",
        )
    return words, field, None


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
        model_slots: threading.Semaphore = MODEL_SLOT_POOL,
        keyword: KeywordTool | None = None,
        keyword_prompt: str | None = None,
        keyword_fields_prompt: str | None = None,
    ) -> None:
        """Bind the model, the search and the limits.

        Args:
            llm (FunctionCallingLLM): The plain (non-reasoning) chat model. Each call
                is abandoned at the deadline; one making a single attempt per call
                ends an abandoned call, and frees its slot, at its own timeout.
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
            model_slots (threading.Semaphore): The model-call slots shared by every
                research in the process (``MODEL_SLOT_POOL``); injectable for tests.
            keyword (KeywordTool | None): The keyword match the turn offers;
                ``None`` keeps research to ``search`` and ``finish``.
            keyword_prompt (str | None): Template override for the prose that
                describes ``keyword``; the localized ``assessment_research_keyword``
                prompt by default.
            keyword_fields_prompt (str | None): Template override for the prose
                that names the fields, added only when more than the text is
                offered; the localized ``assessment_research_keyword_fields``
                prompt by default.
        """
        self._llm = llm
        self._search = search
        self._keyword = keyword
        self._slots = slots
        self._model_slots = model_slots
        self._max_steps = max_steps
        self._ratio = char_token_ratio
        self._prompt = prompt or load_localized_prompt(
            "assessment_research", default=DEFAULT_ASSESSMENT_RESEARCH_PROMPT
        )
        self._keyword_line = ""
        self._wall_clock_s = wall_clock_s
        self._clock = clock
        self._tools = [FunctionTool.from_defaults(fn=_search, name="search", description=SEARCH_DESCRIPTION)]
        schema_tokens = TOOL_SCHEMA_TOKENS
        if keyword is not None:
            # Some chat templates drop the schemas, so the prose names the tool too.
            parts = [
                keyword_prompt
                or load_localized_prompt(
                    "assessment_research_keyword", default=DEFAULT_ASSESSMENT_RESEARCH_KEYWORD_PROMPT
                )
            ]
            if len(keyword.fields) > 1:
                fields_template = keyword_fields_prompt or load_localized_prompt(
                    "assessment_research_keyword_fields", default=DEFAULT_ASSESSMENT_RESEARCH_KEYWORD_FIELDS_PROMPT
                )
                parts.append(fields_template.format(fields=", ".join(keyword.fields)))
            self._keyword_line = " ".join(parts) + "\n"
            tool = FunctionTool.from_defaults(
                fn=_keyword, name="keyword", description=KEYWORD_DESCRIPTION, fn_schema=_keyword_schema(keyword.fields)
            )
            self._tools.append(tool)
            # Measured, not guessed: the schema rides along on every research request.
            schema = json.dumps(tool.metadata.to_openai_tool(skip_length_check=True))
            schema_tokens += estimate_tokens(schema, char_token_ratio)
        self._tools.append(FunctionTool.from_defaults(fn=_finish, name="finish", description=FINISH_DESCRIPTION))
        self._token_limit = int(ctx_window * 0.95) - RESEARCH_REPLY_TOKENS - schema_tokens

    def _transcript_tokens(self, transcript: list[ChatMessage]) -> int:
        """Estimate what the transcript costs to resend.

        Args:
            transcript (list[ChatMessage]): The research conversation so far.

        Returns:
            int: Estimated prompt tokens.
        """
        return sum(estimate_tokens(str(m.content or ""), self._ratio) + MESSAGE_OVERHEAD_TOKENS for m in transcript)

    def _call_search(
        self, start: Callable[[float], concurrent.futures.Future[T] | None], *, started: float, label: str
    ) -> T | _Stopped:
        """Run one call on a search slot within the research budget and read its outcome.

        Args:
            start (Callable[[float], concurrent.futures.Future[T] | None]): Starts
                the call given the seconds left, as :func:`_bounded` does.
            started (float): When research started, on the research clock.
            label (str): What the call is, for the log: ``search`` or ``keyword check``.

        Returns:
            T | _Stopped: The call's result, or why research stops.
        """
        remaining = self._wall_clock_s - (self._clock() - started)
        if remaining <= 0:
            return _Stopped("time_budget")
        try:
            finished = start(remaining)
        except NoSlot:
            logger.warning("Assessment research stopped (research_failed): no search capacity")
            return _Stopped("research_failed")
        except Exception as exc:
            # The call could not start; its slot is already back, and what research found stays.
            logger.warning("Assessment research {} could not start: {}", label, type(exc).__name__)
            return _Stopped("research_failed")
        if finished is None:
            # The slot wait spent the budget, or the call outlasted it.
            logger.warning(
                "Assessment research stopped (time_budget): no {} result arrived within the research budget", label
            )
            return _Stopped("time_budget")
        try:
            return finished.result()
        except Exception as exc:
            logger.warning("Assessment research {} failed: {}", label, type(exc).__name__)
            return _Stopped("research_failed")

    def run(
        self, *, question: str, prior_turn: PriorTurn | None, ledger: EvidenceLedger
    ) -> Iterator[AgentProgress | ResearchOutcome]:
        """Search until the evidence suffices, yielding each search, then one outcome.

        Research reports ``context_budget`` when it ends with the ledger out of room
        before its step cap, or when its last search lost a passage to the room,
        whatever then ended it; a refusal that a later search follows is not reported.
        A keyword check is a search here: it runs on the search slots, adds its
        passages the same way, and ends research when repeated. Its count is a
        fact even when it adds nothing, so only a check whose count failed and
        that added nothing counts toward the two idle rounds that end research.

        Args:
            question (str): The question as the prompts show it: the user's message and,
                when the conversation resolved more, the labelled rewrite (see
                :func:`~docint.agents.assessment.runner.question_block`).
            prior_turn (PriorTurn | None): The previous exchange, if any.
            ledger (EvidenceLedger): The evidence so far; extended in place.

        Yields:
            AgentProgress | ResearchOutcome: One step per search or keyword check, then the outcome.
        """
        digests = "\n".join(digest(number, scored) for number, scored in enumerate(ledger.entries(), start=1))
        prompt = self._prompt.format(
            question=question.strip(),
            prior_turn=prior_turn_block(prior_turn),
            digest=digests or "(none)",
            keyword_tool=self._keyword_line,
        )
        transcript: list[ChatMessage] = [ChatMessage(role=MessageRole.USER, content=prompt)]
        actions: list[dict[str, Any]] = []
        seen: set[str] = set()
        checked: set[tuple[str, frozenset[str], frozenset[Any]]] = set()
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
            remaining = self._wall_clock_s - (self._clock() - started)
            try:
                pending = run_bounded(
                    functools.partial(
                        self._llm.chat_with_tools,
                        self._tools,
                        user_msg=None,
                        chat_history=list(transcript),
                        tool_choice="required",
                        strict=False,
                        allow_parallel_tool_calls=False,
                        max_tokens=RESEARCH_REPLY_TOKENS,
                        # The SDK's own bound, which ends an abandoned call and frees its slot.
                        timeout=max(1.0, remaining),
                    ),
                    remaining,
                    slots=self._model_slots,
                    clock=self._clock,
                    name=MODEL_THREAD_NAME,
                    slot_wait=MODEL_SLOT_WAIT_S,
                )
            except NoSlot:
                reason = "research_failed"
                logger.warning("Assessment research stopped (research_failed): no model capacity")
                break
            except Exception as exc:
                reason = "research_failed"
                logger.warning("Assessment research model call could not start: {}", type(exc).__name__)
                break
            if pending is None:
                # The slot wait spent the budget, or the call outlasted it.
                reason = "time_budget"
                logger.warning(
                    "Assessment research stopped (time_budget): no model reply arrived within the research budget"
                )
                break
            try:
                calls = self._llm.get_tool_calls_from_response(pending.result(), error_on_no_tool_call=False)
            except Exception as exc:
                # A call that fails once the budget is spent, however it failed, ran out of time.
                spent = self._clock() - started > self._wall_clock_s
                reason = "time_budget" if spent else _failure_reason(exc, first=steps == 0)
                logger.warning("Assessment research stopped ({}): {}", reason, type(exc).__name__)
                break
            # A reply that lands after the budget is not acted on.
            if self._clock() - started > self._wall_clock_s:
                reason = "time_budget"
                logger.warning("Assessment research stopped (time_budget): the model replied after the research budget")
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
            if call.tool_name == "keyword" and self._keyword is not None:
                words, field, error = _keyword_request(self._keyword, kwargs)
                if error is not None:
                    transcript.append(_tool_reply(call_id, error))
                    continue
                key = check_key(words, field)
                if key in checked:
                    break
                checked.add(key)
                result = self._call_search(
                    functools.partial(
                        _bounded_check, self._keyword, words, field, slots=self._slots, clock=self._clock
                    ),
                    started=started,
                    label="keyword check",
                )
                if isinstance(result, _Stopped):
                    reason = result.reason
                    break
                found = result.nodes
                if field == DEFAULT_KEYWORD_FIELD:
                    found = [anchor_terms(scored, result.keywords) for scored in found]
                refused_before = ledger.refusals
                added = ledger.add(found)
                refused_last = ledger.refusals > refused_before
                action: dict[str, Any] = {
                    "action": "keyword",
                    "detail": words,
                    "field": field,
                    "found": len(added),
                    "matches": result.matches,
                }
                if result.unchecked > 0:
                    action["unchecked"] = result.unchecked
                actions.append(action)
                yield AgentProgress(
                    step=steps, action="keyword", detail=words, found=len(added), field=field, matches=result.matches
                )
                reply = keyword_reply(
                    matches=result.matches,
                    unchecked=result.unchecked,
                    digests=[digest(number, scored) for number, scored in added],
                )
                productive = bool(added) or result.matches is not None
            else:
                query = str(kwargs.get("query") or "").strip()[:MAX_QUERY_CHARS] if call.tool_name == "search" else ""
                if not query:
                    transcript.append(
                        _tool_reply(call_id, "Error: call search with a non-empty query, or call finish.")
                    )
                    continue
                if query.casefold() in seen:
                    break
                seen.add(query.casefold())
                found_or_stop = self._call_search(
                    functools.partial(_bounded, self._search, query, slots=self._slots, clock=self._clock),
                    started=started,
                    label="search",
                )
                if isinstance(found_or_stop, _Stopped):
                    reason = found_or_stop.reason
                    break
                refused_before = ledger.refusals
                added = ledger.add(found_or_stop)
                refused_last = ledger.refusals > refused_before
                actions.append({"action": "search", "detail": query, "found": len(added)})
                yield AgentProgress(step=steps, action="search", detail=query, found=len(added))
                reply = "\n".join(digest(number, scored) for number, scored in added) or "No new passages."
                productive = bool(added)
            transcript.append(_tool_reply(call_id, reply))
            # Checked after every search, the last one a step cap allows included.
            if self._clock() - started > self._wall_clock_s:
                reason = "time_budget"
                break
            idle = 0 if productive else idle + 1
            if idle >= 2:
                break
        if reason is None and (refused_last or (steps < self._max_steps and ledger.out_of_room)):
            # The room turned away part of the last search, or ended research before its step cap.
            reason = "context_budget"
        yield ResearchOutcome(steps=steps, actions=actions, reason=reason)
