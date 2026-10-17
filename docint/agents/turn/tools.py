"""The tools an agent turn offers the model, and the registry that answers every call it makes.

Every reply is text the model can read. A search shows the passages new to the turn under
the numbers the answer cites them by, names the passages it showed before by number alone
(with a note of their own when they grew), and says how many more matched than it had room
for; those stay unnumbered, so every number the model can cite was shown to it. ``read(n)``
shows one passage again, as much of it as one reply holds. A call the registry cannot run
(a tool the turn does not offer, arguments that are not a JSON object) gets an error reply
saying what to call instead, and a call made before is answered by number from memory,
never run again; a read only while its passage is unchanged.

A call is answered in two steps, so a turn can run the I/O of several calls at once:
:meth:`ToolRegistry.prepare` checks a call and answers it at once (an error, or a call made
before) or returns the work it needs, and :meth:`ToolRegistry.finish` admits and renders.
Only a prepared call's ``fetch``, its I/O, is meant to run off the loop thread: it reads
neither the turn's context nor its ledger. Nothing else here is thread-safe, so ``prepare``
and ``finish`` run on the loop thread, each in the order the model made the calls.
:meth:`ToolRegistry.call` does the steps one after another.
"""

from __future__ import annotations

import copy
import functools
import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from llama_index.core.schema import BaseNode, MetadataMode, NodeWithScore
from loguru import logger

from docint.agents.assessment.ledger import EvidenceLedger
from docint.agents.turn.budget import TurnBudget
from docint.agents.turn.passages import answer_passages, refs, render_entry
from docint.agents.turn.stream import ToolCall
from docint.utils.text_budget import MATCH_SPANS_KEY

SearchFn = Callable[[str], list[NodeWithScore]]
"""Run one search through the turn's engine and return its ranked passages."""

#: Longest main argument a reply carries for the turn's trail.
DETAIL_CHARS = 200
#: Longest query a search sends to retrieval.
MAX_QUERY_CHARS = 300
#: Longest tool name an error reply repeats back.
NAME_CHARS = 64
#: A passage number as a model may write it: ``3``, or ``[3]`` as it is cited.
_NUMBER_RE = re.compile(r"\s*\[?\s*(\d{1,6})\s*\]?\s*")
#: First number no passage can have; a reply never repeats a number this long back.
_NUMBER_LIMIT = 1_000_000

SEARCH_DESCRIPTION = (
    "Search the collection. Returns the passages new to this turn, each under the number [n] to cite it by, "
    "and names passages shown before by their number. Use a few specific words; search again with other words "
    "to find more."
)
SEARCH_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {"query": {"type": "string", "description": "A few specific words to look for."}},
    "required": ["query"],
}
READ_DESCRIPTION = (
    "Show the text of passage [n], one shown before, as much of it as one reply holds; "
    "use it when a search showed only part of a passage."
)
READ_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {"n": {"type": "integer", "description": "The passage's number, as in [n]."}},
    "required": ["n"],
}


@dataclass(frozen=True)
class ToolReply:
    """What one tool call is answered with.

    Attributes:
        text (str): The reply the model reads.
        found (int): Passages the call admitted into the evidence or extended there.
        action (str): The tool's name, for the turn's trail; for a call naming no offered
            tool, the name the model wrote, cut to ``NAME_CHARS``.
        detail (str): The call's main argument, cut to ``DETAIL_CHARS``, for the trail;
            empty when it has none to show. Shown to the user, never logged.
    """

    text: str
    found: int
    action: str
    detail: str


@dataclass(frozen=True)
class PassageReply(ToolReply):
    """A reply that answered its call by showing or naming numbered passages.

    The registry remembers its numbers, so the same call made again is answered by number.

    Attributes:
        numbers (tuple[int, ...]): The ledger numbers the reply showed or named, ascending.
        depends_on (tuple[int, ...]): Entries the reply shows as they stand, as a read
            does: the same call is answered from memory only while each is unchanged, and
            answered anew once one of them has grown.
    """

    numbers: tuple[int, ...] = ()
    depends_on: tuple[int, ...] = ()


@dataclass(frozen=True)
class _Answered:
    """What the registry remembers of a call it answered.

    Attributes:
        numbers (tuple[int, ...]): The passages the answer showed or named.
        detail (str): The call's main argument, for the trail.
        states (tuple[tuple[int, tuple[int, int]], ...]): Each entry the answer depends on,
            with how far it had grown when answered (see :func:`_state`).
    """

    numbers: tuple[int, ...]
    detail: str
    states: tuple[tuple[int, tuple[int, int]], ...] = ()


@dataclass(frozen=True)
class ToolContext:
    """What the tools of one turn work with.

    Attributes:
        ledger (EvidenceLedger): The turn's evidence; a tool admits what it shows into it.
        budget (TurnBudget): The turn's budget, which sizes each reply and each passage in it.
        llm_view (Callable[[BaseNode], BaseNode]): Hides the metadata the model must not see,
            as the grounded prompt does.
        answered (dict[tuple[str, str], _Answered]): The calls this turn has answered, by tool
            name and normalized arguments; kept by the registry.
    """

    ledger: EvidenceLedger
    budget: TurnBudget
    llm_view: Callable[[BaseNode], BaseNode]
    answered: dict[tuple[str, str], _Answered] = field(default_factory=dict, init=False, repr=False, compare=False)


@dataclass(frozen=True)
class ToolWork:
    """The work a call with valid arguments still needs: its I/O, then its answer.

    Attributes:
        detail (str): The call's main argument, cut to ``DETAIL_CHARS``, for the trail.
        fetch (Callable[[], Any] | None): The call's I/O, such as a search. It reads neither
            the turn's context nor its ledger, so it may run off the loop thread; ``None``
            when the answer needs nothing fetched.
        answer (Callable[[Any, ToolContext], ToolReply]): Admits and renders, on the loop
            thread and in call order, from what ``fetch`` returned (``None`` without one).
            It returns a :class:`PassageReply` when it answered the call, so the registry
            can answer the call again by number.
    """

    detail: str
    fetch: Callable[[], Any] | None
    answer: Callable[[Any, ToolContext], ToolReply]


class Tool(Protocol):
    """One tool the model may call.

    Attributes:
        name (str): The name the model calls it by.
        description (str): When to call it, as the model reads it.
        parameters (dict[str, Any]): Its arguments, as the JSON schema of one object.
    """

    name: str
    description: str
    parameters: dict[str, Any]

    def prepare(self, arguments: Mapping[str, Any]) -> ToolReply | ToolWork:
        """Check a call's arguments and say what answering it takes.

        Reads neither the turn's context nor its ledger, which it is not given.

        Args:
            arguments (Mapping[str, Any]): The call's arguments, parsed from JSON.

        Returns:
            ToolReply | ToolWork: An error reply when the arguments will not do, which is
            never remembered; otherwise the call's work.
        """
        ...


def error_reply(action: str, message: str, *, detail: str = "") -> ToolReply:
    """Build the reply to a call that could not be answered as made.

    Args:
        action (str): The tool's name, for the trail.
        message (str): What went wrong and what to call instead, as one sentence.
        detail (str): The call's main argument, for the trail.

    Returns:
        ToolReply: ``Error: <message>``; it admits nothing and is not remembered.
    """
    return ToolReply(text=f"Error: {message}", found=0, action=action, detail=detail)


def _detail(value: Any) -> str:
    """Render a call's main argument as the trail shows it.

    Args:
        value (Any): The argument as parsed from JSON; ``None`` when it was not sent.

    Returns:
        str: Its text on one line, cut to ``DETAIL_CHARS``; empty when it was not sent.
    """
    if value is None:
        return ""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return " ".join(text.split())[:DETAIL_CHARS]


def passage_reply(found: Iterable[NodeWithScore], ctx: ToolContext, *, action: str, detail: str) -> PassageReply:
    """Answer a call with what its search found (see :func:`~docint.agents.turn.passages.answer_passages`).

    Args:
        found (Iterable[NodeWithScore]): The search's passages, best first.
        ctx (ToolContext): The turn's evidence, budget and LLM view.
        action (str): The calling tool's name, for the trail.
        detail (str): The call's main argument, for the trail.

    Returns:
        PassageReply: The passages new to the turn under their numbers, the ones shown
        before named by number, and the notes; ``found`` counts what was admitted or extended.
    """
    rendered = answer_passages(found, ctx, action=action)
    return PassageReply(
        text=rendered.text, found=rendered.found, action=action, detail=detail, numbers=rendered.numbers
    )


class SearchTool:
    """``search(query)``: run the turn's search and answer with what it found."""

    def __init__(self, search: SearchFn) -> None:
        """Bind the search.

        Args:
            search (SearchFn): Runs one search through the turn's engine, with its filters
                and rerank, and returns the ranked passages.
        """
        self.name = "search"
        self.description = SEARCH_DESCRIPTION
        self.parameters = copy.deepcopy(SEARCH_PARAMETERS)
        self._search = search

    def prepare(self, arguments: Mapping[str, Any]) -> ToolReply | ToolWork:
        """Check the query; the search is the call's fetch, and :func:`passage_reply` its answer.

        Args:
            arguments (Mapping[str, Any]): ``query``, the words to look for.

        Returns:
            ToolReply | ToolWork: An error reply when the query is missing, blank or not
            text; otherwise the search to run and the answer to give with what it found.
        """
        raw = arguments.get("query")
        query = " ".join(raw.split())[:MAX_QUERY_CHARS] if isinstance(raw, str) else ""
        if not query:
            return error_reply(
                self.name, "search needs a query: a few specific words to look for.", detail=_detail(raw)
            )
        detail = _detail(query)
        return ToolWork(
            detail=detail,
            fetch=functools.partial(self._search, query),
            answer=functools.partial(passage_reply, action=self.name, detail=detail),
        )


def _passage_number(value: Any) -> int | None:
    """Read a passage number as a model may write it.

    Args:
        value (Any): The ``n`` the model sent.

    Returns:
        int | None: The number, from an integer, a whole float, or text such as ``3`` or
        ``[3]``, when it is positive and under ``_NUMBER_LIMIT``; ``None`` for anything
        else, ``True`` and ``False`` included.
    """
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    elif isinstance(value, str) and (match := _NUMBER_RE.fullmatch(value)):
        value = int(match.group(1))
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if 0 < value < _NUMBER_LIMIT else None


class ReadTool:
    """``read(n)``: show one passage of the turn's evidence again, as much of it as one reply holds."""

    def __init__(self) -> None:
        """Describe the tool."""
        self.name = "read"
        self.description = READ_DESCRIPTION
        self.parameters = copy.deepcopy(READ_PARAMETERS)

    def prepare(self, arguments: Mapping[str, Any]) -> ToolReply | ToolWork:
        """Check that ``n`` reads as a passage number; a read fetches nothing.

        Whether a passage of that number exists is decided at ``answer``, on the loop
        thread, so a search earlier in the same round counts.

        Args:
            arguments (Mapping[str, Any]): ``n``, the passage's number.

        Returns:
            ToolReply | ToolWork: An error reply when ``n`` is no number; otherwise the
            answer, with no fetch.
        """
        raw = arguments.get("n")
        detail = _detail(raw)
        number = _passage_number(raw)
        if number is None:
            return error_reply(
                self.name, "read needs n, the number of a passage shown before, such as 3.", detail=detail
            )
        return ToolWork(detail=detail, fetch=None, answer=functools.partial(self._answer, number, detail))

    def _answer(self, number: int, detail: str, fetched: Any, ctx: ToolContext) -> ToolReply:
        """Show passage ``number`` under its number, as much of it as one reply holds.

        Args:
            number (int): The passage's number.
            detail (str): The call's main argument, for the trail.
            fetched (Any): Nothing; a read has no fetch.
            ctx (ToolContext): The turn's evidence, budget and LLM view.

        Returns:
            ToolReply: The passage with its header, its text cut around its matches only
            past ``result_chars``; an error reply when no passage has that number.
        """
        held = len(ctx.ledger)
        if not 1 <= number <= held:
            if not held:
                known = "no passage has been shown yet"
            elif held == 1:
                known = "the only passage is [1]"
            else:
                known = f"the passages run from [1] to [{held}]"
            return error_reply(self.name, f"there is no passage [{number}]; {known}.", detail=detail)
        text = render_entry(ctx.ledger.entries()[number - 1], number, ctx)
        return PassageReply(
            text=text, found=0, action=self.name, detail=detail, numbers=(number,), depends_on=(number,)
        )


def _arguments(raw: str) -> dict[str, Any] | None:
    """Parse a call's raw arguments.

    Args:
        raw (str): The arguments as the model sent them.

    Returns:
        dict[str, Any] | None: The arguments; empty for blank text or ``null``, which a
        tool then reads as missing; ``None`` when they are not a JSON object.
    """
    if not raw.strip():
        return {}
    try:
        value = json.loads(raw)
    except (ValueError, RecursionError):
        return None
    if value is None:
        return {}
    return value if isinstance(value, dict) else None


def _normalized(value: Any) -> Any:
    """Normalize arguments for comparison: text on one line and case-folded, nested values alike.

    Args:
        value (Any): The arguments, or a value inside them.

    Returns:
        Any: The normalized value.
    """
    if isinstance(value, str):
        return " ".join(value.split()).casefold()
    if isinstance(value, dict):
        return {key: _normalized(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalized(item) for item in value]
    return value


def _example(parameters: Mapping[str, Any]) -> str:
    """Write an example of a tool's arguments from its schema.

    Args:
        parameters (Mapping[str, Any]): The tool's JSON schema.

    Returns:
        str: A JSON object holding each required argument: ``1`` for an integer, ``"..."``
        for anything else.
    """
    properties = parameters.get("properties") or {}
    sample = {
        key: 1 if (properties.get(key) or {}).get("type") == "integer" else "..."
        for key in parameters.get("required") or properties
    }
    return json.dumps(sample, ensure_ascii=False)


def _state(ledger: EvidenceLedger, number: int) -> tuple[int, int]:
    """Return how far a held entry has grown: its text's length and how many matches it holds.

    The ledger only extends an entry, by a window that adds text or a match that adds a
    span, so any change to what a read of it shows changes one of the two.

    Args:
        ledger (EvidenceLedger): The turn's evidence.
        number (int): The entry's number.

    Returns:
        tuple[int, int]: The length of its text and the count of its match spans.
    """
    node = ledger.entries()[number - 1].node
    spans = node.metadata.get(MATCH_SPANS_KEY)
    return len(node.get_content(metadata_mode=MetadataMode.NONE)), len(spans) if isinstance(spans, list) else 0


def _recall(key: tuple[str, str], ctx: ToolContext) -> ToolReply | None:
    """Answer a call from memory, when the same call was answered before in this turn.

    Args:
        key (tuple[str, str]): The tool's name and the call's normalized arguments.
        ctx (ToolContext): The turn's evidence and memory.

    Returns:
        ToolReply | None: The numbers the earlier answer showed or named, admitting
        nothing; ``None`` when nothing is remembered, or when an entry the answer depends
        on has grown since.
    """
    answered = ctx.answered.get(key)
    if answered is None or any(_state(ctx.ledger, number) != state for number, state in answered.states):
        return None
    name = key[0]
    logger.info("A repeated call to the '{}' tool was answered from memory", name)
    text = f"Already called {name} with these arguments; " + (
        f"see {refs(answered.numbers)}." if answered.numbers else "it showed no passages."
    )
    return ToolReply(text=text, found=0, action=name, detail=answered.detail)


@dataclass(frozen=True)
class PreparedCall:
    """A call checked by :meth:`ToolRegistry.prepare`: answered already, or waiting on its work.

    Attributes:
        action (str): The tool's name, for the trail; for a call naming no offered tool,
            the name the model wrote, cut to ``NAME_CHARS``.
        detail (str): The call's main argument, cut to ``DETAIL_CHARS``, for the trail;
            what a reply the loop writes itself, for a fetch that timed out, should show.
        reply (ToolReply | None): The answer when the call needs no work: an error, or the
            memory of the same call earlier in the turn.
        work (ToolWork | None): The work still to do; ``None`` exactly when ``reply`` is set.
        key (tuple[str, str] | None): The tool's name and the normalized arguments, under
            which the answer is remembered.
    """

    action: str
    detail: str
    reply: ToolReply | None = None
    work: ToolWork | None = None
    key: tuple[str, str] | None = None

    @property
    def fetch(self) -> Callable[[], Any] | None:
        """The call's I/O, which may run off the loop thread.

        Returns:
            Callable[[], Any] | None: The fetch; ``None`` when the call needs none.
        """
        return None if self.work is None else self.work.fetch


class ToolRegistry:
    """The tools one turn offers, and the answer to every call the model makes to them."""

    def __init__(self, tools: Iterable[Tool]) -> None:
        """Offer the tools, in the order given.

        Args:
            tools (Iterable[Tool]): The tools.

        Raises:
            ValueError: When two tools share a name, since a call names its tool.
        """
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"Two tools are named {tool.name!r}.")
            self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        """Return the tools as an OpenAI-compatible endpoint receives them.

        Returns:
            list[dict[str, Any]]: One function-tool schema per tool, in the order offered;
            fresh copies the caller may change.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": copy.deepcopy(tool.parameters),
                },
            }
            for tool in self._tools.values()
        ]

    @property
    def names(self) -> frozenset[str]:
        """The names of the tools offered.

        Returns:
            frozenset[str]: The names.
        """
        return frozenset(self._tools)

    def prepare(self, call: ToolCall, ctx: ToolContext) -> PreparedCall:
        """Check one call, and answer it at once or return the work it still needs.

        Runs on the loop thread, in call order. A call naming no offered tool, or sending
        arguments that are not a JSON object, is answered with an error saying what to
        call instead. A call made before in this turn, by tool and arguments compared on
        one line and case-folded, is answered by the numbers its answer showed or named,
        and needs no I/O. Otherwise the tool checks its arguments: an error it finds is the
        answer, never remembered; else the prepared call carries the tool's work.

        Args:
            call (ToolCall): The call as the model made it.
            ctx (ToolContext): The turn's context; only its memory is read.

        Returns:
            PreparedCall: The answer, or the work to fetch and finish.
        """
        name = call.name.strip()
        tool = self._tools.get(name)
        if tool is None:
            logger.warning("A model called a tool this turn does not offer; offered: {}", ", ".join(self._tools))
            called = name[:NAME_CHARS]
            offered = ", ".join(self._tools)
            reply = error_reply(called, f"there is no tool named {called!r}; the tools you can call are {offered}.")
            return PreparedCall(action=called, detail="", reply=reply)
        arguments = _arguments(call.arguments)
        if arguments is None:
            logger.warning("A model sent arguments for the '{}' tool that are not a JSON object", tool.name)
            reply = error_reply(
                tool.name,
                f"the arguments for {tool.name} must be a JSON object, such as {_example(tool.parameters)}.",
            )
            return PreparedCall(action=tool.name, detail="", reply=reply)
        key = (tool.name, json.dumps(_normalized(arguments), sort_keys=True, ensure_ascii=False))
        remembered = _recall(key, ctx)
        if remembered is not None:
            return PreparedCall(action=tool.name, detail=remembered.detail, reply=remembered, key=key)
        work = tool.prepare(arguments)
        if isinstance(work, ToolReply):
            return PreparedCall(action=tool.name, detail=work.detail, reply=work, key=key)
        return PreparedCall(action=tool.name, detail=work.detail, work=work, key=key)

    def finish(self, prepared: PreparedCall, fetched: Any, ctx: ToolContext) -> ToolReply:
        """Answer a prepared call, with what its fetch returned.

        Runs on the loop thread, in call order. A call answered when it was prepared keeps
        that answer. Otherwise memory is checked again first: an identical call earlier in
        the round, prepared beside this one, may have been answered since, and this call
        is then answered from that memory, its own fetch discarded, never admitted again.
        Otherwise the work's answer admits and renders, and a :class:`PassageReply` is
        remembered.

        Args:
            prepared (PreparedCall): The call, from :meth:`prepare`.
            fetched (Any): What its fetch returned; ``None`` when it has none.
            ctx (ToolContext): The turn's evidence, budget, LLM view and memory.

        Returns:
            ToolReply: The reply.

        Raises:
            ValueError: When the prepared call holds neither an answer nor work.
            Exception: Whatever the answer raises; nothing is remembered for the call then.
        """
        if prepared.reply is not None:
            return prepared.reply
        if prepared.work is None or prepared.key is None:
            raise ValueError("A prepared call holds neither an answer nor work.")
        remembered = _recall(prepared.key, ctx)
        if remembered is not None:
            return remembered
        reply = prepared.work.answer(fetched, ctx)
        if isinstance(reply, PassageReply):
            states = tuple((number, _state(ctx.ledger, number)) for number in reply.depends_on)
            ctx.answered[prepared.key] = _Answered(numbers=reply.numbers, detail=reply.detail, states=states)
        return reply

    def call(self, call: ToolCall, ctx: ToolContext) -> ToolReply:
        """Answer one call with a reply the model can read: prepare, fetch and finish in a row.

        Args:
            call (ToolCall): The call as the model made it.
            ctx (ToolContext): The turn's evidence, budget, LLM view and memory.

        Returns:
            ToolReply: The reply (see :meth:`prepare` and :meth:`finish`).

        Raises:
            Exception: Whatever the call's fetch or answer raises, such as a failed search;
                nothing is remembered for the call then.
        """
        prepared = self.prepare(call, ctx)
        fetch = prepared.fetch
        return self.finish(prepared, None if fetch is None else fetch(), ctx)
