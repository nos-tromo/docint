"""The tools an agent turn offers the model, and the registry that answers every call it makes.

Every reply is text the model can read. A search shows the passages new to the turn under
the numbers the answer cites them by, names the passages it showed before by number alone,
and says how many more matched than it had room for; those stay unnumbered, so every number
the model can cite was shown to it. ``read(n)`` shows one passage in full. A call the
registry cannot run (a tool the turn does not offer, arguments that are not a JSON object)
gets an error reply saying what to call instead, and a call made before is answered by
number from memory, never run again.

Nothing here is thread-safe: a call admits passages into the turn's ledger and its answer is
remembered in the turn's context, so the calls of one turn are answered one at a time, in
the order the model made them.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from llama_index.core.schema import BaseNode, NodeWithScore
from loguru import logger

from docint.agents.assessment.ledger import EvidenceLedger
from docint.agents.turn.budget import TurnBudget
from docint.agents.turn.passages import answer_passages, refs, render_entry
from docint.agents.turn.stream import ToolCall

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
READ_DESCRIPTION = "Show the full text of passage [n], when a search showed only part of it."
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
    """

    numbers: tuple[int, ...] = ()


@dataclass(frozen=True)
class _Answered:
    """What the registry remembers of a call it answered.

    Attributes:
        numbers (tuple[int, ...]): The passages the answer showed or named.
        detail (str): The call's main argument, for the trail.
    """

    numbers: tuple[int, ...]
    detail: str


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

    def run(self, arguments: Mapping[str, Any], ctx: ToolContext) -> ToolReply:
        """Answer one call.

        Args:
            arguments (Mapping[str, Any]): The call's arguments, parsed from JSON.
            ctx (ToolContext): The turn's evidence and budget.

        Returns:
            ToolReply: The reply; a :class:`PassageReply` when the call was answered, so
            the registry can answer it again by number.
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

    def run(self, arguments: Mapping[str, Any], ctx: ToolContext) -> ToolReply:
        """Search for the query and answer with the passages found (see :func:`passage_reply`).

        Args:
            arguments (Mapping[str, Any]): ``query``, the words to look for.
            ctx (ToolContext): The turn's evidence, budget and LLM view.

        Returns:
            ToolReply: The passages found, or an error reply when the query is missing,
            blank or not text; the search does not run then.
        """
        raw = arguments.get("query")
        query = " ".join(raw.split())[:MAX_QUERY_CHARS] if isinstance(raw, str) else ""
        if not query:
            return error_reply(
                self.name, "search needs a query: a few specific words to look for.", detail=_detail(raw)
            )
        return passage_reply(self._search(query), ctx, action=self.name, detail=_detail(query))


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
    """``read(n)``: show one passage of the turn's evidence in full."""

    def __init__(self) -> None:
        """Describe the tool."""
        self.name = "read"
        self.description = READ_DESCRIPTION
        self.parameters = copy.deepcopy(READ_PARAMETERS)

    def run(self, arguments: Mapping[str, Any], ctx: ToolContext) -> ToolReply:
        """Show passage ``n`` under its number, as much of it as one reply holds.

        Args:
            arguments (Mapping[str, Any]): ``n``, the passage's number.
            ctx (ToolContext): The turn's evidence, budget and LLM view.

        Returns:
            ToolReply: The passage with its header, its text cut around its matches
            only past ``result_chars``; an error reply when ``n`` is no passage's number.
        """
        raw = arguments.get("n")
        detail = _detail(raw)
        number = _passage_number(raw)
        if number is None:
            return error_reply(
                self.name, "read needs n, the number of a passage shown before, such as 3.", detail=detail
            )
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
        return PassageReply(text=text, found=0, action=self.name, detail=detail, numbers=(number,))


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

    def call(self, call: ToolCall, ctx: ToolContext) -> ToolReply:
        """Answer one call with a reply the model can read.

        A call naming no offered tool, or sending arguments that are not a JSON object, is
        answered with an error saying what to call instead. A call made before in this
        turn, by tool and arguments compared on one line and case-folded, is answered by
        the numbers its first answer showed or named, and does not run again; an error is
        never remembered, so the same call can still be answered once it can be.

        Args:
            call (ToolCall): The call as the model made it.
            ctx (ToolContext): The turn's evidence, budget and LLM view.

        Returns:
            ToolReply: The reply.

        Raises:
            Exception: Whatever the tool raises while it runs, such as a failed search;
                nothing is remembered for the call then.
        """
        name = call.name.strip()
        tool = self._tools.get(name)
        if tool is None:
            logger.warning("A model called a tool this turn does not offer; offered: {}", ", ".join(self._tools))
            called = name[:NAME_CHARS]
            return error_reply(
                called, f"there is no tool named {called!r}; the tools you can call are {', '.join(self._tools)}."
            )
        arguments = _arguments(call.arguments)
        if arguments is None:
            logger.warning("A model sent arguments for the '{}' tool that are not a JSON object", tool.name)
            return error_reply(
                tool.name,
                f"the arguments for {tool.name} must be a JSON object, such as {_example(tool.parameters)}.",
            )
        key = (tool.name, json.dumps(_normalized(arguments), sort_keys=True, ensure_ascii=False))
        answered = ctx.answered.get(key)
        if answered is not None:
            logger.info("A repeated call to the '{}' tool was answered from memory", tool.name)
            text = f"Already called {tool.name} with these arguments; " + (
                f"see {refs(answered.numbers)}." if answered.numbers else "it showed no passages."
            )
            return ToolReply(text=text, found=0, action=tool.name, detail=answered.detail)
        reply = tool.run(arguments, ctx)
        if isinstance(reply, PassageReply):
            ctx.answered[key] = _Answered(numbers=reply.numbers, detail=reply.detail)
        return reply
