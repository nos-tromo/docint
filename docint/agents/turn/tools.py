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
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from llama_index.core.schema import BaseNode, NodeWithScore
from loguru import logger

from docint.agents.assessment.ledger import EvidenceLedger, passage_costs, passage_parts, render_context, stamp
from docint.agents.turn.budget import TurnBudget
from docint.agents.turn.stream import ToolCall
from docint.utils.text_budget import allocate_with_floors

SearchFn = Callable[[str], list[NodeWithScore]]
"""Run one search through the turn's engine and return its ranked passages."""

#: Longest main argument a reply carries for the turn's trail.
DETAIL_CHARS = 200
#: Longest query a search sends to retrieval.
MAX_QUERY_CHARS = 300
#: Longest tool name an error reply repeats back.
NAME_CHARS = 64
#: Longest header a reply shows whole; a longer one keeps its citation line and ``COMPACT_HEADER_CHARS`` of the rest.
HEADER_CHARS = 500
#: What a search reply says when it found nothing the turn had not shown.
NO_NEW_PASSAGES = "No new passages."
#: How a reply joins its passages, and its passages to the notes after them; ``render_context`` reserves it.
_JOIN = "\n\n"
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


def _refs(numbers: Iterable[int]) -> str:
    """Name passages by number, as the answer cites them.

    Args:
        numbers (Iterable[int]): Their ledger numbers.

    Returns:
        str: ``[1], [4]`` in ascending order.
    """
    return ", ".join(f"[{number}]" for number in sorted(numbers))


def _numbered(entry: NodeWithScore, number: int) -> NodeWithScore:
    """Stamp an entry with its number, its citation line first, so every passage a reply shows opens with it.

    Args:
        entry (NodeWithScore): The ledger entry; never written to.
        number (int): Its ledger number.

    Returns:
        NodeWithScore: A stamped clone (see :func:`~docint.agents.assessment.ledger.stamp`).
    """
    stamped = stamp(entry, number)
    metadata = {"citation_index": number, **stamped.node.metadata}
    return NodeWithScore(node=stamped.node.model_copy(update={"metadata": metadata}), score=stamped.score)


def _compact(stamped: NodeWithScore, ctx: ToolContext) -> bool:
    """Return whether a passage's header is too long to show whole.

    Args:
        stamped (NodeWithScore): The numbered passage.
        ctx (ToolContext): The turn's LLM view.

    Returns:
        bool: ``True`` when its header runs past ``HEADER_CHARS``.
    """
    headers, _, _ = passage_parts([stamped], ctx.llm_view)
    return len(headers[0]) > HEADER_CHARS


@dataclass(frozen=True)
class _Planned:
    """A passage a reply may show, priced as the reply renders it.

    Attributes:
        number (int): Its ledger number.
        passage (NodeWithScore): The ledger entry, numbered (see :func:`_numbered`).
        compact (bool): Whether its header is compacted (see :func:`_compact`).
        reserve (int): What its header, joins and cut marker take.
        floor (int): The least text it shows: its legible floor, or ``passage_chars`` if less.
        need (int): The most text it shows: all of it, or ``passage_chars`` if less.
    """

    number: int
    passage: NodeWithScore
    compact: bool
    reserve: int
    floor: int
    need: int


def _plan(entry: NodeWithScore, number: int, ctx: ToolContext) -> _Planned:
    """Price one entry under its number, with the header a reply gives it.

    Args:
        entry (NodeWithScore): The ledger entry, or the entry a new passage would become.
        number (int): Its ledger number.
        ctx (ToolContext): The turn's budget and LLM view.

    Returns:
        _Planned: The numbered entry and its costs.
    """
    numbered = _numbered(entry, number)
    compact = _compact(numbered, ctx)
    headers, bodies, anchors = passage_parts([numbered], ctx.llm_view, compact=compact)
    reserve, floor = passage_costs(headers[0], bodies[0], anchors[0])
    cap = ctx.budget.passage_chars
    return _Planned(
        number=number,
        passage=numbered,
        compact=compact,
        reserve=reserve,
        floor=min(floor, cap),
        need=min(len(bodies[0]), cap),
    )


def _fits(planned: Sequence[_Planned], room: int) -> bool:
    """Return whether passages can all be shown, each at least at its floor, in the room.

    Args:
        planned (Sequence[_Planned]): The passages.
        room (int): Characters they may take together.

    Returns:
        bool: ``True`` when text room remains after their reserves and covers their floors.
    """
    text_room = room - sum(passage.reserve for passage in planned)
    return text_room > 0 and text_room >= sum(passage.floor for passage in planned)


def _render(planned: Sequence[_Planned], room: int, ctx: ToolContext) -> str:
    """Render passages in number order, sharing the room with each kept within its cap.

    Each passage first gets its floor; the rest is shared shortest-first up to its need.
    Each is then rendered alone, so its text is cut window by window around its matches
    to exactly its share, which no shared packing would cap. ``render_context`` defuses
    sentinels in each, and each starts with its citation line, so no join completes one.

    Args:
        planned (Sequence[_Planned]): Passages that fit together (see :func:`_fits`).
        room (int): Characters they may take together.
        ctx (ToolContext): The turn's LLM view.

    Returns:
        str: The passages, each its header and its text, joined by blank lines.
    """
    ordered = sorted(planned, key=lambda passage: passage.number)
    text_room = room - sum(passage.reserve for passage in ordered)
    shares = allocate_with_floors(
        [passage.need for passage in ordered], [passage.floor for passage in ordered], max(0, text_room)
    )
    return _JOIN.join(
        render_context(
            [passage.passage],
            llm_view=ctx.llm_view,
            budget_tokens=passage.reserve + share,
            # The budget is in characters: a reply is sized in characters.
            char_token_ratio=1.0,
            compact=passage.compact,
        ).context
        for passage, share in zip(ordered, shares, strict=True)
    )


def _sections(found: Iterable[NodeWithScore]) -> list[list[NodeWithScore]]:
    """Group a search's passages by node id, so the windows of one section are one passage.

    Args:
        found (Iterable[NodeWithScore]): The passages, best first.

    Returns:
        list[list[NodeWithScore]]: Each section's windows in arrival order, sections in the
        order they first appear.
    """
    sections: dict[str, list[NodeWithScore]] = {}
    for scored in found:
        sections.setdefault(scored.node.node_id, []).append(scored)
    return list(sections.values())


def _merged(windows: Sequence[NodeWithScore]) -> NodeWithScore:
    """Return the entry a section not yet held becomes once its windows are admitted.

    Args:
        windows (Sequence[NodeWithScore]): The section's windows, in arrival order.

    Returns:
        NodeWithScore: The entry, merged by the ledger's own rules on a scratch ledger.
    """
    scratch = EvidenceLedger(cap=1)
    scratch.add(windows)
    return scratch.entries()[0]


def _left_out(count: int, *, full: bool) -> str:
    """Say how many new passages a reply did not show, and what to do about them.

    Args:
        count (int): How many it left out.
        full (bool): Whether the evidence was full, rather than the reply.

    Returns:
        str: One sentence.
    """
    passages = "passage" if count == 1 else "passages"
    if full:
        return (
            f"{count} more matching {passages} left out: the evidence holds all it can, "
            "so answer from the passages shown."
        )
    return f"{count} more matching {passages} did not fit in this reply; search with narrower words to see them."


def _trailer(*, shown: bool, again: Sequence[int], left_out: int, full: bool) -> str:
    """Write the notes after a search reply's passages.

    Args:
        shown (bool): Whether the reply shows any passage.
        again (Sequence[int]): Passages shown before that matched again, by number.
        left_out (int): New passages the reply did not show.
        full (bool): Whether they were left out because the evidence was full.

    Returns:
        str: The notes, one per line; empty when there is nothing to note.
    """
    lines: list[str] = []
    if not shown and not left_out:
        lines.append(NO_NEW_PASSAGES)
    if again:
        lines.append(f"Also matched, shown before: {_refs(again)}.")
    if left_out:
        lines.append(_left_out(left_out, full=full))
    return "\n".join(lines)


def answer_passages(found: Iterable[NodeWithScore], ctx: ToolContext, *, action: str, detail: str) -> PassageReply:
    """Admit what a search found into the turn's evidence and answer the call with it.

    Passages are taken best first, the windows of one section as one passage. A passage
    held before keeps its number: when the search extends it, its grown entry is shown if
    it fits; otherwise it is named by number. A new passage is admitted, under the next
    number, only while the reply can still show every passage at least at its floor; one
    that does not fit, or finds the evidence full, stays unnumbered and the reply says how
    many more matched. The reply keeps within ``result_chars``, room for its notes
    reserved first, and each passage's text within ``passage_chars``.

    Args:
        found (Iterable[NodeWithScore]): The search's passages, best first.
        ctx (ToolContext): The turn's evidence, budget and LLM view.
        action (str): The calling tool's name, for the trail.
        detail (str): The call's main argument, for the trail.

    Returns:
        PassageReply: The reply; ``found`` counts the passages it admitted or extended.
    """
    ledger = ctx.ledger
    sections = _sections(found)
    held = [number for windows in sections if (number := ledger.number_of(windows[0].node.node_id)) is not None]
    fresh = len(sections) - len(held)
    notes = max(len(_trailer(shown=False, again=held, left_out=fresh, full=full)) for full in (False, True))
    room = ctx.budget.result_chars - notes - len(_JOIN)
    shown: list[_Planned] = []
    again: list[int] = []
    left_out, full, added = 0, False, 0
    for windows in sections:
        number = ledger.number_of(windows[0].node.node_id)
        if number is not None:
            if ledger.add(windows):
                added += 1
                planned = _plan(ledger.entries()[number - 1], number, ctx)
                if _fits([*shown, planned], room):
                    shown.append(planned)
                    continue
            again.append(number)
        elif ledger.full:
            left_out, full = left_out + 1, True
        else:
            planned = _plan(_merged(windows), len(ledger) + 1, ctx)
            if _fits([*shown, planned], room) and ledger.add(windows):
                added += 1
                shown.append(planned)
            else:
                left_out += 1
    trailer = _trailer(shown=bool(shown), again=again, left_out=left_out, full=full)
    text = _JOIN.join(part for part in (_render(shown, room, ctx), trailer) if part)
    logger.debug(
        "The '{}' tool showed {} passages, named {} shown before and left {} out",
        action,
        len(shown),
        len(again),
        left_out,
    )
    return PassageReply(
        text=text,
        found=added,
        action=action,
        detail=detail,
        numbers=tuple(sorted({passage.number for passage in shown} | set(again))),
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
        """Search for the query and answer with the passages found (see :func:`answer_passages`).

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
        return answer_passages(self._search(query), ctx, action=self.name, detail=_detail(query))


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
        numbered = _numbered(ctx.ledger.entries()[number - 1], number)
        rendered = render_context(
            [numbered],
            llm_view=ctx.llm_view,
            budget_tokens=ctx.budget.result_chars,
            # The budget is in characters, as a search reply's is.
            char_token_ratio=1.0,
            compact=_compact(numbered, ctx),
        )
        return PassageReply(text=rendered.context, found=0, action=self.name, detail=detail, numbers=(number,))


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
                f"see {_refs(answered.numbers)}." if answered.numbers else "it showed no passages."
            )
            return ToolReply(text=text, found=0, action=tool.name, detail=answered.detail)
        reply = tool.run(arguments, ctx)
        if isinstance(reply, PassageReply):
            ctx.answered[key] = _Answered(numbers=reply.numbers, detail=reply.detail)
        return reply
