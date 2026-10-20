"""The registry that answers every tool call an agent turn makes, and its ``search`` and ``read`` tools.

Every reply is text the model can read. A search shows the passages new to the turn under
the numbers the answer cites them by, names the passages it showed before by number alone
(with a note of their own when they grew), and says how many more matched than it had room
for; those stay unnumbered, so every number the model can cite was shown to it. ``read(n)``
shows one passage again, as much of it as one reply holds. A call the registry cannot run
(a tool the turn does not offer, arguments that are not a JSON object) gets an error reply
saying what to call instead, and a call made before is answered by number from memory,
never run again; a read only while its passage is unchanged. Which calls are one call is the
tool's to say when it defines ``recall_key`` (see :class:`Tool`), else their arguments decide,
compared on one line and case-folded. The image tools, ``search_images`` and ``keyframes``,
live in :mod:`docint.agents.turn.image_tools`.

A call is answered in two steps, so a turn can run the I/O of several calls at once:
:meth:`ToolRegistry.prepare` checks a call and either answers it at once (an error, or a
call made before whose answer no later growth can change, such as a search's) or returns
the work it needs, and :meth:`ToolRegistry.finish` admits and renders. A read made before
is decided at ``finish``, once the calls before it in its round have run, since one of them
may grow its passage. A tool may read the turn's evidence while its call is prepared, as it
stood when the round's calls were prepared: a passage number names an entry there, and one
the same round admits is not there yet. Only a prepared call's ``fetch``, its I/O, is meant
to run off the loop thread: it reads neither the turn's context nor its ledger. Nothing else
here is thread-safe, so ``prepare`` and ``finish`` run on the loop thread, each in the order
the model made the calls. :meth:`ToolRegistry.call` does the steps one after another.
"""

from __future__ import annotations

import copy
import functools
import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

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
#: Deepest nesting of objects and lists a call's arguments may have, the object itself counted.
MAX_ARGUMENT_DEPTH = 32
#: A passage number as a model may write it: ``3``, or ``[3]`` as it is cited.
_NUMBER_RE = re.compile(r"\s*\[?\s*(\d{1,6})\s*\]?\s*")
#: First number no passage can have; a reply never repeats a number this long back.
_NUMBER_LIMIT = 1_000_000

SEARCH_DESCRIPTION = (
    "Search the collection's documents and transcripts for what they say: names, statements, numbers, dates, "
    "quotes. Returns the passages new to this turn, each under the number [n] to cite it by, and names passages "
    "shown before by their number; search again with other words to find more."
)
SEARCH_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {"query": {"type": "string", "description": "A few specific words to look for."}},
    "required": ["query"],
}
READ_DESCRIPTION = (
    "Show the text of passage [n], one shown before, as much of it as one reply holds; "
    "use it for the rest of a passage a reply showed only in part."
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
        error (bool): Whether the reply says why the call was not answered, as every
            :func:`error_reply` does; a turn's trail shows only calls their tool answered.
        informative (bool): Whether the reply told the model something about the collection
            without admitting a passage, as a word count does, so a round with it is not idle.
            A reply answered from memory never is.
        trail (Mapping[str, Any] | None): Extra keys for this call's step in the turn's trail,
            copied verbatim; a key may hold ``None`` on purpose, as a failed count's ``matches``
            does. Its values must be JSON values (str, int, float, bool or ``None``), since the
            stored step is persisted with ``json.dumps``. ``None`` for none.
    """

    text: str
    found: int
    action: str
    detail: str
    error: bool = False
    informative: bool = False
    trail: Mapping[str, Any] | None = None


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
        trail (Mapping[str, Any] | None): A read-only copy of the extra keys the answer's step
            carried, which the step of a call answered from memory shows again.
    """

    numbers: tuple[int, ...]
    detail: str
    states: tuple[tuple[int, tuple[int, int]], ...] = ()
    trail: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ToolContext:
    """What the tools of one turn work with.

    Attributes:
        ledger (EvidenceLedger): The turn's evidence; a tool admits what it shows into it.
        budget (TurnBudget): The turn's budget, which sizes each reply and each passage in it.
        llm_view (Callable[[BaseNode], BaseNode]): Hides the metadata the model must not see,
            as the grounded prompt does.
        answered (dict[tuple[str, str], _Answered]): The calls this turn has answered, by tool
            name and repeat key (see :func:`_repeat_key`); kept by the registry.
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

    A tool may also define ``recall_key(arguments: Mapping[str, Any]) -> str | None`` to say
    which calls are one call (see :class:`HasRecallKey`, which the registry checks for; it is
    left out of this protocol so a tool without it still satisfies it). The registry remembers
    an answer under the tool's name and that string, and answers a later call with the same
    string from memory, whatever its raw arguments. It runs on the loop thread for a call whose
    arguments are a JSON object within ``MAX_ARGUMENT_DEPTH``, before :meth:`prepare` checks them,
    so it must cope with arguments that are missing or of the wrong type. ``None`` leaves the
    call to the default key, its arguments compared on one line and case-folded, as does a tool
    without the method.

    Attributes:
        name (str): The name the model calls it by.
        description (str): When to call it, as the model reads it.
        parameters (dict[str, Any]): Its arguments, as the JSON schema of one object.
    """

    name: str
    description: str
    parameters: dict[str, Any]

    def prepare(self, arguments: Mapping[str, Any], ctx: ToolContext) -> ToolReply | ToolWork:
        """Check a call's arguments and say what answering it takes.

        Runs on the loop thread, in call order. It may read the turn's evidence as it stood
        when the round's calls were prepared, such as the entry a passage number names;
        it never writes the context, whose memory the registry keeps.

        Args:
            arguments (Mapping[str, Any]): The call's arguments, parsed from JSON.
            ctx (ToolContext): The turn's context, to read only.

        Returns:
            ToolReply | ToolWork: An error reply when the arguments will not do, which is
            never remembered; otherwise the call's work.
        """
        ...


@runtime_checkable
class HasRecallKey(Protocol):
    """A tool that says itself which calls are one call: the optional ``recall_key`` of a :class:`Tool`."""

    def recall_key(self, arguments: Mapping[str, Any]) -> str | None:
        """Name the call by what makes it the same call as another.

        Args:
            arguments (Mapping[str, Any]): The call's arguments, parsed from JSON.

        Returns:
            str | None: The string the call is remembered by, under the tool's name; ``None`` to
            leave the call to the default key.
        """
        ...


def error_reply(action: str, message: str, *, detail: str = "") -> ToolReply:
    """Build the reply to a call that could not be answered as made.

    Args:
        action (str): The tool's name, for the trail.
        message (str): What went wrong and what to call instead, as one sentence.
        detail (str): The call's main argument, for the trail.

    Returns:
        ToolReply: ``Error: <message>``, marked as an error; it admits nothing and is not remembered.
    """
    return ToolReply(text=f"Error: {message}", found=0, action=action, detail=detail, error=True)


def trail_detail(value: Any) -> str:
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


def passage_reply(
    found: Iterable[NodeWithScore],
    ctx: ToolContext,
    *,
    action: str,
    detail: str,
    hint: str | None = None,
    note: str = "",
    note_first: bool = False,
) -> PassageReply:
    """Answer a call with what its search found (see :func:`~docint.agents.turn.passages.answer_passages`).

    Args:
        found (Iterable[NodeWithScore]): The search's passages, best first.
        ctx (ToolContext): The turn's evidence, budget and LLM view.
        action (str): The calling tool's name, for the trail.
        detail (str): The call's main argument, for the trail.
        hint (str | None): What the reply suggests for matching passages it had no room for;
            ``None`` to call the same tool again (see :func:`~docint.agents.turn.passages.search_hint`).
        note (str): A note of the calling tool's own, after the others unless ``note_first``;
            empty for none.
        note_first (bool): Whether the note leads the reply instead, ahead of the passages.

    Returns:
        PassageReply: The passages new to the turn under their numbers, the ones shown
        before named by number, and the notes; ``found`` counts what was admitted or extended.
    """
    rendered = answer_passages(found, ctx, action=action, hint=hint, note=note, note_first=note_first)
    return PassageReply(
        text=rendered.text, found=rendered.found, action=action, detail=detail, numbers=rendered.numbers
    )


def search_work(name: str, raw: Any, search: SearchFn, *, missing: str) -> ToolReply | ToolWork:
    """Check a search's words: the search is the call's fetch, and :func:`passage_reply` its answer.

    Args:
        name (str): The calling tool's name.
        raw (Any): The words as the model sent them.
        search (SearchFn): Runs the search with the words, on one line and cut to ``MAX_QUERY_CHARS``.
        missing (str): The error to answer with when there are no words to search for.

    Returns:
        ToolReply | ToolWork: An error reply when the words are missing, blank or not
        text; otherwise the search to run and the answer to give with what it found.
    """
    query = " ".join(raw.split())[:MAX_QUERY_CHARS] if isinstance(raw, str) else ""
    if not query:
        return error_reply(name, missing, detail=trail_detail(raw))
    detail = trail_detail(query)
    return ToolWork(
        detail=detail,
        fetch=functools.partial(search, query),
        answer=functools.partial(passage_reply, action=name, detail=detail),
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

    def prepare(self, arguments: Mapping[str, Any], ctx: ToolContext) -> ToolReply | ToolWork:
        """Check the query; the search is the call's fetch, and :func:`passage_reply` its answer.

        Args:
            arguments (Mapping[str, Any]): ``query``, the words to look for.
            ctx (ToolContext): Unused; a search needs nothing of the evidence to start.

        Returns:
            ToolReply | ToolWork: An error reply when the query is missing, blank or not
            text; otherwise the search to run and the answer to give with what it found.
        """
        return search_work(
            self.name,
            arguments.get("query"),
            self._search,
            missing="search needs a query: a few specific words to look for.",
        )


def passage_number(value: Any) -> int | None:
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


def passage_range(held: int) -> str:
    """Name the passage numbers the evidence holds, for a reply to a number it does not.

    Args:
        held (int): How many passages the evidence holds.

    Returns:
        str: ``the passages run from [1] to [n]``, or the wording for one passage or none.
    """
    if not held:
        return "no passage has been shown yet"
    if held == 1:
        return "the only passage is [1]"
    return f"the passages run from [1] to [{held}]"


class ReadTool:
    """``read(n)``: show one passage of the turn's evidence again, as much of it as one reply holds."""

    def __init__(self) -> None:
        """Describe the tool."""
        self.name = "read"
        self.description = READ_DESCRIPTION
        self.parameters = copy.deepcopy(READ_PARAMETERS)

    def prepare(self, arguments: Mapping[str, Any], ctx: ToolContext) -> ToolReply | ToolWork:
        """Check that ``n`` reads as a passage number; a read fetches nothing.

        Whether a passage of that number exists is decided at ``answer``, on the loop
        thread, so a search earlier in the same round counts.

        Args:
            arguments (Mapping[str, Any]): ``n``, the passage's number.
            ctx (ToolContext): Unused; the number is looked up at ``answer``.

        Returns:
            ToolReply | ToolWork: An error reply when ``n`` is no number; otherwise the
            answer, with no fetch.
        """
        raw = arguments.get("n")
        detail = trail_detail(raw)
        number = passage_number(raw)
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
            past ``result_chars``; an error reply when no passage has that number. It is
            informative: it shows text a reply may have cut, so a round of reads is not idle.
            The same read answered from memory is not (see :func:`_recall`).
        """
        held = len(ctx.ledger)
        if not 1 <= number <= held:
            return error_reply(self.name, f"there is no passage [{number}]; {passage_range(held)}.", detail=detail)
        text = render_entry(ctx.ledger.entries()[number - 1], number, ctx)
        return PassageReply(
            text=text,
            found=0,
            action=self.name,
            detail=detail,
            informative=True,
            numbers=(number,),
            depends_on=(number,),
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


def nested_too_deeply(value: Any) -> bool:
    """Return whether parsed arguments nest objects and lists deeper than ``MAX_ARGUMENT_DEPTH``.

    Measured with a stack, not recursion, so it is safe at any depth JSON can parse. The
    registry refuses such arguments and a turn echoes them back as an empty object: one rule,
    so the two verdicts agree.

    Args:
        value (Any): The parsed arguments.

    Returns:
        bool: ``True`` when an object or a list sits deeper than ``MAX_ARGUMENT_DEPTH``.
    """
    stack: list[tuple[Any, int]] = [(value, 1)]
    while stack:
        item, depth = stack.pop()
        if isinstance(item, (dict, list)):
            if depth > MAX_ARGUMENT_DEPTH:
                return True
            stack.extend((child, depth + 1) for child in (item.values() if isinstance(item, dict) else item))
    return False


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


def _repeat_key(tool: Tool, arguments: Mapping[str, Any]) -> str:
    """Return what a call is remembered by besides its tool: the tool's own repeat key, else its arguments.

    Args:
        tool (Tool): The tool called.
        arguments (Mapping[str, Any]): The call's arguments, a JSON object within ``MAX_ARGUMENT_DEPTH``.

    Returns:
        str: What ``tool.recall_key`` returns, when the tool has it and it returns text; otherwise
        the arguments normalized as :func:`_normalized` does, as sorted JSON.
    """
    if isinstance(tool, HasRecallKey):
        own = tool.recall_key(arguments)
        if isinstance(own, str):
            return own
    return json.dumps(_normalized(arguments), sort_keys=True, ensure_ascii=False)


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


def _recall(key: tuple[str, str], ctx: ToolContext, *, final: bool) -> ToolReply | None:
    """Answer a call from memory, when the same call was answered before in this turn.

    An answer kept by entry state, a read's, is recalled only once every earlier call in
    the round has finished, as at :meth:`ToolRegistry.finish`: until then one of them may
    still grow the entry. An answer without states, a search's, is recalled at any time.

    Args:
        key (tuple[str, str]): The tool's name and the call's repeat key (see :func:`_repeat_key`).
        ctx (ToolContext): The turn's evidence and memory.
        final (bool): Whether every earlier call in the round has finished.

    Returns:
        ToolReply | None: The numbers the earlier answer showed or named, admitting nothing
        and no news, under the trail the earlier answer carried; ``None`` when nothing is
        remembered, when the answer's states cannot be trusted yet, or when an entry the
        answer depends on has grown since.
    """
    answered = ctx.answered.get(key)
    if answered is None or (answered.states and not final):
        return None
    if any(_state(ctx.ledger, number) != state for number, state in answered.states):
        return None
    name = key[0]
    logger.info("A repeated call to the '{}' tool was answered from memory", name)
    text = f"Already called {name} with these arguments; " + (
        f"see {refs(answered.numbers)}." if answered.numbers else "it showed no passages."
    )
    return ToolReply(text=text, found=0, action=name, detail=answered.detail, trail=answered.trail)


@dataclass(frozen=True)
class PreparedCall:
    """A call checked by :meth:`ToolRegistry.prepare`: answered already, or waiting on its work.

    Attributes:
        action (str): The tool's name, for the trail; for a call naming no offered tool,
            the name the model wrote, cut to ``NAME_CHARS``.
        detail (str): The call's main argument, cut to ``DETAIL_CHARS``, for the trail;
            what a reply the loop writes itself, for a fetch that timed out, should show.
        reply (ToolReply | None): The answer when the call needs no work: an error, or the
            memory of the same call earlier in the turn when no growth of a passage can
            change it, as for a search. A remembered read is decided at
            :meth:`ToolRegistry.finish`, once the calls before it in its round have run.
        work (ToolWork | None): The work still to do; ``None`` exactly when ``reply`` is set.
        key (tuple[str, str] | None): The tool's name and its repeat key for the call (see
            :func:`_repeat_key`), under which the answer is remembered.
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
        arguments that are not a JSON object or nest deeper than ``MAX_ARGUMENT_DEPTH``, is
        answered with an error saying what to call instead. A call made before in this turn,
        by tool and repeat key (the tool's own, else its arguments compared on one line and
        case-folded), is answered by the numbers its answer showed or named, and needs no
        I/O; a read is left to :meth:`finish`, since an earlier call in its round may still
        grow its passage. Otherwise the tool checks its arguments, reading the evidence where
        it must: a passage number names an entry as the evidence stood when the round's calls
        were prepared. An error it finds is the answer, never remembered; else the prepared
        call carries its work.

        Args:
            call (ToolCall): The call as the model made it.
            ctx (ToolContext): The turn's context; its memory and evidence are read, never written.

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
        if nested_too_deeply(arguments):
            logger.warning("A model sent arguments for the '{}' tool nested too deeply to read", tool.name)
            reply = error_reply(
                tool.name,
                f"the arguments for {tool.name} are nested too deeply; send a flat JSON object, "
                f"such as {_example(tool.parameters)}.",
            )
            return PreparedCall(action=tool.name, detail="", reply=reply)
        key = (tool.name, _repeat_key(tool, arguments))
        remembered = _recall(key, ctx, final=False)
        if remembered is not None:
            return PreparedCall(action=tool.name, detail=remembered.detail, reply=remembered, key=key)
        work = tool.prepare(arguments, ctx)
        if isinstance(work, ToolReply):
            return PreparedCall(action=tool.name, detail=work.detail, reply=work, key=key)
        return PreparedCall(action=tool.name, detail=work.detail, work=work, key=key)

    def finish(self, prepared: PreparedCall, fetched: Any, ctx: ToolContext) -> ToolReply:
        """Answer a prepared call, with what its fetch returned.

        Runs on the loop thread, in call order. A call answered when it was prepared keeps
        that answer. Otherwise memory is checked first, now that the calls before this one
        have finished: an identical call earlier in the round, prepared beside this one,
        may have been answered since, and a read remembered from an earlier round is
        decided here, while its passage is unchanged. Either way the call is answered from
        memory, its own fetch discarded, never admitted again. Otherwise the work's answer
        admits and renders, and a :class:`PassageReply` is remembered with a copy of its trail.

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
        remembered = _recall(prepared.key, ctx, final=True)
        if remembered is not None:
            return remembered
        reply = prepared.work.answer(fetched, ctx)
        if isinstance(reply, PassageReply):
            states = tuple((number, _state(ctx.ledger, number)) for number in reply.depends_on)
            # A copy, so a tool that reuses and mutates one dict cannot rewrite earlier steps.
            trail = None if reply.trail is None else MappingProxyType(dict(reply.trail))
            ctx.answered[prepared.key] = _Answered(
                numbers=reply.numbers, detail=reply.detail, states=states, trail=trail
            )
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
