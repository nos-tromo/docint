"""The agent's tool registry and its search and read tools. Every passage, query and file here is invented."""

from __future__ import annotations

import json
import pkgutil
import re
import subprocess
import sys
from collections.abc import Sequence
from typing import Any

import pytest
from assessment_fakes import llm_view, scored_node, social_post
from llama_index.core.schema import NodeWithScore
from loguru import logger

import docint.agents.turn
from docint.agents.assessment.ledger import (
    COMPACT_HEADER_CHARS,
    MARKER_RESERVE_CHARS,
    MAX_LEDGER_ENTRIES,
    MIN_TEXT_CHARS,
    WINDOW_JOIN,
    EvidenceLedger,
    passage_costs,
    passage_parts,
    stamp,
)
from docint.agents.turn.budget import TurnBudget
from docint.agents.turn.passages import HEADER_CHARS, NO_NEW_PASSAGES, _fits, _plan
from docint.agents.turn.stream import ToolCall
from docint.agents.turn.tools import PassageReply, ReadTool, SearchTool, ToolContext, ToolRegistry, ToolReply
from docint.core.rag import ParentContextPostprocessor
from docint.utils.text_budget import MATCH_SPANS_KEY

PROGRAMME = (
    "Das Hafenfest wurde wegen der Sturmwarnung auf Sonntag verschoben. Die Stände öffnen um zehn Uhr, Musik ab Mittag."
)
WEATHER = "Für Sonntag meldet der Wetterdienst bedeckten Himmel, Wind aus West und kühle Temperaturen am Hafen."
CLIP = "Im Clip harbour_walk.webm gehen zwei Personen in Regenjacken am Hafen entlang; Person A trägt einen Schirm."
OPENING = "Der Hafenverein lädt am Sonntag zum Hafenfest ein."
CLOSING = "Am Abend spielt eine Band am Kai."

#: The cut marker a shortened passage carries.
MARKER = re.compile(r"\[\.\.\. \d+ of \d+ characters not shown \.\.\.\]")
#: The notes a search reply ends with, after its passages.
NOTES = re.compile(r"\n\n(?:No new passages\.|Also matched, |\d+ more matching ).*\Z", re.DOTALL)


def _budget(*, reasoning: bool = False) -> TurnBudget:
    """Build one turn's budget at an 8k window; the tools read only its reply and passage sizes.

    Args:
        reasoning (bool): Whether the model thinks before it answers.

    Returns:
        TurnBudget: The budget.
    """
    return TurnBudget(
        ctx_window=8192,
        reasoning=reasoning,
        char_token_ratio=3.5,
        system_tokens=900,
        tool_schema_tokens=630,
        history_tokens=1300,
        message_tokens=300,
    )


def _context(*, reasoning: bool = False, cap: int = MAX_LEDGER_ENTRIES) -> ToolContext:
    """Build the context of one turn whose evidence is still empty.

    Args:
        reasoning (bool): Whether the model thinks before it answers, which shortens replies.
        cap (int): The most passages the turn's ledger admits.

    Returns:
        ToolContext: The context.
    """
    return ToolContext(ledger=EvidenceLedger(cap=cap), budget=_budget(reasoning=reasoning), llm_view=llm_view)


class _Search:
    """A search double: each run answers with its next scripted result and records its query."""

    def __init__(self, *results: list[NodeWithScore]) -> None:
        """Hold the results, one per run.

        Args:
            *results (list[NodeWithScore]): What each run returns, in order.
        """
        self._results = list(results)
        self.queries: list[str] = []

    def __call__(self, query: str) -> list[NodeWithScore]:
        """Run one search.

        Args:
            query (str): What the tool searched for.

        Returns:
            list[NodeWithScore]: The next scripted result.

        Raises:
            AssertionError: When the search runs more often than it has results.
        """
        self.queries.append(query)
        if not self._results:
            raise AssertionError("The search ran more often than its script has results.")
        return self._results.pop(0)


def _registry(search: _Search) -> ToolRegistry:
    """Build the registry of a documents turn: search and read.

    Args:
        search (_Search): The search the search tool runs.

    Returns:
        ToolRegistry: The registry.
    """
    return ToolRegistry([SearchTool(search), ReadTool()])


def _call(name: str, **arguments: Any) -> ToolCall:
    """Build one tool call as a model round sends it.

    Args:
        name (str): The tool's name.
        **arguments (Any): The arguments, sent as JSON text.

    Returns:
        ToolCall: The call.
    """
    return ToolCall(id="call_0", name=name, arguments=json.dumps(arguments, ensure_ascii=False))


def _shown(text: str) -> list[int]:
    """Return the numbers a reply shows passages under, in reply order.

    Args:
        text (str): The reply.

    Returns:
        list[int]: Each shown passage's number.
    """
    return [int(number) for number in re.findall(r"^citation_index: (\d+)$", text, flags=re.MULTILINE)]


def _bodies(text: str) -> dict[int, str]:
    """Map each number a reply shows to the passage text under it.

    Args:
        text (str): The reply.

    Returns:
        dict[int, str]: The text shown under each number, window joins and cut markers included.
    """
    bodies: dict[int, str] = {}
    for block in re.split(r"\n\n(?=citation_index: \d+\n)", NOTES.sub("", text)):
        header, _, body = block.partition("\n\n")
        if match := re.match(r"citation_index: (\d+)\n", f"{header}\n"):
            bodies[int(match.group(1))] = body
    return bodies


def _report(number: int, *, sentences: int = 14) -> NodeWithScore:
    """Build a long passage of an invented planning report, about 75 characters per sentence.

    Args:
        number (int): Which report; it also varies the id, file and page.
        sentences (int): How many sentences the passage holds.

    Returns:
        NodeWithScore: The passage.
    """
    text = " ".join(
        f"Abschnitt {number}.{line}: Der Hafenverein plant Stände, Musik und einen Lauf am Kai."
        for line in range(1, sentences + 1)
    )
    return scored_node(f"report-{number}", text, filename=f"planungsbericht_{number}.pdf", page=number)


def _window(text: str) -> NodeWithScore:
    """Build one parent-context window of an invented programme section; every window shares the section's id.

    Args:
        text (str): The windowed text.

    Returns:
        NodeWithScore: The window.
    """
    return scored_node(
        "programme-2",
        text,
        filename="hafenfest_programm.pdf",
        page=2,
        parent_context_windowed=True,
        window_chars=len(text),
    )


def _weather() -> NodeWithScore:
    """Build the invented weather passage.

    Returns:
        NodeWithScore: The passage.
    """
    return scored_node("weather", WEATHER, filename="wetterbericht_hafen.pdf", page=2)


def _programme() -> NodeWithScore:
    """Build the invented programme passage.

    Returns:
        NodeWithScore: The passage.
    """
    return scored_node("programme", PROGRAMME, filename="hafenfest_programm.pdf", page=1)


def test_search_numbers_new_passages_and_refers_to_held_ones() -> None:
    """Passages new to the turn are shown under the next numbers; ones shown before are named by number alone."""
    ctx = _context()
    programme, clip = _programme(), scored_node("clip", CLIP, filename="harbour_walk.webm")
    ctx.ledger.add([programme, clip])
    report = _report(1, sentences=2)
    search = _Search([_weather(), clip, programme, report])

    reply = _registry(search).call(_call("search", query="Hafenfest Wetter am Sonntag"), ctx)

    assert search.queries == ["Hafenfest Wetter am Sonntag"]
    assert _shown(reply.text) == [3, 4]
    assert WEATHER in reply.text
    assert report.node.get_content() in reply.text
    assert PROGRAMME not in reply.text
    assert CLIP not in reply.text
    assert reply.text.endswith("\n\nAlso matched, shown before: [1], [2].")
    assert (reply.found, reply.action, reply.detail) == (2, "search", "Hafenfest Wetter am Sonntag")
    assert [ctx.ledger.number_of(node_id) for node_id in ("programme", "clip", "weather", "report-1")] == [1, 2, 3, 4]


def test_a_result_never_numbers_what_it_did_not_show() -> None:
    """Passages a reply has no room to show stay unnumbered, so every number the model can cite was shown to it."""
    ctx = _context(reasoning=True)
    found = [_report(number) for number in range(1, 7)]

    reply = _registry(_Search(found)).call(_call("search", query="Hafenverein Planung"), ctx)

    admitted = len(ctx.ledger)
    assert 0 < admitted < len(found)
    assert _shown(reply.text) == list(range(1, admitted + 1))
    assert all(ctx.ledger.number_of(scored.node.node_id) is None for scored in found[admitted:])
    assert reply.found == admitted
    assert reply.text.endswith(
        f"\n\n{len(found) - admitted} more matching passages did not fit in this reply; "
        "search with narrower words to see them."
    )
    assert len(reply.text) <= ctx.budget.result_chars
    # Every passage shown is legible: it keeps at least the floor a cut passage must keep.
    assert all(len(MARKER.sub("", body)) >= MIN_TEXT_CHARS for body in _bodies(reply.text).values())


def test_an_extended_window_keeps_its_number() -> None:
    """A later window of a section shown before extends it under the same number, listed by number before a new one."""
    ctx = _context()
    registry = _registry(_Search([_window(OPENING)], [_weather(), _window(CLOSING)]))
    registry.call(_call("search", query="Hafenfest Sonntag"), ctx)

    reply = registry.call(_call("search", query="Band am Kai"), ctx)

    assert _shown(reply.text) == [1, 2]
    assert f"{OPENING}{WINDOW_JOIN}{CLOSING}" in reply.text
    assert WEATHER in reply.text
    assert (reply.found, len(ctx.ledger), ctx.ledger.number_of("programme-2")) == (2, 2, 1)


def test_read_returns_the_full_text_of_a_cut_passage() -> None:
    """A search shows a long passage cut to its share; read(n) shows all of it under the same number."""
    ctx = _context()
    passage = _report(1, sentences=30)
    full_text = passage.node.get_content()
    registry = _registry(_Search([passage]))
    found = registry.call(_call("search", query="Hafenverein"), ctx)
    assert MARKER.search(found.text)

    reply = registry.call(_call("read", n=1), ctx)

    assert _shown(reply.text) == [1]
    assert full_text in reply.text
    assert not MARKER.search(reply.text)
    assert (reply.found, reply.action, reply.detail) == (0, "read", "1")
    assert len(reply.text) <= ctx.budget.result_chars
    assert len(ctx.ledger) == 1


def test_a_repeated_call_is_answered_not_rerun() -> None:
    """The same search again, spaced or cased otherwise, is answered from memory by every number it showed or named."""
    ctx = _context()
    clip = scored_node("clip", CLIP, filename="harbour_walk.webm")
    ctx.ledger.add([clip])
    search = _Search([_programme(), clip, _weather()])
    registry = _registry(search)
    registry.call(_call("search", query="Hafenfest Sonntag"), ctx)

    again = registry.call(_call("search", query="  hafenfest   SONNTAG "), ctx)

    assert search.queries == ["Hafenfest Sonntag"]
    assert again.text == "Already called search with these arguments; see [1], [2], [3]."
    assert (again.found, again.action, again.detail) == (0, "search", "Hafenfest Sonntag")
    assert len(ctx.ledger) == 3


@pytest.mark.parametrize("full", [False, True], ids=["nothing-found", "nothing-admitted"])
def test_a_repeated_call_that_showed_nothing_says_so(full: bool) -> None:
    """A repeat of a search that showed no passage says so, whether it found none or could admit none.

    Args:
        full (bool): Whether the evidence is already full and the search finds new passages.
    """
    ctx = _context(cap=1)
    if full:
        ctx.ledger.add([scored_node("clip", CLIP, filename="harbour_walk.webm")])
    registry = _registry(_Search([_programme(), _weather()] if full else []))
    first = registry.call(_call("search", query="Feuerwerk am Kai"), ctx)

    again = registry.call(_call("search", query="Feuerwerk am Kai"), ctx)

    assert first.text == (
        "2 more matching passages left out: the evidence holds all it can, so answer from the passages shown."
        if full
        else NO_NEW_PASSAGES
    )
    assert (again.text, again.found) == ("Already called search with these arguments; it showed no passages.", 0)


@pytest.mark.parametrize("name", ["search_images", "keyframes", "browse"])
def test_an_unoffered_tool_gets_an_error_reply(name: str) -> None:
    """A tool the turn does not offer is answered with an error naming the tools it does; nothing runs.

    Args:
        name (str): The tool the model called.
    """
    ctx = _context()
    search = _Search()
    call = ToolCall(id="call_0", name=name, arguments='{"description": "Zwei Personen in Regenjacken am Hafen"}')

    reply = _registry(search).call(call, ctx)

    assert reply.text == f"Error: there is no tool named '{name}'; the tools you can call are search, read."
    assert (reply.found, reply.action, reply.detail) == (0, name, "")
    assert search.queries == []
    assert len(ctx.ledger) == 0


@pytest.mark.parametrize(
    ("name", "arguments", "example"),
    [
        ("search", '{"query": "Hafenfest"', '{"query": "..."}'),
        ("search", '["Hafenfest"]', '{"query": "..."}'),
        ("search", '"Hafenfest"', '{"query": "..."}'),
        ("search", "{query: Hafenfest}", '{"query": "..."}'),
        ("read", '{"n": 1', '{"n": 1}'),
    ],
    ids=["cut-off", "list", "string", "unquoted", "read"],
)
def test_malformed_arguments_get_an_error_reply(name: str, arguments: str, example: str) -> None:
    """Arguments that are not a JSON object are answered with an error showing their form; nothing runs.

    Args:
        name (str): The tool called.
        arguments (str): The raw arguments the model sent.
        example (str): The form the reply shows, from the tool's schema.
    """
    ctx = _context()
    search = _Search()

    reply = _registry(search).call(ToolCall(id="call_0", name=name, arguments=arguments), ctx)

    assert reply.text == f"Error: the arguments for {name} must be a JSON object, such as {example}."
    assert (reply.found, reply.action, reply.detail) == (0, name, "")
    assert search.queries == []


@pytest.mark.parametrize(
    ("name", "arguments", "expected"),
    [
        ("search", "", "Error: search needs a query: a few specific words to look for."),
        ("search", "null", "Error: search needs a query: a few specific words to look for."),
        ("search", '{"query": "   "}', "Error: search needs a query: a few specific words to look for."),
        ("search", '{"query": 2026}', "Error: search needs a query: a few specific words to look for."),
        ("read", "{}", "Error: read needs n, the number of a passage shown before, such as 3."),
        ("read", '{"n": "drei"}', "Error: read needs n, the number of a passage shown before, such as 3."),
        ("read", '{"n": true}', "Error: read needs n, the number of a passage shown before, such as 3."),
        ("read", '{"n": 0}', "Error: read needs n, the number of a passage shown before, such as 3."),
        ("read", '{"n": -2}', "Error: read needs n, the number of a passage shown before, such as 3."),
        ("read", '{"n": 1e300}', "Error: read needs n, the number of a passage shown before, such as 3."),
    ],
    ids=["blank", "null", "spaces", "number", "read-missing", "read-word", "read-bool", "zero", "negative", "huge"],
)
def test_a_call_missing_its_argument_gets_the_tools_own_error(name: str, arguments: str, expected: str) -> None:
    """Empty arguments reach the tool, which says what it needs rather than calling them malformed.

    Args:
        name (str): The tool called.
        arguments (str): The raw arguments the model sent.
        expected (str): The reply.
    """
    ctx = _context()
    search = _Search()

    reply = _registry(search).call(ToolCall(id="call_0", name=name, arguments=arguments), ctx)

    assert (reply.text, reply.found, reply.action) == (expected, 0, name)
    assert search.queries == []


def _two_phase(registry: ToolRegistry, calls: Sequence[ToolCall], ctx: ToolContext) -> list[ToolReply]:
    """Answer calls as a turn with parallel calls does: all prepared in order, all fetched, then all finished in order.

    Args:
        registry (ToolRegistry): The turn's tools.
        calls (Sequence[ToolCall]): One round's calls, in the order the model made them.
        ctx (ToolContext): The turn's context.

    Returns:
        list[ToolReply]: One reply per call, in call order.
    """
    prepared = [registry.prepare(call, ctx) for call in calls]
    fetched = [None if ready.fetch is None else ready.fetch() for ready in prepared]
    return [registry.finish(ready, result, ctx) for ready, result in zip(prepared, fetched, strict=True)]


def test_a_two_phase_run_answers_as_call_does() -> None:
    """Preparing a round's calls, fetching, then finishing them in call order answers as calling them one by one."""
    calls = [
        _call("search", query="Hafenfest"),
        _call("search", query="Wetter am Sonntag"),
        _call("read", n=1),
        _call("read", n=7),
        _call("search", query="   "),
        ToolCall(id="call_5", name="browse", arguments="{}"),
        ToolCall(id="call_6", name="read", arguments='{"n": 1'),
    ]
    results = ([_programme(), _window(OPENING)], [_weather(), _programme()])
    sequential_ctx, phased_ctx = _context(), _context()
    one_by_one = _registry(_Search(*results))

    sequential = [one_by_one.call(call, sequential_ctx) for call in calls]
    phased = _two_phase(_registry(_Search(*results)), calls, phased_ctx)

    assert phased == sequential
    for ctx in (phased_ctx, sequential_ctx):
        assert [scored.node.node_id for scored in ctx.ledger.entries()] == ["programme", "programme-2", "weather"]


def test_two_identical_calls_prepared_together_are_admitted_once() -> None:
    """Both pass prepare and fetch; the second is answered from what the first's finish remembered."""
    ctx = _context()
    search = _Search([_programme(), _weather()], [_programme(), _weather()])
    calls = [_call("search", query="Hafenfest Sonntag"), _call("search", query="hafenfest  SONNTAG")]

    first, second = _two_phase(_registry(search), calls, ctx)

    assert search.queries == ["Hafenfest Sonntag", "hafenfest SONNTAG"]
    assert (_shown(first.text), first.found) == ([1, 2], 2)
    assert (second.text, second.found) == ("Already called search with these arguments; see [1], [2].", 0)
    assert len(ctx.ledger) == 2


@pytest.mark.parametrize(
    "call",
    [
        ToolCall(id="call_0", name="search", arguments='{"query": "  "}'),
        ToolCall(id="call_0", name="browse", arguments="{}"),
        ToolCall(id="call_0", name="search", arguments='{"query": '),
    ],
    ids=["argument-error", "unoffered", "malformed"],
)
def test_a_call_answered_at_prepare_needs_no_io(call: ToolCall) -> None:
    """A call whose answer is an error is answered when it is prepared; nothing is fetched or admitted.

    Args:
        call (ToolCall): The call the model made.
    """
    ctx = _context()
    search = _Search()
    registry = _registry(search)

    prepared = registry.prepare(call, ctx)

    assert prepared.fetch is None
    assert registry.finish(prepared, None, ctx).text.startswith("Error: ")
    assert (search.queries, len(ctx.ledger)) == ([], 0)


def test_a_prepare_that_hits_memory_needs_no_io() -> None:
    """A call answered before in the turn is answered when it is prepared, without fetching again."""
    ctx = _context()
    search = _Search([_programme()])
    registry = _registry(search)
    registry.call(_call("search", query="Hafenfest"), ctx)

    prepared = registry.prepare(_call("search", query="HAFENFEST"), ctx)

    assert prepared.fetch is None
    assert registry.finish(prepared, None, ctx).text == "Already called search with these arguments; see [1]."
    assert search.queries == ["Hafenfest"]


def test_read_has_no_io_step() -> None:
    """A read fetches nothing; its range check and render happen at finish, after the calls before it."""
    ctx = _context()
    registry = _registry(_Search([_programme()]))

    prepared = registry.prepare(_call("read", n=1), ctx)
    registry.call(_call("search", query="Hafenfest"), ctx)
    reply = registry.finish(prepared, None, ctx)

    assert prepared.fetch is None
    assert _shown(reply.text) == [1]
    assert PROGRAMME in reply.text


def test_a_prepared_call_carries_what_the_trail_shows() -> None:
    """The loop answers a call whose fetch timed out itself, so a prepared call names its action and argument."""
    ctx = _context()
    registry = _registry(_Search())

    search = registry.prepare(_call("search", query="  Hafenfest   Sonntag "), ctx)
    unoffered = registry.prepare(ToolCall(id="call_1", name="browse", arguments="{}"), ctx)

    assert (search.action, search.detail, search.fetch is not None) == ("search", "Hafenfest Sonntag", True)
    assert (unoffered.action, unoffered.detail, unoffered.fetch) == ("browse", "", None)


def test_a_failed_search_propagates_and_is_not_remembered() -> None:
    """A search that raises fails its call and leaves nothing remembered, so the same call runs again."""
    ctx = _context()
    runs: list[str] = []

    def flaky(query: str) -> list[NodeWithScore]:
        """Fail the first search, then find the programme.

        Args:
            query (str): What the tool searched for.

        Returns:
            list[NodeWithScore]: The programme passage.

        Raises:
            TimeoutError: On the first search.
        """
        runs.append(query)
        if len(runs) == 1:
            raise TimeoutError("The invented search timed out.")
        return [_programme()]

    registry = ToolRegistry([SearchTool(flaky), ReadTool()])
    with pytest.raises(TimeoutError):
        registry.call(_call("search", query="Hafenfest"), ctx)

    reply = registry.call(_call("search", query="Hafenfest"), ctx)

    assert (_shown(reply.text), runs) == ([1], ["Hafenfest", "Hafenfest"])


def test_the_registry_offers_each_tool_as_a_function_schema() -> None:
    """Schemas go to the endpoint as function tools in the order given; changing a copy changes nothing."""
    registry = _registry(_Search())

    schemas = registry.schemas()

    assert [(schema["type"], schema["function"]["name"]) for schema in schemas] == [
        ("function", "search"),
        ("function", "read"),
    ]
    assert all(schema["function"]["description"] for schema in schemas)
    assert schemas[0]["function"]["parameters"]["required"] == ["query"]
    assert schemas[1]["function"]["parameters"]["properties"]["n"]["type"] == "integer"
    assert json.loads(json.dumps(schemas)) == schemas
    assert registry.names == frozenset({"search", "read"})
    schemas[0]["function"]["parameters"]["required"].clear()
    assert registry.schemas()[0]["function"]["parameters"]["required"] == ["query"]


def test_a_registry_refuses_two_tools_of_one_name() -> None:
    """A call names its tool, so two tools of one name could not both be answered."""
    with pytest.raises(ValueError, match="search"):
        ToolRegistry([SearchTool(_Search()), SearchTool(_Search())])


@pytest.mark.parametrize("reasoning", [False, True], ids=["plain", "reasoning"])
@pytest.mark.parametrize("count", [1, 3])
def test_every_passage_in_a_reply_stays_within_passage_chars(reasoning: bool, count: int) -> None:
    """A long passage alone takes no more than its cap, and several share the reply, each within it.

    Args:
        reasoning (bool): Whether the model thinks, which shortens replies and passages.
        count (int): How many long passages the search finds.
    """
    ctx = _context(reasoning=reasoning)
    found = [_report(number, sentences=30) for number in range(1, count + 1)]

    reply = _registry(_Search(found)).call(_call("search", query="Hafenverein"), ctx)

    bodies = _bodies(reply.text)
    assert sorted(bodies) == list(range(1, count + 1))
    assert all(MIN_TEXT_CHARS <= len(MARKER.sub("", body)) <= ctx.budget.passage_chars for body in bodies.values())
    assert len(reply.text) <= ctx.budget.result_chars


def _addendum(number: int) -> NodeWithScore:
    """Build a later window of an invented planning report, which extends the report's entry.

    Args:
        number (int): Which report; it shares the report's id, file and page.

    Returns:
        NodeWithScore: The window.
    """
    text = f"Nachtrag {number}: Der Lauf am Kai beginnt erst nach dem Konzert."
    return scored_node(f"report-{number}", text, filename=f"planungsbericht_{number}.pdf", page=number)


@pytest.mark.parametrize("reasoning", [False, True], ids=["plain", "reasoning"])
@pytest.mark.parametrize(
    ("again", "grown", "cap", "notes"),
    [
        (8, 0, MAX_LEDGER_ENTRIES, ("Also matched, shown before: [1]", "did not fit in this reply")),
        (
            4,
            4,
            MAX_LEDGER_ENTRIES,
            ("shown before: [1], [2], [3], [4].", "new text not shown here: [5], [6], [7], [8];", "did not fit"),
        ),
        (0, 0, 4, ("the evidence holds all it can",)),
    ],
    ids=["named-again", "named-again-and-grown", "evidence-full"],
)
def test_a_reply_with_notes_never_outgrows_result_chars(
    reasoning: bool, again: int, grown: int, cap: int, notes: tuple[str, ...]
) -> None:
    """Room for the notes after the passages is kept before any passage takes it, however the reply ends.

    Args:
        reasoning (bool): Whether the model thinks, which shortens replies.
        again (int): How many found passages were shown before and are found unchanged.
        grown (int): How many were shown before and are found grown, past the reply's room.
        cap (int): The most passages the turn's ledger admits.
        notes (tuple[str, ...]): Parts of the notes the reply must end with.
    """
    ctx = _context(reasoning=reasoning, cap=cap)
    before = [_report(number) for number in range(1, again + grown + 1)]
    ctx.ledger.add(before)
    fresh = [_report(number) for number in range(again + grown + 1, again + grown + 11)]
    found = [*before[:again], *fresh, *(_addendum(number) for number in range(again + 1, again + grown + 1))]

    reply = _registry(_Search(found)).call(_call("search", query="Hafenverein"), ctx)

    assert all(note in reply.text for note in notes)
    assert not set(_shown(reply.text)) & set(range(1, again + grown + 1))
    assert len(reply.text) <= ctx.budget.result_chars


def test_a_long_query_is_cut_for_the_search_and_the_trail() -> None:
    """A run-on query reaches the search on one line within its limit, and the trail shows less of it."""
    ctx = _context()
    search = _Search([])
    query = "Hafenfest   " + " ".join(f"Stand{number}" for number in range(80))

    reply = _registry(search).call(_call("search", query=query), ctx)

    sent = " ".join(query.split())[:300]
    assert search.queries == [sent]
    assert reply.detail == sent[:200]


def test_a_called_name_is_read_without_surrounding_spaces() -> None:
    """A name padded with spaces still reaches its tool."""
    ctx = _context()
    search = _Search([_programme()])

    reply = _registry(search).call(ToolCall(id="call_0", name=" search ", arguments='{"query": "Hafenfest"}'), ctx)

    assert (_shown(reply.text), reply.action) == ([1], "search")


def _matched(number: int) -> NodeWithScore:
    """Build a section of three windows, each holding a retrieved match, whose floor exceeds a thinking passage.

    Args:
        number (int): Which section; it also varies the id and file.

    Returns:
        NodeWithScore: The section, its matches stamped where they sit in its text.
    """
    windows = [f"Fenster {number}.{part}: " + "Der Hafenverein plant Stände am Kai. " * 12 for part in range(1, 4)]
    spans: list[list[int]] = []
    offset = 0
    for window in windows:
        middle = offset + len(window) // 2
        spans.append([middle, middle + 20])
        offset += len(window) + len(WINDOW_JOIN)
    return scored_node(
        f"matched-{number}",
        WINDOW_JOIN.join(windows),
        filename=f"protokoll_{number}.pdf",
        page=1,
        **{MATCH_SPANS_KEY: spans},
    )


def test_a_passage_whose_matches_need_more_than_passage_chars_is_shown_within_it() -> None:
    """The cap wins over a passage's floor, so two such passages still share one reply; read(n) shows the rest."""
    ctx = _context(reasoning=True)
    found = [_matched(1), _matched(2)]
    headers, bodies, anchors = passage_parts([stamp(found[0], 1)], llm_view, compact=True)
    assert passage_costs(headers[0], bodies[0], anchors[0])[1] > ctx.budget.passage_chars

    reply = _registry(_Search(found)).call(_call("search", query="Hafenverein Stände"), ctx)

    bodies_shown = _bodies(reply.text)
    assert sorted(bodies_shown) == [1, 2]
    assert all(len(MARKER.sub("", body)) <= ctx.budget.passage_chars for body in bodies_shown.values())
    assert all(body.count(WINDOW_JOIN) == 2 for body in bodies_shown.values())
    assert len(reply.text) <= ctx.budget.result_chars


def test_an_extension_without_room_is_named_as_grown() -> None:
    """A new window of a passage shown before joins its entry without room to show it, and is named as grown."""
    ctx = _context(reasoning=True)
    opening, closing = OPENING * 8, CLOSING * 12
    ctx.ledger.add([_window(opening)])
    reports = [_report(number) for number in range(1, 5)]

    reply = _registry(_Search([*reports, _window(closing)])).call(_call("search", query="Hafenverein"), ctx)

    assert _shown(reply.text) == [2, 3, 4, 5]
    assert reply.text.endswith("\n\nAlso matched, with new text not shown here: [1]; read(n) shows it.")
    assert "shown before" not in reply.text
    assert isinstance(reply, PassageReply)
    assert (reply.found, reply.numbers) == (5, (1, 2, 3, 4, 5))
    assert ctx.ledger.entries()[0].node.get_content() == f"{opening}{WINDOW_JOIN}{closing}"
    assert len(reply.text) <= ctx.budget.result_chars


def test_a_repeated_read_shows_a_passage_again_once_it_grew() -> None:
    """A read is answered from memory only while its passage is unchanged; once the passage grew, it is shown anew."""
    ctx = _context()
    registry = _registry(_Search([_window(OPENING)], [_window(CLOSING)]))
    registry.call(_call("search", query="Hafenfest Sonntag"), ctx)

    first = registry.call(_call("read", n=1), ctx)
    unchanged = registry.call(_call("read", n=1), ctx)
    registry.call(_call("search", query="Band am Kai"), ctx)
    grown = registry.call(_call("read", n=1), ctx)
    after = registry.call(_call("read", n=1), ctx)

    assert OPENING in first.text
    assert CLOSING not in first.text
    assert unchanged.text == "Already called read with these arguments; see [1]."
    assert _shown(grown.text) == [1]
    assert f"{OPENING}{WINDOW_JOIN}{CLOSING}" in grown.text
    assert after.text == "Already called read with these arguments; see [1]."


def test_a_read_is_shown_anew_once_its_passage_gained_a_match() -> None:
    """A search can add a match deep in a long passage without adding text; a read then keeps that part too."""
    ctx = _context()
    body = " ".join(f"Satz {line}: Der Hafenverein plant Stände und Musik am Kai." for line in range(1, 81))
    start = len(body) - 400
    held = scored_node("programme-9", body, filename="hafenfest_programm.pdf", page=3, **{MATCH_SPANS_KEY: [[60, 100]]})
    window = scored_node(
        "programme-9", body[start:], filename="hafenfest_programm.pdf", page=3, **{MATCH_SPANS_KEY: [[200, 240]]}
    )
    registry = _registry(_Search([held], [window]))
    registry.call(_call("search", query="Hafenverein"), ctx)
    before = registry.call(_call("read", n=1), ctx)
    registry.call(_call("search", query="Musik am Kai"), ctx)

    after = registry.call(_call("read", n=1), ctx)

    late = body[start + 200 : start + 240]
    assert late not in before.text
    assert late in after.text
    assert ctx.ledger.entries()[0].node.get_content() == body


def test_a_remembered_read_waits_for_finish_when_its_round_may_grow_the_passage() -> None:
    """A read remembered in an earlier round is decided at finish, after a search before it in its round grew [1]."""

    def last_reply(two_phase: bool) -> ToolReply:
        """Run two rounds: a search and a read of [1], then a search extending [1] and the same read.

        Args:
            two_phase (bool): Whether the second round is prepared together, as parallel calls are.

        Returns:
            ToolReply: The second round's read.
        """
        ctx = _context()
        registry = _registry(_Search([_window(OPENING)], [_window(CLOSING)]))
        registry.call(_call("search", query="Hafenfest Sonntag"), ctx)
        registry.call(_call("read", n=1), ctx)
        calls = [_call("search", query="Band am Kai"), _call("read", n=1)]
        replies = _two_phase(registry, calls, ctx) if two_phase else [registry.call(call, ctx) for call in calls]
        return replies[1]

    sequential, phased = last_reply(two_phase=False), last_reply(two_phase=True)

    assert phased == sequential
    assert f"{OPENING}{WINDOW_JOIN}{CLOSING}" in phased.text


def test_an_unchanged_remembered_read_is_answered_from_memory_at_finish() -> None:
    """A remembered read needs no I/O; memory answers it at finish, while its passage is unchanged."""
    ctx = _context()
    registry = _registry(_Search([_window(OPENING)]))
    registry.call(_call("search", query="Hafenfest Sonntag"), ctx)
    registry.call(_call("read", n=1), ctx)

    prepared = registry.prepare(_call("read", n=1), ctx)

    assert (prepared.reply, prepared.fetch) == (None, None)
    assert registry.finish(prepared, None, ctx).text == "Already called read with these arguments; see [1]."


def test_a_read_finished_after_an_extending_search_in_its_round_shows_the_growth() -> None:
    """Prepared together, the same read before and after a search that grows its passage is answered twice."""
    ctx = _context()
    registry = _registry(_Search([_window(OPENING)], [_window(CLOSING)]))
    registry.call(_call("search", query="Hafenfest Sonntag"), ctx)
    calls = [_call("read", n=1), _call("search", query="Band am Kai"), _call("read", n=1)]

    before, _, after = _two_phase(registry, calls, ctx)

    assert CLOSING not in before.text
    assert f"{OPENING}{WINDOW_JOIN}{CLOSING}" in after.text


def test_a_long_header_is_compacted_so_it_costs_a_reply_little() -> None:
    """A passage's header is cut to the compact form a reply prices it at, so long names still leave room."""
    ctx = _context()
    found = [
        scored_node(f"long-{number}", _report(number).node.get_content(), filename=f"{number}_" + "hafen" * 200)
        for number in range(1, 6)
    ]

    reply = _registry(_Search(found)).call(_call("search", query="Hafenverein"), ctx)

    headers = [block.partition("\n\n")[0] for block in re.split(r"\n\n(?=citation_index: )", NOTES.sub("", reply.text))]
    assert _shown(reply.text) == [1, 2, 3, 4]
    assert all(
        len(header) <= len("citation_index: 1\n") + COMPACT_HEADER_CHARS + MARKER_RESERVE_CHARS for header in headers
    )
    assert len(reply.text) <= ctx.budget.result_chars


def test_a_post_header_reaches_the_model_whole_with_its_number_first() -> None:
    """Who posted a passage, and when, is evidence too: a header of ordinary length is never cut."""
    ctx = ToolContext(
        ledger=EvidenceLedger(cap=MAX_LEDGER_ENTRIES),
        budget=_budget(),
        llm_view=ParentContextPostprocessor._emit_with_llm_exclusion,
    )
    registry = _registry(_Search([social_post(3)]))

    found = registry.call(_call("search", query="Harbour festival"), ctx)
    read = registry.call(_call("read", n=1), ctx)

    for reply in (found, read):
        header = reply.text.partition("\n\n")[0]
        assert header.startswith("citation_index: 1\nfilename: example_profile/dossier.json\n")
        assert "'author': 'Harbour Events'" in header
        assert "'text_id': '9876543213'" in header
        assert not MARKER.search(header)


@pytest.mark.parametrize(("extra", "whole"), [(0, True), (1, False)], ids=["at-the-limit", "past-the-limit"])
def test_a_header_is_compacted_only_past_header_chars(extra: int, whole: bool) -> None:
    """A header of HEADER_CHARS shows whole; one character more and it keeps its citation line and a cut rest.

    Args:
        extra (int): Characters past the limit the header runs.
        whole (bool): Whether the reply shows it whole.
    """
    ctx = _context()
    name = "h" * (HEADER_CHARS - len("citation_index: 1\nfilename: ") + extra)
    reply = _registry(_Search([scored_node("named", PROGRAMME, filename=name)])).call(
        _call("search", query="Hafenfest"), ctx
    )

    header = reply.text.partition("\n\n")[0]
    assert header.startswith("citation_index: 1\nfilename: hhh")
    assert (header == f"citation_index: 1\nfilename: {name}") is whole
    assert (MARKER.search(header) is None) is whole


def test_a_passage_fits_exactly_when_the_room_covers_its_reserve_and_floor() -> None:
    """Admission at its boundary: text room must remain after the reserves and cover every floor."""
    ctx = _context()
    passage = _plan(_report(1), 1, ctx)
    empty = _plan(scored_node("empty", "", filename="leer.pdf"), 2, ctx)

    assert _fits([passage], passage.reserve + passage.floor)
    assert not _fits([passage], passage.reserve + passage.floor - 1)
    assert not _fits([empty], empty.reserve)
    assert _fits([empty], empty.reserve + 1)


def test_windows_of_a_new_section_in_one_search_share_its_number() -> None:
    """Two windows of one section found together are one passage under one number."""
    ctx = _context()

    reply = _registry(_Search([_window(OPENING), _weather(), _window(CLOSING)])).call(
        _call("search", query="Hafenfest Programm"), ctx
    )

    assert _shown(reply.text) == [1, 2]
    assert f"{OPENING}{WINDOW_JOIN}{CLOSING}" in reply.text
    assert (reply.found, len(ctx.ledger)) == (2, 2)


def test_a_full_ledger_leaves_new_passages_out_and_says_so() -> None:
    """Once the evidence holds all it may, new passages stay unnumbered and the model is told to answer."""
    ctx = _context(cap=1)

    reply = _registry(_Search([_programme(), _weather()])).call(_call("search", query="Hafenfest"), ctx)

    assert _shown(reply.text) == [1]
    assert reply.text.endswith(
        "\n\n1 more matching passage left out: the evidence holds all it can, so answer from the passages shown."
    )
    assert (reply.found, len(ctx.ledger), ctx.ledger.number_of("weather")) == (1, 1, None)


def test_a_search_that_finds_nothing_new_says_so() -> None:
    """Nothing found, or only passages shown before, is answered as no new passages."""
    ctx = _context()
    registry = _registry(_Search([], [_programme()], [_programme()]))

    nothing = registry.call(_call("search", query="Feuerwerk"), ctx)
    registry.call(_call("search", query="Hafenfest"), ctx)
    again = registry.call(_call("search", query="Stände Sonntag"), ctx)

    assert (nothing.text, nothing.found) == (NO_NEW_PASSAGES, 0)
    assert (again.text, again.found) == (f"{NO_NEW_PASSAGES}\nAlso matched, shown before: [1].", 0)


@pytest.mark.parametrize("number", [1, "1", "[1]", 1.0], ids=["int", "text", "bracketed", "float"])
def test_read_accepts_a_number_written_as_text(number: object) -> None:
    """Models write the number as they see it cited; each form reads the same passage.

    Args:
        number (object): The ``n`` the model sent.
    """
    ctx = _context()
    registry = _registry(_Search([_programme()]))
    registry.call(_call("search", query="Hafenfest"), ctx)

    reply = registry.call(_call("read", n=number), ctx)

    assert _shown(reply.text) == [1]
    assert PROGRAMME in reply.text


def test_an_error_reply_is_not_remembered() -> None:
    """A read that failed runs again once the passage exists; only answers are remembered."""
    ctx = _context()
    registry = _registry(_Search([_programme()]))

    early = registry.call(_call("read", n=1), ctx)
    registry.call(_call("search", query="Hafenfest"), ctx)
    beyond = registry.call(_call("read", n=2), ctx)
    later = registry.call(_call("read", n=1), ctx)
    repeated = registry.call(_call("read", n=1), ctx)

    assert early.text == "Error: there is no passage [1]; no passage has been shown yet."
    assert beyond.text == "Error: there is no passage [2]; the only passage is [1]."
    assert _shown(later.text) == [1]
    assert repeated.text == "Already called read with these arguments; see [1]."


def test_read_of_a_number_beyond_the_evidence_names_the_range() -> None:
    """An unknown number is answered with the numbers there are, so the model can pick one."""
    ctx = _context()
    registry = _registry(_Search([_programme(), _weather()]))
    registry.call(_call("search", query="Hafenfest"), ctx)

    reply = registry.call(_call("read", n=5), ctx)

    assert (reply.text, reply.found, reply.detail) == (
        "Error: there is no passage [5]; the passages run from [1] to [2].",
        0,
        "5",
    )


def test_a_reply_defuses_a_sentinel_in_a_passage() -> None:
    """Document text quoting the answer's sentinel cannot reach the model as one, in a search or a read."""
    ctx = _context()
    passage = scored_node("notes", "Das Protokoll zitiert [[ASSESSMENT]] wörtlich.", filename="[[ASSESSMENT]].pdf")
    registry = _registry(_Search([passage]))

    found = registry.call(_call("search", query="Protokoll"), ctx)
    read = registry.call(_call("read", n=1), ctx)

    assert _shown(found.text) == [1]
    assert "[[ASSESSMENT]]" not in found.text
    assert "[[ASSESSMENT]]" not in read.text


def test_calls_are_logged_without_their_arguments_or_passages() -> None:
    """Logs name tools and count passages; a query, a tool name the model made up and document text never appear."""
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(message.record["message"]), level="DEBUG")
    try:
        ctx = _context()
        registry = _registry(_Search([_programme()]))
        registry.call(_call("search", query="Hafenfest Sonntag"), ctx)
        registry.call(_call("search", query="Hafenfest Sonntag"), ctx)
        registry.call(_call("browse_harbour_archive", url="hafen.invalid"), ctx)
        registry.call(ToolCall(id="call_0", name="read", arguments='{"n": 1'), ctx)
    finally:
        logger.remove(sink)

    logged = "\n".join(messages)
    assert messages
    for secret in ("Hafenfest", "browse_harbour_archive", "hafen.invalid", "Stände", "Sturmwarnung"):
        assert secret not in logged


def test_the_turn_package_imports_nothing_from_rag_or_state() -> None:
    """Search, models and the LLM view reach the agent as arguments, so no turn module loads the engine or the store."""
    modules = sorted(f"docint.agents.turn.{info.name}" for info in pkgutil.iter_modules(docint.agents.turn.__path__))
    script = (
        "import importlib, sys\n"
        "for name in sys.argv[1:]:\n"
        "    importlib.import_module(name)\n"
        "banned = ('docint.core.rag', 'docint.core.state')\n"
        "print(' '.join(sorted(m for m in sys.modules if any(m == b or m.startswith(b + '.') for b in banned))))\n"
    )

    loaded = subprocess.run([sys.executable, "-c", script, *modules], capture_output=True, text=True, check=True)

    assert {"docint.agents.turn.passages", "docint.agents.turn.tools"} <= set(modules)
    assert loaded.stdout.split() == []
