"""The agent's tool registry and its search and read tools. Every passage, query and file here is invented."""

from __future__ import annotations

import inspect
import json
import pkgutil
import re
import subprocess
import sys
import threading
from collections.abc import Callable, Mapping
from dataclasses import fields
from typing import Any

import pytest
from _pytest.logging import LogCaptureFixture
from agent_fakes import (
    LEDGER_CAP,
    PROGRAMME,
    ScriptedSearch,
    programme_passage,
    shown_numbers,
    tool_call,
    tool_context,
    two_phase_round,
)
from assessment_fakes import llm_view, scored_node, social_post
from llama_index.core.schema import NodeWithScore
from loguru import logger

import docint.agents.turn
from docint.agents.assessment.ledger import (
    COMPACT_HEADER_CHARS,
    MARKER_RESERVE_CHARS,
    MIN_TEXT_CHARS,
    WINDOW_JOIN,
    passage_costs,
    passage_parts,
    stamp,
)
from docint.agents.turn.calls import FAILED_REPLY, RoundCalls
from docint.agents.turn.passages import HEADER_CHARS, NO_NEW_PASSAGES, _fits, _plan
from docint.agents.turn.stream import ToolCall
from docint.agents.turn.tools import (
    MAX_ARGUMENT_DEPTH,
    PassageReply,
    ReadTool,
    SearchTool,
    ToolContext,
    ToolRegistry,
    ToolReply,
    ToolWork,
    cleared_text,
    error_reply,
    nested_too_deeply,
    passage_reply,
    trail_detail,
)
from docint.core.rag import ParentContextPostprocessor
from docint.utils.text_budget import MATCH_SPANS_KEY

WEATHER = "Für Sonntag meldet der Wetterdienst bedeckten Himmel, Wind aus West und kühle Temperaturen am Hafen."
CLIP = "Im Clip harbour_walk.webm gehen zwei Personen in Regenjacken am Hafen entlang; Person A trägt einen Schirm."
OPENING = "Der Hafenverein lädt am Sonntag zum Hafenfest ein."
CLOSING = "Am Abend spielt eine Band am Kai."

#: The cut marker a shortened passage carries.
MARKER = re.compile(r"\[\.\.\. \d+ of \d+ characters not shown \.\.\.\]")
#: The notes a search reply ends with, after its passages.
NOTES = re.compile(r"\n\n(?:No new passages\.|Also matched, |\d+ more matching ).*\Z", re.DOTALL)


def _registry(search: ScriptedSearch) -> ToolRegistry:
    """Build the registry of a documents turn: search and read.

    Args:
        search (ScriptedSearch): The search the search tool runs.

    Returns:
        ToolRegistry: The registry.
    """
    return ToolRegistry([SearchTool(search), ReadTool()])


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


def test_search_numbers_new_passages_and_refers_to_held_ones() -> None:
    """Passages new to the turn are shown under the next numbers; ones shown before are named by number alone."""
    ctx = tool_context()
    programme, clip = programme_passage(), scored_node("clip", CLIP, filename="harbour_walk.webm")
    ctx.ledger.add([programme, clip])
    report = _report(1, sentences=2)
    search = ScriptedSearch([_weather(), clip, programme, report])

    reply = _registry(search).call(tool_call("search", query="Hafenfest Wetter am Sonntag"), ctx)

    assert search.queries == ["Hafenfest Wetter am Sonntag"]
    assert shown_numbers(reply.text) == [3, 4]
    assert WEATHER in reply.text
    assert report.node.get_content() in reply.text
    assert PROGRAMME not in reply.text
    assert CLIP not in reply.text
    assert reply.text.endswith("\n\nAlso matched, shown before: [1], [2].")
    assert (reply.found, reply.action, reply.detail) == (2, "search", "Hafenfest Wetter am Sonntag")
    assert [ctx.ledger.number_of(node_id) for node_id in ("programme", "clip", "weather", "report-1")] == [1, 2, 3, 4]


def test_a_result_never_numbers_what_it_did_not_show() -> None:
    """Passages a reply has no room to show stay unnumbered, so every number the model can cite was shown to it."""
    ctx = tool_context(reasoning=True)
    found = [_report(number) for number in range(1, 7)]

    reply = _registry(ScriptedSearch(found)).call(tool_call("search", query="Hafenverein Planung"), ctx)

    admitted = len(ctx.ledger)
    assert 0 < admitted < len(found)
    assert shown_numbers(reply.text) == list(range(1, admitted + 1))
    assert all(ctx.ledger.number_of(scored.node.node_id) is None for scored in found[admitted:])
    assert reply.found == admitted
    assert reply.text.endswith(
        f"\n\n{len(found) - admitted} more matching passages did not fit in this reply; "
        "call search again with narrower words to see them."
    )
    assert len(reply.text) <= ctx.budget.result_chars
    # Every passage shown is legible: it keeps at least the floor a cut passage must keep.
    assert all(len(MARKER.sub("", body)) >= MIN_TEXT_CHARS for body in _bodies(reply.text).values())


def test_an_extended_window_keeps_its_number() -> None:
    """A later window of a section shown before extends it under the same number, listed by number before a new one."""
    ctx = tool_context()
    registry = _registry(ScriptedSearch([_window(OPENING)], [_weather(), _window(CLOSING)]))
    registry.call(tool_call("search", query="Hafenfest Sonntag"), ctx)

    reply = registry.call(tool_call("search", query="Band am Kai"), ctx)

    assert shown_numbers(reply.text) == [1, 2]
    assert f"{OPENING}{WINDOW_JOIN}{CLOSING}" in reply.text
    assert WEATHER in reply.text
    assert (reply.found, len(ctx.ledger), ctx.ledger.number_of("programme-2")) == (2, 2, 1)


def test_read_returns_the_full_text_of_a_cut_passage() -> None:
    """A search shows a long passage cut to its share; read(n) shows all of it under the same number."""
    ctx = tool_context()
    passage = _report(1, sentences=30)
    full_text = passage.node.get_content()
    registry = _registry(ScriptedSearch([passage]))
    found = registry.call(tool_call("search", query="Hafenverein"), ctx)
    assert MARKER.search(found.text)

    reply = registry.call(tool_call("read", n=1), ctx)

    assert shown_numbers(reply.text) == [1]
    assert full_text in reply.text
    assert not MARKER.search(reply.text)
    assert (reply.found, reply.action, reply.detail) == (0, "read", "1")
    assert len(reply.text) <= ctx.budget.result_chars
    assert len(ctx.ledger) == 1


def test_a_repeated_call_is_answered_not_rerun() -> None:
    """The same search again, spaced or cased otherwise, is answered from memory by every number it showed or named."""
    ctx = tool_context()
    clip = scored_node("clip", CLIP, filename="harbour_walk.webm")
    ctx.ledger.add([clip])
    search = ScriptedSearch([programme_passage(), clip, _weather()])
    registry = _registry(search)
    registry.call(tool_call("search", query="Hafenfest Sonntag"), ctx)

    again = registry.call(tool_call("search", query="  hafenfest   SONNTAG "), ctx)

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
    ctx = tool_context(cap=1)
    if full:
        ctx.ledger.add([scored_node("clip", CLIP, filename="harbour_walk.webm")])
    registry = _registry(ScriptedSearch([programme_passage(), _weather()] if full else []))
    first = registry.call(tool_call("search", query="Feuerwerk am Kai"), ctx)

    again = registry.call(tool_call("search", query="Feuerwerk am Kai"), ctx)

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
    ctx = tool_context()
    search = ScriptedSearch()
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
    ctx = tool_context()
    search = ScriptedSearch()

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
    ctx = tool_context()
    search = ScriptedSearch()

    reply = _registry(search).call(ToolCall(id="call_0", name=name, arguments=arguments), ctx)

    assert (reply.text, reply.found, reply.action) == (expected, 0, name)
    assert search.queries == []


def test_an_error_reply_must_name_its_kind() -> None:
    """The kind is keyword-only and has no default, so a new error reply cannot go without one."""
    kind = inspect.signature(error_reply).parameters["kind"]

    assert (kind.kind, kind.default) == (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.empty)
    assert error_reply("search", "invented", kind="failed").kind == "failed"


@pytest.mark.parametrize(
    ("call", "kind"),
    [
        pytest.param(tool_call("nope"), "unknown_tool", id="unknown-tool"),
        pytest.param(ToolCall(id="call_0", name="read", arguments="[1]"), "bad_arguments", id="not-an-object"),
        pytest.param(
            ToolCall(id="call_0", name="read", arguments='{"n": ' + "[" * 600 + "]" * 600 + "}"),
            "nested_arguments",
            id="nested-too-deeply",
        ),
        pytest.param(tool_call("read", n="drei"), "bad_argument", id="read-not-a-number"),
        pytest.param(tool_call("read"), "bad_argument", id="read-without-n"),
        pytest.param(tool_call("read", n=9), "no_such_passage", id="read-no-such-passage"),
        pytest.param(tool_call("search"), "missing_argument", id="search-without-query"),
        pytest.param(tool_call("search", query="   "), "missing_argument", id="search-blank-query"),
    ],
)
def test_every_error_reply_names_its_kind(call: ToolCall, kind: str) -> None:
    """Each way the registry and the search and read tools refuse a call is told apart by a kind token.

    Args:
        call (ToolCall): The call the model made.
        kind (str): Why it was refused.
    """
    reply = ToolRegistry([SearchTool(ScriptedSearch()), ReadTool()]).call(call, tool_context())

    assert reply.error and reply.kind == kind


def test_a_two_phase_run_answers_as_call_does() -> None:
    """Preparing a round's calls, fetching, then finishing them in call order answers as calling them one by one."""
    calls = [
        tool_call("search", query="Hafenfest"),
        tool_call("search", query="Wetter am Sonntag"),
        tool_call("read", n=1),
        tool_call("read", n=7),
        tool_call("search", query="   "),
        ToolCall(id="call_5", name="browse", arguments="{}"),
        ToolCall(id="call_6", name="read", arguments='{"n": 1'),
    ]
    results = ([programme_passage(), _window(OPENING)], [_weather(), programme_passage()])
    sequential_ctx, phased_ctx = tool_context(), tool_context()
    one_by_one = _registry(ScriptedSearch(*results))

    sequential = [one_by_one.call(call, sequential_ctx) for call in calls]
    phased = two_phase_round(_registry(ScriptedSearch(*results)), calls, phased_ctx)

    assert phased == sequential
    for ctx in (phased_ctx, sequential_ctx):
        assert [scored.node.node_id for scored in ctx.ledger.entries()] == ["programme", "programme-2", "weather"]


def test_two_identical_calls_prepared_together_are_admitted_once() -> None:
    """Both pass prepare and fetch; the second is answered from what the first's finish remembered."""
    ctx = tool_context()
    search = ScriptedSearch([programme_passage(), _weather()], [programme_passage(), _weather()])
    calls = [tool_call("search", query="Hafenfest Sonntag"), tool_call("search", query="hafenfest  SONNTAG")]

    first, second = two_phase_round(_registry(search), calls, ctx)

    assert search.queries == ["Hafenfest Sonntag", "hafenfest SONNTAG"]
    assert (shown_numbers(first.text), first.found) == ([1, 2], 2)
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
    ctx = tool_context()
    search = ScriptedSearch()
    registry = _registry(search)

    prepared = registry.prepare(call, ctx)

    assert prepared.fetch is None
    assert registry.finish(prepared, None, ctx).text.startswith("Error: ")
    assert (search.queries, len(ctx.ledger)) == ([], 0)


def test_a_prepare_that_hits_memory_needs_no_io() -> None:
    """A call answered before in the turn is answered when it is prepared, without fetching again."""
    ctx = tool_context()
    search = ScriptedSearch([programme_passage()])
    registry = _registry(search)
    registry.call(tool_call("search", query="Hafenfest"), ctx)

    prepared = registry.prepare(tool_call("search", query="HAFENFEST"), ctx)

    assert prepared.fetch is None
    assert registry.finish(prepared, None, ctx).text == "Already called search with these arguments; see [1]."
    assert search.queries == ["Hafenfest"]


def test_read_has_no_io_step() -> None:
    """A read fetches nothing; its range check and render happen at finish, after the calls before it."""
    ctx = tool_context()
    registry = _registry(ScriptedSearch([programme_passage()]))

    prepared = registry.prepare(tool_call("read", n=1), ctx)
    registry.call(tool_call("search", query="Hafenfest"), ctx)
    reply = registry.finish(prepared, None, ctx)

    assert prepared.fetch is None
    assert shown_numbers(reply.text) == [1]
    assert PROGRAMME in reply.text


def test_a_prepared_call_carries_what_the_trail_shows() -> None:
    """The loop answers a call whose fetch timed out itself, so a prepared call names its action and argument."""
    ctx = tool_context()
    registry = _registry(ScriptedSearch())

    search = registry.prepare(tool_call("search", query="  Hafenfest   Sonntag "), ctx)
    unoffered = registry.prepare(ToolCall(id="call_1", name="browse", arguments="{}"), ctx)

    assert (search.action, search.detail, search.fetch is not None) == ("search", "Hafenfest Sonntag", True)
    assert (unoffered.action, unoffered.detail, unoffered.fetch) == ("browse", "", None)


def test_a_failed_search_propagates_and_is_not_remembered() -> None:
    """A search that raises fails its call and leaves nothing remembered, so the same call runs again."""
    ctx = tool_context()
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
        return [programme_passage()]

    registry = ToolRegistry([SearchTool(flaky), ReadTool()])
    with pytest.raises(TimeoutError):
        registry.call(tool_call("search", query="Hafenfest"), ctx)

    reply = registry.call(tool_call("search", query="Hafenfest"), ctx)

    assert (shown_numbers(reply.text), runs) == ([1], ["Hafenfest", "Hafenfest"])


def test_the_registry_offers_each_tool_as_a_function_schema() -> None:
    """Schemas go to the endpoint as function tools in the order given; changing a copy changes nothing."""
    registry = _registry(ScriptedSearch())

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
        ToolRegistry([SearchTool(ScriptedSearch()), SearchTool(ScriptedSearch())])


@pytest.mark.parametrize("reasoning", [False, True], ids=["plain", "reasoning"])
@pytest.mark.parametrize("count", [1, 3])
def test_every_passage_in_a_reply_stays_within_passage_chars(reasoning: bool, count: int) -> None:
    """A long passage alone takes no more than its cap, and several share the reply, each within it.

    Args:
        reasoning (bool): Whether the model thinks, which shortens replies and passages.
        count (int): How many long passages the search finds.
    """
    ctx = tool_context(reasoning=reasoning)
    found = [_report(number, sentences=30) for number in range(1, count + 1)]

    reply = _registry(ScriptedSearch(found)).call(tool_call("search", query="Hafenverein"), ctx)

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
        (8, 0, LEDGER_CAP, ("Also matched, shown before: [1]", "did not fit in this reply")),
        (
            4,
            4,
            LEDGER_CAP,
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
    ctx = tool_context(reasoning=reasoning, cap=cap)
    before = [_report(number) for number in range(1, again + grown + 1)]
    ctx.ledger.add(before)
    fresh = [_report(number) for number in range(again + grown + 1, again + grown + 11)]
    found = [*before[:again], *fresh, *(_addendum(number) for number in range(again + 1, again + grown + 1))]

    reply = _registry(ScriptedSearch(found)).call(tool_call("search", query="Hafenverein"), ctx)

    assert all(note in reply.text for note in notes)
    assert not set(shown_numbers(reply.text)) & set(range(1, again + grown + 1))
    assert len(reply.text) <= ctx.budget.result_chars


def test_a_tools_own_note_goes_last_unless_the_tool_puts_it_first() -> None:
    """Putting the note first only moves it: a tool that does not ask keeps its note after the passages."""
    note = "Eine eigene Notiz des Werkzeugs."
    tail = f"\n\n{note}"

    last = passage_reply([programme_passage()], tool_context(), action="search", detail="Hafenfest", note=note)
    first = passage_reply(
        [programme_passage()], tool_context(), action="search", detail="Hafenfest", note=note, note_first=True
    )

    assert last.text.endswith(tail)
    assert first.text == f"{note}\n\n{last.text.removesuffix(tail)}"
    assert (first.found, first.numbers) == (last.found, last.numbers)


def test_a_long_query_is_cut_for_the_search_and_the_trail() -> None:
    """A run-on query reaches the search on one line within its limit, and the trail shows less of it."""
    ctx = tool_context()
    search = ScriptedSearch([])
    query = "Hafenfest   " + " ".join(f"Stand{number}" for number in range(80))

    reply = _registry(search).call(tool_call("search", query=query), ctx)

    sent = " ".join(query.split())[:300]
    assert search.queries == [sent]
    assert reply.detail == sent[:200]


def test_a_called_name_is_read_without_surrounding_spaces() -> None:
    """A name padded with spaces still reaches its tool."""
    ctx = tool_context()
    search = ScriptedSearch([programme_passage()])

    reply = _registry(search).call(ToolCall(id="call_0", name=" search ", arguments='{"query": "Hafenfest"}'), ctx)

    assert (shown_numbers(reply.text), reply.action) == ([1], "search")


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
    ctx = tool_context(reasoning=True)
    found = [_matched(1), _matched(2)]
    headers, bodies, anchors = passage_parts([stamp(found[0], 1)], llm_view, compact=True)
    assert passage_costs(headers[0], bodies[0], anchors[0])[1] > ctx.budget.passage_chars

    reply = _registry(ScriptedSearch(found)).call(tool_call("search", query="Hafenverein Stände"), ctx)

    bodies_shown = _bodies(reply.text)
    assert sorted(bodies_shown) == [1, 2]
    assert all(len(MARKER.sub("", body)) <= ctx.budget.passage_chars for body in bodies_shown.values())
    assert all(body.count(WINDOW_JOIN) == 2 for body in bodies_shown.values())
    assert len(reply.text) <= ctx.budget.result_chars


def test_an_extension_without_room_is_named_as_grown() -> None:
    """A new window of a passage shown before joins its entry without room to show it, and is named as grown."""
    ctx = tool_context(reasoning=True)
    opening, closing = OPENING * 8, CLOSING * 12
    ctx.ledger.add([_window(opening)])
    reports = [_report(number) for number in range(1, 5)]

    reply = _registry(ScriptedSearch([*reports, _window(closing)])).call(tool_call("search", query="Hafenverein"), ctx)

    assert shown_numbers(reply.text) == [2, 3, 4, 5]
    assert reply.text.endswith("\n\nAlso matched, with new text not shown here: [1]; read(n) shows it.")
    assert "shown before" not in reply.text
    assert isinstance(reply, PassageReply)
    assert (reply.found, reply.numbers) == (5, (1, 2, 3, 4, 5))
    assert ctx.ledger.entries()[0].node.get_content() == f"{opening}{WINDOW_JOIN}{closing}"
    assert len(reply.text) <= ctx.budget.result_chars


def test_a_repeated_read_shows_a_passage_again_once_it_grew() -> None:
    """A read is answered from memory only while its passage is unchanged; once the passage grew, it is shown anew."""
    ctx = tool_context()
    registry = _registry(ScriptedSearch([_window(OPENING)], [_window(CLOSING)]))
    registry.call(tool_call("search", query="Hafenfest Sonntag"), ctx)

    first = registry.call(tool_call("read", n=1), ctx)
    unchanged = registry.call(tool_call("read", n=1), ctx)
    registry.call(tool_call("search", query="Band am Kai"), ctx)
    grown = registry.call(tool_call("read", n=1), ctx)
    after = registry.call(tool_call("read", n=1), ctx)

    assert OPENING in first.text
    assert CLOSING not in first.text
    assert unchanged.text == "Already called read with these arguments; see [1]."
    assert shown_numbers(grown.text) == [1]
    assert f"{OPENING}{WINDOW_JOIN}{CLOSING}" in grown.text
    assert after.text == "Already called read with these arguments; see [1]."


def test_a_read_is_shown_anew_once_its_passage_gained_a_match() -> None:
    """A search can add a match deep in a long passage without adding text; a read then keeps that part too."""
    ctx = tool_context()
    body = " ".join(f"Satz {line}: Der Hafenverein plant Stände und Musik am Kai." for line in range(1, 81))
    start = len(body) - 400
    held = scored_node("programme-9", body, filename="hafenfest_programm.pdf", page=3, **{MATCH_SPANS_KEY: [[60, 100]]})
    window = scored_node(
        "programme-9", body[start:], filename="hafenfest_programm.pdf", page=3, **{MATCH_SPANS_KEY: [[200, 240]]}
    )
    registry = _registry(ScriptedSearch([held], [window]))
    registry.call(tool_call("search", query="Hafenverein"), ctx)
    before = registry.call(tool_call("read", n=1), ctx)
    registry.call(tool_call("search", query="Musik am Kai"), ctx)

    after = registry.call(tool_call("read", n=1), ctx)

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
        ctx = tool_context()
        registry = _registry(ScriptedSearch([_window(OPENING)], [_window(CLOSING)]))
        registry.call(tool_call("search", query="Hafenfest Sonntag"), ctx)
        registry.call(tool_call("read", n=1), ctx)
        calls = [tool_call("search", query="Band am Kai"), tool_call("read", n=1)]
        replies = two_phase_round(registry, calls, ctx) if two_phase else [registry.call(call, ctx) for call in calls]
        return replies[1]

    sequential, phased = last_reply(two_phase=False), last_reply(two_phase=True)

    assert phased == sequential
    assert f"{OPENING}{WINDOW_JOIN}{CLOSING}" in phased.text


def test_a_read_is_news_and_its_repeat_is_not() -> None:
    """A read shows a passage's text, so it is informative; the same read answered from memory is not."""
    ctx = tool_context()
    registry = _registry(ScriptedSearch([_window(OPENING)]))
    registry.call(tool_call("search", query="Hafenfest Sonntag"), ctx)

    first = registry.call(tool_call("read", n=1), ctx)
    again = registry.call(tool_call("read", n=1), ctx)

    assert (first.informative, first.found) == (True, 0)
    assert again.text == "Already called read with these arguments; see [1]."
    assert (again.informative, again.found) == (False, 0)


def test_an_unchanged_remembered_read_is_answered_from_memory_at_finish() -> None:
    """A remembered read needs no I/O; memory answers it at finish, while its passage is unchanged."""
    ctx = tool_context()
    registry = _registry(ScriptedSearch([_window(OPENING)]))
    registry.call(tool_call("search", query="Hafenfest Sonntag"), ctx)
    registry.call(tool_call("read", n=1), ctx)

    prepared = registry.prepare(tool_call("read", n=1), ctx)

    assert (prepared.reply, prepared.fetch) == (None, None)
    assert registry.finish(prepared, None, ctx).text == "Already called read with these arguments; see [1]."


def test_a_read_finished_after_an_extending_search_in_its_round_shows_the_growth() -> None:
    """Prepared together, the same read before and after a search that grows its passage is answered twice."""
    ctx = tool_context()
    registry = _registry(ScriptedSearch([_window(OPENING)], [_window(CLOSING)]))
    registry.call(tool_call("search", query="Hafenfest Sonntag"), ctx)
    calls = [tool_call("read", n=1), tool_call("search", query="Band am Kai"), tool_call("read", n=1)]

    before, _, after = two_phase_round(registry, calls, ctx)

    assert CLOSING not in before.text
    assert f"{OPENING}{WINDOW_JOIN}{CLOSING}" in after.text


def test_a_long_header_is_compacted_so_it_costs_a_reply_little() -> None:
    """A passage's header is cut to the compact form a reply prices it at, so long names still leave room."""
    ctx = tool_context()
    found = [
        scored_node(f"long-{number}", _report(number).node.get_content(), filename=f"{number}_" + "hafen" * 200)
        for number in range(1, 6)
    ]

    reply = _registry(ScriptedSearch(found)).call(tool_call("search", query="Hafenverein"), ctx)

    headers = [block.partition("\n\n")[0] for block in re.split(r"\n\n(?=citation_index: )", NOTES.sub("", reply.text))]
    assert shown_numbers(reply.text) == [1, 2, 3, 4]
    assert all(
        len(header) <= len("citation_index: 1\n") + COMPACT_HEADER_CHARS + MARKER_RESERVE_CHARS for header in headers
    )
    assert len(reply.text) <= ctx.budget.result_chars


def test_a_post_header_reaches_the_model_whole_with_its_number_first() -> None:
    """Who posted a passage, and when, is evidence too: a header of ordinary length is never cut."""
    ctx = tool_context(view=ParentContextPostprocessor._emit_with_llm_exclusion)
    registry = _registry(ScriptedSearch([social_post(3)]))

    found = registry.call(tool_call("search", query="Harbour festival"), ctx)
    read = registry.call(tool_call("read", n=1), ctx)

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
    ctx = tool_context()
    name = "h" * (HEADER_CHARS - len("citation_index: 1\nfilename: ") + extra)
    reply = _registry(ScriptedSearch([scored_node("named", PROGRAMME, filename=name)])).call(
        tool_call("search", query="Hafenfest"), ctx
    )

    header = reply.text.partition("\n\n")[0]
    assert header.startswith("citation_index: 1\nfilename: hhh")
    assert (header == f"citation_index: 1\nfilename: {name}") is whole
    assert (MARKER.search(header) is None) is whole


def test_a_passage_fits_exactly_when_the_room_covers_its_reserve_and_floor() -> None:
    """Admission at its boundary: text room must remain after the reserves and cover every floor."""
    ctx = tool_context()
    passage = _plan(_report(1), 1, ctx)
    empty = _plan(scored_node("empty", "", filename="leer.pdf"), 2, ctx)

    assert _fits([passage], passage.reserve + passage.floor)
    assert not _fits([passage], passage.reserve + passage.floor - 1)
    assert not _fits([empty], empty.reserve)
    assert _fits([empty], empty.reserve + 1)


def test_windows_of_a_new_section_in_one_search_share_its_number() -> None:
    """Two windows of one section found together are one passage under one number."""
    ctx = tool_context()

    reply = _registry(ScriptedSearch([_window(OPENING), _weather(), _window(CLOSING)])).call(
        tool_call("search", query="Hafenfest Programm"), ctx
    )

    assert shown_numbers(reply.text) == [1, 2]
    assert f"{OPENING}{WINDOW_JOIN}{CLOSING}" in reply.text
    assert (reply.found, len(ctx.ledger)) == (2, 2)


def test_a_full_ledger_leaves_new_passages_out_and_says_so() -> None:
    """Once the evidence holds all it may, new passages stay unnumbered and the model is told to answer."""
    ctx = tool_context(cap=1)

    reply = _registry(ScriptedSearch([programme_passage(), _weather()])).call(
        tool_call("search", query="Hafenfest"), ctx
    )

    assert shown_numbers(reply.text) == [1]
    assert reply.text.endswith(
        "\n\n1 more matching passage left out: the evidence holds all it can, so answer from the passages shown."
    )
    assert (reply.found, len(ctx.ledger), ctx.ledger.number_of("weather")) == (1, 1, None)


def test_a_search_that_finds_nothing_new_says_so() -> None:
    """Nothing found, or only passages shown before, is answered as no new passages."""
    ctx = tool_context()
    registry = _registry(ScriptedSearch([], [programme_passage()], [programme_passage()]))

    nothing = registry.call(tool_call("search", query="Feuerwerk"), ctx)
    registry.call(tool_call("search", query="Hafenfest"), ctx)
    again = registry.call(tool_call("search", query="Stände Sonntag"), ctx)

    assert (nothing.text, nothing.found) == (NO_NEW_PASSAGES, 0)
    assert (again.text, again.found) == (f"{NO_NEW_PASSAGES}\nAlso matched, shown before: [1].", 0)


@pytest.mark.parametrize("number", [1, "1", "[1]", 1.0], ids=["int", "text", "bracketed", "float"])
def test_read_accepts_a_number_written_as_text(number: object) -> None:
    """Models write the number as they see it cited; each form reads the same passage.

    Args:
        number (object): The ``n`` the model sent.
    """
    ctx = tool_context()
    registry = _registry(ScriptedSearch([programme_passage()]))
    registry.call(tool_call("search", query="Hafenfest"), ctx)

    reply = registry.call(tool_call("read", n=number), ctx)

    assert shown_numbers(reply.text) == [1]
    assert PROGRAMME in reply.text


def test_an_error_reply_is_not_remembered() -> None:
    """A read that failed runs again once the passage exists; only answers are remembered."""
    ctx = tool_context()
    registry = _registry(ScriptedSearch([programme_passage()]))

    early = registry.call(tool_call("read", n=1), ctx)
    registry.call(tool_call("search", query="Hafenfest"), ctx)
    beyond = registry.call(tool_call("read", n=2), ctx)
    later = registry.call(tool_call("read", n=1), ctx)
    repeated = registry.call(tool_call("read", n=1), ctx)

    assert early.text == "Error: there is no passage [1]; no passage has been shown yet."
    assert beyond.text == "Error: there is no passage [2]; the only passage is [1]."
    assert shown_numbers(later.text) == [1]
    assert repeated.text == "Already called read with these arguments; see [1]."


def test_a_reply_says_whether_the_tool_answered_the_call() -> None:
    """Error replies are marked, so a turn's trail can leave out a call its tool did not answer.

    A search's passages, a read and an answer from memory are answers; an unoffered tool,
    arguments that are no JSON object, a missing argument and a passage that is not held are not.
    """
    ctx = tool_context()
    registry = _registry(ScriptedSearch([programme_passage()]))

    answers = [
        registry.call(tool_call("search", query="Hafenfest"), ctx),
        registry.call(tool_call("search", query="Hafenfest"), ctx),
        registry.call(tool_call("read", n=1), ctx),
    ]
    errors = [
        registry.call(ToolCall(id="call_0", name="keyframes", arguments="{}"), ctx),
        registry.call(ToolCall(id="call_0", name="read", arguments="nicht json"), ctx),
        registry.call(tool_call("search", query=""), ctx),
        registry.call(tool_call("read", n=9), ctx),
    ]

    assert [type(reply) for reply in answers] == [PassageReply, ToolReply, PassageReply]
    assert [reply.error for reply in answers] == [False, False, False]
    assert [reply.kind for reply in answers] == ["", "", ""]
    assert [reply.error for reply in errors] == [True, True, True, True]
    assert error_reply("search", "invented", kind="failed").error is True
    assert ToolReply(text="Keine neuen Passagen.", found=0, action="search", detail="").error is False


# --- News, trail fields and a tool's own repeat key ---------------------------------------------

#: The extra keys of the invented counting tool's step.
COUNT_TRAIL: dict[str, Any] = {"field": "text", "matches": 5, "unchecked": 2}


class _Counting:
    """A tool that answers with the passages it finds and a count of words, news even when no passage is admitted."""

    def __init__(
        self,
        *passages: NodeWithScore,
        plain: bool = False,
        trail_of: Callable[[int], Mapping[str, Any]] | None = None,
    ) -> None:
        """Describe the tool.

        Args:
            *passages (NodeWithScore): What every call finds.
            plain (bool): Whether the answer is a plain reply, which the registry never remembers.
            trail_of (Callable[[int], Mapping[str, Any]] | None): The trail of an answer, from how many answers
                the tool has given including that one; ``COUNT_TRAIL`` for every answer when ``None``.
        """
        self.name = "count"
        self.description = "Count the chunks that hold the words."
        self.parameters: dict[str, Any] = {
            "type": "object",
            "properties": {"words": {"type": "string"}},
            "required": ["words"],
        }
        self.runs: list[str] = []
        self.answers = 0
        self._passages = list(passages)
        self._plain = plain
        self._trail_of = trail_of

    def prepare(self, arguments: Mapping[str, Any], ctx: ToolContext) -> ToolReply | ToolWork:
        """Take any words; the count is the call's fetch and a reply carrying it the answer.

        Args:
            arguments (Mapping[str, Any]): ``words``.
            ctx (ToolContext): Unused.

        Returns:
            ToolReply | ToolWork: The work.
        """
        words = str(arguments.get("words", ""))
        detail = trail_detail(words)

        def _fetch() -> list[NodeWithScore]:
            """Run the count, noting the words it ran for.

            Returns:
                list[NodeWithScore]: The passages found.
            """
            self.runs.append(words)
            return list(self._passages)

        def _answer(fetched: Any, context: ToolContext) -> ToolReply:
            """Show what was found, as news that carries its trail.

            Args:
                fetched (Any): What the fetch returned.
                context (ToolContext): The turn's evidence.

            Returns:
                ToolReply: A plain reply when the tool is plain, else a passage reply.
            """
            shown = passage_reply(fetched, context, action=self.name, detail=detail)
            self.answers += 1
            trail = COUNT_TRAIL if self._trail_of is None else self._trail_of(self.answers)
            if self._plain:
                return ToolReply(
                    text=shown.text,
                    found=shown.found,
                    action=self.name,
                    detail=detail,
                    informative=True,
                    trail=trail,
                )
            return PassageReply(
                text=shown.text,
                found=shown.found,
                action=self.name,
                detail=detail,
                numbers=shown.numbers,
                informative=True,
                trail=trail,
            )

        return ToolWork(detail=detail, fetch=_fetch, answer=_answer)


class _Keyed(_Counting):
    """A counting tool that says itself which calls are one call."""

    def __init__(self, key: Callable[[Mapping[str, Any]], str | None]) -> None:
        """Bind the rule.

        Args:
            key (Callable[[Mapping[str, Any]], str | None]): Maps a call's arguments to its repeat key, or
                ``None`` to leave the call to the default key.
        """
        super().__init__()
        self._key = key

    def recall_key(self, arguments: Mapping[str, Any]) -> str | None:
        """Name the call by the rule.

        Args:
            arguments (Mapping[str, Any]): The call's arguments.

        Returns:
            str | None: The repeat key, or ``None``.
        """
        return self._key(arguments)


def _in_any_order(arguments: Mapping[str, Any]) -> str | None:
    """Key a call by its words in any order and case; a call with no words has no key of its own.

    Args:
        arguments (Mapping[str, Any]): The call's arguments.

    Returns:
        str | None: The folded words, sorted; ``None`` when there are none.
    """
    return " ".join(sorted(str(arguments.get("words", "")).casefold().split())) or None


def test_a_tool_says_which_calls_are_one_call() -> None:
    """Calls its ``recall_key`` maps to one string are one call whatever their arguments: only the first runs."""
    ctx = tool_context()
    tool = _Keyed(_in_any_order)
    registry = ToolRegistry([tool])
    registry.call(tool_call("count", words="Hans Müller"), ctx)

    again = registry.prepare(tool_call("count", words="  müller   HANS "), ctx)
    other = registry.call(tool_call("count", words="Hans Meier"), ctx)

    assert again.key == ("count", "hans müller")
    assert again.fetch is None
    assert registry.finish(again, None, ctx).text == "Already called count with these arguments; it showed no passages."
    assert (other.informative, tool.runs) == (True, ["Hans Müller", "Hans Meier"])


def test_two_calls_with_one_repeat_key_prepared_together_are_admitted_once() -> None:
    """Both pass prepare and fetch; the second is answered at finish from what the first remembered, as no news."""
    ctx = tool_context()
    registry = ToolRegistry([_Keyed(_in_any_order)])
    calls = [tool_call("count", words="Hans Müller"), tool_call("count", words="müller HANS")]

    first, second = two_phase_round(registry, calls, ctx)

    assert (first.informative, first.trail) == (True, COUNT_TRAIL)
    assert (second.text, second.found, second.informative, second.trail) == (
        "Already called count with these arguments; it showed no passages.",
        0,
        False,
        COUNT_TRAIL,
    )


@pytest.mark.parametrize("keyed", [False, True], ids=["no-method", "returns-none"])
def test_a_call_without_a_repeat_key_of_its_own_keeps_the_case_folded_arguments(keyed: bool) -> None:
    """A tool without ``recall_key``, or whose ``recall_key`` returns ``None``, is keyed as every tool was before.

    Args:
        keyed (bool): Whether the tool defines ``recall_key``, answering ``None``.
    """
    ctx = tool_context()
    tool = _Keyed(lambda arguments: None) if keyed else _Counting()
    registry = ToolRegistry([tool])
    registry.call(tool_call("count", words="Hans Müller"), ctx)

    spaced = registry.prepare(tool_call("count", words="  hans   MÜLLER "), ctx)
    reordered = registry.prepare(tool_call("count", words="müller hans"), ctx)

    assert spaced.key == ("count", '{"words": "hans müller"}')
    assert (spaced.fetch is None, reordered.fetch is None) == (True, False)


def test_a_call_a_tool_gives_no_key_to_is_keyed_by_its_arguments() -> None:
    """The tool decides per call: one it cannot key, such as a call with no words, gets the default key."""
    ctx = tool_context()
    registry = ToolRegistry([_Keyed(_in_any_order)])

    blank = registry.prepare(tool_call("count", words="   "), ctx)
    worded = registry.prepare(tool_call("count", words="Hans Müller"), ctx)

    assert blank.key == ("count", '{"words": ""}')
    assert worded.key == ("count", "hans müller")


class _Unkeyable(_Counting):
    """A counting tool whose repeat key raises, as one with a bug in its handling of arguments would."""

    def recall_key(self, arguments: Mapping[str, Any]) -> str | None:
        """Fail.

        Args:
            arguments (Mapping[str, Any]): The call's arguments, unused.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("invented key failure")


def test_a_repeat_key_that_raises_fails_its_call_alone(loguru_caplog: LogCaptureFixture) -> None:
    """A call its tool cannot key is a call that cannot be prepared: it gets the failure reply, the round goes on.

    Args:
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    ctx = tool_context()
    search = ScriptedSearch([programme_passage()])
    calls = RoundCalls(
        registry=ToolRegistry([SearchTool(search), _Unkeyable()]),
        result_chars=ctx.budget.result_chars,
        max_calls_per_round=4,
        char_token_ratio=3.5,
        search_slots=threading.BoundedSemaphore(2),
    )

    prepared = calls.prepare([tool_call("count", words="Hafenfest"), tool_call("search", query="Hafenfest")], ctx)
    answered = calls.answer(prepared, ctx, room=10_000, time_left=lambda: 60.0)

    assert (prepared.failed, answered.failed) == (1, 1)
    assert answered.replies[0] == error_reply("count", FAILED_REPLY, kind="failed")
    assert shown_numbers(answered.replies[1].text) == [1]
    assert search.queries == ["Hafenfest"]
    assert loguru_caplog.messages == ["An agent tool call could not be prepared: RuntimeError"]


def test_a_call_answered_from_memory_keeps_its_trail_and_is_no_news() -> None:
    """A recalled step shows the fields it showed before, but it admits nothing and tells the model nothing new."""
    ctx = tool_context()
    registry = ToolRegistry([_Counting(programme_passage())])
    first = registry.call(tool_call("count", words="Hafenfest"), ctx)

    again = registry.call(tool_call("count", words="HAFENFEST"), ctx)

    assert (first.found, first.informative, first.trail) == (1, True, COUNT_TRAIL)
    assert again.text == "Already called count with these arguments; see [1]."
    assert (again.found, again.informative, again.trail) == (0, False, COUNT_TRAIL)


def _numbered_trail(answer: int) -> dict[str, Any]:
    """Give an answer a trail that tells it from every other answer of the tool.

    Args:
        answer (int): How many answers the tool has given, this one included.

    Returns:
        dict[str, Any]: The count trail with the answer's number.
    """
    return {**COUNT_TRAIL, "answer": answer}


def test_a_call_answered_from_memory_has_the_trail_of_the_first_answer_alone() -> None:
    """A second answer to the same call, which would carry another trail, is never given or remembered."""
    ctx = tool_context()
    tool = _Counting(programme_passage(), trail_of=_numbered_trail)
    registry = ToolRegistry([tool])
    calls = [tool_call("count", words="Hafenfest"), tool_call("count", words="hafenfest")]

    first, duplicate = two_phase_round(registry, calls, ctx)
    later = registry.call(tool_call("count", words="HAFENFEST"), ctx)

    assert first.trail == _numbered_trail(1)
    assert (duplicate.trail, later.trail) == (first.trail, first.trail)
    # Both calls of the round fetched; the registry answered one and recalled the other.
    assert (tool.runs, tool.answers) == (["Hafenfest", "hafenfest"], 1)


def test_the_trail_a_call_is_remembered_by_is_a_copy_the_tool_cannot_rewrite() -> None:
    """A tool that reuses and mutates one dict for its trails cannot rewrite the steps of calls answered before."""
    ctx = tool_context()
    shared = dict(COUNT_TRAIL)
    registry = ToolRegistry([_Counting(programme_passage(), trail_of=lambda answer: shared)])
    registry.call(tool_call("count", words="Hafenfest"), ctx)
    shared["matches"] = 99

    again = registry.call(tool_call("count", words="hafenfest"), ctx)

    assert again.trail == COUNT_TRAIL
    assert shared["matches"] == 99


def test_a_search_answered_from_memory_has_no_trail_and_is_no_news() -> None:
    """A tool that sets no trail leaves none, so a repeated search reads as it always did."""
    ctx = tool_context()
    registry = _registry(ScriptedSearch([programme_passage()]))
    registry.call(tool_call("search", query="Hafenfest"), ctx)

    again = registry.call(tool_call("search", query="hafenfest"), ctx)

    assert (again.found, again.informative, again.trail) == (0, False, None)


def test_a_plain_reply_is_not_remembered_whatever_it_carries() -> None:
    """Only a passage reply is remembered: a count answered plain, as a failed one is, runs again when called again."""
    ctx = tool_context()
    tool = _Counting(plain=True)
    registry = ToolRegistry([tool])

    first = registry.call(tool_call("count", words="Hafenfest"), ctx)
    second = registry.call(tool_call("count", words="Hafenfest"), ctx)

    assert (type(first), first.informative, first.trail) == (ToolReply, True, COUNT_TRAIL)
    assert (second.text, tool.runs) == (first.text, ["Hafenfest", "Hafenfest"])


def test_a_reply_is_by_default_no_news_and_carries_no_trail() -> None:
    """The fields after ``error`` default to nothing, in a plain reply and a passage reply alike."""
    plain = ToolReply(text="Keine neuen Passagen.", found=0, action="search", detail="")
    shown = PassageReply(text="[1]", found=1, action="search", detail="Hafenfest", numbers=(1,))

    defaults = [(reply.informative, reply.trail, reply.kind, reply.keep) for reply in (plain, shown)]
    assert defaults == [(False, None, "", "")] * 2
    # What is left of a reply once cleared, ``keep``, comes last: it is the last thing to happen to a reply.
    base = ["text", "found", "action", "detail", "error", "informative", "trail", "kind", "keep"]
    assert [field.name for field in fields(ToolReply)] == base
    assert [field.name for field in fields(PassageReply)] == [*base, "numbers", "depends_on"]


@pytest.mark.parametrize(
    ("numbers", "keep", "cleared"),
    [
        ((1,), "", "Cleared to make room: this reply showed [1]; read(n) shows a passage again."),
        ((1, 3, 4), "", "Cleared to make room: this reply showed [1], [3], [4]; read(n) shows a passage again."),
        ((), "", "Cleared to make room: this reply showed no passages."),
        (
            (2,),
            "3 Treffer.",
            "3 Treffer.\n\nCleared to make room: this reply showed [2]; read(n) shows a passage again.",
        ),
        ((), "0 Treffer.", "0 Treffer.\n\nCleared to make room: this reply showed no passages."),
    ],
    ids=["one", "several", "none", "kept-text-first", "kept-text-and-none"],
)
def test_a_cleared_reply_keeps_its_kept_text_then_names_its_passages(
    numbers: tuple[int, ...], keep: str, cleared: str
) -> None:
    """A reply cut down to make room names the passages it showed, for read(n), after what it keeps of its text.

    Args:
        numbers (tuple[int, ...]): The passages the reply showed or named.
        keep (str): What it keeps of its text; empty for none.
        cleared (str): What it says once cleared.
    """
    reply = PassageReply(
        text=OPENING * 20, found=len(numbers), action="search", detail="Hafenfest", numbers=numbers, keep=keep
    )

    assert cleared_text(reply) == cleared


def test_arguments_nested_too_deeply_get_an_error_reply() -> None:
    """Arguments that parse as JSON but nest too deeply to compare get an argument error, never a crash."""
    nested = '{"n": ' + "[" * 600 + "]" * 600 + "}"

    reply = _registry(ScriptedSearch()).call(ToolCall(id="call_0", name="read", arguments=nested), tool_context())

    assert reply.error
    assert reply.text.startswith("Error: the arguments for read")


def _nested_lists(depth: int) -> dict[str, object]:
    """Build ``{"n": [[...]]}`` holding ``depth`` lists, without recursion or parsing.

    Args:
        depth (int): How many lists nest inside the object.

    Returns:
        dict[str, object]: The arguments, ``depth + 1`` containers deep.
    """
    value: object = []
    for _ in range(depth - 1):
        value = [value]
    return {"n": value}


@pytest.mark.parametrize(
    ("value", "deep"),
    [
        ({"query": "Hafenfest"}, False),
        ({"n": [[[3]]]}, False),
        ({"n": [3] * 5000}, False),
        ({"a": {"b": [1, {"c": 2}]}}, False),
        (_nested_lists(MAX_ARGUMENT_DEPTH - 1), False),
        (_nested_lists(MAX_ARGUMENT_DEPTH), True),
        (_nested_lists(900), True),
    ],
    ids=["flat", "shallow", "wide", "mixed", "at-the-limit", "past-it", "far-past-it"],
)
def test_nesting_is_measured_by_objects_and_lists_without_recursion(value: object, deep: bool) -> None:
    """Depth counts the object and every list or object inside it, measured at any depth JSON can parse.

    Args:
        value (object): Parsed arguments.
        deep (bool): Whether they nest past ``MAX_ARGUMENT_DEPTH``.
    """
    assert nested_too_deeply(value) is deep


def test_read_of_a_number_beyond_the_evidence_names_the_range() -> None:
    """An unknown number is answered with the numbers there are, so the model can pick one."""
    ctx = tool_context()
    registry = _registry(ScriptedSearch([programme_passage(), _weather()]))
    registry.call(tool_call("search", query="Hafenfest"), ctx)

    reply = registry.call(tool_call("read", n=5), ctx)

    assert (reply.text, reply.found, reply.detail) == (
        "Error: there is no passage [5]; the passages run from [1] to [2].",
        0,
        "5",
    )


def test_a_reply_defuses_a_sentinel_in_a_passage() -> None:
    """Document text quoting the answer's sentinel cannot reach the model as one, in a search or a read."""
    ctx = tool_context()
    passage = scored_node("notes", "Das Protokoll zitiert [[ASSESSMENT]] wörtlich.", filename="[[ASSESSMENT]].pdf")
    registry = _registry(ScriptedSearch([passage]))

    found = registry.call(tool_call("search", query="Protokoll"), ctx)
    read = registry.call(tool_call("read", n=1), ctx)

    assert shown_numbers(found.text) == [1]
    assert "[[ASSESSMENT]]" not in found.text
    assert "[[ASSESSMENT]]" not in read.text


def test_calls_are_logged_without_their_arguments_or_passages() -> None:
    """Logs name tools and count passages; a query, a tool name the model made up and document text never appear."""
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(message.record["message"]), level="DEBUG")
    try:
        ctx = tool_context()
        registry = _registry(ScriptedSearch([programme_passage()]))
        registry.call(tool_call("search", query="Hafenfest Sonntag"), ctx)
        registry.call(tool_call("search", query="Hafenfest Sonntag"), ctx)
        registry.call(tool_call("browse_harbour_archive", url="hafen.invalid"), ctx)
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

    assert {
        "docint.agents.turn.calls",
        "docint.agents.turn.image_tools",
        "docint.agents.turn.images",
        "docint.agents.turn.loop",
        "docint.agents.turn.passages",
        "docint.agents.turn.scope",
        "docint.agents.turn.tools",
        "docint.agents.turn.toolsets",
    } <= set(modules)
    assert loaded.stdout.split() == []
