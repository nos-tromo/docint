"""The research loop. Every query and passage is invented."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextvars import ContextVar
from types import SimpleNamespace
from typing import Any, cast

import httpx
import openai
import pytest
from _pytest.logging import LogCaptureFixture
from assessment_fakes import bad_request, llm_view, scored_node, search_call, tool_llm
from llama_index.core.base.llms.types import MessageRole, ToolCallBlock
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.schema import MetadataMode, NodeWithScore, TextNode

from docint.agents.assessment.ledger import EvidenceLedger, Room
from docint.agents.assessment.research import SEARCH_THREAD_NAME, ResearchLoop
from docint.agents.assessment.runner import ResearchOutcome, SearchFn
from docint.agents.types import AgentProgress
from docint.core.rag import ParentContextPostprocessor

QUESTION = "zzsecretzz What could be improved in this study?"


def _ticking(*ticks: float) -> Callable[[], float]:
    """Build a clock that reads ``ticks`` in turn, then keeps repeating the last one.

    Args:
        *ticks (float): The readings, in order.

    Returns:
        Callable[[], float]: The clock.
    """
    pending = list(ticks)

    def _now() -> float:
        """Read the next tick, or the last one again.

        Returns:
            float: The reading.
        """
        return pending.pop(0) if len(pending) > 1 else pending[0]

    return _now


def _search(results: dict[str, list[NodeWithScore]]) -> Any:
    """Build a search function over canned results.

    Args:
        results (dict[str, list[NodeWithScore]]): Passages per query.

    Returns:
        Any: The search function.
    """

    def _run(query: str) -> list[NodeWithScore]:
        """Return the canned passages for ``query``.

        Args:
            query (str): The search query.

        Returns:
            list[NodeWithScore]: The passages, or none for an unknown query.
        """
        return results.get(query, [])

    return _run


def _loop(llm: Any, search: Any, **overrides: Any) -> ResearchLoop:
    """Build a research loop with generous defaults.

    Args:
        llm (Any): The function-calling model double.
        search (Any): The search function.
        **overrides (Any): Constructor overrides.

    Returns:
        ResearchLoop: The loop.
    """
    kwargs: dict[str, Any] = {
        "max_steps": 6,
        "ctx_window": 32768,
        "char_token_ratio": 3.5,
        "prompt": "{question}{prior_turn}{digest}",
    }
    kwargs.update(overrides)
    return ResearchLoop(llm=cast(Any, llm), search=search, **kwargs)


def _drain(items: Iterator[AgentProgress | ResearchOutcome]) -> tuple[list[AgentProgress], ResearchOutcome]:
    """Split a research run into its progress and its outcome.

    Args:
        items (Iterator[AgentProgress | ResearchOutcome]): The run.

    Returns:
        tuple[list[AgentProgress], ResearchOutcome]: Progress steps and the outcome.
    """
    collected = list(items)
    outcome = collected[-1]
    assert isinstance(outcome, ResearchOutcome)
    progress = [item for item in collected[:-1] if isinstance(item, AgentProgress)]
    assert len(progress) == len(collected) - 1
    return progress, outcome


def test_searches_extend_the_ledger_until_finish() -> None:
    """Each search adds its new passages; ``finish`` ends research cleanly."""
    method = [scored_node("m", "Method section.")]
    limits = [scored_node("l", "Limitations section.")]
    llm = tool_llm([search_call("method"), search_call("limitations")])
    ledger = EvidenceLedger(cap=10)

    progress, outcome = _drain(
        _loop(llm, _search({"method": method, "limitations": limits})).run(
            question=QUESTION, prior_turn=None, ledger=ledger
        )
    )

    assert [(step.action, step.detail, step.found) for step in progress] == [
        ("search", "method", 1),
        ("search", "limitations", 1),
    ]
    assert (outcome.steps, outcome.reason) == (3, None)
    assert len(ledger) == 2


def test_a_new_window_of_a_held_section_is_found_digested_and_resets_the_idle_count() -> None:
    """A later window of a held section is evidence: it is not a repeat, and it does not idle research."""
    opening = [scored_node("sec", "The study opens with a pilot.", filename="paper.pdf")]
    closing = [scored_node("sec", "It closes with a limitations note.", filename="paper.pdf")]
    # Four searches, then the scripted ``finish``: a repeat of the first window would have idled out at the third.
    llm = tool_llm(
        [search_call("opening"), search_call("nothing"), search_call("closing"), search_call("more nothing")]
    )
    ledger = EvidenceLedger(cap=10)

    progress, outcome = _drain(
        _loop(llm, _search({"opening": opening, "closing": closing})).run(
            question=QUESTION, prior_turn=None, ledger=ledger
        )
    )

    assert [(step.detail, step.found) for step in progress] == [
        ("opening", 1),
        ("nothing", 0),
        ("closing", 1),
        ("more nothing", 0),
    ]
    assert (outcome.steps, outcome.reason) == (5, None)
    reply = llm.chat_with_tools.call_args_list[3].kwargs["chat_history"][6]
    assert reply.role == MessageRole.TOOL
    assert "[1] (paper.pdf) It closes with a limitations note." in str(reply.content)
    assert len(ledger) == 1


def test_two_windows_of_one_oversize_section_reach_the_answer_as_one_entry() -> None:
    """The reproduction: the real postprocessor windows a section twice under one id, and neither window may be lost."""
    first_fact = "The first survey wave lost two respondents."
    last_fact = "The final wave was rerun with a larger sample."
    section_text = (
        "Opening filler. " * 300 + first_fact + " Middle filler. " * 300 + last_fact + " Closing filler. " * 300
    )
    section = TextNode(id_="section-1", text=section_text, metadata={"filename": "survey.md", "page": 3})
    hits = {
        query: NodeWithScore(
            node=TextNode(
                id_=f"child-{query}",
                text=fact,
                metadata={"hier.parent_id": "section-1", "docint_hier_type": "fine", "filename": "survey.md"},
            ),
            score=0.8,
        )
        for query, fact in (("first wave", first_fact), ("final wave", last_fact))
    }
    postprocessor = ParentContextPostprocessor(
        docstore=SimpleNamespace(
            get_node=lambda node_id, raise_error=False: section if node_id == "section-1" else None
        ),
        usable_tokens=300,
        per_hit_floor=100,
        char_token_ratio=3.5,
        budget_enforced=True,
    )
    llm = tool_llm([search_call("first wave"), search_call("final wave")])
    ledger = EvidenceLedger(cap=10)

    progress, _ = _drain(
        _loop(llm, lambda query: postprocessor._postprocess_nodes([hits[query]])).run(
            question=QUESTION, prior_turn=None, ledger=ledger
        )
    )

    assert [step.found for step in progress] == [1, 1]
    numbered = ledger.numbered()
    assert len(numbered) == 1
    merged = numbered[0].node.get_content(metadata_mode=MetadataMode.NONE)
    assert first_fact in merged
    assert last_fact in merged
    # Still the two windows, not the section.
    assert len(merged) < len(section_text) // 2
    assert numbered[0].node.metadata["window_chars"] == len(merged)


def test_each_call_requires_a_tool_and_forbids_parallel_calls() -> None:
    """The flags that let vLLM serve research without auto tool choice."""
    llm = tool_llm([])

    _drain(_loop(llm, _search({})).run(question=QUESTION, prior_turn=None, ledger=EvidenceLedger(cap=5)))

    kwargs = llm.chat_with_tools.call_args.kwargs
    assert kwargs["tool_choice"] == "required"
    assert kwargs["allow_parallel_tool_calls"] is False
    assert kwargs["strict"] is False
    assert kwargs["max_tokens"] == 512
    assert kwargs["user_msg"] is None
    assert [tool.metadata.name for tool in llm.chat_with_tools.call_args.args[0]] == ["search", "finish"]


def test_the_transcript_carries_the_call_and_its_result() -> None:
    """The second call sees the first call and a tool reply keyed by the same id."""
    llm = tool_llm([search_call("method", call_id="")])

    _drain(
        _loop(llm, _search({"method": [scored_node("m", "Method section.", filename="study.pdf")]})).run(
            question=QUESTION, prior_turn=None, ledger=EvidenceLedger(cap=5)
        )
    )

    history = llm.chat_with_tools.call_args_list[1].kwargs["chat_history"]
    call_block = history[1].blocks[0]
    assert isinstance(call_block, ToolCallBlock)
    assert call_block.tool_call_id == "call_1"
    assert history[2].role == MessageRole.TOOL
    assert history[2].additional_kwargs["tool_call_id"] == "call_1"
    assert "[1] (study.pdf) Method section." in str(history[2].content)


def test_a_text_reply_ends_research_without_leaking_into_the_answer(loguru_caplog: LogCaptureFixture) -> None:
    """Prose instead of a first tool call means the endpoint treated ``required`` as optional.

    Args:
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    llm = tool_llm(["Here is my assessment already."])

    progress, outcome = _drain(
        _loop(llm, _search({})).run(question=QUESTION, prior_turn=None, ledger=EvidenceLedger(cap=5))
    )

    assert progress == []
    assert (outcome.steps, outcome.reason) == (0, "tools_unavailable")
    assert loguru_caplog.messages == [
        "Assessment research stopped (tools_unavailable): the model replied without a tool call"
    ]


def test_a_text_reply_after_a_search_ends_research_normally() -> None:
    """Once the model has searched, answering in prose is a choice to stop, not missing tool support."""
    llm = tool_llm([search_call("method"), "Here is my assessment already."])

    _, outcome = _drain(
        _loop(llm, _search({"method": [scored_node("m", "M.")]})).run(
            question=QUESTION, prior_turn=None, ledger=EvidenceLedger(cap=5)
        )
    )

    assert (outcome.steps, outcome.reason) == (1, None)


def test_the_step_cap_bounds_research() -> None:
    """``ASSESSMENT_MAX_STEPS`` is a hard bound."""
    llm = tool_llm([search_call(f"q{i}") for i in range(10)])
    results = {f"q{i}": [scored_node(f"n{i}", f"Passage {i}.")] for i in range(10)}

    _, outcome = _drain(
        _loop(llm, _search(results), max_steps=2).run(question=QUESTION, prior_turn=None, ledger=EvidenceLedger(cap=20))
    )

    assert outcome.steps == 2
    assert llm.chat_with_tools.call_count == 2


def test_a_repeated_query_or_two_empty_searches_end_research() -> None:
    """Research that has stopped finding anything stops spending calls."""
    repeat = tool_llm([search_call("method"), search_call("METHOD")])
    _, outcome = _drain(
        _loop(repeat, _search({"method": [scored_node("m", "M.")]})).run(
            question=QUESTION, prior_turn=None, ledger=EvidenceLedger(cap=5)
        )
    )
    assert outcome.steps == 2

    idle = tool_llm([search_call("a"), search_call("b"), search_call("c")])
    _, outcome = _drain(_loop(idle, _search({})).run(question=QUESTION, prior_turn=None, ledger=EvidenceLedger(cap=5)))
    assert outcome.steps == 2


def test_an_invalid_call_is_answered_and_research_continues() -> None:
    """A malformed call costs a step, gets an error reply, and the loop goes on."""
    llm = tool_llm([ToolSelection(tool_id="c1", tool_name="search", tool_kwargs={}), search_call("method")])

    progress, outcome = _drain(
        _loop(llm, _search({"method": [scored_node("m", "M.")]})).run(
            question=QUESTION, prior_turn=None, ledger=EvidenceLedger(cap=5)
        )
    )

    assert [step.detail for step in progress] == ["method"]
    assert outcome.steps == 3
    error_reply = llm.chat_with_tools.call_args_list[1].kwargs["chat_history"][2]
    assert "non-empty query" in str(error_reply.content)


def test_a_non_dict_argument_payload_is_an_invalid_call_not_a_crash() -> None:
    """Arguments that parse to a list or a string get the error reply; research continues."""
    llm = tool_llm([search_call("ignored"), search_call("method")])
    malformed = ToolSelection.model_construct(tool_id="c1", tool_name="search", tool_kwargs=cast(Any, ["method"]))
    parsed = llm.get_tool_calls_from_response.side_effect
    first_reply = iter([[malformed]])

    def _calls(response: Any, **kwargs: Any) -> list[ToolSelection]:
        """Return the malformed call once, then parse responses as the double does.

        Args:
            response (Any): The model response.
            **kwargs (Any): The parser's other arguments.

        Returns:
            list[ToolSelection]: The calls.
        """
        return next(first_reply, None) or parsed(response, **kwargs)

    llm.get_tool_calls_from_response.side_effect = _calls

    progress, outcome = _drain(
        _loop(llm, _search({"method": [scored_node("m", "M.")]})).run(
            question=QUESTION, prior_turn=None, ledger=EvidenceLedger(cap=5)
        )
    )

    assert [step.detail for step in progress] == ["method"]
    assert (outcome.steps, outcome.reason) == (3, None)
    history = llm.chat_with_tools.call_args_list[1].kwargs["chat_history"]
    recorded = history[1].blocks[0]
    assert isinstance(recorded, ToolCallBlock)
    assert recorded.tool_kwargs == {}
    assert "non-empty query" in str(history[2].content)


def test_a_rejected_first_tool_call_reports_tools_unavailable(loguru_caplog: LogCaptureFixture) -> None:
    """An endpoint without tool support degrades the turn; it never fails it.

    Args:
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    llm = tool_llm([bad_request(f"model does not support tools: {QUESTION}")])

    _, outcome = _drain(_loop(llm, _search({})).run(question=QUESTION, prior_turn=None, ledger=EvidenceLedger(cap=5)))

    assert outcome.reason == "tools_unavailable"
    assert "zzsecretzz" not in loguru_caplog.text


@pytest.mark.parametrize(
    ("error", "first_call", "later_call"),
    [
        (
            bad_request("This model's maximum context length is 8192 tokens; your request has 9001 input tokens."),
            "context_budget",
            "context_budget",
        ),
        (
            bad_request("The decoder prompt (length 9001) is longer than the maximum model length of 8192"),
            "context_budget",
            "context_budget",
        ),
        (bad_request("request exceeds the available context size"), "context_budget", "context_budget"),
        (bad_request("Input is too long"), "context_budget", "context_budget"),
        (
            bad_request("auto tool choice requires --enable-auto-tool-choice"),
            "tools_unavailable",
            "research_failed",
        ),
        (RuntimeError("boom"), "research_failed", "research_failed"),
    ],
)
def test_a_failed_call_is_named_by_its_message_and_its_round(
    error: Exception, first_call: str, later_call: str
) -> None:
    """A context-length error is a budget stop on any round; a rejected first call means no tool support.

    Args:
        error (Exception): What the endpoint raises.
        first_call (str): The reason when the first call fails.
        later_call (str): The reason when a call after a search fails.
    """
    first = tool_llm([error])
    _, outcome = _drain(_loop(first, _search({})).run(question=QUESTION, prior_turn=None, ledger=EvidenceLedger(cap=5)))
    assert outcome.reason == first_call

    later = tool_llm([search_call("method"), error])
    _, outcome = _drain(
        _loop(later, _search({"method": [scored_node("m", "M.")]})).run(
            question=QUESTION, prior_turn=None, ledger=EvidenceLedger(cap=5)
        )
    )
    assert outcome.reason == later_call


def test_a_later_failure_reports_research_failed() -> None:
    """Once research has started, a failure keeps what was found."""
    llm = tool_llm([search_call("method"), RuntimeError("boom")])
    ledger = EvidenceLedger(cap=5)

    _, outcome = _drain(
        _loop(llm, _search({"method": [scored_node("m", "M.")]})).run(question=QUESTION, prior_turn=None, ledger=ledger)
    )

    assert outcome.reason == "research_failed"
    assert len(ledger) == 1


def test_the_wall_clock_bounds_research() -> None:
    """A slow endpoint cannot hold the turn open indefinitely, and no search starts once the budget is spent."""
    llm = tool_llm([search_call("method"), search_call("results")])
    searched: list[str] = []

    def _search_recorded(query: str) -> list[NodeWithScore]:
        """Record the query and find one passage.

        Args:
            query (str): The search query.

        Returns:
            list[NodeWithScore]: One invented passage.
        """
        searched.append(query)
        return [scored_node("m", "M.")]

    _, outcome = _drain(
        _loop(llm, _search_recorded, clock=_ticking(0.0, 0.0, 0.0, 200.0), wall_clock_s=120.0).run(
            question=QUESTION, prior_turn=None, ledger=EvidenceLedger(cap=5)
        )
    )

    assert outcome.reason == "time_budget"
    assert searched == []


def test_each_research_call_is_bounded_by_the_time_remaining() -> None:
    """One hung call cannot outlast the research budget; a floor of a second keeps the last call possible."""
    # Start; then per round: the budget check, the call's own timeout, the time left for its search, the check after.
    clock = _ticking(0.0, 0.0, 0.0, 100.0, 100.0, 100.0, 119.5, 200.0)
    llm = tool_llm([search_call("method"), search_call("results")])

    _drain(
        _loop(llm, _search({"method": [scored_node("m", "M.")]}), clock=clock, wall_clock_s=120.0).run(
            question=QUESTION, prior_turn=None, ledger=EvidenceLedger(cap=5)
        )
    )

    assert [call.kwargs["timeout"] for call in llm.chat_with_tools.call_args_list] == [120.0, 1.0]


@pytest.fixture
def held_search() -> Iterator[tuple[threading.Event, SearchFn]]:
    """A search that blocks until released; teardown releases it and joins its thread, so none outlives the test.

    Yields:
        tuple[threading.Event, SearchFn]: The release and the search.
    """
    release = threading.Event()

    def _search(query: str) -> list[NodeWithScore]:
        """Wait for the release, then find one invented passage.

        Args:
            query (str): The search query, unused.

        Returns:
            list[NodeWithScore]: The late passage.
        """
        release.wait(timeout=30)
        return [scored_node("late", "A passage that arrives too late.")]

    yield release, _search
    release.set()
    for thread in threading.enumerate():
        if thread.name == SEARCH_THREAD_NAME:
            thread.join(timeout=5)


def test_a_search_that_outlasts_the_budget_ends_research_on_time(
    held_search: tuple[threading.Event, SearchFn], loguru_caplog: LogCaptureFixture
) -> None:
    """A hung search is abandoned when the budget runs out; its result never reaches the ledger.

    Args:
        held_search (tuple[threading.Event, SearchFn]): The blocking search.
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    _, search = held_search
    ledger = EvidenceLedger(cap=5)
    ledger.add([scored_node("seed", "The seed passage.")])
    started = time.monotonic()

    progress, outcome = _drain(
        _loop(tool_llm([search_call("zzsecretzz method")]), search, wall_clock_s=0.3).run(
            question=QUESTION, prior_turn=None, ledger=ledger
        )
    )

    assert time.monotonic() - started < 3.0
    assert (outcome.reason, progress, outcome.actions) == ("time_budget", [], [])
    assert [scored.node.node_id for scored in ledger.entries()] == ["seed"]
    assert loguru_caplog.messages == [
        "Assessment research stopped (time_budget): a search outlasted the research budget"
    ]


_SCOPE: ContextVar[str | None] = ContextVar("invented_collection_scope", default=None)


def test_a_search_runs_in_the_callers_context() -> None:
    """Retrieval reads the request's collection and reasoning scope from context variables, so a search sees them."""
    seen: list[str | None] = []

    def _search(query: str) -> list[NodeWithScore]:
        """Record the scope the search runs under.

        Args:
            query (str): The search query, unused.

        Returns:
            list[NodeWithScore]: Nothing.
        """
        seen.append(_SCOPE.get())
        return []

    token = _SCOPE.set("invented-collection")
    try:
        _drain(
            _loop(tool_llm([search_call("method")]), _search).run(
                question=QUESTION, prior_turn=None, ledger=EvidenceLedger(cap=5)
            )
        )
    finally:
        _SCOPE.reset(token)

    assert seen == ["invented-collection"]


def test_a_search_raising_its_own_timeout_is_a_failed_search_not_a_spent_budget() -> None:
    """A retrieval dependency's ``TimeoutError`` is the search failing; only the clock spends the budget."""

    def _search(query: str) -> list[NodeWithScore]:
        """Fail as a timed-out storage call would.

        Args:
            query (str): The search query, unused.

        Raises:
            TimeoutError: Always.
        """
        raise TimeoutError("invented storage timeout")

    _, outcome = _drain(
        _loop(tool_llm([search_call("method")]), _search).run(
            question=QUESTION, prior_turn=None, ledger=EvidenceLedger(cap=5)
        )
    )

    assert outcome.reason == "research_failed"


def test_a_model_call_that_times_out_with_the_budget_spent_reports_time_budget() -> None:
    """The call's timeout is the time left, so its timeout error once the clock is spent is the budget running out."""
    timeout = openai.APITimeoutError(request=httpx.Request("POST", "http://inference.invalid/v1/chat/completions"))

    _, outcome = _drain(
        _loop(tool_llm([timeout]), _search({}), clock=_ticking(0.0, 0.0, 0.0, 200.0), wall_clock_s=120.0).run(
            question=QUESTION, prior_turn=None, ledger=EvidenceLedger(cap=5)
        )
    )

    assert outcome.reason == "time_budget"


def test_a_slow_final_search_reports_time_budget() -> None:
    """The last search a step cap allows is still timed: research that overran says so even though it found evidence."""
    now = [0.0]

    def _slow(query: str) -> list[NodeWithScore]:
        """Take five minutes on the injected clock, then find one passage.

        Args:
            query (str): The search query, unused.

        Returns:
            list[NodeWithScore]: One invented passage.
        """
        now[0] += 300.0
        return [scored_node("m", "Method section.")]

    ledger = EvidenceLedger(cap=5)

    progress, outcome = _drain(
        _loop(tool_llm([search_call("method")]), _slow, max_steps=1, clock=lambda: now[0], wall_clock_s=120.0).run(
            question=QUESTION, prior_turn=None, ledger=ledger
        )
    )

    assert outcome.reason == "time_budget"
    assert [step.found for step in progress] == [1]
    assert len(ledger) == 1


def test_a_window_too_small_for_one_step_never_calls_the_model() -> None:
    """The budget check runs before each call, the first one included."""
    llm = tool_llm([search_call("method")])

    _, outcome = _drain(
        _loop(llm, _search({}), ctx_window=1024).run(question=QUESTION, prior_turn=None, ledger=EvidenceLedger(cap=5))
    )

    assert outcome.reason == "context_budget"
    llm.chat_with_tools.assert_not_called()


def test_a_full_ledger_stops_research() -> None:
    """There is no room left for more evidence."""
    ledger = EvidenceLedger(cap=1)
    ledger.add([scored_node("a", "A.")])
    llm = tool_llm([search_call("method")])

    _, outcome = _drain(_loop(llm, _search({})).run(question=QUESTION, prior_turn=None, ledger=ledger))

    assert outcome.steps == 0
    llm.chat_with_tools.assert_not_called()


def _closing_ledger(*, cap: int = 50) -> EvidenceLedger:
    """Hold one seed passage in a room that two passages of :func:`_wide` close.

    Each passage reserves 79 characters for its header, joins and marker and keeps 300: 1,480
    characters leave 343 once two follow the seed, short of the 354 one more bare passage needs.

    Args:
        cap (int): The ledger cap.

    Returns:
        EvidenceLedger: The ledger, restricted.
    """
    ledger = EvidenceLedger(cap=cap)
    ledger.add([scored_node("seed", "Seed passage. " * 30, page=0)])
    ledger.restrict(Room(ledger.numbered(), llm_view=llm_view, budget_tokens=1480, char_token_ratio=1.0))
    return ledger


def _wide(count: int) -> list[NodeWithScore]:
    """Return invented passages too long to show whole.

    Args:
        count (int): How many.

    Returns:
        list[NodeWithScore]: The passages.
    """
    return [scored_node(f"m{number}", "w" * 1000, page=number) for number in range(1, count + 1)]


def test_research_that_runs_out_of_room_before_its_step_cap_says_so() -> None:
    """The room closing ends research early, and the turn reports it as ``context_budget``."""
    ledger = _closing_ledger()
    llm = tool_llm([search_call("method"), search_call("results")])

    progress, outcome = _drain(
        _loop(llm, _search({"method": _wide(3), "results": _wide(6)[3:]})).run(
            question=QUESTION, prior_turn=None, ledger=ledger
        )
    )

    assert [(step.detail, step.found) for step in progress] == [("method", 2)]
    assert ledger.out_of_room is True
    assert (outcome.steps, outcome.reason) == (1, "context_budget")
    assert llm.chat_with_tools.call_count == 1


def test_research_that_reaches_its_step_cap_as_the_room_closes_reports_nothing() -> None:
    """The step cap ended research, not the room."""
    ledger = _closing_ledger()

    _, outcome = _drain(
        _loop(tool_llm([search_call("method")]), _search({"method": _wide(3)}), max_steps=1).run(
            question=QUESTION, prior_turn=None, ledger=ledger
        )
    )

    assert ledger.out_of_room is True
    assert (outcome.steps, outcome.reason) == (1, None)


def test_research_that_fills_the_entry_cap_reports_nothing() -> None:
    """A ledger stopped by its entry cap is not out of room."""
    ledger = _closing_ledger(cap=2)

    _, outcome = _drain(
        _loop(tool_llm([search_call("method")]), _search({"method": _wide(3)})).run(
            question=QUESTION, prior_turn=None, ledger=ledger
        )
    )

    assert (len(ledger), ledger.out_of_room) == (2, False)
    assert (outcome.steps, outcome.reason) == (1, None)


def test_a_refusal_that_does_not_end_research_goes_unreported() -> None:
    """A passage too heavy to admit is passed over; research goes on and ends as it would have."""
    ledger = _closing_ledger()
    heavy = scored_node("heavy", "Heavy passage.", page=1, filename="h" * 1200)

    progress, outcome = _drain(
        _loop(tool_llm([search_call("method")]), _search({"method": [heavy]})).run(
            question=QUESTION, prior_turn=None, ledger=ledger
        )
    )

    assert [step.found for step in progress] == [0]
    assert ledger.full is False
    assert (outcome.steps, outcome.reason) == (2, None)
