"""The research loop. Every query and passage is invented."""

from __future__ import annotations

import contextvars
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextvars import ContextVar
from types import SimpleNamespace
from typing import Any, cast

import httpx
import openai
import pytest
from _pytest.logging import LogCaptureFixture
from assessment_fakes import (
    bad_request,
    fragmented_endpoint,
    join_threads,
    llm_view,
    scored_node,
    search_call,
    tool_llm,
)
from llama_index.core.base.llms.types import ChatResponse, MessageRole, ToolCallBlock
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.schema import MetadataMode, NodeWithScore, TextNode
from typing_extensions import override

from docint.agents.assessment.ledger import EvidenceLedger, Room
from docint.agents.assessment.research import (
    MODEL_SLOT_POOL,
    MODEL_SLOT_WAIT_S,
    MODEL_SLOTS,
    MODEL_THREAD_NAME,
    SEARCH_SLOT_POOL,
    SEARCH_SLOT_WAIT_S,
    SEARCH_SLOTS,
    SEARCH_THREAD_NAME,
    ResearchLoop,
    _bounded,
)
from docint.agents.assessment.runner import ResearchOutcome, SearchFn
from docint.agents.types import AgentProgress
from docint.core.rag import ParentContextPostprocessor
from docint.utils.openai_cfg import LocalOpenAI

QUESTION = "zzsecretzz What could be improved in this study?"


@pytest.fixture(autouse=True)
def _search_capacity_is_whole_again() -> Iterator[None]:
    """Fail any test that leaves one of the process-wide search slots taken.

    Torn down after every fixture the test requested, so a held search is
    released and joined before the slots are counted.

    Yields:
        None: The test runs.
    """
    yield
    taken = 0
    while taken < SEARCH_SLOTS and SEARCH_SLOT_POOL.acquire(blocking=False):
        taken += 1
    for _ in range(taken):
        SEARCH_SLOT_POOL.release()
    assert taken == SEARCH_SLOTS


@pytest.fixture(autouse=True)
def _model_capacity_is_whole_again() -> Iterator[None]:
    """Fail any test that leaves one of the process-wide model-call slots taken.

    Torn down after every fixture the test requested, so a held call is
    released and joined before the slots are counted.

    Yields:
        None: The test runs.
    """
    yield
    taken = 0
    while taken < MODEL_SLOTS and MODEL_SLOT_POOL.acquire(blocking=False):
        taken += 1
    for _ in range(taken):
        MODEL_SLOT_POOL.release()
    assert taken == MODEL_SLOTS


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

    # Start, the budget check and the time left for the call; from the call's slot wait on, 200 s have passed.
    _, outcome = _drain(
        _loop(llm, _search_recorded, clock=_ticking(0.0, 0.0, 0.0, 200.0), wall_clock_s=120.0).run(
            question=QUESTION, prior_turn=None, ledger=EvidenceLedger(cap=5)
        )
    )

    assert outcome.reason == "time_budget"
    assert searched == []


def test_each_research_call_is_bounded_by_the_time_remaining() -> None:
    """One hung call cannot outlast the research budget; a floor of a second keeps the last call possible."""
    # Start; then per round: the budget check, the time left for the call, its slot wait's start and end,
    # the check on its reply, the time left for its search, that slot wait's start and end, the check after.
    clock = _ticking(0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 119.5, 119.5, 119.5, 200.0)
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

    # A fixed clock: only the real wait for the search spends the budget, however the host stalls.
    progress, outcome = _drain(
        _loop(tool_llm([search_call("zzsecretzz method")]), search, wall_clock_s=0.3, clock=lambda: 0.0).run(
            question=QUESTION, prior_turn=None, ledger=ledger
        )
    )

    assert time.monotonic() - started < 3.0
    assert (outcome.reason, progress, outcome.actions) == ("time_budget", [], [])
    assert [scored.node.node_id for scored in ledger.entries()] == ["seed"]
    assert loguru_caplog.messages == [
        "Assessment research stopped (time_budget): no search result arrived within the research budget"
    ]


def _alive_searches() -> int:
    """Count the search threads still running.

    Returns:
        int: Threads named ``SEARCH_THREAD_NAME`` that are alive.
    """
    return sum(thread.name == SEARCH_THREAD_NAME and thread.is_alive() for thread in threading.enumerate())


def test_abandoned_searches_are_capped_at_the_slot_count(held_search: tuple[threading.Event, SearchFn]) -> None:
    """Codex's reproduction: hung searches across turns pile up only until every slot is held, then research stops.

    Args:
        held_search (tuple[threading.Event, SearchFn]): The blocking search.
    """
    _, search = held_search
    calls: list[str] = []

    def _counted(query: str) -> list[NodeWithScore]:
        """Count the search, then block like the held one.

        Args:
            query (str): The search query.

        Returns:
            list[NodeWithScore]: The late passage.
        """
        calls.append(query)
        return search(query)

    # A fixed clock: only the real waits, for a slot and for the search, spend the budget.
    outcomes = [
        _drain(
            _loop(tool_llm([search_call(f"turn {turn}")]), _counted, wall_clock_s=0.2, clock=lambda: 0.0).run(
                question=QUESTION, prior_turn=None, ledger=EvidenceLedger(cap=5)
            )
        )[1].reason
        for turn in range(SEARCH_SLOTS + 1)
    ]

    assert outcomes == ["time_budget"] * SEARCH_SLOTS + ["research_failed"]
    assert len(calls) == SEARCH_SLOTS
    assert _alive_searches() == SEARCH_SLOTS


def _recorded(searched: list[str]) -> SearchFn:
    """Build a search that records each query and finds nothing.

    Args:
        searched (list[str]): Where the queries go.

    Returns:
        SearchFn: The search.
    """

    def _run(query: str) -> list[NodeWithScore]:
        """Record the query.

        Args:
            query (str): The search query.

        Returns:
            list[NodeWithScore]: Nothing.
        """
        searched.append(query)
        return []

    return _run


class _Recording(threading.BoundedSemaphore):
    """Search slots that record how long a search would wait and find them all taken."""

    def __init__(self) -> None:
        """Start with one slot and no waits recorded."""
        super().__init__(1)
        self.waits: list[float | None] = []

    @override
    def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool:
        """Record the wait and report no free slot.

        Args:
            blocking (bool): Whether the caller would block.
            timeout (float | None): How long it would wait.

        Returns:
            bool: Always ``False``.
        """
        self.waits.append(timeout)
        return False


@pytest.mark.parametrize(("budget", "wait"), [(120.0, SEARCH_SLOT_WAIT_S), (2.0, 2.0)])
def test_no_free_slot_stops_research_without_starting_the_search(
    budget: float, wait: float, monkeypatch: pytest.MonkeyPatch, loguru_caplog: LogCaptureFixture
) -> None:
    """A search waits for a slot no longer than the budget allows, and never runs or starts a thread without one.

    Args:
        budget (float): The research budget in seconds.
        wait (float): The longest the search may wait for a slot.
        monkeypatch (pytest.MonkeyPatch): Records thread starts.
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    slots = _Recording()
    searched: list[str] = []
    started: list[str] = []
    real_start = threading.Thread.start

    def _start(thread: threading.Thread) -> None:
        """Record a search thread's start, then start any thread.

        Args:
            thread (threading.Thread): The thread.
        """
        if thread.name == SEARCH_THREAD_NAME:
            started.append(thread.name)
        real_start(thread)

    monkeypatch.setattr(threading.Thread, "start", _start)

    _, outcome = _drain(
        _loop(
            tool_llm([search_call("zzsecretzz method")]),
            _recorded(searched),
            wall_clock_s=budget,
            clock=lambda: 0.0,
            slots=slots,
        ).run(question=QUESTION, prior_turn=None, ledger=EvidenceLedger(cap=5))
    )

    assert (outcome.reason, outcome.actions) == ("research_failed", [])
    assert slots.waits == [wait]
    assert (searched, started) == ([], [])
    assert loguru_caplog.messages == ["Assessment research stopped (research_failed): no search capacity"]


def test_a_timed_out_search_keeps_its_slot_until_it_ends(held_search: tuple[threading.Event, SearchFn]) -> None:
    """Abandoned work stays counted: the slot frees only when retrieval actually finishes.

    Args:
        held_search (tuple[threading.Event, SearchFn]): The blocking search.
    """
    release, search = held_search
    slots = threading.BoundedSemaphore(1)

    # A fixed clock: only the real wait for the search spends the budget, however the host stalls.
    _, outcome = _drain(
        _loop(tool_llm([search_call("method")]), search, wall_clock_s=0.2, clock=lambda: 0.0, slots=slots).run(
            question=QUESTION, prior_turn=None, ledger=EvidenceLedger(cap=5)
        )
    )

    assert outcome.reason == "time_budget"
    assert slots.acquire(blocking=False) is False
    release.set()
    for thread in threading.enumerate():
        if thread.name == SEARCH_THREAD_NAME:
            thread.join(timeout=5)
    assert slots.acquire(blocking=False) is True
    slots.release()


def test_a_search_thread_that_cannot_start_ends_research_and_keeps_what_it_found(
    monkeypatch: pytest.MonkeyPatch, loguru_caplog: LogCaptureFixture
) -> None:
    """The second search's thread cannot start: research fails there, keeping the first search and the slot.

    Args:
        monkeypatch (pytest.MonkeyPatch): Makes the second thread start fail.
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    slots = threading.BoundedSemaphore(1)
    started: list[str] = []
    real_start = threading.Thread.start

    def _start(thread: threading.Thread) -> None:
        """Start the first search thread; fail the next as a process out of threads does.

        Args:
            thread (threading.Thread): The thread.

        Raises:
            RuntimeError: From the second search thread on.
        """
        if thread.name == SEARCH_THREAD_NAME:
            started.append(thread.name)
            if len(started) > 1:
                raise RuntimeError("can't start new thread")
        real_start(thread)

    monkeypatch.setattr(threading.Thread, "start", _start)
    searches = _search({"method": [scored_node("m", "Method section.")], "zzsecretzz results": []})

    progress, outcome = _drain(
        _loop(tool_llm([search_call("method"), search_call("zzsecretzz results")]), searches, slots=slots).run(
            question=QUESTION, prior_turn=None, ledger=EvidenceLedger(cap=5)
        )
    )

    assert [(step.detail, step.found) for step in progress] == [("method", 1)]
    assert (outcome.steps, outcome.reason) == (2, "research_failed")
    assert outcome.actions == [{"action": "search", "detail": "method", "found": 1}]
    assert loguru_caplog.messages == ["Assessment research search could not start: RuntimeError"]
    assert slots.acquire(blocking=False) is True
    slots.release()


class _Abort(BaseException):
    """Ends a search the way an interrupt would, past ``except Exception``."""


def test_a_search_ending_in_a_base_exception_gives_its_slot_back() -> None:
    """The slot is held while retrieval runs and returned however it ends."""
    slots = threading.BoundedSemaphore(1)
    held_while_running: list[bool] = []

    def _search(query: str) -> list[NodeWithScore]:
        """Check the search holds its slot, then end abruptly.

        Args:
            query (str): The search query, unused.

        Raises:
            _Abort: Always.
        """
        free = slots.acquire(blocking=False)
        if free:
            slots.release()
        held_while_running.append(not free)
        raise _Abort

    finished = _bounded(_search, "method", 5.0, slots=slots, clock=time.monotonic)

    assert finished is not None
    assert isinstance(finished.exception(), _Abort)
    assert held_while_running == [True]
    assert slots.acquire(blocking=False) is True
    slots.release()


@pytest.mark.parametrize("failure", ["clock", "context", "thread"])
def test_a_search_that_fails_before_its_thread_runs_gives_its_slot_back(
    failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whatever fails between taking a slot and starting the search's thread gives the slot back.

    Args:
        failure (str): The failing step: the clock read after the wait, copying the context, or building the thread.
        monkeypatch (pytest.MonkeyPatch): Makes that step fail.
    """
    slots = threading.BoundedSemaphore(1)
    reads: list[float] = []

    def _clock() -> float:
        """Read the time once, then fail on the read after the wait when the clock is the failing step.

        Returns:
            float: Zero.

        Raises:
            _Abort: On the second read, when the clock is the failing step.
        """
        if reads and failure == "clock":
            raise _Abort
        reads.append(0.0)
        return 0.0

    def _abort(*args: object, **kwargs: object) -> None:
        """Fail the way the selected step would.

        Args:
            *args (object): Ignored.
            **kwargs (object): Ignored.

        Raises:
            _Abort: Always.
        """
        raise _Abort

    if failure == "context":
        monkeypatch.setattr(contextvars, "copy_context", _abort)
    if failure == "thread":
        monkeypatch.setattr(threading.Thread, "__init__", _abort)

    with pytest.raises(_Abort):
        _bounded(_search({}), "method", 5.0, slots=slots, clock=_clock)

    assert slots.acquire(blocking=False) is True
    slots.release()


class _SlowRelease(threading.BoundedSemaphore):
    """Search slots whose release takes a moment, so its order against the result shows."""

    @override
    def release(self, n: int = 1) -> None:
        """Pause, then give the slot back.

        Args:
            n (int): How many slots to give back.
        """
        time.sleep(0.3)
        super().release(n)


def test_a_search_gives_its_slot_back_before_its_result_is_read() -> None:
    """The caller never finds its own finished search still holding the slot its next search needs."""
    slots = _SlowRelease(1)

    finished = _bounded(_search({}), "method", 5.0, slots=slots, clock=time.monotonic)

    assert finished is not None
    assert slots.acquire(blocking=False) is True
    threading.BoundedSemaphore.release(slots)


class _SlowSlots(threading.BoundedSemaphore):
    """Search slots that take ``delay`` seconds of an injected clock to hand out."""

    def __init__(self, now: list[float], delay: float) -> None:
        """Hold the clock and the delay.

        Args:
            now (list[float]): The injected clock's reading, advanced in place.
            delay (float): Seconds a slot takes to come free.
        """
        super().__init__(1)
        self._now = now
        self._delay = delay

    @override
    def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool:
        """Advance the clock by the delay, then take a slot.

        Args:
            blocking (bool): Whether the caller blocks.
            timeout (float | None): How long it would wait.

        Returns:
            bool: Whether a slot was taken.
        """
        self._now[0] += self._delay
        return super().acquire(blocking, timeout)


def test_the_slot_wait_comes_out_of_the_research_budget(held_search: tuple[threading.Event, SearchFn]) -> None:
    """A slot that takes nearly the whole budget to free leaves the search only what remains.

    Args:
        held_search (tuple[threading.Event, SearchFn]): The blocking search.
    """
    _, search = held_search
    now = [0.0]
    started = time.monotonic()

    _, outcome = _drain(
        _loop(
            tool_llm([search_call("method")]),
            search,
            wall_clock_s=100.0,
            clock=lambda: now[0],
            slots=_SlowSlots(now, 99.9),
        ).run(question=QUESTION, prior_turn=None, ledger=EvidenceLedger(cap=5))
    )

    assert outcome.reason == "time_budget"
    assert time.monotonic() - started < 3.0


def test_a_slot_that_frees_only_after_the_budget_never_starts_the_search(loguru_caplog: LogCaptureFixture) -> None:
    """Once the wait for a slot has used the whole budget, the slot goes back unused.

    Args:
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    now = [0.0]
    slots = _SlowSlots(now, 150.0)
    searched: list[str] = []

    _, outcome = _drain(
        _loop(
            tool_llm([search_call("method")]),
            _recorded(searched),
            wall_clock_s=100.0,
            clock=lambda: now[0],
            slots=slots,
        ).run(question=QUESTION, prior_turn=None, ledger=EvidenceLedger(cap=5))
    )

    assert (outcome.reason, searched) == ("time_budget", [])
    assert loguru_caplog.messages == [
        "Assessment research stopped (time_budget): no search result arrived within the research budget"
    ]
    assert threading.BoundedSemaphore.acquire(slots, blocking=False) is True


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

    # Start, the budget check and the time left for the call; from the call's slot wait on, 200 s have passed.
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


def test_a_room_that_closes_before_the_step_cap_says_so_with_nothing_refused() -> None:
    """Research stops early because the room is full, though its last search fitted whole."""
    ledger = _closing_ledger()
    llm = tool_llm([search_call("method"), search_call("results")])

    _, outcome = _drain(
        _loop(llm, _search({"method": _wide(2), "results": _wide(4)[2:]})).run(
            question=QUESTION, prior_turn=None, ledger=ledger
        )
    )

    assert (len(ledger), ledger.out_of_room, ledger.refusals) == (3, True, 0)
    assert (outcome.steps, outcome.reason) == (1, "context_budget")
    assert llm.chat_with_tools.call_count == 1


def test_a_last_search_that_lost_a_passage_to_the_room_says_so_at_the_step_cap() -> None:
    """The step cap ended research, but its last search offered a passage the closed room turned away."""
    ledger = _closing_ledger()

    _, outcome = _drain(
        _loop(tool_llm([search_call("method")]), _search({"method": _wide(3)}), max_steps=1).run(
            question=QUESTION, prior_turn=None, ledger=ledger
        )
    )

    assert (len(ledger), ledger.out_of_room, ledger.refusals) == (3, True, 1)
    assert (outcome.steps, outcome.reason) == (1, "context_budget")


def test_an_exit_at_the_step_cap_with_nothing_refused_reports_nothing() -> None:
    """The last search fitted whole; that the room is now closed is no reduction."""
    ledger = _closing_ledger()

    _, outcome = _drain(
        _loop(tool_llm([search_call("method")]), _search({"method": _wide(2)}), max_steps=1).run(
            question=QUESTION, prior_turn=None, ledger=ledger
        )
    )

    assert (len(ledger), ledger.out_of_room, ledger.refusals) == (3, True, 0)
    assert (outcome.steps, outcome.reason) == (1, None)


def test_research_that_fills_the_entry_cap_reports_nothing() -> None:
    """A ledger stopped by its entry cap is not out of room."""
    ledger = _closing_ledger(cap=2)

    _, outcome = _drain(
        _loop(tool_llm([search_call("method")]), _search({"method": _wide(3)})).run(
            question=QUESTION, prior_turn=None, ledger=ledger
        )
    )

    assert (len(ledger), ledger.out_of_room, ledger.refusals) == (2, False, 0)
    assert (outcome.steps, outcome.reason) == (1, None)


#: A passage whose header alone outweighs the room :func:`_closing_ledger` has left.
HEAVY = scored_node("heavy", "Heavy passage.", page=1, filename="h" * 1200)


def test_a_last_search_whose_passage_the_room_refused_says_so_after_finish() -> None:
    """Research ended with ``finish``, but the evidence its last search found did not all fit."""
    ledger = _closing_ledger()

    progress, outcome = _drain(
        _loop(tool_llm([search_call("method")]), _search({"method": [HEAVY]})).run(
            question=QUESTION, prior_turn=None, ledger=ledger
        )
    )

    assert [step.found for step in progress] == [0]
    assert (ledger.full, ledger.refusals) == (False, 1)
    assert (outcome.steps, outcome.reason) == (2, "context_budget")


def test_a_refusal_followed_by_a_search_that_adds_goes_unreported() -> None:
    """A later search that fits whole means research went on past the refusal."""
    ledger = _closing_ledger()
    light = [scored_node("light", "Light passage.", page=2)]

    progress, outcome = _drain(
        _loop(
            tool_llm([search_call("method"), search_call("results")]), _search({"method": [HEAVY], "results": light})
        ).run(question=QUESTION, prior_turn=None, ledger=ledger)
    )

    assert [step.found for step in progress] == [0, 1]
    assert ledger.refusals == 1
    assert (outcome.steps, outcome.reason) == (3, None)


def test_a_reason_research_already_gave_outranks_a_last_refusal() -> None:
    """The first cause is kept: a last search that lost a passage and ran out of time reports the time."""
    ledger = _closing_ledger()
    # Start; the budget check, the time left for the call, its slot wait's start and end, the check on its reply,
    # the time left for the search, that slot wait's start and end; then the check after.
    clock = _ticking(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 200.0)

    _, outcome = _drain(
        _loop(tool_llm([search_call("method")]), _search({"method": _wide(3)}), clock=clock).run(
            question=QUESTION, prior_turn=None, ledger=ledger
        )
    )

    assert ledger.refusals == 1
    assert outcome.reason == "time_budget"


#: A research reply in which the model calls ``finish``, as an OpenAI-compatible endpoint sends it.
FINISH_REPLY: dict[str, Any] = {
    "id": "invented-completion",
    "object": "chat.completion",
    "created": 0,
    "model": "invented-model",
    "choices": [
        {
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "finish", "arguments": "{}"}}],
            },
        }
    ],
}


def test_a_reply_that_arrives_in_fragments_past_the_budget_ends_research_on_time(
    loguru_caplog: LogCaptureFixture,
) -> None:
    """Codex's reproduction: each fragment of a ``finish`` lands within the client's timeout, the reply past the budget.

    The SDK's timeout bounds each read, not the call, so only the research deadline can stop it; the
    call is abandoned in its slot, and joined here once it ends.

    Args:
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    with fragmented_endpoint(FINISH_REPLY) as base:
        llm = LocalOpenAI(api_base=base, api_key="test", model="invented-model", max_retries=0)
        started = time.monotonic()
        _, outcome = _drain(
            _loop(llm, _search({}), wall_clock_s=1.0).run(
                question=QUESTION, prior_turn=None, ledger=EvidenceLedger(cap=5)
            )
        )
        elapsed = time.monotonic() - started
        join_threads(MODEL_THREAD_NAME)

    assert elapsed < 2.0
    assert (outcome.reason, outcome.steps) == ("time_budget", 0)
    assert loguru_caplog.messages == [
        "Assessment research stopped (time_budget): no model reply arrived within the research budget"
    ]


@pytest.fixture
def held_model() -> Iterator[tuple[threading.Event, Any]]:
    """A model whose call blocks until released, then calls ``finish``; teardown releases it and joins its thread.

    Yields:
        tuple[threading.Event, Any]: The release and the model double.
    """
    release = threading.Event()
    llm = tool_llm([])
    reply = llm.chat_with_tools.side_effect

    def _held(tools: Sequence[Any], **kwargs: Any) -> ChatResponse:
        """Wait for the release, then reply as the double does.

        Args:
            tools (Sequence[Any]): The tools offered.
            **kwargs (Any): The call's other arguments.

        Returns:
            ChatResponse: A ``finish`` call.
        """
        release.wait(timeout=10)
        return reply(tools, **kwargs)

    llm.chat_with_tools.side_effect = _held
    yield release, llm
    release.set()
    join_threads(MODEL_THREAD_NAME)


def test_a_model_call_that_outlasts_the_budget_keeps_its_slot_until_it_ends(
    held_model: tuple[threading.Event, Any], loguru_caplog: LogCaptureFixture
) -> None:
    """Research ends at its budget, but the abandoned call stays counted until it really finishes.

    Args:
        held_model (tuple[threading.Event, Any]): The blocking model.
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    release, llm = held_model
    slots = threading.BoundedSemaphore(1)
    started = time.monotonic()

    # A fixed clock: only the real wait for the reply spends the budget, however the host stalls.
    _, outcome = _drain(
        _loop(llm, _search({}), wall_clock_s=0.2, clock=lambda: 0.0, model_slots=slots).run(
            question=QUESTION, prior_turn=None, ledger=EvidenceLedger(cap=5)
        )
    )

    assert time.monotonic() - started < 3.0
    assert (outcome.reason, outcome.steps, outcome.actions) == ("time_budget", 0, [])
    assert loguru_caplog.messages == [
        "Assessment research stopped (time_budget): no model reply arrived within the research budget"
    ]
    assert slots.acquire(blocking=False) is False
    release.set()
    join_threads(MODEL_THREAD_NAME)
    assert slots.acquire(blocking=False) is True
    slots.release()


@pytest.mark.parametrize(("budget", "wait"), [(120.0, MODEL_SLOT_WAIT_S), (2.0, 2.0)])
def test_no_free_model_slot_stops_research_without_calling_the_model(
    budget: float, wait: float, loguru_caplog: LogCaptureFixture
) -> None:
    """A model call waits for a slot no longer than the budget allows, and never runs without one.

    Args:
        budget (float): The research budget in seconds.
        wait (float): The longest the call may wait for a slot.
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    slots = _Recording()
    llm = tool_llm([search_call("method")])
    searched: list[str] = []

    _, outcome = _drain(
        _loop(llm, _recorded(searched), wall_clock_s=budget, clock=lambda: 0.0, model_slots=slots).run(
            question=QUESTION, prior_turn=None, ledger=EvidenceLedger(cap=5)
        )
    )

    assert (outcome.reason, outcome.steps, outcome.actions) == ("research_failed", 0, [])
    assert slots.waits == [wait]
    assert searched == []
    llm.chat_with_tools.assert_not_called()
    assert loguru_caplog.messages == ["Assessment research stopped (research_failed): no model capacity"]


def test_a_model_thread_that_cannot_start_ends_research_and_gives_its_slot_back(
    monkeypatch: pytest.MonkeyPatch, loguru_caplog: LogCaptureFixture
) -> None:
    """A process out of threads fails research there, with the model never called and its slot free again.

    Args:
        monkeypatch (pytest.MonkeyPatch): Makes the model thread's start fail.
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    slots = threading.BoundedSemaphore(1)
    llm = tool_llm([search_call("method")])
    real_start = threading.Thread.start

    def _start(thread: threading.Thread) -> None:
        """Fail a model thread's start as a process out of threads does; start any other thread.

        Args:
            thread (threading.Thread): The thread.

        Raises:
            RuntimeError: For a model thread.
        """
        if thread.name == MODEL_THREAD_NAME:
            raise RuntimeError("can't start new thread")
        real_start(thread)

    monkeypatch.setattr(threading.Thread, "start", _start)

    _, outcome = _drain(
        _loop(llm, _search({}), model_slots=slots).run(question=QUESTION, prior_turn=None, ledger=EvidenceLedger(cap=5))
    )

    assert (outcome.reason, outcome.steps) == ("research_failed", 0)
    llm.chat_with_tools.assert_not_called()
    assert loguru_caplog.messages == ["Assessment research model call could not start: RuntimeError"]
    assert slots.acquire(blocking=False) is True
    slots.release()


def _replying_late(reply: ToolSelection | str, now: list[float]) -> Any:
    """Build a model double whose one reply lands at 200 s on the injected clock.

    Args:
        reply (ToolSelection | str): The reply: a tool call, or prose.
        now (list[float]): The injected clock's reading, advanced in place.

    Returns:
        Any: The model double.
    """
    llm = tool_llm([reply])
    scripted = llm.chat_with_tools.side_effect

    def _late(tools: Sequence[Any], **kwargs: Any) -> ChatResponse:
        """Let 200 s pass on the injected clock, then reply as scripted.

        Args:
            tools (Sequence[Any]): The tools offered.
            **kwargs (Any): The call's other arguments.

        Returns:
            ChatResponse: The scripted reply.
        """
        now[0] = 200.0
        return scripted(tools, **kwargs)

    llm.chat_with_tools.side_effect = _late
    return llm


@pytest.mark.parametrize(
    "reply",
    [
        ToolSelection(tool_id="call_1", tool_name="finish", tool_kwargs={}),
        "Here is my assessment already.",
        search_call("method"),
    ],
    ids=["finish", "prose", "search"],
)
def test_a_reply_that_lands_after_the_budget_is_not_acted_on(
    reply: ToolSelection | str, loguru_caplog: LogCaptureFixture
) -> None:
    """A late ``finish`` is no clean end, late prose is no missing tool support, and a late search never runs.

    Args:
        reply (ToolSelection | str): What the model replies, at 200 s of a 120 s budget.
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    now = [0.0]
    searched: list[str] = []

    progress, outcome = _drain(
        _loop(_replying_late(reply, now), _recorded(searched), wall_clock_s=120.0, clock=lambda: now[0]).run(
            question=QUESTION, prior_turn=None, ledger=EvidenceLedger(cap=5)
        )
    )

    assert (outcome.reason, outcome.steps, outcome.actions) == ("time_budget", 0, [])
    assert (progress, searched) == ([], [])
    assert loguru_caplog.messages == [
        "Assessment research stopped (time_budget): the model replied after the research budget"
    ]


def test_a_model_call_runs_on_its_own_thread_in_the_callers_context() -> None:
    """The call is abandoned at the deadline, so it runs on its own thread, still seeing the request's scope."""
    seen: list[tuple[str, str | None]] = []
    llm = tool_llm([])
    reply = llm.chat_with_tools.side_effect

    def _record(tools: Sequence[Any], **kwargs: Any) -> ChatResponse:
        """Record the thread and the scope the call runs under, then reply as the double does.

        Args:
            tools (Sequence[Any]): The tools offered.
            **kwargs (Any): The call's other arguments.

        Returns:
            ChatResponse: A ``finish`` call.
        """
        seen.append((threading.current_thread().name, _SCOPE.get()))
        return reply(tools, **kwargs)

    llm.chat_with_tools.side_effect = _record
    token = _SCOPE.set("invented-collection")
    try:
        _drain(_loop(llm, _search({})).run(question=QUESTION, prior_turn=None, ledger=EvidenceLedger(cap=5)))
    finally:
        _SCOPE.reset(token)

    assert seen == [(MODEL_THREAD_NAME, "invented-collection")]
