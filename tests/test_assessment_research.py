"""The research loop. Every query and passage is invented."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

import httpx
import openai
from _pytest.logging import LogCaptureFixture
from assessment_fakes import scored_node, search_call, tool_llm
from llama_index.core.base.llms.types import MessageRole, ToolCallBlock
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.schema import NodeWithScore

from docint.agents.assessment.ledger import EvidenceLedger
from docint.agents.assessment.research import ResearchLoop
from docint.agents.assessment.runner import ResearchOutcome
from docint.agents.types import AgentProgress

QUESTION = "zzsecretzz What could be improved in this study?"


def _bad_request(message: str) -> openai.BadRequestError:
    """Build the error an OpenAI-compatible endpoint raises on HTTP 400.

    Args:
        message (str): The error message.

    Returns:
        openai.BadRequestError: The error.
    """
    request = httpx.Request("POST", "http://inference.invalid/v1/chat/completions")
    return openai.BadRequestError(message, response=httpx.Response(400, request=request), body=None)


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


def test_a_text_reply_ends_research_without_leaking_into_the_answer() -> None:
    """A model that answers in prose instead of calling a tool simply stops researching."""
    llm = tool_llm(["Here is my assessment already."])

    progress, outcome = _drain(
        _loop(llm, _search({})).run(question=QUESTION, prior_turn=None, ledger=EvidenceLedger(cap=5))
    )

    assert progress == []
    assert (outcome.steps, outcome.reason) == (0, None)


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
    llm = tool_llm([_bad_request(f"model does not support tools: {QUESTION}")])

    _, outcome = _drain(_loop(llm, _search({})).run(question=QUESTION, prior_turn=None, ledger=EvidenceLedger(cap=5)))

    assert outcome.reason == "tools_unavailable"
    assert "zzsecretzz" not in loguru_caplog.text


def test_a_context_error_reports_context_budget() -> None:
    """A context-length 400 is a budget stop, not missing tool support."""
    llm = tool_llm([_bad_request("This model's maximum context length is 8192 tokens; input tokens exceeded")])

    _, outcome = _drain(_loop(llm, _search({})).run(question=QUESTION, prior_turn=None, ledger=EvidenceLedger(cap=5)))

    assert outcome.reason == "context_budget"


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
    """A slow endpoint cannot hold the turn open indefinitely."""
    ticks = iter([0.0, 0.0, 200.0])
    llm = tool_llm([search_call("method"), search_call("results")])

    _, outcome = _drain(
        _loop(llm, _search({"method": [scored_node("m", "M.")]}), clock=lambda: next(ticks), wall_clock_s=120.0).run(
            question=QUESTION, prior_turn=None, ledger=EvidenceLedger(cap=5)
        )
    )

    assert outcome.reason == "time_budget"


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
