"""The agent loop: model rounds, the tool calls they make, and the limits that make the model answer.

Every message, passage, query and tool call here is invented.
"""

from __future__ import annotations

import concurrent.futures
import json
import threading
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping
from contextvars import ContextVar
from typing import Any, cast

import httpx
import openai
import pytest
from _pytest.logging import LogCaptureFixture
from agent_fakes import (
    chunk,
    programme_passage,
    shown_numbers,
    streaming_tool_llm,
    text_chunks,
    tool_call_delta,
    turn_budget,
)
from assessment_fakes import bad_request, join_threads, llm_view, scored_node
from llama_index.core.base.llms.types import ChatMessage, MessageRole, TextBlock, ToolCallBlock
from llama_index.core.schema import NodeWithScore
from llama_index.llms.openai.utils import to_openai_message_dicts
from typing_extensions import override

from docint.agents.assessment.bounded import (
    MODEL_SLOT_POOL,
    MODEL_SLOT_WAIT_S,
    MODEL_SLOTS,
    SEARCH_SLOT_POOL,
    SEARCH_SLOT_WAIT_S,
    SEARCH_SLOTS,
    NoSlot,
)
from docint.agents.assessment.ledger import MAX_LEDGER_ENTRIES, EvidenceLedger
from docint.agents.assessment.split import SENTINEL, assessment_label
from docint.agents.turn.budget import TurnBudget
from docint.agents.turn.calls import (
    FAILED_REPLY,
    LATE_REPLY,
    NO_CAPACITY_REPLY,
    NO_ROOM_REPLY,
    NO_TIME_REPLY,
    SEARCH_THREAD_NAME,
    SKIPPED_REPLY,
    RoundCalls,
)
from docint.agents.turn.history import ConversationTurn, ConversationWindow
from docint.agents.turn.loop import NOTE_CHARS, OVERFLOW_REPLY_SHARE, AgentLoop, AgentResult, ToolsUnavailable
from docint.agents.turn.tools import (
    ReadTool,
    SearchFn,
    SearchTool,
    ToolContext,
    ToolRegistry,
    ToolReply,
    ToolWork,
    error_reply,
)
from docint.agents.types import AgentProgress
from docint.utils.text_budget import fit_with_marker

QUESTION = "zzsecretzz Wann öffnen die Stände beim Hafenfest?"
SYSTEM_PROMPT = "Beantworte Fragen zur Sammlung mit den Werkzeugen und belege jede Aussage mit [n]."
FINAL_NOTE = "Antworte jetzt aus den Passagen oben; rufe keine Werkzeuge mehr auf."
ANSWER = (
    "Die Stände öffnen am Sonntag um zehn Uhr [1]. Das Hafenfest wurde wegen der Sturmwarnung verschoben, "
    "und für Sonntag ist bedeckter Himmel gemeldet [2]."
)
SHORT_ANSWER = "Am Sonntag um zehn Uhr [1]."
WEATHER = "Für Sonntag meldet der Wetterdienst bedeckten Himmel, Wind aus West und kühle Temperaturen am Hafen."
PREFACE = "Ich suche zuerst nach dem Programm des Hafenfests, dann nach dem Wetterbericht für Sonntag."
LONG_PREFACE = "Ich schaue im Programm und im Wetterbericht nach, um die Frage zu beantworten. " * 9
HARBOUR_PLAN = "Der Hafenverein plant das Hafenfest mit Ständen, Musik und einer Regatta am Kai. " * 30
OVERFLOW = "This model's maximum context length is 8192 tokens; your request has 9001 input tokens."
TOOL_CHOICE_REJECTED = (
    '"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set; zzsecretzz'
)
#: A turn whose room holds one full tool reply but not two: 1559 tokens in all.
TIGHT = TurnBudget(
    ctx_window=8192,
    reasoning=False,
    char_token_ratio=3.5,
    system_tokens=2889,
    tool_schema_tokens=630,
    history_tokens=1300,
    message_tokens=300,
)


@pytest.fixture(autouse=True)
def _search_capacity_is_whole_again() -> Iterator[None]:
    """Fail any test that leaves one of the process-wide search slots taken.

    Torn down after every fixture the test requested, so a held search is released
    and joined before the slots are counted.

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
    """Fail any test that leaves one of the process-wide model slots taken.

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


@pytest.fixture
def held() -> Iterator[threading.Event]:
    """Hold a search until the test lets it go; teardown lets go and joins the search threads.

    Yields:
        threading.Event: The event a held search waits on.
    """
    release = threading.Event()
    yield release
    release.set()
    join_threads(SEARCH_THREAD_NAME)


def weather_passage() -> NodeWithScore:
    """Build the invented weather report, a page of a document.

    Returns:
        NodeWithScore: The passage.
    """
    return scored_node("weather", WEATHER, filename="wetterbericht.pdf", page=2)


def long_passages(count: int = 3) -> list[NodeWithScore]:
    """Build passages long enough that a search reply showing them fills its whole length.

    Args:
        count (int): How many.

    Returns:
        list[NodeWithScore]: The passages, each its own page.
    """
    return [
        scored_node(f"plan-{number}", HARBOUR_PLAN[:1500], filename="hafenplan.pdf", page=number)
        for number in range(1, count + 1)
    ]


def _answer(text: str = ANSWER, size: int = 40) -> list[dict[str, Any]]:
    """Script a round that answers: the text in pieces, then the end.

    Args:
        text (str): The answer.
        size (int): Characters per piece.

    Returns:
        list[dict[str, Any]]: The round's chunks.
    """
    return [*text_chunks(text, size), chunk(finish_reason="stop")]


def _calls(*calls: tuple[str, str, Any], text: str = "") -> list[dict[str, Any]]:
    """Script a round that asks for tool calls: optional text first, then every call whole in one chunk.

    Args:
        *calls (tuple[str, str, Any]): Per call its id, the tool's name and the arguments, as an object
            sent as JSON or as raw text sent as it is.
        text (str): What the round writes before its calls.

    Returns:
        list[dict[str, Any]]: The round's chunks.
    """
    deltas = [
        tool_call_delta(
            index=index,
            call_id=call_id,
            name=name,
            arguments=arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False),
        )
        for index, (call_id, name, arguments) in enumerate(calls)
    ]
    return [*text_chunks(text, 40), chunk(tool_calls=deltas, finish_reason="tool_calls")]


def _search_call(call_id: str, query: str) -> tuple[str, str, Any]:
    """Name one ``search`` call.

    Args:
        call_id (str): The call's id.
        query (str): What it searches for.

    Returns:
        tuple[str, str, Any]: The call, as :func:`_calls` takes it.
    """
    return (call_id, "search", {"query": query})


class QuerySearch:
    """A search double answering by query, safe to run on several threads at once."""

    def __init__(self, results: Mapping[str, list[NodeWithScore]]) -> None:
        """Hold the passages each query finds.

        Args:
            results (Mapping[str, list[NodeWithScore]]): Passages per query; any other query finds none.
        """
        self._results = dict(results)
        self.queries: list[str] = []
        self._lock = threading.Lock()

    def __call__(self, query: str) -> list[NodeWithScore]:
        """Run one search.

        Args:
            query (str): What the tool searched for.

        Returns:
            list[NodeWithScore]: The passages for the query.
        """
        with self._lock:
            self.queries.append(query)
        return list(self._results.get(query, []))


def _documents(search: SearchFn) -> ToolRegistry:
    """Offer the documents target's tools.

    Args:
        search (SearchFn): The text search.

    Returns:
        ToolRegistry: ``search`` and ``read``.
    """
    return ToolRegistry([SearchTool(search), ReadTool()])


def _loop(llm: Any, registry: ToolRegistry, **overrides: Any) -> AgentLoop:
    """Build an agent loop with generous limits at an 8k window.

    Args:
        llm (Any): The model double.
        registry (ToolRegistry): The tools offered.
        **overrides (Any): Constructor arguments to change.

    Returns:
        AgentLoop: The loop.
    """
    kwargs: dict[str, Any] = {
        "budget": turn_budget(),
        "system_prompt": SYSTEM_PROMPT,
        "final_note": FINAL_NOTE,
        "max_rounds": 6,
        "time_budget_s": 120.0,
        "max_calls_per_round": 4,
        "char_token_ratio": 3.5,
        "llm_view": llm_view,
    }
    kwargs.update(overrides)
    return AgentLoop(llm=cast(Any, llm), registry=registry, **kwargs)


def _turn(
    loop: AgentLoop,
    *,
    window: ConversationWindow | None = None,
    ledger: EvidenceLedger | None = None,
) -> Iterator[AgentProgress | str | AgentResult]:
    """Run one turn of the user's invented question.

    Args:
        loop (AgentLoop): The loop.
        window (ConversationWindow | None): The conversation before it; none by default.
        ledger (EvidenceLedger | None): The turn's evidence; an empty one by default.

    Returns:
        Iterator[AgentProgress | str | AgentResult]: The turn's output.
    """
    return loop.run(
        window=window or ConversationWindow(),
        message=QUESTION,
        ledger=ledger or EvidenceLedger(cap=MAX_LEDGER_ENTRIES),
    )


def _drain(items: Iterable[AgentProgress | str | AgentResult]) -> tuple[list[AgentProgress], list[str], AgentResult]:
    """Split a turn into its steps, its answer text and its result, checking the result comes last and once.

    Args:
        items (Iterable[AgentProgress | str | AgentResult]): The turn's output.

    Returns:
        tuple[list[AgentProgress], list[str], AgentResult]: Steps, answer pieces and the result.
    """
    collected = list(items)
    result = collected[-1]
    assert isinstance(result, AgentResult)
    progress = [item for item in collected[:-1] if isinstance(item, AgentProgress)]
    texts = [item for item in collected[:-1] if isinstance(item, str)]
    assert len(progress) + len(texts) == len(collected) - 1
    return progress, texts, result


def _until_raised(items: Iterator[AgentProgress | str | AgentResult]) -> tuple[list[Any], BaseException]:
    """Collect a turn's output until it raises.

    Args:
        items (Iterator[AgentProgress | str | AgentResult]): The turn's output.

    Returns:
        tuple[list[Any], BaseException]: What it yielded first, and what it raised.
    """
    collected: list[Any] = []
    try:
        for item in items:
            collected.append(item)
    except BaseException as exc:
        return collected, exc
    raise AssertionError("The turn ended without raising.")


def _sent(llm: Any, call: int) -> list[ChatMessage]:
    """Return the messages one model call was sent.

    Args:
        llm (Any): The model double.
        call (int): The call's position, from zero.

    Returns:
        list[ChatMessage]: The messages.
    """
    return list(llm.stream_chat.call_args_list[call].args[0])


def _tool_replies(llm: Any, call: int) -> dict[str, str]:
    """Return the tool replies one model call was sent, by the id of the call each answers.

    Args:
        llm (Any): The model double.
        call (int): The call's position, from zero.

    Returns:
        dict[str, str]: Reply text by call id, in transcript order.
    """
    return {
        str(message.additional_kwargs["tool_call_id"]): str(message.content)
        for message in _sent(llm, call)
        if message.role == MessageRole.TOOL
    }


def _steps(progress: Iterable[AgentProgress]) -> list[tuple[str, str, int]]:
    """Read the steps as the trail shows them.

    Args:
        progress (Iterable[AgentProgress]): The steps.

    Returns:
        list[tuple[str, str, int]]: Each step's action, detail and found count.
    """
    return [(step.action, step.detail, step.found) for step in progress]


def _advancing(llm: Any, now: list[float], *, after: int, to: float) -> Any:
    """Let the injected clock jump while the model streams, after it has sent ``after`` chunks.

    Args:
        llm (Any): The model double.
        now (list[float]): The clock's reading, changed in place.
        after (int): Chunks streamed before the jump; ``0`` jumps before the first.
        to (float): The reading after the jump.

    Returns:
        Any: The same double; it jumps on its first call only.
    """
    inner = llm.stream_chat.side_effect
    jumped: list[bool] = []

    def _stream(messages: list[ChatMessage], **kwargs: Any) -> Iterator[Any]:
        """Stream as scripted, moving the clock once on the way.

        Args:
            messages (list[ChatMessage]): The prompt.
            **kwargs (Any): Call options.

        Yields:
            Any: The scripted responses.
        """
        for count, response in enumerate(inner(messages, **kwargs)):
            if count == after and not jumped:
                jumped.append(True)
                now[0] = to
            yield response

    llm.stream_chat.side_effect = _stream
    return llm


class _Recording(threading.BoundedSemaphore):
    """Slots that record how long a caller would wait and find them all taken."""

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


class _Granting(threading.BoundedSemaphore):
    """Slots that grant or refuse each request as scripted, recording the wait asked for."""

    def __init__(self, *grants: bool, size: int = 1) -> None:
        """Hold the script.

        Args:
            *grants (bool): Per request, whether a slot comes free; later requests are granted.
            size (int): How many slots there are.
        """
        super().__init__(size)
        self._grants = list(grants)
        self.waits: list[float | None] = []

    @override
    def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool:
        """Record the wait, then grant or refuse as scripted.

        Args:
            blocking (bool): Whether the caller blocks.
            timeout (float | None): How long it would wait.

        Returns:
            bool: Whether a slot was taken.
        """
        self.waits.append(timeout)
        if self._grants and not self._grants.pop(0):
            return False
        return super().acquire(blocking, timeout)


class _SlowSlots(threading.BoundedSemaphore):
    """Slots that take ``delay`` seconds of an injected clock to hand out."""

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


def _status_error(cls: type[openai.APIStatusError], status: int, message: str) -> openai.APIStatusError:
    """Build the error an OpenAI-compatible endpoint raises on an HTTP error status.

    Args:
        cls (type[openai.APIStatusError]): The SDK's error class for the status.
        status (int): The HTTP status.
        message (str): The error message.

    Returns:
        openai.APIStatusError: The error.
    """
    request = httpx.Request("POST", "http://inference.invalid/v1/chat/completions")
    return cls(message, response=httpx.Response(status, request=request), body=None)


# --- The thirteen behaviours the plan names -------------------------------------------------


@pytest.mark.parametrize("answer", [ANSWER, SHORT_ANSWER], ids=["past-the-hold", "within-the-hold"])
def test_a_reply_without_calls_is_the_streamed_answer(answer: str) -> None:
    """A round that calls no tool is the answer: streamed, and returned whole with nothing found.

    Args:
        answer (str): The model's reply, long enough to stream as it comes or short enough to be held.
    """
    llm = streaming_tool_llm([_answer(answer)])
    registry = _documents(QuerySearch({}))

    progress, texts, result = _drain(_turn(_loop(llm, registry)))

    assert ("".join(texts), progress) == (answer, [])
    assert result == AgentResult(raw_text=answer, nodes=[], steps=0, actions=[], reason=None)
    (call,) = llm.stream_chat.call_args_list
    assert (call.kwargs["tools"], call.kwargs["tool_choice"]) == (registry.schemas(), "auto")


@pytest.mark.parametrize(
    "answer",
    [ANSWER[: ANSWER.rindex("[") + 1], "Am Sonntag um zehn Uhr ["],
    ids=["past-the-hold", "within-the-hold"],
)
def test_an_answer_ending_where_a_sentinel_could_begin_is_sent_whole(answer: str) -> None:
    """The splitter holds back a tail that could open a sentinel; the end of the answer releases it.

    Args:
        answer (str): An answer cut off mid-citation, as a reply that hit its length limit ends.
    """
    llm = streaming_tool_llm([_answer(answer)])

    _, texts, result = _drain(_turn(_loop(llm, _documents(QuerySearch({})))))

    assert ("".join(texts), result.raw_text) == (answer, answer)


def test_parallel_calls_run_on_search_slots_and_number_in_call_order() -> None:
    """Two searches run at once, each on a search slot; their passages are numbered in the order they were called."""
    weather_started = threading.Event()
    threads: list[str] = []
    slots = _Granting(size=2)

    def _search(query: str) -> list[NodeWithScore]:
        """Find the programme only once the weather search has begun, so the two cannot run one after the other.

        Args:
            query (str): What the tool searched for.

        Returns:
            list[NodeWithScore]: The passage for the query.

        Raises:
            AssertionError: When the weather search never began alongside.
        """
        threads.append(threading.current_thread().name)
        if query == "Programm":
            if not weather_started.wait(timeout=5):
                raise AssertionError("The searches ran one after the other.")
            return [programme_passage()]
        weather_started.set()
        return [weather_passage()]

    llm = streaming_tool_llm([_calls(_search_call("call_a", "Programm"), _search_call("call_b", "Wetter")), _answer()])

    progress, _, result = _drain(_turn(_loop(llm, _documents(_search), search_slots=slots)))

    assert threads == [SEARCH_THREAD_NAME, SEARCH_THREAD_NAME]
    assert len(slots.waits) == 2
    assert _steps(progress) == [("search", "Programm", 1), ("search", "Wetter", 1)]
    assert {call: shown_numbers(text) for call, text in _tool_replies(llm, 1).items()} == {
        "call_a": [1],
        "call_b": [2],
    }
    assert [(scored.node.node_id, scored.node.metadata["citation_index"]) for scored in result.nodes] == [
        ("programme", 1),
        ("weather", 2),
    ]
    assert result.steps == 1


def test_every_call_gets_a_reply_even_on_timeout(held: threading.Event) -> None:
    """A search that outlives the time left, a read, an unoffered tool and malformed arguments each get their reply.

    Args:
        held (threading.Event): Holds the programme search.
    """

    def _search(query: str) -> list[NodeWithScore]:
        """Hold the programme search until the test ends; find the weather at once.

        Args:
            query (str): What the tool searched for.

        Returns:
            list[NodeWithScore]: The passage for the query.
        """
        if query == "Programm":
            held.wait(timeout=30)
            return [programme_passage()]
        return [weather_passage()]

    now = [0.0]
    llm = streaming_tool_llm(
        [
            _calls(_search_call("call_a", "Wetter")),
            _calls(
                _search_call("call_b", "Programm"),
                ("call_c", "read", {"n": 1}),
                ("call_d", "keyframes", {"file": "harbour_walk.webm"}),
                ("call_e", "read", "nicht json"),
            ),
            _answer(),
        ]
    )
    # Each search slot takes 60 of the turn's 100 s to come free: the second search starts with
    # 40 s left and is waited on with none, so it outlives the time left without a real wait.
    loop = _loop(
        llm, _documents(_search), search_slots=_SlowSlots(now, 60.0), time_budget_s=100.0, clock=lambda: now[0]
    )

    progress, _, result = _drain(_turn(loop))

    replies = _tool_replies(llm, 2)
    assert list(replies) == ["call_a", "call_b", "call_c", "call_d", "call_e"]
    assert replies["call_b"] == error_reply("search", LATE_REPLY).text
    assert shown_numbers(replies["call_c"]) == [1]
    assert replies["call_d"].startswith("Error: there is no tool named 'keyframes'")
    assert replies["call_e"].startswith("Error: the arguments for read must be a JSON object")
    # Only calls their tool answered are steps; the time-out is no failed search, the time limit reports it.
    assert (_steps(progress), result.reason) == ([("search", "Wetter", 1), ("read", "1", 0)], "time_budget")
    held.set()
    join_threads(SEARCH_THREAD_NAME)
    assert [scored.node.node_id for scored in result.nodes] == ["weather"]


@pytest.mark.parametrize("preface", [PREFACE, LONG_PREFACE], ids=["held", "sent"])
def test_text_before_calls_becomes_a_note(preface: str) -> None:
    """A round's text before its calls is a note step ahead of the calls; text already streamed went out first.

    Args:
        preface (str): What the round writes before its call: short enough to be held, or long
            enough to have gone out as provisional answer text.
    """
    llm = streaming_tool_llm([_calls(_search_call("call_a", "Programm"), text=preface), _answer()])

    items = list(_turn(_loop(llm, _documents(QuerySearch({"Programm": [programme_passage()]})))))

    note = AgentProgress(step=1, action="note", detail=" ".join(preface.split())[:NOTE_CHARS], found=0)
    position = items.index(note)
    assert items[position + 1] == AgentProgress(step=1, action="search", detail="Programm", found=1)
    streamed_first = "".join(item for item in items[:position] if isinstance(item, str))
    assert streamed_first == ("" if preface is PREFACE else preface.rstrip())
    result = items[-1]
    assert isinstance(result, AgentResult)
    assert result.actions == [
        {"action": "note", "detail": note.detail, "found": 0},
        {"action": "search", "detail": "Programm", "found": 1},
    ]
    assert len(note.detail) <= NOTE_CHARS
    assert result.raw_text == ANSWER


def test_round_budget_forces_a_tool_less_answer_with_reason_round_budget() -> None:
    """Once the rounds are spent, the model answers in one more round: no tool to choose, the tools still declared."""
    llm = streaming_tool_llm(
        [_calls(_search_call("call_a", "Programm")), _calls(_search_call("call_b", "Wetter")), _answer()]
    )
    registry = _documents(QuerySearch({"Programm": [programme_passage()], "Wetter": [weather_passage()]}))

    progress, texts, result = _drain(_turn(_loop(llm, registry, max_rounds=2)))

    assert [(step.step, step.detail) for step in progress] == [(1, "Programm"), (2, "Wetter")]
    final = llm.stream_chat.call_args_list[2]
    assert (final.kwargs["tool_choice"], final.kwargs["tools"]) == ("none", registry.schemas())
    sent = _sent(llm, 2)
    assert (sent[-1].role, sent[-1].content) == (MessageRole.USER, FINAL_NOTE)
    assert sent[-2].role == MessageRole.TOOL
    assert (result.reason, result.steps, "".join(texts)) == ("round_budget", 2, ANSWER)


@pytest.mark.parametrize("after", [0, 2], ids=["before-the-first-token", "mid-answer"])
def test_time_budget_never_abandons_an_answer_in_flight(after: int) -> None:
    """The time running out while the answer round streams changes nothing: the answer is sent whole.

    Args:
        after (int): Chunks streamed before the clock passes the budget.
    """
    now = [0.0]
    llm = _advancing(streaming_tool_llm([_answer(ANSWER, size=20)]), now, after=after, to=500.0)

    progress, texts, result = _drain(
        _turn(_loop(llm, _documents(QuerySearch({})), clock=lambda: now[0], time_budget_s=120.0))
    )

    assert ("".join(texts), progress, llm.stream_chat.call_count) == (ANSWER, [], 1)
    assert (result.raw_text, result.reason) == (ANSWER, None)


def test_the_context_check_forces_the_answer_before_overflow() -> None:
    """When the room left cannot hold one more full reply, the next round is the tool-less answer."""
    llm = streaming_tool_llm([_calls(_search_call("call_a", "Hafenplan")), _answer()])

    _, texts, result = _drain(_turn(_loop(llm, _documents(QuerySearch({"Hafenplan": long_passages()})), budget=TIGHT)))

    assert llm.stream_chat.call_args_list[1].kwargs["tool_choice"] == "none"
    assert (result.reason, result.steps, "".join(texts)) == ("context_budget", 1, ANSWER)


def test_an_overflow_retries_once_tool_less() -> None:
    """A round the endpoint finds too long is retried once as the answer, every tool reply cut to its share."""
    llm = streaming_tool_llm([_calls(_search_call("call_a", "Hafenplan")), bad_request(OVERFLOW), _answer()])

    _, texts, result = _drain(_turn(_loop(llm, _documents(QuerySearch({"Hafenplan": long_passages()})))))

    assert llm.stream_chat.call_args_list[2].kwargs["tool_choice"] == "none"
    retried = _sent(llm, 2)
    assert retried[-1].content == FINAL_NOTE
    (before,) = _tool_replies(llm, 1).values()
    (after,) = _tool_replies(llm, 2).values()
    assert after == fit_with_marker(before, int(len(before) * OVERFLOW_REPLY_SHARE))[0]
    assert len(after) < len(before)
    assert (result.reason, "".join(texts)) == ("context_budget", ANSWER)


def test_a_rejected_first_call_raises_tools_unavailable_before_any_token(loguru_caplog: LogCaptureFixture) -> None:
    """An endpoint that refuses tool calls is reported before anything is shown, so the caller can answer plainly.

    Args:
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    rejection = bad_request(TOOL_CHOICE_REJECTED)

    shown, raised = _until_raised(_turn(_loop(streaming_tool_llm([rejection]), _documents(QuerySearch({})))))

    assert shown == []
    assert isinstance(raised, ToolsUnavailable)
    assert raised.__cause__ is rejection
    assert "zzsecretzz" not in loguru_caplog.text


def test_a_refused_first_call_is_tools_unavailable_even_when_it_must_answer() -> None:
    """With no tool round allowed, the first call is the answer round; refused as sent, it still reports no tools."""
    rejection = bad_request(TOOL_CHOICE_REJECTED)
    llm = streaming_tool_llm([rejection])

    shown, raised = _until_raised(_turn(_loop(llm, _documents(QuerySearch({})), max_rounds=0)))

    assert (shown, type(raised), raised.__cause__) == ([], ToolsUnavailable, rejection)
    assert llm.stream_chat.call_args.kwargs["tool_choice"] == "none"


def test_two_idle_rounds_force_the_answer() -> None:
    """Two rounds in a row that add no passage make the model answer."""
    llm = streaming_tool_llm(
        [_calls(_search_call("call_a", "Programm")), _calls(_search_call("call_b", "Wetter")), _answer()]
    )

    progress, texts, result = _drain(_turn(_loop(llm, _documents(QuerySearch({})))))

    assert llm.stream_chat.call_args_list[2].kwargs["tool_choice"] == "none"
    assert _steps(progress) == [("search", "Programm", 0), ("search", "Wetter", 0)]
    assert (result.reason, result.steps, "".join(texts)) == ("idle", 2, ANSWER)


@pytest.mark.parametrize(
    ("max_rounds", "reason"), [(6, "context_budget"), (0, "round_budget")], ids=["by-choice", "made-to-answer"]
)
def test_an_answer_cut_at_the_length_limit_is_kept_and_flagged(
    max_rounds: int, reason: str, loguru_caplog: LogCaptureFixture
) -> None:
    """An answer the endpoint cut at its length limit stays as shown, but the turn says evidence was cut short.

    Args:
        max_rounds (int): Tool rounds allowed; with none, the answer round is forced and has a reason already.
        reason (str): The reason the turn reports: ``context_budget``, unless an earlier one stands.
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    cut = ANSWER[:90]
    llm = streaming_tool_llm([[*text_chunks(cut, 30), chunk(finish_reason="length")]])

    _, texts, result = _drain(_turn(_loop(llm, _documents(QuerySearch({})), max_rounds=max_rounds)))

    assert ("".join(texts), result.raw_text, result.reason) == (cut, cut, reason)
    assert loguru_caplog.messages == [f"An agent turn's answer was cut at the length limit ({reason})"]


def test_the_sentinel_is_swapped_for_the_label() -> None:
    """The answer shows the localized label where the model wrote the sentinel; the result keeps the sentinel."""
    judged = (
        "Die Quellen nennen Sonntag als neuen Termin des Hafenfests [1].\n\n"
        f"{SENTINEL}\n\nDie Verschiebung wirkt vernünftig, weil eine Sturmwarnung vorlag."
    )
    llm = streaming_tool_llm([_answer(judged, size=7)])

    _, texts, result = _drain(_turn(_loop(llm, _documents(QuerySearch({})))))

    assert "".join(texts) == judged.replace(SENTINEL, assessment_label(with_rule=True))
    assert result.raw_text == judged


def test_history_is_sent_as_real_messages() -> None:
    """The recent exchanges reach the model as user and assistant messages, after the system prompt."""
    window = ConversationWindow(
        summary="Es ging um das Hafenfest.",
        earlier_questions=("Wer organisiert das Hafenfest?",),
        recent=(
            ConversationTurn("Wann ist das Hafenfest?", "Am Sonntag."),
            ConversationTurn("Und bei Regen?", "Es findet trotzdem statt."),
        ),
    )
    llm = streaming_tool_llm([_answer()])

    _drain(_turn(_loop(llm, _documents(QuerySearch({}))), window=window))

    assert [(message.role, message.content) for message in _sent(llm, 0)] == [
        (MessageRole.SYSTEM, SYSTEM_PROMPT),
        (MessageRole.USER, "Wann ist das Hafenfest?"),
        (MessageRole.ASSISTANT, "Am Sonntag."),
        (MessageRole.USER, "Und bei Regen?"),
        (MessageRole.ASSISTANT, "Es findet trotzdem statt."),
        (MessageRole.USER, QUESTION),
    ]


def test_extra_calls_in_a_round_are_skipped_with_a_reply() -> None:
    """Calls past the round's limit never run, and each is told so under its own id."""
    search = QuerySearch({"Programm": [programme_passage()], "Wetter": [weather_passage()]})
    llm = streaming_tool_llm(
        [
            _calls(
                _search_call("call_a", "Programm"), _search_call("call_b", "Wetter"), _search_call("call_c", "Regatta")
            ),
            _answer(),
        ]
    )

    progress, _, _ = _drain(_turn(_loop(llm, _documents(search), max_calls_per_round=2)))

    assert sorted(search.queries) == ["Programm", "Wetter"]
    replies = _tool_replies(llm, 1)
    assert list(replies) == ["call_a", "call_b", "call_c"]
    assert replies["call_c"] == error_reply("search", SKIPPED_REPLY.format(limit=2)).text
    assert _steps(progress) == [("search", "Programm", 1), ("search", "Wetter", 1)]


# --- Model slots: the first round falls back, a later one still ends the turn ----------------


@pytest.mark.parametrize("max_rounds", [6, 0], ids=["tool-round", "answer-round"])
@pytest.mark.parametrize(("budget", "wait"), [(120.0, MODEL_SLOT_WAIT_S), (2.0, 2.0)])
def test_no_model_slot_for_the_first_round_raises_before_anything_is_yielded(
    budget: float, wait: float, max_rounds: int
) -> None:
    """The first round waits for a model slot no longer than the budget allows; with none, the caller falls back.

    Args:
        budget (float): The turn's time budget in seconds.
        wait (float): The longest the first round may wait for a slot.
        max_rounds (int): Tool rounds allowed; with none, the first round is the answer round.
    """
    slots = _Recording()
    llm = streaming_tool_llm([_answer()])
    loop = _loop(
        llm,
        _documents(QuerySearch({})),
        model_slots=slots,
        time_budget_s=budget,
        clock=lambda: 0.0,
        max_rounds=max_rounds,
    )

    shown, raised = _until_raised(_turn(loop))

    assert (shown, type(raised), slots.waits) == ([], NoSlot, [wait])
    llm.stream_chat.assert_not_called()


def test_a_later_round_without_a_model_slot_answers_from_what_was_found() -> None:
    """A later round that finds no model slot ends research; the answer round waits for one within the time left."""
    slots = _Granting(True, False, True)
    llm = streaming_tool_llm([_calls(_search_call("call_a", "Programm")), _answer()])

    progress, texts, result = _drain(
        _turn(
            _loop(
                llm,
                _documents(QuerySearch({"Programm": [programme_passage()]})),
                model_slots=slots,
                clock=lambda: 0.0,
            )
        )
    )

    assert slots.waits == [MODEL_SLOT_WAIT_S, MODEL_SLOT_WAIT_S, 120.0]
    assert llm.stream_chat.call_args_list[1].kwargs["tool_choice"] == "none"
    assert _steps(progress) == [("search", "Programm", 1)]
    assert (result.reason, result.steps, "".join(texts)) == ("research_failed", 1, ANSWER)


@pytest.mark.parametrize(
    ("preface", "answer_wait"), [(PREFACE, MODEL_SLOT_WAIT_S), ("", 0.0)], ids=["started", "nothing-shown"]
)
def test_an_answer_round_waits_at_least_the_usual_slot_wait_once_the_time_is_up(
    preface: str, answer_wait: float
) -> None:
    """With the time spent, a started turn's answer round still gets the usual wait for a model slot.

    A turn that has shown nothing waits no longer than its time left, as its first round does:
    without a slot it can still be answered another way.

    Args:
        preface (str): What the tool round writes before its call; its note is the one step shown.
        answer_wait (float): The longest the answer round may wait for a slot.
    """
    slots = _Granting()
    now = [0.0]
    llm = _advancing(
        streaming_tool_llm([_calls(_search_call("call_a", "Programm"), text=preface), _answer()]),
        now,
        after=0,
        to=500.0,
    )

    _, _, result = _drain(
        _turn(
            _loop(
                llm,
                _documents(QuerySearch({"Programm": [programme_passage()]})),
                model_slots=slots,
                clock=lambda: now[0],
            )
        )
    )

    assert slots.waits == [MODEL_SLOT_WAIT_S, answer_wait]
    assert result.reason == "time_budget"


def test_a_started_turn_without_a_model_slot_for_its_answer_raises() -> None:
    """An answer round that gets no model slot fails the turn as any failed answer does: no result passes as one."""
    slots = _Granting(True, False, False)
    llm = streaming_tool_llm([_calls(_search_call("call_a", "Programm"))])
    loop = _loop(
        llm,
        _documents(QuerySearch({"Programm": [programme_passage()]})),
        model_slots=slots,
        clock=lambda: 0.0,
    )

    shown, raised = _until_raised(_turn(loop))

    assert (type(raised), llm.stream_chat.call_count) == (NoSlot, 1)
    assert _steps(item for item in shown if isinstance(item, AgentProgress)) == [("search", "Programm", 1)]
    assert not any(isinstance(item, (str, AgentResult)) for item in shown)
    # The answer round still waits the time left, at least the usual wait, before it gives up.
    assert slots.waits == [MODEL_SLOT_WAIT_S, MODEL_SLOT_WAIT_S, 120.0]


def test_a_round_holds_its_model_slot_while_it_streams_and_gives_it_back() -> None:
    """The model slot is taken for the round and given back when it ends, or when the caller stops reading."""
    slots = threading.BoundedSemaphore(1)
    turn = _turn(_loop(streaming_tool_llm([_answer(ANSWER, size=20)]), _documents(QuerySearch({})), model_slots=slots))

    first = next(turn)
    held_while_streaming = not slots.acquire(blocking=False)
    rest = list(turn)

    assert isinstance(first, str) and held_while_streaming
    assert isinstance(rest[-1], AgentResult)
    assert slots.acquire(blocking=False)
    slots.release()

    stopped = cast(
        Generator[Any, None, None],
        _turn(_loop(streaming_tool_llm([_answer(ANSWER, size=20)]), _documents(QuerySearch({})), model_slots=slots)),
    )
    next(stopped)
    stopped.close()
    assert slots.acquire(blocking=False)
    slots.release()


# --- Search slots and the deadline, ported from the research loop ----------------------------


def test_a_search_that_outlives_the_time_left_keeps_its_slot_until_it_ends(
    held: threading.Event, loguru_caplog: LogCaptureFixture
) -> None:
    """An abandoned search is answered on time, admits nothing, and holds its slot until retrieval ends.

    Args:
        held (threading.Event): Holds the search.
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    now = [0.0]
    # Taking the slot spends the turn's whole time, so the search is waited on with none left.
    slots = _SlowSlots(now, 100.0)

    def _search(query: str) -> list[NodeWithScore]:
        """Wait for the release, then find a passage too late.

        Args:
            query (str): What the tool searched for.

        Returns:
            list[NodeWithScore]: The late passage.
        """
        held.wait(timeout=30)
        return [programme_passage()]

    ledger = EvidenceLedger(cap=MAX_LEDGER_ENTRIES)
    llm = streaming_tool_llm([_calls(_search_call("call_a", "zzsecretzz Programm")), _answer()])

    _drain(
        _turn(
            _loop(llm, _documents(_search), search_slots=slots, time_budget_s=100.0, clock=lambda: now[0]),
            ledger=ledger,
        )
    )

    assert _tool_replies(llm, 1) == {"call_a": error_reply("search", LATE_REPLY).text}
    assert len(ledger) == 0
    assert slots.acquire(blocking=False) is False
    held.set()
    join_threads(SEARCH_THREAD_NAME)
    assert slots.acquire(blocking=False) is True
    slots.release()
    assert len(ledger) == 0
    assert "zzsecretzz" not in loguru_caplog.text


def _alive_searches() -> int:
    """Count the agent search threads still running.

    Returns:
        int: Threads named ``SEARCH_THREAD_NAME`` that are alive.
    """
    return sum(thread.name == SEARCH_THREAD_NAME and thread.is_alive() for thread in threading.enumerate())


def test_abandoned_searches_are_capped_at_the_slot_count(held: threading.Event) -> None:
    """Hung searches across turns pile up only until every slot is held; a call after that never starts.

    Args:
        held (threading.Event): Holds every search.
    """
    calls: list[str] = []

    def _search(query: str) -> list[NodeWithScore]:
        """Count the search, then hang until the test ends.

        Args:
            query (str): What the tool searched for.

        Returns:
            list[NodeWithScore]: The late passage.
        """
        calls.append(query)
        held.wait(timeout=30)
        return [programme_passage()]

    replies: list[str] = []
    for turn in range(SEARCH_SLOTS + 1):
        llm = streaming_tool_llm([_calls(_search_call("call_a", f"Hafenfest {turn}")), _answer()])
        # A fixed clock: only the real waits spend the time left. They are short, and the outcome does
        # not hang on them: a held search never answers, and the ninth call never finds a slot.
        _drain(_turn(_loop(llm, _documents(_search), time_budget_s=0.05, clock=lambda: 0.0)))
        replies.extend(_tool_replies(llm, 1).values())

    assert replies == [error_reply("search", LATE_REPLY).text] * SEARCH_SLOTS + [
        error_reply("search", NO_CAPACITY_REPLY).text
    ]
    assert len(calls) == SEARCH_SLOTS
    assert _alive_searches() == SEARCH_SLOTS


@pytest.mark.parametrize(("budget", "wait"), [(120.0, SEARCH_SLOT_WAIT_S), (2.0, 2.0)])
def test_no_free_search_slot_answers_the_call_without_starting_it(
    budget: float, wait: float, monkeypatch: pytest.MonkeyPatch, loguru_caplog: LogCaptureFixture
) -> None:
    """A call waits for a search slot no longer than the time left allows, and never starts a thread without one.

    Args:
        budget (float): The turn's time budget in seconds.
        wait (float): The longest the call may wait for a slot.
        monkeypatch (pytest.MonkeyPatch): Records thread starts.
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    slots = _Recording()
    search = QuerySearch({})
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
    llm = streaming_tool_llm([_calls(_search_call("call_a", "zzsecretzz Programm")), _answer()])

    progress, _, result = _drain(
        _turn(_loop(llm, _documents(search), search_slots=slots, time_budget_s=budget, clock=lambda: 0.0))
    )

    assert slots.waits == [wait]
    assert (search.queries, started) == ([], [])
    assert _tool_replies(llm, 1) == {"call_a": error_reply("search", NO_CAPACITY_REPLY).text}
    assert (_steps(progress), result.reason) == ([], "research_failed")
    assert loguru_caplog.messages == ["No search slot came free for 1 of an agent round's tool calls"]


def test_a_search_thread_that_cannot_start_is_answered_with_an_error(
    monkeypatch: pytest.MonkeyPatch, loguru_caplog: LogCaptureFixture
) -> None:
    """A process out of threads fails the one call: it is told so, its slot comes back and the turn goes on.

    Args:
        monkeypatch (pytest.MonkeyPatch): Makes the search thread's start fail.
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    slots = threading.BoundedSemaphore(1)
    real_start = threading.Thread.start

    def _start(thread: threading.Thread) -> None:
        """Fail a search thread's start as a process out of threads does; start any other thread.

        Args:
            thread (threading.Thread): The thread.

        Raises:
            RuntimeError: For a search thread.
        """
        if thread.name == SEARCH_THREAD_NAME:
            raise RuntimeError("can't start new thread")
        real_start(thread)

    monkeypatch.setattr(threading.Thread, "start", _start)
    llm = streaming_tool_llm([_calls(_search_call("call_a", "Programm")), _answer()])

    _, texts, result = _drain(_turn(_loop(llm, _documents(QuerySearch({})), search_slots=slots)))

    assert _tool_replies(llm, 1) == {"call_a": error_reply("search", FAILED_REPLY).text}
    assert ("".join(texts), result.reason) == (ANSWER, "research_failed")
    assert loguru_caplog.messages == ["An agent's 'search' call could not start: RuntimeError"]
    assert slots.acquire(blocking=False) is True
    slots.release()


_SCOPE: ContextVar[str | None] = ContextVar("invented_collection_scope", default=None)


def test_a_search_runs_in_the_callers_context() -> None:
    """Retrieval reads the request's collection from a context variable, so a search sees the caller's."""
    seen: list[str | None] = []

    def _search(query: str) -> list[NodeWithScore]:
        """Record the scope the search runs under.

        Args:
            query (str): What the tool searched for.

        Returns:
            list[NodeWithScore]: Nothing.
        """
        seen.append(_SCOPE.get())
        return []

    token = _SCOPE.set("invented-collection")
    try:
        _drain(
            _turn(
                _loop(streaming_tool_llm([_calls(_search_call("call_a", "Programm")), _answer()]), _documents(_search))
            )
        )
    finally:
        _SCOPE.reset(token)

    assert seen == ["invented-collection"]


def test_a_search_raising_its_own_timeout_is_a_failed_call_not_a_spent_budget(
    loguru_caplog: LogCaptureFixture,
) -> None:
    """A retrieval dependency's ``TimeoutError`` is the call failing; only the clock spends the time.

    Args:
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """

    def _search(query: str) -> list[NodeWithScore]:
        """Fail as a timed-out storage call would.

        Args:
            query (str): What the tool searched for.

        Raises:
            TimeoutError: Always.
        """
        raise TimeoutError("invented storage timeout")

    llm = streaming_tool_llm([_calls(_search_call("call_a", "Programm")), _answer()])

    _, _, result = _drain(_turn(_loop(llm, _documents(_search))))

    assert _tool_replies(llm, 1) == {"call_a": error_reply("search", FAILED_REPLY).text}
    assert result.reason == "research_failed"
    assert loguru_caplog.messages == ["An agent's 'search' call failed: TimeoutError"]


def test_the_slot_wait_comes_out_of_the_time_budget(held: threading.Event, monkeypatch: pytest.MonkeyPatch) -> None:
    """A search slot that takes nearly the whole budget to free leaves the search only what remains.

    Args:
        held (threading.Event): Holds the search.
        monkeypatch (pytest.MonkeyPatch): Records how long the round waits for its searches.
    """
    now = [0.0]
    waits: list[float | None] = []
    real_wait = concurrent.futures.wait

    def _recorded_wait(futures: Any, timeout: float | None = None, **kwargs: Any) -> Any:
        """Record the wait asked for, then wait as asked.

        Args:
            futures (Any): The futures waited on.
            timeout (float | None): The longest wait.
            **kwargs (Any): Further options.

        Returns:
            Any: What the real wait returns.
        """
        waits.append(timeout)
        return real_wait(futures, timeout=timeout, **kwargs)

    monkeypatch.setattr(concurrent.futures, "wait", _recorded_wait)

    def _search(query: str) -> list[NodeWithScore]:
        """Hang until the test ends.

        Args:
            query (str): What the tool searched for.

        Returns:
            list[NodeWithScore]: The late passage.
        """
        held.wait(timeout=30)
        return [programme_passage()]

    llm = streaming_tool_llm([_calls(_search_call("call_a", "Programm")), _answer()])

    _drain(
        _turn(
            _loop(
                llm,
                _documents(_search),
                search_slots=_SlowSlots(now, 99.9),
                time_budget_s=100.0,
                clock=lambda: now[0],
            )
        )
    )

    assert waits == [pytest.approx(0.1)]
    assert _tool_replies(llm, 1) == {"call_a": error_reply("search", LATE_REPLY).text}


def test_calls_from_a_reply_after_the_time_budget_are_answered_without_running() -> None:
    """A round that ends after the budget has its searches answered unrun, and the answer comes next."""
    now = [0.0]
    search = QuerySearch({"Programm": [programme_passage()]})
    llm = _advancing(
        streaming_tool_llm([_calls(_search_call("call_a", "Programm"), ("call_b", "read", {"n": 4})), _answer()]),
        now,
        after=0,
        to=200.0,
    )

    progress, texts, result = _drain(_turn(_loop(llm, _documents(search), clock=lambda: now[0])))

    assert search.queries == []
    replies = _tool_replies(llm, 1)
    assert replies["call_a"] == error_reply("search", NO_TIME_REPLY).text
    assert replies["call_b"].startswith("Error: there is no passage [4]")
    assert _steps(progress) == []
    assert llm.stream_chat.call_args_list[1].kwargs["tool_choice"] == "none"
    assert (result.reason, "".join(texts)) == ("time_budget", ANSWER)


def test_a_search_back_before_the_deadline_keeps_its_passage_when_the_time_then_runs_out() -> None:
    """A search that came back in time is admitted even if the time runs out while its reply is written."""
    now = [0.0]

    def _slow_view(node: Any) -> Any:
        """Let the time run out while the reply renders, then hide what the test view hides.

        Args:
            node (Any): A ledger node.

        Returns:
            Any: The view of it.
        """
        now[0] = 300.0
        return llm_view(node)

    llm = streaming_tool_llm([_calls(_search_call("call_a", "Programm")), _answer()])

    progress, _, result = _drain(
        _turn(
            _loop(
                llm,
                _documents(QuerySearch({"Programm": [programme_passage()]})),
                clock=lambda: now[0],
                llm_view=_slow_view,
            )
        )
    )

    assert _steps(progress) == [("search", "Programm", 1)]
    assert [scored.node.node_id for scored in result.nodes] == ["programme"]
    assert llm.stream_chat.call_args_list[1].kwargs["tool_choice"] == "none"
    assert result.reason == "time_budget"


# --- How a failed round ends the turn ----------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "raised"),
    [
        (bad_request(OVERFLOW), ValueError),
        (bad_request("The decoder prompt (length 9001) is longer than the maximum model length of 8192"), ValueError),
        (bad_request(TOOL_CHOICE_REJECTED), ToolsUnavailable),
        (_status_error(openai.UnprocessableEntityError, 422, "tools: unsupported field"), ToolsUnavailable),
        (_status_error(openai.RateLimitError, 429, "rate limit reached"), openai.RateLimitError),
        (RuntimeError("invented outage"), RuntimeError),
    ],
    ids=["overflow", "decoder-overflow", "tools-400", "tools-422", "rate-limit", "outage"],
)
def test_a_failed_first_round_raises_before_anything_is_shown(error: Exception, raised: type[Exception]) -> None:
    """Nothing is shown before the first round, so its failure goes to the caller, named so it can fall back.

    An overflow there has no tool reply to shorten, so it is not retried.

    Args:
        error (Exception): What the endpoint raises.
        raised (type[Exception]): What the turn raises.
    """
    llm = streaming_tool_llm([error, _answer()])

    shown, exc = _until_raised(_turn(_loop(llm, _documents(QuerySearch({})))))

    assert (shown, type(exc), llm.stream_chat.call_count) == ([], raised, 1)
    if raised is ValueError:
        assert "context window" in str(exc)


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (bad_request(OVERFLOW), "context_budget"),
        (bad_request(TOOL_CHOICE_REJECTED), "research_failed"),
        (_status_error(openai.RateLimitError, 429, "rate limit reached"), "research_failed"),
        (RuntimeError("invented outage"), "research_failed"),
        ([chunk(tool_calls=[tool_call_delta(index=0, call_id="call_b")]), AssertionError()], "research_failed"),
    ],
    ids=["overflow", "rejected", "rate-limit", "outage", "broken-mid-stream"],
)
def test_a_later_failed_round_answers_from_the_evidence(error: Any, reason: str) -> None:
    """Once the turn has shown a step, a failed round is not the end: the model answers from what was found.

    Args:
        error (Any): What the second round raises, on its first pull or partway through.
        reason (str): Why the turn says it stopped.
    """
    llm = streaming_tool_llm([_calls(_search_call("call_a", "Hafenplan")), error, _answer()])
    search = QuerySearch({"Hafenplan": long_passages(1)})

    progress, texts, result = _drain(_turn(_loop(llm, _documents(search))))

    assert llm.stream_chat.call_args_list[2].kwargs["tool_choice"] == "none"
    assert _steps(progress) == [("search", "Hafenplan", 1)]
    assert (result.reason, result.steps, "".join(texts)) == (reason, 1, ANSWER)
    assert [scored.node.node_id for scored in result.nodes] == ["plan-1"]


def test_a_second_overflow_raises_naming_the_context_window() -> None:
    """The retry is the only one: a retried answer that still does not fit ends the turn with the overflow."""
    llm = streaming_tool_llm(
        [_calls(_search_call("call_a", "Hafenplan")), bad_request(OVERFLOW), bad_request(OVERFLOW)]
    )

    shown, exc = _until_raised(_turn(_loop(llm, _documents(QuerySearch({"Hafenplan": long_passages()})))))

    assert isinstance(exc, ValueError) and "context window" in str(exc)
    assert [type(item) for item in shown] == [AgentProgress]
    assert llm.stream_chat.call_count == 3


def test_a_round_that_breaks_off_with_its_text_showing_is_withdrawn_and_answered_anew() -> None:
    """Text a round showed before it broke off was provisional: a note withdraws it, and the model answers again."""
    llm = streaming_tool_llm([[*text_chunks(ANSWER, 20)[:7], RuntimeError("invented outage")], _answer(SHORT_ANSWER)])

    items = list(_turn(_loop(llm, _documents(QuerySearch({})))))

    (note,) = [item for item in items if isinstance(item, AgentProgress)]
    position = items.index(note)
    withdrawn = "".join(cast(list[str], items[:position]))
    assert (note.step, note.action, note.detail) == (1, "note", " ".join(withdrawn.split()))
    assert len(withdrawn) >= 120
    assert "".join(cast(list[str], items[position + 1 : -1])) == SHORT_ANSWER
    assert llm.stream_chat.call_args_list[1].kwargs["tool_choice"] == "none"
    result = items[-1]
    assert isinstance(result, AgentResult)
    assert (result.raw_text, result.reason, result.steps) == (SHORT_ANSWER, "research_failed", 0)
    assert result.actions == [{"action": "note", "detail": note.detail, "found": 0}]


@pytest.mark.parametrize("path", ["first-round-withdrawn", "no-tool-round-allowed"])
def test_the_final_note_joins_a_user_message_it_would_follow(path: str) -> None:
    """Strict chat templates refuse two user messages in a row, so the final note joins the user's message.

    Args:
        path (str): How the answer round comes to follow the user's message directly: the first
            round broke off and was withdrawn, or no tool round was allowed.
    """
    if path == "first-round-withdrawn":
        broken = [*text_chunks(ANSWER, 20)[:7], RuntimeError("invented outage")]
        llm = streaming_tool_llm([broken, _answer(SHORT_ANSWER)])
        loop = _loop(llm, _documents(QuerySearch({})))
    else:
        llm = streaming_tool_llm([_answer(SHORT_ANSWER)])
        loop = _loop(llm, _documents(QuerySearch({})), max_rounds=0)

    _drain(_turn(loop))

    sent = _sent(llm, llm.stream_chat.call_count - 1)
    roles = [message.role for message in sent]
    assert roles == [MessageRole.SYSTEM, MessageRole.USER]
    wire = to_openai_message_dicts(sent)
    assert isinstance(wire, list)
    assert wire[-1] == {"role": "user", "content": f"{QUESTION}\n\n{FINAL_NOTE}"}


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("invented outage"),
        _status_error(openai.RateLimitError, 429, "rate limit reached"),
        bad_request(TOOL_CHOICE_REJECTED),
    ],
    ids=["outage", "rate-limit", "refused-after-the-first-call"],
)
def test_an_answer_round_that_fails_before_any_token_raises_after_the_steps(error: Exception) -> None:
    """A started turn whose answer round fails at once, with no overflow and no first-call refusal, raises; no result.

    Args:
        error (Exception): What the answer round's first pull raises.
    """
    llm = streaming_tool_llm([_calls(_search_call("call_a", "Programm")), error])
    loop = _loop(llm, _documents(QuerySearch({"Programm": [programme_passage()]})), max_rounds=1)

    shown, raised = _until_raised(_turn(loop))

    assert raised is error
    assert _steps(item for item in shown if isinstance(item, AgentProgress)) == [("search", "Programm", 1)]
    assert not any(isinstance(item, (str, AgentResult)) for item in shown)


def test_an_answer_round_that_breaks_off_raises_after_its_text() -> None:
    """An answer the endpoint stops sending partway is not passed off as whole: the turn raises."""
    broken = [*text_chunks(ANSWER, 20)[:7], RuntimeError("invented outage")]
    llm = streaming_tool_llm([broken, [*broken[:-1], RuntimeError("invented second outage")]])

    shown, exc = _until_raised(_turn(_loop(llm, _documents(QuerySearch({})))))

    assert (type(exc), str(exc)) == (RuntimeError, "invented second outage")
    assert [type(item) for item in shown if not isinstance(item, str)] == [AgentProgress]
    assert isinstance(shown[-1], str)


# --- Context room ---------------------------------------------------------------------------


def test_a_call_without_room_for_its_reply_is_never_run() -> None:
    """A call is run only while one more full reply fits; past that it is told there is no room."""
    search = QuerySearch({"Programm": [programme_passage()], "Wetter": [weather_passage()]})
    llm = streaming_tool_llm([_calls(_search_call("call_a", "Programm"), _search_call("call_b", "Wetter")), _answer()])

    progress, _, _ = _drain(_turn(_loop(llm, _documents(search), budget=TIGHT)))

    assert search.queries == ["Programm"]
    replies = _tool_replies(llm, 1)
    assert shown_numbers(replies["call_a"]) == [1]
    assert replies["call_b"] == error_reply("search", NO_ROOM_REPLY).text
    assert _steps(progress) == [("search", "Programm", 1)]


def _budget_with_room(round_room: int) -> TurnBudget:
    """Build an 8k budget whose room for tool replies is ``round_room`` tokens.

    Args:
        round_room (int): The room wanted.

    Returns:
        TurnBudget: The budget, its system prompt sized to leave that room.
    """
    budget = turn_budget()
    return TurnBudget(
        ctx_window=budget.ctx_window,
        reasoning=False,
        char_token_ratio=budget.char_token_ratio,
        system_tokens=budget.system_tokens + budget.round_room - round_room,
        tool_schema_tokens=budget.tool_schema_tokens,
        history_tokens=budget.history_tokens,
        message_tokens=budget.message_tokens,
    )


def test_the_answer_rounds_note_is_kept_out_of_the_room() -> None:
    """The room for tool replies leaves out the final note: a note leaving no room for a reply means no tool round."""
    llm = streaming_tool_llm([_answer()])

    _, texts, result = _drain(_turn(_loop(llm, _documents(QuerySearch({})), budget=TIGHT, final_note=FINAL_NOTE * 30)))

    assert llm.stream_chat.call_args.kwargs["tool_choice"] == "none"
    assert (llm.stream_chat.call_count, result.reason, "".join(texts)) == (1, "context_budget", ANSWER)


def test_a_rounds_own_message_counts_against_the_room() -> None:
    """The assistant message asking for the calls is part of the transcript the room must hold."""
    search = QuerySearch({"Programm": [programme_passage()]})
    llm = streaming_tool_llm([_calls(_search_call("call_a", "Programm"), text=LONG_PREFACE), _answer()])

    _, _, result = _drain(_turn(_loop(llm, _documents(search), budget=_budget_with_room(1240))))

    assert search.queries == ["Programm"]
    assert llm.stream_chat.call_args_list[1].kwargs["tool_choice"] == "none"
    assert result.reason == "context_budget"


def test_every_calls_reply_is_reserved_before_any_call_runs() -> None:
    """A round's calls each need a reply, so their replies are set aside before a call may take a full one."""
    search = QuerySearch({"Programm": [programme_passage()]})
    unoffered = [(f"call_{number}", "keyframes", {"file": "harbour_walk.webm"}) for number in range(12)]
    llm = streaming_tool_llm([_calls(_search_call("call_a", "Programm"), *unoffered), _answer()])

    _drain(_turn(_loop(llm, _documents(search), budget=TIGHT, max_calls_per_round=16)))

    assert search.queries == []
    assert _tool_replies(llm, 1)["call_a"] == error_reply("search", NO_ROOM_REPLY).text


@pytest.mark.parametrize(("count", "withdrawn"), [(20, True), (2, False)], ids=["too-many", "fits"])
def test_a_round_whose_own_calls_do_not_fit_the_room_is_withdrawn(count: int, withdrawn: bool) -> None:
    """A round's calls, echoed and each given at least a short reply, must fit the room left, or the round is withdrawn.

    A withdrawn round's calls never reach the transcript, so they need no replies; it is no tool
    round, and the model answers next for want of room.

    Args:
        count (int): How many calls the round makes, each to a tool the turn does not offer.
        withdrawn (bool): Whether they are too many for the room left.
    """
    unoffered = [(f"call_{number}", "keyframes", {"file": "harbour_walk.webm"}) for number in range(count)]
    llm = streaming_tool_llm([_calls(*unoffered, text=PREFACE), _answer()])

    progress, texts, result = _drain(_turn(_loop(llm, _documents(QuerySearch({})), budget=TIGHT)))

    kept = [message for message in _sent(llm, 1) if message.role in (MessageRole.ASSISTANT, MessageRole.TOOL)]
    assert progress == [AgentProgress(step=1, action="note", detail=PREFACE, found=0)]
    assert "".join(texts) == ANSWER
    if withdrawn:
        assert kept == []
        assert llm.stream_chat.call_args_list[1].kwargs["tool_choice"] == "none"
        assert (result.reason, result.steps) == ("context_budget", 0)
    else:
        assert len(kept) == 1 + count
        assert llm.stream_chat.call_args_list[1].kwargs["tool_choice"] == "auto"
        assert (result.reason, result.steps) == (None, 1)


def test_a_round_that_adds_evidence_resets_the_idle_count() -> None:
    """Only rounds in a row count as idle: one that adds a passage starts the count again."""
    search = QuerySearch({"Programm": [programme_passage()]})
    llm = streaming_tool_llm(
        [
            _calls(_search_call("call_a", "Regatta")),
            _calls(_search_call("call_b", "Programm")),
            _calls(_search_call("call_c", "Feuerwerk")),
            _answer(),
        ]
    )

    _, _, result = _drain(_turn(_loop(llm, _documents(search))))

    assert llm.stream_chat.call_args_list[3].kwargs["tool_choice"] == "auto"
    assert (result.reason, result.steps) == (None, 3)


class _Verbose:
    """A tool whose reply runs past any reply length, as a reply that lists many clips could."""

    def __init__(self) -> None:
        """Describe the tool."""
        self.name = "verbose"
        self.description = "Answer at length."
        self.parameters: dict[str, Any] = {"type": "object", "properties": {}}

    def prepare(self, arguments: Mapping[str, Any], ctx: ToolContext) -> ToolReply | ToolWork:
        """Answer with a reply far longer than one reply may be.

        Args:
            arguments (Mapping[str, Any]): Unused.
            ctx (ToolContext): Unused.

        Returns:
            ToolReply | ToolWork: The work, which fetches nothing.
        """

        def _answer_long(fetched: Any, context: ToolContext) -> ToolReply:
            """Build the overlong reply.

            Args:
                fetched (Any): Nothing.
                context (ToolContext): Unused.

            Returns:
                ToolReply: Ten thousand characters.
            """
            return ToolReply(text="Hafen " * 2000, found=0, action="verbose", detail="")

        return ToolWork(detail="", fetch=None, answer=_answer_long)


def test_a_reply_longer_than_one_reply_may_be_is_cut(loguru_caplog: LogCaptureFixture) -> None:
    """No tool reply enters the transcript longer than ``result_chars``, whatever the tool wrote.

    Args:
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    llm = streaming_tool_llm([_calls(("call_a", "verbose", {})), _answer()])

    _drain(_turn(_loop(llm, ToolRegistry([_Verbose()]))))

    (reply,) = _tool_replies(llm, 1).values()
    assert len(reply) <= turn_budget().result_chars
    assert reply.endswith("characters not shown ...]")
    assert loguru_caplog.messages == ["A 'verbose' tool reply ran past 3000 characters and was cut"]


# --- The trail and the transcript -------------------------------------------------------------


def _failing_on(query: str, results: Mapping[str, list[NodeWithScore]]) -> SearchFn:
    """Build a search that fails on one query and answers every other one.

    Args:
        query (str): The query whose search fails, as a storage outage would.
        results (Mapping[str, list[NodeWithScore]]): Passages per other query.

    Returns:
        SearchFn: The search.
    """
    answering = QuerySearch(results)

    def _search(asked: str) -> list[NodeWithScore]:
        """Fail on the one query, else answer as the double does.

        Args:
            asked (str): What the tool searched for.

        Returns:
            list[NodeWithScore]: The passages for the query.

        Raises:
            RuntimeError: For the failing query.
        """
        if asked == query:
            raise RuntimeError("invented storage outage")
        return answering(asked)

    return _search


@pytest.mark.parametrize("kind", ["refused", "skipped", "failed", "argument-error"])
def test_a_call_its_tool_did_not_answer_gets_its_reply_but_no_step(kind: str) -> None:
    """A step means the tool answered: a call refused for room, skipped, failed or sent bad arguments shows none.

    Args:
        kind (str): Why the round's second call went unanswered.
    """
    second = _search_call("call_b", "" if kind == "argument-error" else "Wetter")
    results = {"Programm": [programme_passage()], "Wetter": [weather_passage()]}
    search = _failing_on("Wetter", results) if kind == "failed" else QuerySearch(results)
    overrides: dict[str, Any] = {"refused": {"budget": TIGHT}, "skipped": {"max_calls_per_round": 1}}.get(kind, {})
    llm = streaming_tool_llm([_calls(_search_call("call_a", "Programm"), second), _answer()])

    progress, _, result = _drain(_turn(_loop(llm, _documents(search), **overrides)))

    assert _tool_replies(llm, 1)["call_b"].startswith("Error: ")
    assert _steps(progress) == [("search", "Programm", 1)]
    assert result.actions == [{"action": "search", "detail": "Programm", "found": 1}]


@pytest.mark.parametrize("cause", ["failed", "no-slot"])
def test_a_call_that_failed_marks_the_turn_research_failed_and_it_still_answers(cause: str) -> None:
    """Evidence may be missing when a search failed or found no slot; the turn says so, and the model answers by itself.

    Args:
        cause (str): Why the search returned nothing: it failed, or no search slot came free.
    """
    search = _failing_on("Programm", {}) if cause == "failed" else QuerySearch({"Programm": [programme_passage()]})
    overrides: dict[str, Any] = {"search_slots": _Recording()} if cause == "no-slot" else {}
    llm = streaming_tool_llm([_calls(_search_call("call_a", "Programm")), _answer()])

    _, texts, result = _drain(_turn(_loop(llm, _documents(search), **overrides)))

    assert llm.stream_chat.call_args_list[1].kwargs["tool_choice"] == "auto"
    assert (result.reason, result.steps, "".join(texts)) == ("research_failed", 1, ANSWER)


class _Brittle:
    """A tool whose answer fails after its fetch returned, as a renderer that raises would."""

    def __init__(self) -> None:
        """Describe the tool."""
        self.name = "brittle"
        self.description = "Fail while answering."
        self.parameters: dict[str, Any] = {"type": "object", "properties": {}}

    def prepare(self, arguments: Mapping[str, Any], ctx: ToolContext) -> ToolReply | ToolWork:
        """Fetch nothing, then fail to answer.

        Args:
            arguments (Mapping[str, Any]): Unused.
            ctx (ToolContext): Unused.

        Returns:
            ToolReply | ToolWork: The work.
        """

        def _fail(fetched: Any, context: ToolContext) -> ToolReply:
            """Fail as a broken renderer would.

            Args:
                fetched (Any): What the fetch returned.
                context (ToolContext): Unused.

            Raises:
                RuntimeError: Always.
            """
            raise RuntimeError("invented rendering failure")

        return ToolWork(detail="", fetch=list, answer=_fail)


def test_a_call_whose_answer_fails_gets_its_reply_and_marks_research_failed(
    loguru_caplog: LogCaptureFixture,
) -> None:
    """A tool that fails while answering, after its fetch returned, is a failed call like a failed search.

    Args:
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    llm = streaming_tool_llm([_calls(("call_a", "brittle", {})), _answer()])

    progress, _, result = _drain(_turn(_loop(llm, ToolRegistry([_Brittle()]))))

    assert _tool_replies(llm, 1) == {"call_a": error_reply("brittle", FAILED_REPLY).text}
    assert (progress, result.reason) == ([], "research_failed")
    assert loguru_caplog.messages == ["An agent's 'brittle' call failed: RuntimeError"]


class _Unpreparable:
    """A tool whose ``prepare`` raises, as a tool with a bug in its argument check would."""

    def __init__(self) -> None:
        """Describe the tool."""
        self.name = "unpreparable"
        self.description = "Fail while checking the arguments."
        self.parameters: dict[str, Any] = {"type": "object", "properties": {}}

    def prepare(self, arguments: Mapping[str, Any], ctx: ToolContext) -> ToolReply | ToolWork:
        """Fail.

        Args:
            arguments (Mapping[str, Any]): Unused.
            ctx (ToolContext): Unused.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("invented check failure")


def test_a_call_whose_prepare_raises_gets_its_reply_and_the_turn_answers(loguru_caplog: LogCaptureFixture) -> None:
    """A call that cannot even be prepared is answered as a failed call; the started turn goes on and answers.

    Args:
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    registry = ToolRegistry([SearchTool(QuerySearch({"Programm": [programme_passage()]})), _Unpreparable()])
    llm = streaming_tool_llm(
        [_calls(_search_call("call_a", "Programm")), _calls(("call_b", "unpreparable", {})), _answer()]
    )

    progress, texts, result = _drain(_turn(_loop(llm, registry)))

    assert _tool_replies(llm, 2)["call_b"] == error_reply("unpreparable", FAILED_REPLY).text
    assert _steps(progress) == [("search", "Programm", 1)]
    assert llm.stream_chat.call_args_list[2].kwargs["tool_choice"] == "auto"
    assert (result.reason, "".join(texts)) == ("research_failed", ANSWER)
    assert loguru_caplog.messages == ["An agent tool call could not be prepared: RuntimeError"]


def test_arguments_nested_too_deeply_get_an_argument_error_and_no_step() -> None:
    """Arguments that read as JSON but nest too deeply to compare are the model's mistake: an error reply, no crash."""
    nested = '{"n": ' + "[" * 600 + "]" * 600 + "}"
    llm = streaming_tool_llm([_calls(("call_a", "read", nested)), _answer()])

    progress, texts, result = _drain(_turn(_loop(llm, _documents(QuerySearch({})))))

    assert _tool_replies(llm, 1)["call_a"].startswith("Error: the arguments for read")
    assert (progress, result.reason, "".join(texts)) == ([], None, ANSWER)


def test_a_failed_calls_reason_is_kept_when_a_limit_ends_the_turn() -> None:
    """The first cause stands: a limit reached after a failed call does not replace ``research_failed``."""
    llm = streaming_tool_llm([_calls(_search_call("call_a", "Programm")), _answer()])

    _, _, result = _drain(_turn(_loop(llm, _documents(_failing_on("Programm", {})), max_rounds=1)))

    assert llm.stream_chat.call_args_list[1].kwargs["tool_choice"] == "none"
    assert result.reason == "research_failed"


def test_a_memory_hit_and_a_read_still_get_their_step() -> None:
    """A call answered from memory, and a read, are calls their tool answered: each is a step."""
    llm = streaming_tool_llm(
        [
            _calls(_search_call("call_a", "Programm")),
            _calls(_search_call("call_b", "Programm"), ("call_c", "read", {"n": 1})),
            _answer(),
        ]
    )

    progress, _, result = _drain(_turn(_loop(llm, _documents(QuerySearch({"Programm": [programme_passage()]})))))

    assert _steps(progress) == [("search", "Programm", 1), ("search", "Programm", 0), ("read", "1", 0)]
    assert _tool_replies(llm, 2)["call_b"] == "Already called search with these arguments; see [1]."
    assert result.reason is None


def test_a_call_to_a_tool_the_turn_does_not_offer_gets_a_reply_but_no_step() -> None:
    """A call that ran nothing leaves no step in the trail; the model is still answered."""
    llm = streaming_tool_llm([_calls(("call_a", "keyframes", {"file": "harbour_walk.webm"})), _answer()])

    progress, _, result = _drain(_turn(_loop(llm, _documents(QuerySearch({})))))

    assert _tool_replies(llm, 1)["call_a"].startswith("Error: there is no tool named 'keyframes'")
    assert (progress, result.actions, result.steps) == ([], [], 1)


def test_the_transcript_echoes_each_call_and_answers_it_by_id() -> None:
    """The next round reads the calls as the model made them and each reply under its call's id.

    Arguments that are not a JSON object, broken or of another type, are echoed as an empty
    object, which every chat template parses.
    """
    llm = streaming_tool_llm(
        [
            _calls(
                _search_call("call_a", "Programm"),
                ("call_b", "read", "nicht json"),
                ("call_c", "read", "[3]"),
                text=PREFACE,
            ),
            _answer(),
        ]
    )

    _drain(_turn(_loop(llm, _documents(QuerySearch({"Programm": [programme_passage()]})))))

    # The system prompt and the question come first; the round's transcript follows.
    sent = _sent(llm, 1)
    wire = to_openai_message_dicts(sent[2:])
    assert isinstance(wire, list)
    assistant, *replies = (cast(dict[str, Any], message) for message in wire)
    assert (assistant["role"], assistant["content"]) == ("assistant", PREFACE)
    assert [
        (call["id"], call["function"]["name"], call["function"]["arguments"]) for call in assistant["tool_calls"]
    ] == [("call_a", "search", '{"query": "Programm"}'), ("call_b", "read", "{}"), ("call_c", "read", "{}")]
    assert [(reply["role"], reply["tool_call_id"]) for reply in replies] == [
        ("tool", "call_a"),
        ("tool", "call_b"),
        ("tool", "call_c"),
    ]
    assert [block for block in sent[2].blocks if isinstance(block, TextBlock)] == [TextBlock(text=PREFACE)]
    assert [block.tool_call_id for block in sent[2].blocks if isinstance(block, ToolCallBlock)] == [
        "call_a",
        "call_b",
        "call_c",
    ]


@pytest.mark.parametrize("max_rounds", [6, 1], ids=["answered-by-choice", "made-to-answer"])
def test_reasoning_from_every_round_is_kept_on_the_result(max_rounds: int) -> None:
    """The reasoning of each round is kept for the turn, in order, and never streamed as the answer.

    Args:
        max_rounds (int): Tool rounds allowed: the second round answers by choice, or as the last round.
    """
    llm = streaming_tool_llm(
        [
            [
                chunk(reasoning_content="Zuerst das Programm suchen."),
                *_calls(_search_call("call_a", "Programm")),
            ],
            [chunk(reasoning_content="Das Programm nennt zehn Uhr."), *_answer()],
        ]
    )
    registry = _documents(QuerySearch({"Programm": [programme_passage()]}))

    _, texts, result = _drain(_turn(_loop(llm, registry, max_rounds=max_rounds)))

    assert "".join(texts) == ANSWER
    assert result.reasoning == "Zuerst das Programm suchen.\n\nDas Programm nennt zehn Uhr."


def test_a_turn_with_thinking_on_holds_a_reply_that_opens_in_reasoning() -> None:
    """With thinking asked for, text before the first close tag is reasoning, never the answer."""
    thought = "Die Frage zielt auf die Uhrzeit; das Programm sollte sie nennen, also antworte ich direkt daraus."

    def _llm() -> Any:
        """Build a model whose reply opens in a think block the template began.

        Returns:
            Any: The model double.
        """
        return streaming_tool_llm([_answer(f"{thought}</think>{ANSWER}", size=30)])

    _, opened, result = _drain(_turn(_loop(_llm(), _documents(QuerySearch({})), reasoning_open=True)))
    _, plain, _ = _drain(_turn(_loop(_llm(), _documents(QuerySearch({})), reasoning_open=False)))

    assert ("".join(opened), result.reasoning) == (ANSWER, thought)
    assert thought in "".join(plain)


def _counted_view() -> tuple[list[str], Callable[[Any], Any]]:
    """Build an LLM view that records the passages it renders.

    Returns:
        tuple[list[str], Callable[[Any], Any]]: The node ids seen, and the view.
    """
    seen: list[str] = []

    def _view(node: Any) -> Any:
        """Record the node, then hide what the test view hides.

        Args:
            node (Any): A ledger node.

        Returns:
            Any: The view of it.
        """
        seen.append(node.node_id)
        return llm_view(node)

    return seen, _view


def test_a_ratio_other_than_the_budgets_is_refused() -> None:
    """The transcript is priced at the ratio the budget's room was measured at, or the room means nothing."""
    with pytest.raises(ValueError, match="char_token_ratio"):
        _loop(streaming_tool_llm([]), _documents(QuerySearch({})), char_token_ratio=2.5)


@pytest.mark.parametrize("ratio", [0.0, -1.0])
def test_round_calls_refuse_a_ratio_that_is_not_positive(ratio: float) -> None:
    """Replies are priced in characters per token, which must be positive.

    Args:
        ratio (float): The ratio offered.
    """
    with pytest.raises(ValueError, match="char_token_ratio"):
        RoundCalls(
            registry=_documents(QuerySearch({})),
            result_chars=3000,
            max_calls_per_round=4,
            char_token_ratio=ratio,
            search_slots=threading.BoundedSemaphore(1),
        )


def test_the_tools_render_through_the_llm_view_the_loop_was_given() -> None:
    """The loop builds the turn's tool context with its own LLM view."""
    seen, view = _counted_view()
    llm = streaming_tool_llm([_calls(_search_call("call_a", "Programm")), _answer()])

    _drain(_turn(_loop(llm, _documents(QuerySearch({"Programm": [programme_passage()]})), llm_view=view)))

    assert "programme" in seen
