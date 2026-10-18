"""The agent loop: model rounds, the tool calls they make, and the limits that make the model answer.

Every message, passage, query and tool call here is invented.
"""

from __future__ import annotations

import bisect
import concurrent.futures
import json
import math
import threading
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping, Sequence
from contextvars import ContextVar
from typing import Any, cast

import httpx
import openai
import pytest
from _pytest.logging import LogCaptureFixture
from agent_fakes import (
    LEDGER_CAP,
    PROGRAMME,
    ThumbnailStore,
    chunk,
    frame_passage,
    programme_passage,
    shown_numbers,
    streaming_tool_llm,
    text_chunks,
    thumbnail,
    thumbnail_url,
    tool_call_delta,
    turn_budget,
)
from assessment_fakes import bad_request, join_threads, llm_view, scored_node
from llama_index.core.base.llms.types import ChatMessage, ImageBlock, MessageRole, TextBlock, ToolCallBlock
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
from docint.agents.assessment.ledger import EvidenceLedger, render_context
from docint.agents.assessment.split import SENTINEL, assessment_label
from docint.agents.turn import loop as turn_loop
from docint.agents.turn.budget import MESSAGE_OVERHEAD_TOKENS, TurnBudget, message_cost
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
from docint.agents.turn.image_tools import SearchImagesTool
from docint.agents.turn.images import TurnImages
from docint.agents.turn.loop import NOTE_CHARS, OVERFLOW_REPLY_SHARE, AgentLoop, AgentResult, ToolsUnavailable
from docint.agents.turn.scope import PASSAGES_FENCE, EvidenceDoesNotFit
from docint.agents.turn.stream import Provisional
from docint.agents.turn.tools import (
    MAX_ARGUMENT_DEPTH,
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
from docint.core.retrieval.visual import VISUAL_IMAGES_ATTACHED_KEY, render_legend
from docint.utils.text_budget import fit_with_marker

QUESTION = "zzsecretzz Wann öffnen die Stände beim Hafenfest?"
SYSTEM_PROMPT = "Beantworte Fragen zur Sammlung mit den Werkzeugen und belege jede Aussage mit [n]."
FINAL_NOTE = "Antworte jetzt aus den Passagen oben; rufe keine Werkzeuge mehr auf."
SCOPE_NOTE = "Der Nutzer hat diese Passagen ausgewählt; antworte nur aus ihnen und belege jede Aussage mit [n]."
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
    pinned: Sequence[NodeWithScore] | None = None,
) -> Iterator[AgentProgress | str | AgentResult]:
    """Run one turn of the user's invented question.

    Args:
        loop (AgentLoop): The loop.
        window (ConversationWindow | None): The conversation before it; none by default.
        ledger (EvidenceLedger | None): The turn's evidence; an empty one by default.
        pinned (Sequence[NodeWithScore] | None): The passages the user pinned; no scope by default.

    Returns:
        Iterator[AgentProgress | str | AgentResult]: The turn's output.
    """
    return loop.run(
        window=window or ConversationWindow(),
        message=QUESTION,
        # An empty ledger is falsy, so only a missing one is replaced.
        ledger=EvidenceLedger(cap=LEDGER_CAP) if ledger is None else ledger,
        pinned=pinned,
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
def test_a_first_reply_without_calls_is_the_answer_sent_whole(answer: str) -> None:
    """A first round that calls no tool is the answer: held whole, sent once it ends, and returned with nothing found.

    Args:
        answer (str): The model's reply, past the hold or within it: held whole either way.
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


def test_a_search_back_in_time_is_answered_while_one_that_is_late_is_not(
    held: threading.Event, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Of two searches started together, the one back before the deadline shows its passage; the other is late.

    Args:
        held (threading.Event): Holds the programme search.
        monkeypatch (pytest.MonkeyPatch): Lets the deadline pass the moment the first search is back.
    """
    now = [0.0]
    real_wait = concurrent.futures.wait

    def _deadline_after_the_first(futures: Any, timeout: float | None = None, **kwargs: Any) -> Any:
        """Wait until one search is back, let the time run out, then wait as the deadline allows.

        Args:
            futures (Any): The searches waited on.
            timeout (float | None): The wait the round asked for, overruled by the deadline.
            **kwargs (Any): Further options.

        Returns:
            Any: What the real wait returns once no time is left.
        """
        started = list(futures)
        real_wait(started, timeout=30, return_when=concurrent.futures.FIRST_COMPLETED)
        now[0] = 500.0
        return real_wait(started, timeout=0, **kwargs)

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

    monkeypatch.setattr(concurrent.futures, "wait", _deadline_after_the_first)
    llm = streaming_tool_llm([_calls(_search_call("call_a", "Programm"), _search_call("call_b", "Wetter")), _answer()])
    loop = _loop(llm, _documents(_search), search_slots=_Granting(size=2), clock=lambda: now[0])

    progress, _, result = _drain(_turn(loop))

    replies = _tool_replies(llm, 1)
    assert replies["call_a"] == error_reply("search", LATE_REPLY).text
    assert shown_numbers(replies["call_b"]) == [1]
    assert (_steps(progress), result.reason) == ([("search", "Wetter", 1)], "time_budget")


@pytest.mark.parametrize("preface", [PREFACE, LONG_PREFACE], ids=["held", "sent"])
def test_text_before_calls_becomes_a_note(preface: str) -> None:
    """A later round's text before its calls is a note step ahead of the calls; text already streamed went out first.

    The first round is held whole, so only a later one can have streamed its preface.

    Args:
        preface (str): What the second round writes before its call: short enough to be held,
            or long enough to have gone out as provisional answer text.
    """
    llm = streaming_tool_llm(
        [_calls(_search_call("call_w", "Wetter")), _calls(_search_call("call_a", "Programm"), text=preface), _answer()]
    )
    search = QuerySearch({"Wetter": [weather_passage()], "Programm": [programme_passage()]})

    items = list(_turn(_loop(llm, _documents(search))))

    note = AgentProgress(step=2, action="note", detail=" ".join(preface.split())[:NOTE_CHARS], found=0)
    position = items.index(note)
    assert items[position + 1] == AgentProgress(step=2, action="search", detail="Programm", found=1)
    streamed_first = "".join(item for item in items[:position] if isinstance(item, str))
    assert streamed_first == ("" if preface is PREFACE else preface.rstrip())
    result = items[-1]
    assert isinstance(result, AgentResult)
    assert result.actions == [
        {"action": "search", "detail": "Wetter", "found": 1},
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


def test_a_refusal_after_a_silent_first_round_is_raised_as_it_came() -> None:
    """Only the first call can show the endpoint takes no tools: a later refusal, nothing shown yet, is raised as is."""
    rejection = bad_request(TOOL_CHOICE_REJECTED)
    llm = streaming_tool_llm([_calls(("call_a", "keyframes", {"file": "harbour_walk.webm"})), rejection])

    shown, raised = _until_raised(_turn(_loop(llm, _documents(QuerySearch({})))))

    assert (shown, raised) == ([], rejection)


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
    ("max_rounds", "before"), [(6, "none"), (0, "round_budget")], ids=["by-choice", "made-to-answer"]
)
def test_an_answer_cut_at_the_length_limit_is_kept_and_flagged(
    max_rounds: int, before: str, loguru_caplog: LogCaptureFixture
) -> None:
    """An answer the endpoint cut at its length limit stays as written, and the turn says it was cut: ``answer_cut``.

    Args:
        max_rounds (int): Tool rounds allowed; with none, the answer round is forced and has a reason already.
        before (str): The reason that stood before the cut, which ``answer_cut`` replaces, as the log names it.
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    cut = ANSWER[:90]
    llm = streaming_tool_llm([[*text_chunks(cut, 30), chunk(finish_reason="length")]])

    _, texts, result = _drain(_turn(_loop(llm, _documents(QuerySearch({})), max_rounds=max_rounds)))

    assert ("".join(texts), result.raw_text, result.reason) == (cut, cut, "answer_cut")
    assert loguru_caplog.messages == [f"An agent turn's answer was cut at the length limit (earlier reason: {before})"]


@pytest.mark.parametrize("cause", ["research_failed", "round_budget"])
def test_a_cut_answer_replaces_whatever_reason_stood_before_it(cause: str) -> None:
    """A cut answer matters more than a limit or a failure before it, even on the round resent without tools.

    Args:
        cause (str): What stood first: a failed search before a cut answer by choice, or the rounds
            spent before a forced answer that only called tools and was resent, then cut.
    """
    cut = [*text_chunks(ANSWER[:90], 30), chunk(finish_reason="length")]
    if cause == "research_failed":
        llm = streaming_tool_llm([_calls(_search_call("call_a", "Programm")), cut])
        loop = _loop(llm, _documents(_failing_on("Programm", {})))
    else:
        llm = streaming_tool_llm(
            [_calls(_search_call("call_a", "Programm")), _calls(_search_call("call_b", "Wetter")), cut]
        )
        loop = _made_to_answer(llm)

    _, texts, result = _drain(_turn(loop))

    assert ("".join(texts), result.raw_text, result.reason) == (ANSWER[:90], ANSWER[:90], "answer_cut")
    # The cut round: the second, answering by choice, or the resend, which declares no tools.
    calls = llm.stream_chat.call_args_list
    if cause == "research_failed":
        assert (len(calls), calls[1].kwargs["tool_choice"]) == (2, "auto")
    else:
        assert (len(calls), calls[-1].kwargs) == (3, {})


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

    def _streaming() -> Iterator[AgentProgress | str | AgentResult]:
        """Run a turn whose second round streams its answer, the first one held whole having searched.

        Returns:
            Iterator[AgentProgress | str | AgentResult]: The turn's output: the search step, then the answer.
        """
        llm = streaming_tool_llm([_calls(_search_call("call_a", "Programm")), _answer(ANSWER, size=20)])
        return _turn(_loop(llm, _documents(QuerySearch({"Programm": [programme_passage()]})), model_slots=slots))

    turn = _streaming()
    step, first = next(turn), next(turn)
    held_while_streaming = not slots.acquire(blocking=False)
    rest = list(turn)

    assert isinstance(step, AgentProgress) and isinstance(first, str) and held_while_streaming
    assert isinstance(rest[-1], AgentResult)
    assert slots.acquire(blocking=False)
    slots.release()

    stopped = cast(Generator[Any, None, None], _streaming())
    next(stopped)
    next(stopped)
    stopped.close()
    assert slots.acquire(blocking=False)
    slots.release()


# --- Search slots and the deadline -----------------------------------------------------------


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

    ledger = EvidenceLedger(cap=LEDGER_CAP)
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
    broken = [*text_chunks(ANSWER, 20)[:7], RuntimeError("invented outage")]
    llm = streaming_tool_llm([_calls(_search_call("call_a", "Programm")), broken, _answer(SHORT_ANSWER)])

    items = list(_turn(_loop(llm, _documents(QuerySearch({"Programm": [programme_passage()]})))))

    search, note = [item for item in items if isinstance(item, AgentProgress)]
    position = items.index(note)
    withdrawn = "".join(cast(list[str], items[items.index(search) + 1 : position]))
    assert (note.step, note.action, note.detail) == (2, "note", " ".join(withdrawn.split()))
    assert len(withdrawn) >= 120
    assert "".join(cast(list[str], items[position + 1 : -1])) == SHORT_ANSWER
    assert llm.stream_chat.call_args_list[2].kwargs["tool_choice"] == "none"
    result = items[-1]
    assert isinstance(result, AgentResult)
    assert (result.raw_text, result.reason, result.steps) == (SHORT_ANSWER, "research_failed", 1)
    assert result.actions == [
        {"action": "search", "detail": "Programm", "found": 1},
        {"action": "note", "detail": note.detail, "found": 0},
    ]


@pytest.mark.parametrize("path", ["first-round-withdrawn", "no-tool-round-allowed"])
def test_the_final_note_joins_a_user_message_it_would_follow(path: str) -> None:
    """Strict chat templates refuse two user messages in a row, so the final note joins the user's message.

    Args:
        path (str): How the answer round comes to follow the user's message directly: the first
            round's calls did not fit the room and it was withdrawn, or no tool round was allowed.
    """
    if path == "first-round-withdrawn":
        unoffered = [(f"call_{number}", "keyframes", {"file": "harbour_walk.webm"}) for number in range(20)]
        llm = streaming_tool_llm([_calls(*unoffered), _answer(SHORT_ANSWER)])
        loop = _loop(llm, _documents(QuerySearch({})), budget=TIGHT)
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
    broken = [*text_chunks(ANSWER, 20)[:7], RuntimeError("invented second outage")]
    llm = streaming_tool_llm([_calls(_search_call("call_a", "Programm")), broken])
    loop = _loop(llm, _documents(QuerySearch({"Programm": [programme_passage()]})), max_rounds=1)

    shown, exc = _until_raised(_turn(loop))

    assert (type(exc), str(exc)) == (RuntimeError, "invented second outage")
    assert [type(item) for item in shown if not isinstance(item, str)] == [AgentProgress]
    assert isinstance(shown[-1], str)


# --- An answer round that only calls tools ---------------------------------------------------

#: The one warning a resend logs: a count, never a call's arguments.
RESENT = "An agent turn's answer round only called tools (1); it is sent once more with no tools declared"


def _made_to_answer(llm: Any, **overrides: Any) -> AgentLoop:
    """Build a loop allowed one tool round, so the round after a search is the answer round.

    Args:
        llm (Any): The model double.
        **overrides (Any): Constructor arguments to change.

    Returns:
        AgentLoop: The loop.
    """
    return _loop(llm, _documents(QuerySearch({"Programm": [programme_passage()]})), max_rounds=1, **overrides)


def test_an_answer_round_that_only_calls_tools_is_sent_once_more_without_tools(
    loguru_caplog: LogCaptureFixture,
) -> None:
    """An endpoint that ignores ``tool_choice`` answers with calls; sent again declaring no tools, it answers.

    Args:
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    llm = streaming_tool_llm(
        [
            _calls(_search_call("call_a", "Programm")),
            _calls(_search_call("call_b", "zzsecretzz Wetter")),
            _answer(),
        ]
    )

    progress, texts, result = _drain(_turn(_made_to_answer(llm)))

    assert llm.stream_chat.call_count == 3
    answer_round, resent = llm.stream_chat.call_args_list[1:]
    assert answer_round.kwargs["tool_choice"] == "none"
    assert resent.kwargs == {}
    assert resent.args == answer_round.args
    assert ("".join(texts), result.raw_text, result.reason) == (ANSWER, ANSWER, "round_budget")
    assert _steps(progress) == [("search", "Programm", 1)]
    assert [scored.node.node_id for scored in result.nodes] == ["programme"]
    assert loguru_caplog.messages == [RESENT]
    assert "zzsecretzz" not in loguru_caplog.text


def test_a_resend_keeps_the_reasoning_of_the_round_it_resends() -> None:
    """The answer round that only called tools still thought: its reasoning is kept, in order, before the resend's."""
    llm = streaming_tool_llm(
        [
            [chunk(reasoning_content="Erst das Programm suchen."), *_calls(_search_call("call_a", "Programm"))],
            [chunk(reasoning_content="Noch das Wetter prüfen."), *_calls(_search_call("call_b", "Wetter"))],
            [chunk(reasoning_content="Das Programm reicht für die Antwort."), *_answer()],
        ]
    )

    _, texts, result = _drain(_turn(_made_to_answer(llm)))

    assert (llm.stream_chat.call_count, llm.stream_chat.call_args.kwargs, "".join(texts)) == (3, {}, ANSWER)
    assert result.reasoning == (
        "Erst das Programm suchen.\n\nNoch das Wetter prüfen.\n\nDas Programm reicht für die Antwort."
    )


@pytest.mark.parametrize(
    ("resend", "warned"),
    [
        (
            _calls(_search_call("call_c", "Regatta")),
            ["An agent turn's answer round asked for 1 tool calls anyway; none ran"],
        ),
        ([chunk(finish_reason="stop")], []),
    ],
    ids=["calls-again", "nothing"],
)
def test_a_resend_that_is_still_empty_leaves_the_answer_empty_as_before(
    resend: list[dict[str, Any]], warned: list[str], loguru_caplog: LogCaptureFixture
) -> None:
    """The resend is the only one: a reply with no text again ends the turn as an answer round of calls did.

    Args:
        resend (list[dict[str, Any]]): What the resend streams.
        warned (list[str]): The warnings that follow the resend's.
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    llm = streaming_tool_llm(
        [_calls(_search_call("call_a", "Programm")), _calls(_search_call("call_b", "Wetter")), resend]
    )

    _, texts, result = _drain(_turn(_made_to_answer(llm)))

    assert llm.stream_chat.call_count == 3
    assert (texts, result.raw_text, result.reason) == ([], "", "round_budget")
    assert [scored.node.node_id for scored in result.nodes] == ["programme"]
    assert loguru_caplog.messages == [RESENT, *warned]


def test_an_answer_round_with_text_beside_its_calls_is_not_resent(loguru_caplog: LogCaptureFixture) -> None:
    """Text is the answer: a round that wrote some beside its calls keeps it, and none of its calls runs.

    Args:
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    llm = streaming_tool_llm(
        [_calls(_search_call("call_a", "Programm")), _calls(_search_call("call_b", "Wetter"), text=SHORT_ANSWER)]
    )

    _, texts, result = _drain(_turn(_made_to_answer(llm)))

    assert llm.stream_chat.call_count == 2
    assert ("".join(texts), result.raw_text) == (SHORT_ANSWER, SHORT_ANSWER)
    assert loguru_caplog.messages == ["An agent turn's answer round asked for 1 tool calls anyway; none ran"]


def test_a_pinned_round_that_only_calls_tools_is_not_resent() -> None:
    """The pinned round declares no tools already, so it is never sent again without them."""
    llm = streaming_tool_llm([_calls(_search_call("call_a", "Programm"))])

    _, texts, result = _drain(_turn(_loop(llm, _documents(QuerySearch({}))), pinned=[programme_passage()]))

    assert llm.stream_chat.call_count == 1
    assert (texts, result.raw_text) == ([], "")


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("invented outage"),
        _status_error(openai.RateLimitError, 429, "rate limit reached"),
        bad_request(TOOL_CHOICE_REJECTED),
    ],
    ids=["outage", "rate-limit", "refused"],
)
def test_a_resend_that_fails_raises_as_a_failed_answer_round_does(error: Exception) -> None:
    """The resend is the answer round once more: its failure ends the turn, and no empty answer passes as one.

    Args:
        error (Exception): What the resend's first pull raises.
    """
    llm = streaming_tool_llm(
        [_calls(_search_call("call_a", "Programm")), _calls(_search_call("call_b", "Wetter")), error]
    )

    shown, raised = _until_raised(_turn(_made_to_answer(llm)))

    assert (raised, llm.stream_chat.call_count) == (error, 3)
    assert _steps(item for item in shown if isinstance(item, AgentProgress)) == [("search", "Programm", 1)]
    assert not any(isinstance(item, (str, AgentResult)) for item in shown)


def test_a_resend_that_overflows_raises_naming_the_context_window() -> None:
    """An overflowing resend is reported as an answer round's second overflow is."""
    overflow = bad_request(OVERFLOW)
    llm = streaming_tool_llm(
        [_calls(_search_call("call_a", "Programm")), _calls(_search_call("call_b", "Wetter")), overflow]
    )

    _, raised = _until_raised(_turn(_made_to_answer(llm)))

    assert (type(raised), raised.__cause__, llm.stream_chat.call_count) == (ValueError, overflow, 3)
    assert "context window" in str(raised)


def test_a_resend_without_a_model_slot_raises_after_waiting_as_the_answer_does() -> None:
    """The resend waits for a model slot as the answer round does, and without one the turn fails."""
    slots = _Granting(True, True, False)
    llm = streaming_tool_llm([_calls(_search_call("call_a", "Programm")), _calls(_search_call("call_b", "Wetter"))])

    shown, raised = _until_raised(_turn(_made_to_answer(llm, model_slots=slots, clock=lambda: 0.0)))

    assert (type(raised), llm.stream_chat.call_count) == (NoSlot, 2)
    assert not any(isinstance(item, (str, AgentResult)) for item in shown)
    assert slots.waits == [MODEL_SLOT_WAIT_S, 120.0, 120.0]


# --- The search guard: a first round that looked nothing up is asked again ------------------

#: What a guarded loop tells a first round that answered without a tool call.
SEARCH_NOTE = "Du hast für diese Frage noch nichts nachgeschlagen: suche zuerst, dann antworte mit Belegen."
#: A first reply from memory, longer than the hold, so it would stream if the first round were not held whole.
MEMORY_ANSWER = (
    "Die Stände öffnen wie jedes Jahr um zehn Uhr, und das Hafenfest findet trotz der Sturmwarnung "
    "wie geplant am Samstag statt; nachschlagen muss ich das nicht."
)
#: The one line a guard that asks round 1 again logs: no text of the turn.
ASKED_AGAIN = "An agent turn's first round called no tool; it is asked once more, to search first"


def _guarded(llm: Any, registry: ToolRegistry, **overrides: Any) -> AgentLoop:
    """Build a loop whose first round, if it calls no tool, is asked again with the search note.

    Args:
        llm (Any): The model double.
        registry (ToolRegistry): The tools offered.
        **overrides (Any): Constructor arguments to change.

    Returns:
        AgentLoop: The loop.
    """
    return _loop(llm, registry, search_note=SEARCH_NOTE, **overrides)


def _until_text(turn: Iterator[AgentProgress | str | AgentResult]) -> list[AgentProgress | str | AgentResult]:
    """Read a turn up to and including its first answer text.

    Args:
        turn (Iterator[AgentProgress | str | AgentResult]): The turn's output.

    Returns:
        list[AgentProgress | str | AgentResult]: What it yielded so far, the first text last.
    """
    read: list[AgentProgress | str | AgentResult] = []
    for item in turn:
        read.append(item)
        if isinstance(item, str):
            break
    return read


def test_a_first_reply_without_a_tool_call_is_dropped_unseen_and_round_one_asked_again() -> None:
    """A first round that answers from memory shows nothing: round 1 runs again, the search note joined."""
    llm = streaming_tool_llm([_answer(MEMORY_ANSWER), _calls(_search_call("call_a", "Programm")), _answer()])
    registry = _documents(QuerySearch({"Programm": [programme_passage()]}))

    progress, texts, result = _drain(_turn(_guarded(llm, registry)))

    assert ("".join(texts), _steps(progress)) == (ANSWER, [("search", "Programm", 1)])
    assert (result.raw_text, result.steps, result.reason) == (ANSWER, 1, None)
    assert result.actions == [{"action": "search", "detail": "Programm", "found": 1}]
    assert [_wire(message) for message in _sent(llm, 0)] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": QUESTION},
    ]
    assert [_wire(message) for message in _sent(llm, 1)] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{QUESTION}\n\n{SEARCH_NOTE}"},
    ]
    asked_again = llm.stream_chat.call_args_list[1].kwargs
    assert (asked_again["tools"], asked_again["tool_choice"]) == (registry.schemas(), "auto")
    assert llm.stream_chat.call_count == 3
    # The note stays joined for later rounds, once.
    assert _wire(_sent(llm, 2)[1]) == {"role": "user", "content": f"{QUESTION}\n\n{SEARCH_NOTE}"}


def test_a_second_reply_without_a_tool_call_is_the_answer_streamed_as_it_comes() -> None:
    """Round 1 is asked again once: a second reply calling no tool either is the answer, streamed as it comes."""
    first, second = _answer(MEMORY_ANSWER), _answer(ANSWER, size=20)
    llm = streaming_tool_llm([first, second])
    pulled = _pulls(llm)
    turn = _turn(_guarded(llm, _documents(QuerySearch({}))))

    read = _until_text(turn)
    begun = len(pulled)
    progress, texts, result = _drain([*read, *turn])

    assert begun < len(first) + len(second)
    assert ("".join(texts), progress, llm.stream_chat.call_count) == (ANSWER, [], 2)
    assert (result.raw_text, result.steps, result.actions, result.reason) == (ANSWER, 0, [], None)


def test_a_first_round_that_calls_a_tool_is_never_asked_again() -> None:
    """A first round that searched is left as it was: no note joins the message, and the next round streams."""
    first, second = _calls(_search_call("call_a", "Programm"), text=PREFACE), _answer(ANSWER, size=20)
    llm = streaming_tool_llm([first, second])
    pulled = _pulls(llm)
    turn = _turn(_guarded(llm, _documents(QuerySearch({"Programm": [programme_passage()]}))))

    read = _until_text(turn)
    begun = len(pulled)
    progress, texts, result = _drain([*read, *turn])

    assert begun < len(first) + len(second)
    assert _steps(progress) == [("note", PREFACE, 0), ("search", "Programm", 1)]
    assert ("".join(texts), result.raw_text, llm.stream_chat.call_count) == (ANSWER, ANSWER, 2)
    assert all(SEARCH_NOTE not in str(message.content) for call in (0, 1) for message in _sent(llm, call))


def test_a_pinned_scope_is_never_asked_again() -> None:
    """A pinned round offers no tool, so its reply is the answer, whatever the search note says."""
    llm = streaming_tool_llm([_answer(SHORT_ANSWER)])

    _, texts, result = _drain(_turn(_guarded(llm, _documents(QuerySearch({}))), pinned=[programme_passage()]))

    assert ("".join(texts), result.raw_text, llm.stream_chat.call_count) == (SHORT_ANSWER, SHORT_ANSWER, 1)
    assert SEARCH_NOTE not in str(_sent(llm, 0)[-1].content)


def test_a_first_round_offered_no_tool_is_never_asked_again() -> None:
    """Asking a round to look things up means nothing when it was offered no tool: its reply is the answer."""
    llm = streaming_tool_llm([_answer(MEMORY_ANSWER)])

    _, texts, result = _drain(_turn(_guarded(llm, ToolRegistry([]))))

    assert ("".join(texts), result.raw_text, llm.stream_chat.call_count) == (MEMORY_ANSWER, MEMORY_ANSWER, 1)
    assert llm.stream_chat.call_args.kwargs == {}


def test_without_a_search_note_a_first_reply_without_a_tool_call_is_the_answer() -> None:
    """An empty search note asks nothing again: the first round's reply is the answer, sent whole once it ends."""
    llm = streaming_tool_llm([_answer(MEMORY_ANSWER)])

    _, texts, result = _drain(_turn(_loop(llm, _documents(QuerySearch({})))))

    assert ("".join(texts), result.raw_text, llm.stream_chat.call_count) == (MEMORY_ANSWER, MEMORY_ANSWER, 1)


@pytest.mark.parametrize("search_note", ["", SEARCH_NOTE], ids=["unguarded", "guarded"])
def test_a_first_rounds_preface_is_never_sent_as_answer_text(search_note: str) -> None:
    """The first round is held whole: a preface past the hold becomes its note without going out as answer text first.

    Args:
        search_note (str): The loop's search note; none, or the guard's.
    """
    llm = streaming_tool_llm([_calls(_search_call("call_a", "Programm"), text=LONG_PREFACE), _answer()])
    registry = _documents(QuerySearch({"Programm": [programme_passage()]}))

    items = list(_turn(_loop(llm, registry, search_note=search_note)))

    note = AgentProgress(step=1, action="note", detail=" ".join(LONG_PREFACE.split())[:NOTE_CHARS], found=0)
    assert len(LONG_PREFACE) > 120
    assert items[:2] == [note, AgentProgress(step=1, action="search", detail="Programm", found=1)]
    _, texts, result = _drain(items)
    assert ("".join(texts), result.raw_text) == (ANSWER, ANSWER)


def test_the_dropped_reply_reaches_neither_the_transcript_nor_the_reasoning() -> None:
    """The reply the guard drops is gone: no later round is sent it, and the turn keeps none of its reasoning."""
    llm = streaming_tool_llm(
        [
            [chunk(reasoning_content="Das weiß ich noch vom letzten Hafenfest."), *_answer(MEMORY_ANSWER)],
            [chunk(reasoning_content="Erst das Programm suchen."), *_calls(_search_call("call_a", "Programm"))],
            _answer(),
        ]
    )

    _, _, result = _drain(_turn(_guarded(llm, _documents(QuerySearch({"Programm": [programme_passage()]})))))

    sent = [message for call in (1, 2) for message in _sent(llm, call)]
    assert not any(MEMORY_ANSWER in str(message.content) for message in sent)
    assert [message.role for message in _sent(llm, 2)] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
    ]
    assert result.reasoning == "Erst das Programm suchen."


@pytest.mark.parametrize(("budget", "wait"), [(120.0, MODEL_SLOT_WAIT_S), (2.0, 2.0)])
def test_no_model_slot_for_round_one_asked_again_raises_before_anything_is_yielded(budget: float, wait: float) -> None:
    """Round 1 asked again waits for a model slot as round 1 does; without one, nothing shown, the caller falls back.

    Args:
        budget (float): The turn's time budget in seconds.
        wait (float): The longest each of the two may wait for a slot.
    """
    slots = _Granting(True, False)
    llm = streaming_tool_llm([_answer(MEMORY_ANSWER), _answer()])
    loop = _guarded(llm, _documents(QuerySearch({})), model_slots=slots, time_budget_s=budget, clock=lambda: 0.0)

    shown, raised = _until_raised(_turn(loop))

    assert (shown, type(raised), slots.waits, llm.stream_chat.call_count) == ([], NoSlot, [wait, wait], 1)


def test_a_first_round_that_spends_the_time_is_answered_instead_of_asked_again(
    loguru_caplog_info: LogCaptureFixture,
) -> None:
    """Limits are checked before round 1 runs again: with the time spent, the model answers with the final note alone.

    The search note asks for a search the forced round cannot make, so it is never joined.

    Args:
        loguru_caplog_info (LogCaptureFixture): Bridged INFO capture.
    """
    now = [0.0]
    llm = _advancing(streaming_tool_llm([_answer(MEMORY_ANSWER), _answer()]), now, after=0, to=500.0)

    _, texts, result = _drain(_turn(_guarded(llm, _documents(QuerySearch({})), clock=lambda: now[0])))

    assert llm.stream_chat.call_args_list[1].kwargs["tool_choice"] == "none"
    assert _wire(_sent(llm, 1)[-1]) == {"role": "user", "content": f"{QUESTION}\n\n{FINAL_NOTE}"}
    assert ("".join(texts), result.reason, result.steps, llm.stream_chat.call_count) == (ANSWER, "time_budget", 0, 2)
    assert ASKED_AGAIN not in loguru_caplog_info.messages


@pytest.mark.parametrize(("spare", "asked_again"), [(0, True), (-1, False)], ids=["fits", "a-token-short"])
def test_the_search_note_counts_against_the_room(spare: int, asked_again: bool) -> None:
    """Joined to the message, the note costs room: round 1 runs again only while one more full reply still fits.

    Answering for want of room, the model gets the final note alone: the search note is never joined.

    Args:
        spare (int): Tokens the room holds beyond one full reply once the final and the search note are paid.
        asked_again (bool): Whether round 1 runs again, rather than the model answering for want of room.
    """
    full_reply = math.ceil(turn_budget().result_chars / 3.5) + MESSAGE_OVERHEAD_TOKENS
    notes = message_cost(FINAL_NOTE, 3.5) + message_cost(SEARCH_NOTE, 3.5)
    room = notes + full_reply + MESSAGE_OVERHEAD_TOKENS + spare
    llm = streaming_tool_llm([_answer(MEMORY_ANSWER), _answer()])

    _, texts, result = _drain(_turn(_guarded(llm, _documents(QuerySearch({})), budget=_budget_with_room(room))))

    assert llm.stream_chat.call_args_list[1].kwargs["tool_choice"] == ("auto" if asked_again else "none")
    assert ("".join(texts), result.reason) == (ANSWER, None if asked_again else "context_budget")
    joined = SEARCH_NOTE if asked_again else FINAL_NOTE
    assert _wire(_sent(llm, 1)[-1]) == {"role": "user", "content": f"{QUESTION}\n\n{joined}"}


@pytest.mark.parametrize(
    ("error", "raised"),
    [(bad_request(TOOL_CHOICE_REJECTED), openai.BadRequestError), (bad_request(OVERFLOW), ValueError)],
    ids=["refused", "overflow"],
)
def test_round_one_asked_again_fails_as_round_one_does_but_is_not_the_first_call(
    error: Exception, raised: type[Exception]
) -> None:
    """Nothing was shown, so the failure goes to the caller; only the turn's first call can say tools are unavailable.

    Args:
        error (Exception): What round 1 asked again raises on its first pull.
        raised (type[Exception]): What the turn raises: the refusal as it came, or the overflow naming the
            context window.
    """
    llm = streaming_tool_llm([_answer(MEMORY_ANSWER), error])

    shown, exc = _until_raised(_turn(_guarded(llm, _documents(QuerySearch({})))))

    assert (shown, type(exc), llm.stream_chat.call_count) == ([], raised, 2)
    assert (exc if raised is not ValueError else exc.__cause__) is error


def test_asking_round_one_again_is_logged_once_without_any_text(loguru_caplog_info: LogCaptureFixture) -> None:
    """One INFO line says round 1 was asked again; neither the message nor the dropped reply reaches the log.

    Args:
        loguru_caplog_info (LogCaptureFixture): Bridged INFO capture.
    """
    llm = streaming_tool_llm([_answer(f"zzsecretzz {MEMORY_ANSWER}"), _answer()])

    _drain(_turn(_guarded(llm, _documents(QuerySearch({})))))

    assert loguru_caplog_info.messages == [ASKED_AGAIN]
    assert "zzsecretzz" not in loguru_caplog_info.text


# --- A first round that breaks off takes the same one re-ask ---------------------------------

#: The one line a first round's break-off logs: the error's type, no text of the turn.
BROKE_OFF = "An agent turn's first round broke off (RuntimeError) before anything was shown"


def _broken(*, pieces: int = 7, error: Exception | None = None) -> list[Any]:
    """Script a round that breaks off after its first event, partway through the answer.

    Args:
        pieces (int): Twenty-character pieces streamed first; seven pass the hold, three do not.
        error (Exception | None): What breaks the stream; an invented outage by default.

    Returns:
        list[Any]: The round's chunks, the error last.
    """
    return [*text_chunks(ANSWER, 20)[:pieces], error or RuntimeError("invented outage")]


@pytest.mark.parametrize("search_note", ["", SEARCH_NOTE], ids=["unguarded", "guarded"])
def test_a_first_round_that_breaks_off_is_asked_again_once_without_the_search_note(
    search_note: str, loguru_caplog: LogCaptureFixture
) -> None:
    """Held whole, a first round that broke off showed nothing: the same request goes out once more, and answers.

    The break-off is not about searching, so no note is joined; and the turn's one re-ask is
    spent, so a second try that calls no tool is the answer, guard or none.

    Args:
        search_note (str): The loop's search note; none, or the guard's.
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    llm = streaming_tool_llm([_broken(), _answer()])

    progress, texts, result = _drain(_turn(_loop(llm, _documents(QuerySearch({})), search_note=search_note)))

    assert len(ANSWER[: 7 * 20]) > 120
    assert ("".join(texts), progress, llm.stream_chat.call_count) == (ANSWER, [], 2)
    assert (result.raw_text, result.reason, result.steps, result.actions) == (ANSWER, None, 0, [])
    asked = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": QUESTION}]
    assert [[_wire(message) for message in _sent(llm, call)] for call in (0, 1)] == [asked, asked]
    assert llm.stream_chat.call_args_list[1].kwargs["tool_choice"] == "auto"
    assert loguru_caplog.messages == [BROKE_OFF]


def test_a_second_break_off_raises_as_before() -> None:
    """The one re-ask is spent: a second try that breaks off before showing anything fails the turn as round 1 would."""
    second = RuntimeError("invented second outage")
    llm = streaming_tool_llm([_broken(), _broken(pieces=3, error=second)])

    shown, raised = _until_raised(_turn(_loop(llm, _documents(QuerySearch({})))))

    assert (shown, raised, llm.stream_chat.call_count) == ([], second, 2)


def test_a_break_off_after_the_guards_second_try_raises() -> None:
    """The guard and a break-off share the one re-ask: once the guard has taken it, a break-off fails the turn."""
    outage = RuntimeError("invented outage")
    llm = streaming_tool_llm([_answer(MEMORY_ANSWER), _broken(pieces=3, error=outage)])

    shown, raised = _until_raised(_turn(_guarded(llm, _documents(QuerySearch({})))))

    assert (shown, raised, llm.stream_chat.call_count) == ([], outage, 2)
    assert _wire(_sent(llm, 1)[-1]) == {"role": "user", "content": f"{QUESTION}\n\n{SEARCH_NOTE}"}


def test_a_first_round_that_broke_off_after_spending_the_time_is_answered() -> None:
    """Round 1 asked again keeps round 1's rules: with the time spent, the model answers with the final note alone."""
    now = [0.0]
    llm = _advancing(streaming_tool_llm([_broken(), _answer()]), now, after=0, to=500.0)

    _, texts, result = _drain(_turn(_guarded(llm, _documents(QuerySearch({})), clock=lambda: now[0])))

    assert llm.stream_chat.call_args_list[1].kwargs["tool_choice"] == "none"
    assert _wire(_sent(llm, 1)[-1]) == {"role": "user", "content": f"{QUESTION}\n\n{FINAL_NOTE}"}
    assert ("".join(texts), result.reason, llm.stream_chat.call_count) == (ANSWER, "time_budget", 2)


#: What an answer round is sent when no round has answered yet: the question and the final note alone.
FORCED_ALONE = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": f"{QUESTION}\n\n{FINAL_NOTE}"},
]


def test_a_forced_answer_after_the_second_try_broke_off_carries_no_search_note() -> None:
    """The guard's second try broke off after its text showed, adding nothing: the answer has the final note alone.

    No round answered to the search note, and it asks for a search the answer round cannot make.
    """
    llm = streaming_tool_llm([_answer(MEMORY_ANSWER), _broken(), _answer(SHORT_ANSWER)])

    _, _, result = _drain(_turn(_guarded(llm, _documents(QuerySearch({})))))

    assert llm.stream_chat.call_args_list[2].kwargs["tool_choice"] == "none"
    assert [_wire(message) for message in _sent(llm, 2)] == FORCED_ALONE
    assert (result.raw_text, result.reason) == (SHORT_ANSWER, "research_failed")


@pytest.mark.parametrize("answered", ["at-once", "resent"])
def test_a_forced_answer_after_the_second_try_was_withdrawn_carries_no_search_note(answered: str) -> None:
    """The guard's second try had calls the room could not hold, withdrawn unrun: no answer round carries the note.

    Args:
        answered (str): How the forced round answers: at once, or by a resend after it only called tools.
    """
    unoffered = [(f"call_{number}", "keyframes", {"file": "harbour_walk.webm"}) for number in range(20)]
    forced = [_answer(SHORT_ANSWER)]
    if answered == "resent":
        forced.insert(0, _calls(_search_call("call_x", "Programm")))
    llm = streaming_tool_llm([_answer(MEMORY_ANSWER), _calls(*unoffered), *forced])

    _, _, result = _drain(_turn(_guarded(llm, _documents(QuerySearch({})), budget=TIGHT)))

    answer_rounds = range(2, llm.stream_chat.call_count)
    assert len(answer_rounds) == len(forced)
    assert [[_wire(message) for message in _sent(llm, call)] for call in answer_rounds] == [FORCED_ALONE] * len(forced)
    assert (result.raw_text, result.reason, result.steps) == (SHORT_ANSWER, "context_budget", 0)


def test_a_forced_answer_after_the_second_try_searched_keeps_the_search_note() -> None:
    """With the second try's calls in the transcript, the note stays where they answered it; the final note follows."""
    llm = streaming_tool_llm([_answer(MEMORY_ANSWER), _calls(_search_call("call_a", "Programm")), _answer()])
    registry = _documents(QuerySearch({"Programm": [programme_passage()]}))

    _, _, result = _drain(_turn(_guarded(llm, registry, max_rounds=1)))

    sent = _sent(llm, 2)
    roles = [MessageRole.SYSTEM, MessageRole.USER, MessageRole.ASSISTANT, MessageRole.TOOL, MessageRole.USER]
    assert [message.role for message in sent] == roles
    assert (_wire(sent[1]), _wire(sent[-1])) == (
        {"role": "user", "content": f"{QUESTION}\n\n{SEARCH_NOTE}"},
        {"role": "user", "content": FINAL_NOTE},
    )
    assert (result.reason, result.steps, llm.stream_chat.call_args_list[2].kwargs["tool_choice"]) == (
        "round_budget",
        1,
        "none",
    )


def test_a_second_try_that_breaks_off_after_showing_text_is_withdrawn_and_answered_anew() -> None:
    """The second try streams as later rounds do: text it showed before it broke off is withdrawn, then answered."""
    llm = streaming_tool_llm([_broken(), _broken(), _answer(SHORT_ANSWER)])

    items = list(_turn(_loop(llm, _documents(QuerySearch({})))))

    (note,) = [item for item in items if isinstance(item, AgentProgress)]
    withdrawn = "".join(cast(list[str], items[: items.index(note)]))
    assert (note.step, note.action, note.detail) == (1, "note", " ".join(withdrawn.split()))
    assert len(withdrawn) >= 120
    assert llm.stream_chat.call_args_list[2].kwargs["tool_choice"] == "none"
    result = items[-1]
    assert isinstance(result, AgentResult)
    assert (result.raw_text, result.reason, llm.stream_chat.call_count) == (SHORT_ANSWER, "research_failed", 3)


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


def _budget_with_room(round_room: int, *, ratio: float | None = None) -> TurnBudget:
    """Build an 8k budget whose room for tool replies is ``round_room`` tokens.

    Args:
        round_room (int): The room wanted.
        ratio (float | None): The characters per token it is measured at; the usual 3.5 by default.

    Returns:
        TurnBudget: The budget, its system prompt sized to leave that room.
    """
    budget = turn_budget()
    return TurnBudget(
        ctx_window=budget.ctx_window,
        reasoning=False,
        char_token_ratio=budget.char_token_ratio if ratio is None else ratio,
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


def test_arguments_the_registry_refuses_as_too_deep_are_echoed_as_an_empty_object() -> None:
    """Arguments refused as nested too deeply are echoed back as ``{}``, so no endpoint re-parses them."""
    nested = '{"n": ' + "[" * 600 + "]" * 600 + "}"
    llm = streaming_tool_llm([_calls(("call_a", "read", nested)), _answer()])

    _drain(_turn(_loop(llm, _documents(QuerySearch({})))))

    (block,) = [block for block in _sent(llm, 1)[2].blocks if isinstance(block, ToolCallBlock)]
    assert block.tool_kwargs == "{}"
    assert _tool_replies(llm, 1)["call_a"].startswith("Error: the arguments for read are nested too deeply")


@pytest.mark.parametrize("depth", [MAX_ARGUMENT_DEPTH, MAX_ARGUMENT_DEPTH + 1], ids=["at-the-limit", "past-it"])
def test_the_echo_and_the_registry_agree_where_nesting_becomes_too_deep(depth: int) -> None:
    """The echo and the registry share one limit: the registry refuses exactly the arguments echoed as ``{}``.

    Args:
        depth (int): How deeply the arguments nest, the object itself counted.
    """
    arguments = '{"n": ' + "[" * (depth - 1) + "3" + "]" * (depth - 1) + "}"
    llm = streaming_tool_llm([_calls(("call_a", "read", arguments)), _answer()])

    _drain(_turn(_loop(llm, _documents(QuerySearch({})))))

    refused = depth > MAX_ARGUMENT_DEPTH
    (block,) = [block for block in _sent(llm, 1)[2].blocks if isinstance(block, ToolCallBlock)]
    reply = _tool_replies(llm, 1)["call_a"]
    assert block.tool_kwargs == ("{}" if refused else arguments)
    assert reply.startswith("Error: the arguments for read are nested too deeply") is refused


def test_a_call_past_the_rounds_limit_is_never_prepared(loguru_caplog: LogCaptureFixture) -> None:
    """A call the round's limit skips is not looked at: it gets the skipped reply and never fails the turn.

    Args:
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    registry = ToolRegistry([SearchTool(QuerySearch({"Programm": [programme_passage()]})), _Unpreparable()])
    llm = streaming_tool_llm([_calls(_search_call("call_a", "Programm"), ("call_b", "unpreparable", {})), _answer()])

    progress, _, result = _drain(_turn(_loop(llm, registry, max_calls_per_round=1)))

    assert _tool_replies(llm, 1)["call_b"] == error_reply("unpreparable", SKIPPED_REPLY.format(limit=1)).text
    assert (_steps(progress), result.reason) == ([("search", "Programm", 1)], None)
    assert loguru_caplog.messages == []


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


# --- A pinned scope: one round that offers no tools -------------------------------------------


def _pinned_message(nodes: list[NodeWithScore]) -> str:
    """Write the one user message a pinned round sends: the scope note, the passages between fences, the question.

    Args:
        nodes (list[NodeWithScore]): The pinned passages, numbered; short enough to be shown whole.

    Returns:
        str: The message.
    """
    context = render_context(nodes, llm_view=llm_view, budget_tokens=100_000, char_token_ratio=3.5).context
    return f"{SCOPE_NOTE}\n\n{PASSAGES_FENCE}\n{context}\n{PASSAGES_FENCE}\n\n{QUESTION}"


def _pulls(llm: Any) -> list[int]:
    """Count the chunks the model double has streamed, as they are pulled.

    Args:
        llm (Any): The model double.

    Returns:
        list[int]: One entry per chunk pulled so far, growing in place.
    """
    inner = llm.stream_chat.side_effect
    pulled: list[int] = []

    def _stream(messages: list[ChatMessage], **kwargs: Any) -> Iterator[Any]:
        """Stream as scripted, counting each chunk as it is pulled.

        Args:
            messages (list[ChatMessage]): The prompt.
            **kwargs (Any): Call options.

        Yields:
            Any: The scripted responses.
        """
        for response in inner(messages, **kwargs):
            pulled.append(1)
            yield response

    llm.stream_chat.side_effect = _stream
    return pulled


def test_a_tool_less_round_answers_from_pinned_evidence() -> None:
    """A pinned scope is answered in one round that offers no tools, from one user message holding the passages."""
    window = ConversationWindow(recent=(ConversationTurn("Wann ist das Hafenfest?", "Am Sonntag."),))
    search = QuerySearch({"Programm": [programme_passage()]})
    llm = streaming_tool_llm([_answer()])
    loop = _loop(llm, _documents(search), scope_note=SCOPE_NOTE)

    items = list(_turn(loop, window=window, pinned=[programme_passage(), weather_passage()]))

    progress, texts, result = _drain(items)
    seed = AgentProgress(step=0, action="seed", detail="", found=2)
    assert (items[0], progress, "".join(texts)) == (seed, [seed], ANSWER)
    assert [(scored.node.node_id, scored.node.metadata["citation_index"]) for scored in result.nodes] == [
        ("programme", 1),
        ("weather", 2),
    ]
    assert (result.raw_text, result.steps, result.reason, result.images_attached) == (ANSWER, 0, None, 0)
    assert result.actions == [{"action": "seed", "detail": "", "found": 2}]
    (call,) = llm.stream_chat.call_args_list
    assert call.kwargs == {}
    assert [(message.role, message.content) for message in _sent(llm, 0)] == [
        (MessageRole.SYSTEM, SYSTEM_PROMPT),
        (MessageRole.USER, "Wann ist das Hafenfest?"),
        (MessageRole.ASSISTANT, "Am Sonntag."),
        (MessageRole.USER, _pinned_message(result.nodes)),
    ]
    assert search.queries == []


def test_an_oversized_scope_raises_before_any_output() -> None:
    """Pinned passages that cannot all be shown legibly in the round's room raise before the turn shows anything.

    The ledger holds them all, as the caller sizes it for a scope: what does not fit is the room.
    """
    llm = streaming_tool_llm([_answer()])
    pinned = [
        scored_node(f"plan-{number}", HARBOUR_PLAN, filename="hafenplan.pdf", page=number) for number in range(1, 51)
    ]
    ledger = EvidenceLedger(cap=len(pinned))

    shown, raised = _until_raised(
        _turn(_loop(llm, _documents(QuerySearch({})), scope_note=SCOPE_NOTE), ledger=ledger, pinned=pinned)
    )

    assert (shown, type(raised), len(ledger)) == ([], EvidenceDoesNotFit, 50)
    assert isinstance(raised, ValueError)
    assert "context window" in str(raised)
    assert cast(EvidenceDoesNotFit, raised).passages == 50
    llm.stream_chat.assert_not_called()


def test_a_ledger_too_small_for_the_pinned_passages_raises_before_any_output() -> None:
    """Pinned evidence is never dropped: a ledger that cannot hold every pinned passage raises before the round."""
    regatta = scored_node("regatta", "Die Regatta startet um drei Uhr am Kai.", filename="regatta.pdf", page=1)
    llm = streaming_tool_llm([_answer()])

    shown, raised = _until_raised(
        _turn(
            _loop(llm, _documents(QuerySearch({}))),
            ledger=EvidenceLedger(cap=2),
            pinned=[programme_passage(), weather_passage(), regatta],
        )
    )

    assert (shown, type(raised)) == ([], EvidenceDoesNotFit)
    assert cast(EvidenceDoesNotFit, raised).passages == 3
    llm.stream_chat.assert_not_called()


def _note_of(tokens: int) -> str:
    """Write an invented scope note that costs exactly ``tokens`` at 3.5 characters a token.

    Args:
        tokens (int): Its cost.

    Returns:
        str: The note.
    """
    return (SCOPE_NOTE * (tokens // 20 + 1))[: int(3.5 * tokens)]


def test_the_pinned_passages_get_the_loops_room_less_the_scope_note_to_the_token() -> None:
    """The passages' room is the loop's room, the final note kept out, less the scope note and its envelope.

    The note is sized to leave the passages exactly the least room they are legible in, then a token less.
    """
    pinned = [scored_node(f"plan-{number}", HARBOUR_PLAN, filename="hafenplan.pdf", page=number) for number in range(5)]
    ledger = EvidenceLedger(cap=len(pinned))
    ledger.add(pinned)
    numbered = ledger.numbered()

    def _legible(budget: int) -> bool:
        """Say whether the passages are legible in a budget.

        Args:
            budget (int): Tokens the rendered passages may use.

        Returns:
            bool: Whether every passage keeps its floor.
        """
        return render_context(numbered, llm_view=llm_view, budget_tokens=budget, char_token_ratio=3.5).legible

    least = bisect.bisect_left(range(10_000), True, key=_legible)
    spare = turn_budget().round_room - message_cost(FINAL_NOTE, 3.5) - least - MESSAGE_OVERHEAD_TOKENS

    def _scoped(note: str) -> Iterator[AgentProgress | str | AgentResult]:
        """Run the scoped turn with a note.

        Args:
            note (str): The scope note.

        Returns:
            Iterator[AgentProgress | str | AgentResult]: The turn's output.
        """
        return _turn(
            _loop(streaming_tool_llm([_answer()]), _documents(QuerySearch({})), scope_note=note), pinned=pinned
        )

    _, _, result = _drain(_scoped(_note_of(spare)))
    shown, raised = _until_raised(_scoped(_note_of(spare + 1)))

    assert (_legible(least), _legible(least - 1), len(result.nodes)) == (True, False, 5)
    assert (shown, type(raised)) == ([], EvidenceDoesNotFit)


@pytest.mark.parametrize("answer", [ANSWER, SHORT_ANSWER], ids=["past-the-hold", "within-the-hold"])
def test_the_seed_step_comes_with_the_rounds_first_event(answer: str) -> None:
    """The seed step goes out once the round has begun to answer, ahead of the answer's first text.

    Args:
        answer (str): The reply, long enough to stream as it comes or short enough to be held whole.
    """
    llm = streaming_tool_llm([_answer(answer)])
    pulled = _pulls(llm)
    turn = _turn(_loop(llm, _documents(QuerySearch({}))), pinned=[programme_passage()])

    first = next(turn)
    begun = len(pulled)
    rest = list(turn)

    assert (first, begun > 0) == (AgentProgress(step=0, action="seed", detail="", found=1), True)
    assert "".join(cast(list[str], rest[:-1])) == answer
    assert isinstance(rest[-1], AgentResult)


def test_the_seed_step_counts_what_the_scope_admitted_or_extended() -> None:
    """As a search step does, the seed counts each pinned window admitted or extended: one section, two windows."""
    opening = scored_node("programme", PROGRAMME[:66], filename="hafenfest_programm.pdf", page=1)
    closing = scored_node("programme", PROGRAMME[67:], filename="hafenfest_programm.pdf", page=1)
    llm = streaming_tool_llm([_answer()])

    progress, _, result = _drain(_turn(_loop(llm, _documents(QuerySearch({}))), pinned=[opening, closing]))

    assert progress == [AgentProgress(step=0, action="seed", detail="", found=2)]
    assert [scored.node.node_id for scored in result.nodes] == ["programme"]


@pytest.mark.parametrize(
    ("error", "raised"),
    [
        (bad_request(OVERFLOW), EvidenceDoesNotFit),
        (bad_request(TOOL_CHOICE_REJECTED), openai.BadRequestError),
        (_status_error(openai.RateLimitError, 429, "rate limit reached"), openai.RateLimitError),
        (RuntimeError("invented outage"), RuntimeError),
        ([chunk("Am Sonntag"), RuntimeError("invented outage")], RuntimeError),
    ],
    ids=["overflow", "refused", "rate-limit", "outage", "broken-before-any-event"],
)
def test_a_pinned_round_that_fails_before_its_first_event_shows_nothing(
    error: Any, raised: type[BaseException]
) -> None:
    """Before the round's first event nothing is shown, so the caller can still answer the scope another way.

    An overflow raises ``EvidenceDoesNotFit``, which sends the caller down its scoped path; any
    other failure is raised as it came, a refusal included: the round offered no tools.

    Args:
        error (Any): What the endpoint raises, on its first pull or before the round's first event.
        raised (type[BaseException]): What the turn raises.
    """
    llm = streaming_tool_llm([error])

    shown, exc = _until_raised(_turn(_loop(llm, _documents(QuerySearch({}))), pinned=[programme_passage()]))

    assert (shown, type(exc), llm.stream_chat.call_count) == ([], raised, 1)
    cause = error[-1] if isinstance(error, list) else error
    assert (exc.__cause__ if raised is EvidenceDoesNotFit else exc) is cause


@pytest.mark.parametrize(("budget", "wait"), [(120.0, MODEL_SLOT_WAIT_S), (2.0, 2.0)])
def test_a_pinned_round_without_a_model_slot_raises_before_anything_is_yielded(budget: float, wait: float) -> None:
    """The pinned round waits for a model slot as a first round does, never past the time; without one, nothing shows.

    Args:
        budget (float): The turn's time budget in seconds.
        wait (float): The longest the round may wait for a slot.
    """
    slots = _Recording()
    llm = streaming_tool_llm([_answer()])
    loop = _loop(llm, _documents(QuerySearch({})), model_slots=slots, time_budget_s=budget, clock=lambda: 0.0)

    shown, raised = _until_raised(_turn(loop, pinned=[programme_passage()]))

    assert (shown, type(raised), slots.waits) == ([], NoSlot, [wait])
    llm.stream_chat.assert_not_called()


@pytest.mark.parametrize(
    "error", [RuntimeError("invented outage"), bad_request(OVERFLOW)], ids=["outage", "overflow-mid-answer"]
)
def test_a_pinned_answer_that_breaks_off_raises_after_its_text(error: Exception) -> None:
    """An answer the endpoint stops sending partway is not passed off as whole, whatever stopped it.

    Args:
        error (Exception): What breaks the stream once text has gone out.
    """
    llm = streaming_tool_llm([[*text_chunks(ANSWER, 20)[:7], error]])

    shown, exc = _until_raised(_turn(_loop(llm, _documents(QuerySearch({}))), pinned=[programme_passage()]))

    assert exc is error
    assert shown[0] == AgentProgress(step=0, action="seed", detail="", found=1)
    assert len(shown) > 1 and all(isinstance(item, str) for item in shown[1:])


def test_a_pinned_round_that_overflows_after_its_seed_reports_the_context_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once the seed is shown the caller cannot fall back: an overflow before any text is the context window's.

    The answer round reports it so too. Endpoints report an overflow before they stream, so the
    round is faked: its first event is text the splitter holds back, then the overflow.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces the round's stream.
    """
    overflow = bad_request(OVERFLOW)

    def _held_then_overflowing(llm: Any, messages: Any, **kwargs: Any) -> Iterator[Provisional]:
        """Release a fragment that could open the sentinel, then fail as an overflowing request does.

        Args:
            llm (Any): Unused.
            messages (Any): Unused.
            **kwargs (Any): Unused.

        Yields:
            Provisional: The fragment, which the splitter holds back.

        Raises:
            openai.BadRequestError: The overflow.
        """
        yield Provisional(SENTINEL[:6])
        raise overflow

    monkeypatch.setattr(turn_loop, "stream_round", _held_then_overflowing)

    shown, raised = _until_raised(
        _turn(_loop(streaming_tool_llm([]), _documents(QuerySearch({}))), pinned=[programme_passage()])
    )

    assert shown == [AgentProgress(step=0, action="seed", detail="", found=1)]
    assert (type(raised), raised.__cause__) == (ValueError, overflow)
    assert "context window" in str(raised)


def test_a_pinned_answer_cut_at_the_length_limit_is_kept_and_flagged() -> None:
    """The pinned round is an answer round: an answer cut at the length limit stays, flagged ``answer_cut``."""
    cut = ANSWER[:90]
    llm = streaming_tool_llm([[*text_chunks(cut, 30), chunk(finish_reason="length")]])

    _, texts, result = _drain(_turn(_loop(llm, _documents(QuerySearch({}))), pinned=[programme_passage()]))

    assert ("".join(texts), result.raw_text, result.reason) == (cut, cut, "answer_cut")


def test_an_empty_scope_is_still_answered_without_tools() -> None:
    """A scope that pins nothing is still a scope: it is never widened into a turn that searches."""
    llm = streaming_tool_llm([_answer(SHORT_ANSWER)])

    progress, _, result = _drain(_turn(_loop(llm, _documents(QuerySearch({}))), pinned=[]))

    assert progress == [AgentProgress(step=0, action="seed", detail="", found=0)]
    assert (result.nodes, llm.stream_chat.call_args.kwargs) == ([], {})


# --- Pictures for the visual target ------------------------------------------------------------

LEGEND = "Die Bilder unten sind die Quellen, als Bilder beigefügt:\n{image_legend}"
#: How an endpoint that caps the pictures in one prompt refuses a request carrying more.
REFUSED_PICTURES = "At most 1 image(s) may be provided in one prompt; zzsecretzz"


def _pictures(results: Mapping[str, list[NodeWithScore]]) -> ToolRegistry:
    """Offer an image search and a read, as a visual turn offers them.

    Args:
        results (Mapping[str, list[NodeWithScore]]): The frames each description finds.

    Returns:
        ToolRegistry: ``search_images`` and ``read``.
    """
    return ToolRegistry([SearchImagesTool(QuerySearch(results)), ReadTool()])


def _image_call(call_id: str, description: str) -> tuple[str, str, Any]:
    """Name one ``search_images`` call.

    Args:
        call_id (str): The call's id.
        description (str): The visible cues to look for.

    Returns:
        tuple[str, str, Any]: The call, as :func:`_calls` takes it.
    """
    return (call_id, "search_images", {"description": description})


def _visual(llm: Any, registry: ToolRegistry, store: ThumbnailStore, **overrides: Any) -> AgentLoop:
    """Build an agent loop that shows the model the pictures its rounds find.

    Args:
        llm (Any): The model double.
        registry (ToolRegistry): The tools offered.
        store (ThumbnailStore): The thumbnail fetcher.
        **overrides (Any): Constructor arguments to change; six pictures at most by default.

    Returns:
        AgentLoop: The loop.
    """
    return _loop(llm, registry, fetch_thumbnails=store, legend_template=LEGEND, **{"max_images": 6, **overrides})


def _picture_messages(llm: Any, call: int) -> list[ChatMessage]:
    """Return the messages of one model call that carry pictures.

    Args:
        llm (Any): The model double.
        call (int): The call's position, from zero.

    Returns:
        list[ChatMessage]: The picture messages, in order.
    """
    return [message for message in _sent(llm, call) if any(isinstance(b, ImageBlock) for b in message.blocks)]


def _wire(message: ChatMessage) -> dict[str, Any]:
    """Serialise one message as the OpenAI client sends it.

    Args:
        message (ChatMessage): The message.

    Returns:
        dict[str, Any]: Its wire form.
    """
    wire = to_openai_message_dicts([message])
    assert isinstance(wire, list)
    return cast(dict[str, Any], wire[0])


def _picture_wire(legend: list[tuple[int, int]], *, note: str = "") -> dict[str, Any]:
    """Build the wire form of a picture message: the legend, each picture, and a joined note, if any.

    Args:
        legend (list[tuple[int, int]]): Per picture, the number it is cited by and its frame.
        note (str): The final note joined to the message; none when empty.

    Returns:
        dict[str, Any]: The message as the OpenAI client sends it.
    """
    text = render_legend(LEGEND, [(number, *thumbnail(frame)) for number, frame in legend])
    parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
    parts.extend({"type": "image_url", "image_url": {"url": thumbnail_url(frame)}} for _, frame in legend)
    if note:
        parts.append({"type": "text", "text": note})
    return {"role": "user", "content": parts}


def _stamped(result: AgentResult) -> list[int]:
    """Return the numbers of the passages the result marks as shown to the model as pictures.

    Args:
        result (AgentResult): The turn's result.

    Returns:
        list[int]: The stamped passages' numbers.
    """
    return [
        scored.node.metadata["citation_index"]
        for scored in result.nodes
        if scored.node.metadata.get(VISUAL_IMAGES_ATTACHED_KEY)
    ]


@pytest.mark.parametrize(
    ("overrides", "shown", "reason"),
    [
        ({"max_images": 2}, [(1, 0), (2, 1)], None),
        ({"budget": _budget_with_room(1480)}, [(1, 0)], "context_budget"),
    ],
    ids=["by-max-images", "by-the-room-left"],
)
def test_visual_rounds_attach_thumbnails_within_the_allowance(
    overrides: dict[str, Any], shown: list[tuple[int, int]], reason: str | None
) -> None:
    """A round's new pictures are shown after its tool replies, as many as the allowance and the room hold.

    Pictures may use the room up: the next round is then the forced answer, its note joined to them.

    Args:
        overrides (dict[str, Any]): What limits the pictures: the turn's most, or the room left
            (1233 tokens once the round's own messages are paid: a message of one picture costs
            662 as sent and one of two 1268, though the turn's room allows two and its 1420
            before the round would hold them).
        shown (list[tuple[int, int]]): Per picture shown, the number it is cited by and its frame.
        reason (str | None): Why the turn answered: by choice, or for want of room.
    """
    next_choice = "auto" if reason is None else "none"
    frames = [frame_passage(0), frame_passage(1), frame_passage(2)]
    store = ThumbnailStore(0, 1, 2)
    llm = streaming_tool_llm([_calls(_image_call("call_a", "Regenjacken")), _answer()])

    _, texts, result = _drain(_turn(_visual(llm, _pictures({"Regenjacken": frames}), store, **overrides)))

    assert store.fetches == [["frame-0", "frame-1", "frame-2"]]
    sent = _sent(llm, 1)
    roles = [MessageRole.SYSTEM, MessageRole.USER, MessageRole.ASSISTANT, MessageRole.TOOL, MessageRole.USER]
    assert [message.role for message in sent] == roles
    note = FINAL_NOTE if next_choice == "none" else ""
    assert _wire(sent[-1]) == _picture_wire(shown, note=note)
    assert (llm.stream_chat.call_args_list[1].kwargs["tool_choice"], result.reason) == (next_choice, reason)
    assert (result.images_attached, _stamped(result), "".join(texts)) == (len(shown), [n for n, _ in shown], ANSWER)
    assert all(VISUAL_IMAGES_ATTACHED_KEY not in frame.node.metadata for frame in frames)


#: A legend long enough that the ratio a picture message is priced at decides whether it fits.
WORDY_LEGEND = "Die Bilder unten sind die Quellen dieser Antwort, als Bilder beigefügt. " * 40 + "{image_legend}"


def test_pictures_are_priced_at_the_ratio_the_turn_was_measured_with() -> None:
    """A turn measured at 5.0 characters a token prices its picture message at 5.0, not at a ratio of its own.

    The attach is offered 1345 tokens. One picture's message costs 1220 there at 5.0, so it is
    shown; at 3.5 it would cost 1469 and at 1.0 3539, and nothing would be shown or fetched.
    """
    store = ThumbnailStore(0)
    llm = streaming_tool_llm([_calls(_image_call("call_a", "Regenjacken")), _answer()])
    loop = _loop(
        llm,
        _pictures({"Regenjacken": [frame_passage(0)]}),
        budget=_budget_with_room(1516, ratio=5.0),
        char_token_ratio=5.0,
        fetch_thumbnails=store,
        legend_template=WORDY_LEGEND,
        max_images=6,
    )

    _, _, result = _drain(_turn(loop))

    assert (store.fetches, len(_picture_messages(llm, 1)), result.images_attached) == ([["frame-0"]], 1, 1)


def test_a_forced_answer_after_a_picture_round_joins_its_note_to_the_picture_message() -> None:
    """No two user messages in a row: the final note joins the picture message, after its pictures."""
    store = ThumbnailStore(0)
    llm = streaming_tool_llm([_calls(_image_call("call_a", "Regenjacken")), _answer()])

    _, _, result = _drain(_turn(_visual(llm, _pictures({"Regenjacken": [frame_passage(0)]}), store, max_rounds=1)))

    sent = _sent(llm, 1)
    assert [message.role for message in sent][-2:] == [MessageRole.TOOL, MessageRole.USER]
    assert _wire(sent[-1]) == _picture_wire([(1, 0)], note=FINAL_NOTE)
    assert (result.reason, result.images_attached) == ("round_budget", 1)


@pytest.mark.parametrize(("max_rounds", "choice"), [(6, "auto"), (1, "none")], ids=["tool-round", "answer-round"])
def test_a_refused_image_message_is_dropped_and_the_round_retried_once(
    max_rounds: int, choice: str, loguru_caplog: LogCaptureFixture
) -> None:
    """A request with pictures refused before its first event is sent once more without any; the answer follows.

    Args:
        max_rounds (int): Tool rounds allowed: the refused round is a tool round, or the forced answer.
        choice (str): The refused round's ``tool_choice``, which its retry keeps.
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    store = ThumbnailStore(0, 1)
    registry = _pictures({"Regenjacken": [frame_passage(0), frame_passage(1)]})
    llm = streaming_tool_llm([_calls(_image_call("call_a", "Regenjacken")), bad_request(REFUSED_PICTURES), _answer()])

    _, texts, result = _drain(_turn(_visual(llm, registry, store, max_rounds=max_rounds)))

    refused, retried = _sent(llm, 1), _sent(llm, 2)
    assert (len(_picture_messages(llm, 1)), _picture_messages(llm, 2)) == (1, [])
    # The same round without its picture message: the final note then stands alone.
    assert retried[:4] == refused[:4]
    note: list[tuple[MessageRole, str]] = [] if choice == "auto" else [(MessageRole.USER, FINAL_NOTE)]
    assert [(message.role, message.content) for message in retried[4:]] == note
    assert [call.kwargs["tool_choice"] for call in llm.stream_chat.call_args_list[1:]] == [choice, choice]
    assert (llm.stream_chat.call_count, "".join(texts)) == (3, ANSWER)
    assert (result.images_attached, _stamped(result)) == (0, [])
    assert loguru_caplog.messages == [
        "An agent round carrying 2 pictures failed before its first event (BadRequestError); "
        "it is sent again without them"
    ]
    assert "zzsecretzz" not in loguru_caplog.text


class _Clinging(TurnImages):
    """Pictures a drop cannot take back, so only the loop itself can bound the retry of a refused round."""

    @override
    def drop(self, transcript: Sequence[ChatMessage]) -> tuple[list[ChatMessage], int]:
        """Take nothing back.

        Args:
            transcript (Sequence[ChatMessage]): The turn's rounds so far.

        Returns:
            tuple[list[ChatMessage], int]: The transcript as it was, and no tokens freed.
        """
        return list(transcript), 0


def test_a_refused_round_is_sent_again_at_most_once_whatever_its_pictures_hold(
    monkeypatch: pytest.MonkeyPatch, loguru_caplog: LogCaptureFixture
) -> None:
    """The loop bounds the retry itself: refused again, the round fails as any round does, pictures held or not.

    A retry bounded only by the pictures would go on here; the script answers next, so it would
    show as a third send of the tool round, not as a hang.

    Args:
        monkeypatch (pytest.MonkeyPatch): Puts the clinging pictures in the loop's place.
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    monkeypatch.setattr(turn_loop, "TurnImages", _Clinging)
    llm = streaming_tool_llm(
        [
            _calls(_image_call("call_a", "Regenjacken")),
            bad_request(REFUSED_PICTURES),
            bad_request(REFUSED_PICTURES),
            _answer(),
        ]
    )

    _, texts, result = _drain(_turn(_visual(llm, _pictures({"Regenjacken": [frame_passage(0)]}), ThumbnailStore(0))))

    assert [call.kwargs["tool_choice"] for call in llm.stream_chat.call_args_list] == ["auto", "auto", "auto", "none"]
    assert (result.reason, "".join(texts)) == ("research_failed", ANSWER)
    assert loguru_caplog.messages == [
        "An agent round carrying 1 pictures failed before its first event (BadRequestError); "
        "it is sent again without them",
        "An agent round failed (BadRequestError); the model answers from what it found",
    ]


def test_a_turn_whose_pictures_were_refused_shows_no_more() -> None:
    """Once a refusal has taken the pictures back, later rounds' pictures are neither fetched nor shown."""
    store = ThumbnailStore(0, 1, 2)
    registry = _pictures({"Regenjacken": [frame_passage(0), frame_passage(1)], "Schirm": [frame_passage(2)]})
    llm = streaming_tool_llm(
        [
            _calls(_image_call("call_a", "Regenjacken")),
            bad_request(REFUSED_PICTURES),
            _calls(_image_call("call_b", "Schirm")),
            _answer(),
        ]
    )

    _, _, result = _drain(_turn(_visual(llm, registry, store)))

    assert store.fetches == [["frame-0", "frame-1"]]
    assert _picture_messages(llm, 3) == []
    assert (result.images_attached, [scored.node.node_id for scored in result.nodes]) == (
        0,
        ["frame-0", "frame-1", "frame-2"],
    )


def test_a_failed_thumbnail_fetch_shows_no_picture_and_the_turn_goes_on(loguru_caplog: LogCaptureFixture) -> None:
    """A store outage leaves the pictures out; the model answers from their captions, and nothing else changes.

    Args:
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    store = ThumbnailStore(0, failure=RuntimeError("zzsecretzz invented store outage"))
    llm = streaming_tool_llm([_calls(_image_call("call_a", "Regenjacken")), _answer()])

    _, texts, result = _drain(_turn(_visual(llm, _pictures({"Regenjacken": [frame_passage(0)]}), store)))

    assert (_picture_messages(llm, 1), llm.stream_chat.call_args_list[1].kwargs["tool_choice"]) == ([], "auto")
    assert (result.images_attached, result.reason, "".join(texts)) == (0, None, ANSWER)
    assert loguru_caplog.messages == ["An agent round's thumbnails could not be fetched (RuntimeError); none is shown"]


def test_a_picture_shown_before_is_not_attached_again() -> None:
    """A later round that names or reads a picture shown before adds only the pictures new to the turn."""
    store = ThumbnailStore(0, 1)
    registry = _pictures({"Regenjacken": [frame_passage(0)], "Schirm": [frame_passage(0), frame_passage(1)]})
    llm = streaming_tool_llm(
        [
            _calls(_image_call("call_a", "Regenjacken")),
            _calls(_image_call("call_b", "Schirm"), ("call_c", "read", {"n": 1})),
            _answer(),
        ]
    )

    _, _, result = _drain(_turn(_visual(llm, registry, store)))

    assert store.fetches == [["frame-0"], ["frame-1"]]
    assert [_wire(message) for message in _picture_messages(llm, 2)] == [
        _picture_wire([(1, 0)]),
        _picture_wire([(2, 1)]),
    ]
    assert (result.images_attached, _stamped(result)) == (2, [1, 2])


def test_a_retried_tool_round_that_fails_too_makes_the_model_answer_from_what_it_found() -> None:
    """The retry without pictures is the round's one: a tool round failing it too ends research, as any does."""
    store = ThumbnailStore(0)
    llm = streaming_tool_llm(
        [
            _calls(_image_call("call_a", "Regenjacken")),
            bad_request(REFUSED_PICTURES),
            RuntimeError("invented outage"),
            _answer(),
        ]
    )

    _, texts, result = _drain(_turn(_visual(llm, _pictures({"Regenjacken": [frame_passage(0)]}), store)))

    assert (llm.stream_chat.call_count, llm.stream_chat.call_args_list[3].kwargs["tool_choice"]) == (4, "none")
    assert _picture_messages(llm, 3) == []
    assert (result.reason, result.images_attached, "".join(texts)) == ("research_failed", 0, ANSWER)


def test_a_retried_answer_round_that_fails_too_raises_its_error() -> None:
    """The retry without pictures is the round's one: an answer round failing it too raises its error as it came."""
    outage = RuntimeError("invented outage")
    store = ThumbnailStore(0)
    llm = streaming_tool_llm([_calls(_image_call("call_a", "Regenjacken")), bad_request(REFUSED_PICTURES), outage])
    turn = _turn(_visual(llm, _pictures({"Regenjacken": [frame_passage(0)]}), store, max_rounds=1))

    shown, raised = _until_raised(turn)

    assert (raised, llm.stream_chat.call_count) == (outage, 3)
    assert not any(isinstance(item, (str, AgentResult)) for item in shown)


def test_a_round_that_fails_after_its_first_event_keeps_its_pictures() -> None:
    """Only a failure before the first event refuses the pictures: a round that broke off mid-text is withdrawn."""
    store = ThumbnailStore(0)
    broken = [*text_chunks(ANSWER, 20)[:7], RuntimeError("invented outage")]
    llm = streaming_tool_llm([_calls(_image_call("call_a", "Regenjacken")), broken, _answer(SHORT_ANSWER)])

    progress, _, result = _drain(_turn(_visual(llm, _pictures({"Regenjacken": [frame_passage(0)]}), store)))

    assert [step.action for step in progress] == ["search_images", "note"]
    assert _wire(_sent(llm, 2)[-1]) == _picture_wire([(1, 0)], note=FINAL_NOTE)
    assert (result.reason, result.images_attached, result.raw_text) == ("research_failed", 1, SHORT_ANSWER)


def test_a_pinned_scope_shows_no_picture() -> None:
    """A scope outranks the retrieval target: pinned pictures are answered from their captions, none fetched."""
    store = ThumbnailStore(0, 1)
    llm = streaming_tool_llm([_answer()])

    _, _, result = _drain(_turn(_visual(llm, _pictures({}), store), pinned=[frame_passage(0), frame_passage(1)]))

    assert (store.fetches, _picture_messages(llm, 0)) == ([], [])
    assert (result.images_attached, _stamped(result)) == (0, [])


def test_a_refusal_gives_the_pictures_room_back() -> None:
    """Pictures taken back no longer count against the room: a later round may still call tools.

    At this room a round needs 938 tokens: one is left 1001 with the picture's message (662) shown,
    then 1536 after the refusal and another round, which would be 874 with that 662 still counted.
    """
    store = ThumbnailStore(0, 1)
    registry = _pictures({"Regenjacken": [frame_passage(0)], "Schirm": [frame_passage(1)]})
    llm = streaming_tool_llm(
        [
            _calls(_image_call("call_a", "Regenjacken")),
            bad_request(REFUSED_PICTURES),
            _calls(_image_call("call_b", "Schirm")),
            _answer(),
        ]
    )

    _, _, result = _drain(_turn(_visual(llm, registry, store, budget=_budget_with_room(1855))))

    assert [call.kwargs["tool_choice"] for call in llm.stream_chat.call_args_list] == ["auto"] * 4
    assert (result.reason, result.steps, result.images_attached) == (None, 2, 0)


def test_an_overflow_with_pictures_is_taken_as_their_refusal_first() -> None:
    """A round with pictures that overflows is sent again without them before any tool reply is cut."""
    store = ThumbnailStore(0)
    llm = streaming_tool_llm([_calls(_image_call("call_a", "Regenjacken")), bad_request(OVERFLOW), _answer()])

    _, texts, result = _drain(_turn(_visual(llm, _pictures({"Regenjacken": [frame_passage(0)]}), store)))

    assert _tool_replies(llm, 2) == _tool_replies(llm, 1)
    assert llm.stream_chat.call_args_list[2].kwargs["tool_choice"] == "auto"
    assert (result.reason, result.images_attached, "".join(texts)) == (None, 0, ANSWER)
