"""Round streaming and the tool-call accumulator. Every message, query and call here is invented."""

from __future__ import annotations

import random
from typing import Any

import openai
import pytest
from _pytest.logging import LogCaptureFixture
from agent_fakes import chunk, streaming_tool_llm, text_chunks, tool_call_delta
from assessment_fakes import bad_request, sse_endpoint
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from docint.agents.turn.stream import (
    HOLD_CHARS,
    Provisional,
    RoundDone,
    ToolCall,
    ToolCallAccumulator,
    stream_round,
)
from docint.utils.llm_sanitize import strip_reasoning
from docint.utils.openai_cfg import LocalOpenAI

QUESTION = "Wann findet das Hafenfest statt, und wer ist auf den Bildern vom Hafen zu sehen?"
MESSAGES = [ChatMessage(role=MessageRole.USER, content=QUESTION)]


def _tool(name: str, description: str, argument: str) -> dict[str, Any]:
    """Build the schema of a tool that takes one text argument, as the endpoint receives it.

    Args:
        name (str): The tool's name.
        description (str): What the tool is for.
        argument (str): The one argument's name.

    Returns:
        dict[str, Any]: The function-tool schema.
    """
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {argument: {"type": "string", "description": f"The {argument}."}},
                "required": [argument],
            },
        },
    }


SCHEMAS = [
    _tool("search", "Search the collection and return numbered passages.", "query"),
    _tool("search_images", "Find pictures and video frames that match a description.", "description"),
]
SEARCH_ARGS = '{"query": "Hafenfest Sonntag Sturmwarnung"}'
IMAGE_ARGS = '{"description": "Zwei Personen in Regenjacken am Hafen, bedeckter Himmel"}'
SEARCH_CALL = ToolCall(id="call_a", name="search", arguments=SEARCH_ARGS)
IMAGE_CALL = ToolCall(id="call_b", name="search_images", arguments=IMAGE_ARGS)

#: 63 characters: a preface the hold keeps back.
PREFACE = "Ich suche zuerst nach dem Hafenfest und nach Bildern vom Hafen."
#: 157 characters: a preface longer than the hold.
LONG_PREFACE = (
    "Ich suche zuerst nach dem Hafenfest, danach nach Bildern vom Hafen und zuletzt nach "
    "Aufnahmen aus dem Clip harbour_walk.webm, um beide Fragen zu beantworten."
)
#: 214 characters: an answer that outlasts the hold.
ANSWER = (
    "Das Hafenfest wurde wegen der Sturmwarnung auf Sonntag verlegt [1]. "
    "Die Stände öffnen um zehn Uhr, Musik gibt es ab Mittag [1]. "
    "Auf dem Bild vom Hafen stehen zwei Personen in Regenjacken unter bedecktem Himmel [2]."
)
#: 146 characters of reasoning, more than the hold on its own.
REASONING = (
    "Die Frage hat zwei Teile: das Datum des Festes und die Personen auf den Bildern. "
    "Ich durchsuche zuerst die Texte und danach die Bilder vom Hafen. "
)
#: The chunk an endpoint sends last when asked for usage: no choices at all.
USAGE_CHUNK: dict[str, Any] = {
    "id": "chatcmpl-invented",
    "object": "chat.completion.chunk",
    "created": 0,
    "model": "invented-model",
    "choices": [],
    "usage": {"prompt_tokens": 40, "completion_tokens": 30, "total_tokens": 70},
}
#: ``ANSWER`` streamed in 30-character chunks: the held 120 at once, then a piece per chunk. The
#: space at 209 goes out with the next piece, since trailing whitespace waits for text to follow it.
ANSWER_EVENTS = [ANSWER[:120], ANSWER[120:150], ANSWER[150:180], ANSWER[180:209], ANSWER[209:]]


def _whole(call: ToolCall, index: int | None) -> dict[str, Any]:
    """Build the tool-call entry of a call sent whole, as Ollama sends each call.

    Args:
        call (ToolCall): The call.
        index (int | None): Its index; left out when ``None``.

    Returns:
        dict[str, Any]: The entry.
    """
    return tool_call_delta(index=index, call_id=call.id, name=call.name, arguments=call.arguments)


def _accumulate(*chunks: dict[str, Any]) -> list[ToolCall]:
    """Feed raw chunks to a fresh accumulator and read its calls.

    Args:
        *chunks (dict[str, Any]): The chunks, in arrival order.

    Returns:
        list[ToolCall]: The joined calls.
    """
    accumulator = ToolCallAccumulator()
    for raw in chunks:
        accumulator.feed(raw)
    return accumulator.calls()


def _round(llm: Any, **kwargs: Any) -> tuple[list[str], RoundDone]:
    """Stream one round and split its events into the provisional texts and the closing event.

    Args:
        llm (Any): The model.
        **kwargs (Any): Further arguments for :func:`stream_round`.

    Returns:
        tuple[list[str], RoundDone]: Each provisional text in order, and how the round ended.
    """
    *provisional, done = list(stream_round(llm, MESSAGES, tools=SCHEMAS, **kwargs))
    assert isinstance(done, RoundDone)
    texts: list[str] = []
    for event in provisional:
        assert isinstance(event, Provisional)
        texts.append(event.text)
    return texts, done


def _scripted(chunks: list[dict[str, Any]], **kwargs: Any) -> tuple[list[str], RoundDone]:
    """Stream one round of a scripted model that sends ``chunks``.

    Args:
        chunks (list[dict[str, Any]]): The raw chunks, in order.
        **kwargs (Any): Further arguments for :func:`stream_round`.

    Returns:
        tuple[list[str], RoundDone]: Each provisional text in order, and how the round ended.
    """
    return _round(streaming_tool_llm([chunks]), **kwargs)


@pytest.mark.parametrize(
    ("first", "second"),
    [(0, 1), (None, None), (0, 0)],
    ids=["indexed", "without-index", "index-reused"],
)
def test_two_whole_calls_in_one_chunk_both_survive(first: int | None, second: int | None) -> None:
    """Every entry of a chunk is read, so a chunk carrying two whole calls yields both.

    llama-index reads only the first entry of each chunk and drops the second call.

    Args:
        first (int | None): The first call's index.
        second (int | None): The second call's index.
    """
    calls = _accumulate(chunk(tool_calls=[_whole(SEARCH_CALL, first), _whole(IMAGE_CALL, second)]))

    assert calls == [SEARCH_CALL, IMAGE_CALL]


def test_argument_fragments_join_by_index() -> None:
    """Fragments join the call their index names, even when two calls' fragments interleave."""
    calls = _accumulate(
        chunk(tool_calls=[tool_call_delta(index=0, call_id="call_a", name="search", arguments="")]),
        chunk(tool_calls=[tool_call_delta(index=0, arguments=SEARCH_ARGS[:11])]),
        chunk(tool_calls=[tool_call_delta(index=1, call_id="call_b", name="search_images", arguments=IMAGE_ARGS[:9])]),
        chunk(tool_calls=[tool_call_delta(index=0, arguments=SEARCH_ARGS[11:])]),
        chunk(tool_calls=[tool_call_delta(index=1, arguments=IMAGE_ARGS[9:])]),
        chunk(finish_reason="tool_calls"),
    )

    assert calls == [SEARCH_CALL, IMAGE_CALL]


def test_a_delta_without_index_but_a_new_id_starts_a_call() -> None:
    """Without an index, a new id starts a call; a known id, or no id at all, continues one.

    llama-index merges a second index-less call into the first.
    """
    calls = _accumulate(
        chunk(tool_calls=[tool_call_delta(call_id="call_a", name="search", arguments=SEARCH_ARGS[:11])]),
        chunk(tool_calls=[tool_call_delta(arguments=SEARCH_ARGS[11:])]),
        chunk(tool_calls=[tool_call_delta(call_id="call_b", name="search_images", arguments=IMAGE_ARGS[:9])]),
        chunk(tool_calls=[tool_call_delta(call_id="call_b", arguments=IMAGE_ARGS[9:])]),
    )

    assert calls == [SEARCH_CALL, IMAGE_CALL]


def test_calls_without_a_name_are_dropped() -> None:
    """A call that never received a name, or only a blank one, is dropped and counted."""
    accumulator = ToolCallAccumulator()
    accumulator.feed(
        chunk(
            tool_calls=[
                tool_call_delta(index=0, call_id="call_x", arguments=SEARCH_ARGS),
                _whole(IMAGE_CALL, 1),
                tool_call_delta(index=2, call_id="call_y", name="  ", arguments="{}"),
            ]
        )
    )

    assert accumulator.calls() == [IMAGE_CALL]
    assert accumulator.dropped == 2


@pytest.mark.parametrize(
    ("ids", "expected"),
    [
        ((None, None), ["call_0", "call_1"]),
        (("call_a", "call_a"), ["call_a", "call_1"]),
        ((None, "call_0"), ["call_0_1", "call_0"]),
    ],
    ids=["missing", "repeated", "already-taken"],
)
def test_every_call_gets_an_id_no_other_call_in_the_round_holds(
    ids: tuple[str | None, str | None], expected: list[str]
) -> None:
    """A call without an id, or with one an earlier call holds, is given ``call_<position>``, kept unique.

    Args:
        ids (tuple[str | None, str | None]): The ids the two calls arrive with.
        expected (list[str]): The ids they leave with.
    """
    calls = _accumulate(
        chunk(
            tool_calls=[
                tool_call_delta(index=0, call_id=ids[0], name="search", arguments=SEARCH_ARGS),
                tool_call_delta(index=1, call_id=ids[1], name="search_images", arguments=IMAGE_ARGS),
            ]
        )
    )

    assert [call.id for call in calls] == expected
    assert [(call.name, call.arguments) for call in calls] == [("search", SEARCH_ARGS), ("search_images", IMAGE_ARGS)]


def test_feed_reports_whether_a_chunk_carried_a_call() -> None:
    """Text, a usage chunk without choices and a missing raw chunk carry no call and change none."""
    accumulator = ToolCallAccumulator()

    assert accumulator.feed(chunk(tool_calls=[_whole(SEARCH_CALL, 0)])) is True
    assert accumulator.feed(chunk(PREFACE)) is False
    assert accumulator.feed(USAGE_CHUNK) is False
    assert accumulator.feed(None) is False
    assert accumulator.calls() == [SEARCH_CALL]


@pytest.mark.parametrize(
    ("chunks", "text"),
    [
        (
            [*text_chunks(PREFACE, 20), chunk(tool_calls=[_whole(SEARCH_CALL, 0), _whole(IMAGE_CALL, 1)])],
            PREFACE,
        ),
        (
            [
                *text_chunks(LONG_PREFACE[:100], 20),
                chunk(LONG_PREFACE[100:], tool_calls=[_whole(SEARCH_CALL, 0)]),
                chunk(tool_calls=[_whole(IMAGE_CALL, 1)]),
            ],
            LONG_PREFACE,
        ),
        (
            [
                *text_chunks(PREFACE, 20),
                chunk(tool_calls=[_whole(SEARCH_CALL, 0), _whole(IMAGE_CALL, 1)]),
                *text_chunks(ANSWER, 30),
            ],
            PREFACE + ANSWER,
        ),
    ],
    ids=["preface-then-calls", "hold-crossed-in-the-call-chunk", "text-after-the-calls"],
)
def test_held_preface_is_never_forwarded_when_calls_follow(chunks: list[dict[str, Any]], text: str) -> None:
    """Text held when a tool-call delta arrives is never released, nor is any text after it.

    Args:
        chunks (list[dict[str, Any]]): The round's chunks.
        text (str): Its visible text in full.
    """
    texts, done = _scripted([*chunks, chunk(finish_reason="tool_calls")])

    assert texts == []
    assert done == RoundDone(
        text=text, calls=[SEARCH_CALL, IMAGE_CALL], finish_reason="tool_calls", provisional_sent=False
    )


def test_text_past_the_hold_is_forwarded_provisionally() -> None:
    """Nothing goes out before 120 characters; then the held text goes out at once, and each later piece on arrival."""
    texts, done = _scripted([*text_chunks(ANSWER, 30), chunk(finish_reason="stop")])

    assert texts == ANSWER_EVENTS
    assert done == RoundDone(text=ANSWER, calls=[], finish_reason="stop", provisional_sent=True)


@pytest.mark.parametrize(("size", "released"), [(HOLD_CHARS - 1, False), (HOLD_CHARS, True)])
def test_the_hold_releases_at_exactly_hold_chars(size: int, released: bool) -> None:
    """A reply of 119 characters stays held for the caller to send; one of 120 goes out.

    Args:
        size (int): The reply's length.
        released (bool): Whether it is released provisionally.
    """
    text = ANSWER[:size]

    texts, done = _scripted(text_chunks(text, 1))

    assert HOLD_CHARS == 120
    assert texts == ([text] if released else [])
    assert (done.text, done.provisional_sent) == (text, released)


def test_calls_after_the_hold_leave_the_sent_text_provisional() -> None:
    """Text released before calls arrive is reported as sent, so the caller can withdraw it.

    Text after the calls stays held, the tail kept back as a possible tag included.
    """
    after = f"{PREFACE} <th"
    texts, done = _scripted(
        [
            *text_chunks(ANSWER, 30),
            chunk(tool_calls=[_whole(SEARCH_CALL, 0), _whole(IMAGE_CALL, 1)]),
            *text_chunks(after, 30),
            chunk(finish_reason="tool_calls"),
        ]
    )

    assert "".join(texts) == ANSWER
    assert done == RoundDone(
        text=ANSWER + after, calls=[SEARCH_CALL, IMAGE_CALL], finish_reason="tool_calls", provisional_sent=True
    )


@pytest.mark.parametrize(
    ("before", "after"),
    [(ANSWER[:180], ANSWER[180:]), (LONG_PREFACE[:100], LONG_PREFACE[100:])],
    ids=["after-the-hold", "within-the-hold"],
)
def test_an_answer_whose_calls_are_all_dropped_is_sent_in_full(before: str, after: str) -> None:
    """A nameless fragment holds the text as a call would; once it is dropped, the round is an answer and loses nothing.

    Released text then covers all of it, the part after the fragment included; with nothing
    released, the caller sends it all.

    Args:
        before (str): Text ahead of the nameless fragment.
        after (str): Text behind it.
    """
    texts, done = _scripted(
        [
            *text_chunks(before, 30),
            chunk(tool_calls=[tool_call_delta(index=0, call_id="call_x", arguments=SEARCH_ARGS)]),
            *text_chunks(after, 30),
            chunk(finish_reason="stop"),
        ]
    )

    assert (done.text, done.calls, done.provisional_sent) == (before + after, [], len(before) >= HOLD_CHARS)
    assert "".join(texts) == (done.text if done.provisional_sent else "")


@pytest.mark.parametrize(
    ("chunks", "text", "calls", "finish_reason"),
    [
        ([*text_chunks(ANSWER, 30), chunk(finish_reason="stop")], ANSWER, [], "stop"),
        (
            [*text_chunks(LONG_PREFACE, 30), chunk(tool_calls=[_whole(SEARCH_CALL, 0)], finish_reason="tool_calls")],
            LONG_PREFACE,
            [SEARCH_CALL],
            "tool_calls",
        ),
    ],
    ids=["answer", "preface-before-calls"],
)
def test_a_round_held_whole_releases_nothing_and_ends_with_all_of_its_text(
    chunks: list[dict[str, Any]], text: str, calls: list[ToolCall], finish_reason: str
) -> None:
    """Held whole, a round releases no text even past the hold: its end carries all of it, to send or to drop.

    Args:
        chunks (list[dict[str, Any]]): The round's chunks, its text longer than the hold.
        text (str): Its visible text in full.
        calls (list[ToolCall]): The calls it ends in, if any.
        finish_reason (str): Why it ended.
    """
    texts, done = _scripted(chunks, hold=True)

    assert len(text) > HOLD_CHARS
    assert texts == []
    assert done == RoundDone(text=text, calls=calls, finish_reason=finish_reason, provisional_sent=False)


@pytest.mark.parametrize("size", [1, 7, 4096])
def test_reasoning_in_think_tags_never_counts_toward_the_hold(size: int) -> None:
    """146 characters of thinking and a 63-character answer leave the answer held, the thinking gone.

    Args:
        size (int): Characters per chunk, so the tags arrive split or whole.
    """
    raw = f"<think>{REASONING}</think>{PREFACE}"

    texts, done = _scripted([*text_chunks(raw, size), chunk(finish_reason="stop")])

    assert texts == []
    assert (done.text, done.provisional_sent, done.reasoning) == (PREFACE, False, REASONING.strip())


@pytest.mark.parametrize("size", [1, 7, 4096])
def test_reasoning_in_think_tags_is_never_forwarded(size: int) -> None:
    """Only the answer goes out, in any letter case of the tags and wherever the thinking sits.

    Args:
        size (int): Characters per chunk, so the tags arrive split or whole.
    """
    raw = f"<think>{REASONING}</think>{ANSWER[:150]}<THINK>{REASONING}</Think>{ANSWER[150:]}"

    texts, done = _scripted([*text_chunks(raw, size), chunk(finish_reason="stop")])

    assert len(texts[0]) >= HOLD_CHARS
    assert "".join(texts) == ANSWER
    assert (done.text, done.provisional_sent) == (ANSWER, True)
    assert done.reasoning == f"{REASONING.strip()}\n{REASONING.strip()}"


def test_reasoning_left_open_at_the_end_is_never_shown() -> None:
    """A think block the reply never closes, as when it runs out of tokens mid-tag, stays hidden."""
    texts, done = _scripted([*text_chunks(f"{ANSWER}<think>{REASONING}</thi", 7), chunk(finish_reason="length")])

    assert "".join(texts) == ANSWER
    assert (done.text, done.finish_reason, done.reasoning) == (ANSWER, "length", f"{REASONING}</thi".strip())


def test_a_tail_that_never_becomes_a_tag_is_kept() -> None:
    """Characters held back as a possible tag are released once the reply ends without one."""
    texts, done = _scripted(text_chunks(f"{ANSWER} <thi", 1))

    assert "".join(texts) == f"{ANSWER} <thi"
    assert done.text == f"{ANSWER} <thi"


def test_stream_round_offers_the_tools_and_passes_options_through() -> None:
    """The model is called with the messages, the tools, ``tool_choice`` (``auto`` by default) and any options."""
    llm = streaming_tool_llm([[chunk(PREFACE, finish_reason="stop")], [chunk(PREFACE, finish_reason="stop")]])

    _round(llm)
    _round(llm, tool_choice="none", max_tokens=512)

    first, second = llm.stream_chat.call_args_list
    assert first.args == (MESSAGES,)
    assert first.kwargs == {"tools": SCHEMAS, "tool_choice": "auto"}
    assert second.kwargs == {"tools": SCHEMAS, "tool_choice": "none", "max_tokens": 512}


def test_a_round_offered_no_tools_sends_neither_tools_nor_tool_choice() -> None:
    """With no tool to offer, the request carries neither ``tools`` nor ``tool_choice``: an endpoint refuses ``[]``."""
    stream = [*text_chunks(ANSWER, 30), chunk(finish_reason="stop"), USAGE_CHUNK]

    with sse_endpoint(stream) as (base, requests):
        llm = LocalOpenAI(api_base=base, api_key="test", model="invented-model", max_retries=0)
        texts, done = _round_offering(llm, tools=[], tool_choice="none")

    assert ("".join(texts), done.text, done.calls) == (ANSWER, ANSWER, [])
    assert len(requests) == 1
    assert ("tools" in requests[0], "tool_choice" in requests[0]) == (False, False)
    assert requests[0]["messages"] == [{"role": "user", "content": QUESTION}]


def _round_offering(llm: Any, *, tools: list[dict[str, Any]], **kwargs: Any) -> tuple[list[str], RoundDone]:
    """Stream one round offering ``tools``, and split its events as :func:`_round` does.

    Args:
        llm (Any): The model.
        tools (list[dict[str, Any]]): The tool schemas offered.
        **kwargs (Any): Further arguments for :func:`stream_round`.

    Returns:
        tuple[list[str], RoundDone]: Each provisional text in order, and how the round ended.
    """
    *provisional, done = list(stream_round(llm, MESSAGES, tools=tools, **kwargs))
    assert isinstance(done, RoundDone)
    return [event.text for event in provisional if isinstance(event, Provisional)], done


def test_an_error_on_the_first_pull_raises_before_any_event() -> None:
    """An endpoint's HTTP error surfaces from the first pull, before the round has yielded anything."""
    events = stream_round(streaming_tool_llm([bad_request("tool calls are not enabled")]), MESSAGES, tools=SCHEMAS)

    with pytest.raises(openai.BadRequestError):
        next(events)


@pytest.mark.parametrize(
    ("chunks", "finish_reason"),
    [([*text_chunks(PREFACE, 20), chunk(finish_reason="length")], "length"), (text_chunks(PREFACE, 20), None)],
    ids=["sent", "never-sent"],
)
def test_a_round_reports_its_finish_reason(chunks: list[dict[str, Any]], finish_reason: str | None) -> None:
    """The finish reason is the endpoint's, or ``None`` when the stream ended without one.

    Args:
        chunks (list[dict[str, Any]]): The round's chunks.
        finish_reason (str | None): The reason reported.
    """
    _, done = _scripted(chunks)

    assert done.finish_reason == finish_reason


def test_a_dropped_call_is_logged_by_count(loguru_caplog: LogCaptureFixture) -> None:
    """A round that drops nameless calls says how many, and nothing of what they carried.

    Args:
        loguru_caplog (LogCaptureFixture): Bridged WARNING capture.
    """
    _, done = _scripted(
        [chunk(tool_calls=[tool_call_delta(index=0, call_id="call_x", arguments=SEARCH_ARGS), _whole(IMAGE_CALL, 1)])]
    )

    assert done.calls == [IMAGE_CALL]
    assert loguru_caplog.messages == ["A model round sent tool calls without a name; dropped: 1"]


#: Two calls as endpoints stream them: whole in one chunk, in fragments by index, and whole without an index.
SHAPES = {
    "ollama": [chunk(tool_calls=[_whole(SEARCH_CALL, 0), _whole(IMAGE_CALL, 1)])],
    "vllm": [
        chunk(tool_calls=[tool_call_delta(index=0, call_id="call_a", name="search", arguments="")]),
        chunk(tool_calls=[tool_call_delta(index=0, arguments=SEARCH_ARGS[:11])]),
        chunk(tool_calls=[tool_call_delta(index=0, arguments=SEARCH_ARGS[11:])]),
        chunk(tool_calls=[tool_call_delta(index=1, call_id="call_b", name="search_images", arguments="")]),
        chunk(tool_calls=[tool_call_delta(index=1, arguments=IMAGE_ARGS[:9])]),
        chunk(tool_calls=[tool_call_delta(index=1, arguments=IMAGE_ARGS[9:])]),
    ],
    "without-index": [chunk(tool_calls=[_whole(SEARCH_CALL, None)]), chunk(tool_calls=[_whole(IMAGE_CALL, None)])],
}


@pytest.mark.parametrize("shape", list(SHAPES))
def test_the_real_client_reads_both_calls_from_a_loopback_sse_stream(shape: str) -> None:
    """The production client over a loopback SSE stream: both calls survive, and the request offers the tools.

    The ``vllm`` shape also guards the reading order: the client grows an earlier chunk's
    call in place as later fragments arrive, so a chunk read late would count them twice.

    Args:
        shape (str): How the endpoint streams the two calls.
    """
    stream = [*text_chunks(PREFACE, 20), *SHAPES[shape], chunk(finish_reason="tool_calls"), USAGE_CHUNK]

    with sse_endpoint(stream) as (base, requests):
        llm = LocalOpenAI(api_base=base, api_key="test", model="invented-model", max_retries=0)
        texts, done = _round(llm)

    assert texts == []
    assert done == RoundDone(
        text=PREFACE, calls=[SEARCH_CALL, IMAGE_CALL], finish_reason="tool_calls", provisional_sent=False
    )
    assert len(requests) == 1
    assert (requests[0]["tools"], requests[0]["tool_choice"], requests[0]["stream"]) == (SCHEMAS, "auto", True)
    assert requests[0]["messages"] == [{"role": "user", "content": QUESTION}]


@pytest.mark.parametrize(
    ("after", "texts", "text", "calls"),
    [
        (
            [*text_chunks(PREFACE, 20), chunk(tool_calls=[_whole(SEARCH_CALL, 0)])],
            [],
            PREFACE,
            [SEARCH_CALL],
        ),
        (text_chunks(ANSWER, 30), ANSWER_EVENTS, ANSWER, []),
    ],
    ids=["then-calls", "then-answer"],
)
@pytest.mark.parametrize("reasoning_open", [False, True], ids=["plain", "may-open-in-reasoning"])
def test_a_reasoning_field_never_counts_toward_the_hold(
    after: list[dict[str, Any]], texts: list[str], text: str, calls: list[ToolCall], reasoning_open: bool
) -> None:
    """Reasoning a parser sends in its own field is neither counted toward the hold nor released, but reported.

    It also ends the wait of a reply that may open in reasoning: the content that follows streams
    as usual instead of waiting for a close tag that never comes.

    Args:
        after (list[dict[str, Any]]): What follows the 146 characters of reasoning.
        texts (list[str]): The provisional texts expected.
        text (str): The round's visible text.
        calls (list[ToolCall]): The calls expected.
        reasoning_open (bool): Whether the reply may open inside a think block.
    """
    thinking = [chunk(reasoning_content=REASONING[start : start + 40]) for start in range(0, len(REASONING), 40)]

    with sse_endpoint([*thinking, *after, chunk(finish_reason="stop")]) as (base, _):
        llm = LocalOpenAI(api_base=base, api_key="test", model="invented-model", max_retries=0)
        released, done = _round(llm, reasoning_open=reasoning_open)

    assert released == texts
    assert (done.text, done.calls, done.reasoning) == (text, calls, REASONING.strip())


@pytest.mark.parametrize(
    ("after", "texts", "text", "calls"),
    [
        (text_chunks(ANSWER, 30), ANSWER_EVENTS, ANSWER, []),
        ([*text_chunks(PREFACE, 20), chunk(tool_calls=[_whole(SEARCH_CALL, 0)])], [], PREFACE, [SEARCH_CALL]),
    ],
    ids=["then-answer", "then-calls"],
)
def test_a_reply_opened_in_reasoning_is_held_until_its_close_tag(
    after: list[dict[str, Any]], texts: list[str], text: str, calls: list[ToolCall]
) -> None:
    """The template opened the think block, so only a close tag ends it: the 146 characters before it are reasoning.

    Args:
        after (list[dict[str, Any]]): What follows the close tag.
        texts (list[str]): The provisional texts expected.
        text (str): The round's visible text.
        calls (list[ToolCall]): The calls expected.
    """
    thinking = text_chunks(f"{REASONING}</think>\n\n", 7)

    released, done = _scripted([*thinking, *after, chunk(finish_reason="stop")], reasoning_open=True)

    assert released == texts
    assert (done.text, done.calls, done.reasoning) == (text, calls, REASONING.strip())


def test_a_reply_opened_in_reasoning_that_never_closes_is_the_answer() -> None:
    """With no close tag by the end, the held text was the answer: none of it went out, so the caller sends it all."""
    texts, done = _scripted([*text_chunks(ANSWER, 30), chunk(finish_reason="stop")], reasoning_open=True)

    assert texts == []
    assert done == RoundDone(text=ANSWER, calls=[], finish_reason="stop", provisional_sent=False, reasoning=None)


def test_reasoning_from_think_tags_comes_before_a_separate_field() -> None:
    """Both kinds of reasoning are reported, the think text first, joined by a newline."""
    _, done = _scripted(
        [
            chunk(reasoning_content="Erst die Texte. "),
            chunk("<think>Dann die Bilder.</think>"),
            chunk(PREFACE, finish_reason="stop"),
        ]
    )

    assert (done.text, done.reasoning) == (PREFACE, "Dann die Bilder.\nErst die Texte.")


#: Replies whose reasoning is read as ``strip_reasoning`` reads it, and whether each may open in reasoning.
REASONING_REPLIES = {
    "paired": (f"<think>{REASONING}</think>\n\n{ANSWER}\n", False),
    "paired-short": ("<think>Erst das Datum.</think>Das Hafenfest ist am Sonntag.", False),
    "paired-attributes": ('<think mode="plan">Erst das Datum.</think >Das Hafenfest ist am Sonntag.', False),
    "paired-twice": (f"Kurz: <think>Datum?</think>{ANSWER[:150]}<THINK>Bilder?</Think> {ANSWER[150:]}", False),
    "unclosed": (f"{ANSWER}\n<think>{REASONING}", False),
    "unclosed-mid-tag": (f"{PREFACE}<think>{REASONING}</thi", False),
    "dangling": (f"{REASONING}</think>\n\n{ANSWER}", True),
    "dangling-short": ("Erst das Datum.\n</think>\n\nDas Hafenfest ist am Sonntag.", True),
    "dangling-then-paired": (f"Datum?</think>{PREFACE}<think>Bilder?</think> {ANSWER}", True),
    "self-opened": (f"<think>{REASONING}</think>{ANSWER}", True),
    "self-opened-then-paired": (f"<think>Datum?</think>{PREFACE}<think>Bilder?</think> {ANSWER}", True),
    "self-opened-unclosed": (f"<think>{REASONING}", True),
    "never-closed": (ANSWER, True),
}


def _cuts(text: str, seed: int) -> list[list[str]]:
    """Cut a reply into pieces in many ways: fixed sizes, one character at a time included, and seeded random cuts.

    Args:
        text (str): The reply.
        seed (int): Seeds the random cuts, so every run cuts alike.

    Returns:
        list[list[str]]: Each way of cutting, its pieces in order.
    """
    ways = [[text[start : start + size] for start in range(0, len(text), size)] for size in (1, 2, 3, 7, 4096)]
    rng = random.Random(seed)
    for _ in range(24):
        cuts = sorted(rng.sample(range(1, len(text)), rng.randint(1, min(12, len(text) - 1))))
        ways.append([text[start:end] for start, end in zip([0, *cuts], [*cuts, len(text)], strict=True)])
    return ways


@pytest.mark.parametrize("name", list(REASONING_REPLIES))
def test_the_round_reads_reasoning_as_strip_reasoning_does(name: str) -> None:
    """Cut anywhere, a reply's visible text and reasoning are what ``strip_reasoning`` makes of the whole reply.

    Whatever went out provisionally is the whole visible text; when nothing did, the caller sends it.

    Args:
        name (str): Which reply.
    """
    raw, reasoning_open = REASONING_REPLIES[name]
    expected = strip_reasoning(raw)

    for pieces in _cuts(raw, seed=len(raw)):
        texts, done = _scripted(
            [*(chunk(piece) for piece in pieces), chunk(finish_reason="stop")], reasoning_open=reasoning_open
        )

        assert (done.text, done.reasoning) == expected, pieces
        assert "".join(texts) == (done.text if done.provisional_sent else ""), pieces
