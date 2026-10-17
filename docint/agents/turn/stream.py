"""Stream one model round: the answer text as it may be shown, and the round's tool calls.

The calls are read from the raw chunks the client passes through on ``ChatResponse.raw``,
not from llama-index's streaming merge. That merge reads only the first tool-call entry
of each chunk and joins fragments by ``index`` alone, so a chunk carrying two whole calls
(Ollama) loses one, and a call sent without ``index`` merges into the call before it.

A round's text stays provisional while it may still turn out to be a preface to tool
calls: it is held until ``HOLD_CHARS`` visible characters have arrived, and text held
when a tool call starts is never released. Reasoning never counts and is never released:
text inside ``<think>…</think>`` is left out here, and a reasoning field the endpoint sends
apart from the content never reaches ``ChatResponse.delta``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

# The sentinel splitter's think-tag reading, so both agree on what is reasoning.
from docint.agents.assessment.split import _ASCII_LOWER, _THINK_CLOSE, _THINK_OPEN, _partial_suffix

if TYPE_CHECKING:
    from llama_index.core.base.llms.types import ChatMessage
    from llama_index.core.llms import LLM

#: Visible characters a round must have written, with no tool call begun, before its text is released.
HOLD_CHARS = 120


@dataclass(frozen=True)
class ToolCall:
    """One tool call a model round asked for.

    Attributes:
        id (str): The id its reply names; unique within the round.
        name (str): The tool's name as the model wrote it.
        arguments (str): The arguments as raw JSON text, unparsed and unchecked.
    """

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class Provisional:
    """Visible answer text released while the round still streams.

    Attributes:
        text (str): Text not released before, never reasoning. It stays provisional: if the
            round still ends in tool calls, the caller withdraws it.
    """

    text: str


@dataclass(frozen=True)
class RoundDone:
    """How a round ended; always its last event.

    Attributes:
        text (str): The round's visible text in full, released or not; reasoning left out.
        calls (list[ToolCall]): The tool calls, in the order the model began them; empty when
            the text is the answer.
        finish_reason (str | None): The endpoint's finish reason, or ``None`` if it sent none.
        provisional_sent (bool): Whether any of ``text`` went out as :class:`Provisional`. When
            set and the round has no calls, all of ``text`` went out.
    """

    text: str
    calls: list[ToolCall]
    finish_reason: str | None
    provisional_sent: bool


RoundEvent = Provisional | RoundDone
"""What :func:`stream_round` yields."""


def _field(value: Any, name: str) -> Any:
    """Read one field of a raw chunk part, whether the client parsed it or not.

    Args:
        value (Any): An SDK object, a mapping, or ``None``.
        name (str): The field.

    Returns:
        Any: Its value, or ``None`` when absent.
    """
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _text(value: Any) -> str:
    """Return a string field's value, or ``""`` for a missing or malformed one.

    Args:
        value (Any): The field's value.

    Returns:
        str: The value when it is a string.
    """
    return value if isinstance(value, str) else ""


def _choice(raw_chunk: Any) -> Any:
    """Return a raw chunk's first choice.

    Args:
        raw_chunk (Any): The chunk as the client received it.

    Returns:
        Any: The choice, or ``None`` for a chunk without choices, such as a usage chunk.
    """
    choices = _field(raw_chunk, "choices")
    return choices[0] if isinstance(choices, Sequence) and choices else None


@dataclass
class _PartialCall:
    """A tool call still being joined from its fragments."""

    id: str = ""
    name: str = ""
    arguments: str = ""


class ToolCallAccumulator:
    """Join the tool-call fragments in one round's raw chunks into whole calls.

    Every entry of a chunk's ``delta.tool_calls`` is read, so several whole calls in one
    chunk all survive. A fragment joins the call open at its ``index``, unless it carries
    an id other than that call's, which starts a new call there. A fragment without an
    ``index`` joins the call holding its id, or starts one if the id is new; with neither,
    it continues the call fed last.

    Feed each chunk as it arrives: the client grows an earlier chunk's call objects in
    place as later fragments come in, so a chunk read late would count them twice.
    """

    def __init__(self) -> None:
        """Start with no calls."""
        self._calls: list[_PartialCall] = []
        self._by_index: dict[int, _PartialCall] = {}
        self._by_id: dict[str, _PartialCall] = {}
        self._last: _PartialCall | None = None

    def feed(self, raw_chunk: Any) -> bool:
        """Read the tool-call fragments of one raw chunk.

        Args:
            raw_chunk (Any): The chunk on ``ChatResponse.raw``: the SDK's object, or the same
                shape as a mapping; anything else carries no call.

        Returns:
            bool: Whether the chunk carried any tool-call fragment.
        """
        entries = _field(_field(_choice(raw_chunk), "delta"), "tool_calls")
        if not isinstance(entries, Sequence) or not entries:
            return False
        for entry in entries:
            self._join(entry)
        return True

    def _join(self, entry: Any) -> None:
        """Add one fragment to the call it belongs to.

        Args:
            entry (Any): One entry of a chunk's ``delta.tool_calls``.
        """
        index = _field(entry, "index")
        call_id = _text(_field(entry, "id"))
        call = self._target(index if isinstance(index, int) and not isinstance(index, bool) else None, call_id)
        if call_id and not call.id:
            call.id = call_id
            self._by_id.setdefault(call_id, call)
        function = _field(entry, "function")
        call.name += _text(_field(function, "name"))
        call.arguments += _text(_field(function, "arguments"))
        self._last = call

    def _target(self, index: int | None, call_id: str) -> _PartialCall:
        """Find the call a fragment continues, or start the one it begins.

        Args:
            index (int | None): The fragment's index, if it has one.
            call_id (str): Its id; ``""`` when it has none.

        Returns:
            _PartialCall: The call to extend.
        """
        if index is not None:
            call = self._by_index.get(index)
            if call is None or (call_id and call.id and call_id != call.id):
                call = self._start()
                self._by_index[index] = call
            return call
        if call_id:
            return self._by_id.get(call_id) or self._start()
        return self._last or self._start()

    def _start(self) -> _PartialCall:
        """Begin a new call.

        Returns:
            _PartialCall: The empty call, appended in arrival order.
        """
        call = _PartialCall()
        self._calls.append(call)
        return call

    @property
    def dropped(self) -> int:
        """Count the calls :meth:`calls` leaves out because they never received a name.

        Returns:
            int: The count.
        """
        return sum(1 for call in self._calls if not call.name.strip())

    def calls(self) -> list[ToolCall]:
        """Return the calls joined so far, those without a name left out.

        A call without an id, or with one an earlier call already holds, is given
        ``call_<position>``, suffixed until no other call holds it, so every call can be
        answered by its own id.

        Returns:
            list[ToolCall]: The calls, in the order they began.
        """
        named = [call for call in self._calls if call.name.strip()]
        taken = {call.id for call in named if call.id}
        given: set[str] = set()
        calls: list[ToolCall] = []
        for position, call in enumerate(named):
            call_id = call.id
            if not call_id or call_id in given:
                call_id = base = f"call_{position}"
                suffix = 0
                while call_id in taken or call_id in given:
                    suffix += 1
                    call_id = f"{base}_{suffix}"
            given.add(call_id)
            calls.append(ToolCall(id=call_id, name=call.name, arguments=call.arguments))
        return calls


class _VisibleText:
    """Pass a round's text through a token at a time, leaving out what it writes inside think blocks.

    Reads tags as :class:`~docint.agents.assessment.split.SentinelStreamSplitter` does, and
    holds back only a tail that could still grow into one.
    """

    def __init__(self) -> None:
        """Start outside any think block, with nothing held."""
        self._buffer = ""
        self._in_think = False

    def feed(self, token: str) -> str:
        """Accept one token and return the visible text it completes.

        Args:
            token (str): The next piece of the round's content.

        Returns:
            str: Visible text safe to count; empty while a tag may be forming.
        """
        self._buffer += token
        out: list[str] = []
        while self._buffer:
            tag = _THINK_CLOSE if self._in_think else _THINK_OPEN
            lower = self._buffer.translate(_ASCII_LOWER)
            found = lower.find(tag)
            if found < 0:
                kept = len(self._buffer) - _partial_suffix(lower, tag)
                if not self._in_think:
                    out.append(self._buffer[:kept])
                self._buffer = self._buffer[kept:]
                break
            if not self._in_think:
                out.append(self._buffer[:found])
            self._buffer = self._buffer[found + len(tag) :]
            self._in_think = not self._in_think
        return "".join(out)

    def flush(self) -> str:
        """Release a held tail once the round has ended.

        Returns:
            str: The tail, which never became a tag; ``""`` inside a think block the round
            never closed, whose text is reasoning.
        """
        rest, self._buffer = self._buffer, ""
        return "" if self._in_think else rest


def _finish_reason(raw_chunk: Any) -> str | None:
    """Read the finish reason a raw chunk carries.

    Args:
        raw_chunk (Any): The chunk as the client received it.

    Returns:
        str | None: The reason, or ``None`` when the chunk has none.
    """
    return _text(_field(_choice(raw_chunk), "finish_reason")) or None


def stream_round(
    llm: LLM,
    messages: Sequence[ChatMessage],
    *,
    tools: Sequence[dict[str, Any]],
    tool_choice: str = "auto",
    **kwargs: Any,
) -> Iterator[RoundEvent]:
    """Stream one model round, releasing its text only once it is unlikely to be a preface.

    Visible text is held until ``HOLD_CHARS`` characters have arrived with no tool call
    begun; then the held text goes out as one :class:`Provisional` and every later piece as
    its own. Once a tool-call fragment arrives, nothing more is released. A round that ends
    with less than ``HOLD_CHARS`` characters releases nothing, and its text is in
    :class:`RoundDone` for the caller to send.

    Args:
        llm (LLM): The chat model; its ``stream_chat`` must pass the raw chunks on ``raw``.
        messages (Sequence[ChatMessage]): The conversation so far.
        tools (Sequence[dict[str, Any]]): The tool schemas offered, as the endpoint receives them.
        tool_choice (str): ``auto`` lets the model choose; ``none`` asks for an answer.
        **kwargs (Any): Further options for ``stream_chat``.

    Yields:
        RoundEvent: :class:`Provisional` texts, then exactly one :class:`RoundDone`.

    Raises:
        Exception: Whatever the model call raises; an endpoint's HTTP error surfaces on the
            first pull, before anything is yielded.
    """
    accumulator = ToolCallAccumulator()
    reader = _VisibleText()
    text = ""
    released = 0
    calling = False
    finish_reason: str | None = None
    for response in llm.stream_chat(messages, tools=list(tools), tool_choice=tool_choice, **kwargs):
        calling = accumulator.feed(response.raw) or calling
        finish_reason = _finish_reason(response.raw) or finish_reason
        text += reader.feed(response.delta or "")
        if not calling and len(text) > released and (released or len(text) >= HOLD_CHARS):
            yield Provisional(text[released:])
            released = len(text)
    text += reader.flush()
    if released and not calling and len(text) > released:
        yield Provisional(text[released:])
    if accumulator.dropped:
        logger.warning("A model round sent tool calls without a name; dropped: {}", accumulator.dropped)
    yield RoundDone(text=text, calls=accumulator.calls(), finish_reason=finish_reason, provisional_sent=released > 0)
