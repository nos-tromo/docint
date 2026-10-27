"""Tests for the topic labeller."""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import httpx
import openai
import pytest
from loguru import logger

from docint.core.topics.labels import (
    DESCRIPTION_CHARS,
    LABEL_CHARS,
    PASSAGE_CHARS,
    LabelRequest,
    TopicLabeller,
    fallback_label,
    label_response_format,
    parse_label,
    render_prompt,
)

TEMPLATE = "K: {keywords}\n{passages}"


def _reply(label: str = "Harbour traffic", description: str = "Ships and ports.") -> str:
    """Build a well-formed model reply.

    Args:
        label (str): The label value.
        description (str): The description value.

    Returns:
        str: The JSON reply text.
    """
    return json.dumps({"description": description, "label": label})


def _request(topic_id: int = 0, keywords: tuple[str, ...] = ("ship", "port")) -> LabelRequest:
    """Build a request.

    Args:
        topic_id (int): The topic id.
        keywords (tuple[str, ...]): The keywords.

    Returns:
        LabelRequest: The request.
    """
    return LabelRequest(topic_id=topic_id, keywords=keywords, passages=("first passage", "second passage"))


def _bad_request() -> openai.BadRequestError:
    """Build a schema rejection.

    Returns:
        openai.BadRequestError: An HTTP 400 error.
    """
    response = httpx.Response(400, request=httpx.Request("POST", "http://t"))
    return openai.BadRequestError("x", response=response, body=None)


def test_render_prompt_fills_keywords_and_numbered_passages_and_leaves_braces_alone() -> None:
    """Keywords are joined, passages numbered and cut, and other braces survive."""
    request = LabelRequest(0, ("a", "b"), ("x" * (PASSAGE_CHARS + 50), "{keywords} {x}"))

    text = render_prompt("{keywords}|{passages}|{other}", request)

    assert PASSAGE_CHARS == 800  # enough of each passage to say what it is about
    assert text == "a, b|1. " + "x" * PASSAGE_CHARS + "\n2. {keywords} {x}|{other}"
    request = LabelRequest(0, ("{passages}",), ("{keywords}",))
    assert render_prompt("{keywords}|{passages}", request) == "{passages}|1. {keywords}"


def test_parse_label_reads_json_strips_reasoning_and_cuts_long_values() -> None:
    """Reasoning and prose around the object are ignored and long values are cut."""
    raw = "<think>hmm {x}</think>Sure: " + _reply("  Spaced \n out " + "l" * 200, "d" * 500) + " done"

    parsed = parse_label(raw)

    assert parsed is not None
    label, description = parsed
    assert label.startswith("Spaced out ")
    assert len(label) == LABEL_CHARS
    assert len(description) == DESCRIPTION_CHARS


@pytest.mark.parametrize(
    "raw",
    ["", "no json", '{"label": "x"}', '{"label": " ", "description": "d"}', '{"label": 3, "description": "d"}', "[1]"],
)
def test_parse_label_returns_none_for_a_blank_or_missing_value(raw: str) -> None:
    """A reply without two non-blank strings is unusable.

    Args:
        raw (str): The reply text.
    """
    assert parse_label(raw) is None


def test_the_response_format_is_strict_with_both_keys_required() -> None:
    """The schema is strict, closed, and requires both keys."""
    fmt = label_response_format()

    assert fmt["type"] == "json_schema"
    schema = fmt["json_schema"]
    assert schema["strict"] is True
    assert schema["schema"]["additionalProperties"] is False
    assert sorted(schema["schema"]["required"]) == ["description", "label"]


def test_label_all_writes_one_label_per_topic() -> None:
    """Every request gets a written label and progress ticks once per topic."""
    ticks: list[tuple[int, int]] = []
    labeller = TopicLabeller(lambda prompt, fmt: _reply(), template=TEMPLATE)

    out = labeller.label_all([_request(1), _request(2)], progress=lambda d, t: ticks.append((d, t)))

    assert set(out) == {1, 2}
    assert all(v.written and v.label == "Harbour traffic" for v in out.values())
    assert ticks == [(1, 2), (2, 2)]


def test_a_rejected_response_format_switches_off_structured_output_for_the_rest() -> None:
    """After one rejection no later call carries the schema."""
    formats: list[Any] = []
    lock = threading.Lock()

    def complete(prompt: str, fmt: dict[str, Any] | None) -> str:
        with lock:
            formats.append(fmt)
            first = len(formats) == 1
        if first:
            raise _bad_request()
        return _reply()

    labeller = TopicLabeller(complete, template=TEMPLATE, max_in_flight=1)

    out = labeller.label_all([_request(i) for i in range(3)])

    assert all(v.written for v in out.values())
    assert formats[0] is not None
    assert all(f is None for f in formats[1:])
    assert len(formats) == 4


def test_an_unparseable_constrained_reply_is_retried_once_unconstrained() -> None:
    """A bad constrained reply gets one plain retry."""
    formats: list[Any] = []

    def complete(prompt: str, fmt: dict[str, Any] | None) -> str:
        formats.append(fmt)
        return "garbage" if fmt is not None else _reply()

    out = TopicLabeller(complete, template=TEMPLATE).label_all([_request()])

    assert out[0].written
    assert [f is None for f in formats] == [False, True]


def test_a_failing_call_falls_back_to_the_keywords() -> None:
    """A raising model or an unparseable final reply yields the keyword label."""

    def boom(prompt: str, fmt: dict[str, Any] | None) -> str:
        raise RuntimeError("down")

    out = TopicLabeller(boom, template=TEMPLATE).label_all([_request(0, ("a", "b", "c", "d", "e"))])
    junk = TopicLabeller(lambda p, f: "nope", template=TEMPLATE).label_all([_request(1, ("a",))])

    assert out[0].label == "a, b, c, d"
    assert out[0].description == "" and out[0].written is False
    assert junk[1].label == "a" and junk[1].written is False


def test_no_keywords_fall_back_to_the_topic_ref() -> None:
    """With no keywords the label is the topic reference."""
    assert fallback_label(7, ()) == "T7"
    out = TopicLabeller(lambda p, f: "nope", template=TEMPLATE).label_all([_request(7, ())])
    assert out[7].label == "T7"


def test_at_most_four_calls_are_in_flight() -> None:
    """The pool never runs more than four calls at once."""
    lock = threading.Lock()
    state = {"now": 0, "peak": 0}

    def complete(prompt: str, fmt: dict[str, Any] | None) -> str:
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        time.sleep(0.02)
        with lock:
            state["now"] -= 1
        return _reply()

    TopicLabeller(complete, template=TEMPLATE).label_all([_request(i) for i in range(12)])

    assert 1 < state["peak"] <= 4


def test_a_raising_progress_stops_the_remaining_calls() -> None:
    """A progress callback that raises cancels queued work and propagates."""
    calls: list[int] = []

    def complete(prompt: str, fmt: dict[str, Any] | None) -> str:
        calls.append(1)
        time.sleep(0.01)
        return _reply()

    def stop(done: int, total: int) -> None:
        raise KeyboardInterrupt("stop")

    labeller = TopicLabeller(complete, template=TEMPLATE, max_in_flight=1)

    with pytest.raises(KeyboardInterrupt):
        labeller.label_all([_request(i) for i in range(20)], progress=stop)

    assert len(calls) < 20


def test_labels_log_counts_only() -> None:
    """The summary line carries counts and no label, keyword or passage."""
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(str(m)), level="DEBUG")
    try:
        request = LabelRequest(0, ("secretword",), ("secretpassage",))
        TopicLabeller(lambda p, f: _reply("Secretlabel", "Secretdesc"), template=TEMPLATE).label_all([request])
    finally:
        logger.remove(sink)

    joined = "\n".join(messages)
    assert "Topic labels | written=1 fallback=0" in joined
    for leaked in ("secretword", "secretpassage", "Secretlabel", "Secretdesc"):
        assert leaked not in joined
