"""Tests for :mod:`docint.utils.llm_sanitize`."""

from __future__ import annotations

import re

import pytest

from docint.utils.llm_sanitize import (
    ReasoningStripper,
    looks_like_no_image_refusal,
    squeeze_char_runs,
    strip_reasoning,
)


def test_strip_reasoning_removes_single_think_block() -> None:
    """A single paired block should be captured and removed."""
    text = "<think>planning the answer</think>answer body"

    clean, reasoning = strip_reasoning(text)

    assert clean == "answer body"
    assert reasoning == "planning the answer"


def test_strip_reasoning_handles_multiple_think_blocks() -> None:
    """Multiple paired blocks should concatenate into one reasoning string."""
    text = "<think>step 1</think>part A<think>step 2</think>part B"

    clean, reasoning = strip_reasoning(text)

    assert clean == "part Apart B"
    assert reasoning == "step 1\nstep 2"


def test_strip_reasoning_case_insensitive_and_multiline() -> None:
    """Matching should ignore case and span newlines."""
    text = "<THINK>\nline1\nline2\n</Think>\nfinal text"

    clean, reasoning = strip_reasoning(text)

    assert clean == "final text"
    assert reasoning is not None
    assert "line1" in reasoning and "line2" in reasoning


def test_strip_reasoning_dangling_close_tag() -> None:
    """A leading reasoning blob ending with ``</think>`` should be stripped."""
    text = (
        "The user wants me to do OCR. I don't actually see an image. "
        "</think>\n\nI don't see any image attached to your message."
    )

    clean, reasoning = strip_reasoning(text)

    assert "</think>" not in clean
    assert clean.startswith("I don't see any image")
    assert reasoning is not None
    assert "user wants me to do OCR" in reasoning


def test_strip_reasoning_unclosed_open_tag() -> None:
    """An unclosed ``<think>`` at the tail should capture the remainder."""
    text = "visible prefix<think>unfinished reasoning with no close"

    clean, reasoning = strip_reasoning(text)

    assert clean == "visible prefix"
    assert reasoning == "unfinished reasoning with no close"


def test_strip_reasoning_harmony_analysis_and_final_channels() -> None:
    """When a Harmony final channel is present its body becomes the clean text."""
    text = "<|channel|>analysis<|message|>hidden reasoning<|end|><|channel|>final<|message|>visible answer<|end|>"

    clean, reasoning = strip_reasoning(text)

    assert clean == "visible answer"
    assert reasoning == "hidden reasoning"


def test_strip_reasoning_harmony_analysis_only() -> None:
    """Without a Harmony final channel, analysis is stripped from the remainder."""
    text = "prologue <|channel|>analysis<|message|>hidden<|end|> epilogue"

    clean, reasoning = strip_reasoning(text)

    assert clean == "prologue  epilogue".strip() or clean == "prologue   epilogue".strip()
    assert "hidden" not in clean
    assert reasoning == "hidden"


def test_strip_reasoning_preserves_plain_text() -> None:
    """Input without artifacts should be returned unchanged."""
    text = "A perfectly ordinary description with no reasoning tags."

    clean, reasoning = strip_reasoning(text)

    assert clean == text
    assert reasoning is None


def test_strip_reasoning_pure_reasoning_returns_empty_clean() -> None:
    """A response that is entirely a reasoning block should yield empty clean text."""
    text = "<think>only reasoning, no answer</think>"

    clean, reasoning = strip_reasoning(text)

    assert clean == ""
    assert reasoning == "only reasoning, no answer"


def test_strip_reasoning_empty_input_returns_none_reasoning() -> None:
    """An empty string should return empty clean text and ``None`` reasoning."""
    clean, reasoning = strip_reasoning("")

    assert clean == ""
    assert reasoning is None


def test_strip_reasoning_accepts_extra_patterns() -> None:
    """Caller-supplied patterns should be stripped and captured."""
    text = "[[meta: ignore me]] real output"
    pattern = re.compile(r"\[\[meta:.*?\]\]", re.DOTALL)

    clean, reasoning = strip_reasoning(text, extra_patterns=[pattern])

    assert clean == "real output"
    assert reasoning == "[[meta: ignore me]]"


def test_strip_reasoning_removes_stray_harmony_tokens() -> None:
    """Stray Harmony control tokens should never survive into clean text."""
    text = "answer body<|end|>"

    clean, _ = strip_reasoning(text)

    assert clean == "answer body"


def test_looks_like_no_image_refusal_positive_phrases() -> None:
    """Canonical no-image phrases should be detected."""
    samples = [
        "I don't see any image attached to your message.",
        "No image was attached to the request.",
        "Please provide the document image you'd like me to OCR.",
        "There is no image in this conversation so far.",
        "I cannot see an image in your message.",
    ]

    for text in samples:
        assert looks_like_no_image_refusal(text), text


def test_looks_like_no_image_refusal_ignores_normal_descriptions() -> None:
    """Normal captions must not be flagged as refusals."""
    samples = [
        "A technical architecture diagram showing three services.",
        "An OCR-rendered page of Arabic text.",
        "",
        "Black-and-white photograph of a cat on a windowsill.",
    ]

    for text in samples:
        assert not looks_like_no_image_refusal(text), text


def test_looks_like_no_image_refusal_rejects_long_instructional_text() -> None:
    """OCR output containing 'please provide the image' must not be a refusal.

    A document page that happens to contain the phrase as instructional
    text should flow through normally when the response is long enough
    to be real OCR content.
    """
    long_text = (
        "Submission checklist. Please provide the image resolution settings "
        "below along with the requested metadata. The review committee will "
        "not accept submissions missing any of the listed attachments. "
        "Additional notes: see appendix A for sizing constraints."
    )

    assert not looks_like_no_image_refusal(long_text)


def test_looks_like_no_image_refusal_catches_short_please_provide_phrase() -> None:
    """A short refusal like 'Please provide the image to OCR.' stays caught."""
    short_text = "Please provide the image you would like me to OCR."

    assert looks_like_no_image_refusal(short_text)


def test_squeeze_char_runs_collapses_a_degenerate_dot_run() -> None:
    """A repetition loop's output shrinks to a leader-line-sized run.

    Observed live: a vision model transcribing a certificate's dotted
    fill-in line emitted one unbroken run of 65392 dots (a 16-dot token
    repeated until ``max_tokens``), which then became three of a
    document's five chunks.
    """
    text = "URKUNDE\n\nLiebe/r:" + "." * 65392

    squeezed = squeeze_char_runs(text)

    assert squeezed == "URKUNDE\n\nLiebe/r:" + "." * 20


def test_squeeze_char_runs_leaves_ordinary_text_alone() -> None:
    """Real prose, ellipses, and short separators pass through unchanged."""
    text = "Wait... really?\n---\nSection 2.1 covers the gate."

    assert squeeze_char_runs(text) == text


def test_squeeze_char_runs_handles_multiple_runs() -> None:
    """Every oversized run shrinks, not just the first."""
    text = "Name:" + "." * 50 + "\nDatum:" + "_" * 50

    squeezed = squeeze_char_runs(text)

    assert squeezed == "Name:" + "." * 20 + "\nDatum:" + "_" * 20


def test_squeeze_char_runs_keeps_runs_at_the_threshold() -> None:
    """A run exactly at the cap is untouched."""
    text = "-" * 20

    assert squeeze_char_runs(text) == text


def test_squeeze_char_runs_preserves_newline_runs() -> None:
    """Blank-line stretches are layout, not degeneration; leave them be."""
    text = "para one" + "\n" * 30 + "para two"

    assert squeeze_char_runs(text) == text


def _streamed(stripper: ReasoningStripper, text: str, size: int) -> str:
    """Feed ``text`` to ``stripper`` in pieces of ``size`` characters, then flush it.

    Args:
        stripper (ReasoningStripper): The stripper.
        text (str): The reply.
        size (int): Characters per piece.

    Returns:
        str: Everything the stripper returned, in order.
    """
    pieces = [text[start : start + size] for start in range(0, len(text), size)]
    return "".join(stripper.feed(piece) for piece in pieces) + stripper.flush()


@pytest.mark.parametrize(
    ("text", "reasoning_open"),
    [
        ("<think>planning the answer</think>answer body", False),
        ("<THINK>\nline1\nline2\n</Think>\n\nfinal text\n", False),
        ('<think mode="plan">step 1</think >part A <think>step 2</think>part B', False),
        ("visible prefix<think>unfinished reasoning</thi", False),
        ("leading reasoning\n</think>\n\nthe answer", True),
        ("<think>step 1</think>the answer", True),
        ("an answer with no reasoning at all", True),
    ],
    ids=["paired", "paired-whitespace", "attributes", "unclosed", "dangling-close", "self-opened", "never-closed"],
)
@pytest.mark.parametrize("size", [1, 3, 7, 4096])
def test_reasoning_stripper_matches_strip_reasoning(text: str, reasoning_open: bool, size: int) -> None:
    """Streamed in any piece size, the visible text and the reasoning are what ``strip_reasoning`` returns.

    Args:
        text (str): The reply.
        reasoning_open (bool): Whether the reply may open inside a think block.
        size (int): Characters per piece.
    """
    stripper = ReasoningStripper(reasoning_open=reasoning_open)

    assert (_streamed(stripper, text, size), stripper.reasoning) == strip_reasoning(text)


def test_reasoning_stripper_holds_back_only_what_may_become_a_tag() -> None:
    """Text goes out as it arrives, except a tail that could still grow into a tag and trailing whitespace."""
    stripper = ReasoningStripper()

    assert stripper.feed("\nanswer <thi") == "answer"
    assert stripper.feed("s and that") == " <this and that"
    assert stripper.feed(" <think") == ""
    assert stripper.feed(">hidden</th") == ""
    assert stripper.feed("ink>shown ") == " shown"
    assert stripper.flush() == ""
    assert stripper.reasoning == "hidden"


def test_reasoning_stripper_opened_in_reasoning_holds_everything_until_the_close_tag() -> None:
    """A reply that may open in reasoning shows nothing before its first close tag; what came before is reasoning."""
    stripper = ReasoningStripper(reasoning_open=True)

    assert stripper.feed("the user asks for a date") == ""
    assert stripper.feed(", so I search first.</thi") == ""
    assert stripper.feed("nk>\n\nSunday.") == "Sunday."
    assert stripper.flush() == ""
    assert stripper.reasoning == "the user asks for a date, so I search first."


def test_reasoning_stripper_cancel_open_reasoning_reads_held_text_as_text() -> None:
    """When the endpoint sends reasoning apart, held content is the reply's own text, its tags read as usual."""
    stripper = ReasoningStripper(reasoning_open=True)

    assert stripper.feed("\nSunday, ") == ""
    assert stripper.cancel_open_reasoning() == "Sunday,"
    assert stripper.feed("at ten.<think>check</think>") == " at ten."
    assert stripper.cancel_open_reasoning() == ""
    assert (stripper.flush(), stripper.reasoning) == ("", "check")


def test_reasoning_stripper_passes_a_lone_close_tag_through_unless_opened_in_reasoning() -> None:
    """Without ``reasoning_open``, a close tag with no open tag before it is text: what preceded it has gone out."""
    stripper = ReasoningStripper()

    assert stripper.feed("thought</think>answer") == "thought</think>answer"
    assert (stripper.flush(), stripper.reasoning) == ("", None)


def test_reasoning_stripper_opened_in_reasoning_reads_all_held_text_as_reasoning() -> None:
    """Before the first close tag, everything is reasoning, a model's own open tag left out of it.

    ``strip_reasoning`` would show the text ahead of that open tag; a reply that may open in
    reasoning began inside the block, so the stream counts it as reasoning instead.
    """
    stripper = ReasoningStripper(reasoning_open=True)

    assert stripper.feed("first, <think>then</think>answer") == "answer"
    assert (stripper.flush(), stripper.reasoning) == ("", "first, then")
