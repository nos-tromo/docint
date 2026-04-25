"""Tests for :mod:`docint.utils.llm_sanitize`."""

from __future__ import annotations

import re

from docint.utils.llm_sanitize import (
    looks_like_no_image_refusal,
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
    text = (
        "<|channel|>analysis<|message|>hidden reasoning<|end|>"
        "<|channel|>final<|message|>visible answer<|end|>"
    )

    clean, reasoning = strip_reasoning(text)

    assert clean == "visible answer"
    assert reasoning == "hidden reasoning"


def test_strip_reasoning_harmony_analysis_only() -> None:
    """Without a Harmony final channel, analysis is stripped from the remainder."""
    text = (
        "prologue <|channel|>analysis<|message|>hidden<|end|> epilogue"
    )

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
