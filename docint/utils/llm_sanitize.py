"""Sanitize LLM responses that contain reasoning-trace artifacts.

Reasoning-capable models (Qwen QwQ, DeepSeek R1 distills, gpt-oss with
Harmony formatting, etc.) emit their internal scratchpad alongside the
final answer. When the scratchpad survives into downstream storage or
prompts it pollutes retrieval, citations, and display. This module
strips those artifacts in one place so every call site shares the same
behaviour.

The sanitizer is tolerant of three leakage shapes observed in production:

1. Paired ``<think>...</think>`` blocks (zero or more).
2. Dangling ``</think>`` close tags with no matching open (common when
   the decoder drops or truncates the opening tag).
3. Unclosed ``<think>`` open tags at the tail (common when the model's
   output hit ``max_tokens`` mid-reasoning).

It also understands gpt-oss Harmony channels
(``<|channel|>analysis<|message|>...<|end|>``). When a paired ``final``
channel is present its body replaces the rest; otherwise ``analysis``
channels are removed in place.
"""

from __future__ import annotations

import re
from collections.abc import Sequence


_THINK_PAIRED_RE: re.Pattern[str] = re.compile(
    r"<think\b[^>]*>(.*?)</think\s*>",
    re.IGNORECASE | re.DOTALL,
)
_OPEN_THINK_RE: re.Pattern[str] = re.compile(
    r"<think\b[^>]*>",
    re.IGNORECASE,
)
_CLOSE_THINK_RE: re.Pattern[str] = re.compile(
    r"</think\s*>",
    re.IGNORECASE,
)
_HARMONY_ANALYSIS_RE: re.Pattern[str] = re.compile(
    r"<\|channel\|>\s*analysis\s*<\|message\|>(.*?)"
    r"(?=<\|end\|>|<\|return\|>|<\|channel\|>|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_HARMONY_FINAL_RE: re.Pattern[str] = re.compile(
    r"<\|channel\|>\s*final\s*<\|message\|>(.*?)"
    r"(?=<\|end\|>|<\|return\|>|<\|channel\|>|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_HARMONY_STOP_TOKEN_RE: re.Pattern[str] = re.compile(
    r"<\|(?:end|return|start|constrain)\|>",
    re.IGNORECASE,
)

# Phrases that are self-evidently refusals regardless of surrounding context.
_NO_IMAGE_PHRASES_UNCONDITIONAL: tuple[str, ...] = (
    "i don't see any image",
    "i do not see any image",
    "i don't see an image",
    "i do not see an image",
    "no image attached",
    "no image provided",
    "no image was attached",
    "no image was provided",
    "i cannot see an image",
    "i can't see an image",
    "there is no image",
    "there's no image",
    "you haven't provided an image",
    "you have not provided an image",
    "image is not attached",
    "image wasn't attached",
)

# Phrases that are only refusals when the response is very short. OCR'd
# documents can legitimately contain instructional text like "Please provide
# the image number below" — matching those as refusals would silently
# discard valid OCR output.
_NO_IMAGE_PHRASES_LENGTH_GUARDED: tuple[str, ...] = (
    "please provide the document image",
    "please provide the image",
    "please share the image",
)

_NO_IMAGE_LENGTH_GUARD_CHARS: int = 160


def strip_reasoning(
    text: str,
    *,
    extra_patterns: Sequence[re.Pattern[str]] | None = None,
) -> tuple[str, str | None]:
    """Remove reasoning scratchpads from an LLM response.

    Handles paired ``<think>...</think>`` blocks, dangling close tags,
    unclosed open tags, and gpt-oss Harmony channels. The caller can
    supply additional regex patterns to strip via ``extra_patterns``.

    Args:
        text (str): Raw LLM response text.
        extra_patterns (Sequence[re.Pattern[str]] | None): Optional
            extra regex patterns whose matches are treated as reasoning
            and removed from the clean output. Each match's ``group(0)``
            is captured as reasoning.

    Returns:
        tuple[str, str | None]: The sanitized text and a concatenated
        capture of any reasoning that was removed (``None`` when no
        artifact was found).
    """
    if not text:
        return "", None

    reasoning_parts: list[str] = []

    clean = _reduce_harmony_channels(text, reasoning_parts)

    def _paired_sub(match: re.Match[str]) -> str:
        body = match.group(1).strip()
        if body:
            reasoning_parts.append(body)
        return ""

    clean = _THINK_PAIRED_RE.sub(_paired_sub, clean)
    clean = _strip_orphaned_think_tags(clean, reasoning_parts)

    if extra_patterns:
        for pat in extra_patterns:

            def _extra_sub(match: re.Match[str]) -> str:
                body = match.group(0).strip()
                if body:
                    reasoning_parts.append(body)
                return ""

            clean = pat.sub(_extra_sub, clean)

    clean = _HARMONY_STOP_TOKEN_RE.sub("", clean).strip()

    reasoning = "\n".join(p for p in reasoning_parts if p) or None
    return clean, reasoning


def _reduce_harmony_channels(text: str, reasoning_parts: list[str]) -> str:
    """Collapse Harmony channel markup into clean text + captured reasoning.

    When a ``<|channel|>final<|message|>...<|end|>`` block is present,
    its body becomes the clean text. Otherwise all
    ``<|channel|>analysis<|message|>...`` blocks are captured into
    *reasoning_parts* and removed from the clean text.

    Args:
        text (str): Input text.
        reasoning_parts (list[str]): Accumulator appended in place.

    Returns:
        str: The text with Harmony analysis removed (or replaced by
        the final-channel body).
    """
    final_match = _HARMONY_FINAL_RE.search(text)
    if final_match:
        for m in _HARMONY_ANALYSIS_RE.finditer(text):
            body = m.group(1).strip()
            if body:
                reasoning_parts.append(body)
        return final_match.group(1).strip()

    def _analysis_sub(match: re.Match[str]) -> str:
        body = match.group(1).strip()
        if body:
            reasoning_parts.append(body)
        return ""

    return _HARMONY_ANALYSIS_RE.sub(_analysis_sub, text)


def _strip_orphaned_think_tags(text: str, reasoning_parts: list[str]) -> str:
    """Strip orphaned ``<think>`` / ``</think>`` tags from *text*.

    After paired stripping, any remaining tag is unpaired. A dangling
    close tag means the text preceding it is reasoning. An unclosed
    open tag means the text following it is reasoning (the generator
    likely ran out of tokens).

    Args:
        text (str): Input with paired blocks already removed.
        reasoning_parts (list[str]): Accumulator appended in place.

    Returns:
        str: The text with orphan reasoning spans removed.
    """
    close_match = _CLOSE_THINK_RE.search(text)
    open_match = _OPEN_THINK_RE.search(text)

    if close_match and (open_match is None or close_match.start() < open_match.start()):
        before = text[: close_match.start()].strip()
        if before:
            reasoning_parts.append(before)
        text = text[close_match.end() :]
        open_match = _OPEN_THINK_RE.search(text)

    if open_match:
        tail = text[open_match.end() :].strip()
        if tail:
            reasoning_parts.append(tail)
        text = text[: open_match.start()]

    return text


def looks_like_no_image_refusal(text: str) -> bool:
    """Return whether *text* indicates the model did not receive an image.

    Vision calls occasionally return a refusal-like response claiming
    no image was attached, even though the request did include one.
    Callers should treat a ``True`` return as an empty response and log
    a warning rather than storing the text.

    Short "please provide the image" style phrases are only treated as
    refusals when the response is short, to avoid false positives when
    an OCR'd document page legitimately contains that wording.

    Args:
        text (str): Model response text.

    Returns:
        bool: ``True`` when the text matches a known no-image phrase.
    """
    if not text:
        return False
    normalized = " ".join(text.strip().lower().split())
    if any(phrase in normalized for phrase in _NO_IMAGE_PHRASES_UNCONDITIONAL):
        return True
    if len(normalized) <= _NO_IMAGE_LENGTH_GUARD_CHARS:
        return any(phrase in normalized for phrase in _NO_IMAGE_PHRASES_LENGTH_GUARDED)
    return False
