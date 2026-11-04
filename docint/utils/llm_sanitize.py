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

Each function reads its text in time linear in the text's length. The text
has no length limit and can carry a user's own words, so a pattern that scans
to the end again from every tag it finds would make a reply full of unclosed
tags cost quadratic time.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from typing import Literal

# An open tag runs from ``<think`` to the first ``>`` after it: see :func:`_find_open_think`.
_OPEN_THINK_START_RE: re.Pattern[str] = re.compile(
    r"<think\b",
    re.IGNORECASE,
)
_CLOSE_THINK_RE: re.Pattern[str] = re.compile(
    r"</think\s*>",
    re.IGNORECASE,
)
# A Harmony channel's head. Its body runs to the first stop token or channel
# head after it, or to the end: see :func:`_harmony_blocks`.
_HARMONY_ANALYSIS_HEAD_RE: re.Pattern[str] = re.compile(
    r"<\|channel\|>\s*analysis\s*<\|message\|>",
    re.IGNORECASE,
)
_HARMONY_FINAL_HEAD_RE: re.Pattern[str] = re.compile(
    r"<\|channel\|>\s*final\s*<\|message\|>",
    re.IGNORECASE,
)
_HARMONY_BODY_END_RE: re.Pattern[str] = re.compile(
    r"<\|(?:end|return|channel)\|>",
    re.IGNORECASE,
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
    clean = _strip_paired_think_blocks(clean, reasoning_parts)
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
    final = next(_harmony_blocks(text, _HARMONY_FINAL_HEAD_RE), None)
    analysis = list(_harmony_blocks(text, _HARMONY_ANALYSIS_HEAD_RE))
    for _, body_start, body_end in analysis:
        body = text[body_start:body_end].strip()
        if body:
            reasoning_parts.append(body)
    if final is not None:
        return text[final[1] : final[2]].strip()

    kept: list[str] = []
    pos = 0
    for start, _, body_end in analysis:
        kept.append(text[pos:start])
        pos = body_end
    kept.append(text[pos:])
    return "".join(kept)


def _harmony_blocks(text: str, head: re.Pattern[str]) -> Iterator[tuple[int, int, int]]:
    """Yield the Harmony channel blocks that *head* opens, in order.

    A body runs to the first stop token or channel head after it, or to the
    end of the text. Blocks never overlap, so each search starts where the
    last body ended and the text is read once.

    Args:
        text (str): Input text.
        head (re.Pattern[str]): The channel head to look for.

    Yields:
        tuple[int, int, int]: Each block's start, its body's start and its body's end.
    """
    pos = 0
    while (opening := head.search(text, pos)) is not None:
        stop = _HARMONY_BODY_END_RE.search(text, opening.end())
        end = len(text) if stop is None else stop.start()
        yield opening.start(), opening.end(), end
        pos = end


def _find_open_think(text: str, pos: int = 0) -> tuple[int, int] | None:
    """Find the first ``<think ...>`` open tag at or after *pos*.

    The tag runs from ``<think`` to the first ``>`` after it. When no ``>``
    follows the first ``<think``, none follows a later one either, so one
    look settles it; a regex search would scan to the end again from every
    later ``<think``.

    Args:
        text (str): The text to search.
        pos (int): Where to start.

    Returns:
        tuple[int, int] | None: The tag's start and end, or ``None`` when there is none.
    """
    start = _OPEN_THINK_START_RE.search(text, pos)
    if start is None:
        return None
    bracket = text.find(">", start.end())
    if bracket < 0:
        return None
    return start.start(), bracket + 1


def _strip_paired_think_blocks(text: str, reasoning_parts: list[str]) -> str:
    """Remove paired ``<think>...</think>`` blocks from *text*, capturing their bodies.

    A block runs from an open tag to the first close tag after it, so an open
    tag inside a block is part of its reasoning. Once an open tag has no close
    tag after it, no later one has either, and the scan ends there.

    Args:
        text (str): Input text.
        reasoning_parts (list[str]): Accumulator appended in place.

    Returns:
        str: The text without its paired blocks.
    """
    kept: list[str] = []
    pos = 0
    while (opening := _find_open_think(text, pos)) is not None:
        close = _CLOSE_THINK_RE.search(text, opening[1])
        if close is None:
            break
        body = text[opening[1] : close.start()].strip()
        if body:
            reasoning_parts.append(body)
        kept.append(text[pos : opening[0]])
        pos = close.end()
    kept.append(text[pos:])
    return "".join(kept)


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
    opening = _find_open_think(text)

    if close_match and (opening is None or close_match.start() < opening[0]):
        before = text[: close_match.start()].strip()
        if before:
            reasoning_parts.append(before)
        text = text[close_match.end() :]
        opening = _find_open_think(text)

    if opening is not None:
        tail = text[opening[1] :].strip()
        if tail:
            reasoning_parts.append(tail)
        text = text[: opening[0]]

    return text


# A tail that could still grow into an open or a close tag, so a stream holds it back.
_OPEN_THINK_TAIL_RE: re.Pattern[str] = re.compile(r"<(?:t(?:h(?:i(?:n(?:k(?:[^\w>][^>]*)?)?)?)?)?)?\Z", re.IGNORECASE)
_CLOSE_THINK_TAIL_RE: re.Pattern[str] = re.compile(r"<(?:/(?:t(?:h(?:i(?:n(?:k\s*)?)?)?)?)?)?\Z", re.IGNORECASE)


def _tail_start(text: str, tail: re.Pattern[str]) -> int:
    """Return where a tag that may still be forming begins at the end of *text*.

    Args:
        text (str): The text held so far.
        tail (re.Pattern[str]): The partial-tag pattern to look for.

    Returns:
        int: Its start, or ``len(text)`` when the text cannot end inside a tag.
    """
    match = tail.search(text)
    return match.start() if match else len(text)


class ReasoningStripper:
    """Strip reasoning from a reply while it streams: the streaming counterpart of :func:`strip_reasoning`.

    Feed the reply's content a piece at a time; each call returns the visible text that piece
    completes. Held back is only what may still turn out otherwise: a tail that could grow
    into a tag, trailing whitespace, and, for a reply that may open in reasoning, everything up
    to its first close tag. Joined, the returned text is ``strip_reasoning(reply)[0]`` and
    :attr:`reasoning` its second value, for paired blocks, a block left open at the end, and,
    when the reply may open in reasoning, a close tag with no open tag before it.

    Harmony channels are not read. Without ``reasoning_open``, a close tag with no open tag
    before it is passed on as text: the text before it may already have been shown.
    """

    def __init__(self, *, reasoning_open: bool = False) -> None:
        """Start before any content.

        Args:
            reasoning_open (bool): Whether the reply may open inside a think block the chat
                template began in the prompt, so that only a close tag shows where it ends.
        """
        self._state: Literal["held", "text", "think"] = "held" if reasoning_open else "text"
        self._buffer = ""
        self._scan = 0
        self._think = ""
        self._paired: list[str] = []
        self._dangling = ""
        self._unclosed = ""
        self._started = False
        self._space = ""

    @property
    def reasoning(self) -> str | None:
        """Return the reasoning read so far, as :func:`strip_reasoning` joins it.

        Returns:
            str | None: Paired blocks in order, then the text before a close tag with no open
            tag, then a block left open; each stripped, joined by newlines. ``None`` when empty.
        """
        return "\n".join(part for part in (*self._paired, self._dangling, self._unclosed) if part) or None

    def feed(self, text: str) -> str:
        """Accept the next piece of the reply.

        Args:
            text (str): The piece.

        Returns:
            str: The visible text it completes; empty while all of it is held.
        """
        self._buffer += text
        return self._drain()

    def cancel_open_reasoning(self) -> str:
        """Read held content as ordinary text, because the endpoint sends reasoning apart from it.

        Returns:
            str: The visible text the held content yields; empty when nothing was held.
        """
        if self._state != "held":
            return ""
        self._state = "text"
        return self._drain()

    def flush(self) -> str:
        """Release what is still held once the reply has ended.

        A reply still waiting for a close tag never opened in reasoning, so its held content is
        read as text; a block left open is reasoning; a tail that never became a tag is text;
        trailing whitespace is dropped.

        Returns:
            str: The visible text that remained.
        """
        out = self.cancel_open_reasoning()
        if self._state == "think":
            self._unclosed = (self._think + self._buffer).strip()
            self._think = ""
        else:
            out += self._visible(self._buffer)
        self._buffer = self._space = ""
        return out

    def _drain(self) -> str:
        """Read the buffer as far as it can be read.

        Returns:
            str: The visible text read.
        """
        out: list[str] = []
        while self._buffer:
            if self._state == "held":
                close = _CLOSE_THINK_RE.search(self._buffer, self._scan)
                if close is None:
                    # A close tag can only begin at the last "<" or after it.
                    last = self._buffer.rfind("<")
                    self._scan = last if last >= 0 else len(self._buffer)
                    break
                self._end_held(self._buffer[: close.start()])
                self._buffer, self._state = self._buffer[close.end() :], "text"
            elif self._state == "think":
                close = _CLOSE_THINK_RE.search(self._buffer)
                if close is None:
                    cut = _tail_start(self._buffer, _CLOSE_THINK_TAIL_RE)
                    self._think += self._buffer[:cut]
                    self._buffer = self._buffer[cut:]
                    break
                body = (self._think + self._buffer[: close.start()]).strip()
                if body:
                    self._paired.append(body)
                self._think = ""
                self._buffer, self._state = self._buffer[close.end() :], "text"
            else:
                opening = _find_open_think(self._buffer)
                if opening is None:
                    cut = _tail_start(self._buffer, _OPEN_THINK_TAIL_RE)
                    out.append(self._visible(self._buffer[:cut]))
                    self._buffer = self._buffer[cut:]
                    break
                out.append(self._visible(self._buffer[: opening[0]]))
                self._buffer, self._state = self._buffer[opening[1] :], "think"
        return "".join(out)

    def _end_held(self, held: str) -> None:
        """Record what came before the first close tag of a reply that opened in reasoning.

        Args:
            held (str): The held content. If the model wrote its own open tag in it, the tag
                is no part of the reasoning and the block counts as a paired one.
        """
        opening = _find_open_think(held)
        if opening is None:
            self._dangling = held.strip()
            return
        body = (held[: opening[0]] + held[opening[1] :]).strip()
        if body:
            self._paired.append(body)

    def _visible(self, text: str) -> str:
        """Pass visible text on, dropping the reply's leading whitespace and holding trailing whitespace.

        Args:
            text (str): Visible text just read.

        Returns:
            str: What may be shown now.
        """
        if not self._started:
            text = text.lstrip()
            if not text:
                return ""
            self._started = True
        text = self._space + text
        shown = text.rstrip()
        self._space = text[len(shown) :]
        return shown


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


# Longest run of one repeated character allowed through ``squeeze_char_runs``.
# Chosen to keep every legitimate separator (table rules, ASCII dividers,
# dotted leader lines short enough to be typed by hand) while catching
# degenerate generation, which overshoots this by orders of magnitude.
_MAX_CHAR_RUN: int = 20

_CHAR_RUN_PATTERN = re.compile(r"(\S)\1{" + str(_MAX_CHAR_RUN) + ",}")


def squeeze_char_runs(text: str, *, max_run: int = _MAX_CHAR_RUN) -> str:
    """Collapse degenerate character runs to a plausible separator length.

    A vision model transcribing a form's dotted fill-in line can lock into
    repeating the fill character until ``max_tokens`` cuts it off — observed
    live as a single run of 65392 dots that went on to become three of a
    document's five chunks. A human transcriber writes a short leader line;
    so does this.

    Whitespace runs are exempt: blank-line stretches are page layout, and
    they are handled by chunking, not by output hygiene.

    Args:
        text (str): Model output text.
        max_run (int): Longest run of one repeated non-whitespace character
            to keep.

    Returns:
        str: ``text`` with every longer run truncated to ``max_run``.
    """
    if not text:
        return text
    if max_run == _MAX_CHAR_RUN:
        pattern = _CHAR_RUN_PATTERN
    else:
        pattern = re.compile(r"(\S)\1{" + str(max_run) + ",}")
    return pattern.sub(lambda match: match.group(1) * max_run, text)
