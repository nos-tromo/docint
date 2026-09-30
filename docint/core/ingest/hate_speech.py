"""Stance-aware hate-speech helpers shared by docint's chunk and transcript paths.

A verdict is derived from the speaker's (or author's) *stance* toward
group-focused enmity, never from a model-emitted boolean: only ``endorses``
is hate speech. Quoting, reporting, condemning, analysing or asking about hate
is not hate — the use/mention distinction the per-chunk prompt used to miss.
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any, TypedDict

import openai

from docint.utils.llm_sanitize import strip_reasoning

HATE_SPEECH_STANCES: tuple[str, ...] = (
    "endorses",
    "quotes_or_reports",
    "condemns_or_counters",
    "analyzes_or_discusses",
    "unclear",
)
"""Stances the model may assign to content that touches group-focused enmity."""

CHUNK_STANCES: tuple[str, ...] = (*HATE_SPEECH_STANCES, "none")
"""Chunk-level stances; ``none`` means the chunk holds no such content at all."""

HATE_SPEECH_CATEGORIES: tuple[str, ...] = (
    "race",
    "ethnicity",
    "religion",
    "gender",
    "sexual_orientation",
    "disability",
    "nationality",
    "extremism",
    "other",
)
"""GMF categories; unknown labels are normalised to ``other``."""

CHUNK_CATEGORIES: tuple[str, ...] = (*HATE_SPEECH_CATEGORIES, "none")
"""Chunk-level categories; ``none`` for chunks without group-focused content."""

CONFIDENCE_LEVELS: tuple[str, ...] = ("high", "medium", "low")
"""Allowed confidence values; unknown values are normalised to ``low``."""

REASON_MAX_CHARS: int = 500
"""Cap on a stored rationale, so one reply cannot bloat node metadata."""

CHARS_PER_TOKEN: float = 3.0
"""Conservative characters-per-token estimate turning window token budgets into character budgets."""

_TRANSLATION_PREFIX: str = "\n    → "
_EMPTY_BLOCK: str = "—"
_PLACEHOLDER_RE = re.compile(r"\{(language|context_before|segments|context_after|first_index|last_index)\}")
_INDEX_TEXT_RE = re.compile(r"^\[?\s*(\d+)\s*\]?$")

_CONTEXT_LENGTH_ERROR_MARKERS: tuple[str, ...] = (
    "context length",
    "context_length",
    "context window",
    "maximum context",
    "too many tokens",
    "reduce the length",
    "exceeds the maximum",
)


def normalize_choice(value: Any, choices: Sequence[str], default: str) -> str:
    """Normalise an enum-like string (case, whitespace, separators) against ``choices``.

    Args:
        value (Any): The raw value.
        choices (Sequence[str]): Allowed values.
        default (str): Value returned for anything not in ``choices``.

    Returns:
        str: The matching choice, or ``default``.
    """
    if not isinstance(value, str):
        return default
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized if normalized in choices else default


def chunk_response_format() -> dict[str, Any]:
    """Build the ``response_format`` JSON schema for one document-chunk verdict.

    Portable across vLLM, Ollama and OpenAI strict mode: every property is
    required and none may be added. ``reason`` precedes ``stance`` in
    declaration order and alphabetically, so the rationale is generated before
    the decision whichever key order a backend follows.

    Returns:
        dict[str, Any]: The ``response_format`` payload.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "hate_speech_verdict",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["target", "reason", "stance", "category", "confidence"],
                "properties": {
                    "target": {"type": "string"},
                    "reason": {"type": "string"},
                    "stance": {"type": "string", "enum": list(CHUNK_STANCES)},
                    "category": {"type": "string", "enum": list(CHUNK_CATEGORIES)},
                    "confidence": {"type": "string", "enum": list(CONFIDENCE_LEVELS)},
                },
            },
        },
    }


def is_context_length_error(exc: BaseException) -> bool:
    """Report whether an error is a context-window overflow.

    Args:
        exc (BaseException): The error raised by an inference call.

    Returns:
        bool: ``True`` when the message names a context-length overflow.
    """
    message = str(exc).lower()
    return any(marker in message for marker in _CONTEXT_LENGTH_ERROR_MARKERS)


def is_structured_output_rejection(exc: BaseException) -> bool:
    """Report whether an error may be a provider rejecting ``response_format``.

    Context overflows are also reported as HTTP 400 by some providers (vLLM),
    so they are excluded here.

    Args:
        exc (BaseException): The error raised by an inference call.

    Returns:
        bool: ``True`` for HTTP 400/422 client errors that are not overflows.
    """
    return isinstance(exc, openai.APIStatusError) and exc.status_code in (400, 422) and not is_context_length_error(exc)


# ---------------------------------------------------------------------------
# Transcript context windows — semantics mirror Nextext's
# ``nextext/core/hate_speech.py``; keep the two in step.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TranscriptLine:
    """One transcript segment as the window classifier sees it.

    Attributes:
        index (int): Identifier the model reports back (position in the transcript).
        text (str): Segment text, stripped.
        speaker (str | None): Diarization label, or ``None`` when unknown.
        translation (str | None): Translation shown as an aid, or ``None``.
    """

    index: int
    text: str
    speaker: str | None = None
    translation: str | None = None


@dataclass(frozen=True)
class TranscriptWindow:
    """Positions (into the line list) of one classification window.

    The core ``[core_start, core_end)`` is labelled; the margins
    ``[context_start, core_start)`` and ``[core_end, context_end)`` are
    read-only context. Consecutive windows have disjoint cores.

    Attributes:
        context_start (int): First context position before the core.
        core_start (int): First labelled position.
        core_end (int): One past the last labelled position.
        context_end (int): One past the last context position after the core.
    """

    context_start: int
    core_start: int
    core_end: int
    context_end: int


class WindowFinding(TypedDict):
    """One segment whose speaker endorses group-focused enmity.

    Attributes:
        index (int): ``TranscriptLine.index`` of the segment.
        category (str): One of :data:`HATE_SPEECH_CATEGORIES`.
        confidence (str): One of :data:`CONFIDENCE_LEVELS`.
        reason (str): Short rationale in the prompt's language.
    """

    index: int
    category: str
    confidence: str
    reason: str


def _clip(text: str, limit: int, *, keep_tail: bool = False) -> str:
    """Shorten ``text`` to at most ``limit`` characters, marking the cut with ``…``.

    Args:
        text (str): The text to clip.
        limit (int): Maximum length of the result.
        keep_tail (bool): Keep the end of the text instead of its start.

    Returns:
        str: ``text`` unchanged when it fits, otherwise the clipped text.
    """
    if len(text) <= limit:
        return text
    if limit <= 1:
        return "…"
    return "…" + text[-(limit - 1) :] if keep_tail else text[: limit - 1] + "…"


def render_line(line: TranscriptLine, cap: int, *, keep_tail: bool = False) -> str:
    """Render one segment as ``[index] Speaker: text`` plus an optional translation aid line.

    Args:
        line (TranscriptLine): The segment.
        cap (int): Maximum characters kept of the text and, separately, of the translation.
        keep_tail (bool): Clip from the start instead of the end (context before the core).

    Returns:
        str: The rendered segment.
    """
    prefix = f"[{line.index}] {line.speaker}: " if line.speaker else f"[{line.index}] "
    rendered = prefix + _clip(line.text, cap, keep_tail=keep_tail)
    if line.translation:
        rendered += _TRANSLATION_PREFIX + _clip(line.translation, cap, keep_tail=keep_tail)
    return rendered


def _line_cost(line: TranscriptLine, cap: int, *, keep_tail: bool = False) -> int:
    """Return the rendered size of a segment including its trailing newline.

    Args:
        line (TranscriptLine): The segment.
        cap (int): The clip limit it will be rendered with.
        keep_tail (bool): Whether it is rendered tail-first.

    Returns:
        int: Characters the segment occupies in the prompt.
    """
    return len(render_line(line, cap, keep_tail=keep_tail)) + 1


def next_window(lines: Sequence[TranscriptLine], start: int, core_chars: int, context_chars: int) -> TranscriptWindow:
    """Build the classification window whose core begins at ``start``.

    The core takes segments while their rendered cost fits ``core_chars`` but
    always at least one, so a sweep always advances. Context margins walk
    outward: the adjacent segment is always included (clipped to
    ``context_chars``), further ones only while they fit. ``context_chars <= 0``
    disables the margins.

    Args:
        lines (Sequence[TranscriptLine]): All classifiable segments, in order.
        start (int): Position of the first core segment; must be ``< len(lines)``.
        core_chars (int): Character budget of the core.
        context_chars (int): Character budget of each context margin.

    Returns:
        TranscriptWindow: The window's positions.
    """
    core_end = start
    used = 0
    while core_end < len(lines):
        cost = _line_cost(lines[core_end], core_chars)
        if core_end > start and used + cost > core_chars:
            break
        used += cost
        core_end += 1

    context_start = start
    context_end = core_end
    if context_chars > 0:
        remaining = context_chars
        position = start - 1
        while position >= 0 and remaining > 0:
            cost = _line_cost(lines[position], context_chars, keep_tail=True)
            if position < start - 1 and cost > remaining:
                break
            remaining -= cost
            context_start = position
            position -= 1

        remaining = context_chars
        position = core_end
        while position < len(lines) and remaining > 0:
            cost = _line_cost(lines[position], context_chars)
            if position > core_end and cost > remaining:
                break
            remaining -= cost
            position += 1
            context_end = position

    return TranscriptWindow(context_start=context_start, core_start=start, core_end=core_end, context_end=context_end)


def render_window_prompt(
    template: str,
    lines: Sequence[TranscriptLine],
    window: TranscriptWindow,
    *,
    core_chars: int,
    context_chars: int,
    language: str,
) -> str:
    """Fill the transcript prompt template for one window in a single pass.

    Args:
        template (str): Template with ``{language}``, ``{context_before}``,
            ``{segments}``, ``{context_after}``, ``{first_index}`` and
            ``{last_index}`` placeholders.
        lines (Sequence[TranscriptLine]): All classifiable segments, in order.
        window (TranscriptWindow): The window to render.
        core_chars (int): Clip limit for core segments.
        context_chars (int): Clip limit for context segments.
        language (str): Transcript language label.

    Returns:
        str: The rendered prompt; an empty context block renders as ``—``.
    """

    def block(start: int, end: int, cap: int, *, keep_tail: bool = False) -> str:
        """Render segments ``[start, end)`` one per line.

        Args:
            start (int): First position.
            end (int): One past the last position.
            cap (int): Clip limit.
            keep_tail (bool): Clip tail-first.

        Returns:
            str: The rendered segments, or ``—`` when the range is empty.
        """
        rendered = [render_line(lines[position], cap, keep_tail=keep_tail) for position in range(start, end)]
        return "\n".join(rendered) if rendered else _EMPTY_BLOCK

    values = {
        "language": language,
        "context_before": block(window.context_start, window.core_start, context_chars, keep_tail=True),
        "segments": block(window.core_start, window.core_end, core_chars),
        "context_after": block(window.core_end, window.context_end, context_chars),
        "first_index": str(lines[window.core_start].index),
        "last_index": str(lines[window.core_end - 1].index),
    }
    return _PLACEHOLDER_RE.sub(lambda match: values[match.group(1)], template)


def window_response_format(indices: Sequence[int]) -> dict[str, Any]:
    """Build the ``response_format`` JSON schema for one transcript window.

    Args:
        indices (Sequence[int]): Indices of the window's core segments.

    Returns:
        dict[str, Any]: The ``response_format`` payload; ``index`` admits core
            segments only.
    """
    item_schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["index", "target", "reason", "stance", "category", "confidence"],
        "properties": {
            "index": {"type": "integer", "enum": list(indices)},
            "target": {"type": "string"},
            "reason": {"type": "string"},
            "stance": {"type": "string", "enum": list(HATE_SPEECH_STANCES)},
            "category": {"type": "string", "enum": list(HATE_SPEECH_CATEGORIES)},
            "confidence": {"type": "string", "enum": list(CONFIDENCE_LEVELS)},
        },
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "hate_speech_findings",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["findings"],
                "properties": {"findings": {"type": "array", "items": item_schema}},
            },
        },
    }


def _looks_like_findings(candidate: Any) -> bool:
    """Report whether a decoded JSON value can stand in for a findings payload.

    Args:
        candidate (Any): A decoded JSON value.

    Returns:
        bool: ``True`` for a single finding object or a list of objects.
    """
    if isinstance(candidate, dict):
        return "index" in candidate
    if isinstance(candidate, list):
        return all(isinstance(item, dict) for item in candidate)
    return False


def _load_json_payload(text: str) -> Any:
    """Decode the findings payload from a reply that may wrap it in prose or fences.

    Args:
        text (str): The reply with reasoning removed.

    Returns:
        Any: The first ``{"findings": ...}`` object, else the first value that
            looks like findings, else the whole reply decoded (or ``None``).
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    fallback: Any = None
    for position, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            candidate, _ = decoder.raw_decode(text, position)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "findings" in candidate:
            return candidate
        if fallback is None and _looks_like_findings(candidate):
            fallback = candidate
    return fallback


def _coerce_index(value: Any) -> int | None:
    """Convert a model-emitted segment index to ``int``.

    Args:
        value (Any): The ``index`` value.

    Returns:
        int | None: The index, or ``None`` for booleans, fractions and junk.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        match = _INDEX_TEXT_RE.match(value.strip())
        return int(match.group(1)) if match else None
    return None


def parse_window_reply(raw: str, allowed: Collection[int]) -> list[WindowFinding] | None:
    """Parse a window reply into the segments whose speaker endorses hate.

    A segment is a finding if and only if its stance is ``endorses``; items for
    segments outside ``allowed`` are dropped and the first item per segment wins.

    Args:
        raw (str): The raw model reply.
        allowed (Collection[int]): Indices of the window's core segments.

    Returns:
        list[WindowFinding] | None: Findings in order (``[]`` when none), or
            ``None`` when the reply holds no usable findings structure.
    """
    cleaned, _ = strip_reasoning(raw or "")
    payload = _load_json_payload(cleaned.strip())
    items: Any
    if isinstance(payload, dict):
        items = payload.get("findings") if "findings" in payload else ([payload] if "index" in payload else None)
    elif isinstance(payload, list):
        items = payload
    else:
        return None
    if not isinstance(items, list):
        return None

    allowed_set = set(allowed)
    seen: set[int] = set()
    findings: list[WindowFinding] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        index = _coerce_index(item.get("index"))
        if index is None or index not in allowed_set or index in seen:
            continue
        seen.add(index)
        if normalize_choice(item.get("stance"), HATE_SPEECH_STANCES, "unclear") != "endorses":
            continue
        findings.append(
            WindowFinding(
                index=index,
                category=normalize_choice(item.get("category"), HATE_SPEECH_CATEGORIES, "other"),
                confidence=normalize_choice(item.get("confidence"), CONFIDENCE_LEVELS, "low"),
                reason=str(item.get("reason") or "").strip()[:REASON_MAX_CHARS],
            )
        )
    findings.sort(key=lambda finding: finding["index"])
    return findings
