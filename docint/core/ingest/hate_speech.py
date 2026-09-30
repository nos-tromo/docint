"""Stance-aware hate-speech helpers shared by docint's chunk and transcript paths.

A verdict is derived from the speaker's (or author's) *stance* toward
group-focused enmity, never from a model-emitted boolean: only ``endorses``
is hate speech. Quoting, reporting, condemning, analysing or asking about hate
is not hate — the use/mention distinction the per-chunk prompt used to miss.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import openai

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
