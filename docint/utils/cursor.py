"""Opaque cursor tokens for paginated Qdrant scrolls.

The cursor wraps a Qdrant ``next_page_offset`` (a point id — ``int``, ``str``,
or :class:`uuid.UUID`) so it can be passed between client and server as a
single string without leaking internal pagination state.

Cursor format: base64-urlsafe JSON ``{"v": 1, "o": <offset>}``. The version
field reserves room to extend the schema later (e.g. carrying a flushed-filename
set for the ``iter_documents`` generator).
"""

from __future__ import annotations

import base64
import json
import uuid
from typing import Any

from loguru import logger

CURSOR_VERSION = 1


class InvalidCursorError(ValueError):
    """Raised when a cursor token cannot be parsed."""


def encode_cursor(offset: Any, *, extra: dict[str, Any] | None = None) -> str | None:
    """Encode a Qdrant scroll offset as an opaque cursor token.

    Args:
        offset (Any): Native Qdrant point id (``int``, ``str``, or
            :class:`uuid.UUID`) or ``None`` when there are no further pages.
        extra (dict[str, Any] | None): Optional supplementary fields to
            include in the cursor payload (e.g. ``{"flushed": [...]}`` for
            document de-duplication across pages).

    Returns:
        str | None: Base64-urlsafe encoded JSON token, or ``None`` if
        ``offset`` is ``None`` (signalling the end of pagination).
    """
    if offset is None:
        return None

    if isinstance(offset, uuid.UUID):
        offset_value: Any = str(offset)
    else:
        offset_value = offset

    payload: dict[str, Any] = {"v": CURSOR_VERSION, "o": offset_value}
    if extra:
        payload.update(extra)

    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cursor(token: str | None) -> dict[str, Any]:
    """Decode an opaque cursor token into its payload dict.

    Args:
        token (str | None): Cursor token previously produced by
            :func:`encode_cursor`, or ``None`` / empty string for the start
            of the result set.

    Returns:
        dict[str, Any]: The decoded payload with at minimum an ``"o"`` field;
        returns ``{"o": None}`` for the first page.

    Raises:
        InvalidCursorError: If the token is malformed or its version is
            incompatible with this code path.
    """
    if not token:
        return {"o": None}

    try:
        padding = "=" * (-len(token) % 4)
        raw = base64.urlsafe_b64decode(token + padding)
        payload = json.loads(raw.decode("utf-8"))
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("Rejected malformed cursor token: {!r}", token)
        raise InvalidCursorError(f"malformed cursor token: {token!r}") from exc

    if not isinstance(payload, dict):
        raise InvalidCursorError(f"cursor payload is not a dict: {payload!r}")

    version = payload.get("v")
    if version != CURSOR_VERSION:
        raise InvalidCursorError(f"unsupported cursor version: {version!r} (expected {CURSOR_VERSION})")

    if "o" not in payload:
        raise InvalidCursorError("cursor payload missing required 'o' field")

    return payload
