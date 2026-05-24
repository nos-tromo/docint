"""Tests for the opaque cursor token helpers."""

from __future__ import annotations

import base64
import json
import uuid

import pytest

from docint.utils.cursor import (
    CURSOR_VERSION,
    InvalidCursorError,
    decode_cursor,
    encode_cursor,
)


def _encode_raw(payload: object) -> str:
    """Encode an arbitrary payload as a cursor token (test helper)."""
    raw = json.dumps(payload).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def test_encode_none_returns_none() -> None:
    assert encode_cursor(None) is None


def test_encode_decode_round_trip_int() -> None:
    token = encode_cursor(42)
    assert token is not None
    payload = decode_cursor(token)
    assert payload == {"v": CURSOR_VERSION, "o": 42}


def test_encode_decode_round_trip_str() -> None:
    token = encode_cursor("point-abc")
    assert token is not None
    payload = decode_cursor(token)
    assert payload["o"] == "point-abc"
    assert payload["v"] == CURSOR_VERSION


def test_encode_decode_round_trip_uuid() -> None:
    pid = uuid.uuid4()
    token = encode_cursor(pid)
    assert token is not None
    payload = decode_cursor(token)
    assert payload["o"] == str(pid)


def test_encode_with_extra_fields_round_trips() -> None:
    token = encode_cursor("p1", extra={"flushed": ["a", "b"], "n": 3})
    payload = decode_cursor(token)
    assert payload["o"] == "p1"
    assert payload["flushed"] == ["a", "b"]
    assert payload["n"] == 3


def test_decode_empty_token_returns_start_of_results() -> None:
    assert decode_cursor(None) == {"o": None}
    assert decode_cursor("") == {"o": None}


def test_decode_malformed_token_raises() -> None:
    with pytest.raises(InvalidCursorError):
        decode_cursor("$$ not a valid token $$")


def test_decode_wrong_version_raises() -> None:
    bad = _encode_raw({"v": CURSOR_VERSION + 999, "o": "x"})
    with pytest.raises(InvalidCursorError):
        decode_cursor(bad)


def test_decode_non_dict_payload_raises() -> None:
    bad = _encode_raw("just-a-string")
    with pytest.raises(InvalidCursorError):
        decode_cursor(bad)


def test_decode_missing_o_field_raises() -> None:
    bad = _encode_raw({"v": CURSOR_VERSION})
    with pytest.raises(InvalidCursorError):
        decode_cursor(bad)


def test_decode_token_padding_tolerant() -> None:
    token = encode_cursor("padding-test-value")
    assert token is not None
    assert decode_cursor(token)["o"] == "padding-test-value"
    assert decode_cursor(token + "==")["o"] == "padding-test-value"


def test_encode_produces_urlsafe_chars() -> None:
    token = encode_cursor("contains/slash+plus=eq")
    assert token is not None
    assert "/" not in token
    assert "+" not in token
