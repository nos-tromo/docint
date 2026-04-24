"""Tests for :mod:`docint.utils.metadata_sanitize`."""

from __future__ import annotations

import datetime
import decimal
import enum
import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from docint.utils.metadata_sanitize import sanitize_for_json


class _Color(enum.Enum):
    """Sample enum used to exercise enum value unwrapping."""

    RED = "red"
    NUMERIC = 7


class _Opaque:
    """Value-less class whose ``str()`` output is stable and test-visible.

    ``sanitize_for_json`` falls back to ``str(value)`` for unknown types,
    so the fixture overrides ``__str__`` directly rather than relying on
    Python's implicit ``__str__ → __repr__`` fallback, which would make
    the test contract misleading if anyone ever added a real ``__str__``.
    """

    def __str__(self) -> str:
        """Return a stable ``str`` so tests can assert exact fallback output.

        Returns:
            The literal ``"<opaque>"`` marker.
        """
        return "<opaque>"


def test_primitives_pass_through() -> None:
    """Scalars that are already JSON-safe must be returned unchanged."""
    assert sanitize_for_json(None) is None
    assert sanitize_for_json("hello") == "hello"
    assert sanitize_for_json(42) == 42
    assert sanitize_for_json(3.14) == 3.14
    assert sanitize_for_json(True) is True
    assert sanitize_for_json(False) is False


def test_float_nan_and_inf_become_none() -> None:
    """NaN and ±inf values are not JSON-valid; they must collapse to ``None``."""
    assert sanitize_for_json(float("nan")) is None
    assert sanitize_for_json(float("inf")) is None
    assert sanitize_for_json(float("-inf")) is None


def test_datetime_time_uses_isoformat() -> None:
    """``datetime.time`` becomes its ``isoformat`` string (HH:MM:SS)."""
    assert sanitize_for_json(datetime.time(8, 30)) == "08:30:00"
    assert (
        sanitize_for_json(datetime.time(23, 59, 59, 500000)) == "23:59:59.500000"
    )


def test_datetime_date_uses_isoformat() -> None:
    """``datetime.date`` becomes its ISO-8601 date string (YYYY-MM-DD)."""
    assert sanitize_for_json(datetime.date(2026, 4, 24)) == "2026-04-24"


def test_datetime_datetime_uses_isoformat() -> None:
    """``datetime.datetime`` becomes its ISO-8601 string with ``T`` separator."""
    ts = datetime.datetime(2026, 4, 24, 8, 30, 15)
    assert sanitize_for_json(ts) == "2026-04-24T08:30:15"


def test_timedelta_coerced_to_string() -> None:
    """``datetime.timedelta`` becomes ``str(td)`` (e.g. ``0:30:00``)."""
    assert sanitize_for_json(datetime.timedelta(minutes=30)) == "0:30:00"


def test_pandas_timestamp_uses_isoformat() -> None:
    """``pandas.Timestamp`` inherits from ``datetime.datetime`` and uses ISO format."""
    assert sanitize_for_json(pd.Timestamp("2026-04-24 08:30:00")) == (
        "2026-04-24T08:30:00"
    )


def test_pandas_timedelta_coerced_to_string() -> None:
    """``pandas.Timedelta`` inherits from ``datetime.timedelta``; fall back to ``str``."""
    result = sanitize_for_json(pd.Timedelta("1 day 02:00:00"))
    assert isinstance(result, str)
    assert "1 day" in result


def test_pandas_period_coerced_to_string() -> None:
    """``pandas.Period`` is pandas-specific and coerces via ``str``."""
    assert sanitize_for_json(pd.Period("2026-04", freq="M")) == "2026-04"


def test_pandas_na_and_nat_become_none() -> None:
    """``pd.NA`` and ``pd.NaT`` are missing-data sentinels and map to ``None``."""
    assert sanitize_for_json(pd.NA) is None
    assert sanitize_for_json(pd.NaT) is None


def test_numpy_scalar_returns_python_primitive() -> None:
    """``numpy`` scalars convert to the equivalent Python primitive via ``.item()``."""
    assert sanitize_for_json(np.int64(5)) == 5
    assert sanitize_for_json(np.float32(1.5)) == pytest.approx(1.5)
    assert sanitize_for_json(np.bool_(True)) is True


def test_numpy_nan_scalar_becomes_none() -> None:
    """A numpy NaN scalar unwraps to Python ``float('nan')`` then to ``None``."""
    assert sanitize_for_json(np.float64("nan")) is None


def test_numpy_ndarray_becomes_plain_list() -> None:
    """``numpy.ndarray`` values recursively convert to native Python lists."""
    arr = np.array([[1, 2], [3, 4]])
    assert sanitize_for_json(arr) == [[1, 2], [3, 4]]


def test_decimal_uses_string_for_precision() -> None:
    """``Decimal`` values must preserve precision via ``str``, not ``float``."""
    assert sanitize_for_json(decimal.Decimal("1.2345678901234567890")) == (
        "1.2345678901234567890"
    )


def test_uuid_path_and_bytes_coerce_to_string() -> None:
    """UUID / Path / bytes become human-readable strings."""
    uid = uuid.UUID("12345678-1234-5678-1234-567812345678")
    assert sanitize_for_json(uid) == "12345678-1234-5678-1234-567812345678"
    assert sanitize_for_json(Path("/tmp/x.txt")) == "/tmp/x.txt"
    assert sanitize_for_json(b"hello") == "hello"
    assert sanitize_for_json(bytearray(b"world")) == "world"


def test_bytes_with_invalid_utf8_uses_replacement() -> None:
    """Undecodable bytes must not raise — they use the replacement character."""
    result = sanitize_for_json(b"\xff\xfe")
    assert isinstance(result, str)


def test_enum_unwraps_to_inner_value() -> None:
    """Enums return their ``.value`` after recursive sanitation."""
    assert sanitize_for_json(_Color.RED) == "red"
    assert sanitize_for_json(_Color.NUMERIC) == 7


def test_sets_become_sorted_lists() -> None:
    """Sets and frozensets become lists; sortable members are emitted in order."""
    assert sanitize_for_json({3, 1, 2}) == [1, 2, 3]
    assert sanitize_for_json(frozenset({"b", "a"})) == ["a", "b"]


def test_tuple_becomes_list() -> None:
    """Tuples convert to lists for JSON compatibility."""
    assert sanitize_for_json((1, "x", 3.0)) == [1, "x", 3.0]


def test_nested_dict_recurses_and_coerces_keys() -> None:
    """Nested containers recurse; non-string dict keys are stringified."""
    payload = {
        1: datetime.time(8, 30),
        "nested": {
            "time": datetime.time(9, 0),
            "numbers": [np.int64(1), np.int64(2)],
        },
    }
    assert sanitize_for_json(payload) == {
        "1": "08:30:00",
        "nested": {"time": "09:00:00", "numbers": [1, 2]},
    }


def test_mixed_payload_round_trips_through_json_dumps() -> None:
    """The sanitized output of a realistic mixed payload must serialize cleanly."""
    payload = {
        "start": datetime.time(8, 30),
        "end": datetime.datetime(2026, 4, 24, 9, 45),
        "timedelta": datetime.timedelta(minutes=15),
        "decimal": decimal.Decimal("3.14"),
        "arr": np.array([1, 2, 3]),
        "not_a_number": float("nan"),
        "missing": pd.NaT,
        "nested": {"tags": {"a", "b"}},
    }
    sanitized = sanitize_for_json(payload)
    # Strict JSON parse: NaN / Infinity must not appear as literal values.
    roundtrip = json.loads(json.dumps(sanitized))
    assert roundtrip["start"] == "08:30:00"
    assert roundtrip["end"] == "2026-04-24T09:45:00"
    assert roundtrip["timedelta"] == "0:15:00"
    assert roundtrip["decimal"] == "3.14"
    assert roundtrip["arr"] == [1, 2, 3]
    assert roundtrip["not_a_number"] is None
    assert roundtrip["missing"] is None
    assert roundtrip["nested"] == {"tags": ["a", "b"]}


def test_sanitize_is_idempotent() -> None:
    """Applying the helper twice must produce the same output as applying it once."""
    payload = {
        "t": datetime.time(8, 30),
        "items": [decimal.Decimal("1.5"), {"nested": np.int64(3)}],
    }
    first = sanitize_for_json(payload)
    second = sanitize_for_json(first)
    assert first == second
    assert json.dumps(first) == json.dumps(second)


def test_unknown_type_falls_back_to_str() -> None:
    """Unknown objects must coerce via ``str(...)`` rather than raising."""
    assert sanitize_for_json(_Opaque()) == "<opaque>"
