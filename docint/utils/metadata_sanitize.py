"""JSON-safe coercion for node / document metadata before persistence.

The LlamaIndex ``SQLiteKVStore`` backing ``DocStore.add_documents`` persists
node metadata by round-tripping it through :func:`json.dumps`. Pandas-backed
readers (see :mod:`docint.core.readers.tables`) can surface
``datetime.time``, ``datetime.datetime``, ``pandas.Timestamp``, ``numpy``
scalars, ``decimal.Decimal`` and other non-JSON-serializable leaves that
would otherwise crash the entire persist batch with ``TypeError: Object of
type X is not JSON serializable``.

This module provides a single recursive helper that converts those values
into JSON-friendly primitives. It is the source-side counterpart to the
``default=`` handler in :mod:`docint.core.storage.sqlite_kvstore`; applying
it at source preserves type fidelity — for example ``datetime.datetime``
values become ISO-8601 strings (``2024-01-01T08:30:00``) via
``.isoformat()`` rather than the space-separated representation that
``str(...)`` produces.
"""

from __future__ import annotations

import datetime
import decimal
import enum
import math
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

_LOGGED_UNKNOWN_TYPES: set[str] = set()


def _is_pandas_missing(value: Any) -> bool:
    """Return ``True`` when *value* is a pandas / numpy missing-data sentinel.

    Detects ``pandas.NA``, ``pandas.NaT``, and ``numpy.datetime64('NaT')``.
    Callers route missing-data scalars to ``None`` rather than letting them
    fall through to a stringified ``"NaT"`` / ``"<NA>"``.

    Args:
        value: Candidate scalar. Container types are filtered out by the
            caller before this function is invoked.

    Returns:
        ``True`` when ``pd.isna(value)`` returns a scalar ``True``;
        ``False`` otherwise, including when ``pd.isna`` cannot classify
        the value (e.g. a non-scalar that slipped through).
    """
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(result, bool):
        return result
    return False


def sanitize_for_json(value: Any) -> Any:
    """Return *value* as a JSON-serializable Python object.

    Recurses through dicts, lists, tuples, sets, and frozensets; primitive
    types pass through unchanged. Temporal, decimal, numpy, pandas, and
    common stdlib types are normalized to strings, numbers, or ``None``
    as appropriate. A residual non-JSON leaf is coerced via ``str(...)``
    and its type name is logged once at ``DEBUG`` level so surprising
    leaks remain visible without spamming the hot path.

    The helper is idempotent: calling it a second time on its own output
    is a no-op. It performs no I/O and does not consult the environment.

    Args:
        value: Candidate value from node metadata (scalar or container).

    Returns:
        A JSON-serializable equivalent of *value*. Containers are
        rebuilt as plain ``dict`` / ``list`` instances.
    """
    if value is None:
        return None
    # ``bool`` subclasses ``int``; both pass through unchanged.
    if isinstance(value, (str, bool, int)) and not isinstance(value, enum.Enum):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    if isinstance(value, dict):
        return {str(k): sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, (set, frozenset)):
        try:
            items = sorted(value)
        except TypeError:
            items = list(value)
        return [sanitize_for_json(item) for item in items]

    if _is_pandas_missing(value):
        return None

    # ``datetime.datetime`` must precede ``datetime.date`` (subclass order);
    # ``pandas.Timestamp`` inherits from ``datetime.datetime`` and
    # ``pandas.Timedelta`` from ``datetime.timedelta``, so both are handled
    # by the stdlib branches below and produce ISO-8601 strings.
    if isinstance(value, datetime.datetime):
        return value.isoformat()
    if isinstance(value, datetime.date):
        return value.isoformat()
    if isinstance(value, datetime.time):
        return value.isoformat()
    if isinstance(value, datetime.timedelta):
        return str(value)

    if isinstance(value, pd.Period):
        return str(value)

    if isinstance(value, np.ndarray):
        return [sanitize_for_json(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return sanitize_for_json(value.item())

    if isinstance(value, decimal.Decimal):
        return str(value)
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, enum.Enum):
        return sanitize_for_json(value.value)
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, bytearray):
        return bytes(value).decode("utf-8", "replace")

    # check-then-add is intentionally racy under concurrent first-contact on
    # the same novel type — the worst outcome is a duplicate log line, never
    # a crash or an incorrect sanitation result.
    type_name = type(value).__name__
    if type_name not in _LOGGED_UNKNOWN_TYPES:
        _LOGGED_UNKNOWN_TYPES.add(type_name)
        logger.debug("sanitize_for_json: coerced unknown type {} via str()", type_name)
    return str(value)
