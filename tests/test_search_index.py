"""Unit tests for the full-text search payload field and its index."""

from __future__ import annotations

from typing import Any

from qdrant_client import models

from docint.core.search.index import (
    SEARCH_TEXT_FIELD,
    ensure_search_index,
    search_index_params,
    write_search_text,
)


class _FakeClient:
    """Records payload-index calls without talking to a server."""

    def __init__(self, *, fail: bool = False) -> None:
        """Initialize the fake.

        Args:
            fail (bool): When True, every call raises, to exercise fail-soft paths.
        """
        self.fail = fail
        self.index_calls: list[dict[str, Any]] = []

    def create_payload_index(self, **kwargs: Any) -> None:
        """Record an index creation, or raise when configured to fail."""
        if self.fail:
            raise RuntimeError("qdrant unreachable")
        self.index_calls.append(kwargs)


def test_search_index_uses_a_lowercase_prefix_tokenizer() -> None:
    """Case-insensitivity on non-ASCII text depends on this exact index.

    Un-indexed MatchText only case-folds ASCII, so a German title-case token
    would not match its lowercase form without a lowercase text index.
    """
    params = search_index_params()

    assert params.tokenizer == models.TokenizerType.PREFIX
    assert params.lowercase is True
    assert params.min_token_len == 2


def test_ensure_search_index_creates_the_index_on_the_field() -> None:
    """The index must be created on the dedicated search field."""
    client = _FakeClient()

    assert ensure_search_index(client, "col") is True
    assert len(client.index_calls) == 1
    assert client.index_calls[0]["field_name"] == SEARCH_TEXT_FIELD
    assert client.index_calls[0]["collection_name"] == "col"


def test_ensure_search_index_is_fail_soft() -> None:
    """A Qdrant outage must not break the caller — search degrades, not crashes."""
    client = _FakeClient(fail=True)

    assert ensure_search_index(client, "col") is False


class _RecordingClient(_FakeClient):
    """Fake client that records batched payload updates."""

    def __init__(self, *, fail: bool = False) -> None:
        """Initialize the recorder."""
        super().__init__(fail=fail)
        self.batches: list[list[Any]] = []

    def batch_update_points(self, **kwargs: Any) -> list[Any]:
        """Record one batched update call."""
        if self.fail:
            raise RuntimeError("qdrant unreachable")
        self.batches.append(list(kwargs["update_operations"]))
        return []


def _written(client: _RecordingClient) -> dict[Any, str]:
    """Flatten recorded operations into ``{point_id: text}``.

    Args:
        client (_RecordingClient): The fake that recorded the calls.

    Returns:
        dict[Any, str]: Every point id written, mapped to its text.
    """
    out: dict[Any, str] = {}
    for batch in client.batches:
        for op in batch:
            payload = op.set_payload
            for point_id in payload.points or []:
                out[point_id] = payload.payload[SEARCH_TEXT_FIELD]
    return out


def test_write_search_text_writes_one_operation_per_point() -> None:
    """Distinct texts per point must survive into distinct operations."""
    client = _RecordingClient()

    assert write_search_text(client, "col", {"a": "first chunk", "b": "second chunk"}) == 2
    assert _written(client) == {"a": "first chunk", "b": "second chunk"}


def test_write_search_text_preserves_the_point_id_type() -> None:
    """Qdrant ids are unsigned ints or UUIDs; coercing an int id writes nothing.

    A str("1") targets a point that does not exist, so the write silently
    lands nowhere and the collection stays unsearchable.
    """
    client = _RecordingClient()

    assert write_search_text(client, "col", {7: "seventh chunk"}) == 1
    assert _written(client) == {7: "seventh chunk"}


def test_write_search_text_batches_by_batch_size() -> None:
    """A large collection must not become one enormous request."""
    client = _RecordingClient()
    texts = {str(i): f"chunk {i}" for i in range(7)}

    assert write_search_text(client, "col", texts, batch_size=3) == 7
    assert [len(batch) for batch in client.batches] == [3, 3, 1]


def test_write_search_text_ignores_empty_input() -> None:
    """No points means no request at all."""
    client = _RecordingClient()

    assert write_search_text(client, "col", {}) == 0
    assert client.batches == []
