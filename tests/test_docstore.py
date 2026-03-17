"""Tests for Qdrant-backed key-value docstore retry behavior."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from docint.core.storage.docstore import QdrantKVStore


@pytest.fixture
def client() -> MagicMock:
    """Return a mock Qdrant client with existing collection."""
    mock = MagicMock()
    mock.collection_exists.return_value = True
    return mock


def test_get_retries_on_connection_reset(client: MagicMock, monkeypatch) -> None:
    """Retry transient connection reset errors during get()."""
    sleep_calls: list[float] = []
    monkeypatch.setattr(
        "docint.core.storage.docstore.time.sleep",
        lambda delay: sleep_calls.append(delay),
    )

    payload_point = SimpleNamespace(payload={"val": {"ok": True}})
    client.retrieve.side_effect = [
        ConnectionResetError(104, "Connection reset by peer"),
        [payload_point],
    ]

    store = QdrantKVStore(
        client=client,
        collection_name="test_dockv",
        max_retries=2,
        retry_backoff_seconds=0.1,
        retry_backoff_max_seconds=0.2,
    )

    result = store.get("k1")

    assert result == {"ok": True}
    assert client.retrieve.call_count == 2
    assert sleep_calls == [0.1]


def test_put_all_retries_batch_upsert(client: MagicMock, monkeypatch) -> None:
    """Retry transient upsert errors for a single batch."""
    monkeypatch.setattr("docint.core.storage.docstore.time.sleep", lambda _: None)
    client.upsert.side_effect = [
        ConnectionResetError(104, "Connection reset by peer"),
        None,
    ]

    store = QdrantKVStore(
        client=client,
        collection_name="test_dockv",
        max_retries=1,
        retry_backoff_seconds=0.0,
    )
    store.put_all([("k1", {"v": 1})], batch_size=10)

    assert client.upsert.call_count == 2


def test_get_non_retryable_error_raises_immediately(client: MagicMock) -> None:
    """Propagate non-retryable exceptions without retrying."""
    client.retrieve.side_effect = ValueError("schema mismatch")

    store = QdrantKVStore(
        client=client,
        collection_name="test_dockv",
        max_retries=3,
    )

    with pytest.raises(ValueError, match="schema mismatch"):
        store.get("k1")

    assert client.retrieve.call_count == 1


def test_get_raises_after_retry_exhaustion(client: MagicMock, monkeypatch) -> None:
    """Raise the last transient exception once retries are exhausted."""
    monkeypatch.setattr("docint.core.storage.docstore.time.sleep", lambda _: None)
    client.retrieve.side_effect = ConnectionResetError(104, "Connection reset by peer")

    store = QdrantKVStore(
        client=client,
        collection_name="test_dockv",
        max_retries=2,
        retry_backoff_seconds=0.0,
    )

    with pytest.raises(ConnectionResetError):
        store.get("k1")

    assert client.retrieve.call_count == 3
