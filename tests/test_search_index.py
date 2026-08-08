"""Unit tests for the full-text search payload field and its index."""

from __future__ import annotations

from typing import Any

from qdrant_client import models

from docint.core.search.index import SEARCH_TEXT_FIELD, ensure_search_index, search_index_params


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
