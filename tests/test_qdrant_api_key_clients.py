"""The Qdrant clients docint builds carry ``QDRANT_API_KEY`` when it is set.

Both the sync and async client in ``RAG`` and the self-built client in
``ImageIngestionService`` must pass ``api_key`` through, and must pass
``None`` (not ``""``) when the variable is unset, so a data-plane without
server-side auth keeps working unchanged.
"""

from __future__ import annotations

from typing import Any, ClassVar, cast

import pytest

from docint.core import rag as rag_module
from docint.core.rag import RAG


class _Capture:
    """Stand-in for a Qdrant client class that records constructor kwargs."""

    calls: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, **kwargs: Any) -> None:
        type(self).calls.append(kwargs)


@pytest.fixture(autouse=True)
def _reset_capture() -> None:
    """Clear captured constructor calls between tests."""
    _Capture.calls = []


def test_sync_client_receives_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The sync client is built with the configured QDRANT_API_KEY."""
    monkeypatch.setenv("QDRANT_HOST", "http://qdrant:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "change-me-qdrant-key")
    monkeypatch.setattr(rag_module, "QdrantClient", _Capture)
    rag = RAG(qdrant_collection="test")

    _ = rag.qdrant_client

    assert _Capture.calls == [{"url": "http://qdrant:6333", "api_key": "change-me-qdrant-key"}]


def test_async_client_receives_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The async client is built with the configured QDRANT_API_KEY."""
    monkeypatch.setenv("QDRANT_HOST", "http://qdrant:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "change-me-qdrant-key")
    monkeypatch.setattr(rag_module, "AsyncQdrantClient", _Capture)
    rag = RAG(qdrant_collection="test")

    _ = rag.qdrant_aclient

    assert _Capture.calls == [{"url": "http://qdrant:6333", "api_key": "change-me-qdrant-key"}]


def test_clients_pass_none_when_key_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both clients pass api_key=None (not "") when QDRANT_API_KEY is unset."""
    monkeypatch.setenv("QDRANT_HOST", "http://qdrant:6333")
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    monkeypatch.setattr(rag_module, "QdrantClient", _Capture)
    monkeypatch.setattr(rag_module, "AsyncQdrantClient", _Capture)
    rag = RAG(qdrant_collection="test")

    _ = rag.qdrant_client
    _ = rag.qdrant_aclient

    assert [c["api_key"] for c in _Capture.calls] == [None, None]


def test_image_service_self_built_client_receives_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The self-built client in ImageIngestionService receives the configured QDRANT_API_KEY."""
    from docint.core.ingest import images_service as images_module
    from docint.core.ingest.images_service import ImageIngestionService

    monkeypatch.setenv("QDRANT_HOST", "http://qdrant:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "change-me-qdrant-key")
    monkeypatch.setattr(images_module, "QdrantClient", _Capture)

    ImageIngestionService()

    assert _Capture.calls == [{"url": "http://qdrant:6333", "api_key": "change-me-qdrant-key"}]


def test_image_service_self_built_client_passes_none_when_key_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """The self-built client in ImageIngestionService passes api_key=None when QDRANT_API_KEY is unset."""
    from docint.core.ingest import images_service as images_module
    from docint.core.ingest.images_service import ImageIngestionService

    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    monkeypatch.setenv("QDRANT_HOST", "http://qdrant:6333")
    monkeypatch.setattr(images_module, "QdrantClient", _Capture)

    ImageIngestionService()

    assert _Capture.calls == [{"url": "http://qdrant:6333", "api_key": None}]


def test_image_service_injected_client_is_not_replaced(monkeypatch: pytest.MonkeyPatch) -> None:
    """An injected qdrant_client is kept as-is and no client is constructed."""
    from docint.core.ingest import images_service as images_module
    from docint.core.ingest.images_service import ImageIngestionService

    monkeypatch.setattr(images_module, "QdrantClient", _Capture)
    sentinel = object()

    service = ImageIngestionService(qdrant_client=cast(Any, sentinel))

    assert service.qdrant_client is sentinel
    assert _Capture.calls == []


def test_key_absent_from_rag_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    """QDRANT_API_KEY never appears in RAG's repr or its host_config's repr."""
    monkeypatch.setenv("QDRANT_HOST", "http://qdrant:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "change-me-qdrant-key")
    rag = RAG(qdrant_collection="test")

    assert "change-me-qdrant-key" not in repr(rag)
    assert "change-me-qdrant-key" not in repr(rag.host_config)
