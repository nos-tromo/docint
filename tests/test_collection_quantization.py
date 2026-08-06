"""Creation-site tests: every Qdrant collection is created with TurboQuant.

Pins that all three dense-vector creation sites — the main collection
(``RAG.create_collection_if_missing``), the ``_images`` CLIP companion
(``ImageIngestionService._ensure_collection``), and the ``_entities``
companion (``EntityStore.ensure_collection``) — pass the shared
``build_quantization_config()`` payload, and that ``QDRANT_QUANTIZATION=none``
turns it off everywhere at once.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from qdrant_client.http import models as qdrant_models

from docint.core.entities.store import EntityStore
from docint.core.ingest.images_service import ImageIngestionService
from docint.core.rag import RAG
from docint.utils.env_cfg import ImageIngestionConfig


class _CaptureQdrant:
    """Fake Qdrant client capturing ``create_collection`` keyword arguments."""

    def __init__(self) -> None:
        self.create_calls: list[dict[str, Any]] = []

    def collection_exists(self, collection_name: str) -> bool:
        return False

    def get_collection(self, collection_name: str) -> Any:
        raise RuntimeError("missing")

    def create_collection(self, **kwargs: Any) -> None:
        self.create_calls.append(kwargs)

    def create_payload_index(self, **kwargs: Any) -> None:
        return None


def _quantization_of(call: dict[str, Any]) -> Any:
    assert "quantization_config" in call, "create_collection must pass quantization_config"
    return call["quantization_config"]


def _make_rag(client: _CaptureQdrant) -> RAG:
    rag = RAG(qdrant_collection="test")
    rag._qdrant_client = cast(Any, client)
    rag.openai_dimensions = 8  # skip the embed probe
    return rag


def test_main_collection_dense_only_gets_turboquant() -> None:
    """The dense-only create passes the TurboQuant payload."""
    client = _CaptureQdrant()
    rag = _make_rag(client)
    rag.enable_hybrid = False
    rag.create_collection_if_missing()
    assert isinstance(_quantization_of(client.create_calls[0]), qdrant_models.TurboQuantization)


def test_main_collection_hybrid_gets_turboquant() -> None:
    """The hybrid create passes the TurboQuant payload too."""
    client = _CaptureQdrant()
    rag = _make_rag(client)
    rag.enable_hybrid = True
    rag.create_collection_if_missing()
    call = client.create_calls[0]
    assert "sparse_vectors_config" in call
    assert isinstance(_quantization_of(call), qdrant_models.TurboQuantization)


def test_main_collection_respects_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """``QDRANT_QUANTIZATION=none`` creates the collection unquantized."""
    monkeypatch.setenv("QDRANT_QUANTIZATION", "none")
    client = _CaptureQdrant()
    rag = _make_rag(client)
    rag.enable_hybrid = False
    rag.create_collection_if_missing()
    assert _quantization_of(client.create_calls[0]) is None


def test_entity_store_gets_turboquant() -> None:
    """The ``_entities`` companion create passes the TurboQuant payload."""
    client = _CaptureQdrant()
    store = EntityStore(cast(Any, client), collection="docs_entities", dim=2, embed_model="test-embed")
    store.ensure_collection()
    assert isinstance(_quantization_of(client.create_calls[0]), qdrant_models.TurboQuantization)


def test_images_companion_gets_turboquant() -> None:
    """The ``_images`` companion create passes the TurboQuant payload."""
    cfg = ImageIngestionConfig(
        enabled=True,
        embedding_enabled=True,
        tagging_enabled=True,
        collection_name="test-images",
        vector_name="image-dense",
        cache_by_hash=True,
        fail_on_embedding_error=False,
        fail_on_tagging_error=False,
        retrieve_top_k=5,
    )
    client = _CaptureQdrant()
    service = ImageIngestionService(
        img_ingestion_config=cfg,
        model_config=cast(Any, SimpleNamespace(image_embed_model="openai/clip-vit-base-patch32")),
        qdrant_client=cast(Any, client),
        vector_store=cast(Any, None),
        embedding_backend=cast(Any, None),
        tagging_backend=cast(Any, None),
    )
    service._ensure_collection(collection_name="test-images", vector_dim=4)
    assert isinstance(_quantization_of(client.create_calls[0]), qdrant_models.TurboQuantization)
