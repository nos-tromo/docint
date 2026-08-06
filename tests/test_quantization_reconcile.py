"""Tests for the best-effort startup quantization reconcile.

``RAG.reconcile_quantization()`` upgrades pre-existing collections to the
configured TurboQuant setting via ``update_collection``. It is add-only
(a no-op when ``QDRANT_QUANTIZATION=none``), skips vector-less collections,
never overwrites a different quantization family, and swallows every
failure — startup must not block on it.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from qdrant_client.http import models as qdrant_models

from docint.core.rag import RAG


def _collection_info(
    *,
    vectors: Any,
    quantization: Any = None,
) -> SimpleNamespace:
    """Build a fake ``get_collection`` result with the fields reconcile reads."""
    return SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(vectors=vectors),
            quantization_config=quantization,
        )
    )


class _FakeQdrant:
    """Fake client driving ``reconcile_quantization`` through its branches."""

    def __init__(self, infos: dict[str, Any]) -> None:
        self.infos = infos
        self.update_calls: list[dict[str, Any]] = []

    def get_collections(self) -> SimpleNamespace:
        return SimpleNamespace(collections=[SimpleNamespace(name=name) for name in self.infos])

    def get_collection(self, name: str) -> Any:
        info = self.infos[name]
        if isinstance(info, Exception):
            raise info
        return info

    def update_collection(self, *, collection_name: str, quantization_config: Any) -> None:
        self.update_calls.append(
            {"collection_name": collection_name, "quantization_config": quantization_config}
        )


def _make_rag(client: Any) -> RAG:
    rag = RAG(qdrant_collection="test")
    rag._qdrant_client = cast(Any, client)
    return rag


def _dense_params() -> dict[str, Any]:
    return {"text-dense": qdrant_models.VectorParams(size=4, distance=qdrant_models.Distance.COSINE)}


def _turbo(bits: str = "bits4", always_ram: bool | None = None) -> qdrant_models.TurboQuantization:
    return qdrant_models.TurboQuantization(
        turbo=qdrant_models.TurboQuantQuantizationConfig(
            bits=qdrant_models.TurboQuantBitSize(bits), always_ram=always_ram
        )
    )


def test_reconcile_updates_unquantized_collection() -> None:
    """A dense collection without quantization gets a TurboQuant update."""
    client = _FakeQdrant({"docs": _collection_info(vectors=_dense_params())})
    assert _make_rag(client).reconcile_quantization() == 1
    call = client.update_calls[0]
    assert call["collection_name"] == "docs"
    assert isinstance(call["quantization_config"], qdrant_models.TurboQuantization)


def test_reconcile_skips_matching_collection() -> None:
    """A collection already at the target config is left alone."""
    client = _FakeQdrant({"docs": _collection_info(vectors=_dense_params(), quantization=_turbo())})
    assert _make_rag(client).reconcile_quantization() == 0
    assert client.update_calls == []


def test_reconcile_treats_false_and_none_always_ram_as_equal() -> None:
    """A server reporting ``always_ram=False`` matches a ``None`` target."""
    client = _FakeQdrant(
        {"docs": _collection_info(vectors=_dense_params(), quantization=_turbo(always_ram=False))}
    )
    assert _make_rag(client).reconcile_quantization() == 0
    assert client.update_calls == []


def test_reconcile_noop_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """``QDRANT_QUANTIZATION=none`` never strips quantization (add-only)."""
    monkeypatch.setenv("QDRANT_QUANTIZATION", "none")
    client = _FakeQdrant({"docs": _collection_info(vectors=_dense_params())})
    assert _make_rag(client).reconcile_quantization() == 0
    assert client.update_calls == []


def test_reconcile_skips_vectorless_collection() -> None:
    """Collections without dense vector params are skipped."""
    client = _FakeQdrant(
        {
            "docs_dockv": _collection_info(vectors=None),
            "docs": _collection_info(vectors=_dense_params()),
        }
    )
    assert _make_rag(client).reconcile_quantization() == 1
    assert [c["collection_name"] for c in client.update_calls] == ["docs"]


def test_reconcile_preserves_other_quantization_family() -> None:
    """A deliberately configured non-TurboQuant family is not overwritten."""
    scalar = qdrant_models.ScalarQuantization(
        scalar=qdrant_models.ScalarQuantizationConfig(type=qdrant_models.ScalarType.INT8)
    )
    client = _FakeQdrant({"docs": _collection_info(vectors=_dense_params(), quantization=scalar)})
    assert _make_rag(client).reconcile_quantization() == 0
    assert client.update_calls == []


def test_reconcile_continues_past_per_collection_failure() -> None:
    """One broken collection does not stop the others from reconciling."""
    client = _FakeQdrant(
        {
            "broken": RuntimeError("boom"),
            "docs": _collection_info(vectors=_dense_params()),
        }
    )
    assert _make_rag(client).reconcile_quantization() == 1
    assert [c["collection_name"] for c in client.update_calls] == ["docs"]


def test_reconcile_survives_listing_failure() -> None:
    """An unreachable Qdrant degrades to a logged no-op, never an exception."""

    class _DeadQdrant:
        def get_collections(self) -> Any:
            raise ConnectionError("qdrant down")

    assert _make_rag(_DeadQdrant()).reconcile_quantization() == 0
