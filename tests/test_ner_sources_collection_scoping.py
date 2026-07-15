"""Regression tests: NER sources are cached per collection, not process-globally.

``RAG.get_collection_ner`` used to memoize the Qdrant scroll into a single
un-keyed ``ner_sources`` list on the process-wide RAG singleton. Whichever
collection populated it first was served to every later request — across
collections *and* tenants — and deleting that collection (which runs without a
``collection_scope`` bound) never cleared it, so its chunks kept surfacing on
the analysis page. The memo is now ``_ner_sources_cache``, keyed by physical
collection like ``_documents_cache``, and ``_invalidate_ner_cache`` pops the
key unconditionally.
"""

from __future__ import annotations

from typing import Any

import pytest

from docint.core.rag import RAG


def _patch_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the Qdrant scroll return rows tagged with the active collection."""

    def _fake_load(self: RAG, **_kwargs: Any) -> list[dict[str, Any]]:
        return [{"collection": self.qdrant_collection, "file_name": f"{self.qdrant_collection}.mp4"}]

    monkeypatch.setattr(RAG, "_load_collection_ner_sources", _fake_load)


def test_ner_sources_isolated_per_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each collection scope sees only its own NER sources."""
    _patch_loader(monkeypatch)
    rag = RAG(qdrant_collection="")

    with rag.collection_scope("collA"):
        first = rag.get_collection_ner()
    assert first[0]["collection"] == "collA"

    with rag.collection_scope("collB"):
        second = rag.get_collection_ner()
    assert second[0]["collection"] == "collB", (
        f"collection 'collB' was served NER sources from {second[0]['collection']!r}"
    )

    # Repeat reads still hit the right per-collection entries.
    with rag.collection_scope("collA"):
        assert rag.get_collection_ner()[0]["collection"] == "collA"


def test_ner_sources_cleared_by_unscoped_delete(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalidating by name with no request scope bound evicts the entry.

    ``delete_collection`` calls ``_invalidate_ner_cache(physical)`` outside any
    ``collection_scope``; the deleted collection's rows must not stay servable.
    """
    _patch_loader(monkeypatch)
    rag = RAG(qdrant_collection="")

    with rag.collection_scope("collA"):
        rag.get_collection_ner()

    rag._invalidate_ner_cache("collA")
    assert "collA" not in rag._ner_sources_cache

    with rag.collection_scope("collB"):
        after_delete = rag.get_collection_ner()
    assert all(row["collection"] == "collB" for row in after_delete), (
        f"deleted collection's NER sources still served: {after_delete}"
    )


def test_ner_sources_refresh_reloads_only_active_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    """``refresh=True`` re-fetches the scoped collection without touching others."""
    _patch_loader(monkeypatch)
    rag = RAG(qdrant_collection="")

    with rag.collection_scope("collA"):
        rag.get_collection_ner()
    with rag.collection_scope("collB"):
        rag.get_collection_ner()

    with rag.collection_scope("collB"):
        refreshed = rag.get_collection_ner(refresh=True)

    assert refreshed[0]["collection"] == "collB"
    assert rag._ner_sources_cache["collA"][0]["collection"] == "collA"
