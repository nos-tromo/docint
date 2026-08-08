"""Tests for the ``resolve`` CLI entry point."""

from __future__ import annotations

import types

import pytest

import docint.cli.resolve as cli
from docint.core.entities.resolution import ResolutionSummary


def test_resolve_cli_invokes_rag_and_unloads(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI constructs a RAG for the collection, resolves, then unloads.

    Args:
        monkeypatch: Fixture to swap the RAG class for a fake.
    """
    calls: dict[str, object] = {}

    class _FakeRAG:
        def __init__(self, *, qdrant_collection: str) -> None:
            calls["collection"] = qdrant_collection
            # The CLI resolves logical -> physical before running; a client
            # that reports the name already exists makes that step a no-op so
            # this test stays about the resolve call, not name resolution.
            self.qdrant_client = types.SimpleNamespace(collection_exists=lambda collection_name: True)

        def resolve_entities(self, *, progress_callback: object = None) -> ResolutionSummary:
            calls["resolved"] = True
            return ResolutionSummary(processed=3, minted=1, attached=1, skipped=1, entities_touched=2)

        def unload_models(self) -> None:
            calls["unloaded"] = True

    monkeypatch.setattr(cli, "RAG", _FakeRAG)

    cli.resolve_entities("mycol")

    assert calls == {"collection": "mycol", "resolved": True, "unloaded": True}


def test_resolve_cli_unloads_even_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Models are unloaded even when resolution raises.

    Args:
        monkeypatch: Fixture to swap the RAG class for a fake.
    """
    unloaded: dict[str, bool] = {}

    class _BoomRAG:
        def __init__(self, *, qdrant_collection: str) -> None:
            # The CLI resolves logical -> physical before running; a client
            # that reports the name already exists makes that step a no-op so
            # these tests stay about unloading, not name resolution.
            self.qdrant_client = types.SimpleNamespace(collection_exists=lambda collection_name: True)

        def resolve_entities(self, *, progress_callback: object = None) -> ResolutionSummary:
            raise RuntimeError("boom")

        def unload_models(self) -> None:
            unloaded["done"] = True

    monkeypatch.setattr(cli, "RAG", _BoomRAG)

    with pytest.raises(RuntimeError, match="boom"):
        cli.resolve_entities("mycol")

    assert unloaded == {"done": True}
