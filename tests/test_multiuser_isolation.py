"""WS5: comprehensive multi-user isolation across the analysis path.

Complements the other multi-tenant guards:
- ``test_collection_owner_manager.py`` — ownership store unit tests.
- ``test_api_collections_ownership.py`` — ownership/visibility on list/ingest/
  select/delete/preview.
- ``test_rag_stateless_concurrency.py`` — per-request ``/query`` isolation.
- ``test_session_concurrency.py`` — per-request session ownership.

This module adds the **analysis** endpoints (``/collections/documents``), proving
they are owner-gated and isolated per request the same way the chat path is —
end-to-end through the API against a real RAG + real ownership store. Reads are
concurrent (no SQLite write contention, so no lock flakiness); the ownership
rows are registered sequentially in setup.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from docint.core.rag import RAG


def _make_rag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> RAG:
    """Build a real RAG on an in-process SQLite store with two owners of 'docs'.

    ``list_collections`` is stubbed empty so the ownership backfill never touches
    Qdrant. Alice and Bob each register the same logical name ``docs`` (distinct
    physical collections).

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture.
        tmp_path (Path): Per-test temp dir for the SQLite store.

    Returns:
        RAG: The configured RAG instance.
    """
    monkeypatch.setattr(RAG, "list_collections", lambda self: [])
    rag = RAG(qdrant_collection="")
    rag.init_session_store(f"sqlite:///{tmp_path / 'state.db'}")
    com = rag.ensure_collection_owner_manager()
    com.register("alice", "docs")
    com.register("bob", "docs")
    return rag


def _wire(monkeypatch: pytest.MonkeyPatch, rag: RAG) -> Any:
    """Install ``rag`` as the API singleton and set a default identity.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture.
        rag (RAG): The RAG instance to install.

    Returns:
        Any: The ``docint.core.api`` module.
    """
    import docint.core.api as api_module

    monkeypatch.setattr(api_module, "rag", rag)
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "operator")
    return api_module


def test_documents_endpoint_is_owner_gated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The analysis path resolves the owner's logical name and 404s non-owners.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture.
        tmp_path (Path): Per-test temp dir for the SQLite store.
    """
    rag = _make_rag(monkeypatch, tmp_path)
    monkeypatch.setattr(RAG, "list_documents", lambda self: [{"collection": self.qdrant_collection}])
    api_module = _wire(monkeypatch, rag)
    client = TestClient(api_module.app)

    physical_alice = rag.ensure_collection_owner_manager().resolve("alice", "docs")
    ok = client.get("/collections/documents", params={"collection": "docs"}, headers={"X-Auth-User": "alice"})
    assert ok.status_code == 200, ok.text
    assert ok.json()["documents"] == [{"collection": physical_alice}]  # own physical, not the logical name

    # A principal who owns nothing cannot read another user's collection.
    denied = client.get("/collections/documents", params={"collection": "docs"}, headers={"X-Auth-User": "carol"})
    assert denied.status_code == 404


def test_concurrent_documents_isolated_per_owner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Two concurrent analysis requests for the same logical name see only their own.

    A barrier inside ``list_documents`` forces both requests to capture their
    scoped collection before either returns, so a shared/global active collection
    would make both echo the last writer and fail. With the per-request scope each
    sees only its owner's physical collection.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture.
        tmp_path (Path): Per-test temp dir for the SQLite store.
    """
    rag = _make_rag(monkeypatch, tmp_path)
    capture_barrier = threading.Barrier(2)

    def _docs(self: RAG) -> list[dict[str, Any]]:
        captured = self.qdrant_collection
        capture_barrier.wait(timeout=10)
        return [{"collection": captured}]

    monkeypatch.setattr(RAG, "list_documents", _docs)
    api_module = _wire(monkeypatch, rag)
    client = TestClient(api_module.app)

    com = rag.ensure_collection_owner_manager()
    physical_alice = com.resolve("alice", "docs")
    physical_bob = com.resolve("bob", "docs")
    assert physical_alice and physical_bob and physical_alice != physical_bob

    results: dict[str, Any] = {}
    errors: dict[str, BaseException] = {}

    def call(user: str) -> None:
        try:
            results[user] = client.get(
                "/collections/documents", params={"collection": "docs"}, headers={"X-Auth-User": user}
            )
        except BaseException as exc:
            errors[user] = exc

    threads = [threading.Thread(target=call, args=(u,)) for u in ("alice", "bob")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert results["alice"].json()["documents"] == [{"collection": physical_alice}]
    assert results["bob"].json()["documents"] == [{"collection": physical_bob}]


def test_admin_capability_boundary_on_analysis_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Pins the admin cross-owner boundary on the analysis path too (WS6).

    Complements ``test_api_collections_ownership.py``'s admin-select/-delete/
    -ingest coverage with the same guarantee on ``/collections/documents``:
    - a non-admin passing ``?owner=`` still resolves in their own namespace
      (404 here, since carol owns nothing named 'docs') — the query param is
      not a privilege escalation by itself;
    - an admin with no ``?owner=`` is scoped to their own (empty) namespace,
      not silently granted every owner's data.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture.
        tmp_path (Path): Per-test temp dir for the SQLite store.
    """
    rag = _make_rag(monkeypatch, tmp_path)
    monkeypatch.setattr(RAG, "list_documents", lambda self: [{"collection": self.qdrant_collection}])
    api_module = _wire(monkeypatch, rag)
    client = TestClient(api_module.app)

    # Non-admin: the owner param is inert without admin rights.
    denied = client.get(
        "/collections/documents",
        params={"collection": "docs", "owner": "alice"},
        headers={"X-Auth-User": "carol"},
    )
    assert denied.status_code == 404

    # Admin, no owner param: scoped to their own (unregistered) namespace.
    admin_own = client.get(
        "/collections/documents",
        params={"collection": "docs"},
        headers={"X-Auth-User": "root", "X-Auth-Groups": "admins"},
    )
    assert admin_own.status_code == 404

    # Admin with an explicit owner param: resolves alice's physical collection.
    physical_alice = rag.ensure_collection_owner_manager().resolve("alice", "docs")
    admin_cross = client.get(
        "/collections/documents",
        params={"collection": "docs", "owner": "alice"},
        headers={"X-Auth-User": "root", "X-Auth-Groups": "admins"},
    )
    assert admin_cross.status_code == 200, admin_cross.text
    assert admin_cross.json()["documents"] == [{"collection": physical_alice}]
