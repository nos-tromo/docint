"""Concurrency / ownership tests for the per-request chat runtime (WS3).

These pin the session-ownership invariant fixed in WS3: two principals driving
concurrent *session-mode* chat turns must each persist their turn under their
own ``(session_id, owner)`` and never cross-attribute.

Before WS3, :class:`docint.core.state.session_manager.SessionManager` kept the
active ``session_id`` / ``_owner`` (and ``chat_engine`` sentinel) in shared
instance fields that ``start_session`` mutated and ``chat`` read back -- a
TOCTOU race across the FastAPI threadpool. With both requests in flight the
shared fields hold the *last writer's* session id/owner, so an interleaved turn
lands in the wrong conversation. The headline test below drives that exact
interleaving and asserts correct persistence; it fails under the old
shared-field model and passes once the runtime is threaded per request.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest
from llama_index.core.base.response.schema import Response

from docint.core.rag import RAG


class _ChatModeStub:
    """Tiny enum-like stub exposing a ``.value`` for the response-mode field."""

    value = "default"


class _Owners:
    """Owner-manager stub mapping ``(owner, logical)`` to a per-owner physical name."""

    def resolve(self, owner: str | None, logical: str) -> str:
        """Return a deterministic per-owner physical collection name.

        Args:
            owner (str | None): The calling principal.
            logical (str): The caller's logical collection name.

        Returns:
            str: A physical name unique to ``(owner, logical)``.
        """
        return f"u_{owner}__{logical}"


def _install_chat_stubs(monkeypatch: pytest.MonkeyPatch, build_query_engine: Any) -> RAG:
    """Patch the inference-heavy RAG surface and return a fresh RAG instance.

    Everything that would touch a real model or Qdrant is stubbed; the real
    ``SessionManager`` (start_session / persistence / ownership scoping) runs
    against a real SQLite store so the test exercises the genuine write path.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture.
        build_query_engine (Any): The fake ``build_query_engine`` to install.

    Returns:
        RAG: A RAG instance with the chat surface stubbed.
    """
    monkeypatch.setattr(RAG, "build_query_engine", build_query_engine)
    monkeypatch.setattr(RAG, "create_index", lambda self: None)
    monkeypatch.setattr(RAG, "create_query_engine", lambda self: None)
    monkeypatch.setattr(
        RAG,
        "rewrite_retrieval_query",
        lambda self, *, user_msg, conversation_context="": user_msg,
    )
    monkeypatch.setattr(RAG, "expand_query_with_graph_with_debug", lambda self, q: (q, {}))
    monkeypatch.setattr(RAG, "_infer_collection_profile", lambda self: {"coverage_unit": "documents"})
    monkeypatch.setattr(RAG, "_resolve_chat_response_mode", lambda self: _ChatModeStub())
    monkeypatch.setattr(
        RAG,
        "_normalize_response_data",
        lambda self, user_msg, resp, **kw: {"response": getattr(resp, "response", ""), "sources": []},
    )
    monkeypatch.setattr(RAG, "ensure_collection_owner_manager", lambda self: _Owners())
    return RAG(qdrant_collection="")


def _wire_api(monkeypatch: pytest.MonkeyPatch, rag: RAG, tmp_path: Path) -> Any:
    """Point the API singleton at ``rag`` and force the chat build branch.

    ``build_metadata_filters`` is stubbed non-``None`` so ``chat`` always takes
    its ``build_query_engine`` branch (where the rendezvous barrier lives),
    independent of the request body.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture.
        rag (RAG): The RAG instance to install as the API singleton.
        tmp_path (Path): Per-test temp dir for the SQLite session store.

    Returns:
        Any: The ``docint.core.api`` module (already wired).
    """
    import docint.core.api as api_module

    rag.init_session_store(f"sqlite:///{tmp_path / 'sessions.db'}")
    monkeypatch.setattr(api_module, "rag", rag)
    monkeypatch.setattr(api_module, "_validation_payload", lambda **kwargs: {})
    monkeypatch.setattr(api_module, "build_metadata_filters", lambda rules: object())
    monkeypatch.setattr(api_module, "build_qdrant_filter", lambda rules: None)
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "operator")
    return api_module


def test_concurrent_session_chats_persist_under_correct_owner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Interleaved alice/bob session turns each persist under their own session.

    Each request enters ``chat`` and rendezvous on a barrier inside
    ``build_query_engine`` -- the first thing ``chat`` does -- so both requests
    have completed ``start_session`` before either reaches its turn. Under the
    old shared-field model the active session id/owner now hold the last
    writer's values and BOTH turns land in one conversation. With the runtime
    threaded per request each turn lands in its own.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture.
        tmp_path (Path): Per-test temp dir for the SQLite session store.
    """
    from fastapi.testclient import TestClient

    build_barrier = threading.Barrier(2)

    def _fake_build(self: RAG, **_kwargs: Any) -> Any:
        build_barrier.wait(timeout=10)

        class _Engine:
            def query(self, prompt: str) -> Response:
                return Response(response=f"answer::{prompt}", source_nodes=[])

        return _Engine()

    rag = _install_chat_stubs(monkeypatch, _fake_build)
    api_module = _wire_api(monkeypatch, rag, tmp_path)

    client = TestClient(api_module.app)
    results: dict[str, Any] = {}
    errors: dict[str, BaseException] = {}

    def call(user: str, session_id: str, question: str) -> None:
        try:
            results[user] = client.post(
                "/query",
                json={
                    "question": question,
                    "collection": "docs",
                    "session_id": session_id,
                    "retrieval_mode": "session",
                },
                headers={"X-Auth-User": user},
            )
        except BaseException as exc:
            errors[user] = exc

    threads = [
        threading.Thread(target=call, args=("alice", "sess-alice", "alice-question")),
        threading.Thread(target=call, args=("bob", "sess-bob", "bob-question")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert results["alice"].status_code == 200, results["alice"].text
    assert results["bob"].status_code == 200, results["bob"].text
    assert results["alice"].json()["session_id"] == "sess-alice"
    assert results["bob"].json()["session_id"] == "sess-bob"

    sm = rag.ensure_session_manager()
    alice_users = [m["content"] for m in sm.get_session_history("sess-alice", "alice") if m["role"] == "user"]
    bob_users = [m["content"] for m in sm.get_session_history("sess-bob", "bob") if m["role"] == "user"]

    # Each turn must persist under its own (session_id, owner), never cross-attributed.
    assert alice_users == ["alice-question"], alice_users
    assert bob_users == ["bob-question"], bob_users

    # Owner scoping: alice cannot read bob's session (treated as 404 / empty).
    assert sm.get_session_history("sess-bob", "alice") == []
    assert {s["id"] for s in sm.list_sessions("alice")} == {"sess-alice"}
    assert {s["id"] for s in sm.list_sessions("bob")} == {"sess-bob"}


def test_session_collection_pin_conflict_returns_409(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Reusing a session id against a different collection is rejected with 409.

    A conversation is pinned to the collection it was created under. A later
    session-mode request that resolves to a *different* physical collection for
    the same owned session id must be refused rather than silently appending a
    turn retrieved from the wrong corpus.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture.
        tmp_path (Path): Per-test temp dir for the SQLite session store.
    """
    from fastapi.testclient import TestClient

    def _fake_build(self: RAG, **_kwargs: Any) -> Any:
        class _Engine:
            def query(self, prompt: str) -> Response:
                return Response(response=f"answer::{prompt}", source_nodes=[])

        return _Engine()

    rag = _install_chat_stubs(monkeypatch, _fake_build)
    api_module = _wire_api(monkeypatch, rag, tmp_path)

    client = TestClient(api_module.app)

    first = client.post(
        "/query",
        json={"question": "q1", "collection": "docs", "session_id": "s1", "retrieval_mode": "session"},
        headers={"X-Auth-User": "operator"},
    )
    assert first.status_code == 200, first.text

    # Same owned session, different collection -> pinned mismatch -> 409.
    conflict = client.post(
        "/query",
        json={"question": "q2", "collection": "other", "session_id": "s1", "retrieval_mode": "session"},
        headers={"X-Auth-User": "operator"},
    )
    assert conflict.status_code == 409, conflict.text
