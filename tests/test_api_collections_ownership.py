"""API tests for per-user collection ownership & visibility (WS1).

Patches ``api_module.rag`` with a minimal dummy whose
``ensure_collection_owner_manager`` returns a *real* CollectionOwnerManager
backed by a shared in-memory SQLite DB, plus just enough Qdrant-touching
methods (``list_collections`` / ``select_collection`` / ``delete_collection``)
and a faked ingestion module, so the endpoints are exercised end-to-end.

Identity is carried by ``X-Auth-User`` (default header); requests with no header
fall back to ``DOCINT_DEFAULT_IDENTITY`` ("test-operator").
"""

from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from _pytest.logging import LogCaptureFixture
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import docint.core.api as api_module
from docint.core.retention import NOTICE_DAYS
from docint.core.state.base import Base
from docint.core.state.collection_owner_manager import CollectionOwnerManager
from docint.core.state.collection_ownership import CollectionOwnership


class _FakeIngest:
    """Fake ingestion module: records the physical collection and 'creates' it."""

    def __init__(self, dummy: "_OwnRAG") -> None:
        self.dummy = dummy
        self.calls: list[str] = []

    def ingest_docs(self, collection: str, *args: Any, **kwargs: Any) -> None:
        # The JSON /ingest path passes (collection, data_dir, hybrid=...); the
        # streaming /ingest/upload path passes (collection, data_dir, hybrid,
        # progress_callback) positionally. Accept both.
        self.calls.append(collection)
        self.dummy.existing.add(collection)


class _SpySessions:
    """Records the physical collections whose sessions were cascade-deleted.

    Also records the (owner, collection) pairs any session listing was scoped to.
    """

    def __init__(self) -> None:
        self.deleted_for: list[str] = []
        self.listed: list[tuple[str | None, str | None]] = []

    def delete_sessions_for_collection(self, collection: str) -> int:
        self.deleted_for.append(collection)
        return 0

    def list_sessions(self, owner: str | None, collection: str | None = None) -> list[dict[str, Any]]:
        self.listed.append((owner, collection))
        return [{"id": "s1", "owner": owner, "collection": collection}]

    def init_session_store_if_needed(self) -> None:
        """Satisfy the lifespan's eager store init without touching a DB."""
        return None


class _OwnRAG:
    """Minimal RAG stand-in for collection-ownership endpoint tests."""

    def __init__(self) -> None:
        engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(engine)
        self.session_store = "sqlite://"
        self._com = CollectionOwnerManager(rag=cast(Any, self))
        self._com._SessionMaker = sessionmaker(bind=engine)
        self.existing: set[str] = set()
        self.active: str = ""
        self.deleted: list[str] = []
        self._sessions = _SpySessions()
        self._backfilled = False

    def probe_rerank_endpoint(self) -> None:
        """Satisfy the lifespan rerank probe without touching the network."""
        return None

    def probe_qdrant(self) -> bool:
        """Satisfy the lifespan startup probe without touching the network."""
        return True

    def reconcile_quantization(self) -> int:
        """Satisfy the lifespan quantization reconcile without touching Qdrant."""
        return 0

    def list_collections(self) -> list[str]:
        return sorted(self.existing)

    def ensure_collection_owner_manager(self) -> CollectionOwnerManager:
        if not self._backfilled:
            self._com.backfill_legacy(self.list_collections(), "test-operator")
            self._backfilled = True
        return self._com

    def ensure_session_manager(self) -> _SpySessions:
        return self._sessions

    def select_collection(self, name: str) -> None:
        if name not in self.existing:
            raise ValueError(f"Collection '{name}' does not exist.")
        self.active = name

    def delete_collection(self, name: str) -> None:
        self.existing.discard(name)
        self.deleted.append(name)


@pytest.fixture(autouse=True)
def _patch_rag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _OwnRAG:
    """Patch the module-level RAG singleton + ingestion seam with test doubles."""
    monkeypatch.delenv("DOCINT_AUTH_HEADER", raising=False)
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "test-operator")
    dummy = _OwnRAG()
    monkeypatch.setattr(api_module, "rag", dummy)
    monkeypatch.setattr(api_module, "ingest_module", _FakeIngest(dummy))
    monkeypatch.setattr(api_module, "_resolve_data_dir", lambda: tmp_path)
    return dummy


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """A TestClient bound to the FastAPI app."""
    with TestClient(api_module.app) as test_client:
        yield test_client


def _ingest(client: TestClient, user: str, logical: str) -> Any:
    return client.post("/ingest", json={"collection": logical}, headers={"X-Auth-User": user})


def _list(client: TestClient, user: str) -> list[str]:
    resp = client.get("/collections/list", headers={"X-Auth-User": user})
    assert resp.status_code == 200, resp.text
    return cast(list[str], resp.json())


def test_ingest_registers_ownership_and_list_is_scoped(client: TestClient) -> None:
    """After alice ingests, only alice sees the collection."""
    assert _ingest(client, "alice", "alpha").status_code == 200
    assert _list(client, "alice") == ["alpha"]
    assert _list(client, "bob") == []


def test_same_logical_name_is_independent_per_user(client: TestClient, _patch_rag: _OwnRAG) -> None:
    """Alice and Bob can both ingest 'mydocs'; they map to distinct physical collections."""
    assert _ingest(client, "alice", "mydocs").status_code == 200
    assert _ingest(client, "bob", "mydocs").status_code == 200
    assert _list(client, "alice") == ["mydocs"]
    assert _list(client, "bob") == ["mydocs"]
    assert len(_patch_rag.existing) == 2  # two distinct physical collections


def test_select_is_owner_gated(client: TestClient, _patch_rag: _OwnRAG) -> None:
    """Select validates ownership (200 owned / 404 not) and never mutates state (WS2)."""
    _ingest(client, "alice", "alpha")
    assert (
        client.post("/collections/select", json={"name": "alpha"}, headers={"X-Auth-User": "alice"}).status_code == 200
    )
    # WS2: selection is a non-mutating ownership check — no server-side active
    # collection is set, so concurrent users cannot clobber each other.
    assert _patch_rag.active == ""
    resp = client.post("/collections/select", json={"name": "alpha"}, headers={"X-Auth-User": "bob"})
    assert resp.status_code == 404


def test_delete_is_owner_gated(client: TestClient, _patch_rag: _OwnRAG) -> None:
    """A non-owner cannot delete; the owner can, and it disappears from their list."""
    _ingest(client, "alice", "alpha")
    assert client.delete("/collections/alpha", headers={"X-Auth-User": "bob"}).status_code == 404
    assert _patch_rag.deleted == []
    assert client.delete("/collections/alpha", headers={"X-Auth-User": "alice"}).status_code == 200
    assert len(_patch_rag.deleted) == 1  # the physical name was deleted from Qdrant
    assert _list(client, "alice") == []


def test_delete_collection_cascades_sessions(client: TestClient, _patch_rag: _OwnRAG) -> None:
    """Deleting a collection cascade-deletes its chat sessions (by physical name)."""
    assert _ingest(client, "alice", "alpha").status_code == 200
    physical = _patch_rag.ensure_collection_owner_manager().resolve("alice", "alpha")
    assert physical is not None

    assert client.delete("/collections/alpha", headers={"X-Auth-User": "alice"}).status_code == 200
    assert _patch_rag._sessions.deleted_for == [physical]


def test_legacy_collections_backfilled_to_default_identity(_patch_rag: _OwnRAG) -> None:
    """A pre-existing collection is owned by the default identity, not by other users.

    Startup records the retention window through the ownership manager, so the
    backfill runs at boot: the collection must exist before the app starts.
    """
    _patch_rag.existing.add("legacy1")
    with TestClient(api_module.app) as client:
        # The default identity (no header) sees the backfilled legacy collection...
        assert "legacy1" in client.get("/collections/list").json()
        # ...but a different principal does not.
        assert _list(client, "alice") == []


def test_preview_source_is_owner_gated_and_uses_physical(
    client: TestClient, _patch_rag: _OwnRAG, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """/sources/preview owner-gates the logical name and resolves it to physical.

    A non-owner gets 404 (and no file lookup is attempted); the owner's request
    resolves the logical name to its owner-namespaced physical collection before
    touching the source store, so previews work for namespaced users.
    """
    _ingest(client, "alice", "docs")
    captured: dict[str, str] = {}
    src = tmp_path / "x.txt"
    src.write_text("hi", encoding="utf-8")

    def _fake_resolve(collection: str, file_hash: str, **_kw: Any) -> Path:
        captured["collection"] = collection
        return src

    monkeypatch.setattr(api_module, "_resolve_source_file_path", _fake_resolve)

    ok = client.get(
        "/sources/preview", params={"collection": "docs", "file_hash": "h"}, headers={"X-Auth-User": "alice"}
    )
    assert ok.status_code == 200, ok.text
    assert captured["collection"] != "docs"  # resolved to the owner-namespaced physical name

    captured.clear()
    denied = client.get(
        "/sources/preview", params={"collection": "docs", "file_hash": "h"}, headers={"X-Auth-User": "bob"}
    )
    assert denied.status_code == 404
    assert captured == {}  # gate rejected before any source lookup


def test_upload_registers_ownership(
    client: TestClient, _patch_rag: _OwnRAG, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Files uploaded via /ingest/upload register ownership, scoped to the uploader."""
    monkeypatch.setattr(api_module, "_resolve_qdrant_src_dir", lambda: tmp_path)
    resp = client.post(
        "/ingest/upload",
        data={"collection": "uploaded", "hybrid": "true"},
        files=[("files", ("a.txt", b"hello", "text/plain"))],
        headers={"X-Auth-User": "alice"},
    )
    assert resp.status_code == 200, resp.text
    assert _list(client, "alice") == ["uploaded"]
    assert _list(client, "bob") == []


ADMIN = {"X-Auth-User": "root", "X-Auth-Groups": "admins"}


def test_admin_accesses_cross_owner_with_owner_param(client: TestClient) -> None:
    """An admin with ?owner=<user> operates in that user's namespace."""
    _ingest(client, "alice", "alpha")

    resp = client.post("/collections/select?owner=alice", json={"name": "alpha"}, headers=ADMIN)
    assert resp.status_code == 200

    # Without the owner param the admin is in their own (empty) namespace.
    assert client.post("/collections/select", json={"name": "alpha"}, headers=ADMIN).status_code == 404


def test_non_admin_owner_param_still_404s(client: TestClient) -> None:
    """A non-admin passing ?owner= resolves in their own namespace: 404, not 403."""
    _ingest(client, "alice", "alpha")

    resp = client.post("/collections/select?owner=alice", json={"name": "alpha"}, headers={"X-Auth-User": "bob"})
    assert resp.status_code == 404


def test_admin_deletes_cross_owner_collection(client: TestClient) -> None:
    """Admin delete with ?owner= removes the user's mapping (full owner powers)."""
    _ingest(client, "alice", "alpha")

    assert client.delete("/collections/alpha?owner=alice", headers=ADMIN).status_code == 200
    assert client.get("/collections/list", headers={"X-Auth-User": "alice"}).json() == []


def test_admin_ingests_into_cross_owner_collection(client: TestClient) -> None:
    """Admin ingest with ?owner= registers under that owner, not the admin."""
    _ingest(client, "alice", "alpha")
    resp = client.post("/ingest?owner=alice", json={"collection": "alpha"}, headers=ADMIN)
    assert resp.status_code == 200

    # Still exactly alice's collection — not duplicated into the admin's namespace.
    assert client.get("/collections/list", headers={"X-Auth-User": "alice"}).json() == ["alpha"]
    assert client.get("/collections/list", headers=ADMIN).json() == []


def test_admin_lists_cross_owner_sessions_with_owner_param(client: TestClient, _patch_rag: _OwnRAG) -> None:
    """Admin's /sessions/list?owner=alice lists *alice's* sessions in alice's collection.

    Both the collection lookup and the session-owner scope use the effective
    owner (alice): an admin browsing a foreign namespace must see that owner's
    chats there, not their own empty list.
    """
    _ingest(client, "alice", "alpha")

    resp = client.get("/sessions/list", params={"collection": "alpha", "owner": "alice"}, headers=ADMIN)
    assert resp.status_code == 200
    # A resolved collection reaches the session manager (non-empty fake payload);
    # the old bug's fallback for an unresolvable collection was a hard {"sessions": []}.
    assert resp.json()["sessions"] != []

    owner_arg, collection_arg = _patch_rag._sessions.listed[-1]
    assert owner_arg == "alice"  # session scope follows effective_owner, not the admin's name
    assert collection_arg is not None  # and the collection resolved under alice's namespace


def test_non_admin_owner_param_does_not_rescope_sessions(client: TestClient, _patch_rag: _OwnRAG) -> None:
    """A non-admin passing ?owner= keeps their own session scope."""
    resp = client.get("/sessions/list", params={"owner": "alice"}, headers={"X-Auth-User": "bob"})
    assert resp.status_code == 200

    owner_arg, _ = _patch_rag._sessions.listed[-1]
    assert owner_arg == "bob"


def test_collections_list_all_admin_shape(client: TestClient) -> None:
    """?all=true for an admin returns mine + per-owner groups, mine excluded from others."""
    _ingest(client, "alice", "alpha")
    _ingest(client, "bob", "beta")
    _ingest(client, "root", "own")

    body = client.get("/collections/list?all=true", headers=ADMIN).json()

    assert body == {
        "mine": ["own"],
        "others": [
            {"owner": "alice", "collections": ["alpha"]},
            {"owner": "bob", "collections": ["beta"]},
        ],
    }


def test_collections_list_all_ignored_for_non_admin(client: TestClient) -> None:
    """?all=true for a non-admin is silently ignored: plain string[] as always."""
    _ingest(client, "alice", "alpha")
    _ingest(client, "bob", "beta")

    assert client.get("/collections/list?all=true", headers={"X-Auth-User": "alice"}).json() == ["alpha"]


def test_collections_list_no_params_unchanged_for_admin(client: TestClient) -> None:
    """Without ?all the admin gets the plain owner-scoped string[] like anyone else."""
    _ingest(client, "root", "own")

    assert client.get("/collections/list", headers=ADMIN).json() == ["own"]


# --- Retention clock and expiry listing (docs/retention.md) ---

LONG_AGO = datetime(2020, 1, 1, tzinfo=UTC)


def _age(rag: _OwnRAG, owner: str, logical: str, stamp: datetime | None = LONG_AGO) -> None:
    """Backdate a collection's last activity and forget the in-memory throttle."""
    mgr = rag.ensure_collection_owner_manager()
    with mgr._session_scope() as s:
        s.query(CollectionOwnership).filter(
            CollectionOwnership.owner == owner, CollectionOwnership.logical_name == logical
        ).update({CollectionOwnership.last_activity_at: stamp})
        s.commit()
    mgr._touch_due.clear()


def _last_activity(rag: _OwnRAG, owner: str, logical: str) -> datetime | None:
    """The recorded last activity of one collection."""
    rows = rag.ensure_collection_owner_manager().list_activity(owner)
    return next(row.last_activity_at for row in rows if row.logical == logical)


def test_an_owner_gated_request_moves_the_clock(client: TestClient, _patch_rag: _OwnRAG) -> None:
    """Every collection-scoped endpoint passes the ownership gate, so every one counts."""
    _ingest(client, "alice", "alpha")
    _age(_patch_rag, "alice", "alpha")

    assert (
        client.post("/collections/select", json={"name": "alpha"}, headers={"X-Auth-User": "alice"}).status_code == 200
    )

    stamp = _last_activity(_patch_rag, "alice", "alpha")
    assert stamp is not None and stamp > LONG_AGO


def test_reingesting_moves_the_clock(client: TestClient, _patch_rag: _OwnRAG) -> None:
    """Registering an existing collection does not write, so the register path stamps it."""
    _ingest(client, "alice", "alpha")
    _age(_patch_rag, "alice", "alpha")

    assert _ingest(client, "alice", "alpha").status_code == 200

    stamp = _last_activity(_patch_rag, "alice", "alpha")
    assert stamp is not None and stamp > LONG_AGO


def test_admin_work_in_another_namespace_moves_that_clock(client: TestClient, _patch_rag: _OwnRAG) -> None:
    """An admin working on a user's collection is activity on it."""
    _ingest(client, "alice", "alpha")
    _age(_patch_rag, "alice", "alpha")

    assert client.post("/collections/select?owner=alice", json={"name": "alpha"}, headers=ADMIN).status_code == 200

    stamp = _last_activity(_patch_rag, "alice", "alpha")
    assert stamp is not None and stamp > LONG_AGO


def test_listing_sessions_does_not_move_the_clock(client: TestClient, _patch_rag: _OwnRAG) -> None:
    """The sidebar lists sessions on every page load; that is looking, not working."""
    _ingest(client, "alice", "alpha")
    _age(_patch_rag, "alice", "alpha")

    resp = client.get("/sessions/list", params={"collection": "alpha"}, headers={"X-Auth-User": "alice"})

    assert resp.status_code == 200
    assert _last_activity(_patch_rag, "alice", "alpha") == LONG_AGO


def test_the_retention_listing_does_not_move_the_clock(client: TestClient, _patch_rag: _OwnRAG) -> None:
    """Reading a collection's expiry must never postpone it."""
    _ingest(client, "alice", "alpha")
    _age(_patch_rag, "alice", "alpha")

    assert client.get("/collections/retention", headers={"X-Auth-User": "alice"}).status_code == 200

    assert _last_activity(_patch_rag, "alice", "alpha") == LONG_AGO


def test_a_failed_stamp_never_fails_the_request(
    client: TestClient, _patch_rag: _OwnRAG, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The clock is bookkeeping; the user's request is the work."""
    _ingest(client, "alice", "alpha")
    _age(_patch_rag, "alice", "alpha")

    def _locked(self: CollectionOwnerManager, owner: str | None, logical: str, **_kw: Any) -> bool:
        raise RuntimeError("database is locked")

    monkeypatch.setattr(CollectionOwnerManager, "touch", _locked)

    assert (
        client.post("/collections/select", json={"name": "alpha"}, headers={"X-Auth-User": "alice"}).status_code == 200
    )
    assert _ingest(client, "alice", "alpha").status_code == 200


def test_retention_listing_when_retention_is_off(client: TestClient, _patch_rag: _OwnRAG) -> None:
    """With retention off nothing has a deadline, but the clock is still shown."""
    _ingest(client, "alice", "alpha")

    body = client.get("/collections/retention", headers={"X-Auth-User": "alice"}).json()

    assert body["window"] == "off"
    [row] = body["collections"]
    assert row["name"] == "alpha"
    assert row["owner"] is None
    assert row["last_activity_at"] is not None
    assert row["expires_at"] is None
    assert row["warning"] is False


def test_retention_listing_dates_and_flags_each_collection(
    monkeypatch: pytest.MonkeyPatch, _patch_rag: _OwnRAG
) -> None:
    """A collection idle past the window is due when the grace period ends, and is flagged."""
    monkeypatch.setenv("COLLECTION_RETENTION", "6m")
    with TestClient(api_module.app) as client:
        _ingest(client, "alice", "fresh")
        _ingest(client, "alice", "stale")
        _age(_patch_rag, "alice", "stale")
        state = _patch_rag.ensure_collection_owner_manager().retention_window()
        assert state is not None and state.window == "6m"

        body = client.get("/collections/retention", headers={"X-Auth-User": "alice"}).json()

    assert body["window"] == "6m"
    stale, fresh = body["collections"]
    assert stale["name"] == "stale"
    assert datetime.fromisoformat(stale["expires_at"]) == state.set_at + timedelta(days=NOTICE_DAYS)
    assert stale["warning"] is True
    assert fresh["name"] == "fresh"
    assert datetime.fromisoformat(fresh["expires_at"]) > datetime.now(UTC) + timedelta(days=150)
    assert fresh["warning"] is False


def test_an_unstamped_collection_never_expires(monkeypatch: pytest.MonkeyPatch, _patch_rag: _OwnRAG) -> None:
    """No recorded activity is listed without a deadline, and last."""
    monkeypatch.setenv("COLLECTION_RETENTION", "6m")
    with TestClient(api_module.app) as client:
        _ingest(client, "alice", "alpha")
        _ingest(client, "alice", "beta")
        _age(_patch_rag, "alice", "alpha", stamp=None)

        body = client.get("/collections/retention", headers={"X-Auth-User": "alice"}).json()

    beta, alpha = body["collections"]
    assert (beta["name"], alpha["name"]) == ("beta", "alpha")
    assert alpha["expires_at"] is None
    assert alpha["warning"] is False


def test_retention_listing_is_owner_scoped(client: TestClient) -> None:
    """A non-admin's ``all=true`` is ignored, so no other owner's collection leaks."""
    _ingest(client, "alice", "alpha")
    _ingest(client, "bob", "beta")

    body = client.get("/collections/retention?all=true", headers={"X-Auth-User": "alice"}).json()

    assert [(row["name"], row["owner"]) for row in body["collections"]] == [("alpha", None)]


def test_retention_listing_shows_admins_every_owner(client: TestClient) -> None:
    """With ``all=true`` an admin sees every collection, foreign ones named by owner."""
    _ingest(client, "alice", "alpha")
    _ingest(client, "root", "own")

    body = client.get("/collections/retention?all=true", headers=ADMIN).json()

    assert sorted((row["name"], row["owner"]) for row in body["collections"]) == [("alpha", "alice"), ("own", None)]


def test_startup_records_the_window_in_force(monkeypatch: pytest.MonkeyPatch, _patch_rag: _OwnRAG) -> None:
    """The grace period is anchored on the startup that changed the window."""
    monkeypatch.setenv("COLLECTION_RETENTION", "12m")
    with TestClient(api_module.app):
        pass

    state = _patch_rag.ensure_collection_owner_manager().retention_window()
    assert state is not None and state.window == "12m"


def test_startup_logs_the_window(
    monkeypatch: pytest.MonkeyPatch, _patch_rag: _OwnRAG, loguru_caplog_info: LogCaptureFixture
) -> None:
    """One greppable line says whether retention is on — a typo that disabled it shows up here."""
    monkeypatch.setenv("COLLECTION_RETENTION", "18m")
    with TestClient(api_module.app):
        pass

    assert any("Collection retention | window=18m" in str(r.msg) for r in loguru_caplog_info.records)


def test_config_advertises_the_retention_window(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """The SPA reads the window from ``/config`` before asking for any expiry dates."""
    assert client.get("/config").json()["collection_retention"] == "off"
    monkeypatch.setenv("COLLECTION_RETENTION", "24m")
    assert client.get("/config").json()["collection_retention"] == "24m"
