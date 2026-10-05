"""Tests for the Reports API endpoints (CRUD, items, and the five exports).

Patches ``api_module.rag`` with a minimal dummy whose ``ensure_report_manager``
returns a *real* :class:`ReportManager` backed by a shared in-memory SQLite DB
(``StaticPool`` so every TestClient worker thread sees the same data), so the
endpoints are exercised against the true manager logic end-to-end.
"""

import io
import logging
import threading
import time
import zipfile
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import docint.core.api as api_module
from docint.core.jobs import TERMINAL_STATUSES, IngestJobManager
from docint.core.state import report_render
from docint.core.state.base import Base
from docint.core.state.collection_owner_manager import CollectionOwnerManager
from docint.core.state.collection_ownership import CollectionOwnership
from docint.core.state.report_manager import ReportManager
from docint.core.state.report_pdf_store import ReportPdfStore


class _ReportRAG:
    """Minimal RAG stand-in exposing only ``ensure_report_manager``.

    Also carries the collection-owner manager, the ``list_documents`` /
    ``collection_scope`` seams that collection-overview capture needs, and the
    ``_image_collection_name`` seam add-time thumbnail enrichment uses.
    """

    def __init__(self) -> None:
        engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(engine)
        self.session_store = "sqlite://"
        self._rm = ReportManager(rag=cast(Any, self))
        self._rm._SessionMaker = sessionmaker(bind=engine)
        self._com = CollectionOwnerManager(rag=cast(Any, self))
        self._com._SessionMaker = sessionmaker(bind=engine)

    def probe_rerank_endpoint(self) -> None:
        """Satisfy the lifespan rerank probe without touching the network."""
        return None

    def probe_qdrant(self) -> bool:
        """Satisfy the lifespan startup probe without touching the network."""
        return True

    def ensure_session_manager(self) -> SimpleNamespace:
        """Satisfy the lifespan's eager session-store init without a DB.

        Returns:
            SimpleNamespace: A stand-in whose ``init_session_store_if_needed``
            is a no-op.
        """
        return SimpleNamespace(init_session_store_if_needed=lambda: None)

    def reconcile_quantization(self) -> int:
        """Satisfy the lifespan quantization reconcile without touching Qdrant."""
        return 0

    def ensure_report_manager(self) -> ReportManager:
        return self._rm

    def ensure_collection_owner_manager(self) -> CollectionOwnerManager:
        return self._com

    @contextmanager
    def reasoning_scope(self, enabled: bool | None) -> Iterator[None]:
        """No-op mirror of :meth:`RAG.reasoning_scope`; the stub has no model to switch."""
        _ = enabled

        yield

    @contextmanager
    def collection_scope(self, physical: str) -> Iterator[None]:
        """No-op scope: this stand-in carries no per-request collection state."""
        yield

    def list_documents(self) -> list[dict[str, Any]]:
        """Default empty document list; individual tests monkeypatch this."""
        return []

    def _image_collection_name(self, collection: str | None = None) -> str:
        """Name the ``_images`` companion the way the real engine does."""
        return f"{collection}_images"


@pytest.fixture(autouse=True)
def _patch_rag(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the module-level RAG singleton with the report-only dummy."""
    monkeypatch.delenv("DOCINT_AUTH_HEADER", raising=False)
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "test-operator")
    monkeypatch.setattr(api_module, "rag", _ReportRAG())


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """A TestClient bound to the FastAPI app."""
    with TestClient(api_module.app) as test_client:
        yield test_client


def _create(client: TestClient, title: str = "Case A", collection: str | None = "docs") -> dict[str, Any]:
    resp = client.post("/reports", json={"title": title, "collection_name": collection})
    assert resp.status_code == 200, resp.text
    return cast(dict[str, Any], resp.json())


def _entity_payload(chunk_id: str = "c1", *, translation: dict[str, Any] | None = None) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "chunk_id": chunk_id,
        "entity_label": "Acme [ORG]",
        "chunk_text": "Acme met Bob",
        "filename": "a.pdf",
        "page": 1,
        "entities": [{"text": "Acme", "type": "ORG"}],
    }
    if translation is not None:
        snapshot["translation"] = translation
    return {
        "artifact_type": "entity_finding",
        "dedupe_key": f"entity:{chunk_id}",
        "snapshot": snapshot,
    }


def _own_collection(name: str, owner: str = "test-operator") -> None:
    """Register ``owner`` as the owner of ``name`` directly on the manager.

    This module's RAG stand-in has no ``/ingest`` endpoint (only report
    plumbing), so a collection can't be owned via HTTP; tests register it the
    same way ``tests/test_collection_owner_manager.py`` does, straight through
    :class:`CollectionOwnerManager`.
    """
    api_module.rag.ensure_collection_owner_manager().register(owner, name)


def _stub_documents(monkeypatch: pytest.MonkeyPatch, documents: list[dict[str, Any]]) -> None:
    """Monkeypatch ``rag.list_documents`` to return a fixed document list."""
    monkeypatch.setattr(api_module.rag, "list_documents", lambda: documents)


def test_created_report_defaults_show_toc_on(client: TestClient) -> None:
    """A freshly created report has the table-of-contents flag on by default."""
    assert _create(client)["show_toc"] is True


def test_patch_show_toc(client: TestClient) -> None:
    """PATCH /reports/{id} toggles the table-of-contents flag and it persists."""
    rid = _create(client)["id"]
    resp = client.patch(f"/reports/{rid}", json={"show_toc": False})
    assert resp.status_code == 200, resp.text
    assert resp.json()["show_toc"] is False
    assert client.get(f"/reports/{rid}").json()["show_toc"] is False


def test_create_and_list(client: TestClient) -> None:
    """POST /reports creates a report that then appears in GET /reports."""
    created = _create(client, title="Case A")
    assert created["title"] == "Case A"
    assert created["items"] == []

    listed = client.get("/reports").json()["reports"]
    assert any(r["id"] == created["id"] for r in listed)


def test_list_filtered_by_collection(client: TestClient) -> None:
    """GET /reports?collection= filters reports by collection."""
    _create(client, title="A", collection="docs")
    _create(client, title="B", collection="other")
    docs = client.get("/reports", params={"collection": "docs"}).json()["reports"]
    assert {r["title"] for r in docs} == {"A"}


def test_get_missing_report_404(client: TestClient) -> None:
    """GET on an unknown report id returns 404."""
    assert client.get("/reports/99999").status_code == 404


def test_add_item_is_idempotent(client: TestClient) -> None:
    """Re-posting the same dedupe key returns the existing item (no duplicate)."""
    rid = _create(client)["id"]
    first = client.post(f"/reports/{rid}/items", json=_entity_payload("c1"))
    again = client.post(f"/reports/{rid}/items", json=_entity_payload("c1"))
    assert first.status_code == 200 and again.status_code == 200
    assert first.json()["id"] == again.json()["id"]

    report = client.get(f"/reports/{rid}").json()
    assert len(report["items"]) == 1


def test_add_item_to_missing_report_404(client: TestClient) -> None:
    """Adding an item to an unknown report id returns 404."""
    assert client.post("/reports/99999/items", json=_entity_payload()).status_code == 404


def test_remove_item(client: TestClient) -> None:
    """DELETE on an item removes it from the report."""
    rid = _create(client)["id"]
    item = client.post(f"/reports/{rid}/items", json=_entity_payload("c1")).json()
    assert client.delete(f"/reports/{rid}/items/{item['id']}").status_code == 200
    assert client.get(f"/reports/{rid}").json()["items"] == []


def test_annotate_item(client: TestClient) -> None:
    """PATCH on an item sets its investigator note."""
    rid = _create(client)["id"]
    item = client.post(f"/reports/{rid}/items", json=_entity_payload("c1")).json()
    resp = client.patch(f"/reports/{rid}/items/{item['id']}", json={"note": "evidence"})
    assert resp.status_code == 200
    assert resp.json()["note"] == "evidence"


def test_reorder_items(client: TestClient) -> None:
    """POST .../items/reorder reorders the report's items."""
    rid = _create(client)["id"]
    a = client.post(f"/reports/{rid}/items", json=_entity_payload("c1")).json()
    b = client.post(f"/reports/{rid}/items", json=_entity_payload("c2")).json()
    resp = client.post(f"/reports/{rid}/items/reorder", json={"item_ids": [b["id"], a["id"]]})
    assert resp.status_code == 200
    assert [i["id"] for i in resp.json()["items"]] == [b["id"], a["id"]]


def test_update_and_delete(client: TestClient) -> None:
    """PATCH renames a report; DELETE removes it (subsequent GET is 404)."""
    rid = _create(client, title="Old")["id"]
    renamed = client.patch(f"/reports/{rid}", json={"title": "New"})
    assert renamed.status_code == 200 and renamed.json()["title"] == "New"
    assert client.delete(f"/reports/{rid}").status_code == 200
    assert client.get(f"/reports/{rid}").status_code == 404


def test_export_md_html_json_zip(client: TestClient) -> None:
    """The md/html/json/zip exports return the right content types and bodies."""
    rid = _create(client)["id"]
    client.post(f"/reports/{rid}/items", json=_entity_payload("c1"))

    md = client.get(f"/reports/{rid}/export.md")
    assert md.status_code == 200 and "text/markdown" in md.headers["content-type"]

    html = client.get(f"/reports/{rid}/export.html")
    assert html.status_code == 200 and "text/html" in html.headers["content-type"]
    assert "inline" in html.headers["content-disposition"]

    js = client.get(f"/reports/{rid}/export.json")
    assert js.status_code == 200 and "application/json" in js.headers["content-type"]
    assert js.json()["id"] == rid

    zb = client.get(f"/reports/{rid}/export.zip")
    assert zb.status_code == 200 and "application/zip" in zb.headers["content-type"]
    names = zipfile.ZipFile(io.BytesIO(zb.content)).namelist()
    assert "entity-findings.csv" in names


def test_export_missing_report_404(client: TestClient) -> None:
    """Exporting an unknown report id returns 404."""
    assert client.get("/reports/99999/export.md").status_code == 404


def test_export_pdf_503_when_engine_unavailable(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """The PDF export returns 503 when the WeasyPrint engine is unavailable."""
    rid = _create(client)["id"]
    monkeypatch.setattr(report_render, "_load_weasyprint", lambda: (None, ImportError("no native libs")))
    resp = client.get(f"/reports/{rid}/export.pdf")
    assert resp.status_code == 503


def test_export_pdf_ok_when_engine_available(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """The PDF export returns application/pdf bytes when the engine is available."""
    rid = _create(client)["id"]

    class _FakeHTML:
        def __init__(self, string: str) -> None:
            self.string = string

        def write_pdf(self) -> bytes:
            return b"%PDF-1.7 fake"

    monkeypatch.setattr(report_render, "_load_weasyprint", lambda: (_FakeHTML, None))
    resp = client.get(f"/reports/{rid}/export.pdf")
    assert resp.status_code == 200
    assert "application/pdf" in resp.headers["content-type"]
    assert resp.content.startswith(b"%PDF")


def test_patch_toggles_collection_overview(client: TestClient) -> None:
    """PATCH /reports/{id} toggles the document-overview flag and it persists."""
    rid = _create(client)["id"]
    resp = client.patch(f"/reports/{rid}", json={"show_collection_overview": False})
    assert resp.status_code == 200, resp.text
    assert resp.json()["show_collection_overview"] is False


def test_refresh_builds_snapshot(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """POST .../collection-overview/refresh rebuilds the snapshot from current documents."""
    _own_collection("c1")
    rid = _create(client, collection="c1")["id"]
    _stub_documents(
        monkeypatch,
        [
            {
                "filename": "a.pdf",
                "mimetype": "application/pdf",
                "file_hash": "h",
                "node_count": 2,
                "page_count": 3,
                "entity_types": ["ORG"],
            }
        ],
    )
    resp = client.post(f"/reports/{rid}/collection-overview/refresh")
    assert resp.status_code == 200, resp.text
    overview = resp.json()["collection_overview"]
    assert overview["document_count"] == 1
    assert overview["documents"][0]["type_label"] == "PDF"


def test_refresh_cross_owner_is_404(client: TestClient) -> None:
    """A non-owner cannot trigger a refresh on someone else's report (404, no leak)."""
    _own_collection("c1")
    rid = _create(client, collection="c1")["id"]
    resp = client.post(f"/reports/{rid}/collection-overview/refresh", headers={"X-Auth-User": "alice"})
    assert resp.status_code == 404


def test_refresh_no_collection_is_400(client: TestClient) -> None:
    """Refreshing a report with no collection scope is a 400, not a crash."""
    rid = _create(client, collection=None)["id"]
    resp = client.post(f"/reports/{rid}/collection-overview/refresh")
    assert resp.status_code == 400


def test_refresh_build_failure_is_502(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A manifest-build failure during refresh surfaces as 502, not 500 or a silent no-op."""
    _own_collection("c1")
    rid = _create(client, collection="c1")["id"]

    def _boom() -> list[dict[str, Any]]:
        raise RuntimeError("qdrant unreachable")

    monkeypatch.setattr(api_module.rag, "list_documents", _boom)
    resp = client.post(f"/reports/{rid}/collection-overview/refresh")
    assert resp.status_code == 502


def test_refresh_unowned_collection_is_404_not_502(client: TestClient) -> None:
    """A report's collection that no longer resolves to an owned collection surfaces 404.

    Covers a ``collection_name`` that was never registered (or was deleted after
    the report was created): the ownership resolver's 404 must pass through
    refresh verbatim, not get masked as a generic 502 by the catch-all handler.
    """
    rid = _create(client, collection="ghost-collection")["id"]
    resp = client.post(f"/reports/{rid}/collection-overview/refresh")
    assert resp.status_code == 404


def test_create_captures_overview_when_collection_registered(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /reports captures the document overview once, at create time."""
    _own_collection("c1")
    _stub_documents(
        monkeypatch,
        [
            {
                "filename": "a.pdf",
                "mimetype": "application/pdf",
                "file_hash": "h",
                "node_count": 2,
                "page_count": 3,
                "entity_types": ["ORG"],
            }
        ],
    )
    created = _create(client, collection="c1")
    overview = created["collection_overview"]
    assert overview is not None
    assert overview["document_count"] >= 1


def test_create_is_failsoft_when_capture_raises(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A capture failure at create time must not fail report creation."""
    _own_collection("c1")

    def _boom() -> list[dict[str, Any]]:
        raise RuntimeError("qdrant unreachable")

    monkeypatch.setattr(api_module.rag, "list_documents", _boom)
    created = _create(client, collection="c1")
    assert created["collection_overview"] is None


# --- admin cross-owner access (owner query param) ---

ADMIN = {"X-Auth-User": "root", "X-Auth-Groups": "admins"}


def test_admin_lists_cross_owner_reports_with_owner_param(client: TestClient) -> None:
    """GET /reports?owner=alice as an admin returns alice's reports."""
    resp = client.post(
        "/reports", json={"title": "Alice case", "collection_name": None}, headers={"X-Auth-User": "alice"}
    )
    assert resp.status_code == 200

    listed = client.get("/reports", params={"owner": "alice"}, headers=ADMIN).json()["reports"]
    assert [r["title"] for r in listed] == ["Alice case"]

    # Without the owner param the admin is in their own (empty) namespace.
    assert client.get("/reports", headers=ADMIN).json()["reports"] == []


def test_admin_reads_cross_owner_report_with_owner_param(client: TestClient) -> None:
    """GET /reports/{id}?owner=alice as an admin returns alice's report; without it, 404."""
    created = client.post(
        "/reports", json={"title": "Alice case", "collection_name": None}, headers={"X-Auth-User": "alice"}
    ).json()
    rid = created["id"]

    assert client.get(f"/reports/{rid}", params={"owner": "alice"}, headers=ADMIN).status_code == 200
    assert client.get(f"/reports/{rid}", headers=ADMIN).status_code == 404


def test_non_admin_owner_param_does_not_rescope_reports(client: TestClient) -> None:
    """A non-admin passing ?owner= keeps their own report scope (404 cross-owner)."""
    created = client.post(
        "/reports", json={"title": "Alice case", "collection_name": None}, headers={"X-Auth-User": "alice"}
    ).json()
    rid = created["id"]

    assert client.get(f"/reports/{rid}", params={"owner": "alice"}, headers={"X-Auth-User": "bob"}).status_code == 404
    assert client.get("/reports", params={"owner": "alice"}, headers={"X-Auth-User": "bob"}).json()["reports"] == []


# ---------------------------------------------------------------------------
# Add-time thumbnail enrichment — visual evidence frozen into the snapshot
# ---------------------------------------------------------------------------


class _FakeImageQdrant:
    """Companion-collection double: answers image_id scrolls from a fixed map."""

    def __init__(self, points_by_image_id: dict[str, dict[str, Any]]) -> None:
        self.points = points_by_image_id
        self.scrolled_collections: list[str] = []

    def scroll(self, collection_name: str, **kwargs: Any) -> tuple[list[Any], None]:
        self.scrolled_collections.append(collection_name)
        scroll_filter = kwargs.get("scroll_filter")
        wanted: list[str] = []
        for cond in getattr(scroll_filter, "must", []) or []:
            match = getattr(cond, "match", None)
            wanted = list(getattr(match, "any", None) or ([match.value] if getattr(match, "value", None) else []))
        hits = [SimpleNamespace(id=i, payload=dict(p)) for i, p in self.points.items() if p.get("image_id") in wanted]
        return hits, None


def _physical(name: str = "docs", owner: str = "test-operator") -> str:
    resolved = api_module.rag.ensure_collection_owner_manager().resolve(owner, name)
    assert resolved is not None
    return resolved


def _wire_images(monkeypatch: pytest.MonkeyPatch, points: dict[str, dict[str, Any]]) -> _FakeImageQdrant:
    """Own 'docs' and hang a fake companion-collection client off the RAG stub."""
    _own_collection("docs")
    fake = _FakeImageQdrant(points)
    monkeypatch.setattr(api_module.rag, "qdrant_client", fake, raising=False)
    return fake


def _chat_payload_with_image(image_collection: str | None) -> dict[str, Any]:
    source: dict[str, Any] = {
        "filename": "figure.png",
        "page": None,
        "row": None,
        "score": 0.8,
        "text": "A bar chart.",
        "reference_metadata": None,
        "image_id": "img-1",
    }
    if image_collection is not None:
        source["image_collection"] = image_collection
    return {
        "artifact_type": "chat_answer",
        "dedupe_key": "chat:s1:0",
        "snapshot": {
            "session_id": "s1",
            "turn_idx": 0,
            "user_text": "What does the chart show?",
            "model_response": "It shows totals.",
            "sources": [source],
        },
    }


def test_add_chat_item_freezes_source_thumbnail(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A chat source carrying image identity gains a data-URI thumbnail at add-time."""
    fake = _wire_images(
        monkeypatch,
        {
            "p1": {
                "image_id": "img-1",
                "thumbnail_b64": "QUJD",
                "thumbnail_mime": "image/jpeg",
                "width": 320,
                "height": 180,
                "source_type": "document",
            }
        },
    )
    rid = _create(client)["id"]
    companion = f"{_physical()}_images"

    resp = client.post(f"/reports/{rid}/items", json=_chat_payload_with_image(companion))

    assert resp.status_code == 200, resp.text
    thumb = resp.json()["snapshot"]["sources"][0]["thumbnail"]
    assert thumb["data_uri"] == "data:image/jpeg;base64,QUJD"
    assert thumb["kind"] == "image"
    assert thumb["width"] == 320
    assert fake.scrolled_collections == [companion]
    stored = client.get(f"/reports/{rid}").json()["items"][0]
    assert stored["snapshot"]["sources"][0]["thumbnail"]["data_uri"] == "data:image/jpeg;base64,QUJD"


def test_add_finding_item_freezes_keyframe_thumbnail(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A finding whose chunk is a keyframe gains a snapshot-level video thumbnail."""
    _wire_images(
        monkeypatch,
        {
            "p1": {
                "image_id": "kf-1",
                "thumbnail_b64": "REVG",
                "thumbnail_mime": "image/jpeg",
                "source_type": "video_keyframe",
            }
        },
    )
    rid = _create(client)["id"]
    payload = _entity_payload()
    payload["snapshot"]["image_id"] = "kf-1"

    resp = client.post(f"/reports/{rid}/items", json=payload)

    assert resp.status_code == 200, resp.text
    thumb = resp.json()["snapshot"]["thumbnail"]
    assert thumb["data_uri"] == "data:image/jpeg;base64,REVG"
    assert thumb["kind"] == "video_keyframe"


def test_add_item_without_thumbnail_payload_stays_text_only(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A point that predates thumbnails adds cleanly with no thumbnail key."""
    _wire_images(monkeypatch, {"p1": {"image_id": "img-1", "source_type": "document"}})
    rid = _create(client)["id"]

    resp = client.post(f"/reports/{rid}/items", json=_chat_payload_with_image(f"{_physical()}_images"))

    assert resp.status_code == 200, resp.text
    assert "thumbnail" not in resp.json()["snapshot"]["sources"][0]


def test_add_item_survives_companion_scroll_failure(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A Qdrant hiccup degrades to a text-only item, never a 500."""
    fake = _wire_images(monkeypatch, {})

    def _boom(collection_name: str, **kwargs: Any) -> tuple[list[Any], None]:
        raise RuntimeError("qdrant down")

    monkeypatch.setattr(fake, "scroll", _boom)
    rid = _create(client)["id"]

    resp = client.post(f"/reports/{rid}/items", json=_chat_payload_with_image(f"{_physical()}_images"))

    assert resp.status_code == 200, resp.text
    assert "thumbnail" not in resp.json()["snapshot"]["sources"][0]


def test_add_item_ignores_foreign_image_collection(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller-supplied companion name is a cross-check, never an address."""
    fake = _wire_images(
        monkeypatch,
        {
            "p1": {
                "image_id": "img-1",
                "thumbnail_b64": "QUJD",
                "thumbnail_mime": "image/jpeg",
                "source_type": "document",
            }
        },
    )
    rid = _create(client)["id"]

    resp = client.post(f"/reports/{rid}/items", json=_chat_payload_with_image("uDEADBEEF__other_images"))

    assert resp.status_code == 200, resp.text
    assert "thumbnail" not in resp.json()["snapshot"]["sources"][0]
    assert fake.scrolled_collections == []


def test_add_item_without_report_collection_skips_enrichment(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A report with no collection has no companion to consult — item still adds."""
    fake = _wire_images(monkeypatch, {})
    rid = _create(client, collection=None)["id"]

    resp = client.post(f"/reports/{rid}/items", json=_chat_payload_with_image(None))

    assert resp.status_code == 200, resp.text
    assert "thumbnail" not in resp.json()["snapshot"]["sources"][0]
    assert fake.scrolled_collections == []


def test_enriched_item_stays_idempotent_on_re_add(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-adding the same dedupe key returns the frozen item unchanged."""
    _wire_images(
        monkeypatch,
        {
            "p1": {
                "image_id": "img-1",
                "thumbnail_b64": "QUJD",
                "thumbnail_mime": "image/jpeg",
                "source_type": "document",
            }
        },
    )
    rid = _create(client)["id"]
    payload = _chat_payload_with_image(f"{_physical()}_images")

    first = client.post(f"/reports/{rid}/items", json=payload).json()
    second = client.post(f"/reports/{rid}/items", json=payload).json()

    assert second["id"] == first["id"]
    assert second["snapshot"] == first["snapshot"]


def test_add_item_resolves_the_request_collection_not_the_report_one(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The artifact's own collection addresses the companion, not the report's.

    An investigator switches collections and keeps adding to the open report:
    the evidence lives where it was retrieved, so a lookup against the report's
    companion finds nothing.
    """
    fake = _wire_images(
        monkeypatch,
        {
            "p1": {
                "image_id": "img-1",
                "thumbnail_b64": "QUJD",
                "thumbnail_mime": "image/jpeg",
                "source_type": "document",
            }
        },
    )
    _own_collection("other")
    rid = _create(client)["id"]
    payload = _chat_payload_with_image(None)
    payload["collection"] = "other"

    resp = client.post(f"/reports/{rid}/items", json=payload)

    assert resp.status_code == 200, resp.text
    assert resp.json()["snapshot"]["sources"][0]["thumbnail"]["data_uri"] == "data:image/jpeg;base64,QUJD"
    assert fake.scrolled_collections == [f"{_physical('other')}_images"]


def test_add_item_falls_back_to_the_report_collection_when_unowned(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A request collection the caller does not own never redirects the lookup."""
    fake = _wire_images(
        monkeypatch,
        {
            "p1": {
                "image_id": "img-1",
                "thumbnail_b64": "QUJD",
                "thumbnail_mime": "image/jpeg",
                "source_type": "document",
            }
        },
    )
    rid = _create(client)["id"]
    payload = _chat_payload_with_image(None)
    payload["collection"] = "someone-elses"

    resp = client.post(f"/reports/{rid}/items", json=payload)

    assert resp.status_code == 200, resp.text
    assert fake.scrolled_collections == [f"{_physical()}_images"]


def test_add_chat_item_freezes_every_image_source(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Three cited images freeze three thumbnails in one companion round-trip."""
    fake = _wire_images(
        monkeypatch,
        {
            f"p{n}": {
                "image_id": f"img-{n}",
                "thumbnail_b64": f"QUJ{n}",
                "thumbnail_mime": "image/jpeg",
                "source_type": "standalone",
            }
            for n in (1, 2, 3)
        },
    )
    rid = _create(client)["id"]
    payload = _chat_payload_with_image(None)
    payload["snapshot"]["sources"] = [
        {"filename": f"image-{n}.png", "text": "", "image_id": f"img-{n}"} for n in (1, 2, 3)
    ]

    resp = client.post(f"/reports/{rid}/items", json=payload)

    assert resp.status_code == 200, resp.text
    sources = resp.json()["snapshot"]["sources"]
    assert [s["thumbnail"]["data_uri"] for s in sources] == [f"data:image/jpeg;base64,QUJ{n}" for n in (1, 2, 3)]
    assert len(fake.scrolled_collections) == 1


def test_add_summary_item_is_never_enriched(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Summaries carry no visual evidence — the companion is not consulted."""
    fake = _wire_images(monkeypatch, {})
    rid = _create(client)["id"]

    resp = client.post(
        f"/reports/{rid}/items",
        json={
            "artifact_type": "summary",
            "dedupe_key": "summary:docs",
            "snapshot": {"collection": "docs", "text": "A summary.", "image_id": "img-1"},
            "collection": "docs",
        },
    )

    assert resp.status_code == 200, resp.text
    assert "thumbnail" not in resp.json()["snapshot"]
    assert fake.scrolled_collections == []


# ---------------------------------------------------------------------------
# Batch add — POST /reports/{id}/items/batch ("Add all" in an Analysis section)
# ---------------------------------------------------------------------------


def test_batch_add_items(client: TestClient) -> None:
    """A batch adds every item in one request and reports the counts."""
    rid = _create(client)["id"]

    resp = client.post(
        f"/reports/{rid}/items/batch",
        json={"items": [_entity_payload("c1"), _entity_payload("c2"), _entity_payload("c3")]},
    )

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"added": 3, "skipped": 0, "updated": 0, "item_count": 3}
    report = client.get(f"/reports/{rid}").json()
    assert [i["dedupe_key"] for i in report["items"]] == ["entity:c1", "entity:c2", "entity:c3"]


def test_batch_add_skips_items_already_in_the_report(client: TestClient) -> None:
    """Re-sending a key already present counts as skipped and adds no duplicate."""
    rid = _create(client)["id"]
    client.post(f"/reports/{rid}/items", json=_entity_payload("c1"))

    resp = client.post(f"/reports/{rid}/items/batch", json={"items": [_entity_payload("c1"), _entity_payload("c2")]})

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"added": 1, "skipped": 1, "updated": 0, "item_count": 2}
    assert len(client.get(f"/reports/{rid}").json()["items"]) == 2


def test_batch_add_backfills_a_translation_into_an_item_already_in_the_report(client: TestClient) -> None:
    """A duplicate carrying a translation amends the stored snapshot and counts as updated."""
    rid = _create(client)["id"]
    client.post(f"/reports/{rid}/items", json=_entity_payload("c1"))
    translation = {"text": "Acme traf Bob", "target_lang": "de", "model": "test-model"}

    resp = client.post(
        f"/reports/{rid}/items/batch",
        json={"items": [_entity_payload("c1", translation=translation)]},
    )

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"added": 0, "skipped": 0, "updated": 1, "item_count": 1}
    stored = client.get(f"/reports/{rid}").json()["items"][0]["snapshot"]
    assert stored["translation"] == translation
    assert stored["chunk_text"] == "Acme met Bob"


def test_batch_add_rejects_an_empty_batch(client: TestClient) -> None:
    """An empty item list is a client error, not a silent success."""
    rid = _create(client)["id"]
    assert client.post(f"/reports/{rid}/items/batch", json={"items": []}).status_code == 422


def test_batch_add_rejects_an_oversize_batch(client: TestClient) -> None:
    """Above the hard cap the request is refused before anything is written."""
    rid = _create(client)["id"]
    items = [_entity_payload(f"c{i}") for i in range(api_module.REPORT_BATCH_MAX_ITEMS + 1)]

    resp = client.post(f"/reports/{rid}/items/batch", json={"items": items})

    assert resp.status_code == 422
    assert client.get(f"/reports/{rid}").json()["items"] == []


def test_batch_add_to_missing_report_404(client: TestClient) -> None:
    """A batch against an unknown report id returns 404."""
    assert client.post("/reports/99999/items/batch", json={"items": [_entity_payload()]}).status_code == 404


def test_batch_add_cross_owner_404(client: TestClient) -> None:
    """A batch against another owner's report is 404, like every other item route."""
    rid = client.post("/reports", json={"title": "Alice case"}, headers={"X-Auth-User": "alice"}).json()["id"]

    resp = client.post(
        f"/reports/{rid}/items/batch", json={"items": [_entity_payload("c1")]}, headers={"X-Auth-User": "bob"}
    )

    assert resp.status_code == 404


def test_batch_add_freezes_thumbnails_in_one_companion_scroll(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every image-bearing item in a batch is enriched, with a single scroll."""
    fake = _wire_images(
        monkeypatch,
        {
            "p1": {
                "image_id": "kf-1",
                "thumbnail_b64": "QUJD",
                "thumbnail_mime": "image/jpeg",
                "source_type": "document",
            },
            "p2": {
                "image_id": "kf-2",
                "thumbnail_b64": "REVG",
                "thumbnail_mime": "image/jpeg",
                "source_type": "video_keyframe",
            },
        },
    )
    rid = _create(client)["id"]
    first, second, third = _entity_payload("c1"), _entity_payload("c2"), _entity_payload("c3")
    first["snapshot"]["image_id"] = "kf-1"
    second["snapshot"]["image_id"] = "kf-2"

    resp = client.post(f"/reports/{rid}/items/batch", json={"items": [first, second, third], "collection": "docs"})

    assert resp.status_code == 200, resp.text
    assert fake.scrolled_collections == [f"{_physical()}_images"]
    items = {i["dedupe_key"]: i for i in client.get(f"/reports/{rid}").json()["items"]}
    assert items["entity:c1"]["snapshot"]["thumbnail"]["data_uri"] == "data:image/jpeg;base64,QUJD"
    assert items["entity:c2"]["snapshot"]["thumbnail"]["kind"] == "video_keyframe"
    assert "thumbnail" not in items["entity:c3"]["snapshot"]


def test_batch_add_survives_companion_scroll_failure(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A dead companion degrades the batch to text-only items, never a failed add."""
    _wire_images(monkeypatch, {})

    def _boom(*_args: Any, **_kwargs: Any) -> tuple[list[Any], None]:
        raise RuntimeError("qdrant down")

    monkeypatch.setattr(api_module.rag.qdrant_client, "scroll", _boom, raising=False)
    rid = _create(client)["id"]
    payload = _entity_payload("c1")
    payload["snapshot"]["image_id"] = "kf-1"

    resp = client.post(f"/reports/{rid}/items/batch", json={"items": [payload], "collection": "docs"})

    assert resp.status_code == 200, resp.text
    assert resp.json()["added"] == 1
    assert "thumbnail" not in client.get(f"/reports/{rid}").json()["items"][0]["snapshot"]


def _backdate(mgr: CollectionOwnerManager, owner: str, logical: str, stamp: datetime) -> None:
    """Set a collection's last activity and forget the in-memory throttle."""
    with mgr._session_scope() as s:
        s.query(CollectionOwnership).filter(
            CollectionOwnership.owner == owner, CollectionOwnership.logical_name == logical
        ).update({CollectionOwnership.last_activity_at: stamp})
        s.commit()
    mgr._touch_due.clear()


def test_report_work_moves_its_collections_retention_clock(client: TestClient) -> None:
    """Building a report from a collection is work on that collection."""
    mgr = cast(_ReportRAG, api_module.rag)._com
    mgr.register("test-operator", "docs")
    rid = _create(client, collection="docs")["id"]
    long_ago = datetime(2020, 1, 1, tzinfo=UTC)
    _backdate(mgr, "test-operator", "docs", long_ago)

    assert client.get(f"/reports/{rid}").status_code == 200

    [row] = mgr.list_activity("test-operator")
    assert row.last_activity_at is not None and row.last_activity_at > long_ago


def test_a_report_without_a_collection_touches_no_clock(client: TestClient) -> None:
    """A report started outside any collection still opens."""
    rid = _create(client, collection=None)["id"]

    assert client.get(f"/reports/{rid}").status_code == 200


# ---------------------------------------------------------------------------
# PDF as a background job — POST /reports/{id}/pdf, GET …/pdf, GET …/pdf/status
# ---------------------------------------------------------------------------
_PROGRESS = logging.getLogger("weasyprint.progress")


@pytest.fixture
def pdf_jobs(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Generator[IngestJobManager, None, None]:
    """A private job registry running the real PDF runner, over a scratch PDF store."""
    monkeypatch.setenv("REPORT_PDF_DIR", str(tmp_path / "report-pdfs"))
    manager = IngestJobManager(runner=api_module._run_job)
    api_module.app.dependency_overrides[api_module.get_job_manager] = lambda: manager
    yield manager
    # A render outlives its request; let it finish before the store's env var unwinds.
    for _ in range(200):
        if all(state.status in TERMINAL_STATUSES for state in manager._jobs.values()):
            break
        time.sleep(0.02)
    api_module.app.dependency_overrides.pop(api_module.get_job_manager, None)


def _fake_engine(monkeypatch: pytest.MonkeyPatch, gate: threading.Event | None = None) -> None:
    """Stand WeasyPrint in with a render that reports two pages and, given a gate, waits on it."""

    class _FakeHTML:
        def __init__(self, string: str) -> None:
            self.string = string

        def write_pdf(self) -> bytes:
            _PROGRESS.info("Step 5 - Creating layout - Page %d", 1)
            if gate is not None:
                assert gate.wait(timeout=5)
            _PROGRESS.info("Step 5 - Creating layout - Page %d", 2)
            _PROGRESS.info("Step 6 - Creating PDF")
            return b"%PDF-1.7 fake"

    monkeypatch.setattr(report_render, "_load_weasyprint", lambda: (_FakeHTML, None))


def _settled(client: TestClient, rid: int, **kwargs: Any) -> dict[str, Any]:
    """Poll a report's PDF status until its newest job has finished."""
    status: dict[str, Any] = {}
    for _ in range(200):
        status = client.get(f"/reports/{rid}/pdf/status", **kwargs).json()
        if status["job"] and status["job"]["status"] in {"completed", "failed", "cancelled"}:
            return status
        time.sleep(0.02)
    raise AssertionError(f"render did not finish; last status={status}")


def test_a_queued_pdf_renders_and_downloads(
    client: TestClient, pdf_jobs: IngestJobManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The PDF renders off the request and downloads when the job is done."""
    _fake_engine(monkeypatch)
    rid = _create(client, title="Case A")["id"]

    queued = client.post(f"/reports/{rid}/pdf")
    assert queued.status_code == 202
    status = _settled(client, rid)

    assert status["job"]["job_id"] == queued.json()["job_id"]
    assert status["job"]["status"] == "completed"
    assert status["pdf"]["current"] is True
    assert status["pdf"]["filename"] == f"report-{rid}-Case A.pdf"
    download = client.get(f"/reports/{rid}/pdf")
    assert download.status_code == 200
    assert download.headers["content-type"] == "application/pdf"
    assert "attachment" in download.headers["content-disposition"]
    assert download.content == b"%PDF-1.7 fake"


def test_a_render_reports_its_pages_as_it_goes(
    client: TestClient, pdf_jobs: IngestJobManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tab watching the job sees the page WeasyPrint has reached."""
    gate = threading.Event()
    _fake_engine(monkeypatch, gate)
    rid = _create(client)["id"]
    job_id = client.post(f"/reports/{rid}/pdf").json()["job_id"]

    reached = False
    for _ in range(200):
        if pdf_jobs._jobs[job_id].message == "Laying out page 1":
            reached = True
            break
        time.sleep(0.02)
    gate.set()
    _settled(client, rid)

    assert reached


def test_a_second_request_adopts_the_render_in_flight(
    client: TestClient, pdf_jobs: IngestJobManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A double click, or a second tab, joins the running render instead of starting another."""
    gate = threading.Event()
    _fake_engine(monkeypatch, gate)
    rid = _create(client)["id"]

    first = client.post(f"/reports/{rid}/pdf")
    second = client.post(f"/reports/{rid}/pdf")
    gate.set()

    assert second.status_code == 409
    assert second.json()["detail"]["job_id"] == first.json()["job_id"]
    assert _settled(client, rid)["job"]["job_id"] == first.json()["job_id"]


def test_two_reports_render_without_refusing_each_other(
    client: TestClient, pdf_jobs: IngestJobManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The in-flight check is per report: another report's render queues rather than 409s."""
    gate = threading.Event()
    _fake_engine(monkeypatch, gate)
    first, second = _create(client, title="A")["id"], _create(client, title="B")["id"]

    assert client.post(f"/reports/{first}/pdf").status_code == 202
    assert client.post(f"/reports/{second}/pdf").status_code == 202
    gate.set()
    assert _settled(client, first)["pdf"] is not None
    assert _settled(client, second)["pdf"] is not None


def test_pdf_routes_hide_another_owners_report(
    client: TestClient, pdf_jobs: IngestJobManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every PDF route answers 404 across owners, so a report's existence never leaks."""
    _fake_engine(monkeypatch)
    alice = {"X-Auth-User": "alice"}
    rid = client.post("/reports", json={"title": "Alice case", "collection_name": None}, headers=alice).json()["id"]
    bob = {"X-Auth-User": "bob"}

    assert client.post(f"/reports/{rid}/pdf", headers=bob).status_code == 404
    assert client.get(f"/reports/{rid}/pdf", headers=bob).status_code == 404
    assert client.get(f"/reports/{rid}/pdf/status", headers=bob).status_code == 404
    assert client.post("/reports/99999/pdf").status_code == 404
    assert client.post(f"/reports/{rid}/pdf", headers=alice).status_code == 202


def test_an_admin_renders_another_owners_pdf_with_the_owner_param(
    client: TestClient, pdf_jobs: IngestJobManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An admin working in a user's namespace can render and fetch that user's PDF."""
    _fake_engine(monkeypatch)
    alice = {"X-Auth-User": "alice"}
    rid = client.post("/reports", json={"title": "Alice case", "collection_name": None}, headers=alice).json()["id"]
    as_alice = {"params": {"owner": "alice"}, "headers": ADMIN}

    assert client.post(f"/reports/{rid}/pdf", **as_alice).status_code == 202
    assert _settled(client, rid, **as_alice)["pdf"]["current"] is True
    assert client.get(f"/reports/{rid}/pdf", **as_alice).status_code == 200
    assert client.get(f"/reports/{rid}/pdf", headers=ADMIN).status_code == 404


def test_no_render_is_queued_without_the_pdf_engine(
    client: TestClient, pdf_jobs: IngestJobManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without WeasyPrint the request fails at once instead of queueing a job that cannot run."""
    monkeypatch.setattr(report_render, "_load_weasyprint", lambda: (None, ImportError("no native libs")))
    rid = _create(client)["id"]

    assert client.post(f"/reports/{rid}/pdf").status_code == 503
    assert pdf_jobs._jobs == {}


def test_an_edit_makes_the_stored_pdf_outdated(
    client: TestClient, pdf_jobs: IngestJobManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A PDF missing what was added since it rendered must not pass as the current one."""
    _fake_engine(monkeypatch)
    rid = _create(client)["id"]
    assert client.get(f"/reports/{rid}/pdf/status").json() == {"job": None, "pdf": None}
    client.post(f"/reports/{rid}/pdf")
    assert _settled(client, rid)["pdf"]["current"] is True

    client.post(f"/reports/{rid}/items", json=_entity_payload())

    assert client.get(f"/reports/{rid}/pdf/status").json()["pdf"]["current"] is False


def test_deleting_a_report_deletes_its_pdf(
    client: TestClient, pdf_jobs: IngestJobManager, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Nothing rendered from a deleted report stays behind."""
    _fake_engine(monkeypatch)
    rid = _create(client)["id"]
    client.post(f"/reports/{rid}/pdf")
    _settled(client, rid)

    assert client.delete(f"/reports/{rid}").status_code == 200

    assert list((tmp_path / "report-pdfs").iterdir()) == []


def test_a_report_deleted_mid_render_leaves_no_pdf(
    client: TestClient, pdf_jobs: IngestJobManager, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A render that finishes after its report is gone discards what it wrote."""
    gate = threading.Event()
    _fake_engine(monkeypatch, gate)
    rid = _create(client)["id"]
    job_id = client.post(f"/reports/{rid}/pdf").json()["job_id"]

    assert client.delete(f"/reports/{rid}").status_code == 200
    gate.set()
    for _ in range(200):
        if pdf_jobs._jobs[job_id].status in TERMINAL_STATUSES:
            break
        time.sleep(0.02)

    assert pdf_jobs._jobs[job_id].empty is True
    assert not any((tmp_path / "report-pdfs").glob("*.pdf"))


def test_a_reused_report_id_never_serves_the_deleted_reports_pdf(
    client: TestClient, pdf_jobs: IngestJobManager, tmp_path: Path
) -> None:
    """SQLite reuses a deleted report's id, and the PDF left under it belongs to the old report."""
    rid = _create(client)["id"]
    ReportPdfStore(tmp_path / "report-pdfs").write(
        rid,
        b"%PDF-1.7 someone else's",
        report_created_at="2020-01-01T00:00:00",
        report_updated_at="2020-01-01T00:00:00",
        filename="report-old.pdf",
        pages=1,
        now=datetime(2020, 1, 1, tzinfo=UTC),
    )

    assert client.get(f"/reports/{rid}/pdf").status_code == 404
    assert client.get(f"/reports/{rid}/pdf/status").json()["pdf"] is None


def test_queueing_a_pdf_counts_as_activity_on_its_collection(
    client: TestClient, pdf_jobs: IngestJobManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exporting a collection's report keeps it alive under retention, like every other export."""
    _fake_engine(monkeypatch)
    touched: list[tuple[str | None, str]] = []
    monkeypatch.setattr(api_module, "_record_activity", lambda owner, logical: touched.append((owner, logical)))
    rid = _create(client, collection="docs")["id"]
    touched.clear()

    client.post(f"/reports/{rid}/pdf")

    assert touched == [("test-operator", "docs")]


def test_a_reused_report_id_does_not_inherit_the_deleted_reports_render(
    client: TestClient, pdf_jobs: IngestJobManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new report that took a deleted one's id starts with no render job of its own."""
    _fake_engine(monkeypatch)
    old = _create(client, title="Old")["id"]
    client.post(f"/reports/{old}/pdf")
    _settled(client, old)
    client.delete(f"/reports/{old}")
    time.sleep(0.002)

    new = _create(client, title="New")["id"]

    assert new == old
    assert client.get(f"/reports/{new}/pdf/status").json() == {"job": None, "pdf": None}
