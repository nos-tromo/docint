"""Tests for the Reports API endpoints (CRUD, items, and the five exports).

Patches ``api_module.rag`` with a minimal dummy whose ``ensure_report_manager``
returns a *real* :class:`ReportManager` backed by a shared in-memory SQLite DB
(``StaticPool`` so every TestClient worker thread sees the same data), so the
endpoints are exercised against the true manager logic end-to-end.
"""

import io
import zipfile
from collections.abc import Generator
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import docint.core.api as api_module
from docint.core.state import report_render
from docint.core.state.base import Base
from docint.core.state.report_manager import ReportManager


class _ReportRAG:
    """Minimal RAG stand-in exposing only ``ensure_report_manager``."""

    def __init__(self) -> None:
        engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(engine)
        self.session_store = "sqlite://"
        self._rm = ReportManager(rag=cast(Any, self))
        self._rm._SessionMaker = sessionmaker(bind=engine)

    def ensure_report_manager(self) -> ReportManager:
        return self._rm


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


def _entity_payload(chunk_id: str = "c1") -> dict[str, Any]:
    return {
        "artifact_type": "entity_finding",
        "dedupe_key": f"entity:{chunk_id}",
        "snapshot": {
            "chunk_id": chunk_id,
            "entity_label": "Acme [ORG]",
            "chunk_text": "Acme met Bob",
            "filename": "a.pdf",
            "page": 1,
            "entities": [{"text": "Acme", "type": "ORG"}],
        },
    }


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
