"""Endpoint tests for the extract routes.

Fixtures mirror ``tests/test_summarize_api.py``: a private
:class:`~docint.core.jobs.IngestJobManager` per test, a ``rag`` stub with no
Qdrant behind it, and a temporary extract store. Every payload is synthetic.
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from docint.core import api as api_module
from docint.core.extract.store import ExtractStore
from docint.core.jobs import IngestJobManager, JobRunner

_NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


class _StubOwners:
    """Passthrough ownership manager: every logical name is already owned."""

    def register(self, owner: str | None, logical: str) -> str:
        """Return ``logical`` unchanged."""
        return logical

    def resolve(self, owner: str | None, logical: str) -> str | None:
        """Return ``logical`` unchanged, except for a name reserved as foreign."""
        return None if logical == "not-mine" else logical


class _StubRAG:
    """Minimal stand-in for :class:`~docint.core.rag.RAG`."""

    def __init__(self) -> None:
        """Initialize with one collection and no points."""
        self.qdrant_collection = "col"
        self.qdrant_client = object()
        self._owners = _StubOwners()

    def probe_rerank_endpoint(self) -> None:
        """Satisfy the lifespan rerank probe."""
        return None

    def probe_qdrant(self) -> bool:
        """Satisfy the lifespan startup probe."""
        return True

    def ensure_session_manager(self) -> SimpleNamespace:
        """Satisfy the lifespan session-store init."""
        return SimpleNamespace(init_session_store_if_needed=lambda: None)

    def reconcile_quantization(self) -> int:
        """Satisfy the lifespan quantization reconcile."""
        return 0

    def list_collections(self) -> list[str]:
        """Return the fixed collection list."""
        return ["col"]

    def ensure_collection_owner_manager(self) -> _StubOwners:
        """Return the passthrough ownership manager."""
        return self._owners

    def _image_collection_name(self, collection: str | None = None) -> str:
        """Return the companion name for a collection."""
        return f"{collection or self.qdrant_collection}_images"

    @contextmanager
    def collection_scope(self, physical: str) -> Iterator[None]:
        """Bind then restore the active collection."""
        prev = self.qdrant_collection
        self.qdrant_collection = physical
        try:
            yield
        finally:
            self.qdrant_collection = prev


def _node(text: str) -> str:
    """Serialize a llama-index-style node blob."""
    return json.dumps({"text": text, "start_char_idx": 0})


#: One document chunk and one image, as the gather layer would hand them over.
_MAIN_POINTS: list[tuple[str, dict[str, Any]]] = [
    ("p1", {"file_hash": "a1b2c3d4", "file_name": "report.pdf", "page": 1, "_node_content": _node("body")})
]
_IMAGE_POINTS: list[tuple[str, dict[str, Any]]] = [
    (
        "i1",
        {
            "image_id": "img-1",
            "source_type": "document",
            "source_doc_id": "a1b2c3d4",
            "llm_description": "a chart",
            "llm_tags": [],
            "ocr_text": "",
            "thumbnail_b64": "//9k",
            "thumbnail_mime": "image/jpeg",
        },
    )
]


def _posting(point: str, uuid: str) -> tuple[str, dict[str, Any]]:
    """Build a synthetic postings-table row."""
    return point, {
        "source": "table",
        "file_hash": "table-hash",
        "file_name": "postings.csv",
        "reference_metadata": {
            "uuid": uuid,
            "type": "posting",
            "network": "examplenet",
            "author": "Example Account",
            "timestamp": "2026-01-02T03:04:05",
            "text": "words",
        },
        "_node_content": _node("words"),
    }


_POSTING_POINTS = [_posting("r1", "uuid-1"), _posting("r2", "uuid-2")]


def _default_runner(state: Any, push: Any) -> dict[str, Any]:
    """Stand-in runner that resolves immediately."""
    return {"empty": False, "resolution": None}


@pytest.fixture
def store_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the extract store at a temporary directory."""
    root = tmp_path / "extracts"
    monkeypatch.setenv("EXTRACT_DIR", str(root))
    return root


@pytest.fixture
def make_client(monkeypatch: pytest.MonkeyPatch, store_root: Path) -> Generator[Callable[..., TestClient], None, None]:
    """Build a TestClient with a private job manager and a stub ``rag``."""
    monkeypatch.setattr(api_module, "rag", _StubRAG())
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "test-operator")
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    monkeypatch.delenv("DOCINT_AUTH_HEADER", raising=False)
    monkeypatch.setattr(
        api_module,
        "scroll_collection",
        lambda client, collection, image_collection, source_id=None: (_MAIN_POINTS, _IMAGE_POINTS),
    )
    clients: list[TestClient] = []
    entered: list[TestClient] = []

    def _make(runner: JobRunner | None = None) -> TestClient:
        manager = IngestJobManager(runner=runner or api_module._run_job)
        api_module.app.dependency_overrides[api_module.get_job_manager] = lambda: manager
        ctx = TestClient(api_module.app)
        clients.append(ctx)
        client = ctx.__enter__()
        entered.append(client)
        return client

    yield _make

    # Drain before teardown: a job outlives the request that queued it, and the
    # store reads its root from EXTRACT_DIR at write time — so a job still
    # running when this fixture's monkeypatch unwinds writes its bundle into
    # the *next* test's directory.
    for client in entered:
        for _ in range(200):
            jobs = client.get("/ingest/jobs").json()["jobs"]
            if all(job["status"] in {"completed", "failed"} for job in jobs):
                break
            time.sleep(0.05)
    for ctx in clients:
        ctx.__exit__(None, None, None)
    api_module.app.dependency_overrides.clear()


@pytest.fixture
def client(make_client: Callable[..., TestClient]) -> TestClient:
    """A TestClient running the real extract job runner."""
    return make_client()


def _seed(store_root: Path, physical: str = "col", **meta: Any) -> str:
    """Write one stored extract directly to disk, with no job involved."""
    store = ExtractStore(store_root)
    record = store.write(
        physical,
        zip_bytes=b"PK-payload",
        meta={"collection": "col", "target": None, "counts": {}, "pdf_skipped": False, **meta},
        now=_NOW,
    )
    return str(record["extract_id"])


# --------------------------------------------------------------------------- #
# Queueing
# --------------------------------------------------------------------------- #
def test_post_queues_a_job(client: TestClient) -> None:
    """Building an extract is a background job, not a blocking request."""
    response = client.post("/collections/col/extracts")
    assert response.status_code == 202
    assert response.json()["job_id"]


def test_a_second_build_is_refused_with_the_in_flight_job(make_client: Callable[..., TestClient]) -> None:
    """Two concurrent builds of one collection would race on the store."""
    import threading

    gate = threading.Event()

    def blocking(state: Any, push: Any) -> dict[str, Any]:
        gate.wait(timeout=5)
        return {"empty": False, "resolution": None}

    client = make_client(blocking)
    first = client.post("/collections/col/extracts")
    second = client.post("/collections/col/extracts")
    assert first.status_code == 202
    assert second.status_code == 409
    assert second.json()["detail"]["job_id"] == first.json()["job_id"]
    gate.set()


def test_a_cross_owner_collection_is_not_found(client: TestClient) -> None:
    """A foreign collection 404s so its existence never leaks."""
    assert client.post("/collections/not-mine/extracts").status_code == 404


def test_the_job_writes_an_artifact_that_lists_and_downloads(client: TestClient) -> None:
    """The whole round trip: queue, store, list, download."""
    job_id = client.post("/collections/col/extracts").json()["job_id"]
    for _ in range(100):
        snapshot = client.get(f"/ingest/jobs/{job_id}").json()
        if snapshot["status"] in {"completed", "failed"}:
            break
        time.sleep(0.05)
    assert snapshot["status"] == "completed"
    assert snapshot["artifact"]["filename"].startswith("col-extract-")

    listed = client.get("/collections/col/extracts").json()["extracts"]
    assert [record["extract_id"] for record in listed] == [snapshot["artifact"]["extract_id"]]

    download = client.get(f"/collections/col/extracts/{listed[0]['extract_id']}/download")
    assert download.status_code == 200
    assert download.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
        assert any(name.endswith("README.md") for name in archive.namelist())


# --------------------------------------------------------------------------- #
# Listing, downloading, deleting
# --------------------------------------------------------------------------- #
def test_the_listing_reads_the_store_not_the_registry(client: TestClient, store_root: Path) -> None:
    """Jobs are in-memory and evicted; the artifacts on disk outlive them."""
    extract_id = _seed(store_root)
    listed = client.get("/collections/col/extracts").json()["extracts"]
    assert [record["extract_id"] for record in listed] == [extract_id]


def test_download_names_the_file_after_the_collection(client: TestClient, store_root: Path) -> None:
    """A downloads folder must show what the archive is."""
    extract_id = _seed(store_root)
    response = client.get(f"/collections/col/extracts/{extract_id}/download")
    assert "col-extract-20260102-0304.zip" in response.headers["content-disposition"]
    assert response.headers["cache-control"] == "no-store"


def test_download_of_an_unknown_extract_is_not_found(client: TestClient) -> None:
    """An id nothing matches is a 404, never a traversal attempt."""
    assert client.get("/collections/col/extracts/20260102-030405-deadbeef/download").status_code == 404
    assert client.get("/collections/col/extracts/..%2F..%2Fetc/download").status_code == 404


def test_delete_removes_a_stored_extract(client: TestClient, store_root: Path) -> None:
    """An operator can clear a build they no longer want."""
    extract_id = _seed(store_root)
    assert client.delete(f"/collections/col/extracts/{extract_id}").status_code == 200
    assert client.get("/collections/col/extracts").json()["extracts"] == []
    assert client.delete(f"/collections/col/extracts/{extract_id}").status_code == 404


# --------------------------------------------------------------------------- #
# Per-source downloads
# --------------------------------------------------------------------------- #
def test_a_source_renders_markdown_immediately(client: TestClient) -> None:
    """One document is small enough to answer on the request."""
    response = client.get("/collections/col/sources/a1b2c3d4/extract.md")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert "report.pdf" in response.text


def test_a_source_renders_a_zip(client: TestClient) -> None:
    """The same source as a bundle, figures included."""
    response = client.get("/collections/col/sources/a1b2c3d4/extract.zip")
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert any(name.endswith("img-1.jpg") for name in archive.namelist())


def test_an_unknown_source_is_not_found(client: TestClient) -> None:
    """A source the collection does not hold is a 404, not an empty bundle."""
    assert client.get("/collections/col/sources/nope/extract.md").status_code == 404


def test_an_oversize_source_is_refused_with_its_size(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A postings table expands to every post in it; it belongs in a job."""
    monkeypatch.setenv("EXTRACT_SYNC_MAX_UNITS", "1")
    monkeypatch.setattr(
        api_module,
        "scroll_collection",
        lambda client, collection, image_collection, source_id=None: (_POSTING_POINTS, []),
    )
    response = client.get("/collections/col/sources/table-hash/extract.md")
    assert response.status_code == 413
    assert response.json()["detail"]["units"] == 2


def test_a_missing_pdf_engine_degrades_to_503(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A PDF outage must not read as a missing source."""
    from docint.core.state.report_render import PdfEngineUnavailableError

    def _unavailable(document: str) -> bytes:
        raise PdfEngineUnavailableError("no pango")

    monkeypatch.setattr(api_module, "html_to_pdf", _unavailable)
    assert client.get("/collections/col/sources/a1b2c3d4/extract.pdf").status_code == 503


def test_an_unknown_format_is_rejected(client: TestClient) -> None:
    """Only the three documented formats are routable."""
    assert client.get("/collections/col/sources/a1b2c3d4/extract.docx").status_code == 422


def test_the_case_file_travels_from_the_request_to_the_stored_bundle(client: TestClient) -> None:
    """An appendix is filed under the report the caller had open.

    The listing is what the SPA shows, so the fields have to survive the job
    and land in the store's sidecar, not merely reach the renderer.
    """
    response = client.post(
        "/collections/col/extracts",
        json={"reference_number": "AZ-12/26", "operator": "A. Analyst"},
    )
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    for _ in range(100):
        snapshot = client.get(f"/ingest/jobs/{job_id}").json()
        if snapshot["status"] in {"completed", "failed"}:
            break
        time.sleep(0.05)
    assert snapshot["status"] == "completed"

    (record,) = client.get("/collections/col/extracts").json()["extracts"]
    assert record["reference_number"] == "AZ-12/26"
    assert record["operator"] == "A. Analyst"


def test_a_single_source_download_takes_the_case_file_too(client: TestClient) -> None:
    """The same appendix chrome on the one-source route.

    Checked through the ZIP, whose README carries the header; the bare ``md``
    format is one unit's own document and has no header block to put it in.
    """
    response = client.get(
        "/collections/col/sources/a1b2c3d4/extract.zip",
        params={"reference_number": "AZ-12/26", "operator": "A. Analyst"},
    )
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        readme = archive.read(next(n for n in archive.namelist() if n.endswith("README.md"))).decode("utf-8")
    assert "AZ-12/26" in readme
    assert "A. Analyst" in readme
