"""API tests for the topics routes, the topics job runner and the ``/config`` flag."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Callable, Generator, Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from topics_fakes import DIM, FakeFit, add_points, blob_vectors, make_collection, uuid_ids

import docint.core.api as api_module
import docint.core.rag as rag_module
from docint.core.jobs import TERMINAL_STATUSES, IngestJobManager, IngestJobState, JobRunner, JobStatus
from docint.core.rag import RAG
from docint.core.state.base import Base
from docint.core.state.collection_owner_manager import CollectionOwnerManager
from docint.core.topics.service import TopicsRefused, progress_payload
from docint.utils.cursor import InvalidCursorError
from docint.utils.topics_client import TopicsUnreachable

PREFIX = "/collections/{name}/topics"
CREATED = "2026-01-01T00:00:00+00:00"


class _Sessions:
    """Satisfies the lifespan's session-store init."""

    def init_session_store_if_needed(self) -> None:
        """Do nothing."""


class _Store:
    """A topic store stand-in whose active run is scripted."""

    def __init__(self) -> None:
        """Start with no run."""
        self.run: object | None = None
        self.asked: list[tuple[str, str | None]] = []

    def active(self, physical: str, *, owner_created_at: str | None) -> object | None:
        """Return the scripted run.

        Args:
            physical: Physical collection name.
            owner_created_at: The ownership row's creation time.

        Returns:
            The scripted run, or ``None``.
        """
        self.asked.append((physical, owner_created_at))
        return self.run


class _TopicsRAG:
    """A RAG stand-in: real ownership, scripted topic reads."""

    def __init__(self) -> None:
        """Wire an in-memory ownership store."""
        engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(engine)
        self.session_store = "sqlite://"
        self._com = CollectionOwnerManager(rag=cast(Any, self))
        self._com._SessionMaker = sessionmaker(bind=engine)
        self.store = _Store()
        self.overview: dict[str, Any] | None = None
        self.evidence: list[dict[str, Any]] | None = None
        self.documents: dict[str, Any] | None = None
        self.chunks: dict[str, Any] | None = None
        self.service: Any = None
        self.overview_error: Exception | None = None
        self.reads: list[tuple[str, str, int | None]] = []

    def probe_rerank_endpoint(self) -> None:
        """Skip the lifespan rerank probe."""

    def probe_qdrant(self) -> bool:
        """Skip the lifespan Qdrant probe.

        Returns:
            Always true.
        """
        return True

    def reconcile_quantization(self) -> int:
        """Skip the lifespan quantization reconcile.

        Returns:
            Always 0.
        """
        return 0

    def list_collections(self) -> list[str]:
        """List no Qdrant collections.

        Returns:
            An empty list.
        """
        return []

    def ensure_collection_owner_manager(self) -> CollectionOwnerManager:
        """Return the real ownership manager.

        Returns:
            The manager.
        """
        return self._com

    def ensure_session_manager(self) -> _Sessions:
        """Return the session stand-in.

        Returns:
            The stand-in.
        """
        return _Sessions()

    def topic_store(self) -> _Store:
        """Return the scripted store.

        Returns:
            The store.
        """
        return self.store

    def _active_topic_run(self, physical: str) -> object | None:
        """Read the scripted store as the real engine does.

        Args:
            physical: Physical collection name.

        Returns:
            The scripted run, or ``None``.
        """
        return self.store.active(physical, owner_created_at=self.collection_created_at(physical))

    def collection_created_at(self, physical: str) -> str | None:
        """Return a fixed creation time.

        Args:
            physical: Physical collection name.

        Returns:
            The fixed time.
        """
        return CREATED

    def topics_service(self) -> Any:
        """Return the scripted service.

        Returns:
            The service.
        """
        return self.service

    def topics_overview(self, physical: str) -> dict[str, Any] | None:
        """Return the scripted overview.

        Args:
            physical: Physical collection name.

        Returns:
            The overview.

        Raises:
            Exception: The scripted read failure.
        """
        self.reads.append(("overview", physical, None))
        if self.overview_error is not None:
            raise self.overview_error
        return self.overview

    def topic_evidence(
        self, physical: str, topic_id: int, *, limit: int, run_id: str | None = None
    ) -> list[dict[str, Any]] | None:
        """Return the scripted evidence.

        Args:
            physical: Physical collection name.
            topic_id: Topic id.
            limit: Most passages.
            run_id: Expected run id.

        Returns:
            The evidence.
        """
        self.reads.append(("evidence", physical, topic_id))
        return self.evidence

    def topic_documents(
        self, physical: str, topic_id: int, *, limit: int, run_id: str | None = None
    ) -> dict[str, Any] | None:
        """Return the scripted documents.

        Args:
            physical: Physical collection name.
            topic_id: Topic id.
            limit: Most documents.
            run_id: Expected run id.

        Returns:
            The documents.
        """
        self.reads.append(("documents", physical, topic_id))
        return self.documents

    def topic_chunks(
        self, physical: str, topic_id: int, *, cursor: str | None, limit: int, run_id: str | None = None
    ) -> dict[str, Any] | None:
        """Return the scripted page, refusing a cursor that says ``bad``.

        Args:
            physical: Physical collection name.
            topic_id: Topic id.
            cursor: Page cursor.
            limit: Page size.
            run_id: Expected run id.

        Returns:
            The page.

        Raises:
            InvalidCursorError: For the cursor ``bad``.
        """
        self.reads.append(("chunks", physical, topic_id))
        if cursor == "bad":
            raise InvalidCursorError("bad")
        return self.chunks


def _idle_runner(state: IngestJobState, push: Callable[[str, dict[str, Any]], None]) -> dict[str, Any]:
    """Finish at once.

    Args:
        state: The job.
        push: The event publisher.

    Returns:
        An empty result.
    """
    return {"empty": True, "resolution": None}


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> _TopicsRAG:
    """Install the RAG stand-in, with ``docs`` owned by the default identity.

    Args:
        monkeypatch: The monkeypatch fixture.

    Returns:
        The stand-in.
    """
    monkeypatch.delenv("DOCINT_AUTH_HEADER", raising=False)
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "test-operator")
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    dummy = _TopicsRAG()
    dummy._com.register("test-operator", "docs")
    monkeypatch.setattr(api_module, "rag", dummy)
    return dummy


@pytest.fixture
def make_client(fake: _TopicsRAG) -> Iterator[Callable[[JobRunner], TestClient]]:
    """Build TestClients backed by a job manager private to the test.

    Args:
        fake: The RAG stand-in.

    Yields:
        A factory taking the job runner.
    """
    clients: list[TestClient] = []

    def _make(runner: JobRunner = _idle_runner) -> TestClient:
        """Build one client.

        Args:
            runner: The job runner.

        Returns:
            The entered client.
        """
        manager = IngestJobManager(runner=runner)
        api_module.app.dependency_overrides[api_module.get_job_manager] = lambda: manager
        ctx = TestClient(api_module.app)
        clients.append(ctx)
        return ctx.__enter__()

    yield _make
    for ctx in clients:
        ctx.__exit__(None, None, None)
    api_module.app.dependency_overrides.clear()


@pytest.fixture
def client(make_client: Callable[[JobRunner], TestClient]) -> TestClient:
    """A client whose jobs finish at once.

    Args:
        make_client: The client factory.

    Returns:
        The client.
    """
    return make_client(_idle_runner)


def _physical(fake: _TopicsRAG, logical: str = "docs") -> str:
    """Resolve a logical name for the default identity.

    Args:
        fake: The RAG stand-in.
        logical: Logical name.

    Returns:
        The physical name.
    """
    physical = fake._com.resolve("test-operator", logical)
    assert physical is not None
    return physical


def _await_terminal(client: TestClient, job_id: str) -> dict[str, Any]:
    """Poll a job until it finishes.

    Args:
        client: The client.
        job_id: Job id.

    Returns:
        The final snapshot.

    Raises:
        AssertionError: When it does not finish in time.
    """
    snapshot: dict[str, Any] = {}
    for _ in range(300):
        snapshot = client.get(f"/ingest/jobs/{job_id}").json()
        if snapshot["status"] in {"completed", "failed", "cancelled"}:
            return snapshot
        time.sleep(0.01)
    raise AssertionError(f"job did not finish; last snapshot={snapshot}")


def _topics_routes() -> list[tuple[str, str]]:
    """Every ``(method, path)`` under the topics prefix, read off the app.

    Returns:
        The routes, sorted.
    """
    return sorted(
        (method, route.path)
        for route in api_module.app.routes
        if isinstance(route, APIRoute) and route.path.startswith(PREFIX)
        for method in route.methods or ()
    )


def _call(client: TestClient, method: str, path: str, logical: str, headers: dict[str, str] | None = None) -> Any:
    """Call one topics route for a collection.

    Args:
        client: The client.
        method: HTTP method.
        path: Route path template.
        logical: Logical collection name.
        headers: Extra headers.

    Returns:
        The response.
    """
    url = path.replace("{name}", logical).replace("{topic_id}", "1")
    return client.request(method, url, json={} if method == "POST" else None, headers=headers or {})


def test_the_topics_routes_exist() -> None:
    """The enumeration below covers all five routes."""
    assert _topics_routes() == [
        ("GET", PREFIX),
        ("GET", f"{PREFIX}/{{topic_id}}/chunks"),
        ("GET", f"{PREFIX}/{{topic_id}}/documents"),
        ("GET", f"{PREFIX}/{{topic_id}}/evidence"),
        ("POST", PREFIX),
    ]


def test_topics_routes_honor_admin_owner_context(client: TestClient, fake: _TopicsRAG) -> None:
    """An admin can read and build another owner's topics; an ordinary caller remains in their own namespace.

    Args:
        client: The HTTP client.
        fake: The RAG stand-in with real ownership checks.
    """
    physical = fake._com.register("analyst-b", "shared-docs")
    fake.overview = {"run_id": "synthetic-run"}
    fake.evidence = []
    fake.documents = {"documents": [], "more": False}
    fake.chunks = {"items": [], "next_cursor": None}
    admin = {"X-Auth-User": "root", "X-Auth-Groups": "admins"}
    for method, path in _topics_routes():
        url = path.replace("{name}", "shared-docs").replace("{topic_id}", "0")
        response = client.request(method, url, params={"owner": "analyst-b"}, headers=admin)
        assert response.status_code == (202 if method == "POST" else 200)
        denied = client.request(method, url, params={"owner": "analyst-b"}, headers={"X-Auth-User": "analyst-a"})
        assert denied.status_code == 404
    assert fake.reads and all(read[1] == physical for read in fake.reads)


def test_post_queues_a_topics_job_with_its_action(client: TestClient, fake: _TopicsRAG) -> None:
    """A build is queued as a topics job; a discover with a run carries its action.

    Args:
        client: The client.
        fake: The RAG stand-in.
    """
    res = client.post("/collections/docs/topics", json={})
    assert res.status_code == 202
    built = _await_terminal(client, res.json()["job_id"])
    assert built["kind"] == "topics"
    assert built["action"] == "build"
    assert built["collection"] == "docs"
    assert fake.store.asked == []

    fake.store.run = object()
    res = client.post("/collections/docs/topics", json={"action": "discover"})
    assert res.status_code == 202
    assert _await_terminal(client, res.json()["job_id"])["action"] == "discover"
    assert fake.store.asked == [(_physical(fake), CREATED)]
    assert _physical(fake) not in json.dumps(built)

    assert client.post("/collections/docs/topics", json={"action": "other"}).status_code == 422


def test_a_second_post_answers_409_with_the_running_job_id(make_client: Callable[[JobRunner], TestClient]) -> None:
    """A run in flight is reported with its id rather than started twice.

    Args:
        make_client: The client factory.
    """
    gate = threading.Event()

    def blocking(state: IngestJobState, push: Callable[[str, dict[str, Any]], None]) -> dict[str, Any]:
        """Hold the job until released.

        Args:
            state: The job.
            push: The event publisher.

        Returns:
            An empty result.
        """
        gate.wait(timeout=5)
        return {"empty": True, "resolution": None}

    client = make_client(blocking)
    first = client.post("/collections/docs/topics", json={})
    second = client.post("/collections/docs/topics", json={})
    gate.set()

    assert first.status_code == 202
    assert second.status_code == 409
    assert second.json()["detail"] == {
        "message": "A topics run is already in progress.",
        "job_id": first.json()["job_id"],
    }


@pytest.mark.parametrize("action", ["discover", "align"])
def test_discover_or_align_while_a_job_is_active_answers_409_not_400(
    make_client: Callable[[JobRunner], TestClient], action: str
) -> None:
    """A busy collection is reported as busy even before its first run exists.

    Args:
        make_client: The client factory.
        action: The action.
    """
    gate = threading.Event()

    def blocking(state: IngestJobState, push: Callable[[str, dict[str, Any]], None]) -> dict[str, Any]:
        """Hold the job until released.

        Args:
            state: The job.
            push: The event publisher.

        Returns:
            An empty result.
        """
        gate.wait(timeout=5)
        return {"empty": True, "resolution": None}

    client = make_client(blocking)
    first = client.post("/collections/docs/topics", json={})
    second = client.post("/collections/docs/topics", json={"action": action})
    gate.set()

    assert second.status_code == 409
    assert second.json()["detail"]["job_id"] == first.json()["job_id"]


@pytest.mark.parametrize("action", ["discover", "align"])
def test_discover_or_align_without_a_run_answers_400(client: TestClient, action: str) -> None:
    """Extending or aligning needs a run to extend.

    Args:
        client: The client.
        action: The action.
    """
    res = client.post("/collections/docs/topics", json={"action": action})

    assert res.status_code == 400
    assert res.json()["detail"] == "Build topics first."


@pytest.mark.parametrize(("method", "path"), _topics_routes())
def test_topics_routes_answer_404_when_disabled(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, method: str, path: str
) -> None:
    """Disabled topics answer 404 before ownership is even looked at.

    Args:
        client: The client.
        monkeypatch: The monkeypatch fixture.
        method: HTTP method.
        path: Route path template.
    """
    monkeypatch.setenv("TOPICS_ENABLED", "false")
    for logical in ("docs", "nobody-owns-this"):
        res = _call(client, method, path, logical)
        assert res.status_code == 404
        assert res.json()["detail"] == "Topics are disabled."


@pytest.mark.parametrize(("method", "path"), _topics_routes())
def test_topics_routes_answer_404_for_another_owners_collection(
    client: TestClient, fake: _TopicsRAG, method: str, path: str
) -> None:
    """Another owner's collection is indistinguishable from a missing one.

    Args:
        client: The client.
        fake: The RAG stand-in.
        method: HTTP method.
        path: Route path template.
    """
    fake.overview = {"run_id": "r"}
    fake.evidence = []
    fake.documents = {"documents": [], "more": False}
    fake.chunks = {"items": [], "next_cursor": None}
    fake.store.run = object()

    res = _call(client, method, path, "docs", headers={"X-Auth-User": "bob"})

    assert res.status_code == 404
    assert res.json()["detail"] == "Collection 'docs' not found"
    assert fake.reads == []


def test_get_topics_answers_200_with_the_overview_and_204_without_one(client: TestClient, fake: _TopicsRAG) -> None:
    """The overview is served as is, and its absence is an empty 204.

    Args:
        client: The client.
        fake: The RAG stand-in.
    """
    none = client.get("/collections/docs/topics")
    assert none.status_code == 204
    assert none.content == b""

    fake.overview = {"run_id": "20260101-000000-abcdef", "topics": []}
    res = client.get("/collections/docs/topics")
    assert res.status_code == 200
    assert res.json() == fake.overview
    assert fake.reads[-1] == ("overview", _physical(fake), None)


def test_get_topics_answers_500_when_the_read_fails(client: TestClient, fake: _TopicsRAG) -> None:
    """A failed read answers static copy.

    Args:
        client: The client.
        fake: The RAG stand-in.
    """
    fake.overview_error = RuntimeError("http://internal-host:6333 refused")
    res = client.get("/collections/docs/topics")

    assert res.status_code == 500
    assert res.json()["detail"] == "Request failed."
    assert "internal-host" not in res.text


def test_evidence_documents_and_chunks_routes(client: TestClient, fake: _TopicsRAG) -> None:
    """The three topic reads serve their payloads, 404 an unknown topic and 400 a bad cursor.

    Args:
        client: The client.
        fake: The RAG stand-in.
    """
    for route in ("evidence", "documents", "chunks"):
        res = client.get(f"/collections/docs/topics/7/{route}")
        assert res.status_code == 404, route
        assert res.json()["detail"] == "Topic not found."

    fake.evidence = [{"id": "p1", "text": "synthetic"}]
    fake.documents = {"documents": [{"file_hash": "h", "filename": "f.pdf", "count": 2}], "more": False}
    fake.chunks = {"items": [{"id": "p1"}], "next_cursor": None}

    assert client.get("/collections/docs/topics/7/evidence").json() == {"topic": 7, "sources": fake.evidence}
    assert client.get("/collections/docs/topics/7/documents").json() == fake.documents
    assert client.get("/collections/docs/topics/7/chunks").json() == fake.chunks
    assert ("evidence", _physical(fake), 7) in fake.reads

    bad = client.get("/collections/docs/topics/7/chunks", params={"cursor": "bad"})
    assert bad.status_code == 400
    assert bad.json()["detail"] == "Invalid request."

    assert client.get("/collections/docs/topics/7/evidence", params={"limit": 21}).status_code == 422
    assert client.get("/collections/docs/topics/7/documents", params={"limit": 501}).status_code == 422
    assert client.get("/collections/docs/topics/7/chunks", params={"limit": 201}).status_code == 422
    assert client.get("/collections/docs/topics/x/chunks").status_code == 422


def test_config_reports_topics_enabled(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """``/config`` tells the SPA whether to offer topics.

    Args:
        client: The client.
        monkeypatch: The monkeypatch fixture.
    """
    assert client.get("/config").json()["topics_enabled"] is True
    monkeypatch.setenv("TOPICS_ENABLED", "false")
    assert client.get("/config").json()["topics_enabled"] is False


class _Service:
    """A topics service stand-in whose run raises."""

    def __init__(self, exc: Exception) -> None:
        """Script the failure.

        Args:
            exc: What ``run`` raises.
        """
        self.exc = exc

    def run(self, action: str, physical: str, **_kwargs: Any) -> Any:
        """Raise the scripted failure.

        Args:
            action: The action.
            physical: Physical collection name.
            **_kwargs: Ignored.

        Raises:
            Exception: The scripted one.
        """
        raise self.exc


def _topics_state(physical: str, action: str = "build") -> IngestJobState:
    """Build a topics job state.

    Args:
        physical: Physical collection name.
        action: The action.

    Returns:
        The state.
    """
    return IngestJobState(
        job_id="job-1", owner="test-operator", logical_name="docs", physical=physical, kind="topics", action=action
    )


def test_the_runner_reports_a_refusal_as_a_coded_warning_and_an_empty_completion(fake: _TopicsRAG) -> None:
    """A refusal completes the job as empty, with the refusal as a coded warning.

    Args:
        fake: The RAG stand-in.
    """
    fake.service = _Service(TopicsRefused("no_run"))
    pushed: list[tuple[str, dict[str, Any]]] = []

    result = api_module._run_topics_job(
        _topics_state("p1", "align"), lambda event, payload: pushed.append((event, payload))
    )

    assert result == {"empty": True, "resolution": None}
    assert pushed == [
        ("topics_progress", progress_payload("starting", 0, 0)),
        ("warning", {"message": "Build topics first.", "code": "no_run"}),
    ]


@pytest.fixture
def anyio_backend() -> str:
    """Run the async tests on asyncio only.

    Returns:
        The backend name.
    """
    return "asyncio"


@pytest.mark.anyio
async def test_the_runner_reports_an_endpoint_fault_as_a_static_url_free_warning_and_fails(
    fake: _TopicsRAG,
) -> None:
    """An endpoint fault warns with static copy, then fails the job as ``topics_failed``.

    Args:
        fake: The RAG stand-in.
    """
    fake.service = _Service(TopicsUnreachable("http://topics-secret-host:4000/topics/fit could not be reached"))
    manager = IngestJobManager(runner=api_module._run_job)
    state = await manager.create(owner="test-operator", logical_name="docs", physical="p1", kind="topics")
    for _ in range(300):
        if state.status in TERMINAL_STATUSES:
            break
        await asyncio.sleep(0.01)

    assert state.status is JobStatus.FAILED
    frames = state.history()
    assert "topics-secret-host" not in json.dumps(frames)
    events = [(f.split("\n", 1)[0].removeprefix("event: "), json.loads(f.split("data: ", 1)[1])) for f in frames]
    warning = next(data for name, data in events if name == "warning")
    assert warning["code"] == "topics_unreachable"
    assert warning["message"] == "The topics service could not be reached."
    assert events[-1][0] == "error"
    assert events[-1][1]["code"] == "topics_failed"
    await manager.stop()


@pytest.fixture
def real_rag(monkeypatch: pytest.MonkeyPatch) -> Generator[RAG, None, None]:
    """Install a real RAG over an in-memory collection, with a scripted fit.

    Args:
        monkeypatch: The monkeypatch fixture.

    Yields:
        The RAG.
    """
    blobs = np.eye(3, DIM, dtype=np.float64)

    class _Client:
        """Stands in for the fit client."""

        def __init__(self, _config: Any) -> None:
            """Script the fit.

            Args:
                _config: Ignored.
            """
            self.fit = FakeFit(blobs)

    monkeypatch.setenv("TOPICS_LABELS", "false")
    monkeypatch.setattr(rag_module, "TopicsClient", _Client)
    monkeypatch.setattr(RAG, "collection_created_at", lambda self, physical: CREATED)
    client = QdrantClient(location=":memory:")
    make_collection(client, "p1")
    vectors = blob_vectors(0, blobs, 30, 0.05)
    add_points(
        client,
        "p1",
        [
            (pid, vec, {"search_text": f"passage {pid} blob{i // 30} shared words for a synthetic topic"})
            for i, (pid, vec) in enumerate(zip(uuid_ids(len(vectors)), vectors, strict=True))
        ],
    )
    rag = RAG(qdrant_collection="")
    rag._qdrant_client = client
    monkeypatch.setattr(api_module, "rag", rag)
    yield rag
    client.close()


def test_the_runner_forgets_the_document_cache_after_writing(real_rag: RAG, tmp_path: Path) -> None:
    """A run that wrote assignments drops the collection's cached document rows.

    Args:
        real_rag: The installed RAG.
        tmp_path: Per-test directory.
    """
    real_rag._documents_cache["p1"] = [{"filename": "f.pdf", "file_hash": "h"}]
    real_rag._documents_cache["other"] = [{"filename": "g.pdf", "file_hash": "g"}]
    pushed: list[tuple[str, dict[str, Any]]] = []

    result = api_module._run_topics_job(_topics_state("p1"), lambda event, payload: pushed.append((event, payload)))

    assert result["empty"] is False
    assert result["stats"] == {"topics": 3, "added": 3, "assigned": 90}
    assert result["artifact"]["action"] == "build"
    assert "p1" not in real_rag._documents_cache
    assert "other" in real_rag._documents_cache
    assert pushed[0] == ("topics_progress", progress_payload("starting", 0, 0))
    assert all(event in {"topics_progress", "warning"} for event, _ in pushed)


def test_detail_routes_refuse_a_run_replaced_by_a_rebuild(
    client: TestClient, real_rag: RAG, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HTTP detail reads pin a real service run and refuse the same topic number after a rebuild.

    Args:
        client: The HTTP client.
        real_rag: The RAG with synthetic Qdrant data and a scripted remote fit.
        monkeypatch: Pytest monkeypatch.
    """
    monkeypatch.setattr(api_module, "_require_owned_collection", lambda name, principal: "p1")
    first = api_module._run_topics_job(_topics_state("p1"), lambda event, payload: None)
    old_id = first["artifact"]["run_id"]
    second = api_module._run_topics_job(_topics_state("p1"), lambda event, payload: None)
    run_id = second["artifact"]["run_id"]
    assert old_id != run_id
    for read in ("evidence", "documents", "chunks"):
        path = f"/collections/docs/topics/0/{read}"
        assert client.get(path, params={"run_id": old_id}).status_code == 404
        assert client.get(path, params={"run_id": run_id}).status_code == 200
        assert client.get(path).status_code == 200
