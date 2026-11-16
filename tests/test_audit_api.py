"""API tests for the data-protection audit trail (``docs/audit.md``).

One test per recorded behaviour, through the real routes over ``test_api``'s
``DummyRAG``: the startup event, the actor mapping, the purpose header and the
purpose rule, fail closed on a broken audit stream, per-action events with
their targets and recipients, long id lists, jobs carrying their request's
context, and the retention sweep. The per-test chain (and the lifespan's,
which lands on the same stream) is verified by ``conftest.audit_stream``.
"""

import io
import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from conftest import run_ingest
from fastapi.testclient import TestClient
from starlette.requests import Request as StarletteRequest
from test_api import ANALYSIS_ID, STORED_ANALYSIS, DummyRAG, _sse_frames
from typing_extensions import override

import docint.core.api as api_module
from docint.core import audit
from docint.core.entities.resolution import ResolutionSummary
from docint.core.jobs import IngestJobManager
from docint.core.rag import IngestStats
from docint.core.search.fulltext import parse_keywords
from docint.core.state.session_manager import SessionCollectionMismatchError
from docint.utils.audit_disclose import Disclosure, DisclosureRefused, current_ledger, record_disclosure
from docint.utils.audit_event import AuditEmitter, sha256_text
from docint.utils.audit_forward import CHAT_REASON, forward_headers
from docint.utils.audit_meter import TransferKey, count_transfer, staged_meter, take_staged
from docint.utils.audit_runtime import PURPOSE_HEADER, encode_purpose
from docint.utils.hashing import compute_file_hash
from docint.utils.translate_client import TranslateResult

ALICE = {"X-Auth-User": "alice"}
ADMIN = {"X-Auth-User": "root", "X-Auth-Groups": "admins"}
REPO_ROOT = Path(__file__).resolve().parents[1]
#: A workbench job id as the client reads it off an answer.
JOB = "0" * 32


class _FailingStream(io.StringIO):
    """A stream that takes writes until ``fail`` is set, then raises like a full disk.

    With ``refuse`` set it refuses only the lines that contain it.
    """

    fail = False
    refuse: str | None = None

    @override
    def write(self, s: str) -> int:
        """Write, or raise once the stream is set to fail.

        Args:
            s (str): The text.

        Returns:
            int: Characters written.

        Raises:
            OSError: Once ``fail`` is set, or for a line holding ``refuse``.
        """
        if self.fail or (self.refuse is not None and self.refuse in s):
            raise OSError("no space left on device")
        return super().write(s)


@pytest.fixture
def rag(monkeypatch: pytest.MonkeyPatch) -> DummyRAG:
    """Put ``test_api``'s RAG double behind the app, with the dev default identity.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.

    Returns:
        DummyRAG: The double.
    """
    monkeypatch.delenv("DOCINT_AUTH_HEADER", raising=False)
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "test-operator")
    dummy = DummyRAG()
    monkeypatch.setattr(api_module, "rag", dummy)
    return dummy


@pytest.fixture
def client(rag: DummyRAG) -> Iterator[TestClient]:
    """Run the app with its lifespan, so its own chain opens on the test stream.

    Jobs run on a manager of the test's own (the real runner), so none is left
    behind on the application's.

    Args:
        rag (DummyRAG): The RAG double.

    Yields:
        TestClient: The client.
    """
    manager = IngestJobManager(runner=api_module._run_job)
    api_module.app.dependency_overrides[api_module.get_job_manager] = lambda: manager
    try:
        with TestClient(api_module.app) as test_client:
            yield test_client
    finally:
        api_module.app.dependency_overrides.pop(api_module.get_job_manager, None)


@contextmanager
def _failing_chain(refuse: str | None = None) -> Iterator[_FailingStream]:
    """Install, over the running app's chain, one whose stream fails every write.

    Args:
        refuse (str | None): Fail only the lines holding this text instead.

    Yields:
        _FailingStream: The failing stream.
    """
    stream = _FailingStream()
    emitter = AuditEmitter(audit.SERVICE, "0+test", stream=stream, heartbeat_interval_s=0)
    emitter.start()
    auditor = audit.Auditor(emitter)
    audit.install(auditor)
    stream.fail = refuse is None
    stream.refuse = refuse
    try:
        yield stream
    finally:
        audit.uninstall(auditor)
        stream.fail = False
        stream.refuse = None
        auditor.close()


@pytest.fixture
def broken_auditor(client: TestClient) -> Iterator[_FailingStream]:
    """Break the audit stream of the running app.

    Args:
        client (TestClient): The running app.

    Yields:
        _FailingStream: The failing stream.
    """
    with _failing_chain() as stream:
        yield stream


def _events(stream: io.StringIO) -> list[dict[str, Any]]:
    """Parse every audit line written so far.

    Args:
        stream (io.StringIO): The test's audit stream.

    Returns:
        list[dict[str, Any]]: The events.
    """
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def _ops(stream: io.StringIO) -> list[dict[str, Any]]:
    """Return the operation events, without the chains' lifecycle and config events.

    Args:
        stream (io.StringIO): The test's audit stream.

    Returns:
        list[dict[str, Any]]: The events, in order.
    """
    return [
        event for event in _events(stream) if event["action"] != "lifecycle" and event["operation"] != "config.audit"
    ]


def test_the_api_chain_opens_with_its_operating_mode(client: TestClient, audit_stream: io.StringIO) -> None:
    """The lifespan's chain states the identity trust and the purpose rule, in the federation's keys.

    Args:
        client (TestClient): The running app.
        audit_stream (io.StringIO): The test's audit stream.
    """
    configs = [event for event in _events(audit_stream) if event["operation"] == "config.audit"]

    assert len(configs) == 1
    assert configs[0]["service"] == "docint"
    assert configs[0]["actor"] == {"user": None, "via": "system"}
    assert configs[0]["criteria"] == {
        "edge_secret_enforced": False,
        "default_identity_enabled": True,
        "document_retention_days": 365,
        "retention_notice_days": 30,
        "require_purpose": [],
    }


def test_an_unknown_required_action_stops_startup(rag: DummyRAG, monkeypatch: pytest.MonkeyPatch) -> None:
    """A misspelt ``AUDIT_REQUIRE_PURPOSE`` must not quietly switch a requirement off.

    Args:
        rag (DummyRAG): The RAG double.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setenv("AUDIT_REQUIRE_PURPOSE", "qurey")

    with pytest.raises(ValueError, match="AUDIT_REQUIRE_PURPOSE"), TestClient(api_module.app):
        pass


def test_a_chat_turn_records_the_query_and_the_answers_model_call(
    client: TestClient, audit_stream: io.StringIO
) -> None:
    """One ``query`` naming the sources, one ``ai_call`` beneath it; the question only by hash.

    Args:
        client (TestClient): The running app.
        audit_stream (io.StringIO): The test's audit stream.
    """
    question = "What did Alice Example post about the meeting?"
    headers = {**ALICE, "X-Request-ID": "req-0001", PURPOSE_HEADER: encode_purpose("case-0001", "review")}

    response = client.post("/query", json={"question": question, "collection": "alpha"}, headers=headers)

    assert response.status_code == 200
    query, answer = _ops(audit_stream)
    assert (query["action"], query["operation"], query["outcome"]) == ("query", "chat.query", "ok")
    assert query["actor"] == {"user": "alice", "via": "edge"}
    assert query["target"] == {"kind": "chunk", "ids": ["1"], "count": 1, "container": "alpha"}
    assert query["purpose"] == {"case_ref": "case-0001", "reason": "review"}
    assert query["request_id"] == "req-0001"
    assert query["criteria"]["question_sha256"] == sha256_text(question)
    assert (answer["action"], answer["operation"]) == ("ai_call", "chat.answer")
    assert answer["parent_event_id"] == query["event_id"]
    assert answer["recipient"] == "inference:chat"
    assert answer["ai"]["output_sha256"] == sha256_text("answer")
    assert answer["ai"]["sources"] == ["1"]
    assert "Alice Example" not in audit_stream.getvalue()


def test_a_streamed_turn_is_recorded_with_its_final_frame(client: TestClient, audit_stream: io.StringIO) -> None:
    """The streamed turn records the same two events, and the sources still arrive.

    Args:
        client (TestClient): The running app.
        audit_stream (io.StringIO): The test's audit stream.
    """
    response = client.post("/stream_query", json={"question": "Who wrote it?", "collection": "alpha"}, headers=ALICE)

    frames = _sse_frames(response.text)
    assert any("sources" in frame for frame in frames)
    assert [(event["action"], event["operation"]) for event in _ops(audit_stream)] == [
        ("query", "chat.query"),
        ("ai_call", "chat.answer"),
    ]


def test_a_failed_audit_write_ends_the_stream_without_its_sources(
    client: TestClient, broken_auditor: _FailingStream
) -> None:
    """Fail closed on a stream: the generic error frame instead of the sources.

    Args:
        client (TestClient): The running app.
        broken_auditor (_FailingStream): The failing audit stream.
    """
    response = client.post("/stream_query", json={"question": "Who wrote it?", "collection": "alpha"}, headers=ALICE)

    frames = _sse_frames(response.text)
    assert frames[-1] == {"error": "Internal server error", "code": "generation_failed"}
    assert not any("sources" in frame for frame in frames)


@pytest.mark.parametrize(
    ("method", "url", "body", "data_key"),
    [
        ("POST", "/query", {"question": "Who wrote it?", "collection": "alpha"}, "answer"),
        ("GET", "/collections/documents?collection=alpha", None, "documents"),
        ("GET", "/collections/alpha/export/documents.csv", None, "file_hash"),
    ],
)
def test_a_failed_audit_write_returns_no_data(rag: DummyRAG, method: str, url: str, body: Any, data_key: str) -> None:
    """Fail closed on a request: the generic 500, never the data.

    The client hands back the 500 the generic handler sends rather than
    re-raising the exception it logs, as a production server does.

    Args:
        rag (DummyRAG): The RAG double.
        method (str): The request method.
        url (str): The route.
        body (Any): The JSON body, if any.
        data_key (str): A key the successful response would carry.
    """
    rag.documents = [{"file_hash": "h1"}]
    with TestClient(api_module.app, raise_server_exceptions=False) as client, _failing_chain():
        response = client.request(method, url, json=body, headers=ALICE)

    assert response.status_code == 500
    assert data_key not in response.text


def test_a_malformed_purpose_header_is_refused_and_records_nothing(
    client: TestClient, audit_stream: io.StringIO
) -> None:
    """A repeated purpose field is ambiguous: 400, and no operation was started.

    Args:
        client (TestClient): The running app.
        audit_stream (io.StringIO): The test's audit stream.
    """
    headers = {**ALICE, PURPOSE_HEADER: "case_ref=case-0001&case_ref=case-0002"}

    response = client.get("/collections/documents", params={"collection": "alpha"}, headers=headers)

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid request."}
    assert _ops(audit_stream) == []


def test_a_required_purpose_is_enforced_per_action(
    rag: DummyRAG, monkeypatch: pytest.MonkeyPatch, audit_stream: io.StringIO
) -> None:
    """Required for ``query``: 428 and one denied event without a purpose; other actions unaffected.

    Args:
        rag (DummyRAG): The RAG double.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        audit_stream (io.StringIO): The test's audit stream.
    """
    monkeypatch.setenv("AUDIT_REQUIRE_PURPOSE", "query")
    with TestClient(api_module.app) as client:
        refused = client.get("/collections/documents", params={"collection": "alpha"}, headers=ALICE)
        denied = _ops(audit_stream)
        stated = client.get(
            "/collections/documents",
            params={"collection": "alpha"},
            headers={**ALICE, PURPOSE_HEADER: encode_purpose("case-0001", None)},
        )
        unaffected = client.delete("/sessions/s1/scope", headers=ALICE)

    assert refused.status_code == 428
    assert "detail" in refused.json()
    assert len(denied) == 1
    assert (denied[0]["action"], denied[0]["operation"], denied[0]["outcome"]) == (
        "query",
        "documents.list",
        "denied",
    )
    assert "target" not in denied[0]
    assert denied[0]["criteria"] == {"collection": "alpha"}
    assert stated.status_code == 200
    assert unaffected.status_code != 428


def test_no_purpose_is_required_by_default(client: TestClient) -> None:
    """The code default is empty: nothing asks for a purpose unless the knob is set.

    Args:
        client (TestClient): The running app.
    """
    assert client.get("/collections/documents", params={"collection": "alpha"}, headers=ALICE).status_code == 200


def test_the_config_advertises_the_purpose_rule_in_force(
    rag: DummyRAG, monkeypatch: pytest.MonkeyPatch, audit_stream: io.StringIO
) -> None:
    """``GET /config`` names the required actions, sorted, so the SPA can insist on a purpose.

    It is read on the SPA's first load, before anyone signs in, so it needs no
    principal; and it is configuration, so it records nothing.

    Args:
        rag (DummyRAG): The RAG double.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        audit_stream (io.StringIO): The test's audit stream.
    """
    monkeypatch.setenv("AUDIT_REQUIRE_PURPOSE", "query, Disclose")
    monkeypatch.delenv("DOCINT_DEFAULT_IDENTITY")
    with TestClient(api_module.app) as client:
        response = client.get("/config")

    assert response.status_code == 200
    assert response.json()["require_purpose"] == ["disclose", "query"]
    assert _ops(audit_stream) == []


def test_the_config_advertises_no_purpose_rule_by_default(client: TestClient) -> None:
    """With ``AUDIT_REQUIRE_PURPOSE`` unset the list is empty, and the purpose bar stays optional.

    Args:
        client (TestClient): The running app.
    """
    assert client.get("/config").json()["require_purpose"] == []


def test_a_deployment_requires_a_purpose_for_queries_and_disclosures() -> None:
    """Compose requires a purpose for ``query`` and ``disclose`` unless the variable is set, even to empty.

    ``${VAR-default}`` substitutes only for an unset variable, so
    ``AUDIT_REQUIRE_PURPOSE=`` in ``.env`` switches the requirement off;
    ``${VAR:-default}`` would bring the default back for an empty value and
    leave no way to require nothing. The dev overlay must not override it.
    """
    compose = (REPO_ROOT / "docker" / "compose.yaml").read_text(encoding="utf-8")
    overlay = (REPO_ROOT / "docker" / "compose.override.yaml").read_text(encoding="utf-8")

    assert compose.count("AUDIT_REQUIRE_PURPOSE: ${AUDIT_REQUIRE_PURPOSE-query,disclose}") == 1
    assert "${AUDIT_REQUIRE_PURPOSE:-" not in compose
    assert "AUDIT_REQUIRE_PURPOSE" not in overlay


@pytest.mark.parametrize(
    ("method", "url", "body"),
    [
        ("GET", "/collections/documents?collection=alpha", None),
        ("POST", "/search", {"question": "Alice Example", "collection": "alpha"}),
        ("POST", "/stream_query", {"question": "Who wrote it?", "collection": "alpha"}),
        ("GET", "/sessions/list?collection=alpha", None),
        ("GET", "/sources/thumbnail?collection=alpha&image_id=i1", None),
        ("GET", "/collections/alpha/export/documents.csv", None),
        ("GET", "/sessions/s1/sources.zip", None),
    ],
)
def test_the_deployment_default_refuses_queries_and_disclosures_without_a_purpose(
    rag: DummyRAG, monkeypatch: pytest.MonkeyPatch, audit_stream: io.StringIO, method: str, url: str, body: Any
) -> None:
    """Under ``query,disclose`` every read and every download the SPA makes needs the bar's purpose.

    Args:
        rag (DummyRAG): The RAG double.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        audit_stream (io.StringIO): The test's audit stream.
        method (str): The request method.
        url (str): The route.
        body (Any): The JSON body, if any.
    """
    monkeypatch.setenv("AUDIT_REQUIRE_PURPOSE", "query,disclose")
    with TestClient(api_module.app) as client:
        refused = client.request(method, url, json=body, headers=ALICE)

    assert refused.status_code == 428
    assert [event["outcome"] for event in _ops(audit_stream)] == ["denied"]


@pytest.mark.parametrize(
    ("method", "url", "body"),
    [
        ("POST", "/ingest/finalize", {"collection": "alpha"}),
        ("DELETE", "/sessions/s1/scope", None),
        ("POST", "/reports", {"title": "Case", "collection_name": "alpha"}),
        ("DELETE", "/sessions/s1", None),
        ("POST", "/translate", {"text": "Hallo"}),
    ],
)
def test_the_deployment_default_lets_other_work_through_without_a_purpose(
    rag: DummyRAG, monkeypatch: pytest.MonkeyPatch, method: str, url: str, body: Any
) -> None:
    """Collection, alteration, erasure and model calls are not refused for want of a purpose.

    They still carry one when it is stated; ``query,disclose`` only makes it
    mandatory for reads and disclosures. A queued run lands on a manager of
    the test's own whose runner does nothing, so none is left behind on the
    application's.

    Args:
        rag (DummyRAG): The RAG double.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        method (str): The request method.
        url (str): The route.
        body (Any): The JSON body, if any.
    """
    monkeypatch.setenv("AUDIT_REQUIRE_PURPOSE", "query,disclose")
    manager = IngestJobManager(runner=lambda _state, _push: {})
    api_module.app.dependency_overrides[api_module.get_job_manager] = lambda: manager
    try:
        with TestClient(api_module.app, raise_server_exceptions=False) as client:
            response = client.request(method, url, json=body, headers=ALICE)
    finally:
        api_module.app.dependency_overrides.pop(api_module.get_job_manager, None)

    assert response.status_code != 428


def test_requiring_ai_call_gates_a_chat_turn(
    rag: DummyRAG, monkeypatch: pytest.MonkeyPatch, audit_stream: io.StringIO
) -> None:
    """A chat turn's model call counts: requiring ``ai_call`` refuses a turn without a purpose.

    Args:
        rag (DummyRAG): The RAG double.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        audit_stream (io.StringIO): The test's audit stream.
    """
    monkeypatch.setenv("AUDIT_REQUIRE_PURPOSE", "ai_call")
    with TestClient(api_module.app) as client:
        response = client.post("/query", json={"question": "Who wrote it?", "collection": "alpha"}, headers=ALICE)

    assert response.status_code == 428
    assert [(event["action"], event["operation"], event["outcome"]) for event in _ops(audit_stream)] == [
        ("query", "chat.query", "denied")
    ]


def test_requiring_combine_gates_both_ingest_routes(
    rag: DummyRAG, monkeypatch: pytest.MonkeyPatch, audit_stream: io.StringIO
) -> None:
    """A run may join a social export's postings, so requiring ``combine`` refuses either ingest route.

    Args:
        rag (DummyRAG): The RAG double.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        audit_stream (io.StringIO): The test's audit stream.
    """
    monkeypatch.setenv("AUDIT_REQUIRE_PURPOSE", "combine")
    with TestClient(api_module.app) as client:
        direct = client.post("/ingest", json={"collection": "alpha"}, headers=ALICE)
        queued = client.post("/ingest/finalize", json={"collection": "alpha"}, headers=ALICE)

    assert (direct.status_code, queued.status_code) == (428, 428)
    assert [(event["action"], event["operation"], event["outcome"]) for event in _ops(audit_stream)] == [
        ("collect", "ingest.run", "denied"),
        ("collect", "ingest.run", "denied"),
    ]


def test_an_admin_in_another_namespace_acts_on_that_owners_behalf(
    client: TestClient, audit_stream: io.StringIO
) -> None:
    """``?owner=`` from an admin is recorded with ``on_behalf_of``.

    Args:
        client (TestClient): The running app.
        audit_stream (io.StringIO): The test's audit stream.
    """
    response = client.get("/collections/documents", params={"collection": "alpha", "owner": "bob"}, headers=ADMIN)

    assert response.status_code == 200
    assert _ops(audit_stream)[0]["actor"] == {"user": "root", "via": "edge", "on_behalf_of": "bob"}


def test_a_non_admin_naming_another_owner_is_refused_and_recorded(
    client: TestClient, audit_stream: io.StringIO
) -> None:
    """The same request from a non-admin: 403, and a denied event naming whom it tried to act for.

    Args:
        client (TestClient): The running app.
        audit_stream (io.StringIO): The test's audit stream.
    """
    response = client.get("/collections/documents", params={"collection": "alpha", "owner": "bob"}, headers=ALICE)

    assert response.status_code == 403
    (denied,) = _ops(audit_stream)
    assert (denied["operation"], denied["outcome"]) == ("documents.list", "denied")
    assert denied["actor"] == {"user": "alice", "via": "edge", "on_behalf_of": "bob"}
    assert denied["criteria"] == {"collection": "alpha"}
    assert "target" not in denied


def test_the_dev_default_identity_acts_as_the_system(client: TestClient, audit_stream: io.StringIO) -> None:
    """A request with no identity header, under the dev fallback, is attributed to ``via: system``.

    Args:
        client (TestClient): The running app.
        audit_stream (io.StringIO): The test's audit stream.
    """
    assert client.get("/collections/documents", params={"collection": "alpha"}).status_code == 200

    assert _ops(audit_stream)[0]["actor"] == {"user": "test-operator", "via": "system"}


def test_an_unauthenticated_request_records_nothing(
    client: TestClient, audit_stream: io.StringIO, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No actor, no event: a 401 must not let anyone fill the trail.

    Args:
        client (TestClient): The running app.
        audit_stream (io.StringIO): The test's audit stream.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.delenv("DOCINT_DEFAULT_IDENTITY")

    assert client.get("/collections/documents", params={"collection": "alpha"}).status_code == 401
    assert _ops(audit_stream) == []


def test_a_server_error_is_recorded_as_failed_without_its_text(
    client: TestClient, rag: DummyRAG, audit_stream: io.StringIO, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 500 settles the operation as ``error`` with its parameters; the exception's text stays out.

    Args:
        client (TestClient): The running app.
        rag (DummyRAG): The RAG double.
        audit_stream (io.StringIO): The test's audit stream.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """

    def _broken() -> list[dict[str, Any]]:
        raise RuntimeError("secret detail from the store")

    monkeypatch.setattr(rag, "list_documents", _broken)

    assert client.get("/collections/documents", params={"collection": "alpha"}, headers=ALICE).status_code == 500
    (failed,) = _ops(audit_stream)
    assert (failed["operation"], failed["outcome"]) == ("documents.list", "error")
    assert failed["criteria"] == {"limit": 0, "page": False}
    assert "secret detail" not in audit_stream.getvalue()


def test_a_long_listing_is_recorded_in_full(client: TestClient, rag: DummyRAG, audit_stream: io.StringIO) -> None:
    """450 documents: the primary event plus two continuations, every hash recorded.

    Args:
        client (TestClient): The running app.
        rag (DummyRAG): The RAG double.
        audit_stream (io.StringIO): The test's audit stream.
    """
    rag.documents = [{"file_hash": f"{index:064x}"} for index in range(450)]

    assert client.get("/collections/documents", params={"collection": "alpha"}, headers=ALICE).status_code == 200

    events = _ops(audit_stream)
    assert [event["operation"] for event in events] == ["documents.list", "documents.list.ids", "documents.list.ids"]
    assert events[0]["target"]["count"] == 450
    assert sum(len(event["target"]["ids"]) for event in events) == 450
    assert all(event["parent_event_id"] == events[0]["event_id"] for event in events[1:])


def test_a_keyword_search_records_its_keywords_and_hits(
    client: TestClient, rag: DummyRAG, audit_stream: io.StringIO, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Search terms are the query's criteria; the hits are named by point id.

    Args:
        client (TestClient): The running app.
        rag (DummyRAG): The RAG double.
        audit_stream (io.StringIO): The test's audit stream.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """

    def _search(query: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "status": "ok",
            "hits": [{"id": "p1", "kind": "text", "preview": "..."}, {"id": "p2", "kind": "image"}],
            "total": 2,
            "next_cursor": None,
            "index_status": {},
        }

    monkeypatch.setattr(rag, "search_fulltext", _search)
    payload = {"question": "Alice Example", "collection": "alpha", "field": "author"}

    assert client.post("/search", json=payload, headers=ALICE).status_code == 200

    (event,) = _ops(audit_stream)
    assert (event["action"], event["operation"]) == ("query", "search.fulltext")
    assert event["target"] == {"kind": "chunk", "ids": ["p1", "p2"], "count": 2, "container": "alpha"}
    assert event["criteria"]["keywords"] == parse_keywords("Alice Example")
    assert event["criteria"]["field"] == "author"


def test_an_export_is_a_disclosure_to_a_download(client: TestClient, rag: DummyRAG, audit_stream: io.StringIO) -> None:
    """A CSV export names the documents it carries and its format as the recipient.

    Args:
        client (TestClient): The running app.
        rag (DummyRAG): The RAG double.
        audit_stream (io.StringIO): The test's audit stream.
    """
    rag.documents = [{"file_hash": "h1", "filename": "statement-0001.pdf"}]

    response = client.get("/collections/alpha/export/documents.csv", headers=ALICE)

    assert response.status_code == 200
    (event,) = _ops(audit_stream)
    assert (event["action"], event["operation"], event["recipient"]) == (
        "disclose",
        "export.documents_csv",
        "download:csv",
    )
    assert event["target"] == {"kind": "document", "ids": ["h1"], "count": 1, "container": "alpha"}


def test_a_collection_delete_records_its_cascade(
    client: TestClient, audit_stream: io.StringIO, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Erasure of a collection, with the sessions and reports that went with it.

    Args:
        client (TestClient): The running app.
        audit_stream (io.StringIO): The test's audit stream.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setattr(
        api_module, "_purge_collection", lambda owner, logical, physical: {"sessions_deleted": 1, "reports_deleted": 2}
    )

    assert client.delete("/collections/alpha", headers=ALICE).status_code == 200

    (event,) = _ops(audit_stream)
    assert (event["action"], event["operation"]) == ("delete", "collection.delete")
    assert event["target"] == {"kind": "collection", "ids": ["alpha"], "count": 1, "container": "alpha"}
    assert event["criteria"] == {"sessions_deleted": 1, "reports_deleted": 2}


def test_entity_resolution_is_a_combine_with_its_transfers(
    client: TestClient, rag: DummyRAG, audit_stream: io.StringIO, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolution records the linking, then one ``ai_call`` per model it sent names to.

    Args:
        client (TestClient): The running app.
        rag (DummyRAG): The RAG double.
        audit_stream (io.StringIO): The test's audit stream.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """

    def _resolve(*, progress_callback: Any = None) -> ResolutionSummary:
        count_transfer("embed", "inference:embed", "embed-model", 4)
        count_transfer("tiebreak", "inference:chat", "chat-model", kind="entity")
        return ResolutionSummary(processed=4, minted=2, attached=1, skipped=1, entities_touched=2)

    monkeypatch.setattr(rag, "resolve_entities", _resolve)

    assert (
        client.post("/collections/entities/resolve", params={"collection": "alpha"}, headers=ALICE).status_code == 200
    )

    combine, *transfers = _ops(audit_stream)
    assert (combine["action"], combine["operation"]) == ("combine", "entities.resolve")
    assert combine["target"] == {"kind": "entity", "count": 2, "container": "alpha"}
    assert combine["criteria"] == {"processed": 4, "minted": 2, "attached": 1, "skipped": 1}
    assert [(event["operation"], event["target"]["kind"], event["target"]["count"]) for event in transfers] == [
        ("entities.resolve.embed", "text", 4),
        ("entities.resolve.tiebreak", "entity", 1),
    ]
    assert all(event["parent_event_id"] == combine["event_id"] for event in transfers)


def test_a_translation_is_an_ai_call_by_hash(
    client: TestClient, audit_stream: io.StringIO, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The snippet goes to the chat model: recorded by length and hash, cache hits marked.

    Args:
        client (TestClient): The running app.
        audit_stream (io.StringIO): The test's audit stream.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    calls: list[str] = []

    def _translate(text: str) -> TranslateResult:
        if text not in calls:
            calls.append(text)
            count_transfer("translate", "inference:chat", "chat-model")
        return TranslateResult(ok=True, translation="Hello, Alice", model="chat-model", target_lang="en")

    monkeypatch.setattr(api_module, "translate", _translate)
    snippet = "Hallo, Alice Example"

    for _ in range(2):
        assert client.post("/translate", json={"text": snippet}, headers=ALICE).status_code == 200

    first, second = _ops(audit_stream)
    assert (first["action"], first["operation"], first["recipient"]) == (
        "ai_call",
        "translate.snippet",
        "inference:chat",
    )
    assert first["ai"] == {"model_alias": "chat-model", "output_sha256": sha256_text("Hello, Alice")}
    assert first["criteria"]["text_sha256"] == sha256_text(snippet)
    assert (first["criteria"]["cached"], second["criteria"]["cached"]) == (False, True)
    assert "Alice Example" not in audit_stream.getvalue()


def test_an_upload_records_the_files_it_staged(
    client: TestClient, audit_stream: io.StringIO, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Staging is collection: the files by hash, once the batch is on disk.

    Args:
        client (TestClient): The running app.
        audit_stream (io.StringIO): The test's audit stream.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        tmp_path (Path): The sources root.
    """
    monkeypatch.setattr(api_module, "_resolve_qdrant_src_dir", lambda: tmp_path)
    content = b"synthetic statement text"
    reference = tmp_path / "reference.txt"
    reference.write_bytes(content)

    response = client.post(
        "/ingest/upload",
        data={"collection": "alpha"},
        files=[("files", ("statement-0001.txt", content, "text/plain"))],
        headers=ALICE,
    )

    assert response.status_code == 200
    assert "upload_complete" in response.text
    (event,) = _ops(audit_stream)
    assert (event["action"], event["operation"]) == ("collect", "ingest.upload")
    assert event["target"] == {
        "kind": "document",
        "ids": [compute_file_hash(reference)],
        "count": 1,
        "container": "alpha",
    }
    assert event["criteria"] == {"preprocess": False, "bytes": len(content)}


def test_the_work_an_upload_starts_states_the_uploads_purpose(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A file's upload-time preprocessing (a clip's Nextext round trip) forwards the upload's purpose.

    Args:
        client (TestClient): The running app.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        tmp_path (Path): The sources root.
    """
    monkeypatch.setattr(api_module, "_resolve_qdrant_src_dir", lambda: tmp_path)
    monkeypatch.setenv("INGEST_PREPROCESS_ON_UPLOAD", "true")
    forwarded: list[dict[str, str]] = []
    monkeypatch.setattr(api_module, "submit_file", lambda *_args, **_kwargs: forwarded.append(forward_headers()))
    headers = {**ALICE, "X-Request-ID": "req-0003", PURPOSE_HEADER: encode_purpose("case-0001", "Review")}

    response = client.post(
        "/ingest/upload",
        data={"collection": "alpha"},
        files=[("files", ("clip-0001.mp4", b"synthetic media bytes", "video/mp4"))],
        headers=headers,
    )

    assert response.status_code == 200
    assert forwarded == [{"X-Audit-Purpose": "case_ref=case-0001&reason=Review", "X-Request-ID": "req-0003"}]


def _fake_ingest(hashes: tuple[str, ...]) -> Any:
    """Build an ``ingest_docs`` stand-in that calls two models and ingests ``hashes``.

    Args:
        hashes (tuple[str, ...]): The file hashes the run reports.

    Returns:
        Any: The stand-in.
    """

    def _ingest(
        collection: str,
        data_dir: Path,
        hybrid: bool | None = None,
        progress_callback: Any = None,
        *,
        ner: bool | None = None,
        hate_speech: bool | None = None,
    ) -> IngestStats:
        for _ in range(2):
            count_transfer("embed", "inference:embed", "embed-model", 12)
        for _ in range(3):
            count_transfer("ner", "inference:ner", "gliner")
        return IngestStats(files_processed=2, files_skipped=0, files_failed=0, docs=2, nodes=24, file_hashes=hashes)

    return _ingest


def test_an_ingest_run_records_what_it_collected_and_what_it_sent(
    client: TestClient, audit_stream: io.StringIO, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The job records with the finalize request's context: the files, then one ``ai_call`` per stage.

    Upload-time model calls (the staged meter) are counted for this run.

    Args:
        client (TestClient): The running app.
        audit_stream (io.StringIO): The test's audit stream.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        tmp_path (Path): The sources root.
    """
    monkeypatch.setenv("TOPICS_ENABLED", "false")
    monkeypatch.setattr(api_module, "_resolve_qdrant_src_dir", lambda: tmp_path)
    (tmp_path / "alpha").mkdir()
    hashes = ("a" * 64, "b" * 64)
    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", _fake_ingest(hashes))
    staged_meter("alpha").add(TransferKey("caption", "inference:vision", "vision-model", "image"), 4)
    headers = {**ALICE, "X-Request-ID": "req-0002", PURPOSE_HEADER: encode_purpose("case-0001", None)}

    snapshot = run_ingest(client, "alpha", headers=headers, extra={"ner": False, "summary": False, "topics": False})

    assert snapshot["status"] == "completed"
    run, *transfers = _ops(audit_stream)
    assert (run["action"], run["operation"], run["outcome"]) == ("collect", "ingest.run", "ok")
    assert run["actor"] == {"user": "alice", "via": "edge"}
    assert run["request_id"] == "req-0002"
    assert run["purpose"] == {"case_ref": "case-0001"}
    assert run["target"] == {"kind": "document", "ids": list(hashes), "count": 2, "container": "alpha"}
    assert run["criteria"]["files_processed"] == 2
    assert run["criteria"]["source"] == "upload"
    assert [
        (event["operation"], event["target"]["count"], event["criteria"]["requests"], event["recipient"])
        for event in transfers
    ] == [
        ("ingest.caption", 4, 1, "inference:vision"),
        ("ingest.embed", 24, 2, "inference:embed"),
        ("ingest.ner", 3, 3, "inference:ner"),
    ]
    assert all(event["parent_event_id"] == run["event_id"] for event in transfers)
    assert all(event["request_id"] == "req-0002" for event in transfers)
    assert take_staged("alpha") is None


@pytest.mark.parametrize(
    ("purpose", "forwarded"),
    [
        (encode_purpose(None, "Review of the uploads"), "reason=Review+of+the+uploads"),
        (None, "reason=docint+ingest"),
    ],
)
def test_an_ingest_runs_calls_to_nextext_state_the_finalize_purpose(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, purpose: str | None, forwarded: str
) -> None:
    """The job forwards the purpose its finalize request stated, else docint's own reason for asking.

    Args:
        client (TestClient): The running app.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        tmp_path (Path): The sources root.
        purpose (str | None): The finalize request's ``X-Audit-Purpose``.
        forwarded (str): What the run's calls to Nextext state.
    """
    monkeypatch.setenv("TOPICS_ENABLED", "false")
    monkeypatch.setattr(api_module, "_resolve_qdrant_src_dir", lambda: tmp_path)
    (tmp_path / "alpha").mkdir()
    fake = _fake_ingest(("c" * 64,))
    seen: list[dict[str, str]] = []

    def _ingest(*args: Any, **kwargs: Any) -> IngestStats:
        seen.append(forward_headers())
        return fake(*args, **kwargs)

    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", _ingest)
    headers = {**ALICE, "X-Request-ID": "req-0004", **({PURPOSE_HEADER: purpose} if purpose else {})}

    snapshot = run_ingest(client, "alpha", headers=headers, extra={"ner": False, "summary": False, "topics": False})

    assert snapshot["status"] == "completed"
    assert seen == [{"X-Audit-Purpose": forwarded, "X-Request-ID": "req-0004"}]


def test_a_job_whose_audit_write_fails_is_marked_failed(
    client: TestClient, broken_auditor: _FailingStream, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Fail closed in a job: the run does not complete unrecorded.

    Args:
        client (TestClient): The running app.
        broken_auditor (_FailingStream): The failing audit stream.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        tmp_path (Path): The sources root.
    """
    monkeypatch.setenv("TOPICS_ENABLED", "false")
    monkeypatch.setattr(api_module, "_resolve_qdrant_src_dir", lambda: tmp_path)
    (tmp_path / "alpha").mkdir()
    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", _fake_ingest(("c" * 64,)))

    snapshot = run_ingest(client, "alpha", headers=ALICE, extra={"ner": False, "summary": False, "topics": False})

    assert snapshot["status"] == "failed"


def test_the_admin_listing_across_owners_is_an_admin_event(
    client: TestClient, rag: DummyRAG, audit_stream: io.StringIO, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the admin's view of every owner's collections is recorded, as ``admin``.

    Args:
        client (TestClient): The running app.
        rag (DummyRAG): The RAG double.
        audit_stream (io.StringIO): The test's audit stream.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    owners = rag.ensure_collection_owner_manager()
    monkeypatch.setattr(owners, "list_all", lambda: [("alice", "a1"), ("bob", "b1"), ("bob", "b2")], raising=False)

    assert client.get("/collections/list", headers=ALICE).status_code == 200
    assert _ops(audit_stream) == []
    assert client.get("/collections/list", params={"all": "true"}, headers=ADMIN).status_code == 200

    (event,) = _ops(audit_stream)
    assert (event["action"], event["operation"]) == ("admin", "collections.list_all")
    assert event["target"] == {"kind": "collection", "count": 3}
    assert event["criteria"] == {"owners": 2}


def _hand_over(point_ids: tuple[str, ...] = ("1", "2")) -> Disclosure:
    """Record one ``run_python`` call in the turn's ledger, as the tool does before it sends.

    Args:
        point_ids (tuple[str, ...]): The points whose passages the call hands over.

    Returns:
        Disclosure: The call's disclosure handle.
    """
    return record_disclosure(
        point_ids=list(point_ids), passages=len(point_ids), keyword_checks=1, topic_listings=0, data_bytes=2048
    )


class _PythonRAG(DummyRAG):
    """``test_api``'s RAG double whose chat turns make one ``run_python`` call before they answer.

    The call is recorded the way the tool records it, and the purpose it would state to workbench
    is kept in ``forwarded``.
    """

    def __init__(self) -> None:
        """Answer every call with ``JOB``, and let the turn succeed."""
        super().__init__()
        #: The job the call's answer names; ``None`` when no answer came.
        self.job: str | None = JOB
        #: Whether the turn fails after its call.
        self.fail = False
        self.forwarded: list[dict[str, str]] = []

    def _run_python(self) -> None:
        """Make the turn's call: state its purpose, record it, settle it.

        Raises:
            RuntimeError: After the call, when the turn is set to fail.
        """
        self.forwarded.append(forward_headers(system_reason=CHAT_REASON))
        disclosure = _hand_over()
        if self.job is None:
            disclosure.unanswered()
        else:
            disclosure.answered(self.job)
        if self.fail:
            raise RuntimeError("secret detail of a failed turn")

    def chat(self, question: str, **kwargs: Any) -> dict[str, Any]:
        """Call workbench once, then answer as the double does.

        Args:
            question (str): The question.
            **kwargs (Any): The turn's parameters.

        Returns:
            dict[str, Any]: The double's answer.
        """
        self._run_python()
        return super().chat(question, **kwargs)

    def stream_chat(self, question: str, **kwargs: Any) -> Generator[str | dict[str, Any], None, None]:
        """Call workbench once, then stream as the double does.

        Args:
            question (str): The question.
            **kwargs (Any): The turn's parameters.

        Yields:
            str | dict[str, Any]: The double's tokens, then its final payload.
        """
        self._run_python()
        yield from super().stream_chat(question, **kwargs)


@pytest.fixture
def python_rag(rag: DummyRAG, monkeypatch: pytest.MonkeyPatch) -> _PythonRAG:
    """Put the double whose turns call workbench behind the app, in place of ``rag``.

    Args:
        rag (DummyRAG): The plain double it replaces.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.

    Returns:
        _PythonRAG: The double.
    """
    double = _PythonRAG()
    monkeypatch.setattr(api_module, "rag", double)
    return double


_TURN = {"question": "How many posts name the meeting?", "collection": "alpha"}


def test_a_turn_that_ran_python_records_one_disclose_per_call(
    client: TestClient, python_rag: _PythonRAG, audit_stream: io.StringIO
) -> None:
    """The call is a ``disclose`` beneath the turn's ``query``, by reference, before the answer's ``ai_call``.

    Args:
        client (TestClient): The running app.
        python_rag (_PythonRAG): The double whose turn calls workbench.
        audit_stream (io.StringIO): The test's audit stream.
    """
    headers = {**ALICE, "X-Request-ID": "req-0005", PURPOSE_HEADER: encode_purpose("case-0001", "review")}

    assert client.post("/query", json=_TURN, headers=headers).status_code == 200

    query, disclose, answer = _ops(audit_stream)
    assert [(e["action"], e["operation"]) for e in (query, disclose, answer)] == [
        ("query", "chat.query"),
        ("disclose", "chat.workbench"),
        ("ai_call", "chat.answer"),
    ]
    assert disclose["parent_event_id"] == query["event_id"] and disclose["recipient"] == "workbench:run_python"
    assert disclose["outcome"] == "ok"
    assert disclose["target"] == {"kind": "chunk", "ids": ["1", "2"], "count": 2, "container": "alpha"}
    assert {
        k: disclose["criteria"][k]
        for k in ("workbench_job", "passages", "keyword_checks", "topic_listings", "data_bytes")
    } == {"workbench_job": JOB, "passages": 2, "keyword_checks": 1, "topic_listings": 0, "data_bytes": 2048}
    assert "incomplete" not in disclose["criteria"]
    assert (disclose["purpose"], disclose["request_id"]) == (query["purpose"], query["request_id"])
    assert answer["parent_event_id"] == query["event_id"]


def test_a_streamed_turn_discloses_before_its_final_frame(
    client: TestClient, python_rag: _PythonRAG, audit_stream: io.StringIO
) -> None:
    """The streamed turn writes the same events; a disclosure that cannot be written withholds the sources.

    Args:
        client (TestClient): The running app.
        python_rag (_PythonRAG): The double whose turn calls workbench.
        audit_stream (io.StringIO): The test's audit stream.
    """
    frames = _sse_frames(client.post("/stream_query", json=_TURN, headers=ALICE).text)

    assert "sources" in frames[-1]
    query, disclose, answer = _ops(audit_stream)
    assert [(e["action"], e["operation"]) for e in (query, disclose, answer)] == [
        ("query", "chat.query"),
        ("disclose", "chat.workbench"),
        ("ai_call", "chat.answer"),
    ]
    assert disclose["parent_event_id"] == query["event_id"]

    with _failing_chain(refuse='"action":"disclose"'):
        refused = _sse_frames(client.post("/stream_query", json=_TURN, headers=ALICE).text)

    assert refused[-1] == {"error": "Internal server error", "code": "generation_failed"}
    assert not any("sources" in frame for frame in refused)


def test_a_call_that_got_no_answer_is_disclosed_as_an_error(
    client: TestClient, python_rag: _PythonRAG, audit_stream: io.StringIO
) -> None:
    """The data left even though no answer came: an ``error`` disclosure without a job, the turn still ``ok``.

    Args:
        client (TestClient): The running app.
        python_rag (_PythonRAG): The double whose turn calls workbench.
        audit_stream (io.StringIO): The test's audit stream.
    """
    python_rag.job = None

    assert client.post("/query", json=_TURN, headers=ALICE).status_code == 200

    query, disclose, _answer = _ops(audit_stream)
    assert query["outcome"] == "ok"
    assert (disclose["operation"], disclose["outcome"]) == ("chat.workbench", "error")
    assert "workbench_job" not in disclose["criteria"]
    assert "incomplete" not in disclose["criteria"]


def test_a_failed_turn_writes_its_disclosures_beneath_the_failure_event(
    client: TestClient, python_rag: _PythonRAG, audit_stream: io.StringIO
) -> None:
    """A turn that fails after its call: a 500, the ``query`` as failed and the ``disclose`` beneath it.

    Args:
        client (TestClient): The running app.
        python_rag (_PythonRAG): The double whose turn calls workbench.
        audit_stream (io.StringIO): The test's audit stream.
    """
    python_rag.fail = True

    assert client.post("/query", json=_TURN, headers=ALICE).status_code == 500

    query, disclose = _ops(audit_stream)
    assert (query["operation"], query["outcome"]) == ("chat.query", "error")
    assert query["criteria"]["incomplete"] is True
    assert disclose["parent_event_id"] == query["event_id"]
    assert (disclose["operation"], disclose["outcome"]) == ("chat.workbench", "ok")
    assert disclose["criteria"]["workbench_job"] == JOB
    assert "secret detail" not in audit_stream.getvalue()


def test_a_stream_whose_client_went_away_discloses_every_call_sent_and_sends_no_more(
    client: TestClient, python_rag: _PythonRAG, audit_stream: io.StringIO, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The client leaves while a call waits: the call is written at the seal, and the turn's next call is refused.

    Args:
        client (TestClient): The running app.
        python_rag (_PythonRAG): The double whose turn calls workbench.
        audit_stream (io.StringIO): The test's audit stream.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setattr(api_module, "STREAM_DISCONNECT_POLL_INTERVAL_S", 0.05)
    refused: list[DisclosureRefused] = []
    finished = threading.Event()

    def _gone(question: str, **_kwargs: Any) -> Generator[str | dict[str, Any], None, None]:
        """Send one call, wait while the client goes away, then try a second one."""
        try:
            _hand_over()
            ledger = current_ledger()
            deadline = time.monotonic() + 5.0
            while ledger is not None and not ledger.sealed and time.monotonic() < deadline:
                time.sleep(0.01)
            try:
                _hand_over(("3",))
            except DisclosureRefused as exc:
                refused.append(exc)
        finally:
            finished.set()
        yield {"response": "answer", "sources": [{"id": 3}], "session_id": "generated-session"}

    async def _disconnected(_self: StarletteRequest) -> bool:
        """Report the client as gone."""
        return True

    monkeypatch.setattr(python_rag, "stream_chat", _gone)
    monkeypatch.setattr(StarletteRequest, "is_disconnected", _disconnected)

    with client.stream("POST", "/stream_query", json=_TURN, headers=ALICE) as response:
        body = "".join(chunk.decode() for chunk in response.iter_raw())

    assert finished.wait(5.0)
    assert len(refused) == 1
    assert '"sources"' not in body
    query, disclose = _ops(audit_stream)
    assert (query["outcome"], query["criteria"]["incomplete"]) == ("error", True)
    assert disclose["parent_event_id"] == query["event_id"]
    assert (disclose["operation"], disclose["outcome"]) == ("chat.workbench", "error")
    assert disclose["criteria"]["incomplete"] is True
    assert "workbench_job" not in disclose["criteria"]
    assert disclose["target"]["ids"] == ["1", "2"]


@pytest.mark.parametrize("route", ["/query", "/stream_query"])
@pytest.mark.parametrize(
    ("headers", "forwarded"),
    [
        (
            {"X-Request-ID": "req-0006", PURPOSE_HEADER: encode_purpose("case-0001", "review")},
            {"X-Audit-Purpose": "case_ref=case-0001&reason=review", "X-Request-ID": "req-0006"},
        ),
        ({}, {"X-Audit-Purpose": "reason=docint+chat"}),
    ],
)
def test_calls_to_workbench_state_the_turns_purpose(
    client: TestClient, python_rag: _PythonRAG, route: str, headers: dict[str, str], forwarded: dict[str, str]
) -> None:
    """A turn's call to workbench states the turn's purpose and request id, else docint's reason for asking.

    Args:
        client (TestClient): The running app.
        python_rag (_PythonRAG): The double whose turn calls workbench.
        route (str): The chat route.
        headers (dict[str, str]): The turn's purpose and request id headers.
        forwarded (dict[str, str]): What the call to workbench states.
    """
    response = client.post(route, json=_TURN, headers={**ALICE, **headers})

    assert response.status_code == 200
    if route == "/stream_query":
        assert "sources" in _sse_frames(response.text)[-1]
    assert python_rag.forwarded == [forwarded]


def test_a_client_error_still_records_nothing(
    client: TestClient, python_rag: _PythonRAG, audit_stream: io.StringIO, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A session pinned to another collection is a 409 before any call, and the trail stays empty.

    Args:
        client (TestClient): The running app.
        python_rag (_PythonRAG): The double whose turn calls workbench.
        audit_stream (io.StringIO): The test's audit stream.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """

    def _pinned(session_id: str | None = None, owner: str | None = None) -> str:
        raise SessionCollectionMismatchError(session_id="session-0001", pinned="beta", requested="alpha")

    monkeypatch.setattr(python_rag, "start_session", _pinned)

    response = client.post("/query", json={**_TURN, "session_id": "session-0001"}, headers=ALICE)

    assert response.status_code == 409
    assert python_rag.forwarded == []
    assert _ops(audit_stream) == []


def test_a_disclose_write_that_fails_part_way_counts_what_went_unwritten(
    client: TestClient, python_rag: _PythonRAG, monkeypatch: pytest.MonkeyPatch, loguru_caplog: pytest.LogCaptureFixture
) -> None:
    """The turn fails closed at the call it cannot write; the log counts that call and those after it, by count only.

    Args:
        client (TestClient): The running app.
        python_rag (_PythonRAG): The double whose turn calls workbench.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        loguru_caplog (pytest.LogCaptureFixture): Loguru's WARNING and ERROR lines.
    """

    def _three_calls(question: str, **kwargs: Any) -> dict[str, Any]:
        for point_ids in (("1", "2"), ("3",), ("4",)):
            _hand_over(point_ids).answered(JOB)
        return DummyRAG.chat(python_rag, question, **kwargs)

    monkeypatch.setattr(python_rag, "chat", _three_calls)

    with _failing_chain(refuse='"ids":["3"]'):
        response = client.post("/query", json=_TURN, headers=ALICE)

    assert response.status_code == 500
    assert "2 disclosure(s) of audited operation 'chat.query' were not written" in loguru_caplog.messages


def test_a_streamed_turn_that_fails_after_its_call_discloses_beneath_the_failure(
    client: TestClient, python_rag: _PythonRAG, audit_stream: io.StringIO
) -> None:
    """The stream ends with its error frame; the ``query`` is an ``error`` and the call's ``disclose`` sits beneath it.

    Args:
        client (TestClient): The running app.
        python_rag (_PythonRAG): The double whose turn calls workbench.
        audit_stream (io.StringIO): The test's audit stream.
    """
    python_rag.fail = True

    frames = _sse_frames(client.post("/stream_query", json=_TURN, headers=ALICE).text)

    assert frames == [{"error": "Internal server error", "code": "generation_failed"}]
    query, disclose = _ops(audit_stream)
    assert (query["operation"], query["outcome"]) == ("chat.query", "error")
    assert disclose["parent_event_id"] == query["event_id"]
    assert (disclose["operation"], disclose["outcome"]) == ("chat.workbench", "ok")
    assert disclose["criteria"]["workbench_job"] == JOB
    assert "secret detail" not in audit_stream.getvalue()


def test_a_client_gone_during_the_corrective_retry_is_no_ok_turn(
    client: TestClient, rag: DummyRAG, audit_stream: io.StringIO, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The retry's answer reached no one: the turn is a failed, incomplete ``query``, never ``ok`` without sources.

    Args:
        client (TestClient): The running app.
        rag (DummyRAG): The RAG double.
        audit_stream (io.StringIO): The test's audit stream.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setattr(api_module, "STREAM_DISCONNECT_POLL_INTERVAL_S", 0.05)
    retrying, release, finished = threading.Event(), threading.Event(), threading.Event()
    weak = "Evidence insufficient."

    def _stream_chat(question: str, *, retrieval_query: str | None = None, **_kwargs: Any) -> Iterator[Any]:
        """Answer weakly at once; on the retry, wait until the client has gone."""
        final = {"response": weak, "sources": [{"id": 1}], "session_id": "session-1", "turn_idx": 0}
        if retrieval_query is None:
            yield weak
            yield {**final, "retrieval_mode": "rewrite_compact"}
            return
        retrying.set()
        try:
            release.wait(5.0)
        finally:
            finished.set()
        yield {**final, "response": "A grounded answer."}

    async def _gone(_self: StarletteRequest) -> bool:
        """Report the client as gone once the retry has started."""
        return retrying.is_set()

    rejected = {"validation_checked": True, "validation_mismatch": True, "validation_reason": "unsupported"}
    monkeypatch.setattr(rag, "stream_chat", _stream_chat)
    monkeypatch.setattr(api_module, "_validation_payload", lambda **_kwargs: dict(rejected))
    monkeypatch.setattr(api_module, "_reformulated_query", lambda *_args: "reformulated query")
    monkeypatch.setattr(StarletteRequest, "is_disconnected", _gone)

    try:
        frames = _sse_frames(client.post("/stream_query", json=_TURN, headers=ALICE).text)
    finally:
        release.set()

    assert finished.wait(5.0)
    assert frames[-1] == {"retry": {"query": "reformulated query"}}
    (query,) = _ops(audit_stream)
    assert (query["operation"], query["outcome"], query["criteria"]["incomplete"]) == ("chat.query", "error", True)


#: Imports the app with the environment it is given and asks both chat routes, without a purpose, what they answer.
_CHAT_PURPOSE_SCRIPT = textwrap.dedent(
    """
    import io
    import json
    import types

    from fastapi.testclient import TestClient

    import docint.core.api as api
    from docint.core import audit
    from docint.utils.env_cfg import load_audit_env

    audit.open_audit_stream = lambda output: io.StringIO()
    audit.install(audit.open_auditor(load_audit_env(), mode={}, heartbeat_interval_s=0))
    unowned = types.SimpleNamespace(resolve=lambda owner, name: None)
    api.rag = types.SimpleNamespace(ensure_collection_owner_manager=lambda: unowned)
    client = TestClient(api.app, raise_server_exceptions=False)
    body = {"question": "How many posts name the meeting?", "collection": "docs"}
    print(json.dumps([client.post(route, json=body).status_code for route in ("/query", "/stream_query")]))
    """
)


@pytest.mark.parametrize(
    ("key", "statuses"),
    [("x" * 40, [428, 428]), ("", [404, 404])],
    ids=["workbench configured", "workbench not configured"],
)
def test_a_chat_turn_needs_a_purpose_to_disclose_only_while_workbench_is_configured(
    tmp_path: Path, key: str, statuses: list[int]
) -> None:
    """``AUDIT_REQUIRE_PURPOSE=disclose`` refuses either chat route without a purpose only while workbench is set.

    What a chat turn may record is read once, when the app is imported, so each case imports it in a process of
    its own. Without workbench the routes pass the purpose rule and answer 404 for a collection nobody owns.

    Args:
        tmp_path (Path): A working directory without a ``.env``.
        key (str): ``WORKBENCH_API_KEY``; empty leaves workbench unconfigured.
        statuses (list[int]): What ``/query`` and ``/stream_query`` answer.
    """
    env = {
        **os.environ,
        "AUDIT_REQUIRE_PURPOSE": "disclose",
        "WORKBENCH_API_KEY": key,
        "WORKBENCH_API_BASE": "http://workbench-api:8000",
        "DOCINT_DEFAULT_IDENTITY": "user-a",
        "DOCINT_EDGE_SECRET": "",
        "DOCINT_AUTH_HEADER": "X-Auth-User",
    }

    result = subprocess.run(
        [sys.executable, "-c", _CHAT_PURPOSE_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == statuses


def test_reading_an_analysis_is_a_query_of_kind_analysis(
    client: TestClient, rag: DummyRAG, audit_stream: io.StringIO, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reading an analysis is a ``query`` naming it by id and its session; none of its content is recorded.

    An analysis that is not found records nothing, as a client error.

    Args:
        client (TestClient): The running app.
        rag (DummyRAG): The RAG double.
        audit_stream (io.StringIO): The test's audit stream.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setattr(rag.sessions, "get_session_collection", lambda session_id, owner=None: "alpha")
    monkeypatch.setattr(
        rag.sessions,
        "get_analysis",
        lambda session_id, analysis_id, owner: STORED_ANALYSIS if analysis_id == ANALYSIS_ID else None,
    )

    missing = client.get(f"/sessions/session-1/analyses/{'b' * 32}", params={"collection": "alpha"}, headers=ALICE)

    assert missing.status_code == 404
    assert _ops(audit_stream) == []

    read = client.get(f"/sessions/session-1/analyses/{ANALYSIS_ID}", params={"collection": "alpha"}, headers=ALICE)

    assert read.status_code == 200
    (event,) = _ops(audit_stream)
    assert (event["action"], event["operation"], event["outcome"]) == ("query", "session.analysis", "ok")
    assert event["target"] == {"kind": "analysis", "ids": [ANALYSIS_ID], "count": 1, "container": "alpha"}
    assert event["criteria"] == {"session_id": "session-1"}
    trail = audit_stream.getvalue()
    for content in ("Posts per author", "print(2)", "2 authors", "counts.csv", "Who wrote most about the harbour?"):
        assert content not in trail
