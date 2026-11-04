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
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from conftest import run_ingest
from fastapi.testclient import TestClient
from test_api import DummyRAG, _sse_frames
from typing_extensions import override

import docint.core.api as api_module
from docint.core import audit
from docint.core.entities.resolution import ResolutionSummary
from docint.core.jobs import IngestJobManager
from docint.core.rag import IngestStats
from docint.core.search.fulltext import parse_keywords
from docint.utils.audit_event import AuditEmitter, sha256_text
from docint.utils.audit_meter import TransferKey, count_transfer, staged_meter, take_staged
from docint.utils.audit_runtime import PURPOSE_HEADER, encode_purpose
from docint.utils.hashing import compute_file_hash
from docint.utils.translate_client import TranslateResult

ALICE = {"X-Auth-User": "alice"}
ADMIN = {"X-Auth-User": "root", "X-Auth-Groups": "admins"}


class _FailingStream(io.StringIO):
    """A stream that takes writes until ``fail`` is set, then raises like a full disk."""

    fail = False

    @override
    def write(self, s: str) -> int:
        """Write, or raise once the stream is set to fail.

        Args:
            s (str): The text.

        Returns:
            int: Characters written.

        Raises:
            OSError: Once ``fail`` is set.
        """
        if self.fail:
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
def _failing_chain() -> Iterator[_FailingStream]:
    """Install, over the running app's chain, one whose stream fails every write.

    Yields:
        _FailingStream: The failing stream.
    """
    stream = _FailingStream()
    emitter = AuditEmitter(audit.SERVICE, "0+test", stream=stream, heartbeat_interval_s=0)
    emitter.start()
    auditor = audit.Auditor(emitter)
    audit.install(auditor)
    stream.fail = True
    try:
        yield stream
    finally:
        audit.uninstall(auditor)
        stream.fail = False
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


@dataclass(frozen=True)
class _Clock:
    """A collection's retention clock."""

    owner: str | None
    logical: str
    physical: str
    last_activity_at: datetime | None


def test_the_retention_sweep_records_a_system_delete(
    rag: DummyRAG, audit_stream: io.StringIO, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An expired collection's purge is the system's erasure, naming the owner it belonged to.

    Args:
        rag (DummyRAG): The RAG double.
        audit_stream (io.StringIO): The test's audit stream.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    now = datetime.now(UTC)
    idle = now - timedelta(days=400)
    clock = _Clock("alice", "old-docs", "u000000000000__old-docs", idle)

    class _Owners:
        def get_activity(self, physical: str) -> _Clock:
            return clock

    monkeypatch.setattr(rag, "ensure_collection_owner_manager", lambda: _Owners())
    monkeypatch.setattr(
        api_module, "_purge_collection", lambda owner, logical, physical: {"sessions_deleted": 1, "reports_deleted": 2}
    )

    assert api_module._purge_if_still_expired(clock, 6, idle, now) is True

    (event,) = _ops(audit_stream)
    assert (event["action"], event["operation"], event["outcome"]) == ("delete", "retention.purge", "ok")
    assert event["actor"] == {"user": None, "via": "system"}
    assert event["target"] == {"kind": "collection", "ids": ["old-docs"], "count": 1, "container": "old-docs"}
    assert event["criteria"] == {
        "owner": "alice",
        "window_months": 6,
        "last_activity_at": idle.isoformat(),
        "sessions_deleted": 1,
        "reports_deleted": 2,
    }
    assert "purpose" not in event and "request_id" not in event


def test_a_failed_retention_purge_is_recorded_as_failed(
    rag: DummyRAG, audit_stream: io.StringIO, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A purge that raises is an ``error`` delete, and the sweep sees the failure.

    Args:
        rag (DummyRAG): The RAG double.
        audit_stream (io.StringIO): The test's audit stream.
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    now = datetime.now(UTC)
    idle = now - timedelta(days=400)
    clock = _Clock("alice", "old-docs", "u000000000000__old-docs", idle)

    class _Owners:
        def get_activity(self, physical: str) -> _Clock:
            return clock

    def _purge(owner: str | None, logical: str, physical: str) -> dict[str, int]:
        raise RuntimeError("store offline")

    monkeypatch.setattr(rag, "ensure_collection_owner_manager", lambda: _Owners())
    monkeypatch.setattr(api_module, "_purge_collection", _purge)

    with pytest.raises(RuntimeError):
        api_module._purge_if_still_expired(clock, 6, idle, now)

    (event,) = _ops(audit_stream)
    assert (event["operation"], event["outcome"]) == ("retention.purge", "error")
    assert event["target"]["ids"] == ["old-docs"]
