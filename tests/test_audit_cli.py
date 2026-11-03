"""Tests for the command-line tools' audit chains (``docs/audit.md``).

Each command opens a chain of its own (``via: cli``) on the stream
``audit.open_audit_stream`` returns, which ``conftest.audit_stream`` points at
the test's stream and verifies at teardown. RAG is faked throughout.
"""

import io
import json
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
from typing_extensions import override

import docint.cli.extract as extract_cli
import docint.cli.ingest as ingest_cli
import docint.cli.query as query_cli
import docint.cli.resolve as resolve_cli
import docint.cli.search_index as search_index_cli
import docint.cli.topics as topics_cli
import docint.cli.verify as verify_cli
from docint.core import audit
from docint.core.entities.resolution import ResolutionSummary
from docint.core.rag import IngestStats
from docint.utils.audit_event import AuditWriteError, sha256_text
from docint.utils.audit_meter import count_transfer


class _FailingStream(io.StringIO):
    """A stream that takes ``ok_writes`` writes, then raises like a full disk."""

    def __init__(self, ok_writes: int) -> None:
        """Count down from ``ok_writes``.

        Args:
            ok_writes (int): Writes that succeed before the stream fails.
        """
        super().__init__()
        self.ok_writes = ok_writes

    @override
    def write(self, s: str) -> int:
        """Write while writes remain, then raise.

        Args:
            s (str): The text.

        Returns:
            int: Characters written.

        Raises:
            OSError: Once the writes ran out.
        """
        if self.ok_writes <= 0:
            raise OSError("no space left on device")
        self.ok_writes -= 1
        return super().write(s)


@pytest.fixture(autouse=True)
def _operator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every command as an invented operator stating a case reference.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setenv("AUDIT_CLI_USER", "operator-1")
    monkeypatch.setenv("AUDIT_CASE_REF", "case-0001")
    monkeypatch.delenv("DOCINT_EDGE_SECRET", raising=False)
    monkeypatch.delenv("DOCINT_DEFAULT_IDENTITY", raising=False)


def _chains(stream: io.StringIO) -> dict[str, list[dict[str, Any]]]:
    """Group the command chains on the stream by instance, the test's own chain left out.

    Args:
        stream (io.StringIO): The test's audit stream.

    Returns:
        dict[str, list[dict[str, Any]]]: Each command chain's events, in order.
    """
    own = audit.current()._emitter.instance
    chains: dict[str, list[dict[str, Any]]] = {}
    for line in stream.getvalue().splitlines():
        event = json.loads(line)
        if event["instance"] != own:
            chains.setdefault(event["instance"], []).append(event)
    return chains


def _only_chain(stream: io.StringIO) -> list[dict[str, Any]]:
    """Return the one command chain on the stream.

    Args:
        stream (io.StringIO): The test's audit stream.

    Returns:
        list[dict[str, Any]]: Its events.
    """
    chains = list(_chains(stream).values())
    assert len(chains) == 1
    return chains[0]


def _ops(chain: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop a chain's lifecycle and config events.

    Args:
        chain (list[dict[str, Any]]): The chain's events.

    Returns:
        list[dict[str, Any]]: The operation events.
    """
    return [event for event in chain if event["action"] != "lifecycle" and event["operation"] != "config.audit"]


class _ResolveRAG:
    """RAG stand-in for the resolve command: one model call of each kind."""

    def __init__(self, *, qdrant_collection: str) -> None:
        """Accept any collection name as physical.

        Args:
            qdrant_collection (str): The collection.
        """
        self.qdrant_client = types.SimpleNamespace(collection_exists=lambda collection_name: True)

    def resolve_entities(self, *, progress_callback: object = None) -> ResolutionSummary:
        """Count two transfers and report fixed counts.

        Args:
            progress_callback (object): Ignored.

        Returns:
            ResolutionSummary: The counts.
        """
        count_transfer("embed", "inference:embed", "embed-model", 3)
        count_transfer("tiebreak", "inference:chat", "chat-model", kind="entity")
        return ResolutionSummary(processed=3, minted=1, attached=1, skipped=1, entities_touched=2)

    def unload_models(self) -> None:
        """Nothing to unload."""


def test_a_command_writes_its_own_chain(monkeypatch: pytest.MonkeyPatch, audit_stream: io.StringIO) -> None:
    """``resolve``: start, the mode, the combine and its transfers, shutdown, all as the operator.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        audit_stream (io.StringIO): The test's audit stream.
    """
    monkeypatch.setattr(resolve_cli, "RAG", _ResolveRAG)

    resolve_cli.resolve_entities("docs")

    chain = _only_chain(audit_stream)
    assert [(event["action"], event["operation"]) for event in chain] == [
        ("lifecycle", "start"),
        ("admin", "config.audit"),
        ("combine", "entities.resolve"),
        ("ai_call", "entities.resolve.embed"),
        ("ai_call", "entities.resolve.tiebreak"),
        ("lifecycle", "shutdown"),
    ]
    assert chain[1]["criteria"] == {
        "edge_secret_enforced": False,
        "default_identity_enabled": False,
        "require_purpose": [],
        "command": "resolve",
    }
    combine = chain[2]
    assert combine["actor"] == {"user": "operator-1", "via": "cli"}
    assert combine["purpose"] == {"case_ref": "case-0001"}
    assert combine["target"] == {"kind": "entity", "count": 2, "container": "docs"}
    assert "request_id" not in combine
    assert all(event["parent_event_id"] == combine["event_id"] for event in chain[3:5])


def test_a_command_needing_a_purpose_is_refused_without_one(
    monkeypatch: pytest.MonkeyPatch, audit_stream: io.StringIO
) -> None:
    """Exit code 2 and one denied event; the work never starts.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        audit_stream (io.StringIO): The test's audit stream.
    """
    monkeypatch.setenv("AUDIT_REQUIRE_PURPOSE", "combine")
    monkeypatch.delenv("AUDIT_CASE_REF")
    started: list[str] = []
    monkeypatch.setattr(resolve_cli, "RAG", lambda **kwargs: started.append("rag"))

    with pytest.raises(SystemExit) as exc_info:
        resolve_cli.resolve_entities("docs")

    assert exc_info.value.code == 2
    assert started == []
    (denied,) = _ops(_only_chain(audit_stream))
    assert (denied["action"], denied["operation"], denied["outcome"]) == ("combine", "entities.resolve", "denied")
    assert denied["actor"] == {"user": "operator-1", "via": "cli"}


def test_a_required_purpose_is_satisfied_by_the_environment(
    monkeypatch: pytest.MonkeyPatch, audit_stream: io.StringIO
) -> None:
    """With ``AUDIT_CASE_REF`` stated, the same command runs.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        audit_stream (io.StringIO): The test's audit stream.
    """
    monkeypatch.setenv("AUDIT_REQUIRE_PURPOSE", "combine, ai_call")
    monkeypatch.setattr(resolve_cli, "RAG", _ResolveRAG)

    resolve_cli.resolve_entities("docs")

    assert [event["outcome"] for event in _ops(_only_chain(audit_stream))] == ["ok", "ok", "ok"]


def test_a_malformed_purpose_is_refused_before_any_chain(
    monkeypatch: pytest.MonkeyPatch, audit_stream: io.StringIO
) -> None:
    """A control character in the purpose: exit code 2, nothing written.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        audit_stream (io.StringIO): The test's audit stream.
    """
    monkeypatch.setenv("AUDIT_CASE_REF", "case-0001\nforged line")
    monkeypatch.setattr(resolve_cli, "RAG", _ResolveRAG)

    with pytest.raises(SystemExit) as exc_info:
        resolve_cli.resolve_entities("docs")

    assert exc_info.value.code == 2
    assert _chains(audit_stream) == {}


def test_a_failed_audit_write_fails_the_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """The run's own event cannot be written: the command fails instead of finishing unrecorded.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    stream = _FailingStream(ok_writes=2)  # the chain's start and its config event
    monkeypatch.setattr(audit, "open_audit_stream", lambda output: stream)
    monkeypatch.setattr(resolve_cli, "RAG", _ResolveRAG)

    with pytest.raises(AuditWriteError):
        resolve_cli.resolve_entities("docs")

    stream.ok_writes = 1_000


@dataclass
class _Unit:
    """An extract unit stand-in."""

    kind: str
    key: str


def test_an_extract_is_a_disclosure_to_a_file(
    monkeypatch: pytest.MonkeyPatch, audit_stream: io.StringIO, tmp_path: Path
) -> None:
    """``extract``: the units it wrote, by kind, to ``file:extract``.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        audit_stream (io.StringIO): The test's audit stream.
        tmp_path (Path): Where the archive goes.
    """
    rag = types.SimpleNamespace(
        qdrant_client=types.SimpleNamespace(collection_exists=lambda collection_name: True),
        _image_collection_name=lambda name: f"{name}_images",
        unload_models=lambda: None,
    )
    monkeypatch.setattr(extract_cli, "RAG", lambda **kwargs: rag)
    monkeypatch.setattr(extract_cli, "scroll_collection", lambda *args, **kwargs: ([], []))
    monkeypatch.setattr(
        extract_cli, "partition", lambda main, images: [_Unit("document", "h1"), _Unit("posting", "u1")]
    )
    monkeypatch.setattr(
        extract_cli,
        "build_bundle",
        lambda units, **kwargs: types.SimpleNamespace(zip_bytes=b"zip", counts={"documents": 1, "postings": 1}),
    )

    path = extract_cli.build_extract(
        "docs", target=None, out_dir=tmp_path, with_pdf=False, reference_number="case-0001", operator="Alice Example"
    )

    assert path.read_bytes() == b"zip"
    (event,) = _ops(_only_chain(audit_stream))
    assert (event["action"], event["operation"], event["recipient"]) == ("disclose", "extract.build", "file:extract")
    assert event["target"] == {"kind": "mixed", "ids": ["document:h1", "posting:u1"], "count": 2, "container": "docs"}
    assert event["criteria"]["counts"] == {"documents": 1, "postings": 1}


def test_topics_status_is_a_query(monkeypatch: pytest.MonkeyPatch, audit_stream: io.StringIO) -> None:
    """``topics status`` shows labels and keywords: a query naming the topics.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        audit_stream (io.StringIO): The test's audit stream.
    """
    overview = {
        "run_id": "run-1",
        "coverage": {"assigned": 3, "total": 4, "unassigned": 1, "pending": 0},
        "topics": [{"id": 0, "size": 2, "label": "Meetings", "keywords": [("meeting", 0.9)]}],
    }
    rag = types.SimpleNamespace(
        qdrant_client=types.SimpleNamespace(collection_exists=lambda collection_name: True),
        topics_overview=lambda physical: overview,
        unload_models=lambda: None,
    )
    monkeypatch.setattr(topics_cli, "RAG", lambda **kwargs: rag)

    assert topics_cli.run("status", "docs") == 0

    (event,) = _ops(_only_chain(audit_stream))
    assert (event["action"], event["operation"]) == ("query", "topics.overview")
    assert event["target"] == {"kind": "topic", "ids": ["0"], "count": 1, "container": "docs"}
    assert event["criteria"] == {"run_id": "run-1"}


def test_search_index_all_records_every_collection_on_one_chain(
    monkeypatch: pytest.MonkeyPatch, audit_stream: io.StringIO
) -> None:
    """A bulk backport is one chain with one ``modify`` per collection.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        audit_stream (io.StringIO): The test's audit stream.
    """

    class _IndexRAG:
        """RAG stand-in: two physical collections, no image companions."""

        _extract_indexable_text = staticmethod(lambda payload: "")

        def __init__(self, **kwargs: Any) -> None:
            self.qdrant_client = types.SimpleNamespace(
                collection_exists=lambda collection_name: not collection_name.endswith("_images")
            )

        def list_collections(self) -> list[str]:
            return ["u1__alpha", "u2__beta"]

        def unload_models(self) -> None:
            """Nothing to unload."""

    monkeypatch.setattr(search_index_cli, "RAG", _IndexRAG)
    monkeypatch.setattr(search_index_cli, "ensure_search_index", lambda c, name: True)
    monkeypatch.setattr(search_index_cli, "ensure_field_indexes", lambda c, name: True)
    monkeypatch.setattr(
        search_index_cli,
        "backfill_search_text",
        lambda c, name, **kwargs: types.SimpleNamespace(scanned=3, written=2, skipped=1, empty=0),
    )

    search_index_cli.build_all_search_indexes()

    events = _ops(_only_chain(audit_stream))
    assert [(event["action"], event["operation"], event["target"]["container"]) for event in events] == [
        ("modify", "search_index.build", "u1__alpha"),
        ("modify", "search_index.build", "u2__beta"),
    ]
    assert events[0]["target"] == {"kind": "chunk", "count": 2, "container": "u1__alpha"}
    assert events[0]["criteria"] == {"scanned": 3, "written": 2, "skipped": 1, "empty": 0}


def test_verify_records_only_a_repair(monkeypatch: pytest.MonkeyPatch, audit_stream: io.StringIO) -> None:
    """A check reads counts and ids and is not audited; a repair is a ``delete`` of the orphans.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        audit_stream (io.StringIO): The test's audit stream.
    """
    report: dict[str, Any] = {
        "collection": "docs",
        "qdrant_count": 2,
        "kv_count": 3,
        "kv_orphans": ["n3"],
        "qdrant_orphans": [],
        "expected_coarse_only": [],
        "missing_parent_ids": [],
        "repaired_ids": [],
    }

    def _verify(collection: str, *, repair: bool) -> dict[str, Any]:
        return {**report, "repaired_ids": ["n3"] if repair else []}

    monkeypatch.setattr(verify_cli, "RAG", lambda **kwargs: types.SimpleNamespace(verify_collection=_verify))

    verify_cli.main(["-c", "docs", "--json"])
    assert _chains(audit_stream) == {}

    verify_cli.main(["-c", "docs", "--repair", "--json"])
    (event,) = _ops(_only_chain(audit_stream))
    assert (event["action"], event["operation"]) == ("delete", "verify.repair")
    assert event["target"] == {"kind": "node", "ids": ["n3"], "count": 1, "container": "docs"}


def test_an_ingest_run_needs_a_purpose_where_combine_does(
    monkeypatch: pytest.MonkeyPatch, audit_stream: io.StringIO
) -> None:
    """A run may join a social export's postings: requiring ``combine`` refuses it before anything is asked.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        audit_stream (io.StringIO): The test's audit stream.
    """
    monkeypatch.setenv("AUDIT_REQUIRE_PURPOSE", "combine")
    monkeypatch.delenv("AUDIT_CASE_REF")
    asked: list[str] = []

    def _ask() -> str:
        asked.append("collection")
        return "docs"

    monkeypatch.setattr(ingest_cli, "get_collection", _ask)

    with pytest.raises(SystemExit) as exc_info:
        ingest_cli.main()

    assert exc_info.value.code == 2
    assert asked == []
    (denied,) = _ops(_only_chain(audit_stream))
    assert (denied["action"], denied["operation"], denied["outcome"]) == ("collect", "ingest.run", "denied")


def test_an_ingest_run_records_the_files_and_the_model_calls(
    monkeypatch: pytest.MonkeyPatch, audit_stream: io.StringIO
) -> None:
    """``ingest``: the collected files by hash, then one ``ai_call`` per stage.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        audit_stream (io.StringIO): The test's audit stream.
    """

    def _ingest(collection: str, data_dir: Path, *args: Any, **kwargs: Any) -> IngestStats:
        count_transfer("embed", "inference:embed", "embed-model", 5)
        return IngestStats(files_processed=1, files_skipped=0, files_failed=0, docs=1, nodes=5, file_hashes=("h1",))

    monkeypatch.setattr(ingest_cli, "get_collection", lambda: "docs")
    monkeypatch.setattr(ingest_cli, "ingest_docs", _ingest)

    ingest_cli.main()

    run, transfer = _ops(_only_chain(audit_stream))
    assert (run["action"], run["operation"]) == ("collect", "ingest.run")
    assert run["target"] == {"kind": "document", "ids": ["h1"], "count": 1, "container": "docs"}
    assert run["criteria"]["source"] == "data_dir"
    assert (transfer["operation"], transfer["target"]["count"], transfer["parent_event_id"]) == (
        "ingest.embed",
        5,
        run["event_id"],
    )


def test_a_query_run_records_the_question_by_hash_and_its_result_file(
    monkeypatch: pytest.MonkeyPatch, audit_stream: io.StringIO, tmp_path: Path
) -> None:
    """``query``: the retrieval, the answer's model call and the file it writes, never the text.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        audit_stream (io.StringIO): The test's audit stream.
        tmp_path (Path): Where results go.
    """
    question = "What did Alice Example post?"
    rag = types.SimpleNamespace(
        qdrant_collection="docs",
        text_model_id="chat-model",
        run_query=lambda query, expanded_query=None: {"response": "An answer.", "sources": [{"id": "p1"}]},
    )
    monkeypatch.setattr(query_cli, "_store_text_output", lambda filename, data, output_path=None: None)

    query_cli.run_query(cast(Any, rag), question, index=1, output_path=tmp_path)

    query, answer, written = _ops(_only_chain(audit_stream))
    assert (query["action"], query["operation"]) == ("query", "chat.query")
    assert query["criteria"]["question_sha256"] == sha256_text(question)
    assert (answer["action"], answer["operation"], answer["recipient"]) == ("ai_call", "chat.answer", "inference:chat")
    assert answer["ai"] == {"model_alias": "chat-model", "output_sha256": sha256_text("An answer."), "sources": ["p1"]}
    assert (written["action"], written["recipient"]) == ("disclose", "file:query")
    assert "Alice Example" not in audit_stream.getvalue()
