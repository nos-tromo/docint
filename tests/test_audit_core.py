"""Unit tests for docint's audit glue, its transfer meter and its target builders.

The per-test audit chain (``audit_stream`` in ``conftest.py``) is verified at
teardown; tests that need a chain of their own build one on a private
``StringIO`` and verify it themselves.
"""

import io
import json
import os
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from typing_extensions import override

from docint.core import audit
from docint.core.audit_targets import chunk_target, entity_key, entity_target, report_items_target, unit_target
from docint.core.auth.principal import Principal
from docint.utils.audit_event import Actor, AuditEmitter, AuditWriteError, Target, verify_chain
from docint.utils.audit_meter import (
    TransferKey,
    TransferMeter,
    count_transfer,
    in_context,
    metering,
    staged_meter,
    take_staged,
)
from docint.utils.audit_runtime import PurposeError
from docint.utils.env_cfg import AuditConfig, PrincipalConfig, load_audit_cli_env, load_audit_env


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


def _auditor(stream: io.StringIO, require: frozenset[str] = frozenset()) -> audit.Auditor:
    """Start a private chain on ``stream``.

    Args:
        stream (io.StringIO): Where its lines go.
        require (frozenset[str]): Actions that need a stated purpose.

    Returns:
        audit.Auditor: The started chain.
    """
    emitter = AuditEmitter(audit.SERVICE, "0+test", stream=stream, heartbeat_interval_s=0)
    emitter.start()
    return audit.Auditor(emitter, require_purpose=require)


def _events(stream: io.StringIO) -> list[dict[str, Any]]:
    """Parse a stream's audit lines.

    Args:
        stream (io.StringIO): The stream.

    Returns:
        list[dict[str, Any]]: The events.
    """
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def _context() -> audit.AuditContext:
    """Return a request-like context for an invented user.

    Returns:
        audit.AuditContext: Alice, with a stated purpose and a request id.
    """
    return audit.AuditContext(
        actor=Actor("alice", "edge"),
        purpose=audit.stated_purpose("case-0001", None),
        request_id="req-0001",
    )


def test_long_id_lists_are_split_into_continuation_events() -> None:
    """More ids than one event holds: the first chunk on the event, the rest on continuations."""
    stream = io.StringIO()
    auditor = _auditor(stream)
    ids = [f"chunk-{index:04d}" for index in range(450)]

    primary = auditor.emit(
        "query",
        "search.fulltext",
        Actor("alice", "edge"),
        target=Target(kind="chunk", ids=ids, count=450, container="docs"),
        request_id="req-0001",
    )
    auditor.close()

    events = [event for event in _events(stream) if event["action"] == "query"]
    assert [event["operation"] for event in events] == ["search.fulltext", "search.fulltext.ids", "search.fulltext.ids"]
    assert events[0]["target"]["count"] == 450
    assert [len(event["target"]["ids"]) for event in events] == [200, 200, 50]
    assert all("ids_truncated" not in event["target"] for event in events)
    assert all(event["parent_event_id"] == primary["event_id"] for event in events[1:])
    assert all(event["request_id"] == "req-0001" for event in events)
    assert [item for event in events for item in event["target"]["ids"]] == ids
    assert verify_chain(_events(stream)).ok


def test_a_target_without_a_count_counts_its_ids() -> None:
    """``count`` is always set, from the ids when the caller gave none."""
    stream = io.StringIO()
    auditor = _auditor(stream)

    auditor.emit("query", "search.chunk", Actor("alice", "edge"), target=Target(kind="chunk", ids=["c1", "c2"]))

    assert _events(stream)[-1]["target"] == {"kind": "chunk", "ids": ["c1", "c2"], "count": 2}


def test_an_operation_records_once_and_its_children_point_at_it() -> None:
    """The primary event is written once; a child names it; a second record is a bug."""
    stream = io.StringIO()
    op = audit.Operation(_auditor(stream), _context(), "query", "chat.query", container="docs")

    with pytest.raises(RuntimeError):
        op.child("ai_call", "chat.answer")
    event_id = op.record(target=Target(kind="chunk", ids=["c1"], count=1))
    child_id = op.child("ai_call", "chat.answer", recipient="inference:chat")
    with pytest.raises(RuntimeError):
        op.record()
    op.fail()

    events = [event for event in _events(stream) if event["action"] != "lifecycle"]
    assert [event["event_id"] for event in events] == [event_id, child_id]
    assert events[0]["target"]["container"] == "docs"
    assert events[0]["purpose"] == {"case_ref": "case-0001"}
    assert events[1]["parent_event_id"] == event_id
    assert events[1]["request_id"] == "req-0001"


def test_settle_quietly_never_raises_on_a_broken_stream() -> None:
    """The work is failing anyway: a failed error write is logged, not raised."""
    stream = _FailingStream()
    op = audit.Operation(_auditor(stream), _context(), "query", "documents.list")
    stream.fail = True

    audit.settle_quietly(op, criteria={"incomplete": True})

    assert op.settled is True
    assert op.event_id is None
    stream.fail = False
    op.auditor.close()


def test_a_closed_chain_refuses_to_write() -> None:
    """Emitting on a closed chain fails closed with ``AuditTrailClosed``."""
    stream = io.StringIO()
    auditor = _auditor(stream)
    auditor.close()

    with pytest.raises(audit.AuditTrailClosed):
        auditor.emit("query", "documents.list", Actor("alice", "edge"))


def test_a_failed_write_raises_audit_write_error() -> None:
    """A write the stream refuses surfaces as ``AuditWriteError``, one of ``AUDIT_FAILURES``."""
    stream = _FailingStream()
    op = audit.Operation(_auditor(stream), _context(), "query", "documents.list")
    stream.fail = True

    with pytest.raises(audit.AUDIT_FAILURES):
        op.record()
    with pytest.raises(AuditWriteError):
        audit.Operation(op.auditor, _context(), "query", "documents.list").record()
    stream.fail = False
    op.auditor.close()


def test_no_installed_chain_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a chain, audited work cannot start.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setattr(audit, "_installed", [])

    with pytest.raises(audit.AuditTrailClosed):
        audit.current()


def test_installed_chains_stack() -> None:
    """A chain installed over another hands back to it when it ends."""
    below = audit.current()
    stream = io.StringIO()
    above = _auditor(stream)

    audit.install(above)
    try:
        assert audit.current() is above
    finally:
        audit.uninstall(above)
        above.close()

    assert audit.current() is below


def test_metered_records_a_failure_with_what_it_sent() -> None:
    """A failing operation records ``error`` with static criteria, then its transfers."""
    stream = io.StringIO()
    op = audit.Operation(_auditor(stream), _context(), "collect", "ingest.run", container="docs")

    with pytest.raises(RuntimeError, match="secret detail"):
        with audit.metered(op, prefix="ingest", on_error=lambda exc: {"cancelled": True}):
            count_transfer("embed", "inference:embed", "embed-model", 3)
            count_transfer("embed", "inference:embed", "embed-model", 2)
            raise RuntimeError("secret detail")

    events = [event for event in _events(stream) if event["action"] != "lifecycle"]
    assert [(event["action"], event["operation"], event["outcome"]) for event in events] == [
        ("collect", "ingest.run", "error"),
        ("ai_call", "ingest.embed", "ok"),
    ]
    assert events[0]["criteria"] == {"cancelled": True}
    assert events[1]["parent_event_id"] == events[0]["event_id"]
    assert events[1]["target"] == {"kind": "text", "count": 5, "container": "docs"}
    assert events[1]["criteria"] == {"requests": 2}
    assert events[1]["recipient"] == "inference:embed"
    assert events[1]["ai"] == {"model_alias": "embed-model"}
    assert "secret detail" not in stream.getvalue()


def test_metered_writes_nothing_more_after_an_audit_failure() -> None:
    """An audit failure inside the block is not followed by further writes."""
    stream = io.StringIO()
    op = audit.Operation(_auditor(stream), _context(), "collect", "ingest.run")

    with pytest.raises(AuditWriteError):
        with audit.metered(op):
            count_transfer("embed", "inference:embed", "embed-model", 3)
            raise AuditWriteError("audit event could not be written")

    assert op.settled is False
    assert [event["action"] for event in _events(stream)] == ["lifecycle"]


def test_open_auditor_records_the_operating_mode(audit_stream: io.StringIO) -> None:
    """A new chain opens with ``config.audit``, stating the trust and purpose rules and nothing secret.

    Args:
        audit_stream (io.StringIO): The test's audit stream.
    """
    principal_cfg = PrincipalConfig(
        header_name="X-Auth-User",
        default_identity="operator",
        groups_header="X-Auth-Groups",
        admin_group="admins",
        default_groups=None,
        edge_secret="test-edge-secret-0123456789abcdefghij",
    )
    config = AuditConfig(output=None, require_purpose=frozenset({"query", "disclose"}))

    auditor = audit.open_auditor(config, mode=audit.identity_mode(principal_cfg))
    auditor.close()

    events = [event for event in _events(audit_stream) if event["instance"] != audit.current()._emitter.instance]
    assert [(event["action"], event["operation"]) for event in events] == [
        ("lifecycle", "start"),
        ("admin", "config.audit"),
        ("lifecycle", "shutdown"),
    ]
    assert events[1]["actor"] == {"user": None, "via": "system"}
    assert events[1]["criteria"] == {
        "edge_secret_enforced": True,
        "default_identity_enabled": False,
        "require_purpose": ["disclose", "query"],
    }
    assert "test-edge-secret" not in audit_stream.getvalue()


@pytest.mark.parametrize(
    ("secret", "default", "expected"),
    [
        (None, None, (False, False)),
        (None, "operator", (False, True)),
        ("test-edge-secret-0123456789abcdefghij", None, (True, False)),
        ("test-edge-secret-0123456789abcdefghij", "operator", (True, False)),
    ],
)
def test_identity_mode_reports_the_fallback_only_without_the_secret(
    secret: str | None, default: str | None, expected: tuple[bool, bool]
) -> None:
    """``default_identity_enabled`` is true only while the dev fallback is in effect.

    Args:
        secret (str | None): The edge secret.
        default (str | None): The default identity.
        expected (tuple[bool, bool]): ``(edge_secret_enforced, default_identity_enabled)``.
    """
    cfg = PrincipalConfig("X-Auth-User", default, "X-Auth-Groups", "admins", None, edge_secret=secret)

    mode = audit.identity_mode(cfg)

    assert (mode["edge_secret_enforced"], mode["default_identity_enabled"]) == expected


def test_load_audit_env_parses_and_rejects_unknown_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    """A misspelt action stops startup instead of switching a requirement off.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    assert load_audit_env().require_purpose == frozenset()

    monkeypatch.setenv("AUDIT_REQUIRE_PURPOSE", " Query, disclose ,")
    assert load_audit_env().require_purpose == frozenset({"query", "disclose"})

    for value in ("qurey", "lifecycle"):
        monkeypatch.setenv("AUDIT_REQUIRE_PURPOSE", value)
        with pytest.raises(ValueError, match="AUDIT_REQUIRE_PURPOSE"):
            load_audit_env()


def test_load_audit_cli_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The operator and the stated purpose of a command-line run; blanks are not stated.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setenv("AUDIT_CLI_USER", " alice ")
    monkeypatch.setenv("AUDIT_CASE_REF", "")
    monkeypatch.setenv("AUDIT_REASON", " review ")

    cli = load_audit_cli_env()

    assert (cli.user, cli.case_ref, cli.reason) == ("alice", None, "review")

    monkeypatch.setenv("AUDIT_REASON", "line\nbreak")
    with pytest.raises(PurposeError):
        load_audit_cli_env()


def test_an_actor_names_whom_an_admin_acts_for() -> None:
    """The caller is the actor; an admin in another namespace acts on that owner's behalf."""
    assert audit.actor_for(Principal(name="alice")) == Actor("alice", "edge")
    assert audit.actor_for(Principal(name="operator", via="system")) == Actor("operator", "system")
    admin = Principal(name="root", groups=frozenset({"admins"}), is_admin=True, requested_owner="bob")
    assert audit.actor_for(admin) == Actor("root", "edge", on_behalf_of="bob")
    # A non-admin's owner parameter changes nothing it acts in.
    assert audit.actor_for(Principal(name="carol", requested_owner="bob")) == Actor("carol", "edge")


def test_count_transfer_without_a_meter_does_nothing() -> None:
    """A model call outside any audited batch is not counted (and never fails)."""
    count_transfer("embed", "inference:embed", "embed-model", 4)


def test_a_nested_meter_keeps_its_counts_apart() -> None:
    """A sub-operation counts into its own meter only."""
    with metering() as outer:
        count_transfer("embed", "inference:embed", "embed-model", 1)
        with metering() as inner:
            count_transfer("tiebreak", "inference:chat", "chat-model", 2, kind="entity")
        count_transfer("embed", "inference:embed", "embed-model", 0)

    assert [(key.stage, tally.items) for key, tally in outer.snapshot()] == [("embed", 1)]
    assert [(key.stage, key.kind, tally.items) for key, tally in inner.snapshot()] == [("tiebreak", "entity", 2)]


def test_in_context_carries_the_meter_into_a_thread_pool() -> None:
    """Work submitted through ``in_context`` counts; a bare submission does not."""
    with metering() as meter, ThreadPoolExecutor(max_workers=2) as pool:
        pool.submit(in_context(count_transfer), "ner", "inference:ner", "gliner", 5).result()
        pool.submit(count_transfer, "ner", "inference:ner", "gliner", 7).result()

    assert [(key.stage, tally.items, tally.requests) for key, tally in meter.snapshot()] == [("ner", 5, 1)]


def test_staged_meters_are_taken_once() -> None:
    """Upload-time counts wait per collection until a run takes them."""
    staged = staged_meter("zz-staged")
    assert staged_meter("zz-staged") is staged
    staged.add(TransferKey("caption", "inference:vision", "vision-model", "image"), 3)

    taken = take_staged("zz-staged")

    assert taken is staged
    assert take_staged("zz-staged") is None


def test_meters_merge_and_report_in_a_stable_order() -> None:
    """Merged counts add up, and the snapshot is sorted by transfer kind."""
    first, second = TransferMeter(), TransferMeter()
    first.add(TransferKey("ner", "inference:ner", "gliner", "text"), 2)
    second.add(TransferKey("embed", "inference:embed", "embed-model", "text"), 4, requests=2)
    second.add(TransferKey("ner", "inference:ner", "gliner", "text"), 1)

    first.merge(second)

    assert [(key.stage, tally.items, tally.requests) for key, tally in first.snapshot()] == [
        ("embed", 4, 2),
        ("ner", 3, 2),
    ]
    assert bool(first) and not bool(TransferMeter())


def test_chunk_targets_name_rows_by_point_id() -> None:
    """The point id first, then the reader's chunk id; every row counted, every id once."""
    rows = [{"id": "p1", "chunk_id": "c1"}, {"chunk_id": "c2"}, {"id": "p1"}, {"text": "no identity"}]

    target = chunk_target(rows, container="docs")

    assert (target.kind, list(target.ids), target.count, target.container) == ("chunk", ["p1", "c2"], 4, "docs")


def test_entity_keys_follow_the_spa() -> None:
    """``<text>::<type>``, an explicit key winning, and no key without text."""
    assert entity_key({"text": "Alice Example", "type": "person"}) == "Alice Example::person"
    assert entity_key({"text": "Example GmbH"}) == "Example GmbH::Unlabeled"
    assert entity_key({"key": "k-1", "text": "ignored"}) == "k-1"
    assert entity_key({"type": "person"}) is None
    assert entity_target([{"text": "Alice Example", "type": "person"}]).ids == ["Alice Example::person"]


def test_report_items_are_named_by_dedupe_key() -> None:
    """A report's evidence is traced by what each item is."""
    report = {"items": [{"id": 1, "dedupe_key": "entity:c1"}, {"id": 2, "dedupe_key": "chat:s1:2"}]}

    target = report_items_target(report, container="docs")

    assert (target.kind, list(target.ids), target.count) == ("report_item", ["entity:c1", "chat:s1:2"], 2)


@dataclass
class _Unit:
    """An extract unit stand-in."""

    kind: str
    key: str


def test_unit_targets_prefix_kinds_only_when_mixed() -> None:
    """One kind of unit is named by key; several kinds carry their kind."""
    assert unit_target([_Unit("document", "h1"), _Unit("document", "h2")]) == Target(
        kind="document", ids=["h1", "h2"], count=2
    )
    assert unit_target([_Unit("posting", "u1"), _Unit("image", "i1")]) == Target(
        kind="mixed", ids=["posting:u1", "image:i1"], count=2
    )


def test_uvicorn_access_lines_never_reach_stdout(tmp_path: Path) -> None:
    """The backend's stdout carries audit lines only: uvicorn's access log goes to stderr.

    uvicorn configures its access handler on stdout before the app is imported;
    ``init_logger`` (run at app import) must take it over.

    Args:
        tmp_path (Path): A working directory without a ``.env``.
    """
    script = textwrap.dedent(
        """
        import logging
        import logging.config

        import uvicorn.config

        logging.config.dictConfig(uvicorn.config.LOGGING_CONFIG)

        from docint.utils.logger_cfg import init_logger

        init_logger()
        logging.getLogger("uvicorn.access").info(
            '%s - "%s %s HTTP/%s" %d', "10.0.0.1:1234", "GET", "/collections/list", "1.1", 200
        )
        logging.getLogger("uvicorn.error").info("Started server process")
        """
    )
    env = {**os.environ, "LOG_LEVEL": "INFO"}

    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env, cwd=tmp_path, timeout=120, check=False
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert "/collections/list" in result.stderr
