"""Per-request enrichment toggles, auto-resolve, and the ingest-defaults endpoint.

The env config stays the deployment default; explicit per-request flags
override it for one ingest run only.
"""

from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from conftest import run_ingest
from fastapi.testclient import TestClient

import docint.cli.ingest as ingest_cli
import docint.core.api as api_module
import docint.core.ingest.ingestion_pipeline as pipeline_module
from docint.core.entities.resolution import ResolutionSummary
from docint.core.ingest.ingestion_pipeline import DocumentIngestionPipeline
from docint.core.jobs import IngestJobState


@pytest.fixture(autouse=True)
def _default_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a default authenticated identity for every request in this module."""
    monkeypatch.delenv("DOCINT_AUTH_HEADER", raising=False)
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "test-operator")


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Create a TestClient against the real app.

    Entered as a context manager so a single portal (and its background
    event-loop thread) stays alive for the whole test: ingest jobs run as a
    detached ``asyncio`` task meant to outlive the request that queued them,
    and a bare, non-context-managed ``TestClient`` opens a brand-new
    throwaway event loop per call — orphaning that task the instant the
    queuing request returns.
    """
    with TestClient(api_module.app, raise_server_exceptions=False) as client:
        yield client


def _pipeline(tmp_path: Path, **overrides: Any) -> DocumentIngestionPipeline:
    """Construct a cheap pipeline instance for override tests.

    Args:
        tmp_path: Directory used as the pipeline data dir.
        **overrides: Enrichment override kwargs under test.

    Returns:
        DocumentIngestionPipeline: The constructed pipeline.
    """
    pipeline = DocumentIngestionPipeline(data_dir=tmp_path, ner_model=None, progress_callback=None, **overrides)
    pipeline.image_ingestion_service = cast(Any, SimpleNamespace())
    return pipeline


# --- pipeline-level overrides -------------------------------------------------


def test_ner_override_disables_env_enabled_ner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """ner_override=False wins over NER_ENABLED=true; no NER client is built."""
    monkeypatch.setenv("NER_ENABLED", "true")
    calls: list[bool] = []
    monkeypatch.setattr(pipeline_module, "build_remote_ner_extractor", lambda: calls.append(True))
    pipeline = _pipeline(tmp_path, ner_override=False)
    assert calls == []
    assert pipeline.entity_extractor is None


def test_ner_override_enables_env_disabled_ner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """ner_override=True wins over NER_ENABLED=false; the NER client is built."""
    monkeypatch.setenv("NER_ENABLED", "false")
    calls: list[bool] = []
    monkeypatch.setattr(pipeline_module, "build_remote_ner_extractor", lambda: calls.append(True) or SimpleNamespace())
    _pipeline(tmp_path, ner_override=True)
    assert calls == [True]


def test_hate_override_wins_over_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """hate_speech_override replaces the env default in both directions."""
    monkeypatch.setenv("ENABLE_HATE_SPEECH_DETECTION", "false")
    assert _pipeline(tmp_path, hate_speech_override=True).hate_speech_enabled is True
    monkeypatch.setenv("ENABLE_HATE_SPEECH_DETECTION", "true")
    assert _pipeline(tmp_path, hate_speech_override=False).hate_speech_enabled is False


def test_no_override_keeps_env_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Absent overrides leave the env-driven behavior untouched."""
    monkeypatch.setenv("NER_ENABLED", "false")
    monkeypatch.setenv("ENABLE_HATE_SPEECH_DETECTION", "false")
    pipeline = _pipeline(tmp_path)
    assert pipeline.entity_extractor is None
    assert pipeline.hate_speech_enabled is False


# --- ingest_docs threading ----------------------------------------------------


def test_ingest_docs_threads_overrides_to_rag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """CLI-level ingest_docs forwards the per-request flags into RAG.ingest_docs."""
    recorded: dict[str, Any] = {}

    class FakeRAG:
        def __init__(self, **kwargs: Any) -> None:
            recorded["init"] = kwargs

        def ingest_docs(self, data_dir: Any, **kwargs: Any) -> None:
            recorded["ingest"] = kwargs

        def unload_models(self) -> None:
            recorded["unloaded"] = True

    monkeypatch.setattr(ingest_cli, "RAG", FakeRAG)
    ingest_cli.ingest_docs("col", tmp_path, ner=False, hate_speech=True)
    assert recorded["ingest"]["ner"] is False
    assert recorded["ingest"]["hate_speech"] is True
    assert recorded["unloaded"] is True


# --- API surface --------------------------------------------------------------


def test_ingest_defaults_endpoint_reflects_env(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """GET /config/ingest-defaults mirrors the deployment env defaults."""
    monkeypatch.setenv("NER_ENABLED", "true")
    monkeypatch.setenv("ENABLE_HATE_SPEECH_DETECTION", "false")
    monkeypatch.setenv("SUMMARY_ON_INGEST", "false")
    resp = client.get("/config/ingest-defaults")
    assert resp.status_code == 200
    assert resp.json() == {"ner": True, "hate_speech": False, "summary": False}

    monkeypatch.setenv("SUMMARY_ON_INGEST", "true")
    assert client.get("/config/ingest-defaults").json()["summary"] is True


def test_ingest_finalize_forwards_flags(monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path) -> None:
    """POST /ingest/finalize threads explicit ner/hate_speech flags to ingest_docs.

    ``/ingest/upload`` only stages files now — it never calls ``ingest_docs``.
    The per-request enrichment flags take effect at finalize time, since that
    is what actually queues the ingest job.
    """
    monkeypatch.setattr(api_module, "_resolve_qdrant_src_dir", lambda: tmp_path)
    recorded: dict[str, Any] = {}

    def fake_ingest(
        collection: str, path: Path, hybrid: bool = True, progress_callback: Any = None, **kwargs: Any
    ) -> None:
        recorded.update(kwargs)

    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", fake_ingest)
    staged = client.post(
        "/ingest/upload",
        data={"collection": "flags-col"},
        files={"files": ("a.txt", b"hello", "text/plain")},
    )
    assert staged.status_code == 200

    snapshot = run_ingest(client, "flags-col", extra={"ner": False, "hate_speech": True})

    assert snapshot["status"] == "completed"
    assert recorded.get("ner") is False
    assert recorded.get("hate_speech") is True


def test_ingest_finalize_does_not_force_hybrid_when_unspecified(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path
) -> None:
    """POST /ingest/finalize with no explicit ``hybrid`` must not force it on.

    Regression test for the critical defect where every ingest caller (this
    endpoint's ``IngestIn.hybrid`` default plus its ``... if ... is not None
    else True`` coercions) forced ``hybrid=True`` downstream regardless of
    ``resolve_enable_hybrid()``'s derived default. On a non-vLLM provider
    with no ``SPARSE_API_BASE`` set, that forced every ingest to probe a
    sparse endpoint (``/pooling``) that was never configured, failing every
    ingest job. An unspecified ``hybrid`` must now propagate as ``None`` all
    the way to ``ingest_docs`` so the RAG engine's own resolved default
    decides; an explicit ``hybrid`` must still win.
    """
    monkeypatch.setattr(api_module, "_resolve_qdrant_src_dir", lambda: tmp_path)
    recorded: dict[str, Any] = {}

    def fake_ingest(
        collection: str, path: Path, hybrid: bool | None = None, progress_callback: Any = None, **kwargs: Any
    ) -> None:
        recorded["hybrid"] = hybrid

    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", fake_ingest)

    snapshot = _stage_then_finalize(client, "hybrid-unset-col")

    assert snapshot["status"] == "completed"
    assert recorded["hybrid"] is None


def _stage_then_finalize(
    client: TestClient, collection: str, finalize_body: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Stage one file, then finalize and wait for the job to finish.

    Args:
        client: The API test client.
        collection: Logical collection name for the run.
        finalize_body: Extra ``IngestIn`` fields for the finalize call.

    Returns:
        dict[str, Any]: The job's terminal snapshot.
    """
    resp = client.post(
        "/ingest/upload",
        data={"collection": collection},
        files={"files": ("a.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 200
    return run_ingest(client, collection, extra=finalize_body)


def test_ingest_finalize_auto_resolves_when_ner_active(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path
) -> None:
    """Resolution runs automatically whenever the run's effective NER is on."""
    monkeypatch.setattr(api_module, "_resolve_qdrant_src_dir", lambda: tmp_path)
    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", lambda *a, **k: None)
    monkeypatch.setenv("NER_ENABLED", "false")
    summary = ResolutionSummary(processed=3, minted=1, attached=2, skipped=0, entities_touched=2)
    calls: list[bool] = []

    def fake_resolve(self: Any, *, progress_callback: Any = None) -> ResolutionSummary:
        calls.append(True)
        return summary

    monkeypatch.setattr(type(api_module.rag), "resolve_entities", fake_resolve)

    snapshot = _stage_then_finalize(client, "auto-col", {"ner": True})

    assert snapshot["status"] == "completed"
    assert calls == [True]
    assert snapshot["resolution"] is not None
    assert snapshot["resolution"]["minted"] == 1


def test_ingest_finalize_env_ner_default_triggers_auto_resolve(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path
) -> None:
    """With no per-request flag, NER_ENABLED=true alone triggers resolution."""
    monkeypatch.setattr(api_module, "_resolve_qdrant_src_dir", lambda: tmp_path)
    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", lambda *a, **k: None)
    monkeypatch.setenv("NER_ENABLED", "true")
    calls: list[bool] = []
    monkeypatch.setattr(
        type(api_module.rag),
        "resolve_entities",
        lambda self, *, progress_callback=None: (
            calls.append(True),
            ResolutionSummary(processed=0, minted=0, attached=0, skipped=0, entities_touched=0),
        )[1],
    )
    _stage_then_finalize(client, "env-auto-col")
    assert calls == [True]


def test_ingest_finalize_env_kill_switch_disables_auto_resolve(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path
) -> None:
    """RES_AUTO_RESOLVE=false suppresses auto-resolution even with NER on."""
    monkeypatch.setattr(api_module, "_resolve_qdrant_src_dir", lambda: tmp_path)
    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", lambda *a, **k: None)
    monkeypatch.setenv("NER_ENABLED", "true")
    monkeypatch.setenv("RES_AUTO_RESOLVE", "false")
    calls: list[bool] = []
    monkeypatch.setattr(
        type(api_module.rag),
        "resolve_entities",
        lambda self, *, progress_callback=None: calls.append(True),
    )
    _stage_then_finalize(client, "kill-col")
    assert calls == []


def test_ingest_finalize_without_resolve_skips_resolution(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path
) -> None:
    """A run with NER off (explicitly and via env) never touches resolution."""
    monkeypatch.setattr(api_module, "_resolve_qdrant_src_dir", lambda: tmp_path)
    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", lambda *a, **k: None)
    calls: list[bool] = []
    monkeypatch.setattr(
        type(api_module.rag),
        "resolve_entities",
        lambda self, *, progress_callback=None: calls.append(True),
    )

    monkeypatch.setenv("NER_ENABLED", "false")
    snapshot = _stage_then_finalize(client, "no-auto-col", {"ner": False})
    assert snapshot["status"] == "completed"
    assert calls == []


# --- the summary stage's per-request override ---------------------------------


def _summary_job_state(tmp_path: Path, summary: bool | None) -> Any:
    """Build an ingest job state whose only interesting field is ``summary``.

    Args:
        tmp_path: Directory standing in for the staged batch.
        summary: The per-request summary override under test.

    Returns:
        IngestJobState: A state the ingest runner can execute directly.
    """
    return IngestJobState(
        job_id="job-1",
        owner="test-operator",
        logical_name="sum-col",
        physical="u000000000000__sum-col",
        batch_dir=tmp_path,
        summary=summary,
    )


def _run_job_recording_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, on_ingest: str, summary: bool | None
) -> tuple[list[bool], list[str]]:
    """Run the ingest job runner and record whether it summarized.

    Args:
        monkeypatch: Fixture used to stub the pipeline and the summary build.
        tmp_path: Staged batch directory.
        on_ingest: Value for the ``SUMMARY_ON_INGEST`` deployment default.
        summary: The per-request override carried by the job.

    Returns:
        tuple[list[bool], list[str]]: Summary-build calls, and the messages
        pushed to the client.
    """
    monkeypatch.setenv("SUMMARY_ON_INGEST", on_ingest)
    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", lambda *a, **k: None)
    builds: list[bool] = []
    monkeypatch.setattr(
        type(api_module.rag),
        "build_tree_summary",
        lambda self, *, progress=None: builds.append(True),
    )
    messages: list[str] = []
    api_module._run_ingest_job(
        _summary_job_state(tmp_path, summary),
        lambda event, payload: messages.append(str(payload.get("message", ""))),
    )
    return builds, messages


def test_summary_override_skips_the_rebuild(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """summary=False wins over SUMMARY_ON_INGEST=true, and says so.

    The rebuild maps the whole collection, so a run that skips it must not
    look like one whose summary silently vanished.
    """
    builds, messages = _run_job_recording_summary(monkeypatch, tmp_path, on_ingest="true", summary=False)
    assert builds == []
    assert "Collection summary skipped." in messages


def test_summary_override_forces_the_rebuild(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """summary=True wins over SUMMARY_ON_INGEST=false."""
    builds, messages = _run_job_recording_summary(monkeypatch, tmp_path, on_ingest="false", summary=True)
    assert builds == [True]
    assert "Building collection summary..." in messages


def test_absent_summary_override_follows_the_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An unspecified override leaves SUMMARY_ON_INGEST deciding, both ways."""
    assert _run_job_recording_summary(monkeypatch, tmp_path, on_ingest="true", summary=None)[0] == [True]
    assert _run_job_recording_summary(monkeypatch, tmp_path, on_ingest="false", summary=None)[0] == []


def test_ingest_finalize_forwards_summary_flag(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path
) -> None:
    """POST /ingest/finalize threads an explicit summary flag onto the job."""
    monkeypatch.setattr(api_module, "_resolve_qdrant_src_dir", lambda: tmp_path)
    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", lambda *a, **k: None)
    recorded: list[bool | None] = []
    original = api_module._run_ingest_job
    monkeypatch.setattr(
        api_module,
        "_run_ingest_job",
        lambda state, push: (recorded.append(state.summary), original(state, push))[1],
    )

    assert _stage_then_finalize(client, "sum-flag-col", {"summary": False})["status"] == "completed"
    assert recorded == [False]


def test_ingest_finalize_leaves_summary_unspecified_by_default(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path
) -> None:
    """No summary field in the request must reach the job as ``None``.

    ``None`` is what keeps ``SUMMARY_ON_INGEST`` in charge; a client that
    says nothing must never be read as having asked for the summary off.
    """
    monkeypatch.setattr(api_module, "_resolve_qdrant_src_dir", lambda: tmp_path)
    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", lambda *a, **k: None)
    recorded: list[bool | None] = []
    original = api_module._run_ingest_job
    monkeypatch.setattr(
        api_module,
        "_run_ingest_job",
        lambda state, push: (recorded.append(state.summary), original(state, push))[1],
    )

    assert _stage_then_finalize(client, "sum-default-col")["status"] == "completed"
    assert recorded == [None]
