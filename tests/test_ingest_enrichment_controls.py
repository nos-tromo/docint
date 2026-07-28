"""Per-request enrichment toggles, auto-resolve, and the ingest-defaults endpoint.

The env config stays the deployment default; explicit per-request flags
override it for one ingest run only.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

import docint.cli.ingest as ingest_cli
import docint.core.api as api_module
import docint.core.ingest.ingestion_pipeline as pipeline_module
from docint.core.entities.resolution import ResolutionSummary
from docint.core.ingest.ingestion_pipeline import DocumentIngestionPipeline


@pytest.fixture(autouse=True)
def _default_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a default authenticated identity for every request in this module."""
    monkeypatch.delenv("DOCINT_AUTH_HEADER", raising=False)
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "test-operator")


@pytest.fixture
def client() -> TestClient:
    """Create a TestClient against the real app."""
    return TestClient(api_module.app, raise_server_exceptions=False)


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
    resp = client.get("/config/ingest-defaults")
    assert resp.status_code == 200
    assert resp.json() == {"ner": True, "hate_speech": False}


def test_ingest_upload_forwards_flags(monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path) -> None:
    """POST /ingest/upload threads explicit ner/hate_speech flags to ingest_docs."""
    monkeypatch.setattr(api_module, "_resolve_qdrant_src_dir", lambda: tmp_path)
    recorded: dict[str, Any] = {}

    def fake_ingest(
        collection: str, path: Path, hybrid: bool = True, progress_callback: Any = None, **kwargs: Any
    ) -> None:
        recorded.update(kwargs)

    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", fake_ingest)
    resp = client.post(
        "/ingest/upload",
        data={"collection": "flags-col", "ner": "false", "hate_speech": "true"},
        files={"files": ("a.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 200
    assert recorded.get("ner") is False
    assert recorded.get("hate_speech") is True


def test_ingest_finalize_auto_resolve_runs_and_reports(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path
) -> None:
    """Finalize with resolve=true runs resolution and reports its summary."""
    monkeypatch.setattr(api_module, "_resolve_qdrant_src_dir", lambda: tmp_path)
    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", lambda *a, **k: None)
    summary = ResolutionSummary(processed=3, minted=1, attached=2, skipped=0, entities_touched=2)
    calls: list[bool] = []

    def fake_resolve(self: Any, *, progress_callback: Any = None) -> ResolutionSummary:
        calls.append(True)
        return summary

    monkeypatch.setattr(type(api_module.rag), "resolve_entities", fake_resolve)

    # Stage a file first so finalize has something to ingest.
    resp = client.post(
        "/ingest/upload",
        data={"collection": "auto-col", "defer_ingest": "true"},
        files={"files": ("a.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 200

    resp = client.post("/ingest/finalize", json={"collection": "auto-col", "resolve": True})
    assert resp.status_code == 200
    body = resp.text
    assert calls == [True]
    assert '"resolution"' in body
    assert '"minted": 1' in body


def test_ingest_finalize_without_resolve_skips_resolution(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, tmp_path: Path
) -> None:
    """Finalize without the flag never touches resolution."""
    monkeypatch.setattr(api_module, "_resolve_qdrant_src_dir", lambda: tmp_path)
    monkeypatch.setattr(api_module.ingest_module, "ingest_docs", lambda *a, **k: None)
    calls: list[bool] = []
    monkeypatch.setattr(
        type(api_module.rag),
        "resolve_entities",
        lambda self, *, progress_callback=None: calls.append(True),
    )

    resp = client.post(
        "/ingest/upload",
        data={"collection": "no-auto-col", "defer_ingest": "true"},
        files={"files": ("a.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 200
    resp = client.post("/ingest/finalize", json={"collection": "no-auto-col"})
    assert resp.status_code == 200
    assert calls == []
