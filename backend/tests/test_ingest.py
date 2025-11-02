from pathlib import Path
from types import SimpleNamespace

import pytest

import docint.core.ingest as ingest


def test_get_inputs_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("builtins.input", lambda _: "collection")
    monkeypatch.setattr(ingest, "DATA_PATH", tmp_path)
    (tmp_path).mkdir(exist_ok=True)
    name, path = ingest.get_inputs()
    assert name == "collection"
    assert path == tmp_path


def test_get_inputs_missing_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    monkeypatch.setattr("builtins.input", lambda _: "collection")
    monkeypatch.setattr(ingest, "DATA_PATH", missing)
    with pytest.raises(ValueError):
        ingest.get_inputs()


def test_ingest_docs_invokes_rag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = SimpleNamespace(args=None)

    class DummyRAG:
        def __init__(self, qdrant_collection: str, enable_hybrid: bool) -> None:
            calls.args = (qdrant_collection, enable_hybrid)

        def ingest_docs(self, path: Path) -> None:  # type: ignore[override]
            calls.path = path

    monkeypatch.setattr(ingest, "RAG", DummyRAG)
    data_dir = tmp_path
    ingest.ingest_docs("demo", data_dir, hybrid=False)
    assert calls.args == ("demo", False)
    assert calls.path == data_dir


def test_main_executes_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[str] = []

    def fake_setup() -> None:
        order.append("setup")

    def fake_get() -> tuple[str, Path]:
        order.append("inputs")
        return "demo", Path("/tmp")

    def fake_ingest(*args, **kwargs) -> None:
        order.append("ingest")

    monkeypatch.setattr(ingest, "setup_logging", fake_setup)
    monkeypatch.setattr(ingest, "get_inputs", fake_get)
    monkeypatch.setattr(ingest, "ingest_docs", fake_ingest)

    ingest.main()
    assert order == ["setup", "inputs", "ingest"]
