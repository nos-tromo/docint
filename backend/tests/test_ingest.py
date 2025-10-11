from pathlib import Path

import pytest

import docint.utils.ingest as ingest


def test_get_inputs_uses_env_path(monkeypatch, tmp_path: Path) -> None:
    home_data = tmp_path / "data"
    home_data.mkdir()

    monkeypatch.setattr(ingest, "DATA_PATH", str(home_data), raising=False)
    monkeypatch.setattr("builtins.input", lambda prompt: "demo-collection")

    collection, data_dir = ingest.get_inputs()

    assert collection == "demo-collection"
    assert data_dir == home_data


def test_get_inputs_raises_for_missing_dir(monkeypatch, tmp_path: Path) -> None:
    missing_dir = tmp_path / "missing"
    monkeypatch.setattr(ingest, "DATA_PATH", str(missing_dir), raising=False)
    monkeypatch.setattr("builtins.input", lambda prompt: "demo")

    with pytest.raises(ValueError):
        ingest.get_inputs()


def test_ingest_docs_invokes_rag(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "docs"
    data_dir.mkdir()

    calls: dict[str, Path] = {}

    class DummyRAG:
        def __init__(self, qdrant_collection: str, enable_hybrid: bool = True):
            calls["collection"] = qdrant_collection
            calls["hybrid"] = enable_hybrid
            self.called_with: Path | None = None

        def ingest_docs(self, directory: Path) -> None:
            self.called_with = Path(directory)
            calls["data_dir"] = self.called_with

    monkeypatch.setattr(ingest, "RAG", DummyRAG)

    ingest.ingest_docs("the-col", data_dir, hybrid=False)

    assert calls["collection"] == "the-col"
    assert calls["hybrid"] is False
    assert calls["data_dir"] == data_dir
