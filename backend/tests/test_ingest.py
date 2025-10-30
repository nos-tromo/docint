from pathlib import Path

import pytest

import docint.core.ingest as ingest


def test_get_inputs_uses_env_path(monkeypatch, tmp_path: Path) -> None:
    """
    Tests that the environment path is used for data input.

    Args:
        monkeypatch (_type_): The monkeypatch fixture.
        tmp_path (Path): The temporary directory path.
    """
    home_data = tmp_path / "data"
    if not home_data.exists():
        home_data.mkdir()

    monkeypatch.setattr(ingest, "DATA_PATH", str(home_data), raising=False)
    monkeypatch.setattr("builtins.input", lambda prompt: "demo-collection")

    collection, data_dir = ingest.get_inputs()

    assert collection == "demo-collection"
    assert data_dir == home_data


def test_get_inputs_raises_for_missing_dir(monkeypatch, tmp_path: Path) -> None:
    """
    Tests that a ValueError is raised when the data directory is missing.

    Args:
        monkeypatch (_type_): The monkeypatch fixture.
        tmp_path (Path): The temporary directory path.
    """
    missing_dir = tmp_path / "missing"
    monkeypatch.setattr(ingest, "DATA_PATH", str(missing_dir), raising=False)
    monkeypatch.setattr("builtins.input", lambda prompt: "demo")

    with pytest.raises(ValueError):
        ingest.get_inputs()


def test_ingest_docs_invokes_rag(monkeypatch, tmp_path: Path) -> None:
    """
    Tests that the ingest_docs function invokes the RAG model correctly.

    Args:
        monkeypatch (_type_): The monkeypatch fixture.
        tmp_path (Path): The temporary directory path.
    """
    data_dir = tmp_path / "docs"
    if not data_dir.exists():
        data_dir.mkdir()

    calls: dict[str, Path] = {}

    class DummyRAG:
        """
        A dummy Retrieval-Augmented Generation (RAG) model for testing purposes.
        """

        def __init__(self, qdrant_collection: str, enable_hybrid: bool = True) -> None:
            """
            Initializes the DummyRAG model.

            Args:
                qdrant_collection (str): The name of the Qdrant collection.
                enable_hybrid (bool, optional): Whether to enable hybrid search. Defaults to True.
            """
            calls["collection"] = qdrant_collection
            calls["hybrid"] = enable_hybrid
            self.called_with: Path | None = None

        def ingest_docs(self, directory: Path) -> None:
            """
            Ingests documents from the specified directory.

            Args:
                directory (Path): The directory containing documents to ingest.
            """
            self.called_with = Path(directory)
            calls["data_dir"] = self.called_with

    monkeypatch.setattr(ingest, "RAG", DummyRAG)

    ingest.ingest_docs("the-col", data_dir, hybrid=False)

    assert calls["collection"] == "the-col"
    assert calls["hybrid"] is False
    assert calls["data_dir"] == data_dir
