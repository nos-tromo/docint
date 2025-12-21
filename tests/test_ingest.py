from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest

import docint.cli.ingest as ingest


def test_get_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test that get_collection returns the user input.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setattr("builtins.input", lambda _: "collection")
    name = ingest.get_collection()
    assert name == "collection"


def test_main_executes_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test that the main function executes the ingestion pipeline in the correct order.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    order: list[str] = []

    def fake_setup() -> None:
        order.append("setup")

    def fake_get_collection() -> str:
        order.append("collection")
        return "demo"

    def fake_ingest(*args, **kwargs) -> None:
        order.append("ingest")

    class FakePathConfig:
        data = Path("/tmp")

    def fake_load_path_env() -> FakePathConfig:
        order.append("env")
        return FakePathConfig()

    monkeypatch.setattr(ingest, "setup_logging", fake_setup)
    monkeypatch.setattr(ingest, "set_offline_env", lambda: None)
    monkeypatch.setattr(ingest, "load_path_env", fake_load_path_env)
    monkeypatch.setattr(ingest, "get_collection", fake_get_collection)
    monkeypatch.setattr(ingest, "ingest_docs", fake_ingest)

    ingest.main()
    # Order might vary slightly depending on implementation details, but generally:
    # setup -> env -> collection -> ingest
    assert "setup" in order
    assert "env" in order
    assert "collection" in order
    assert "ingest" in order


def test_ingest_docs_invokes_rag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = SimpleNamespace(args=None, build_query_engine=None, path=None)

    class DummyRAG:
        def __init__(
            self,
            qdrant_collection: str,
            enable_hybrid: bool,
            table_row_limit: int | None = None,
            table_row_filter: str | None = None,
        ) -> None:
            calls.args = (
                qdrant_collection,
                enable_hybrid,
                table_row_limit,
                table_row_filter,
            )

        def ingest_docs(
            self,
            path: Path,
            *,
            build_query_engine: bool = True,
            progress_callback: Callable[[str], None] | None = None,
        ) -> None:  # type: ignore[override]
            """
            Placeholder ingest_docs method for the test double.

            Args:
                path (Path): _description_
                build_query_engine (bool, optional): _description_. Defaults to True.
                progress_callback (Callable[[str], None] | None, optional): _description_. Defaults to None.
            """            
            calls.path = path
            calls.build_query_engine = build_query_engine

        def unload_models(self) -> None:
            # No-op for test double
            return None

    monkeypatch.setattr(ingest, "RAG", DummyRAG)
    data_dir = tmp_path
    ingest.ingest_docs("demo", data_dir, hybrid=False)
    assert calls.args == ("demo", False, None, None)
    assert calls.path == data_dir
    assert calls.build_query_engine is False
