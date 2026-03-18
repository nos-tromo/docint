"""Tests for the retrieval evaluation CLI."""

from pathlib import Path
from typing import Any


import docint.cli.eval as eval_cli


def test_load_eval_queries_supports_plain_text(tmp_path: Path) -> None:
    """Plain-text query files should load as simple query specs."""
    query_file = tmp_path / "queries.txt"
    query_file.write_text("one\n\ntwo\n", encoding="utf-8")

    loaded = eval_cli.load_eval_queries(query_file, tmp_path)

    assert loaded == [{"query": "one"}, {"query": "two"}]


def test_load_eval_queries_supports_json_records(tmp_path: Path) -> None:
    """JSON query files should preserve labels and expectations."""
    query_file = tmp_path / "queries.json"
    query_file.write_text(
        ('[{"query": "one", "label": "Q1", "expected_filenames": ["a.pdf"]}]'),
        encoding="utf-8",
    )

    loaded = eval_cli.load_eval_queries(query_file, tmp_path)

    assert loaded[0]["query"] == "one"
    assert loaded[0]["label"] == "Q1"
    assert loaded[0]["expected_filenames"] == ["a.pdf"]


def test_evaluate_retrieval_compares_modes_and_expectations() -> None:
    """Eval harness should compare multiple retrieval modes and compute hit rates."""

    class DummyRAG:
        def __init__(self) -> None:
            self.enable_hybrid = True
            self.qdrant_collection = "alpha"
            self.calls: list[tuple[str, str]] = []

        def run_query(
            self,
            prompt: str,
            *,
            retrieval_options: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            mode = str(
                (retrieval_options or {}).get("vector_store_query_mode") or "default"
            )
            self.calls.append((prompt, mode))
            return {
                "response": f"{mode}:{prompt}",
                "sources": [
                    {
                        "filename": "a.pdf" if mode == "hybrid" else "b.pdf",
                        "file_hash": "ha" if mode == "hybrid" else "hb",
                        "reference_metadata": {
                            "text_id": "t1" if mode == "hybrid" else "t2"
                        },
                    }
                ],
                "vector_query_mode": mode,
                "retrieval_profile": mode,
                "parent_context_enabled": False,
            }

    rag = DummyRAG()
    report = eval_cli.evaluate_retrieval(
        rag,
        [
            {
                "query": "find alpha",
                "expected_filenames": ["a.pdf"],
                "expected_file_hashes": ["ha"],
                "expected_text_ids": ["t1"],
            }
        ],
        modes=["default", "hybrid"],
    )

    assert report["modes"] == ["default", "hybrid"]
    assert len(report["results"]) == 1
    default_result, hybrid_result = report["results"][0]["results"]
    assert default_result["expectations"]["filename_hit"] is False
    assert hybrid_result["expectations"]["filename_hit"] is True
    assert report["summary"]["hybrid"]["filename_hit_rate"] == 1.0
    assert ("find alpha", "default") in rag.calls
    assert ("find alpha", "hybrid") in rag.calls


def test_load_eval_queries_falls_back_to_summarize_prompt(tmp_path: Path) -> None:
    """Missing query files should fall back to the summarize prompt."""
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "summarize.txt").write_text("Summarize collection.", encoding="utf-8")

    loaded = eval_cli.load_eval_queries(tmp_path / "missing.txt", prompts)

    assert loaded == [{"query": "Summarize collection."}]
