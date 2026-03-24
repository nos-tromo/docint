# ruff: noqa: E402

"""CLI for corpus-specific retrieval evaluation across retrieval modes."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter, time
from typing import Any, Protocol

from docint.utils.env_cfg import bootstrap_config, load_path_env, set_offline_env

bootstrap_config(role="worker")

from loguru import logger

from docint.cli.query import _store_output, get_col_name, rag_pipeline
from docint.utils.logging_cfg import setup_logging


class RetrievalEvalRAG(Protocol):
    """Structural type for the retrieval-eval helper."""

    enable_hybrid: bool
    qdrant_collection: str

    def run_query(
        self,
        prompt: str,
        *,
        retrieval_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...


def load_eval_queries(queries_path: Path, prompts_path: Path) -> list[dict[str, Any]]:
    """Load retrieval-eval queries from JSON, JSONL, or plain text.

    Supported formats:
    - ``.json``: list of objects or strings
    - ``.jsonl``: one object or string per line
    - text: one query per non-empty line

    Query objects may include optional expectations such as
    ``expected_filenames``, ``expected_file_hashes``, and ``expected_text_ids``.

    Args:
        queries_path: Path to the queries file.
        prompts_path: Path to the prompts directory (used for default query if no file is found

    Returns:
        List of normalized query specifications, each with a mandatory 'query' key and optional expectation keys
    """
    if queries_path.exists():
        suffix = queries_path.suffix.lower()
        if suffix == ".json":
            data = json.loads(queries_path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                raise ValueError("Eval JSON must contain a list of queries.")
            return [_normalize_query_spec(item) for item in data]
        if suffix == ".jsonl":
            specs: list[dict[str, Any]] = []
            with open(queries_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    specs.append(_normalize_query_spec(json.loads(line)))
            return specs

        with open(queries_path, "r", encoding="utf-8") as handle:
            return [
                _normalize_query_spec(line.strip()) for line in handle if line.strip()
            ]

    logger.info("No queries file found, using default summarize prompt")
    summarize_prompt_path = prompts_path / "summarize.txt"
    with open(summarize_prompt_path, "r", encoding="utf-8") as handle:
        return [_normalize_query_spec(handle.read().strip())]


def _normalize_query_spec(raw: Any) -> dict[str, Any]:
    """Normalize one eval-query record into a plain dictionary.

    Args:
        raw: The raw query record, which may be a string or a dictionary.

    Returns:
        A normalized query specification dictionary with a mandatory 'query' key.
    """
    if isinstance(raw, str):
        return {"query": raw}
    if not isinstance(raw, dict):
        raise ValueError(f"Unsupported eval query record: {type(raw).__name__}")
    query = str(raw.get("query") or raw.get("question") or "").strip()
    if not query:
        raise ValueError("Eval query record requires a non-empty 'query'.")
    normalized = dict(raw)
    normalized["query"] = query
    return normalized


def _match_expectations(
    sources: list[dict[str, Any]],
    query_spec: dict[str, Any],
) -> dict[str, Any]:
    """Compute simple expectation hits for a query result.

    Checks whether any retrieved source matches any of the expected filenames, file hashes, or text IDs specified in the query spec.

    Args:
        sources: The list of retrieved source documents, each as a dictionary with optional 'filename',
            'file_hash', and 'reference_metadata.text_id' fields.
        query_spec: The query specification dictionary, which may include 'expected_filenames',
            'expected_file_hashes', and 'expected_text_ids' as lists of expected values.

    Returns:
        A dictionary containing the sets of expected values and boolean flags indicating whether any hits were found.
    """
    filenames = {
        str(source.get("filename") or "").strip()
        for source in sources
        if str(source.get("filename") or "").strip()
    }
    file_hashes = {
        str(source.get("file_hash") or "").strip()
        for source in sources
        if str(source.get("file_hash") or "").strip()
    }
    text_ids = {
        str((source.get("reference_metadata") or {}).get("text_id") or "").strip()
        for source in sources
        if isinstance(source.get("reference_metadata"), dict)
        and str((source.get("reference_metadata") or {}).get("text_id") or "").strip()
    }

    expected_filenames = {
        str(value).strip() for value in (query_spec.get("expected_filenames") or [])
    }
    expected_file_hashes = {
        str(value).strip() for value in (query_spec.get("expected_file_hashes") or [])
    }
    expected_text_ids = {
        str(value).strip() for value in (query_spec.get("expected_text_ids") or [])
    }

    return {
        "expected_filenames": sorted(expected_filenames),
        "expected_file_hashes": sorted(expected_file_hashes),
        "expected_text_ids": sorted(expected_text_ids),
        "filename_hit": bool(
            expected_filenames and filenames.intersection(expected_filenames)
        ),
        "file_hash_hit": bool(
            expected_file_hashes and file_hashes.intersection(expected_file_hashes)
        ),
        "text_id_hit": bool(
            expected_text_ids and text_ids.intersection(expected_text_ids)
        ),
    }


def evaluate_retrieval(
    rag: RetrievalEvalRAG,
    query_specs: list[dict[str, Any]],
    *,
    modes: list[str] | None = None,
) -> dict[str, Any]:
    """Run the same query set across multiple retrieval modes.

    The output is intended for corpus-specific inspection rather than benchmark
    publication: it preserves answers, sources, timing, and optional expectation hits.

    Args:
        rag: The RAG pipeline instance to use for running queries.
        query_specs: A list of query specifications, each as a dictionary with a mandatory 'query' key and optional expectation keys.
        modes: An optional list of retrieval modes to test. If None, defaults to ["default", "hybrid", "sparse"] if hybrid retrieval is enabled, otherwise ["default"].

    Returns:
        A dictionary containing the collection name, tested modes, query count, a summary of results by mode, and the detailed results for each query and mode.
    """
    selected_modes = modes or (
        ["default", "hybrid", "sparse"] if rag.enable_hybrid else ["default"]
    )
    results: list[dict[str, Any]] = []
    summary_by_mode: dict[str, dict[str, Any]] = {}

    for query_index, query_spec in enumerate(query_specs, start=1):
        query_text = str(query_spec["query"])
        mode_results: list[dict[str, Any]] = []
        for mode in selected_modes:
            started_at = perf_counter()
            payload = rag.run_query(
                query_text,
                retrieval_options={"vector_store_query_mode": mode},
            )
            latency_ms = round((perf_counter() - started_at) * 1000, 2)
            sources = [
                source
                for source in (payload.get("sources") or [])
                if isinstance(source, dict)
            ]
            expectations = _match_expectations(sources, query_spec)
            mode_result = {
                "mode": mode,
                "latency_ms": latency_ms,
                "response": payload.get("response"),
                "vector_query_mode": payload.get("vector_query_mode"),
                "retrieval_profile": payload.get("retrieval_profile"),
                "parent_context_enabled": payload.get("parent_context_enabled"),
                "source_count": len(sources),
                "unique_filenames": sorted(
                    {
                        str(source.get("filename") or "").strip()
                        for source in sources
                        if str(source.get("filename") or "").strip()
                    }
                ),
                "sources": sources,
                "expectations": expectations,
            }
            mode_results.append(mode_result)

            mode_summary = summary_by_mode.setdefault(
                mode,
                {
                    "queries": 0,
                    "latency_ms_total": 0.0,
                    "source_count_total": 0,
                    "filename_hits": 0,
                    "file_hash_hits": 0,
                    "text_id_hits": 0,
                    "expected_queries": 0,
                },
            )
            mode_summary["queries"] += 1
            mode_summary["latency_ms_total"] += latency_ms
            mode_summary["source_count_total"] += len(sources)
            if any(
                expectations[key]
                for key in ("filename_hit", "file_hash_hit", "text_id_hit")
            ):
                mode_summary["expected_queries"] += 1
            if expectations["filename_hit"]:
                mode_summary["filename_hits"] += 1
            if expectations["file_hash_hit"]:
                mode_summary["file_hash_hits"] += 1
            if expectations["text_id_hit"]:
                mode_summary["text_id_hits"] += 1

        results.append(
            {
                "query_index": query_index,
                "query": query_text,
                "label": query_spec.get("label"),
                "results": mode_results,
            }
        )

    summary: dict[str, Any] = {}
    for mode, stats in summary_by_mode.items():
        query_count = max(1, int(stats["queries"]))
        expected_queries = int(stats["expected_queries"])
        summary[mode] = {
            "queries": int(stats["queries"]),
            "avg_latency_ms": round(float(stats["latency_ms_total"]) / query_count, 2),
            "avg_source_count": round(
                float(stats["source_count_total"]) / query_count, 2
            ),
            "filename_hit_rate": round(stats["filename_hits"] / expected_queries, 4)
            if expected_queries
            else None,
            "file_hash_hit_rate": round(stats["file_hash_hits"] / expected_queries, 4)
            if expected_queries
            else None,
            "text_id_hit_rate": round(stats["text_id_hits"] / expected_queries, 4)
            if expected_queries
            else None,
        }

    return {
        "collection": rag.qdrant_collection,
        "modes": selected_modes,
        "query_count": len(query_specs),
        "summary": summary,
        "results": results,
    }


def main() -> None:
    """Run retrieval evaluation for the active collection."""
    setup_logging()
    set_offline_env()
    col_name = get_col_name()
    rag = rag_pipeline(col_name=col_name)
    path_config = load_path_env()
    query_specs = load_eval_queries(
        queries_path=path_config.queries,
        prompts_path=path_config.prompts,
    )
    report = evaluate_retrieval(rag, query_specs)
    timestamp = str(int(time()))
    _store_output(
        filename=f"{timestamp}_retrieval_eval",
        data=report,
        output_path=path_config.results,
    )
    rag.unload_models()
    logger.info("Retrieval evaluation complete.")


if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parents[2].resolve()))
    main()
