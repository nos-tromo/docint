"""CLI entry point for chat queries and collection-level exports."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from time import time
from typing import Any, Sequence

from loguru import logger

from docint.agents.generation import ResultValidationResponseAgent
from docint.agents.types import RetrievalResult, Turn
from docint.core.rag import RAG
from docint.utils.env_cfg import (
    load_path_env,
    load_response_validation_env,
    set_offline_env,
)
from docint.utils.logger_cfg import init_logger
from docint.utils.reference_metadata import REFERENCE_METADATA_FIELDS

DEFAULT_CHAT_SENTINEL = "__default_chat_queries__"
DEFAULT_ENTITY_LIMIT = 50


def get_col_name() -> str:
    """Prompt the user to enter a collection name.

    Returns:
        str: The entered collection name.
    """
    return input("Enter collection name: ")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the query CLI.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Run batch chat queries and collection analysis exports.",
    )
    parser.add_argument(
        "-c",
        "--collection",
        metavar="NAME",
        help="Use NAME as the collection instead of prompting interactively.",
    )
    parser.add_argument(
        "-q",
        "--query",
        nargs="?",
        const=DEFAULT_CHAT_SENTINEL,
        metavar="PATH",
        help=(
            "Run query prompts from PATH. When provided without PATH, use the default "
            "queries file location. If no query file exists, chat queries are skipped."
        ),
    )
    parser.add_argument(
        "-s",
        "--summary",
        action="store_true",
        help="Generate a collection summary using the same backend flow as the frontend.",
    )
    parser.add_argument(
        "-e",
        "--entities",
        action="store_true",
        help="Export the 50 most frequent entities and mention counts as a text file.",
    )
    parser.add_argument(
        "-h8",
        "--hate-speech",
        dest="hate_speech",
        action="store_true",
        help="Export flagged hate-speech findings using the frontend text format.",
    )
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Run chat, summary, entities, and hate-speech exports together.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv (Sequence[str] | None): Optional explicit argument sequence for tests.

    Returns:
        argparse.Namespace: Parsed CLI namespace.
    """
    return build_parser().parse_args(list(argv) if argv is not None else None)


def rag_pipeline(col_name: str) -> RAG:
    """Initialize a Retrieval-Augmented Generation session.

    Args:
        col_name (str): The collection to query.

    Returns:
        RAG: Initialized RAG instance.
    """
    logger.info("Initializing RAG pipeline...")
    rag = RAG(qdrant_collection=col_name)
    rag.create_index()
    rag.create_query_engine()
    return rag


def load_queries(
    queries_path: Path,
    prompts_path: Path | None = None,
) -> list[str]:
    """Load query strings from a text file.

    Args:
        queries_path (Path): Path to the query text file.
        prompts_path (Path | None): Unused compatibility argument retained for callers/tests.

    Returns:
        list[str]: Non-empty query strings loaded from the file, or an empty list when the
        file does not exist.
    """
    _ = prompts_path
    if not isinstance(queries_path, Path):
        queries_path = Path(queries_path).expanduser()

    if not queries_path.exists():
        logger.info("No queries file found at {}; skipping chat queries.", queries_path)
        return []

    logger.info("Loading queries from {}", queries_path)
    with open(queries_path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _ensure_output_path(output_path: str | Path) -> Path:
    """Ensure the output directory exists and return it.

    Args:
        output_path (str | Path): Output directory path.

    Returns:
        Path: Resolved output directory path.
    """
    if not isinstance(output_path, Path):
        output_path = Path(output_path).expanduser()

    if not output_path.exists():
        logger.info("Creating output directory at {}", output_path)
        output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def _store_output(filename: str, data: dict | list, output_path: str | Path) -> None:
    """Store structured output data to a JSON file.

    Args:
        filename (str): Output filename without extension.
        data (dict | list): Data payload to serialize.
        output_path (str | Path): Directory that will receive the file.
    """
    resolved_output_path = _ensure_output_path(output_path)

    if isinstance(data, dict):
        with open(
            resolved_output_path / f"{filename}.json", "w", encoding="utf-8"
        ) as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
    else:
        serializable = []
        for item in data:
            if hasattr(item, "to_dict"):
                serializable.append(item.to_dict())
            else:
                serializable.append(str(item))

        with open(
            resolved_output_path / f"{filename}.json", "w", encoding="utf-8"
        ) as handle:
            json.dump(serializable, handle, ensure_ascii=False, indent=2)
    logger.info("Results stored in {}", resolved_output_path / f"{filename}.json")


def _store_text_output(filename: str, data: str, output_path: str | Path) -> None:
    """Store plain-text output data to a text file.

    Args:
        filename (str): Output filename without extension.
        data (str): Text payload to write.
        output_path (str | Path): Directory that will receive the file.
    """
    resolved_output_path = _ensure_output_path(output_path)
    target = resolved_output_path / f"{filename}.txt"
    target.write_text(data, encoding="utf-8")
    logger.info("Results stored in {}", target)


def _store_csv_output(
    filename: str, rows: list[dict[str, Any]], output_path: str | Path
) -> None:
    """Store tabular data to a CSV file.

    Args:
        filename (str): Output filename without extension.
        rows (list[dict[str, Any]]): List of dicts; keys of the first row become column headers.
        output_path (str | Path): Directory that will receive the file.
    """
    if not rows:
        return
    resolved_output_path = _ensure_output_path(output_path)
    target = resolved_output_path / f"{filename}.csv"
    fieldnames = list(rows[0].keys())
    with target.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Results stored in {}", target)


def _get_validation_payload(
    rag: RAG,
    *,
    question: str,
    answer: str | None,
    sources: list[dict[str, Any]],
    summary_diagnostics: dict[str, Any] | None = None,
) -> dict[str, bool | str | None]:
    """Validate an answer against sources using the same logic as API/frontend flows.

    Args:
        rag (RAG): Active RAG instance.
        question (str): Question or summarize prompt.
        answer (str | None): Generated answer text.
        sources (list[dict[str, Any]]): Retrieved or summary sources.
        summary_diagnostics (dict[str, Any] | None): Optional summary diagnostics payload.

    Returns:
        dict[str, bool | str | None]: Validation metadata dictionary.
    """
    validation_cfg = load_response_validation_env()
    validation_llm = None
    if getattr(rag, "text_model_id", None):
        try:
            validation_llm = rag.text_model
        except Exception as exc:
            logger.warning("Failed to initialize validation LLM: {}", exc)

    validator = ResultValidationResponseAgent(
        enabled=validation_cfg.enabled,
        llm=validation_llm,
    )
    retrieval = RetrievalResult(
        answer=answer,
        sources=sources,
        summary_diagnostics=summary_diagnostics,
    )
    validated = validator.finalize(retrieval, Turn(user_input=question))
    return {
        "validation_checked": validated.validation_checked,
        "validation_mismatch": validated.validation_mismatch,
        "validation_reason": validated.validation_reason,
    }


def _sanitize_filename_fragment(value: str) -> str:
    """Convert an arbitrary string into a filesystem-friendly filename fragment.

    Args:
        value (str): Raw string value.

    Returns:
        str: Sanitized filename fragment.
    """
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return normalized.strip("_") or "collection"


def _build_run_output_path(base_output_path: Path, *, collection_name: str) -> Path:
    """Build the per-run results directory path.

    Args:
        base_output_path (Path): Root results directory.
        collection_name (str): Active collection name.

    Returns:
        Path: Per-run output directory in the form
        ``{base_output_path}/{unix_timestamp}_{collection_name}``.
    """
    timestamp = str(int(time()))
    collection_fragment = _sanitize_filename_fragment(collection_name)
    return Path(base_output_path).expanduser() / f"{timestamp}_{collection_fragment}"


def _reference_metadata_text_block(
    src: dict[str, Any],
    *,
    include_text: bool = True,
) -> str:
    """Return a multi-line text block for reference metadata.

    Args:
        src (dict[str, Any]): Source dictionary containing optional reference metadata.
        include_text (bool): Whether to include the raw ``text`` field.

    Returns:
        str: Text block suitable for exports, or an empty string.
    """
    raw = src.get("reference_metadata")
    if not isinstance(raw, dict):
        return ""

    lines: list[str] = []
    for key, label in REFERENCE_METADATA_FIELDS.items():
        if not include_text and key == "text":
            continue
        value = raw.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        lines.append(f"- {label}: {text}")
    return "\n".join(lines)


def _build_sources_txt(sources: list[dict[str, Any]]) -> str:
    """Build a text block for retrieved or summary sources.

    Args:
        sources (list[dict[str, Any]]): Source dictionaries attached to a query or summary result.

    Returns:
        str: Formatted text block for source details.
    """
    if not sources:
        return "No sources available.\n"

    lines: list[str] = []
    for index, source in enumerate(sources, start=1):
        filename = str(source.get("filename") or source.get("source_ref") or "Unknown")
        page = source.get("page")
        row = source.get("row")
        chunk_id = str(source.get("chunk_id") or "").strip() or "n/a"
        content = str(
            source.get("chunk_text")
            or source.get("text")
            or source.get("preview_text")
            or ""
        ).strip()
        lines.append(
            f"[{index}] {filename} page={page if page is not None else 'n/a'} "
            f"row={row if row is not None else 'n/a'} chunk_id={chunk_id}"
        )
        metadata_block = _reference_metadata_text_block(
            source,
            include_text=not bool(content),
        )
        if metadata_block:
            lines.append(metadata_block)
        if content:
            lines.append(content)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _build_query_result_txt(query: str, result: dict[str, Any]) -> str:
    """Build the text export for one query result.

    Args:
        query (str): Original user query.
        result (dict[str, Any]): Query result payload.

    Returns:
        str: Text payload for the query export.
    """
    answer = str(result.get("response") or result.get("answer") or "").strip()
    validation_reason = result.get("validation_reason")
    lines = [f"Query: {query}", "", "Answer:", answer or "", ""]
    lines.extend(
        [
            "Validation:",
            f"- checked: {result.get('validation_checked')}",
            f"- mismatch: {result.get('validation_mismatch')}",
            f"- reason: {validation_reason if validation_reason is not None else 'n/a'}",
            "",
        ]
    )

    graph_debug = result.get("graph_debug")
    if isinstance(graph_debug, dict):
        lines.extend(
            [
                "Graph debug:",
                json.dumps(graph_debug, ensure_ascii=False, indent=2),
                "",
            ]
        )

    lines.extend(["Sources:", _build_sources_txt(list(result.get("sources") or []))])
    return "\n".join(lines).strip() + "\n"


def _build_summary_txt(collection: str, payload: dict[str, Any]) -> str:
    """Build the text export for a collection summary.

    Args:
        collection (str): Active collection name.
        payload (dict[str, Any]): Summary payload with validation and diagnostics.

    Returns:
        str: Text payload for the summary export.
    """
    summary = str(payload.get("summary") or payload.get("response") or "").strip()
    validation_reason = payload.get("validation_reason")
    lines = [f"Collection summary: {collection}", "", "Summary:", summary or "", ""]
    lines.extend(
        [
            "Validation:",
            f"- checked: {payload.get('validation_checked')}",
            f"- mismatch: {payload.get('validation_mismatch')}",
            f"- reason: {validation_reason if validation_reason is not None else 'n/a'}",
            "",
        ]
    )

    summary_diagnostics = payload.get("summary_diagnostics")
    if isinstance(summary_diagnostics, dict):
        lines.extend(
            [
                "Summary diagnostics:",
                json.dumps(summary_diagnostics, ensure_ascii=False, indent=2),
                "",
            ]
        )

    lines.extend(["Sources:", _build_sources_txt(list(payload.get("sources") or []))])
    return "\n".join(lines).strip() + "\n"


def _build_entities_txt(top_entities: list[dict[str, Any]], *, collection: str) -> str:
    """Build the entity-frequency export text.

    Args:
        top_entities (list[dict[str, Any]]): Ranked entity rows from collection NER stats.
        collection (str): Active collection name.

    Returns:
        str: Text payload for the entity export.
    """
    lines = [
        f"Top {DEFAULT_ENTITY_LIMIT} entities for collection: {collection}",
        "",
    ]
    if not top_entities:
        lines.append("No entities found in this collection.")
        return "\n".join(lines).strip() + "\n"

    for index, row in enumerate(top_entities[:DEFAULT_ENTITY_LIMIT], start=1):
        entity = str(row.get("text") or "Unknown")
        entity_type = str(row.get("type") or "Unlabeled")
        mentions = int(row.get("mentions", row.get("count", 0)) or 0)
        lines.append(f"{index}. {entity} [{entity_type}] - {mentions}")
    return "\n".join(lines).strip() + "\n"


def _build_entities_csv(
    top_entities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build entity-frequency rows for CSV export.

    Args:
        top_entities (list[dict[str, Any]]): Ranked entity rows from collection NER stats.

    Returns:
        list[dict[str, Any]]: List of dicts with keys: rank, entity, type, mentions.
    """
    rows = []
    for index, row in enumerate(top_entities[:DEFAULT_ENTITY_LIMIT], start=1):
        rows.append(
            {
                "rank": index,
                "entity": str(row.get("text") or "Unknown"),
                "type": str(row.get("type") or "Unlabeled"),
                "mentions": int(row.get("mentions", row.get("count", 0)) or 0),
            }
        )
    return rows


def _build_hate_speech_csv(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build hate-speech finding rows for CSV export.

    Args:
        findings (list[dict[str, Any]]): Flagged hate-speech rows.

    Returns:
        list[dict[str, Any]]: List of dicts covering all finding fields.
    """
    rows = []
    for chunk in findings:
        ref = chunk.get("reference_metadata") or {}
        rows.append(
            {
                "source": chunk.get("source_ref"),
                "page": chunk.get("page"),
                "row": chunk.get("row"),
                "chunk_id": chunk.get("chunk_id"),
                "category": chunk.get("category"),
                "confidence": chunk.get("confidence"),
                "reason": chunk.get("reason"),
                "chunk_text": chunk.get("chunk_text") or chunk.get("text"),
                "network": ref.get("network"),
                "ref_type": ref.get("type"),
                "uuid": ref.get("uuid"),
                "timestamp": ref.get("timestamp"),
                "author": ref.get("author"),
                "author_id": ref.get("author_id"),
                "vanity": ref.get("vanity"),
                "text_id": ref.get("text_id"),
                "parent_text": ref.get("parent_text"),
                "anchor_text": ref.get("anchor_text"),
            }
        )
    return rows


def _build_hate_speech_txt(findings: list[dict[str, Any]]) -> str:
    """Build the hate-speech export text using the frontend format.

    Args:
        findings (list[dict[str, Any]]): Flagged hate-speech rows.

    Returns:
        str: Text payload for the export.
    """
    if not findings:
        return "No hate speech flags detected for this collection.\n"

    lines = ["Flagged hate-speech chunks", ""]
    for idx, chunk in enumerate(findings, start=1):
        location_label = "page" if chunk.get("page") is not None else "row"
        location_value = (
            chunk.get("page")
            if chunk.get("page") is not None
            else chunk.get("row", "n/a")
        )
        lines.append(
            f"[{idx}]\n"
            f"- source: {chunk.get('source_ref')}\n"
            f"- {location_label}: {location_value}\n"
            f"- chunk_id: {chunk.get('chunk_id')}\n"
            f"- category: {chunk.get('category')}\n"
            f"- confidence: {chunk.get('confidence')}"
        )
        lines.append(f"reason: {chunk.get('reason')}")
        metadata_block = _reference_metadata_text_block(chunk)
        if metadata_block:
            lines.append(metadata_block)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def run_query(rag: RAG, query: str, index: int, output_path: str | Path) -> None:
    """Run one query against the RAG instance and store the result.

    Args:
        rag (RAG): Active RAG instance.
        query (str): Query string.
        index (int): Query index for naming.
        output_path (str | Path): Directory where results are stored.
    """
    logger.info("Running query {}: {}", index, query)

    retrieval_query = query
    graph_debug: dict[str, Any] | None = None
    expand_with_debug = getattr(rag, "expand_query_with_graph_with_debug", None)
    if callable(expand_with_debug):
        try:
            expanded, debug_payload = expand_with_debug(query)
            retrieval_query = str(expanded)
            if isinstance(debug_payload, dict):
                graph_debug = debug_payload
        except Exception as exc:
            logger.warning("Graph debug expansion failed in query CLI: {}", exc)

    result = rag.run_query(retrieval_query)
    if graph_debug is not None:
        result["graph_debug"] = graph_debug

    raw_sources = result.get("sources")
    sources: list[dict[str, Any]] = []
    if isinstance(raw_sources, list):
        sources = [src for src in raw_sources if isinstance(src, dict)]

    result.update(
        _get_validation_payload(
            rag,
            question=query,
            answer=str(result.get("response") or result.get("answer") or ""),
            sources=sources,
        )
    )

    timestamp = str(int(time()))
    _store_text_output(
        filename=f"{timestamp}_{index}_result",
        data=_build_query_result_txt(query, result),
        output_path=output_path,
    )


def export_chat_queries(
    rag: RAG,
    *,
    queries_path: Path,
    prompts_path: Path,
    output_path: Path,
) -> None:
    """Run batch chat queries from a text file.

    Args:
        rag (RAG): Active RAG instance.
        queries_path (Path): Query file path.
        prompts_path (Path): Prompts directory, kept for compatibility with load_queries.
        output_path (Path): Results directory.
    """
    queries = load_queries(queries_path=queries_path, prompts_path=prompts_path)
    if not queries:
        logger.info("No chat queries to run.")
        return

    for index, query in enumerate(queries, start=1):
        run_query(rag=rag, query=query, index=index, output_path=output_path)


def export_summary(rag: RAG, *, output_path: Path) -> None:
    """Generate and store a collection summary payload.

    Args:
        rag (RAG): Active RAG instance.
        output_path (Path): Results directory.
    """
    logger.info("Generating collection summary...")
    data = rag.summarize_collection()
    summary = str(data.get("response") or data.get("answer") or "")
    sources = [src for src in data.get("sources", []) if isinstance(src, dict)]
    summary_diagnostics = data.get("summary_diagnostics")
    if not isinstance(summary_diagnostics, dict):
        summary_diagnostics = None
    payload = {
        "summary": summary,
        "sources": sources,
        "summary_diagnostics": summary_diagnostics,
        **_get_validation_payload(
            rag,
            question=str(getattr(rag, "summarize_prompt", "Summarize the collection.")),
            answer=summary,
            sources=sources,
            summary_diagnostics=summary_diagnostics,
        ),
    }
    collection = _sanitize_filename_fragment(str(rag.qdrant_collection or "collection"))
    _store_text_output(
        f"summary_{collection}",
        _build_summary_txt(str(rag.qdrant_collection or collection), payload),
        output_path,
    )


def export_entities(rag: RAG, *, output_path: Path) -> None:
    """Export the top collection entities and their mention counts as text.

    Args:
        rag (RAG): Active RAG instance.
        output_path (Path): Results directory.
    """
    logger.info("Exporting top entities...")
    stats = rag.get_collection_ner_stats(
        top_k=DEFAULT_ENTITY_LIMIT,
        min_mentions=1,
        include_relations=True,
        entity_merge_mode="orthographic",
    )
    top_entities = [
        row for row in list(stats.get("top_entities") or []) if isinstance(row, dict)
    ]
    collection = _sanitize_filename_fragment(str(rag.qdrant_collection or "collection"))
    _store_text_output(
        f"entities_{collection}",
        _build_entities_txt(top_entities, collection=str(rag.qdrant_collection or "")),
        output_path,
    )
    _store_csv_output(
        f"entities_{collection}", _build_entities_csv(top_entities), output_path
    )


def export_hate_speech(rag: RAG, *, output_path: Path) -> None:
    """Export flagged hate-speech findings as text.

    Args:
        rag (RAG): Active RAG instance.
        output_path (Path): Results directory.
    """
    logger.info("Exporting hate-speech findings...")
    findings = [
        row for row in rag.get_collection_hate_speech() if isinstance(row, dict)
    ]
    collection = _sanitize_filename_fragment(str(rag.qdrant_collection or "collection"))
    _store_text_output(
        f"hate_speech_{collection}",
        _build_hate_speech_txt(findings),
        output_path,
    )
    _store_csv_output(
        f"hate_speech_{collection}", _build_hate_speech_csv(findings), output_path
    )


def _should_run_chat(args: argparse.Namespace) -> bool:
    """Return whether chat-query mode is active.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        bool: Whether chat mode should run.
    """
    if args.all or args.query is not None:
        return True
    return not (args.summary or args.entities or args.hate_speech)


def _resolve_chat_queries_path(
    args: argparse.Namespace,
    *,
    default_queries_path: Path,
) -> Path:
    """Resolve the queries file path for chat mode.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.
        default_queries_path (Path): Default configured queries file path.

    Returns:
        Path: Resolved queries file path.
    """
    if args.query in {None, DEFAULT_CHAT_SENTINEL}:
        return default_queries_path
    return Path(str(args.query)).expanduser()


def _resolve_collection_name(args: argparse.Namespace) -> str:
    """Resolve the target collection name from arguments or interactive input.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        str: Selected collection name.
    """
    collection_name = str(getattr(args, "collection", "") or "").strip()
    if collection_name:
        return collection_name
    return get_col_name()


def main(argv: Sequence[str] | None = None) -> None:
    """Run the query CLI workflow.

    Args:
        argv (Sequence[str] | None): Optional explicit argument sequence for tests.
    """
    init_logger()
    set_offline_env()
    args = parse_args(argv)

    col_name = _resolve_collection_name(args)
    rag = rag_pipeline(col_name=col_name)
    path_config = load_path_env()
    run_output_path = _build_run_output_path(
        path_config.results,
        collection_name=col_name,
    )

    try:
        if _should_run_chat(args):
            export_chat_queries(
                rag,
                queries_path=_resolve_chat_queries_path(
                    args, default_queries_path=path_config.queries
                ),
                prompts_path=path_config.prompts,
                output_path=run_output_path,
            )

        if args.all or args.summary:
            export_summary(rag, output_path=run_output_path)

        if args.all or args.entities:
            export_entities(rag, output_path=run_output_path)

        if args.all or args.hate_speech:
            export_hate_speech(rag, output_path=run_output_path)
    finally:
        rag.unload_models()

    logger.info("Query CLI work completed.")


if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parents[2].resolve()))
    main()
