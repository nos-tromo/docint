"""CLI entry point for verifying cross-store consistency.

Wraps :meth:`docint.core.rag.RAG.verify_collection` to report (and
optionally repair) drift between the Qdrant vector store and the
SQLite KV docstore.
"""

from __future__ import annotations

import argparse
import json
import sys

from loguru import logger

from docint.core.rag import RAG
from docint.utils.env_cfg import set_offline_env
from docint.utils.logger_cfg import init_logger


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the verify CLI.

    Returns:
        A configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Check that a collection's Qdrant vector store and SQLite KV "
            "docstore are in sync, and optionally repair KV-only orphans."
        ),
    )
    parser.add_argument(
        "-c",
        "--collection",
        required=True,
        help="Qdrant collection name to verify.",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help=(
            "Delete KV-only orphan nodes from the docstore. "
            "Qdrant-only orphans and broken parent references are never "
            "repaired automatically; they require re-ingestion."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the report as a single JSON object on stdout.",
    )
    return parser


def _print_human_report(report: dict) -> None:
    """Print a compact human-readable verification report.

    Args:
        report: The dict returned by :meth:`RAG.verify_collection`.
    """
    sys.stdout.write(f"Collection: {report['collection']}\n")
    sys.stdout.write(f"  qdrant_count:          {report['qdrant_count']}\n")
    sys.stdout.write(f"  kv_count:              {report['kv_count']}\n")
    sys.stdout.write(f"  kv_orphans:            {len(report['kv_orphans'])}\n")
    sys.stdout.write(f"  qdrant_orphans:        {len(report['qdrant_orphans'])}\n")
    sys.stdout.write(
        f"  expected_coarse_only:  {len(report['expected_coarse_only'])}\n"
    )
    sys.stdout.write(
        f"  missing_parent_ids:    {len(report['missing_parent_ids'])}\n"
    )
    sys.stdout.write(f"  repaired_ids:          {len(report['repaired_ids'])}\n")
    if report["kv_orphans"]:
        sys.stdout.write("\nKV orphans (first 20):\n")
        for node_id in report["kv_orphans"][:20]:
            sys.stdout.write(f"  - {node_id}\n")
    if report["qdrant_orphans"]:
        sys.stdout.write("\nQdrant orphans (first 20) — require re-ingestion:\n")
        for node_id in report["qdrant_orphans"][:20]:
            sys.stdout.write(f"  - {node_id}\n")
    if report["missing_parent_ids"]:
        sys.stdout.write("\nMissing parent ids (first 20):\n")
        for parent_id in report["missing_parent_ids"][:20]:
            sys.stdout.write(f"  - {parent_id}\n")


def main(argv: list[str] | None = None) -> int:
    """Run the verify CLI.

    Args:
        argv: Optional argument list (primarily for tests).  When ``None``
            the arguments are read from ``sys.argv``.

    Returns:
        Process exit code — ``0`` when no drift is present, ``1`` when
        any orphans or missing parents were found.
    """
    init_logger()
    set_offline_env()
    parser = build_parser()
    args = parser.parse_args(argv)

    rag = RAG(qdrant_collection=args.collection)
    try:
        report = rag.verify_collection(args.collection, repair=args.repair)
    except Exception as exc:
        logger.error("verify_collection failed: {}", exc)
        return 2

    if args.json:
        sys.stdout.write(json.dumps(report, indent=2) + "\n")
    else:
        _print_human_report(report)

    has_drift = bool(
        report["kv_orphans"]
        or report["qdrant_orphans"]
        or report["missing_parent_ids"]
    )
    return 1 if has_drift else 0


if __name__ == "__main__":
    raise SystemExit(main())
