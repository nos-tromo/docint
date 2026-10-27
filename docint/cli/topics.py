"""CLI entry point for building, extending and inspecting a collection's topics.

The offline twin of ``POST /collections/{name}/topics``: same service, run in a
one-off backend container so it reaches the Qdrant and inference aliases.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from loguru import logger

from docint.cli._collection import CollectionNotFoundError, resolve_collection_name
from docint.core.rag import RAG
from docint.core.topics.service import Action, TopicsRefused, progress_message
from docint.utils.env_cfg import load_logging_env, load_topics_env, set_offline_env
from docint.utils.logfmt import ProgressLogThrottle
from docint.utils.logger_cfg import init_logger
from docint.utils.topics_client import TopicsEndpointError

OPS = ("build", "align", "discover", "status")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the topics CLI's arguments.

    Args:
        argv (Sequence[str] | None): Argument vector; defaults to ``sys.argv``.

    Returns:
        argparse.Namespace: ``op`` and ``collection``.
    """
    parser = argparse.ArgumentParser(description="Build, extend or inspect a collection's topics.")
    parser.add_argument("op", choices=OPS, help="build, align, discover or status.")
    parser.add_argument("collection", nargs="?", help="Logical or physical collection name.")
    return parser.parse_args(argv)


def _print_status(rag: RAG, physical: str) -> None:
    """Print the active run's overview.

    Args:
        rag (RAG): The engine.
        physical (str): Physical collection name.
    """
    overview = rag.topics_overview(physical)
    if overview is None:
        print("No topics yet.")
        return
    coverage = overview["coverage"]
    print(
        f"run={overview['run_id']} topics={len(overview['topics'])} assigned={coverage['assigned']}/{coverage['total']}"
        f" unassigned={coverage['unassigned']} pending={coverage['pending']}"
    )
    for topic in overview["topics"]:
        keywords = ", ".join(word for word, _score in topic["keywords"][:3])
        print(f"T{topic['id']}  {topic['size']}  {topic['label']} — {keywords}")


def _run_action(work: RAG, op: str, physical: str, logical: str) -> int:
    """Run a build, discover or align and print its outcome.

    Args:
        work (RAG): The engine bound to the physical collection.
        op (str): ``build``, ``discover`` or ``align``.
        physical (str): Physical collection name.
        logical (str): The name as typed, for the service's refusal log line.

    Returns:
        int: 0 on success, 1 on an endpoint fault, 2 on a refusal.
    """
    throttle = ProgressLogThrottle(load_logging_env().progress_interval_s)

    def _progress(stage: str, done: int, total: int) -> None:
        """Log one throttled progress line.

        Args:
            stage (str): The stage.
            done (int): Units done.
            total (int): Units in all, 0 when unknown.
        """
        for line in throttle.observe(progress_message(stage, done, total)):
            logger.info("{}", line)

    def _warn(code: str, message: str) -> None:
        """Log one coded warning.

        Args:
            code (str): The warning code.
            message (str): Its static message.
        """
        logger.warning("{} ({})", message, code)

    try:
        outcome = work.topics_service().run(
            cast(Action, op),
            physical,
            logical=logical,
            owner_created_at=lambda: work.collection_created_at(physical),
            progress=_progress,
            warn=_warn,
        )
    except TopicsRefused as exc:
        logger.error("{}", exc.describe())
        return 2
    except TopicsEndpointError as exc:
        logger.error("{}", str(exc))
        return 1
    for line in throttle.flush():
        logger.info("{}", line)
    print(
        f"{outcome.action}: run={outcome.run_id} topics={outcome.topics}"
        f" added={outcome.added} assigned={outcome.assigned}"
    )
    return 0


def run(op: str, collection: str) -> int:
    """Run one topics operation.

    Args:
        op (str): One of :data:`OPS`.
        collection (str): Logical or physical collection name.

    Returns:
        int: 0 on success, 1 on an endpoint fault, 2 on a refusal, a disabled
            feature or an unknown collection.
    """
    if not load_topics_env().enabled:
        logger.error("Topics are switched off (TOPICS_ENABLED=false).")
        return 2
    rag = RAG(qdrant_collection=collection)
    work: RAG | None = None
    try:
        try:
            physical = resolve_collection_name(rag, collection)
        except CollectionNotFoundError as exc:
            logger.error("{}", exc)
            return 2
        work = RAG(qdrant_collection=physical)
        if op == "status":
            _print_status(work, physical)
            return 0
        return _run_action(work, op, physical, collection)
    finally:
        if work is not None:
            work.unload_models()
        rag.unload_models()


def main(argv: Sequence[str] | None = None) -> None:
    """Run the topics CLI.

    Args:
        argv (Sequence[str] | None): Argument vector; defaults to ``sys.argv``.

    Raises:
        SystemExit: Always, with :func:`run`'s code.
    """
    init_logger()
    set_offline_env()
    args = parse_args(argv)
    collection = args.collection or input("Enter collection name: ").strip()
    if not collection:
        logger.error("A collection name is required.")
        raise SystemExit(2)
    raise SystemExit(run(args.op, collection))


if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parents[2].resolve()))
    main()
