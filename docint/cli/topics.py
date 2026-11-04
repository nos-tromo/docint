"""CLI entry point for building, extending and inspecting a collection's topics.

The offline twin of ``POST /collections/{name}/topics``: same service, run in
the running backend container so it reaches the Qdrant and inference aliases.

``build``, ``align`` and ``discover`` are audited as ``modify``
(``topics.<op>``) with the run's model calls as ``ai_call`` transfers;
``status`` shows the topics' labels and keywords, so it is a ``query``
(``topics.overview``).
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from loguru import logger

from docint.cli._audit import collection_criteria, session
from docint.cli._collection import CollectionNotFoundError, resolve_collection_name
from docint.core.audit import CliAudit, Operation, metered
from docint.core.audit_targets import topic_target
from docint.core.rag import RAG
from docint.core.topics.service import Action, TopicsRefused, progress_message
from docint.utils.audit_event import Target
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


def _print_status(rag: RAG, physical: str, op: Operation) -> None:
    """Print the active run's overview.

    Recorded before it is printed; a collection without topics shows nothing
    and records nothing.

    Args:
        rag (RAG): The engine.
        physical (str): Physical collection name.
        op (Operation): The open ``query`` operation.
    """
    with metered(op):
        overview = rag.topics_overview(physical)
    if overview is None:
        print("No topics yet.")
        return
    op.record(target=topic_target(list(overview.get("topics") or [])), criteria={"run_id": overview.get("run_id")})
    coverage = overview["coverage"]
    print(
        f"run={overview['run_id']} topics={len(overview['topics'])} assigned={coverage['assigned']}/{coverage['total']}"
        f" unassigned={coverage['unassigned']} pending={coverage['pending']}"
    )
    for topic in overview["topics"]:
        keywords = ", ".join(word for word, _score in topic["keywords"][:3])
        print(f"T{topic['id']}  {topic['size']}  {topic['label']} — {keywords}")


def _error_criteria(exc: BaseException) -> dict[str, object] | None:
    """Name how a topics run failed, by its static code only.

    Args:
        exc (BaseException): What ended the run.

    Returns:
        dict[str, object] | None: The refusal or endpoint code, else nothing.
    """
    if isinstance(exc, TopicsRefused):
        return {"refused": exc.code}
    if isinstance(exc, TopicsEndpointError):
        return {"endpoint_error": exc.code}
    return None


def _run_action(work: RAG, op: str, physical: str, logical: str, audit_op: Operation) -> int:
    """Run a build, discover or align and print its outcome.

    Args:
        work (RAG): The engine bound to the physical collection.
        op (str): ``build``, ``discover`` or ``align``.
        physical (str): Physical collection name.
        logical (str): The name as typed, for the service's refusal log line.
        audit_op (Operation): The open ``modify`` operation; the run's model
            calls are its transfers.

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
        with metered(audit_op, on_error=_error_criteria) as meter:
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
    audit_op.record(
        target=Target(kind="topic", count=outcome.topics),
        criteria={
            "run_id": outcome.run_id,
            "added": outcome.added,
            "assigned": outcome.assigned,
            "deferred": outcome.deferred,
        },
    )
    audit_op.transfers(meter)
    print(
        f"{outcome.action}: run={outcome.run_id} topics={outcome.topics}"
        f" added={outcome.added} assigned={outcome.assigned}"
    )
    return 0


def run(op: str, collection: str, *, audit: CliAudit | None = None) -> int:
    """Run one topics operation.

    Args:
        op (str): One of :data:`OPS`.
        collection (str): Logical or physical collection name.
        audit (CliAudit | None): The command's audit chain; one is opened for
            this run when omitted.

    Returns:
        int: 0 on success, 1 on an endpoint fault, 2 on a refusal, a disabled
            feature or an unknown collection.
    """
    if not load_topics_env().enabled:
        logger.error("Topics are switched off (TOPICS_ENABLED=false).")
        return 2
    action, operation = ("query", "topics.overview") if op == "status" else ("modify", f"topics.{op}")
    also = () if op == "status" else ("ai_call",)
    with session(audit, "topics", action, operation, also=also) as cli:
        audit_op = cli.operation(action, operation, container=collection)
        rag = RAG(qdrant_collection=collection)
        work: RAG | None = None
        try:
            try:
                physical = resolve_collection_name(rag, collection)
            except CollectionNotFoundError as exc:
                logger.error("{}", exc)
                audit_op.fail(criteria={"collection_not_found": True})
                return 2
            audit_op.criteria.update(collection_criteria(collection, physical))
            work = RAG(qdrant_collection=physical)
            if op == "status":
                _print_status(work, physical, audit_op)
                return 0
            return _run_action(work, op, physical, collection, audit_op)
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
