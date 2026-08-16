"""CLI entry point for ingesting documents into a collection."""

import sys
import time
from collections.abc import Callable
from pathlib import Path

from loguru import logger

from docint.core.rag import RAG, EmptyIngestionError, IngestStats
from docint.utils.duration import format_elapsed
from docint.utils.env_cfg import load_path_env, set_offline_env
from docint.utils.logger_cfg import init_logger


def get_collection() -> str:
    """Get user inputs for the Qdrant collection name.

    Returns:
        str: Qdrant collection name.
    """
    return input("Enter Qdrant collection name: ").strip()


def ingest_docs(
    qdrant_col: str,
    data_dir: Path,
    hybrid: bool | None = None,
    progress_callback: Callable[[str], None] | None = None,
    *,
    ner: bool | None = None,
    hate_speech: bool | None = None,
) -> IngestStats | None:
    """Ingest documents from the specified directory into the Qdrant collection.

    Args:
        qdrant_col (str): Qdrant collection name.
        data_dir (Path): Path to the data directory.
        hybrid (bool | None): Per-request hybrid-search override; ``None``
            keeps :func:`docint.utils.env_cfg.resolve_enable_hybrid`'s
            derived default instead of forcing it on. Passing ``True`` or
            ``False`` explicitly always wins.
        progress_callback (Callable[[str], None] | None): Optional callback for
            reporting ingestion progress.
        ner (bool | None): Per-request NER override; ``None`` keeps the env
            default (``NER_ENABLED``).
        hate_speech (bool | None): Per-request hate-speech override; ``None``
            keeps the env default (``ENABLE_HATE_SPEECH_DETECTION``).

    Returns:
        IngestStats | None: What the run did, forwarded for the job layer's
        run summary. ``None`` only if a test double stands in for ``RAG``.

    Raises:
        EmptyIngestionError: When the source directory yielded no usable
            content for a fresh collection. Re-raised so programmatic
            callers (e.g. the streaming ingest API) can branch on the
            soft-empty outcome without parsing log messages. The terminal
            CLI :func:`main` catches this and exits cleanly.

    Notes:
        The CLI skips query engine creation so that large generation and reranker models
        are not loaded unnecessarily during ingestion jobs.
    """
    # Deliberately untimed: this is one stage of a run, not a run. Under the
    # job API the same run also resolves entities and builds the collection
    # summary afterwards, and a duration logged here would exclude both — plus
    # the RAG construction just above, whose embedding-tokenizer load alone
    # cost 1.8 s of a 9.4 s run in the trace that motivated this. Each caller
    # that owns a whole run times it instead (see :func:`main`,
    # ``core/api.py``'s ``ingest``, and ``IngestJobManager._run``).
    rag = (
        RAG(qdrant_collection=qdrant_col) if hybrid is None else RAG(qdrant_collection=qdrant_col, enable_hybrid=hybrid)
    )
    try:
        # The ``logger.info`` fallback applies only when the caller supplied
        # no callback — i.e. on the terminal CLI path. Callers that pass one
        # (the job runner passes an SSE publisher) do not lose the narrative
        # any more: ``core/jobs.py`` tees every pushed message to the log.
        return rag.ingest_docs(
            data_dir,
            build_query_engine=False,
            progress_callback=progress_callback or (lambda msg: logger.info(msg)),
            ner=ner,
            hate_speech=hate_speech,
        )
    finally:
        rag.unload_models()


def main() -> None:
    """Main function for the ingestion CLI.

    Catches :class:`EmptyIngestionError` so that a no-content ingestion
    surfaces as a warning and a clean exit rather than a traceback or
    non-zero exit code — the underlying ``RAG`` already removed the
    empty SQLite KV store and retained the uploaded source files.

    Times the run from here rather than inside :func:`ingest_docs` so the
    duration covers everything the operator waited for, model loading
    included. There is no job and no ingest card on this path, so this line
    is the only record of how long the run took.
    """
    init_logger()
    set_offline_env()
    data_path = load_path_env().data
    qdrant_col = get_collection()
    started_ticks = time.monotonic()
    try:
        ingest_docs(qdrant_col, data_path)
    except EmptyIngestionError as exc:
        logger.warning(
            "Ingestion produced no content for '{}'. Empty KV store has been "
            "removed; uploaded source files are retained.",
            exc.collection_name,
        )
    else:
        logger.info("Ingestion complete in {}.", format_elapsed(time.monotonic() - started_ticks))


if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parents[2].resolve()))
    main()
