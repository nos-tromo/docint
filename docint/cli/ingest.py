"""CLI entry point for ingesting documents into a collection."""

import sys
import time
from collections.abc import Callable
from pathlib import Path

from loguru import logger

from docint.core.rag import RAG, EmptyIngestionError
from docint.utils.env_cfg import load_path_env, set_offline_env
from docint.utils.logger_cfg import init_logger

_SECONDS_PER_DAY = 86_400
_SECONDS_PER_HOUR = 3_600


def _format_elapsed(seconds: float) -> str:
    """Format an elapsed duration for the completion log line.

    Mirrors the SPA's ``formatDuration``
    (``frontend/src/lib/ingestStatus.ts``) so a log line and the ingest
    card's frozen timer can be compared without converting units. The
    duration *scales* rather than overflowing one column: rolling hours
    into the minutes place renders a ~42 h run as ``2500:37``.

    Args:
        seconds (float): Elapsed wall-clock seconds. Non-positive and
            non-finite inputs yield ``"00:00"`` rather than a negative or
            nonsense duration.

    Returns:
        str: ``MM:SS`` under an hour, ``H:MM:SS`` under a day, and
        ``Nd HH:MM:SS`` beyond (DIN 1301 day symbol, shared across locales).
    """
    total = int(seconds)
    if total <= 0:
        return "00:00"
    days, remainder = divmod(total, _SECONDS_PER_DAY)
    hours, remainder = divmod(remainder, _SECONDS_PER_HOUR)
    minutes, secs = divmod(remainder, 60)
    if days > 0:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


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
) -> None:
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
    rag = (
        RAG(qdrant_collection=qdrant_col) if hybrid is None else RAG(qdrant_collection=qdrant_col, enable_hybrid=hybrid)
    )
    started_at = time.monotonic()
    try:
        rag.ingest_docs(
            data_dir,
            build_query_engine=False,
            progress_callback=progress_callback or (lambda msg: logger.info(msg)),
            ner=ner,
            hate_speech=hate_speech,
        )
    finally:
        rag.unload_models()
    # Both API call sites route through here, so this is the only place a
    # completed ingest's duration reaches an operator reading the backend
    # log — the SPA's frozen timer is not visible there. Measured across
    # the whole call including model unload: that is the wall clock the
    # caller actually waited. It therefore covers less than the SPA's
    # client-anchored timer, which starts at the upload rather than here.
    logger.info("Ingestion complete in {}.", _format_elapsed(time.monotonic() - started_at))


def main() -> None:
    """Main function for the ingestion CLI.

    Catches :class:`EmptyIngestionError` so that a no-content ingestion
    surfaces as a warning and a clean exit rather than a traceback or
    non-zero exit code — the underlying ``RAG`` already removed the
    empty SQLite KV store and retained the uploaded source files.
    """
    init_logger()
    set_offline_env()
    data_path = load_path_env().data
    qdrant_col = get_collection()
    try:
        ingest_docs(qdrant_col, data_path)
    except EmptyIngestionError as exc:
        logger.warning(
            "Ingestion produced no content for '{}'. Empty KV store has been "
            "removed; uploaded source files are retained.",
            exc.collection_name,
        )


if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parents[2].resolve()))
    main()
