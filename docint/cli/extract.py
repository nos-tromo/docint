"""CLI entry point for rendering a collection's written data extract.

The offline twin of ``POST /collections/{name}/extracts``: same gather,
partition and bundle code, writing the ZIP to ``RESULTS_PATH`` instead of the
extract store. Useful on a host with no HTTP access to the backend, and for
scripting a batch of collections.
"""

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from docint.cli._collection import CollectionNotFoundError, resolve_collection_name
from docint.core.extract.bundle import build_bundle
from docint.core.extract.gather import scroll_collection
from docint.core.extract.units import partition, resolve_target
from docint.core.rag import RAG
from docint.core.state.report_render import PdfEngineUnavailableError, html_to_pdf
from docint.utils.env_cfg import load_extract_env, load_path_env, set_offline_env
from docint.utils.logger_cfg import init_logger


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the extract CLI's arguments.

    Args:
        argv (list[str] | None): Argument vector; defaults to ``sys.argv``.

    Returns:
        argparse.Namespace: ``collection``, ``target``, ``out``, ``no_pdf``.
    """
    parser = argparse.ArgumentParser(description="Render a collection's data extract as a ZIP bundle.")
    parser.add_argument("collection", nargs="?", help="Logical or physical collection name.")
    parser.add_argument("-t", "--target", help="Render one source: a file hash, an image id or a posting uuid.")
    parser.add_argument("-o", "--out", type=Path, help="Directory to write the bundle into (default: RESULTS_PATH).")
    parser.add_argument("--no-pdf", action="store_true", help="Skip the combined PDF.")
    return parser.parse_args(argv)


def build_extract(collection: str, *, target: str | None, out_dir: Path, with_pdf: bool) -> Path:
    """Render one collection's extract and write it to disk.

    Args:
        collection (str): Logical or physical collection name.
        target (str | None): One source to render, or ``None`` for all of it.
        out_dir (Path): Directory the bundle is written into.
        with_pdf (bool): Whether to render the combined PDF.

    Returns:
        Path: The written archive.

    Raises:
        SystemExit: When the collection cannot be resolved, or holds nothing
            the target names.
    """
    rag = RAG(qdrant_collection=collection)
    try:
        try:
            physical = resolve_collection_name(rag, collection)
        except CollectionNotFoundError as exc:
            logger.error("{}", exc)
            raise SystemExit(1) from exc
        if physical != collection:
            logger.info("Resolved '{}' to the physical collection '{}'.", collection, physical)

        main_points, image_points = scroll_collection(
            rag.qdrant_client, physical, rag._image_collection_name(physical), source_id=target
        )
        units = partition(main_points, image_points)
        if target:
            units = resolve_target(units, target)
        if not units:
            logger.error("Nothing to extract for '{}'{}.", collection, f" (target {target!r})" if target else "")
            raise SystemExit(1)

        engine = None
        if with_pdf:
            try:
                html_to_pdf("<html><body></body></html>")
                engine = html_to_pdf
            except PdfEngineUnavailableError as exc:
                logger.warning("PDF engine unavailable, writing the bundle without one: {}", exc)

        now = datetime.now(tz=UTC)
        bundle = build_bundle(
            units,
            collection=collection,
            cfg=load_extract_env(),
            pdf=engine,
            now=now,
            progress=lambda rendered, total: logger.info("Rendering {}/{}", rendered, total),
        )
    finally:
        rag.unload_models()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{collection}-extract-{now:%Y%m%d-%H%M}.zip"
    path.write_bytes(bundle.zip_bytes)
    logger.info(
        "Extract written | file={!r} bytes={} {}",
        str(path),
        len(bundle.zip_bytes),
        " ".join(f"{key}={value}" for key, value in bundle.counts.items()),
    )
    return path


def main(argv: list[str] | None = None) -> None:
    """Run the extract CLI.

    Args:
        argv (list[str] | None): Argument vector; defaults to ``sys.argv``.
    """
    init_logger()
    set_offline_env()
    args = parse_args(argv)
    collection = args.collection or input("Enter collection name: ").strip()
    if not collection:
        logger.error("A collection name is required.")
        raise SystemExit(1)
    build_extract(
        collection,
        target=args.target,
        out_dir=args.out or load_path_env().results,
        with_pdf=not args.no_pdf,
    )


if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parents[2].resolve()))
    main()
