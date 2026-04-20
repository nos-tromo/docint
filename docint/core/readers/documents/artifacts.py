"""Artifact persistence for the document processing pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from loguru import logger

from docint.core.readers.documents.models import (
    ChunkResult,
    DocumentManifest,
    ImageResult,
    LayoutBlock,
    PageText,
    TableResult,
)


def _ensure_dir(path: Path) -> Path:
    """Create *path* if it does not exist and return it.

    Args:
        path (Path): The directory path to ensure exists.

    Returns:
        Path: The same path that was ensured to exist.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_manifest(manifest: DocumentManifest, artifacts_dir: Path) -> Path:
    """Write ``manifest.json`` to the document's artifact directory.

    Args:
        manifest (DocumentManifest): The document manifest to persist.
        artifacts_dir (Path): Root artifacts directory.

    Returns:
        Path: Path to the written file.
    """
    doc_dir = _ensure_dir(artifacts_dir / manifest.doc_id)
    out = doc_dir / "manifest.json"
    out.write_text(json.dumps(asdict(manifest), indent=2, default=str))
    logger.debug("Saved manifest: {}", out)
    return out


def save_layout(
    doc_id: str, page_index: int, blocks: list[LayoutBlock], artifacts_dir: Path
) -> Path:
    """Write ``layout.json`` for a single page.

    Args:
        doc_id (str): Document identifier.
        page_index (int): Page index.
        blocks (list[LayoutBlock]): Layout blocks for the page.
        artifacts_dir (Path): Root artifacts directory.

    Returns:
        Path: Path to the written file.
    """
    page_dir = _ensure_dir(artifacts_dir / doc_id / "pages" / str(page_index))
    out = page_dir / "layout.json"
    out.write_text(json.dumps([asdict(b) for b in blocks], indent=2, default=str))
    logger.debug("Saved layout for page {}: {}", page_index, out)
    return out


def save_page_text(doc_id: str, page_text: PageText, artifacts_dir: Path) -> Path:
    """Write ``text.json`` for a single page.

    Args:
        doc_id (str): Document identifier.
        page_text (PageText): Page text extraction result.
        artifacts_dir (Path): Root artifacts directory.

    Returns:
        Path: Path to the written file.
    """
    page_dir = _ensure_dir(artifacts_dir / doc_id / "pages" / str(page_text.page_index))
    out = page_dir / "text.json"
    out.write_text(json.dumps(asdict(page_text), indent=2, default=str))
    logger.debug("Saved text for page {}: {}", page_text.page_index, out)
    return out


def save_table(doc_id: str, table: TableResult, artifacts_dir: Path) -> Path:
    """Write ``{table_id}.json`` to the tables directory.

    Args:
        doc_id (str): Document identifier.
        table (TableResult): Table extraction result.
        artifacts_dir (Path): Root artifacts directory.

    Returns:
        Path: Path to the written file.
    """
    tables_dir = _ensure_dir(artifacts_dir / doc_id / "tables")
    out = tables_dir / f"{table.table_id}.json"
    out.write_text(json.dumps(asdict(table), indent=2, default=str))
    logger.debug("Saved table {}: {}", table.table_id, out)

    # Write CSV if cell grid is available
    if table.cell_grid:
        csv_path = tables_dir / f"{table.table_id}.csv"
        lines = [",".join(row) for row in table.cell_grid]
        csv_path.write_text("\n".join(lines))
        logger.debug("Saved table CSV: {}", csv_path)

    return out


def save_image_metadata(doc_id: str, image: ImageResult, artifacts_dir: Path) -> Path:
    """Write ``{image_id}.json`` to the images directory.

    Args:
        doc_id (str): Document identifier.
        image (ImageResult): Image extraction result.
        artifacts_dir (Path): Root artifacts directory.

    Returns:
        Path: Path to the written file.
    """
    images_dir = _ensure_dir(artifacts_dir / doc_id / "images")
    out = images_dir / f"{image.image_id}.json"
    out.write_text(json.dumps(asdict(image), indent=2, default=str))
    logger.debug("Saved image metadata {}: {}", image.image_id, out)
    return out


def save_chunks(doc_id: str, chunks: list[ChunkResult], artifacts_dir: Path) -> Path:
    """Write ``chunks.jsonl`` to the document's artifact directory.

    Args:
        doc_id (str): Document identifier.
        chunks (list[ChunkResult]): Chunk results to persist.
        artifacts_dir (Path): Root artifacts directory.

    Returns:
        Path: Path to the written file.
    """
    doc_dir = _ensure_dir(artifacts_dir / doc_id)
    out = doc_dir / "chunks.jsonl"
    with out.open("w") as fh:
        for chunk in chunks:
            fh.write(json.dumps(asdict(chunk), default=str) + "\n")
    logger.debug("Saved {} chunks: {}", len(chunks), out)
    return out


def load_manifest(doc_id: str, artifacts_dir: Path) -> DocumentManifest | None:
    """Load an existing manifest if present.

    Args:
        doc_id (str): Document identifier.
        artifacts_dir (Path): Root artifacts directory.

    Returns:
        DocumentManifest | None: A ``DocumentManifest`` or ``None`` if not found.
    """
    manifest_path = artifacts_dir / doc_id / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        data = json.loads(manifest_path.read_text())
        from docint.core.readers.documents.models import PageInfo

        pages = [PageInfo(**p) for p in data.pop("pages", [])]
        return DocumentManifest(pages=pages, **data)
    except Exception as exc:
        logger.warning("Failed to load manifest {}: {}", manifest_path, exc)
        return None
