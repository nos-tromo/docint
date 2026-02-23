"""Pipeline orchestrator with per-page error isolation and idempotency."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

from loguru import logger

from docint.core.pipeline.artifacts import (
    load_manifest,
    save_chunks,
    save_image_metadata,
    save_layout,
    save_manifest,
    save_page_text,
    save_table,
)
from docint.core.pipeline.chunking import chunk_document
from docint.core.pipeline.config import PipelineConfig, load_pipeline_config
from docint.core.pipeline.extraction import extract_images, extract_tables
from docint.core.pipeline.layout import analyze_document
from docint.core.pipeline.models import (
    ChunkResult,
    DocumentManifest,
    ImageResult,
    LayoutBlock,
    PageText,
    TableResult,
)
from docint.core.pipeline.ocr import extract_text_for_pages
from docint.core.pipeline.triage import triage_pdf
from docint.utils.hashing import compute_file_hash


class DocumentPipelineOrchestrator:
    """Orchestrate the full document processing pipeline for a single PDF.

    Stages are executed per-page where possible with retry logic.
    Failures on individual pages are isolated — they do not crash the
    entire document.  Results are persisted as artifacts for debugging
    and reprocessing.

    Args:
        config: Pipeline configuration.  When ``None`` the config is
            loaded from environment variables.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or load_pipeline_config()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, file_path: str | Path) -> DocumentManifest:
        """Run the full pipeline on *file_path* and return the manifest.

        Args:
            file_path: Path to a PDF file.

        Returns:
            A ``DocumentManifest`` summarising the processing outcome.
        """
        file_path = Path(file_path)
        doc_id = compute_file_hash(file_path)
        artifacts_dir = Path(self.config.artifacts_dir)

        # --- Idempotency check ---
        if not self.config.force_reprocess:
            existing = load_manifest(doc_id, artifacts_dir)
            if (
                existing
                and existing.pipeline_version == self.config.pipeline_version
                and existing.status == "completed"
            ):
                logger.info(
                    "Skipping {} — artifacts exist for pipeline v{}",
                    file_path.name,
                    self.config.pipeline_version,
                )
                return existing

        start = time.monotonic()
        manifest = DocumentManifest(
            doc_id=doc_id,
            file_path=str(file_path),
            file_name=file_path.name,
            pipeline_version=self.config.pipeline_version,
        )

        # --- Stage 1: Triage ---
        pages = self._run_with_retry(
            "triage", lambda: triage_pdf(file_path, self.config)
        )
        if pages is None:
            manifest.status = "failed"
            manifest.error = "Triage failed after retries"
            save_manifest(manifest, artifacts_dir)
            return manifest

        manifest.pages = pages
        manifest.pages_total = len(pages)
        manifest.pages_ocr = sum(1 for p in pages if p.needs_ocr)
        manifest.pages_failed = sum(1 for p in pages if p.status == "failed")

        # --- Stage 2: Layout analysis ---
        layout: dict[int, list[LayoutBlock]] = {}
        for page_info in pages:
            if page_info.status == "failed":
                layout[page_info.page_index] = []
                continue
            result = self._run_with_retry(
                f"layout-page-{page_info.page_index}",
                lambda pi=page_info: analyze_document(  # type: ignore[misc]
                    file_path, [pi]
                ),
            )
            if result is not None:
                layout.update(result)
                save_layout(
                    doc_id,
                    page_info.page_index,
                    result.get(page_info.page_index, []),
                    artifacts_dir,
                )
            else:
                layout[page_info.page_index] = []
                page_info.status = "failed"
                page_info.error = "Layout analysis failed"

        # --- Stage 3: OCR / text extraction ---
        page_texts: dict[int, PageText] = {}
        for page_info in pages:
            if page_info.status == "failed":
                continue
            result = self._run_with_retry(
                f"ocr-page-{page_info.page_index}",
                lambda pi=page_info: extract_text_for_pages(  # type: ignore[misc]
                    file_path, [pi], layout
                ),
            )
            if result is not None:
                page_texts.update(result)
                for pt in result.values():
                    save_page_text(doc_id, pt, artifacts_dir)
            else:
                page_info.status = "failed"
                page_info.error = "Text extraction failed"

        # --- Stage 4: Table extraction ---
        tables: list[TableResult] = self._run_with_retry(
            "tables", lambda: extract_tables(layout)
        ) or []
        for table in tables:
            save_table(doc_id, table, artifacts_dir)
        manifest.tables_found = len(tables)

        # --- Stage 5: Image extraction ---
        images_dir = artifacts_dir / doc_id / "images"
        images: list[ImageResult] = self._run_with_retry(
            "images",
            lambda: extract_images(layout, file_path, images_dir),
        ) or []
        for image in images:
            save_image_metadata(doc_id, image, artifacts_dir)
        manifest.images_found = len(images)

        # --- Stage 6: Chunking ---
        chunks: list[ChunkResult] = self._run_with_retry(
            "chunking",
            lambda: chunk_document(doc_id, layout, page_texts, tables, images),
        ) or []
        if chunks:
            save_chunks(doc_id, chunks, artifacts_dir)

        # --- Finalize ---
        manifest.pages_failed = sum(1 for p in pages if p.status == "failed")
        manifest.status = "completed"
        duration_ms = round((time.monotonic() - start) * 1000)
        save_manifest(manifest, artifacts_dir)

        self._log_summary(manifest, duration_ms, len(chunks))
        return manifest

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_with_retry(self, stage: str, fn, retries: int | None = None):  # type: ignore[no-untyped-def]
        """Execute *fn* with retry logic, returning ``None`` on exhaustion.

        Args:
            stage: Human-readable stage name for logging.
            fn: Callable to execute.
            retries: Override for ``config.max_retries``.

        Returns:
            The result of *fn* or ``None`` if all retries failed.
        """
        max_tries = (retries or self.config.max_retries) + 1
        for attempt in range(1, max_tries + 1):
            try:
                return fn()
            except Exception as exc:
                logger.warning(
                    "Stage '{}' attempt {}/{} failed: {}",
                    stage,
                    attempt,
                    max_tries,
                    exc,
                )
                if attempt == max_tries:
                    logger.error("Stage '{}' failed after {} attempts", stage, max_tries)
                    return None

    @staticmethod
    def _log_summary(
        manifest: DocumentManifest, duration_ms: int, chunks_count: int
    ) -> None:
        """Emit a structured summary log for the processed document."""
        logger.info(
            "Pipeline complete | doc_id={} file={} duration_ms={} "
            "pages_total={} pages_ocr={} pages_failed={} "
            "tables={} images={} chunks={}",
            manifest.doc_id[:12],
            manifest.file_name,
            duration_ms,
            manifest.pages_total,
            manifest.pages_ocr,
            manifest.pages_failed,
            manifest.tables_found,
            manifest.images_found,
            chunks_count,
        )
