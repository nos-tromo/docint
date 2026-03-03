"""Pipeline orchestrator with per-page error isolation and idempotency."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from docint.core.readers.documents.artifacts import (
    load_manifest,
    save_chunks,
    save_image_metadata,
    save_layout,
    save_manifest,
    save_page_text,
    save_table,
)
from docint.core.readers.documents.chunking import chunk_document
from docint.utils.env_cfg import PipelineConfig, load_pipeline_config
from docint.core.readers.documents.extraction import extract_images, extract_tables
from docint.core.readers.documents.layout import analyze_document
from docint.core.readers.documents.models import (
    BBox,
    BlockType,
    ChunkResult,
    DocumentManifest,
    ImageResult,
    LayoutBlock,
    PageText,
    TableResult,
)
from docint.core.readers.documents.ocr import (
    OCREngine,
    VisionOCREngine,
    extract_text_for_pages,
)
from docint.core.readers.documents.triage import triage_pdf
from docint.utils.hashing import compute_file_hash


class DocumentPipelineOrchestrator:
    """Orchestrate the full document processing pipeline for a single PDF.

    Stages are executed per-page where possible with retry logic.
    Failures on individual pages are isolated — they do not crash the
    entire document.  Results are persisted as artifacts for debugging
    and reprocessing.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        """Initialize the orchestrator with the given configuration.

        Args:
            config (PipelineConfig | None, optional): Pipeline configuration. When ``None``
            the config is loaded from environment variables.
        """
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

            def _analyze_layout(pi=page_info) -> dict[int, list[LayoutBlock]]:
                result = analyze_document(file_path, [pi])
                return result if result is not None else {}

            result = self._run_with_retry(
                f"layout-page-{page_info.page_index}",
                _analyze_layout,
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
        vision_engine: OCREngine | None = None
        if self.config.enable_vision_ocr and any(p.needs_ocr for p in pages):
            try:
                vision_engine = VisionOCREngine(
                    file_path,
                    timeout=self.config.vision_ocr_timeout,
                    max_retries=self.config.vision_ocr_max_retries,
                    max_image_dimension=self.config.vision_ocr_max_image_dimension,
                    max_tokens=self.config.vision_ocr_max_tokens,
                )
                logger.info("Vision OCR engine initialised for {}", file_path.name)
            except Exception as exc:
                logger.debug("Vision OCR engine not available: {}", exc)

        page_texts: dict[int, PageText] = {}
        for page_info in pages:
            if page_info.status == "failed":
                continue

            def _extract_text(pi=page_info) -> dict[int, PageText]:
                return (
                    extract_text_for_pages(
                        file_path,
                        [pi],
                        layout,
                        vision_engine=vision_engine,
                    )
                    or {}
                )

            result = self._run_with_retry(
                f"ocr-page-{page_info.page_index}",
                _extract_text,
            )
            if result is not None:
                page_texts.update(result)
                for pt in result.values():
                    save_page_text(doc_id, pt, artifacts_dir)
            else:
                page_info.status = "failed"
                page_info.error = "Text extraction failed"

        if vision_engine is not None and hasattr(vision_engine, "close"):
            vision_engine.close()

        # Inject synthetic TEXT blocks for OCR pages whose layout only
        # contains non-text blocks (e.g. a lone FIGURE for a scanned
        # page).  Without a TEXT block the chunker has nothing to emit.
        for page_info in pages:
            idx = page_info.page_index
            pt = page_texts.get(idx)
            if not pt or not pt.full_text.strip():
                continue
            page_blocks = layout.get(idx, [])
            if any(b.type == BlockType.TEXT for b in page_blocks):
                continue
            next_order = max((b.reading_order for b in page_blocks), default=-1) + 1
            page_blocks.append(
                LayoutBlock(
                    block_id=f"ocr-text-{idx}-synth",
                    page_index=idx,
                    type=BlockType.TEXT,
                    bbox=BBox(
                        x0=0.0,
                        y0=0.0,
                        x1=page_info.width or 612.0,
                        y1=page_info.height or 792.0,
                    ),
                    reading_order=next_order,
                    confidence=pt.confidence,
                    text=pt.full_text,
                )
            )
            save_layout(doc_id, idx, page_blocks, artifacts_dir)
            logger.info(
                "Injected synthetic TEXT block for OCR page {} (text_len={})",
                idx,
                len(pt.full_text),
            )

        # --- Stage 4: Table extraction ---
        tables: list[TableResult] = (
            self._run_with_retry("tables", lambda: extract_tables(layout)) or []
        )
        for table in tables:
            save_table(doc_id, table, artifacts_dir)
        manifest.tables_found = len(tables)

        # --- Stage 5: Image extraction ---
        images_dir = artifacts_dir / doc_id / "images"
        images: list[ImageResult] = (
            self._run_with_retry(
                "images",
                lambda: extract_images(layout, file_path, images_dir),
            )
            or []
        )
        for image in images:
            save_image_metadata(doc_id, image, artifacts_dir)
        manifest.images_found = len(images)

        # --- Stage 6: Chunking ---
        chunks: list[ChunkResult] = (
            self._run_with_retry(
                "chunking",
                lambda: chunk_document(doc_id, layout, page_texts, tables, images),
            )
            or []
        )
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

    def _run_with_retry(
        self, stage: str, fn: Callable[[], Any], retries: int | None = None
    ) -> Any:
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
                    logger.error(
                        "Stage '{}' failed after {} attempts", stage, max_tries
                    )
                    return None

    @staticmethod
    def _log_summary(
        manifest: DocumentManifest, duration_ms: int, chunks_count: int
    ) -> None:
        """Emit a structured summary log for the processed document.

        Args:
            manifest: The document manifest containing processing details.
            duration_ms: Total processing time in milliseconds.
            chunks_count: Total number of chunks produced.
        """
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
