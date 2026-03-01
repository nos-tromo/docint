"""Pipeline configuration with configurable thresholds."""

from __future__ import annotations

import os
from dataclasses import dataclass

PIPELINE_VERSION = "1.0.0"


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the document processing pipeline.

    Attributes:
        text_coverage_threshold: Minimum characters-per-area ratio below which
            a page is classified as needing OCR.
        pipeline_version: Semver string identifying the pipeline logic version.
        artifacts_dir: Root directory for artifact output.
        max_retries: Maximum retry attempts per stage on a given page.
        force_reprocess: When True, ignore existing artifacts and reprocess.
        max_workers: Maximum parallel workers for document-level processing.
    """

    text_coverage_threshold: float
    pipeline_version: str
    artifacts_dir: str
    max_retries: int
    force_reprocess: bool
    max_workers: int


def load_pipeline_config() -> PipelineConfig:
    """Build a ``PipelineConfig`` from environment variables with sensible defaults.

    Environment variables
    ---------------------
    PIPELINE_TEXT_COVERAGE_THRESHOLD : float
        Characters per area unit (default ``0.01``).
    PIPELINE_ARTIFACTS_DIR : str
        Root artifacts directory (default ``"artifacts"``).
    PIPELINE_MAX_RETRIES : int
        Max retries per page stage (default ``2``).
    PIPELINE_FORCE_REPROCESS : str
        ``"true"`` / ``"1"`` to force (default ``"false"``).
    PIPELINE_MAX_WORKERS : int
        Document-level parallelism (default ``4``).

    Returns:
        A fully-initialised ``PipelineConfig``.
    """
    return PipelineConfig(
        text_coverage_threshold=float(
            os.getenv("PIPELINE_TEXT_COVERAGE_THRESHOLD", "0.01")
        ),
        pipeline_version=PIPELINE_VERSION,
        artifacts_dir=os.getenv("PIPELINE_ARTIFACTS_DIR", "artifacts"),
        max_retries=int(os.getenv("PIPELINE_MAX_RETRIES", "2")),
        force_reprocess=os.getenv("PIPELINE_FORCE_REPROCESS", "false").lower()
        in {"true", "1", "yes"},
        max_workers=int(os.getenv("PIPELINE_MAX_WORKERS", "4")),
    )
