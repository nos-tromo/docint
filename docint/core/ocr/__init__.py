"""Reading text out of pixels — one engine for every caller.

A scanned PDF page, a table's region, a photographed letter, a video keyframe:
all of them are "read this image", so all of them go through
:class:`~docint.core.ocr.engine.DocumentOcrEngine`. What the configured model
expects and returns lives in :mod:`docint.core.ocr.families`; captioning an
image ("describe what this shows") is a different task and stays in
``core/ingest/images_service.py``.
"""

from docint.core.ocr.engine import (
    DocumentOcrEngine,
    OcrError,
    OcrRejected,
    OcrStats,
    OcrUnreachable,
    build_engine,
)
from docint.core.ocr.families import OcrBlock, OcrBox, OcrCategory, OcrFrame, OcrLimits, OcrTask
from docint.core.ocr.html_table import grid_to_text, parse_html_table

__all__ = [
    "DocumentOcrEngine",
    "OcrBlock",
    "OcrBox",
    "OcrCategory",
    "OcrError",
    "OcrFrame",
    "OcrLimits",
    "OcrRejected",
    "OcrStats",
    "OcrTask",
    "OcrUnreachable",
    "build_engine",
    "grid_to_text",
    "parse_html_table",
]
