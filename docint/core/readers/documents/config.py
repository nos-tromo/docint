"""Pipeline configuration — re-exported from ``docint.utils.env_cfg``.

All environment configuration lives in :pymod:`docint.utils.env_cfg`.
This module re-exports symbols for backward compatibility so existing
imports of the form ``from docint.core.readers.documents.config import …``
continue to work.
"""

from docint.utils.env_cfg import (  # noqa: F401
    PIPELINE_VERSION,
    PipelineConfig,
    load_pipeline_config,
)

__all__ = ["PIPELINE_VERSION", "PipelineConfig", "load_pipeline_config"]
