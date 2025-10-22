"""Utilities for keeping fastembed usable in offline environments."""

from __future__ import annotations

import logging
import os
from typing import Any

_LOGGER = logging.getLogger(__name__)


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def ensure_fastembed_offline_patch(force: bool | None = None) -> None:
    """Monkeypatch fastembed to skip Hugging Face repo checks when offline."""
    if force is None:
        should_patch = _is_truthy(os.getenv("DOCINT_FASTEMBED_OFFLINE"))
    else:
        should_patch = force

    try:
        from fastembed.common import model_management
    except Exception as exc:  # pragma: no cover - import failure is fine when unused
        if should_patch:
            _LOGGER.warning("Could not import fastembed for offline patch: %s", exc)
        return

    cls = model_management.ModelManagement

    if getattr(cls, "_docint_fastembed_offline_patched", False):
        return

    original_classmethod = None
    for ancestor in cls.__mro__:
        candidate = ancestor.__dict__.get("download_files_from_huggingface")
        if candidate is not None:
            original_classmethod = candidate
            break

    if original_classmethod is None:
        raise AttributeError(
            "fastembed ModelManagement is missing download_files_from_huggingface"
        )

    original_func = original_classmethod.__func__

    def patched(
        subclass: type,
        hf_source_repo: str,
        cache_dir: str,
        extra_patterns: list[str],
        local_files_only: bool = False,
        **kwargs: Any,
    ) -> str:
        call_kwargs = dict(kwargs)
        call_kwargs.setdefault("local_files_only", local_files_only)

        if force is None and not _is_truthy(os.getenv("DOCINT_FASTEMBED_OFFLINE")):
            return original_func(
                subclass,
                hf_source_repo,
                cache_dir,
                extra_patterns,
                **call_kwargs,
            )

        call_kwargs["local_files_only"] = True

        try:
            return original_func(
                subclass,
                hf_source_repo,
                cache_dir,
                extra_patterns,
                **call_kwargs,
            )
        except Exception as exc:  # pragma: no cover - propagate with context
            msg = (
                "Fastembed offline mode failed. Ensure model files are pre-cached "
                "or unset DOCINT_FASTEMBED_OFFLINE to allow online downloads."
            )
            raise RuntimeError(msg) from exc

    setattr(cls, "download_files_from_huggingface", classmethod(patched))
    setattr(cls, "_docint_fastembed_offline_patched", True)

    if should_patch:
        _LOGGER.info("Applied fastembed offline patch; downloads will use cached files only.")
