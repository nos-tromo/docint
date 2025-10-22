"""Utilities to keep fastembed usable in fully offline environments."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_PATCH_FLAG = "_docint_fastembed_offline_patched"


def _should_force_offline(force: bool | None) -> bool:
    if force is not None:
        return force

    env_value = os.getenv("DOCINT_FASTEMBED_FORCE_OFFLINE", "1").strip().lower()
    return env_value not in {"0", "false", "no", "off"}


def enable_fastembed_offline_mode(*, force: bool | None = None) -> None:
    """Monkeypatch fastembed downloads to avoid network calls.

    Fastembed performs metadata checks against Hugging Face the first time a model
    is loaded. In air‑gapped deployments this results in a hard failure even when
    the required assets are already present on disk. This helper forces
    fastembed's download helpers into ``local_files_only`` mode so that all
    verification happens against the local cache without contacting the network.

    Parameters
    ----------
    force:
        Explicit override used mostly for tests. When ``None`` the behaviour is
        controlled by the ``DOCINT_FASTEMBED_FORCE_OFFLINE`` environment variable
        which defaults to ``"1"`` (enabled).
    """

    if not _should_force_offline(force):
        return

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    try:
        from fastembed.common import model_management as mm
    except Exception as exc:  # pragma: no cover - fastembed optional in tests
        logger.debug("Skipping fastembed offline patch: %s", exc)
        return

    if getattr(mm, _PATCH_FLAG, False):
        return

    original_download_files = (
        mm.ModelManagement.download_files_from_huggingface.__func__
    )
    original_download_model = mm.ModelManagement.download_model.__func__

    def _download_files_offline(
        cls,
        hf_source_repo: str,
        cache_dir: str,
        extra_patterns: list[str],
        local_files_only: bool = False,
        **kwargs: Any,
    ) -> str:
        patched_kwargs = {**kwargs, "local_files_only": True}
        return original_download_files(
            cls,
            hf_source_repo,
            cache_dir,
            extra_patterns,
            **patched_kwargs,
        )

    def _download_model_offline(
        cls,
        model: Any,
        cache_dir: str,
        retries: int = 3,
        **kwargs: Any,
    ):
        patched_kwargs = {**kwargs, "local_files_only": True}
        return original_download_model(
            cls,
            model,
            cache_dir,
            retries=1,
            **patched_kwargs,
        )

    mm.ModelManagement.download_files_from_huggingface = classmethod(
        _download_files_offline
    )
    mm.ModelManagement.download_model = classmethod(_download_model_offline)
    setattr(mm, _PATCH_FLAG, True)
    logger.debug("Enabled fastembed offline mode")

