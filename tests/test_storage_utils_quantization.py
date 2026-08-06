"""Tests for the shared Qdrant quantization-config builder."""

from __future__ import annotations

import pytest
from qdrant_client.http import models as qdrant_models

from docint.core.storage.utils import build_quantization_config
from docint.utils.env_cfg import QdrantQuantizationConfig


def test_builder_returns_turbo_config() -> None:
    """An explicit turbo config maps onto the TurboQuantization payload."""
    cfg = QdrantQuantizationConfig(mode="turbo", bits="bits2", always_ram=True)
    built = build_quantization_config(cfg)
    assert isinstance(built, qdrant_models.TurboQuantization)
    assert built.turbo.bits == qdrant_models.TurboQuantBitSize.BITS2
    assert built.turbo.always_ram is True


def test_builder_returns_none_when_disabled() -> None:
    """``mode='none'`` yields no quantization payload."""
    cfg = QdrantQuantizationConfig(mode="none", bits="bits4", always_ram=None)
    assert build_quantization_config(cfg) is None


def test_builder_defaults_load_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no explicit config, the builder resolves settings from the env."""
    monkeypatch.delenv("QDRANT_QUANTIZATION", raising=False)
    monkeypatch.delenv("QDRANT_TURBOQUANT_BITS", raising=False)
    monkeypatch.delenv("QDRANT_QUANTIZATION_ALWAYS_RAM", raising=False)
    built = build_quantization_config()
    assert isinstance(built, qdrant_models.TurboQuantization)
    assert built.turbo.bits == qdrant_models.TurboQuantBitSize.BITS4
    assert built.turbo.always_ram is None
