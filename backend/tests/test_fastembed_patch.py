from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _install_fastembed_module(monkeypatch, cls):
    fastembed_module = types.ModuleType("fastembed")
    common_module = types.ModuleType("fastembed.common")
    model_management_module = types.ModuleType("fastembed.common.model_management")
    model_management_module.ModelManagement = cls

    common_module.model_management = model_management_module

    monkeypatch.setitem(sys.modules, "fastembed", fastembed_module)
    monkeypatch.setitem(sys.modules, "fastembed.common", common_module)
    monkeypatch.setitem(sys.modules, "fastembed.common.model_management", model_management_module)


class _DummyModelManagement:
    called_with: tuple[tuple, dict] | None = None

    @classmethod
    def download_files_from_huggingface(
        cls,
        repo: str,
        cache_dir: str,
        extra_patterns: list[str],
        local_files_only: bool = False,
        **kwargs,
    ) -> str:
        cls.called_with = ((repo, cache_dir, tuple(extra_patterns)), {**kwargs, "local_files_only": local_files_only})
        return "result"


def test_fastembed_patch_forces_offline(monkeypatch):
    class OfflineModelManagement(_DummyModelManagement):
        called_with: tuple[tuple, dict] | None = None

    _install_fastembed_module(monkeypatch, OfflineModelManagement)
    monkeypatch.setenv("DOCINT_FASTEMBED_OFFLINE", "1")

    fastembed_patch = importlib.import_module("docint.utils.fastembed_patch")
    importlib.reload(fastembed_patch)

    fastembed_patch.ensure_fastembed_offline_patch()

    OfflineModelManagement.download_files_from_huggingface("repo", "cache", ["pattern"], local_files_only=False)

    assert OfflineModelManagement.called_with is not None
    _, kwargs = OfflineModelManagement.called_with
    assert kwargs["local_files_only"] is True


def test_fastembed_patch_respects_online(monkeypatch):
    class OnlineModelManagement(_DummyModelManagement):
        called_with: tuple[tuple, dict] | None = None

    _install_fastembed_module(monkeypatch, OnlineModelManagement)
    monkeypatch.delenv("DOCINT_FASTEMBED_OFFLINE", raising=False)

    fastembed_patch = importlib.import_module("docint.utils.fastembed_patch")
    importlib.reload(fastembed_patch)

    fastembed_patch.ensure_fastembed_offline_patch()

    OnlineModelManagement.download_files_from_huggingface("repo", "cache", ["pattern"], local_files_only=False)

    assert OnlineModelManagement.called_with is not None
    _, kwargs = OnlineModelManagement.called_with
    assert kwargs["local_files_only"] is False
