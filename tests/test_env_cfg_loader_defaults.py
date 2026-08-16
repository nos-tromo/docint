"""Tests that a loader's ``default_*`` parameter is the value it actually falls back to.

Three loaders used to hardcode a literal in the ``os.getenv`` call while still
advertising a ``default_*`` parameter beside it. A caller passing anything
other than the hardcoded value was ignored silently — the parameter read as
configuration and behaved as decoration, which is the worst of both: no error,
no effect, and a signature that documents a lie.

These tests pin the contract for every boolean loader default in `env_cfg`:
unset env var means the caller's parameter wins, and the env var still
overrides it when set.
"""

from __future__ import annotations

import pytest

from docint.utils.env_cfg import (
    load_ingestion_env,
    load_pipeline_config,
    load_resolution_env,
)


class TestHierarchicalChunking:
    """``HIERARCHICAL_CHUNKING_ENABLED`` / ``default_hierarchical_chunking_enabled``."""

    def test_caller_default_is_honoured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An explicit False from the caller survives an unset env var."""
        monkeypatch.delenv("HIERARCHICAL_CHUNKING_ENABLED", raising=False)
        assert load_ingestion_env(default_hierarchical_chunking_enabled=False).hierarchical_chunking_enabled is False

    def test_shipped_default_is_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With no caller override and no env var, chunking stays hierarchical."""
        monkeypatch.delenv("HIERARCHICAL_CHUNKING_ENABLED", raising=False)
        assert load_ingestion_env().hierarchical_chunking_enabled is True

    def test_env_var_still_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The operator's value beats the caller's default, in both directions."""
        monkeypatch.setenv("HIERARCHICAL_CHUNKING_ENABLED", "false")
        assert load_ingestion_env(default_hierarchical_chunking_enabled=True).hierarchical_chunking_enabled is False
        monkeypatch.setenv("HIERARCHICAL_CHUNKING_ENABLED", "true")
        assert load_ingestion_env(default_hierarchical_chunking_enabled=False).hierarchical_chunking_enabled is True


class TestForceReprocess:
    """``PIPELINE_FORCE_REPROCESS`` / ``default_force_reprocess``."""

    def test_caller_default_is_honoured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A caller asking for a forced re-run gets one without setting the env var."""
        monkeypatch.delenv("PIPELINE_FORCE_REPROCESS", raising=False)
        assert load_pipeline_config(default_force_reprocess=True).force_reprocess is True

    def test_shipped_default_reuses_artifacts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Nothing reprocesses by default — cached artifacts are the point."""
        monkeypatch.delenv("PIPELINE_FORCE_REPROCESS", raising=False)
        assert load_pipeline_config().force_reprocess is False

    def test_env_var_still_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The operator's value beats the caller's default, in both directions."""
        monkeypatch.setenv("PIPELINE_FORCE_REPROCESS", "true")
        assert load_pipeline_config(default_force_reprocess=False).force_reprocess is True
        monkeypatch.setenv("PIPELINE_FORCE_REPROCESS", "false")
        assert load_pipeline_config(default_force_reprocess=True).force_reprocess is False


class TestAutoResolve:
    """``RES_AUTO_RESOLVE`` / ``default_auto_resolve`` (the parameter was missing entirely)."""

    def test_caller_default_is_honoured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A caller can turn the in-job resolution stage off without an env var."""
        monkeypatch.delenv("RES_AUTO_RESOLVE", raising=False)
        assert load_resolution_env(default_auto_resolve=False).auto_resolve is False

    def test_shipped_default_is_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Resolution runs inside the ingest job unless told otherwise."""
        monkeypatch.delenv("RES_AUTO_RESOLVE", raising=False)
        assert load_resolution_env().auto_resolve is True

    def test_env_var_still_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The operator's value beats the caller's default, in both directions."""
        monkeypatch.setenv("RES_AUTO_RESOLVE", "false")
        assert load_resolution_env(default_auto_resolve=True).auto_resolve is False
        monkeypatch.setenv("RES_AUTO_RESOLVE", "true")
        assert load_resolution_env(default_auto_resolve=False).auto_resolve is True


class TestStreamingReaders:
    """The one that was only ever a docstring lie — behaviour was already right."""

    def test_streaming_is_on_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The loader's docstring said "default false" against a default of True."""
        monkeypatch.delenv("STREAMING_READERS_ENABLED", raising=False)
        assert load_ingestion_env().streaming_readers_enabled is True

    def test_caller_default_is_honoured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """This one already honoured its parameter; pin it so it stays that way."""
        monkeypatch.delenv("STREAMING_READERS_ENABLED", raising=False)
        assert load_ingestion_env(default_streaming_readers_enabled=False).streaming_readers_enabled is False


def test_a_typo_reads_as_false_not_as_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Booleans are truthy-list membership, so `on` is false — including against a True default.

    Worth pinning because it is the surprising half of the rule: setting a
    variable to a plausible-looking value does not leave the default in place,
    it actively selects false.
    """
    monkeypatch.setenv("HIERARCHICAL_CHUNKING_ENABLED", "on")
    assert load_ingestion_env(default_hierarchical_chunking_enabled=True).hierarchical_chunking_enabled is False
