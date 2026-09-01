"""Tests for ExtractConfig, the extracts directory and extract concurrency."""

from __future__ import annotations

from pathlib import Path

import pytest

from docint.utils.env_cfg import load_extract_concurrency, load_extract_env, load_path_env

_VARS = (
    "EXTRACT_DIR",
    "EXTRACT_RETENTION_DAYS",
    "EXTRACT_MAX_PER_COLLECTION",
    "EXTRACT_PDF_MAX_UNITS",
    "EXTRACT_PDF_MAX_FIGURES",
    "EXTRACT_SYNC_MAX_UNITS",
    "DOCINT_EXTRACT_CONCURRENCY",
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear every extract variable so an ambient ``.env`` cannot skew defaults."""
    for var in _VARS:
        monkeypatch.delenv(var, raising=False)


def test_defaults() -> None:
    """The shipped defaults are the documented ones."""
    cfg = load_extract_env()
    assert cfg.retention_days == 7
    assert cfg.max_per_collection == 5
    assert cfg.pdf_max_units == 200
    assert cfg.pdf_max_figures == 400
    assert cfg.sync_max_units == 50


def test_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every knob reads its own variable."""
    monkeypatch.setenv("EXTRACT_RETENTION_DAYS", "30")
    monkeypatch.setenv("EXTRACT_MAX_PER_COLLECTION", "2")
    monkeypatch.setenv("EXTRACT_PDF_MAX_UNITS", "10")
    monkeypatch.setenv("EXTRACT_PDF_MAX_FIGURES", "20")
    monkeypatch.setenv("EXTRACT_SYNC_MAX_UNITS", "5")
    cfg = load_extract_env()
    assert (cfg.retention_days, cfg.max_per_collection) == (30, 2)
    assert (cfg.pdf_max_units, cfg.pdf_max_figures, cfg.sync_max_units) == (10, 20, 5)


def test_unparseable_values_fall_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """A misconfigured value keeps the default rather than crashing an export."""
    monkeypatch.setenv("EXTRACT_PDF_MAX_UNITS", "lots")
    assert load_extract_env().pdf_max_units == 200


def test_negative_caps_clamp_to_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """A negative cap means "never", not a nonsense comparison."""
    monkeypatch.setenv("EXTRACT_PDF_MAX_UNITS", "-3")
    assert load_extract_env().pdf_max_units == 0


def test_retention_below_one_clamps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retention under a day would delete an extract the user just built."""
    monkeypatch.setenv("EXTRACT_RETENTION_DAYS", "0")
    assert load_extract_env().retention_days == 1


def test_extracts_dir_defaults_under_docint_home(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no override the store sits beside the other pipeline directories."""
    paths = load_path_env()
    assert paths.extracts == paths.docint_home_dir / "extracts"


def test_extracts_dir_reads_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``EXTRACT_DIR`` relocates the store, as compose does onto its volume."""
    monkeypatch.setenv("EXTRACT_DIR", str(tmp_path / "e"))
    assert load_path_env().extracts == tmp_path / "e"


def test_concurrency_defaults_to_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """One export at a time, so a bundle render cannot starve an ingest."""
    assert load_extract_concurrency() == 1


def test_concurrency_reads_env_and_clamps(monkeypatch: pytest.MonkeyPatch) -> None:
    """The variable raises the bound; a sub-1 value can never disable the pool."""
    monkeypatch.setenv("DOCINT_EXTRACT_CONCURRENCY", "3")
    assert load_extract_concurrency() == 3
    monkeypatch.setenv("DOCINT_EXTRACT_CONCURRENCY", "0")
    assert load_extract_concurrency() == 1
