"""Tests for the optional machine-translation block in report exports."""

from typing import Any

import pytest

from docint.core.state import report_render as rr
from docint.utils import csv_stream as cs


def _entity_snap(**extra: Any) -> dict[str, Any]:
    """Build a minimal entity-finding snapshot, optionally carrying a translation."""
    snap: dict[str, Any] = {"entity_label": "ACME", "filename": "f.pdf", "chunk_text": "original text"}
    snap.update(extra)
    return snap


def test_md_entity_renders_translation_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """An entity snapshot carrying a translation renders a labeled block after the chunk."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    snap = _entity_snap(translation={"text": "übersetzter Text", "target_lang": "de", "model": "m"})
    out = "\n".join(rr._md_entity(snap, None))
    assert "Machine translation" in out
    assert "übersetzter Text" in out
    assert "original text" in out  # original preserved


def test_md_entity_without_translation_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Absent translation field renders no translation block (byte-identical to today)."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    snap = _entity_snap()
    out = "\n".join(rr._md_entity(snap, None))
    assert "Machine translation" not in out


def test_html_hate_renders_translation_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hate-speech snapshot carrying a translation renders a labeled block in HTML."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "en")
    snap: dict[str, Any] = {
        "category": "X",
        "confidence": "high",
        "chunk_text": "orig",
        "translation": {"text": "übersetzt", "target_lang": "de", "model": "m"},
    }
    out = rr._html_hate(snap, None)
    assert "Machine translation" in out
    assert "übersetzt" in out


def test_ner_row_includes_translation() -> None:
    """A translated chunk carries its translated text into the entity CSV row."""
    row = cs.ner_source_row(
        {"chunk_id": "c1", "chunk_text": "orig", "translation": {"text": "übersetzt"}},
        entity_label="ACME",
    )
    assert "translation" in cs.NER_SOURCE_COLUMNS
    assert row["translation"] == "übersetzt"


def test_hate_row_translation_absent_is_blank() -> None:
    """Absent translation renders as a blank cell (existing rows stay unchanged)."""
    row = cs.hate_speech_row({"chunk_id": "c1", "chunk_text": "orig"})
    assert "translation" in cs.HATE_SPEECH_COLUMNS
    assert row["translation"] == ""
