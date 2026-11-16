"""Tests for the locale-aware ``ui_string`` helper."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from docint.utils.ui_strings import UI_STRINGS, ui_string


def test_ui_string_returns_english_when_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ``RESPONSE_LANGUAGE`` set, ``ui_string`` returns the English entry."""
    monkeypatch.delenv("RESPONSE_LANGUAGE", raising=False)

    assert ui_string("clarify_generic") == "Could you clarify what you need?"


def test_ui_string_returns_german_when_de(monkeypatch: pytest.MonkeyPatch) -> None:
    """With ``RESPONSE_LANGUAGE=de`` the German variant is returned."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "de")

    assert ui_string("clarify_generic") == "Können Sie präzisieren, was Sie benötigen?"


def test_ui_string_table_is_complete_across_languages() -> None:
    """Every language must define the exact same key set as English."""
    english_keys = set(UI_STRINGS["en"])

    for lang, strings in UI_STRINGS.items():
        assert set(strings) == english_keys, f"language '{lang}' has divergent UI keys"


def test_ui_string_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown key surfaces as ``KeyError`` rather than silently returning empty."""
    monkeypatch.delenv("RESPONSE_LANGUAGE", raising=False)

    with pytest.raises(KeyError):
        ui_string("not_a_real_key")


def test_collection_overview_keys_present_both_locales() -> None:
    """All report collection-overview keys must be present in both en and de."""
    keys = {
        "report_section_collection_overview",
        "report_overview_documents",
        "report_overview_nodes",
        "report_overview_file_types",
        "report_overview_entity_types",
        "report_overview_col_document",
        "report_overview_col_type",
        "report_overview_col_units",
        "report_overview_col_hash",
    }
    assert keys <= set(UI_STRINGS["en"])
    assert keys <= set(UI_STRINGS["de"])
    assert UI_STRINGS["de"]["report_section_collection_overview"] == "Dokumentenübersicht"


def test_analysis_report_strings_are_worded_in_both_locales() -> None:
    """The analyses section's heading and labels, with the placeholders their renderers fill."""
    expected = {
        "en": {
            "report_section_analyses": "Analyses",
            "report_label_code": "Code",
            "report_label_output": "Output",
            "report_label_computed_from": "Computed from",
            "report_label_workbench_job": "workbench {version}, job {job}",
            "report_label_table_shape": "{rows} rows × {columns} columns",  # noqa: RUF001
            "report_label_table_cut": "first {shown} shown",
            "report_label_table_trimmed": "shown cut",
            "report_label_analysis_untitled": "Analysis",
        },
        "de": {
            "report_section_analyses": "Analysen",
            "report_label_code": "Code",
            "report_label_output": "Ausgabe",
            "report_label_computed_from": "Berechnet aus",
            "report_label_workbench_job": "workbench {version}, Job {job}",
            "report_label_table_shape": "{rows} Zeilen × {columns} Spalten",  # noqa: RUF001
            "report_label_table_cut": "die ersten {shown} gezeigt",
            "report_label_table_trimmed": "Anzeige gekürzt",
            "report_label_analysis_untitled": "Analyse",
        },
    }
    for lang, strings in expected.items():
        assert {key: UI_STRINGS[lang][key] for key in strings} == strings


_SPA_CATALOGS = Path(__file__).resolve().parents[1] / "frontend" / "src" / "i18n"


@pytest.mark.parametrize("lang", ["en", "de"])
def test_hate_speech_labels_match_the_spa_catalog(lang: str) -> None:
    """A finding's category and confidence read the same in the SPA and in every report export."""
    catalog = (_SPA_CATALOGS / f"{lang}.ts").read_text(encoding="utf-8")
    spa = dict(re.findall(r"'hate\.((?:category|confidence)_\w+)': '([^']*)'", catalog))
    exported = {
        key.removeprefix("hate_"): label
        for key, label in UI_STRINGS[lang].items()
        if key.startswith(("hate_category_", "hate_confidence_"))
    }
    assert len(spa) == 14
    assert exported == spa
