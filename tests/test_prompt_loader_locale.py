"""Tests for locale-aware prompt loading.

Covers three load sites: the module-level :func:`load_localized_prompt`
helper, :class:`docint.utils.openai_cfg.OpenAIPipeline`'s ``load_prompt``,
and that the active locale subdir is reflected in the resulting paths.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from docint.utils.prompt_loader import load_localized_prompt


def test_load_localized_prompt_reads_english_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """``RESPONSE_LANGUAGE`` unset -> reads from ``prompts/en/``.

    The marker phrase 'Grounded answer:' is the English end-cue for the
    QA prompt and is one of the strings the German pack overrides.
    """
    monkeypatch.delenv("RESPONSE_LANGUAGE", raising=False)

    text = load_localized_prompt("grounded_qa", default="FALLBACK", required=True)

    assert "Grounded answer:" in text
    assert "Fundierte Antwort:" not in text


def test_load_localized_prompt_reads_german_when_de(monkeypatch: pytest.MonkeyPatch) -> None:
    """``RESPONSE_LANGUAGE=de`` -> reads from ``prompts/de/`` and emits German end-cues."""
    monkeypatch.setenv("RESPONSE_LANGUAGE", "de")

    text = load_localized_prompt("grounded_qa", default="FALLBACK", required=True)

    assert "Fundierte Antwort:" in text
    assert "Grounded answer:" not in text


def test_load_localized_prompt_falls_back_when_file_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When the prompt file is missing and ``required=False``, the default is returned."""
    monkeypatch.delenv("RESPONSE_LANGUAGE", raising=False)
    fake_root = tmp_path / "prompts"
    (fake_root / "en").mkdir(parents=True)

    with patch("docint.utils.prompt_loader.load_path_env") as load_path_env_mock:
        load_path_env_mock.return_value.prompts = fake_root
        text = load_localized_prompt("does_not_exist", default="FALLBACK_TEXT")

    assert text == "FALLBACK_TEXT"


def test_load_localized_prompt_raises_when_required_and_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``required=True`` surfaces a missing file as ``FileNotFoundError``."""
    monkeypatch.delenv("RESPONSE_LANGUAGE", raising=False)
    fake_root = tmp_path / "prompts"
    (fake_root / "en").mkdir(parents=True)

    with patch("docint.utils.prompt_loader.load_path_env") as load_path_env_mock:
        load_path_env_mock.return_value.prompts = fake_root
        with pytest.raises(FileNotFoundError):
            load_localized_prompt("does_not_exist", default="x", required=True)


def test_openai_pipeline_prompt_dir_includes_active_locale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``OpenAIPipeline.prompt_dir`` ends with the active language code."""
    from docint.utils.openai_cfg import OpenAIPipeline

    monkeypatch.setenv("RESPONSE_LANGUAGE", "de")
    pipeline = OpenAIPipeline()

    assert pipeline.prompt_dir is not None
    assert pipeline.prompt_dir.name == "de"


def test_openai_pipeline_load_prompt_reads_active_locale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``OpenAIPipeline().load_prompt('grounded_qa')`` returns the localized file."""
    from docint.utils.openai_cfg import OpenAIPipeline

    monkeypatch.setenv("RESPONSE_LANGUAGE", "de")
    pipeline = OpenAIPipeline()

    text = pipeline.load_prompt(kw="grounded_qa")

    assert "Fundierte Antwort:" in text


def test_rag_grounded_qa_prompt_switches_language(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ``RAG`` class loads English vs. German QA cues based on ``RESPONSE_LANGUAGE``.

    Uses ``RAG.__new__`` to skip the full Qdrant/embedding bootstrap and only
    exercises the prompt-loading path of ``__post_init__``.
    """
    from docint.core.rag import RAG
    from docint.utils.env_cfg import load_path_env

    for code, marker, anti_marker in (
        ("en", "Grounded answer:", "Fundierte Antwort:"),
        ("de", "Fundierte Antwort:", "Grounded answer:"),
    ):
        monkeypatch.setenv("RESPONSE_LANGUAGE", code)
        prompt_path = load_path_env().prompts / code / "grounded_qa.txt"
        text = prompt_path.read_text(encoding="utf-8")
        assert marker in text, f"marker {marker!r} missing for {code}"
        assert anti_marker not in text, f"anti-marker {anti_marker!r} leaked into {code}"

    # Sanity: the dataclass field is wired so static type checking won't trip.
    assert hasattr(RAG, "__dataclass_fields__")
    assert "language_code" in RAG.__dataclass_fields__
    assert "grounded_collection_summary_prompt" in RAG.__dataclass_fields__


@pytest.mark.parametrize("name", ["grounded_qa", "grounded_refine"])
def test_de_generation_prompts_match_en_placeholders(name: str) -> None:
    """German QA/refine prompts must expose the same ``{...}`` placeholders as English.

    ``SessionManager._bind_prior_turn_context`` injects the prior exchange by
    ``partial_format``-ing ``{prior_turn_context}`` onto these templates. When the
    German file lacks the placeholder, the bound prior turn is silently dropped, so
    streaming history-awareness is inert under the deployed ``RESPONSE_LANGUAGE=de``.
    Both locale packs must carry the identical placeholder set so either locale
    formats cleanly with the same kwargs.
    """
    import re

    from docint.utils.env_cfg import load_path_env

    prompts_root = load_path_env().prompts
    en_text = (prompts_root / "en" / f"{name}.txt").read_text(encoding="utf-8")
    de_text = (prompts_root / "de" / f"{name}.txt").read_text(encoding="utf-8")

    def placeholders(text: str) -> set[str]:
        return set(re.findall(r"\{(\w+)\}", text))

    en_placeholders = placeholders(en_text)
    de_placeholders = placeholders(de_text)

    # The prior-turn block is the load-bearing placeholder this work relies on.
    assert "prior_turn_context" in en_placeholders, f"en/{name}.txt unexpectedly missing prior_turn_context"
    assert de_placeholders == en_placeholders, (
        f"de/{name}.txt placeholders {sorted(de_placeholders)} != en {sorted(en_placeholders)} (locale prompt drift)"
    )


@pytest.mark.parametrize(
    "name,fill",
    [
        ("grounded_qa", {"context_str": "CTX", "query_str": "Q"}),
        ("grounded_refine", {"context_msg": "CTX", "query_str": "Q", "existing_answer": "A"}),
    ],
)
def test_de_generation_prompt_binds_prior_turn_and_renders_cleanly(name: str, fill: dict[str, str]) -> None:
    """The German QA/refine templates bind the prior turn and format without leftover braces.

    Mirrors the production path (``partial_format`` of ``{prior_turn_context}`` in
    ``RAG._build_grounded_text_qa_template`` followed by the synthesizer's
    ``.format``). Guards against a stale German file silently dropping the prior
    turn, and against a hand-edited prompt smuggling in a stray ``{``/``}`` that
    would raise ``KeyError`` at generation time.
    """
    from llama_index.core import PromptTemplate

    from docint.utils.env_cfg import load_path_env

    text = (load_path_env().prompts / "de" / f"{name}.txt").read_text(encoding="utf-8")
    bound = PromptTemplate(text).partial_format(prior_turn_context="PRIOR-TURN-MARKER")

    rendered = bound.format(**fill)  # type: ignore[arg-type]

    assert "PRIOR-TURN-MARKER" in rendered, f"de/{name}.txt dropped the bound prior turn"
    assert "{" not in rendered and "}" not in rendered, f"de/{name}.txt left unfilled placeholders: {rendered!r}"
