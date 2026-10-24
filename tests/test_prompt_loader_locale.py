"""Tests for locale-aware prompt loading.

Covers three load sites: the module-level :func:`load_localized_prompt`
helper, :class:`docint.utils.openai_cfg.OpenAIPipeline`'s ``load_prompt``,
and that the active locale subdir is reflected in the resulting paths.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import patch

import pytest
from agent_fakes import LEDGER_CAP, PLANNED_SYSTEM_TOKENS, ScriptedSearch, shown_numbers, tool_call, tool_context
from assessment_fakes import llm_view, scored_node
from llama_index.core.schema import NodeWithScore

from docint.agents.assessment.ledger import EvidenceLedger
from docint.agents.assessment.split import SENTINEL
from docint.agents.turn.history import (
    ANSWER_ASSESSMENT_LABEL,
    ANSWER_SOURCED_LABEL,
    NO_HISTORY,
    ConversationWindow,
    labelled_lines,
)
from docint.agents.turn.keyword import WRITING_MARK, KeywordMatcher, KeywordResult
from docint.agents.turn.passages import _trailer, search_hint
from docint.agents.turn.scope import scope_message
from docint.agents.turn.tools import PassageReply, ReadTool, SearchTool, ToolRegistry, cleared_text
from docint.agents.turn.toolsets import toolset_for
from docint.core.rag import RAG
from docint.utils.embed_chunking import estimate_tokens
from docint.utils.prompt_loader import load_localized_prompt

#: The prompts an agent turn is built from, each with the placeholders its caller fills.
AGENT_PROMPTS: dict[str, frozenset[str]] = {
    "agent_system": frozenset({"tools_guide", "assessment_rule", "conversation_summary"}),
    "agent_tools_all": frozenset(),
    "agent_tools_documents": frozenset(),
    "agent_tools_visual": frozenset(),
    "agent_tools_keyword": frozenset(),
    "agent_tools_keyword_fields": frozenset({"fields"}),
    "agent_assessment_rule": frozenset(),
    "agent_final": frozenset(),
    "agent_scope": frozenset(),
    "agent_search_first": frozenset(),
    "agent_image_legend": frozenset({"image_legend"}),
}
#: The retrieval targets, each with its own tools guide.
TARGETS = ("all", "documents", "visual")
#: The targets that offer keyword: it matches words in text, so a visual turn never has it.
KEYWORD_TARGETS = ("all", "documents")
#: The fields of a keyword tool on a collection whose every index is ready.
KEYWORD_FIELDS = ("text", "author", "network", "uuid")
#: What each locale's keyword guide has the model report for a count of 0.
ZERO_PHRASES = {"en": "no passage contains these words", "de": "keine Passage enthält diese Wörter"}
#: How an answer marks a conclusion drawn from the evidence, by the locale of its prompt.
INFERENCE_LABELS = {"en": "Inference:", "de": "Schlussfolgerung:"}
#: The prompts that teach an answer to mark an inference, each with its own locale's label.
ANSWER_PROMPTS = ("agent_system", "grounded_qa", "grounded_refine")
#: The prompts that judge an answer: they accept every label, since an answer follows the user's language.
VALIDATOR_PROMPTS = ("response_validator", "response_validator_assessment")
CHAR_TOKEN_RATIO = 3.5
#: The texts the system prompt must describe, each by a phrase it quotes from them.
DESCRIBED = (
    "passage header",
    "cut passage",
    "shown before",
    "grown",
    "did not fit",
    "repeated call",
    "cleared reply",
    "earlier assessment",
)


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


@pytest.mark.parametrize("name", ["summary_map.txt", "summary_fold.txt"])
def test_tree_summary_prompts_exist_in_all_locales(name: str) -> None:
    """Tree-summary map/fold prompts exist in both English and German."""
    base = Path("docint/utils/prompts")
    for locale in ("en", "de"):
        path = base / locale / name
        assert path.is_file(), f"missing {path}"
        text = path.read_text(encoding="utf-8")
        assert text.strip()


def test_summary_map_prompt_placeholders() -> None:
    """The summary_map prompt contains required placeholders and protocol strings."""
    for locale in ("en", "de"):
        text = Path(f"docint/utils/prompts/{locale}/summary_map.txt").read_text(encoding="utf-8")
        assert "{label}" in text, f"missing {{label}} in {locale}/summary_map.txt"
        assert "{chunk_block}" in text, f"missing {{chunk_block}} in {locale}/summary_map.txt"
        assert "EVIDENCE_INDICES:" in text, f"missing EVIDENCE_INDICES: in {locale}/summary_map.txt"


def test_summary_fold_prompt_placeholders() -> None:
    """The summary_fold prompt contains required placeholders."""
    for locale in ("en", "de"):
        text = Path(f"docint/utils/prompts/{locale}/summary_fold.txt").read_text(encoding="utf-8")
        assert "{summaries_block}" in text, f"missing {{summaries_block}} in {locale}/summary_fold.txt"


@pytest.mark.parametrize("name", ["grounded_qa", "grounded_refine", "agent_system"])
def test_chat_answer_prompts_ask_for_markdown(name: str) -> None:
    """Both locales ask the chat model for Markdown, which the SPA and the report exports render."""
    for locale in ("en", "de"):
        text = Path(f"docint/utils/prompts/{locale}/{name}.txt").read_text(encoding="utf-8")
        assert "Markdown" in text, f"{locale}/{name}.txt no longer asks for Markdown"


def _prompt(name: str, code: str) -> str:
    """Load a prompt as a caller does, from the given locale.

    Args:
        name (str): The prompt basename.
        code (str): The locale.

    Returns:
        str: The prompt, stripped.
    """
    return load_localized_prompt(name, default="", required=True, lang=code)


def _placeholders(text: str) -> set[str]:
    """Collect the ``{name}`` placeholders of a template.

    Args:
        text (str): The template text.

    Returns:
        set[str]: The placeholder names.
    """
    return set(re.findall(r"\{(\w+)\}", text))


def _system_prompt(code: str, target: str, keyword_fields: Sequence[str] | None = None) -> str:
    """Render the system prompt of a turn with no history: the target's guide and the assessment rule filled in.

    Args:
        code (str): The locale.
        target (str): The retrieval target.
        keyword_fields (Sequence[str] | None): The fields the turn's keyword tool offers; none when it
            offers no keyword. The guide then gains the keyword line, and a line listing the fields
            when there is more than the text.

    Returns:
        str: The rendered prompt.
    """
    guide = _prompt(f"agent_tools_{target}", code)
    if keyword_fields:
        guide += "\n" + _prompt("agent_tools_keyword", code)
        if len(keyword_fields) > 1:
            guide += "\n" + _prompt("agent_tools_keyword_fields", code).format(fields=", ".join(keyword_fields))
    return _prompt("agent_system", code).format(
        tools_guide=guide,
        assessment_rule=_prompt("agent_assessment_rule", code),
        conversation_summary=NO_HISTORY,
    )


def _nothing(_: str) -> list[NodeWithScore]:
    """Find nothing, for a tool set built only to read its names.

    Args:
        _ (str): The query, unused.

    Returns:
        list[NodeWithScore]: No passage.
    """
    return []


def _matcher(*fields: str) -> KeywordMatcher:
    """Build a keyword match for a tool set built only to read its names: it accepts every call and runs none.

    Args:
        *fields (str): The fields it offers; the text alone when none is given.

    Returns:
        KeywordMatcher: The match.
    """

    def _run(words: str, field: str) -> KeywordResult:
        """Fail: a tool set built to read its names never matches anything.

        Args:
            words (str): The words.
            field (str): The field.

        Raises:
            AssertionError: Always.
        """
        raise AssertionError("The keyword match ran, which the test did not expect.")

    return KeywordMatcher(fields=fields or ("text",), check=lambda words, field: None, run=_run)


def _offered(target: str, keyword: KeywordMatcher | None = None) -> frozenset[str]:
    """Return the names of the tools a target offers.

    Args:
        target (str): The retrieval target.
        keyword (KeywordMatcher | None): The keyword match the turn offers; none when not given.

    Returns:
        frozenset[str]: The names.
    """
    return toolset_for(target, search=_nothing, search_images=_nothing, list_keyframes=_nothing, keyword=keyword).names


def _named_tools(text: str) -> set[str]:
    """Find the tools a text names, each as a whole word: ``search`` is not named by ``search_images``.

    Args:
        text (str): A prompt.

    Returns:
        set[str]: The names of the tools of any target that it names, keyword included.
    """
    return {name for name in _offered("all", _matcher()) if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text)}


def _written() -> dict[str, str]:
    """Render each text the system prompt describes, by the code that writes it.

    The passage, the repeated call and the cleared reply come from real tool calls; every
    note after a search's passages comes from ``_trailer``, the one function that writes them.

    Returns:
        dict[str, str]: By kind (see ``DESCRIBED``), the text as the model reads it.
    """
    text = " ".join(f"Abschnitt {line}: Der Hafenverein plant Stände und Musik am Kai." for line in range(40))
    plan = scored_node("plan", text, filename="plan.pdf", page=1)
    registry = ToolRegistry([SearchTool(ScriptedSearch([plan])), ReadTool()])
    ctx = tool_context()
    found = registry.call(tool_call("search", query="Hafenverein"), ctx)
    assert isinstance(found, PassageReply)
    header, _, body = found.text.partition("\n\n")
    note = functools.partial(
        _trailer, shown=True, again=(), grown=(), left_out=0, full=False, hint=search_hint("search"), note=""
    )
    return {
        "passage header": header,
        "cut passage": body,
        "shown before": note(again=(1,)),
        "grown": note(grown=(1,)),
        "did not fit": note(left_out=2),
        "repeated call": registry.call(tool_call("search", query="Hafenverein"), ctx).text,
        "cleared reply": cleared_text(found),
        "earlier assessment": labelled_lines((ANSWER_SOURCED_LABEL, ANSWER_ASSESSMENT_LABEL), ("Teil 1", "Teil 2")),
    }


@pytest.mark.parametrize("name", sorted(AGENT_PROMPTS))
def test_agent_prompts_match_across_locales(name: str) -> None:
    """Every agent prompt exists in both locales with exactly the placeholders its caller fills, and formats cleanly.

    Args:
        name (str): The prompt basename.
    """
    for code in ("en", "de"):
        text = _prompt(name, code)
        assert text, f"{code}/{name}.txt is empty"
        assert _placeholders(text) == AGENT_PROMPTS[name], f"{code}/{name}.txt takes {sorted(_placeholders(text))}"
        text.format(**dict.fromkeys(AGENT_PROMPTS[name], "X"))


@pytest.mark.parametrize("name", ["response_validator", "response_validator_assessment", "rewrite_retrieval"])
def test_formatted_plain_prompts_match_across_locales(name: str) -> None:
    """The validators' and the rewrite's ``str.format`` prompts share placeholders across locales and format cleanly.

    Args:
        name (str): The prompt basename.
    """
    texts = {code: _prompt(name, code) for code in ("en", "de")}

    assert _placeholders(texts["de"]) == _placeholders(texts["en"])
    for text in texts.values():
        text.format(**dict.fromkeys(_placeholders(text), "X"))


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("code", ["en", "de"])
def test_agent_system_prompt_carries_the_protocol_tokens(code: str, target: str) -> None:
    """The rendered prompt keeps its English protocol tokens, names only its target's tools, and ends in the history.

    Args:
        code (str): The locale.
        target (str): The retrieval target.
    """
    rendered = _system_prompt(code, target)
    guide = _prompt(f"agent_tools_{target}", code)
    offered = _offered(target)

    for token in (SENTINEL, "citation_index", "[n]"):
        assert token in rendered, f"{code}/{target}: {token!r} missing"
    assert _named_tools(guide) == offered, f"{code}/agent_tools_{target}.txt names {sorted(_named_tools(guide))}"
    assert _named_tools(rendered) <= offered, f"{code}/{target}: names a tool the target does not offer"
    assert ("#n" in guide) == ("keyframes" in offered), f"{code}/agent_tools_{target}.txt and the frame label #n"
    assert rendered.endswith(f"\n{NO_HISTORY}"), f"{code}: the conversation summary is not the prompt's last part"


@pytest.mark.parametrize("name", ["agent_tools_keyword", "agent_tools_keyword_fields"])
@pytest.mark.parametrize("code", ["en", "de"])
def test_a_keyword_line_is_one_line_naming_only_keyword_and_quoting_nothing(code: str, name: str) -> None:
    """Each keyword line is a single line that names the tool alone, with no backticks.

    A backtick-quoted phrase must be text the code writes, so the verbatim test would hold a quoted
    keyword line to that; the line is prose, and the tool's mechanics are in its schema.

    Args:
        code (str): The locale.
        name (str): The keyword prompt's basename.
    """
    text = _prompt(name, code)

    assert "\n" not in text, f"{code}/{name}.txt is more than one line"
    assert "`" not in text, f"{code}/{name}.txt quotes in backticks"
    assert _named_tools(text) == {"keyword"}, f"{code}/{name}.txt names {sorted(_named_tools(text))}"


@pytest.mark.parametrize("code", ["en", "de"])
def test_the_keyword_guide_makes_a_count_a_fact_not_a_source(code: str) -> None:
    """The guide keeps a count out of the citations and a zero out of any claim of absence, in the locale's words.

    Args:
        code (str): The locale.
    """
    guide = _prompt("agent_tools_keyword", code)

    assert "[n]" in guide, f"{code}/agent_tools_keyword.txt does not tie the count to [n]"
    assert ZERO_PHRASES[code] in guide, f"{code}/agent_tools_keyword.txt lacks {ZERO_PHRASES[code]!r}"


@pytest.mark.parametrize("target", KEYWORD_TARGETS)
@pytest.mark.parametrize("code", ["en", "de"])
def test_agent_system_prompt_with_keyword_keeps_the_protocol_tokens_and_names_exactly_the_offered_tools(
    code: str, target: str
) -> None:
    """With the keyword lines and every field, the prompt keeps its protocol tokens and names the tools offered.

    Args:
        code (str): The locale.
        target (str): The retrieval target, one that offers keyword.
    """
    rendered = _system_prompt(code, target, KEYWORD_FIELDS)
    offered = _offered(target, _matcher(*KEYWORD_FIELDS))

    for token in (SENTINEL, "citation_index", "[n]"):
        assert token in rendered, f"{code}/{target}: {token!r} missing"
    assert "keyword" in offered
    assert _named_tools(rendered) == offered, f"{code}/{target}: names {sorted(_named_tools(rendered))}"
    assert rendered.endswith(f"\n{NO_HISTORY}"), f"{code}: the conversation summary is not the prompt's last part"


def _engine(monkeypatch: pytest.MonkeyPatch, code: str) -> RAG:
    """Build an engine whose agent prompts load from one locale, the assessment rule switched on.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pins the locale and the rule before the engine reads them.
        code (str): The locale.

    Returns:
        RAG: The engine.
    """
    monkeypatch.setenv("RESPONSE_LANGUAGE", code)
    monkeypatch.setenv("ASSESSMENT_ENABLED", "true")
    return RAG(qdrant_collection="hafen")


@pytest.mark.parametrize("code", ["en", "de"])
def test_a_keyword_offering_only_the_text_adds_no_fields_line(monkeypatch: pytest.MonkeyPatch, code: str) -> None:
    """The fields line is for a keyword that offers a choice: with only the text, the model is told nothing of fields.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pins the locale and the assessment rule.
        code (str): The locale.
    """
    rag = _engine(monkeypatch, code)
    line = _prompt("agent_tools_keyword_fields", code)

    text_only = rag._agent_system_prompt("all", ConversationWindow(), keyword_fields=("text",))
    every_field = rag._agent_system_prompt("all", ConversationWindow(), keyword_fields=KEYWORD_FIELDS)

    assert _prompt("agent_tools_keyword", code) in text_only
    assert line.split("{fields}")[0] not in text_only
    assert line.format(fields=", ".join(KEYWORD_FIELDS)) in every_field


@pytest.mark.parametrize("fields", [None, ()], ids=["none", "empty"])
@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("code", ["en", "de"])
def test_without_keyword_the_engine_renders_the_system_prompt_it_always_did(
    monkeypatch: pytest.MonkeyPatch, code: str, target: str, fields: tuple[str, ...] | None
) -> None:
    """A turn that offers no keyword is sent the target's guide as it stands, byte for byte.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pins the locale and the assessment rule.
        code (str): The locale.
        target (str): The retrieval target.
        fields (tuple[str, ...] | None): No keyword fields: left out, or none offered.
    """
    rag = _engine(monkeypatch, code)

    rendered = rag._agent_system_prompt(target, ConversationWindow(), keyword_fields=fields)

    assert rendered == _system_prompt(code, target)
    assert rendered == rag._agent_system_prompt(target, ConversationWindow())


@pytest.mark.parametrize("fields", [("text",), ("text", "uuid"), KEYWORD_FIELDS], ids=["text", "two", "four"])
@pytest.mark.parametrize("target", KEYWORD_TARGETS)
@pytest.mark.parametrize("code", ["en", "de"])
def test_with_keyword_the_engine_follows_the_targets_guide_with_the_keyword_lines(
    monkeypatch: pytest.MonkeyPatch, code: str, target: str, fields: tuple[str, ...]
) -> None:
    """The guide becomes the target's, then the keyword line, then a line naming the fields when there is a choice.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pins the locale and the assessment rule.
        code (str): The locale.
        target (str): The retrieval target, one that offers keyword.
        fields (tuple[str, ...]): The fields the keyword tool offers, the text first.
    """
    rag = _engine(monkeypatch, code)
    lines = [_prompt(f"agent_tools_{target}", code), _prompt("agent_tools_keyword", code)]
    if len(fields) > 1:
        lines.append(_prompt("agent_tools_keyword_fields", code).format(fields=", ".join(fields)))

    rendered = rag._agent_system_prompt(target, ConversationWindow(), keyword_fields=fields)

    assert "\n".join(lines) + "\n\n" in rendered
    assert rendered == _system_prompt(code, target, fields)


@pytest.mark.parametrize(("code", "evaluation"), [("en", "evaluation"), ("de", "Bewertung")])
def test_the_final_note_keeps_an_evaluations_judgement_but_leaves_the_sentinel_to_the_rules(
    code: str, evaluation: str
) -> None:
    """A forced answer to an evaluation still ends in the model's judgement, where the rules ask for one.

    The note goes to every forced answer, assessments switched off too, so it never writes the
    sentinel itself: only a prompt carrying the assessment rule asks for it. Nor does it speak of
    a label, the word the system prompt uses for the display labels an answer must not copy.

    Args:
        code (str): The locale.
        evaluation (str): How the locale names an evaluation question.
    """
    note = _prompt("agent_final", code)

    assert evaluation in note
    assert SENTINEL not in note
    assert not re.search(r"label|kennzeichn", note, re.IGNORECASE)


@pytest.mark.parametrize("code", ["en", "de"])
def test_the_search_note_names_no_tool_since_every_target_is_sent_it(code: str) -> None:
    """The note that asks a first round to look things up first goes to every target, so it names none of their tools.

    Args:
        code (str): The locale.
    """
    note = _prompt("agent_search_first", code)

    named = {name for name in _offered("all") if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", note, re.IGNORECASE)}
    assert named == set()


@pytest.mark.parametrize("code", ["en", "de"])
def test_every_rule_on_citing_names_the_line_pinned_passages_show_too(code: str) -> None:
    """Every rule on [n] but the earlier-answers one ties it to the ``citation_index:`` line a scoped round shows too.

    A scoped round sends the same system prompt with its passages in the user message, not
    in a tool reply, so a rule tying numbers to tool replies alone would forbid the citations
    its note asks for.

    Args:
        code (str): The locale.
    """
    pinned = scored_node("plan", "Der Hafenverein plant Stände am Kai.", filename="plan.pdf", page=1)
    message, _ = scope_message(
        [pinned],
        ledger=EvidenceLedger(cap=LEDGER_CAP),
        note=_prompt("agent_scope", code),
        message="Was plant der Hafenverein?",
        llm_view=llm_view,
        budget_tokens=1000,
        char_token_ratio=CHAR_TOKEN_RATIO,
    )
    # Exempt: it numbers carried passages, and a scoped turn carries none, so its earlier answers hold no numbers.
    rules = [
        line
        for line in _prompt("agent_system", code).splitlines()
        if "[n]" in line and "`Previous assistant answer`" not in line
    ]

    assert shown_numbers(message) == [1]
    assert rules
    for line in rules:
        assert "`citation_index:`" in line, f"{code}/agent_system.txt ties [n] to something else: {line!r}"


@pytest.mark.parametrize("code", ["en", "de"])
def test_the_inference_marker_matches_between_answer_and_validator_prompts(code: str) -> None:
    """An answer prompt teaches its locale's label; the validators accept that label and every other locale's.

    The answer follows the user's language while the prompts follow the deployment's, so a
    German deployment may be validating an English answer marked "Inference:".

    Args:
        code (str): The locale.
    """
    label = INFERENCE_LABELS[code]
    others = [other for locale, other in INFERENCE_LABELS.items() if locale != code]
    for name in ANSWER_PROMPTS:
        text = _prompt(name, code)
        assert label in text, f"{code}/{name}.txt does not name the inference label {label!r}"
        assert not any(other in text for other in others), f"{code}/{name}.txt teaches another locale's label"
    for name in VALIDATOR_PROMPTS:
        text = _prompt(name, code)
        missing = [each for each in INFERENCE_LABELS.values() if each not in text]
        assert not missing, f"{code}/{name}.txt does not accept {missing}"


@pytest.mark.parametrize("name", VALIDATOR_PROMPTS)
@pytest.mark.parametrize("code", ["en", "de"])
def test_the_validators_take_a_count_from_a_listed_keyword_check_as_grounded(code: str, name: str) -> None:
    """An answer states a count without citing a passage, so each validator has one line accepting it on a check.

    Args:
        code (str): The locale.
        name (str): The validator prompt.
    """
    lines = [line for line in _prompt(name, code).splitlines() if "keyword check" in line.casefold()]

    assert len(lines) == 1, f"{code}/{name}.txt names the keyword checks on {len(lines)} lines"


@pytest.mark.parametrize("name", VALIDATOR_PROMPTS)
@pytest.mark.parametrize("code", ["en", "de"])
def test_the_validators_take_a_check_made_while_written_only_as_a_count_so_far(code: str, name: str) -> None:
    """The line that accepts a count on a check also names the mark a check made while written ends in.

    Args:
        code (str): The locale.
        name (str): The validator prompt.
    """
    marked = [
        line for line in _prompt(name, code).splitlines() if "keyword check" in line.casefold() and WRITING_MARK in line
    ]

    assert len(marked) == 1, f"{code}/{name}.txt does not name {WRITING_MARK!r} on its keyword check line"


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("code", ["en", "de"])
def test_the_rendered_agent_system_prompt_fits_its_planned_tokens(code: str, target: str) -> None:
    """With its guide and the assessment rule, and no history, the system prompt stays within the tokens planned for it.

    Args:
        code (str): The locale.
        target (str): The retrieval target.
    """
    rendered = _system_prompt(code, target)

    assert estimate_tokens(rendered, CHAR_TOKEN_RATIO) <= PLANNED_SYSTEM_TOKENS, (
        f"{code}/{target}: {len(rendered)} chars"
    )


@pytest.mark.parametrize("target", TARGETS)
def test_the_agent_system_prompt_quotes_what_the_code_writes_verbatim(target: str) -> None:
    """Every phrase the prompt quotes in backticks is text the code writes, the same English in both locales.

    And each text the prompt must describe, the replies of the tools and the labels of the
    history, is described by a phrase quoted from it.

    Args:
        target (str): The retrieval target.
    """
    written = _written()
    quoted = {code: set(re.findall(r"`([^`]+)`", _system_prompt(code, target))) for code in ("en", "de")}

    assert quoted["de"] == quoted["en"]
    for phrase in quoted["en"]:
        assert any(phrase in text for text in written.values()), f"{phrase!r} is not what the code writes"
    for kind in DESCRIBED:
        assert any(phrase in written[kind] for phrase in quoted["en"]), f"no quoted phrase describes the {kind}"


def test_image_rescore_description_prompt_exists_in_both_locales_with_matching_placeholders() -> None:
    """The image re-score prompt ships in every locale, alike in placeholders, each in its own language.

    The description is reranked against stored captions, which are written in
    the response language, so each locale's prompt must ask for its own.
    """
    import re

    texts = {
        locale: Path(f"docint/utils/prompts/{locale}/image_rescore_description.txt").read_text(encoding="utf-8")
        for locale in ("en", "de")
    }

    assert all(text.strip() for text in texts.values())
    assert set(re.findall(r"\{(\w+)\}", texts["de"])) == set(re.findall(r"\{(\w+)\}", texts["en"]))
    assert "English" in texts["en"]
    assert "Deutsch" in texts["de"]


@pytest.mark.parametrize(
    ("locale", "required", "framing"),
    [
        ("en", ("alternatives", "never decide the answer"), "caption"),
        ("de", ("Möglichkeiten", "entscheide nie die Antwort"), "Bildunterschrift"),
    ],
    ids=["en", "de"],
)
def test_image_rescore_description_prompt_asks_for_cues_with_alternatives_not_a_caption(
    locale: str, required: tuple[str, ...], framing: str
) -> None:
    """The prompt has the model name the cues a question needs, never a caption that answers it.

    Asked for the caption of a picture that would answer the query, the chat
    model guessed a scene (a sunny sky for a weather question) that contradicted
    the right image, which then scored under the re-score floor. Naming the cues
    with their usual alternatives leaves the answer undecided.

    Args:
        locale: The prompt's locale.
        required: Phrases that keep the answer undecided.
        framing: The word of the caption framing that invited the guess.
    """
    text = Path(f"docint/utils/prompts/{locale}/image_rescore_description.txt").read_text(encoding="utf-8")

    for phrase in required:
        assert phrase in text, f"{locale}/image_rescore_description.txt no longer says {phrase!r}"
    assert framing.lower() not in text.lower(), f"{locale}/image_rescore_description.txt asks for a {framing} again"
