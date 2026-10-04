"""Tests for stance-aware hate-speech detection on document chunks."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, cast

import httpx
import openai
import pytest

import docint.core.ingest.ingestion_pipeline as pipeline_module
from docint.core.ingest.hate_speech import CHUNK_CATEGORIES, CHUNK_STANCES, CONFIDENCE_LEVELS
from docint.core.ingest.ingestion_pipeline import DocumentIngestionPipeline


def _chunk_reply(stance: str, **overrides: Any) -> str:
    """Serialize a document-chunk verdict the way the stance-aware prompt asks for it.

    Args:
        stance (str): The author's stance.
        **overrides (Any): Field overrides.

    Returns:
        str: The JSON reply.
    """
    payload: dict[str, Any] = {
        "target": "refugees",
        "reason": "Dehumanizes a group.",
        "stance": stance,
        "category": "ethnicity",
        "confidence": "high",
    }
    payload.update(overrides)
    return json.dumps(payload)


def _status_error(status_code: int, message: str) -> openai.APIStatusError:
    """Build the SDK error a provider rejection raises.

    Args:
        status_code (int): HTTP status.
        message (str): Error message.

    Returns:
        openai.APIStatusError: The error.
    """
    request = httpx.Request("POST", "http://inference.invalid/v1/chat/completions")
    return openai.APIStatusError(message, response=httpx.Response(status_code, request=request), body=None)


# ---------------------------------------------------------------------------
# _parse_hate_speech_reply
# ---------------------------------------------------------------------------


def test_parse_flags_only_an_endorsing_author() -> None:
    """The verdict comes from stance: endorsement is hate, condemnation is not."""
    endorsed = pipeline_module._parse_hate_speech_reply(_chunk_reply("endorses", confidence="Medium"))
    condemned = pipeline_module._parse_hate_speech_reply(_chunk_reply("condemns_or_counters"))

    assert endorsed == {
        "hate_speech": True,
        "category": "ethnicity",
        "confidence": "medium",
        "reason": "Dehumanizes a group.",
    }
    assert condemned is not None
    assert condemned["hate_speech"] is False


@pytest.mark.parametrize(
    "payload",
    [
        '{"hate_speech": "false", "category": "ethnicity", "confidence": "high", "reason": "x"}',
        '{"hate_speech": true, "category": "ethnicity", "confidence": "high", "reason": "x"}',
        _chunk_reply("quotes_or_reports"),
        _chunk_reply("analyzes_or_discusses"),
        _chunk_reply("unclear"),
        _chunk_reply("none", category="none"),
    ],
)
def test_parse_never_flags_without_an_endorsing_stance(payload: str) -> None:
    """A stray boolean (even the string "false") or a non-endorsing stance is never a positive.

    Args:
        payload (str): A model reply.
    """
    parsed = pipeline_module._parse_hate_speech_reply(payload)

    assert parsed is not None
    assert parsed["hate_speech"] is False


def test_parse_maps_unknown_categories_to_other() -> None:
    """Categories outside the GMF enum are normalised instead of passed through."""
    parsed = pipeline_module._parse_hate_speech_reply(_chunk_reply("endorses", category="racism"))

    assert parsed is not None
    assert parsed["category"] == "other"


# ---------------------------------------------------------------------------
# Document-chunk requests: schema, fallbacks, coarse nodes
# ---------------------------------------------------------------------------


class _RecordingModel:
    """Stand-in for the llama-index chat model that records ``complete`` kwargs."""

    def __init__(self, respond: Any) -> None:
        """Store the scripted responder.

        Args:
            respond (Any): Callable ``(prompt, kwargs) -> str`` that may raise.
        """
        self.respond = respond
        self.calls: list[dict[str, Any]] = []

    def complete(self, prompt: str, **kwargs: Any) -> SimpleNamespace:
        """Record the call and return the scripted reply.

        Args:
            prompt (str): The rendered prompt.
            **kwargs (Any): Request keyword arguments.

        Returns:
            SimpleNamespace: A response exposing ``text``.
        """
        self.calls.append({"prompt": prompt, **kwargs})
        return SimpleNamespace(text=self.respond(prompt, kwargs))


def _pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, model: _RecordingModel) -> DocumentIngestionPipeline:
    """Build a pipeline with hate-speech detection on and every loader stubbed.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        tmp_path (Path): Temporary directory for the pipeline.
        model (_RecordingModel): The hate-speech model.

    Returns:
        DocumentIngestionPipeline: The pipeline.
    """

    class FakeNERConfig:
        enabled = False
        max_chars = 256
        max_workers = 1

    class FakeHateSpeechConfig:
        enabled = True
        max_chars = 512
        max_workers = 1
        window_tokens = 1000
        context_tokens = 300

    class FakeIngestionConfig:
        ingestion_batch_size = 2
        sentence_splitter_chunk_size = 512
        sentence_splitter_chunk_overlap = 64
        supported_filetypes: ClassVar[list[str]] = []
        hierarchical_chunking_enabled = False
        coarse_chunk_size = 1024
        fine_chunk_size = 256
        fine_chunk_overlap = 32
        streaming_readers_enabled = False

    class FakeOpenAIPipeline:
        def load_prompt(self, kw: str) -> str:
            """Return a compact template per prompt keyword.

            Args:
                kw (str): The prompt keyword.

            Returns:
                str: The template.
            """
            return "Classify:\n{text}" if kw == "hate_speech" else "S({first_index}-{last_index}):\n{segments}"

    monkeypatch.setattr(pipeline_module, "load_ner_env", lambda: FakeNERConfig())
    monkeypatch.setattr(pipeline_module, "load_hate_speech_env", lambda: FakeHateSpeechConfig())
    monkeypatch.setattr(pipeline_module, "load_ingestion_env", lambda: FakeIngestionConfig())
    monkeypatch.setattr(pipeline_module, "OpenAIPipeline", FakeOpenAIPipeline)
    pipeline = DocumentIngestionPipeline(
        data_dir=tmp_path, ner_model=None, progress_callback=None, hate_speech_model=cast(Any, model)
    )
    pipeline.entity_extractor = None
    return pipeline


def _node(text: str, node_id: str, **metadata: Any) -> SimpleNamespace:
    """Build a node stub.

    Args:
        text (str): Node text.
        node_id (str): Node id.
        **metadata (Any): Extra metadata.

    Returns:
        SimpleNamespace: The node.
    """
    return SimpleNamespace(text=text, node_id=node_id, metadata={"file_path": "doc.pdf", **metadata})


def test_chunk_requests_carry_a_json_schema(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Document chunks are classified under a strict single-verdict schema.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        tmp_path (Path): Temporary directory.
    """
    model = _RecordingModel(lambda prompt, kwargs: _chunk_reply("endorses"))
    pipeline = _pipeline(monkeypatch, tmp_path, model)
    nodes: list[Any] = [_node("Some chunk.", "n-1")]

    pipeline._enrich_nodes_in_place(nodes)

    response_format = model.calls[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert set(response_format["json_schema"]["schema"]["required"]) == {
        "target",
        "reason",
        "stance",
        "category",
        "confidence",
    }
    assert nodes[0].metadata["hate_speech"]["hate_speech"] is True


def test_a_rejected_schema_falls_back_to_unconstrained_requests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A provider rejecting ``response_format`` gets the chunk again without it, and never again.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        tmp_path (Path): Temporary directory.
    """

    def respond(prompt: str, kwargs: dict[str, Any]) -> str:
        if "response_format" in kwargs:
            raise _status_error(400, "response_format is not supported")
        return _chunk_reply("endorses")

    model = _RecordingModel(respond)
    pipeline = _pipeline(monkeypatch, tmp_path, model)
    nodes: list[Any] = [_node("One.", "n-1"), _node("Two.", "n-2")]

    pipeline._enrich_nodes_in_place(nodes)

    assert ["response_format" in call for call in model.calls] == [True, False, False]
    assert all(node.metadata["hate_speech"]["hate_speech"] for node in nodes)


def test_an_unparseable_structured_reply_is_retried_unconstrained(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A constrained reply that cannot be parsed is retried once without the schema.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        tmp_path (Path): Temporary directory.
    """
    model = _RecordingModel(lambda prompt, kwargs: "" if "response_format" in kwargs else _chunk_reply("endorses"))
    pipeline = _pipeline(monkeypatch, tmp_path, model)
    nodes: list[Any] = [_node("One.", "n-1")]

    pipeline._enrich_nodes_in_place(nodes)

    assert ["response_format" in call for call in model.calls] == [True, False]
    assert nodes[0].metadata["hate_speech"]["hate_speech"] is True


def test_coarse_parent_chunks_are_not_classified(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Coarse hierarchical parents are never stored as vectors, so they cost no request.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        tmp_path (Path): Temporary directory.
    """
    model = _RecordingModel(lambda prompt, kwargs: _chunk_reply("endorses"))
    pipeline = _pipeline(monkeypatch, tmp_path, model)
    nodes: list[Any] = [
        _node("Parent text.", "coarse-1", docint_hier_type="coarse"),
        _node("Child text.", "fine-1", docint_hier_type="fine"),
    ]

    pipeline._enrich_nodes_in_place(nodes)

    assert [call["prompt"] for call in model.calls] == ["Classify:\nChild text."]
    assert "hate_speech" not in nodes[0].metadata


# ---------------------------------------------------------------------------
# hate_speech prompt contract (en + de)
# ---------------------------------------------------------------------------

_PROMPT_DIR = Path(pipeline_module.__file__).resolve().parents[2] / "utils" / "prompts"


@pytest.mark.parametrize("locale", ["en", "de"])
def test_chunk_prompt_asks_for_the_stance_verdict_the_parser_reads(locale: str) -> None:
    """The per-chunk prompt names every enum value the parser accepts and carries one text slot.

    Args:
        locale (str): Prompt locale directory.
    """
    template = (_PROMPT_DIR / locale / "hate_speech.txt").read_text(encoding="utf-8")

    assert template.count("{text}") == 1
    for value in (*CHUNK_STANCES, *CHUNK_CATEGORIES, *CONFIDENCE_LEVELS):
        assert value in template, value
    assert '"hate_speech"' not in template


def test_german_chunk_prompt_does_not_exempt_the_term_it_uses_for_slurs() -> None:
    """The not-GMF list must not name "Schimpfwörter", which the GMF list uses for slurs."""
    template = (_PROMPT_DIR / "de" / "hate_speech.txt").read_text(encoding="utf-8")
    not_gmf = next(line for line in template.splitlines() if line.startswith("Keine GMF"))

    assert "Schimpfw" not in not_gmf


@pytest.mark.parametrize("locale", ["en", "de"])
def test_chunk_prompt_output_example_is_not_a_verdict(locale: str) -> None:
    """A model that copies the output template verbatim must not produce a finding.

    Args:
        locale (str): Prompt locale directory.
    """
    template = (_PROMPT_DIR / locale / "hate_speech.txt").read_text(encoding="utf-8")
    example = next(line for line in template.splitlines() if line.startswith('{"target"'))

    parsed = pipeline_module._parse_hate_speech_reply(example)

    assert parsed is None or parsed["hate_speech"] is False


@pytest.mark.parametrize("stance", ["Endorses.", "endorsed"])
def test_chunk_parser_reads_unconstrained_stances(stance: str) -> None:
    """Unconstrained inflections of the stance count, as in the window parser.

    Args:
        stance (str): The stance as an unconstrained model wrote it.
    """
    parsed = pipeline_module._parse_hate_speech_reply(_chunk_reply(stance))

    assert parsed is not None
    assert parsed["hate_speech"] is True


def test_chunk_parser_reads_a_list_wrapped_verdict() -> None:
    """A verdict wrapped in a list (a common unconstrained quirk) is still read."""
    parsed = pipeline_module._parse_hate_speech_reply("[" + _chunk_reply("endorses") + "]")

    assert parsed is not None
    assert parsed["hate_speech"] is True


def test_chunk_parser_reports_the_endorsement_in_a_fenced_verdict_list() -> None:
    """A fenced per-statement list is judged by its endorsing verdict, not by its first item."""
    reply = (
        "```json\n["
        + _chunk_reply("none", target="", reason="Insults one person.", category="none")
        + ", "
        + _chunk_reply(
            "endorses", reason="Calls for violence against <Gruppe>.", category="religion", confidence="medium"
        )
        + "]\n```"
    )

    parsed = pipeline_module._parse_hate_speech_reply(reply)

    assert parsed == {
        "hate_speech": True,
        "category": "religion",
        "confidence": "medium",
        "reason": "Calls for violence against <Gruppe>.",
    }


@pytest.mark.parametrize(
    ("first", "second"),
    [("condemns_or_counters", "endorses"), ("endorses", "condemns_or_counters")],
)
def test_chunk_parser_flags_a_bare_verdict_list_with_any_endorsement(first: str, second: str) -> None:
    """An unfenced list is aggregated the same way: one endorsement anywhere makes the chunk a finding.

    Args:
        first (str): Stance of the first listed verdict.
        second (str): Stance of the second listed verdict.
    """
    reply = "[" + _chunk_reply(first) + ", " + _chunk_reply(second) + "]"

    parsed = pipeline_module._parse_hate_speech_reply(reply)

    assert parsed is not None
    assert parsed["hate_speech"] is True


def test_chunk_parser_skips_bracketed_prose_before_the_verdict() -> None:
    """A bracket in the prose is not a verdict list; the object after it is still read."""
    parsed = pipeline_module._parse_hate_speech_reply("Statement [1] decides it: " + _chunk_reply("endorses"))

    assert parsed is not None
    assert parsed["hate_speech"] is True


def test_chunk_parser_reads_an_empty_list_as_no_verdict() -> None:
    """A list holding no verdict object is unparseable, never a crash."""
    assert pipeline_module._parse_hate_speech_reply("[]") is None
