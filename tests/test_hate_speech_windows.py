"""Tests for classifying Nextext transcript segments in context windows."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, cast

import httpx
import openai
import pytest
from llama_index.core import Document
from typing_extensions import override

import docint.core.ingest.ingestion_pipeline as pipeline_module
from docint.core.ingest.hate_speech import (
    TranscriptLine,
    TranscriptWindow,
    next_window,
    parse_window_reply,
    render_window_prompt,
    window_response_format,
)
from docint.core.ingest.ingestion_pipeline import DocumentIngestionPipeline
from docint.core.jobs import JobCancelled
from docint.core.readers.json import CustomJSONReader

_TEMPLATE = "L={language}\nB:\n{context_before}\nS({first_index}-{last_index}):\n{segments}\nA:\n{context_after}"


def _window_reply(*items: tuple[int, str]) -> str:
    """Serialize ``(index, stance)`` pairs as a findings reply.

    Args:
        *items (tuple[int, str]): Row index and stance per item.

    Returns:
        str: The JSON reply.
    """
    return json.dumps(
        {
            "findings": [
                {
                    "index": index,
                    "target": "refugees",
                    "reason": "Dehumanizes a group.",
                    "stance": stance,
                    "category": "ethnicity",
                    "confidence": "high",
                }
                for index, stance in items
            ]
        }
    )


# ---------------------------------------------------------------------------
# Helpers (semantics shared with Nextext's nextext/core/hate_speech.py)
# ---------------------------------------------------------------------------


def test_next_window_packs_core_and_frames_it_with_context() -> None:
    """The core fits its budget; the adjacent rows are shown as read-only context."""
    lines = [TranscriptLine(index=i, text="xxxx") for i in range(5)]

    window = next_window(lines, 2, core_chars=9, context_chars=9)

    assert window == TranscriptWindow(context_start=1, core_start=2, core_end=3, context_end=4)


def test_small_cores_never_clip_a_row_below_the_per_row_floor() -> None:
    """A tiny core budget still shows each labelled segment whole (up to the old 2048-char cap)."""
    lines = [
        TranscriptLine(index=0, text="Die gehören alle weg, sagt er, und meint es ernst."),
        TranscriptLine(index=1, text="x"),
    ]
    window = next_window(lines, 0, core_chars=3, context_chars=0)

    prompt = render_window_prompt("{segments}", lines, window, core_chars=3, context_chars=0, language="de")

    assert (window.core_start, window.core_end) == (0, 1)
    assert prompt == "[0] Die gehören alle weg, sagt er, und meint es ernst."


def test_render_window_prompt_substitutes_once() -> None:
    """Placeholder-like transcript text is inserted verbatim, never substituted again."""
    lines = [TranscriptLine(index=0, text="Er sagte {segments}.", speaker="Speaker 1")]
    window = TranscriptWindow(context_start=0, core_start=0, core_end=1, context_end=1)

    prompt = render_window_prompt(
        "{segments}|{last_index}", lines, window, core_chars=100, context_chars=0, language="de"
    )

    assert prompt == "[0] Speaker 1: Er sagte {segments}.|0"


def test_parse_window_reply_keeps_endorsed_core_rows_only() -> None:
    """Condemnations and context rows are dropped."""
    raw = _window_reply((1, "condemns_or_counters"), (2, "endorses"), (7, "endorses"))

    findings = parse_window_reply(raw, allowed=[1, 2])

    assert findings == [{"index": 2, "category": "ethnicity", "confidence": "high", "reason": "Dehumanizes a group."}]


def test_parse_window_reply_salvages_every_complete_item_of_a_cut_off_reply() -> None:
    """A reply cut off mid-list keeps every complete item, not just the first one."""
    items = json.loads(_window_reply((3, "quotes_or_reports"), (4, "endorses")))["findings"]
    raw = '{"findings": [' + json.dumps(items[0]) + ", " + json.dumps(items[1]) + ', {"index": 5, "stan'

    assert [f["index"] for f in parse_window_reply(raw, allowed=[3, 4, 5]) or []] == [4]


def test_parse_window_reply_lets_an_endorsing_duplicate_win() -> None:
    """A segment listed as quoted and then endorsed is reported, whatever the order."""
    assert [
        f["index"] for f in parse_window_reply(_window_reply((7, "quotes_or_reports"), (7, "endorses")), [7]) or []
    ] == [7]
    assert [
        f["index"] for f in parse_window_reply(_window_reply((7, "endorses"), (7, "quotes_or_reports")), [7]) or []
    ] == [7]


@pytest.mark.parametrize(("stance", "flagged"), [("Endorses.", True), ("endorsed", True), ("does not endorse", False)])
def test_parse_window_reply_reads_unconstrained_stances(stance: str, flagged: bool) -> None:
    """Inflected endorsement counts; negations do not.

    Args:
        stance (str): The stance as an unconstrained model wrote it.
        flagged (bool): Whether the segment must be reported.
    """
    raw = _window_reply((1, "endorses")).replace('"endorses"', json.dumps(stance))

    assert bool(parse_window_reply(raw, allowed=[1])) is flagged


def test_parse_window_reply_without_readable_indices_is_unparseable() -> None:
    """Items that name the segment under another key cannot be read and must not pass as clean."""
    raw = json.dumps({"findings": [{"row": 3, "stance": "endorses"}]})

    assert parse_window_reply(raw, allowed=[3]) is None


def test_parse_window_reply_reports_unparseable_replies() -> None:
    """A reply without a findings structure is ``None``, not a clean window."""
    assert parse_window_reply("Sorry, I can't.", allowed=[0]) is None
    assert parse_window_reply('{"findings": []}', allowed=[0]) == []


def test_window_response_format_admits_only_core_indices() -> None:
    """The schema's index enum is the core, so enforcing backends cannot flag context rows."""
    schema = window_response_format([3, 4])["json_schema"]["schema"]

    assert schema["properties"]["findings"]["items"]["properties"]["index"]["enum"] == [3, 4]


# ---------------------------------------------------------------------------
# Pipeline wiring
# ---------------------------------------------------------------------------


class _RecordingModel:
    """Stand-in for the llama-index chat model that records ``complete`` calls."""

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


def _pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model: _RecordingModel,
    *,
    window_tokens: int = 1000,
    context_tokens: int = 300,
    batch_size: int = 2,
    transcript_prompt: bool = True,
    progress: list[str] | None = None,
) -> DocumentIngestionPipeline:
    """Build a pipeline with hate-speech detection on and every loader stubbed.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        tmp_path (Path): Temporary directory.
        model (_RecordingModel): The hate-speech model.
        window_tokens (int): ``HATE_SPEECH_WINDOW_TOKENS``.
        context_tokens (int): ``HATE_SPEECH_CONTEXT_TOKENS``.
        batch_size (int): ``INGESTION_BATCH_SIZE``.
        transcript_prompt (bool): Whether the transcript prompt is available.
        progress (list[str] | None): Collects progress messages when given.

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
        max_workers = 2
        window_tokens = 1000
        context_tokens = 300

    FakeHateSpeechConfig.window_tokens = window_tokens
    FakeHateSpeechConfig.context_tokens = context_tokens

    class FakeIngestionConfig:
        ingestion_batch_size = batch_size
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

            Raises:
                FileNotFoundError: When the transcript prompt is switched off.
            """
            if kw == "hate_speech_transcript":
                if not transcript_prompt:
                    raise FileNotFoundError(kw)
                return _TEMPLATE
            return "Classify:\n{text}"

    monkeypatch.setattr(pipeline_module, "load_ner_env", lambda: FakeNERConfig())
    monkeypatch.setattr(pipeline_module, "load_hate_speech_env", lambda: FakeHateSpeechConfig())
    monkeypatch.setattr(pipeline_module, "load_ingestion_env", lambda: FakeIngestionConfig())
    monkeypatch.setattr(pipeline_module, "OpenAIPipeline", FakeOpenAIPipeline)
    pipeline = DocumentIngestionPipeline(
        data_dir=tmp_path,
        ner_model=None,
        progress_callback=progress.append if progress is not None else None,
        hate_speech_model=cast(Any, model),
    )
    pipeline.entity_extractor = None
    return pipeline


def _segment(text: str, index: int, *, file_hash: str = "sha256:a", speaker: str | None = "Speaker 1") -> Any:
    """Build a transcript-segment node stub as docint's Nextext reader produces it.

    Args:
        text (str): Segment text.
        index (int): ``sentence_index``.
        file_hash (str): Source file hash (groups segments per transcript).
        speaker (str | None): Diarization label.

    Returns:
        Any: The node stub.
    """
    metadata: dict[str, Any] = {
        "docint_doc_kind": "transcript_segment",
        "sentence_index": index,
        "file_hash": file_hash,
        "file_path": f"{file_hash}.wav",
        "whisper_language": "de",
    }
    if speaker:
        metadata["speaker"] = speaker
    return SimpleNamespace(text=text, node_id=f"{file_hash}-{index}", metadata=metadata)


def test_segments_are_classified_in_order_with_their_neighbours(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One window per transcript, rows in sentence order, only the endorsing row flagged.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        tmp_path (Path): Temporary directory.
    """
    model = _RecordingModel(lambda prompt, kwargs: _window_reply((0, "quotes_or_reports"), (1, "endorses")))
    pipeline = _pipeline(monkeypatch, tmp_path, model)
    nodes = [
        _segment("Die gehören alle weg.", 1),
        _segment("In die Unterkunft sind viele Geflüchtete gezogen.", 0),
        _segment("Das ist menschenverachtend.", 2, speaker="Speaker 2"),
    ]

    monkeypatch.setattr(DocumentIngestionPipeline, "_create_nodes_without_enrichment", lambda self, docs: nodes)
    pipeline._create_nodes([Document(text="x")])

    assert len(model.calls) == 1
    assert model.calls[0]["prompt"] == (
        "L=de\nB:\n—\nS(0-2):\n[0] Speaker 1: In die Unterkunft sind viele Geflüchtete gezogen.\n"
        "[1] Speaker 1: Die gehören alle weg.\n[2] Speaker 2: Das ist menschenverachtend.\nA:\n—"
    )
    assert nodes[0].metadata["hate_speech"] == {
        "hate_speech": True,
        "category": "ethnicity",
        "confidence": "high",
        "reason": "Dehumanizes a group.",
        "chunk_id": "sha256:a-1",
        "chunk_text": "Die gehören alle weg.",
        "source_ref": "sha256:a.wav",
    }
    assert "hate_speech" not in nodes[1].metadata
    assert "hate_speech" not in nodes[2].metadata


def test_transcripts_are_windowed_per_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Segments of different files never share a window.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        tmp_path (Path): Temporary directory.
    """
    model = _RecordingModel(lambda prompt, kwargs: '{"findings": []}')
    pipeline = _pipeline(monkeypatch, tmp_path, model)
    nodes = [_segment("Eins.", 0, file_hash="sha256:a"), _segment("Zwei.", 0, file_hash="sha256:b")]

    monkeypatch.setattr(DocumentIngestionPipeline, "_create_nodes_without_enrichment", lambda self, docs: nodes)
    pipeline._create_nodes([Document(text="x")])

    assert sorted(call["prompt"].split("S(0-0):\n")[1].split("\n")[0] for call in model.calls) == [
        "[0] Speaker 1: Eins.",
        "[0] Speaker 1: Zwei.",
    ]


def test_windows_span_the_whole_transcript_despite_small_node_batches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Streaming enrichment batches nodes in twos; transcript windows still see every neighbour.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        tmp_path (Path): Temporary directory.
    """
    model = _RecordingModel(lambda prompt, kwargs: '{"findings": []}')
    # 4 tokens -> a 12-char core: one 9-char row per window, unclipped; 100 context tokens show every neighbour.
    pipeline = _pipeline(monkeypatch, tmp_path, model, window_tokens=4, context_tokens=100, batch_size=2)
    nodes = [_segment(text, i, speaker=None) for i, text in enumerate(["eins", "zwei", "drei", "vier", "fünf"])]
    monkeypatch.setattr(DocumentIngestionPipeline, "_create_nodes_without_enrichment", lambda self, docs: nodes)
    monkeypatch.setattr(DocumentIngestionPipeline, "_attach_clean_text", lambda self, docs: docs)
    monkeypatch.setattr(DocumentIngestionPipeline, "_ensure_file_hashes", lambda self, docs: docs)
    monkeypatch.setattr(DocumentIngestionPipeline, "_filter_docs_by_existing_hashes", lambda self, docs, hashes: docs)

    batches = list(pipeline._stream_processed_batch([Document(text="x")], None))

    assert sum(len(batch_nodes) for _, batch_nodes, _ in batches) == 5
    assert len(model.calls) == 5
    assert model.calls[2]["prompt"] == "L=de\nB:\n[0] eins\n[1] zwei\nS(2-2):\n[2] drei\nA:\n[3] vier\n[4] fünf"


def test_windowed_segments_skip_the_per_chunk_detector(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Transcript segments cost one window request, not an extra isolated per-chunk request each.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        tmp_path (Path): Temporary directory.
    """
    model = _RecordingModel(lambda prompt, kwargs: '{"findings": []}')
    pipeline = _pipeline(monkeypatch, tmp_path, model)
    document_chunk = SimpleNamespace(text="Ein Absatz.", node_id="doc-1", metadata={"file_path": "a.pdf"})
    nodes = [_segment("Eins.", 0), _segment("Zwei.", 1), document_chunk]

    monkeypatch.setattr(DocumentIngestionPipeline, "_create_nodes_without_enrichment", lambda self, docs: nodes)
    pipeline._create_nodes([Document(text="x")])

    prompts = [call["prompt"] for call in model.calls]
    assert len(prompts) == 2
    assert prompts[1] == "Classify:\nEin Absatz."


def test_without_the_transcript_prompt_segments_fall_back_to_the_chunk_detector(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A missing transcript prompt keeps today's per-segment behaviour rather than skipping detection.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        tmp_path (Path): Temporary directory.
    """
    model = _RecordingModel(
        lambda prompt, kwargs: '{"target": "", "reason": "", "stance": "none", "category": "none", "confidence": "low"}'
    )
    pipeline = _pipeline(monkeypatch, tmp_path, model, transcript_prompt=False)
    nodes = [_segment("Eins.", 0), _segment("Zwei.", 1)]

    monkeypatch.setattr(DocumentIngestionPipeline, "_create_nodes_without_enrichment", lambda self, docs: nodes)
    pipeline._create_nodes([Document(text="x")])

    assert [call["prompt"] for call in model.calls] == ["Classify:\nEins.", "Classify:\nZwei."]


def test_a_rejected_schema_is_retried_unconstrained_for_windows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Windows share the chunk path's fallback when the provider rejects ``response_format``.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        tmp_path (Path): Temporary directory.
    """

    def respond(prompt: str, kwargs: dict[str, Any]) -> str:
        if "response_format" in kwargs:
            request = httpx.Request("POST", "http://inference.invalid/v1/chat/completions")
            raise openai.APIStatusError(
                "json_schema unsupported", response=httpx.Response(400, request=request), body=None
            )
        return _window_reply((0, "endorses"))

    model = _RecordingModel(respond)
    pipeline = _pipeline(monkeypatch, tmp_path, model)
    nodes = [_segment("Die gehören alle weg.", 0)]

    monkeypatch.setattr(DocumentIngestionPipeline, "_create_nodes_without_enrichment", lambda self, docs: nodes)
    pipeline._create_nodes([Document(text="x")])

    assert ["response_format" in call for call in model.calls] == [True, False]
    assert nodes[0].metadata["hate_speech"]["hate_speech"] is True


def test_window_progress_uses_the_messages_the_spa_parses(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Progress keeps the ``Detecting hate speech: N/M chunks processed`` shape.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        tmp_path (Path): Temporary directory.
    """
    progress: list[str] = []
    model = _RecordingModel(lambda prompt, kwargs: '{"findings": []}')
    pipeline = _pipeline(monkeypatch, tmp_path, model, window_tokens=1, context_tokens=0, progress=progress)
    nodes = [_segment("Eins.", 0), _segment("Zwei.", 1)]

    monkeypatch.setattr(DocumentIngestionPipeline, "_create_nodes_without_enrichment", lambda self, docs: nodes)
    pipeline._create_nodes([Document(text="x")])

    assert progress[:2] == [
        "Detecting hate speech: 1/2 chunks processed",
        "Detecting hate speech: 2/2 chunks processed",
    ]


# ---------------------------------------------------------------------------
# Prompt parity with Nextext
# ---------------------------------------------------------------------------

_PROMPT_DIR = Path(pipeline_module.__file__).resolve().parents[2] / "utils" / "prompts"


@pytest.mark.parametrize("locale", ["en", "de"])
def test_transcript_prompt_exposes_every_placeholder(locale: str) -> None:
    """The transcript template carries every block the renderer fills.

    Args:
        locale (str): Prompt locale directory.
    """
    template = (_PROMPT_DIR / locale / "hate_speech_transcript.txt").read_text(encoding="utf-8")

    for name in ("{language}", "{context_before}", "{segments}", "{context_after}", "{first_index}", "{last_index}"):
        assert name in template, name


def test_transcript_prompts_are_pinned_to_nextexts_copy() -> None:
    """Docint and Nextext ship byte-identical transcript prompts; change both repos together."""
    digests = {
        locale: hashlib.sha256((_PROMPT_DIR / locale / "hate_speech_transcript.txt").read_bytes()).hexdigest()
        for locale in ("en", "de")
    }

    assert digests == _NEXTEXT_TRANSCRIPT_PROMPT_SHA256


_NEXTEXT_TRANSCRIPT_PROMPT_SHA256: dict[str, str] = {
    "en": "208e798b79810203d1d82398d760fe14807287d9caac8f741ca07ea38aabfda4",
    "de": "99703816dc6dbd4a213358dd6af940b238591867fdb97be8c7f6e4a13b62f576",
}
"""SHA-256 of Nextext's ``nextext/utils/prompts/<locale>/hate_speech_transcript.txt``."""


# ---------------------------------------------------------------------------
# Review fixes: cancellation, failure breaker, grouping, truncation, contract
# ---------------------------------------------------------------------------


def test_stop_cancels_the_windows_still_queued(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A cancelled job stops calling the model within one round of in-flight windows.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        tmp_path (Path): Temporary directory.
    """
    model = _RecordingModel(lambda prompt, kwargs: '{"findings": []}')

    def cancel(_: str) -> None:
        raise JobCancelled("stop requested")

    pipeline = _pipeline(monkeypatch, tmp_path, model, window_tokens=1, context_tokens=0)
    pipeline.progress_callback = cancel
    nodes = [_segment(f"Satz {i}.", i) for i in range(20)]
    monkeypatch.setattr(DocumentIngestionPipeline, "_create_nodes_without_enrichment", lambda self, docs: nodes)

    with pytest.raises(JobCancelled):
        pipeline._create_nodes([Document(text="x")])

    assert len(model.calls) <= pipeline.hate_speech_max_workers + 1


def test_consecutive_failures_stop_the_window_pass(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A dead endpoint is not called once per remaining window; the pass gives up and moves on.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        tmp_path (Path): Temporary directory.
    """

    def respond(prompt: str, kwargs: dict[str, Any]) -> str:
        raise ConnectionError("router unreachable")

    model = _RecordingModel(respond)
    pipeline = _pipeline(monkeypatch, tmp_path, model, window_tokens=1, context_tokens=0)
    nodes = [_segment(f"Satz {i}.", i) for i in range(20)]
    monkeypatch.setattr(DocumentIngestionPipeline, "_create_nodes_without_enrichment", lambda self, docs: nodes)

    pipeline._create_nodes([Document(text="x")])

    assert len(model.calls) <= 3 + pipeline.hate_speech_max_workers
    assert all("hate_speech" not in node.metadata for node in nodes)


def test_a_failed_window_is_not_resent_segment_by_segment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Segments of a window whose request failed are not re-judged alone by the per-chunk detector.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        tmp_path (Path): Temporary directory.
    """

    def respond(prompt: str, kwargs: dict[str, Any]) -> str:
        raise ConnectionError("router unreachable")

    model = _RecordingModel(respond)
    pipeline = _pipeline(monkeypatch, tmp_path, model)
    nodes = [_segment("Eins.", 0), _segment("Zwei.", 1)]
    monkeypatch.setattr(DocumentIngestionPipeline, "_create_nodes_without_enrichment", lambda self, docs: nodes)

    pipeline._create_nodes([Document(text="x")])

    assert len(model.calls) == 1


def test_reposted_clips_are_windowed_per_posting(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The same clip attached to two postings yields two transcripts, not one interleaved window.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        tmp_path (Path): Temporary directory.
    """
    model = _RecordingModel(lambda prompt, kwargs: '{"findings": []}')
    pipeline = _pipeline(monkeypatch, tmp_path, model)
    nodes = []
    for posting in ("p-1", "p-2"):
        for index, text in enumerate(["Guten Abend.", "Willkommen."]):
            node = _segment(text, index)
            node.node_id = f"{posting}-{index}"
            node.metadata["posting_uuid"] = posting
            nodes.append(node)
    monkeypatch.setattr(DocumentIngestionPipeline, "_create_nodes_without_enrichment", lambda self, docs: nodes)

    pipeline._create_nodes([Document(text="x")])

    assert len(model.calls) == 2
    assert all(call["prompt"].count("Guten Abend.") == 1 for call in model.calls)


def test_a_cut_off_window_reply_falls_back_to_one_segment_per_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A window reply stopped at the output cap is not trusted; its segments are asked one by one.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        tmp_path (Path): Temporary directory.
    """
    truncated = SimpleNamespace(choices=[SimpleNamespace(finish_reason="length")])

    class _TruncatingModel(_RecordingModel):
        """Cuts off the multi-segment window, answers single segments normally."""

        @override
        def complete(self, prompt: str, **kwargs: Any) -> SimpleNamespace:
            """Record the call and cut the reply off when the core holds several segments.

            Args:
                prompt (str): The rendered prompt.
                **kwargs (Any): Request keyword arguments.

            Returns:
                SimpleNamespace: A response exposing ``text`` and ``raw``.
            """
            self.calls.append({"prompt": prompt, **kwargs})
            if "S(0-2)" in prompt:
                return SimpleNamespace(text='{"findings": [{"index": 0, "sta', raw=truncated)
            reply = _window_reply((2, "endorses")) if "S(2-2)" in prompt else '{"findings": []}'
            return SimpleNamespace(text=reply, raw=SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop")]))

    model = _TruncatingModel(lambda prompt, kwargs: "")
    pipeline = _pipeline(monkeypatch, tmp_path, model)
    nodes = [_segment("Eins.", 0), _segment("Zwei.", 1), _segment("Die gehören alle weg.", 2)]
    monkeypatch.setattr(DocumentIngestionPipeline, "_create_nodes_without_enrichment", lambda self, docs: nodes)

    pipeline._create_nodes([Document(text="x")])

    assert len(model.calls) == 4
    assert model.calls[0]["max_tokens"] >= 1024
    assert "hate_speech" in nodes[2].metadata
    assert "hate_speech" not in nodes[0].metadata


def test_real_nextext_jsonl_reaches_the_window_pass(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Contract with the Nextext reader: its metadata keys drive grouping, order, speakers and language.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        tmp_path (Path): Temporary directory.
    """
    rows = [
        (0, "Speaker 1", "In die Unterkunft sind viele Geflüchtete gezogen."),
        (1, "Speaker 1", "Die gehören alle weg."),
        (2, "Speaker 2", "So redet man nicht über Menschen."),
    ]
    transcript = tmp_path / "talk.docint.jsonl"
    transcript.write_text(
        "".join(
            json.dumps(
                {
                    "source_file": "talk.wav",
                    "language": "de",
                    "sentence_index": index,
                    "start_seconds": float(index * 5),
                    "end_seconds": float(index * 5 + 4),
                    "start_ts": f"00:00:{index * 5:02d}",
                    "end_ts": f"00:00:{index * 5 + 4:02d}",
                    "speaker": speaker,
                    "text": text,
                },
                ensure_ascii=False,
            )
            + "\n"
            for index, speaker, text in rows
        ),
        encoding="utf-8",
    )
    model = _RecordingModel(lambda prompt, kwargs: _window_reply((1, "endorses")))
    pipeline = _pipeline(monkeypatch, tmp_path, model)
    pipeline.md_node_parser = cast(Any, SimpleNamespace(get_nodes_from_documents=lambda docs: []))
    pipeline.docling_node_parser = cast(Any, SimpleNamespace(get_nodes_from_documents=lambda docs: []))

    docs = list(CustomJSONReader(is_jsonl=True).iter_documents(transcript))
    nodes = pipeline._create_nodes(docs)

    assert len(model.calls) == 1
    assert (
        "[0] Speaker 1: In die Unterkunft sind viele Geflüchtete gezogen.\n"
        "[1] Speaker 1: Die gehören alle weg.\n[2] Speaker 2: So redet man nicht über Menschen."
    ) in model.calls[0]["prompt"]
    assert model.calls[0]["prompt"].startswith("L=de\n")
    flagged = [node.get_content() for node in nodes if "hate_speech" in (node.metadata or {})]
    assert flagged == ["Die gehören alle weg."]
