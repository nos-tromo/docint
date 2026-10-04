"""NER and hate-speech results stay on the metadata dict, out of the embed and prompt text."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, cast

import pytest
from llama_index.core import Document
from llama_index.core.schema import BaseNode, MetadataMode, TextNode

import docint.core.ingest.ingestion_pipeline as pipeline_module
from docint.core.ingest.enrichment import set_enrichment
from docint.core.ingest.ingestion_pipeline import DocumentIngestionPipeline
from docint.core.readers.documents import CorePDFPipelineReader
from docint.core.readers.json import CustomJSONReader
from docint.core.storage.hierarchical import HierarchicalNodeParser

ENRICHMENT_KEYS = ("entities", "relations", "hate_speech")
# Sentinels appear in no chunk text, so finding one in a rendering means metadata leaked.
ENTITY = "Zephyrine Holdings"
REASON = "Sentinel verdict reason."
_WINDOW_TEMPLATE = "S({first_index}-{last_index}):\n{segments}\n{context_before}{context_after}{language}"


def _extractor(text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return one sentinel entity and relation for any text.

    Args:
        text (str): The chunk text.

    Returns:
        tuple[list[dict[str, Any]], list[dict[str, Any]]]: Entities and relations.
    """
    return (
        [{"text": ENTITY, "type": "org", "score": 0.9}],
        [{"source": ENTITY, "target": "hub", "label": "linked"}],
    )


def _reply(prompt: str) -> str:
    """Endorse every chunk, and the second row of every transcript window.

    Args:
        prompt (str): The rendered prompt.

    Returns:
        str: The JSON reply.
    """
    verdict = {"target": "refugees", "reason": REASON, "stance": "endorses", "category": "other", "confidence": "high"}
    if prompt.startswith("S("):
        return json.dumps({"findings": [{"index": 1, **verdict}]})
    return json.dumps(verdict)


def _pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, hierarchical: bool) -> DocumentIngestionPipeline:
    """Build a pipeline with real node parsers and stubbed NER and hate-speech models.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        tmp_path (Path): Temporary directory.
        hierarchical (bool): Whether hierarchical chunking is enabled.

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
        ingestion_batch_size = 64
        sentence_splitter_chunk_size = 256
        sentence_splitter_chunk_overlap = 0
        supported_filetypes: ClassVar[list[str]] = []
        hierarchical_chunking_enabled = hierarchical
        coarse_chunk_size = 1024
        fine_chunk_size = 256
        fine_chunk_overlap = 0
        streaming_readers_enabled = False

    class FakeOpenAIPipeline:
        def load_prompt(self, kw: str) -> str:
            """Return a compact template per prompt keyword.

            Args:
                kw (str): The prompt keyword.

            Returns:
                str: The template.
            """
            return _WINDOW_TEMPLATE if kw == "hate_speech_transcript" else "Classify:\n{text}"

    monkeypatch.setattr(pipeline_module, "load_ner_env", lambda: FakeNERConfig())
    monkeypatch.setattr(pipeline_module, "load_hate_speech_env", lambda: FakeHateSpeechConfig())
    monkeypatch.setattr(pipeline_module, "load_ingestion_env", lambda: FakeIngestionConfig())
    monkeypatch.setattr(pipeline_module, "OpenAIPipeline", FakeOpenAIPipeline)
    model = SimpleNamespace(complete=lambda prompt, **kwargs: SimpleNamespace(text=_reply(prompt)))
    pipeline = DocumentIngestionPipeline(
        data_dir=tmp_path, ner_model=None, progress_callback=None, hate_speech_model=cast(Any, model)
    )
    pipeline._load_node_parsers()
    pipeline.entity_extractor = _extractor
    return pipeline


def _assert_enrichment_hidden(nodes: list[BaseNode], *, expected: set[str]) -> None:
    """Assert enriched nodes keep their results but render none of them.

    Args:
        nodes (list[BaseNode]): Nodes after enrichment.
        expected (set[str]): Keys at least one node must carry.
    """
    carried = {key for node in nodes for key in ENRICHMENT_KEYS if key in node.metadata}
    assert carried == expected
    for node in nodes:
        for mode in (MetadataMode.EMBED, MetadataMode.LLM):
            rendered = node.get_content(metadata_mode=mode)
            assert ENTITY not in rendered and REASON not in rendered, (mode, node.metadata.get("docint_hier_type"))
            assert not any(f"{key}:" in rendered for key in ENRICHMENT_KEYS)


@pytest.mark.parametrize("hierarchical", [True, False])
def test_text_chunks_embed_without_their_enrichment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hierarchical: bool
) -> None:
    """Entities and the hate-speech verdict never reach the dense/sparse input or the prompt.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        tmp_path (Path): Temporary directory.
        hierarchical (bool): Whether hierarchical chunking is enabled.
    """
    pipeline = _pipeline(monkeypatch, tmp_path, hierarchical=hierarchical)
    text = " ".join(f"Sentence number {i} is complete and clear." for i in range(60))
    doc = Document(text=text, metadata={"file_type": "text/plain", "file_name": "a.txt", "file_hash": "h1"})

    nodes = pipeline._create_nodes([doc])

    _assert_enrichment_hidden(nodes, expected=set(ENRICHMENT_KEYS))
    fine = [n for n in nodes if n.metadata.get("docint_hier_type") == "fine"]
    assert bool(fine) == hierarchical
    for node in fine:
        assert node.get_content(metadata_mode=MetadataMode.EMBED) == node.get_content(metadata_mode=MetadataMode.NONE)


def test_transcript_window_findings_embed_without_their_enrichment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A segment flagged by the window pass keeps its reader's own exclusions and adds the verdict's.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        tmp_path (Path): Temporary directory.
    """
    transcript = tmp_path / "talk.docint.jsonl"
    transcript.write_text(
        "".join(
            json.dumps(
                {
                    "source_file": "talk.wav",
                    "language": "en",
                    "sentence_index": index,
                    "start_seconds": float(index * 5),
                    "end_seconds": float(index * 5 + 4),
                    "speaker": "Speaker 1",
                    "text": text,
                }
            )
            + "\n"
            for index, text in enumerate(["First line.", "Second line.", "Third line."])
        ),
        encoding="utf-8",
    )
    pipeline = _pipeline(monkeypatch, tmp_path, hierarchical=False)

    nodes = pipeline._create_nodes(list(CustomJSONReader(is_jsonl=True).iter_documents(transcript)))

    flagged = [n for n in nodes if "hate_speech" in n.metadata]
    assert [n.get_content() for n in flagged] == ["Second line."]
    assert "reference_metadata" in flagged[0].excluded_embed_metadata_keys
    _assert_enrichment_hidden(nodes, expected=set(ENRICHMENT_KEYS))


def test_pdf_children_and_mirrored_parents_hide_their_enrichment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The PDF lane's pass over its children and the reader's parent mirror write hidden keys too.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        tmp_path (Path): Temporary directory.
    """
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    body = " ".join(f"Sentence number {i} is complete and clear." for i in range(200))
    _docs, nodes = CorePDFPipelineReader._build_nodes(
        file_path=pdf_path,
        doc_id="hashZ",
        pipeline_version="2.0.0",
        chunks=[{"chunk_id": "c1", "text": body, "page_range": [0], "section_path": ["Intro"]}],
        hierarchical_node_parser=HierarchicalNodeParser(
            coarse_chunk_size=100_000, fine_chunk_size=1024, fine_chunk_overlap=0
        ),
    )

    pipeline = _pipeline(monkeypatch, tmp_path, hierarchical=True)

    CorePDFPipelineReader(data_dir=tmp_path, enrich_nodes=pipeline.enrich_nodes)._enrich_nodes(nodes)

    coarse = [n for n in nodes if n.metadata.get("docint_hier_type") == "coarse"]
    assert coarse and all(n.metadata.get("entities") for n in coarse)
    _assert_enrichment_hidden(nodes, expected=set(ENRICHMENT_KEYS))


def test_set_enrichment_extends_exclusions_without_duplicates() -> None:
    """Existing exclusions survive, a re-written key is listed once, and empty results write nothing."""
    node = TextNode(text="Body.", metadata={"file_name": "a.txt"}, excluded_embed_metadata_keys=["file_name"])

    set_enrichment(node, {"entities": [{"text": ENTITY}], "relations": []})
    set_enrichment(node, {"entities": [{"text": ENTITY}]})

    assert "relations" not in node.metadata
    assert node.excluded_embed_metadata_keys == ["file_name", "entities"]
    assert node.excluded_llm_metadata_keys == ["entities"]
    assert node.get_content(metadata_mode=MetadataMode.EMBED) == "Body."
