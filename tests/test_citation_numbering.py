"""Tests that chat sources carry the same number the generator saw.

Answers refer to "source 1", "source 2". Until the snippets are numbered in
the prompt those ordinals are the model's own count of the order it happened
to receive them in, and nothing pins them to a card in the chat window. These
tests cover the number's whole path: stamped onto the final node set, rendered
into the LLM's view of the snippet, carried onto the normalized source, and
replayed from persisted citations.
"""

from collections.abc import Generator, Iterator
from contextlib import contextmanager
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from llama_index.core.schema import MetadataMode, NodeWithScore, TextNode
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from docint.agents.generation import ResultValidationResponseAgent
from docint.core.rag import RAG, CitationNumberingPostprocessor
from docint.core.state.base import Base
from docint.core.state.session_manager import SessionManager


def _hit(node_id: str, text: str, metadata: dict[str, Any] | None = None) -> NodeWithScore:
    """Build a scored node stand-in.

    Args:
        node_id (str): The node's stable id.
        text (str): The chunk body.
        metadata (dict[str, Any] | None): Optional node metadata.

    Returns:
        NodeWithScore: The wrapped node.
    """
    return NodeWithScore(node=TextNode(id_=node_id, text=text, metadata=dict(metadata or {})), score=0.5)


def test_numbering_stamps_a_one_based_index_in_node_order() -> None:
    """Each node in the synthesized set gets its position, counting from one."""
    nodes = [_hit("a", "first"), _hit("b", "second"), _hit("c", "third")]

    out = CitationNumberingPostprocessor()._postprocess_nodes(nodes, None)

    assert [n.node.metadata["citation_index"] for n in out] == [1, 2, 3]


def test_numbering_does_not_mutate_the_incoming_nodes() -> None:
    """The docstore's cached parents must not inherit one query's numbering."""
    hit = _hit("a", "first")

    CitationNumberingPostprocessor()._postprocess_nodes([hit], None)

    assert "citation_index" not in hit.node.metadata


def test_numbering_preserves_node_identity_score_and_text() -> None:
    """Only the number is added; everything a citation needs survives."""
    hit = _hit("a", "first", {"filename": "handbook.pdf", "page": 26})

    out = CitationNumberingPostprocessor()._postprocess_nodes([hit], None)

    assert out[0].node.node_id == "a"
    assert out[0].node.get_content(metadata_mode=MetadataMode.NONE) == "first"
    assert out[0].score == 0.5
    assert out[0].node.metadata["filename"] == "handbook.pdf"
    assert out[0].node.metadata["page"] == 26


def test_the_number_reaches_the_llms_view_of_the_snippet() -> None:
    """The model reads its number rather than counting the snippets itself.

    The synthesizer splices each node in via ``get_content(MetadataMode.LLM)``;
    a key excluded from that rendering would leave the prompt unnumbered no
    matter what the metadata says.
    """
    hit = _hit("a", "first", {"filename": "handbook.pdf", "entities": ["noise"]})
    hit.node.excluded_llm_metadata_keys = ["entities"]

    out = CitationNumberingPostprocessor()._postprocess_nodes([hit], None)

    rendered = out[0].node.get_content(metadata_mode=MetadataMode.LLM)
    assert "citation_index: 1" in rendered
    assert "noise" not in rendered


def test_prompt_visible_metadata_whitelists_the_number() -> None:
    """The LLM-visible whitelist admits the number.

    ``ParentContextPostprocessor`` derives its exclusion list from this set,
    so a number added before that runs would otherwise be stripped from the
    prompt.
    """
    from docint.core.rag import LLM_VISIBLE_METADATA_KEYS

    assert "citation_index" in LLM_VISIBLE_METADATA_KEYS


def test_normalized_source_carries_the_number() -> None:
    """The number rides the source payload out to the chat window."""
    payload: dict[str, Any] = {"filename": "handbook.pdf", "citation_index": 3}

    src = RAG._source_from_payload(collection="c", payload=payload, node_id="node-1")

    assert src["citation_index"] == 3


def test_source_without_a_number_omits_the_key() -> None:
    """Image sources are retrieved after generation and were never numbered."""
    src = RAG._source_from_payload(collection="c", payload={"filename": "shot.png"})

    assert "citation_index" not in src


def test_validator_headers_use_the_carried_number() -> None:
    """The validator numbers sources the way the answer does.

    It re-enumerated its own truncated slice, so a dropped or unnumbered
    source shifted every header after it out of step with the answer.
    """
    agent = ResultValidationResponseAgent.__new__(ResultValidationResponseAgent)
    sources: list[dict[str, Any]] = [
        {"filename": "handbook.pdf", "text": "gate", "citation_index": 2},
        {"filename": "notes.md", "text": "key", "citation_index": 5},
    ]

    text, _ = agent._sources_to_text(sources)

    assert "Source 2 [handbook.pdf" in text
    assert "Source 5 [notes.md" in text
    assert "Source 1 [" not in text


def test_validator_falls_back_to_position_when_unnumbered() -> None:
    """Sources with no carried number still get stable headers."""
    agent = ResultValidationResponseAgent.__new__(ResultValidationResponseAgent)
    sources: list[dict[str, Any]] = [{"filename": "shot.png", "text": "a"}, {"filename": "b.md", "text": "b"}]

    text, _ = agent._sources_to_text(sources)

    assert "Source 1 [shot.png" in text
    assert "Source 2 [b.md" in text


class _ReplayRag:
    """RAG stand-in for history replay, resolving any node id.

    Attributes:
        session_store: Session store path read by ``SessionManager``.
    """

    def __init__(self) -> None:
        """Initialize the stand-in."""
        self.session_store: str = ""
        self.qdrant_collection: str = "uabc123__testbatch"

    @contextmanager
    def collection_scope(self, physical: str) -> Iterator[None]:
        """Bind ``physical`` as the active collection for the block.

        Args:
            physical (str): Physical collection name to activate.

        Yields:
            None: Control returns with the scope active.
        """
        yield

    def get_source_by_node_id(self, node_id: str, *, score: float | None = None) -> dict[str, Any]:
        """Resolve a node id into a source payload.

        Args:
            node_id (str): Persisted citation node id.
            score (float | None): Retrieval score to echo back.

        Returns:
            dict[str, Any]: The rehydrated source payload.
        """
        return {"text": f"body of {node_id}", "preview_text": f"body of {node_id}", "node_id": node_id, "score": score}


@pytest.fixture
def replay_manager() -> Generator[SessionManager, None, None]:
    """SessionManager on an in-memory DB wired to the replay stand-in.

    Returns:
        Generator[SessionManager, None, None]: The session manager.
    """
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    sm = SessionManager(rag=cast(Any, _ReplayRag()))
    sm._SessionMaker = sessionmaker(bind=engine)
    yield sm
    engine.dispose()


def _persist_turn(sm: SessionManager, session_id: str, owner: str, node_ids: list[str]) -> None:
    """Persist one turn citing ``node_ids`` in order.

    Args:
        sm (SessionManager): The session manager under test.
        session_id (str): Conversation id to write into.
        owner (str): Owning principal.
        node_ids (list[str]): Node ids to persist as citations, in the order
            the generator saw them.
    """
    source_nodes = []
    for node_id in node_ids:
        node = MagicMock()
        node.node_id = node_id
        node.metadata = {"file_name": f"{node_id}.md"}
        scored = MagicMock()
        scored.node = node
        scored.score = 0.5
        source_nodes.append(scored)

    resp = MagicMock()
    resp.metadata = cast(dict[str, Any], {})
    resp.source_nodes = source_nodes
    sm._persist_turn(session_id, "question?", resp, {"response": "answer", "reasoning": None}, owner=owner)


def test_replayed_history_numbers_sources_in_persisted_order(replay_manager: SessionManager) -> None:
    """A reloaded conversation numbers its citations the way the answer did.

    Citation rows are written in ``source_nodes`` order, which is the order
    the generator numbered. Without a number the replayed cards lose the link
    to the ordinals still sitting in the stored answer text.
    """
    _persist_turn(replay_manager, "sess-a", "alice", ["node-a", "node-b", "node-c"])

    messages = replay_manager.get_session_history("sess-a", "alice")

    assistant = next(m for m in messages if m["role"] == "assistant")
    sources = cast(list[dict[str, Any]], assistant["sources"])
    assert [s["citation_index"] for s in sources] == [1, 2, 3]
    assert [s["node_id"] for s in sources] == ["node-a", "node-b", "node-c"]
