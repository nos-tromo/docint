"""Tests that replayed session history resolves sources in the pinned collection.

Citations persist only a ``node_id``; the source text is rehydrated from the
docstore at read time. That lookup is collection-scoped, so replaying a
conversation without binding its pinned collection resolves against the
ambient (usually empty) collection and silently yields sources with no text —
the citation panel then shows a filename and a score but no evidence.
"""

from collections.abc import Generator, Iterator
from contextlib import contextmanager
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from agent_fakes import an_analysis
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from docint.core.state.base import Base
from docint.core.state.conversation import Conversation
from docint.core.state.session_manager import SessionManager

CHUNK_TEXT = "NLP is a subfield of artificial intelligence."
PINNED_COLLECTION = "uabc123__testbatch"


class _ScopedSourceRag:
    """RAG stand-in whose node lookup only resolves inside the right scope.

    Mirrors the real :class:`~docint.core.rag.RAG` contract: the active
    collection is per-context state bound by ``collection_scope``, and
    ``get_source_by_node_id`` reads the store of whatever collection is
    active at call time.

    Attributes:
        active_collection: The currently bound physical collection name.
        session_store: Session store path read by ``SessionManager``.
    """

    def __init__(self) -> None:
        """Initialize the stand-in with no collection bound."""
        self.active_collection: str = ""
        self.session_store: str = ""

    @property
    def index(self) -> Any:
        """Return the index of the active collection, if it holds one.

        The docstore is per collection, so an index only resolves while the
        owning collection is bound.

        Returns:
            Any: A docstore-bearing index stub, or ``None`` outside the scope.
        """
        if self.active_collection != PINNED_COLLECTION:
            return None
        node = MagicMock()
        node.text = CHUNK_TEXT
        index = MagicMock()
        index.storage_context.docstore.get_node.return_value = node
        return index

    @property
    def qdrant_collection(self) -> str:
        """Return the collection bound by the innermost active scope.

        Returns:
            str: The active physical collection name.
        """
        return self.active_collection

    @contextmanager
    def reasoning_scope(self, enabled: bool | None) -> Iterator[None]:
        """No-op mirror of :meth:`RAG.reasoning_scope`; the stub has no model to switch."""
        _ = enabled

        yield

    @contextmanager
    def collection_scope(self, physical: str) -> Iterator[None]:
        """Bind ``physical`` as the active collection for the block.

        Args:
            physical (str): Physical collection name to activate.

        Yields:
            None: Control returns with the scope active.
        """
        previous = self.active_collection
        self.active_collection = physical
        try:
            yield
        finally:
            self.active_collection = previous

    def get_source_by_node_id(self, node_id: str, *, score: float | None = None) -> dict[str, Any] | None:
        """Resolve a node id against the active collection.

        Args:
            node_id (str): Persisted citation node id.
            score (float | None): Retrieval score to echo back.

        Returns:
            dict[str, Any] | None: The source payload, or ``None`` when the
            active collection does not hold the node.
        """
        if self.active_collection != PINNED_COLLECTION:
            return None
        return {"text": CHUNK_TEXT, "preview_text": CHUNK_TEXT[:280], "node_id": node_id, "score": score}


@pytest.fixture
def scoped_rag() -> _ScopedSourceRag:
    """Provide the collection-aware RAG stand-in.

    Returns:
        _ScopedSourceRag: The stand-in instance.
    """
    return _ScopedSourceRag()


@pytest.fixture
def session_manager(scoped_rag: _ScopedSourceRag) -> Generator[SessionManager, None, None]:
    """SessionManager on an in-memory DB wired to the scoped RAG stand-in.

    Args:
        scoped_rag (_ScopedSourceRag): The RAG stand-in fixture.

    Returns:
        Generator[SessionManager, None, None]: The session manager.
    """
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    sm = SessionManager(rag=cast(Any, scoped_rag))
    sm._SessionMaker = sessionmaker(bind=engine)
    yield sm
    engine.dispose()


def _persist_turn_with_citation(sm: SessionManager, session_id: str, owner: str) -> None:
    """Persist one turn carrying a single node-id citation.

    Args:
        sm (SessionManager): The session manager under test.
        session_id (str): Conversation id to write into.
        owner (str): Owning principal.
    """
    with sm._session_scope() as s:
        conv = sm._load_or_create_convo(s, session_id, owner)
        conv.collection_name = PINNED_COLLECTION
        s.commit()

    node = MagicMock()
    node.node_id = "node-1"
    node.metadata = {"file_name": "python_documentation.md"}
    source_node = MagicMock()
    source_node.node = node
    source_node.score = 0.5

    resp_mock = MagicMock()
    resp_mock.metadata = cast(dict[str, Any], {})
    resp_mock.source_nodes = [source_node]
    sm._persist_turn(
        session_id,
        "Gibt es Dokumente über KI?",
        resp_mock,
        {"response": "Ja.", "reasoning": None},
        owner=owner,
    )


def test_history_rehydrates_source_text_in_the_pinned_collection(
    session_manager: SessionManager,
) -> None:
    """Replayed history carries the chunk text, not an empty citation.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    _persist_turn_with_citation(session_manager, "sess-a", "alice")

    messages = session_manager.get_session_history("sess-a", "alice")

    assistant = next(m for m in messages if m["role"] == "assistant")
    sources = cast(list[dict[str, Any]], assistant["sources"])
    assert [s["text"] for s in sources] == [CHUNK_TEXT]
    assert sources[0]["preview_text"]


def test_exported_transcript_quotes_the_source_excerpt(
    session_manager: SessionManager,
    tmp_path: Any,
) -> None:
    """The exported transcript quotes chunk text, not the unavailable marker.

    Args:
        session_manager (SessionManager): The session manager fixture.
        tmp_path (Any): Pytest temporary directory.
    """
    _persist_turn_with_citation(session_manager, "sess-a", "alice")

    with session_manager._session_scope() as s:
        conv = s.get(Conversation, "sess-a")
        assert conv is not None
        session_manager._export_transcript(tmp_path, conv, "")

    transcript = (tmp_path / "transcript.md").read_text(encoding="utf-8")
    assert CHUNK_TEXT in transcript
    assert "[source text unavailable]" not in transcript


def test_history_lookup_does_not_leak_the_scope(
    session_manager: SessionManager,
    scoped_rag: _ScopedSourceRag,
) -> None:
    """The pinned collection is unbound again once history has been read.

    Args:
        session_manager (SessionManager): The session manager fixture.
        scoped_rag (_ScopedSourceRag): The RAG stand-in fixture.
    """
    _persist_turn_with_citation(session_manager, "sess-a", "alice")

    session_manager.get_session_history("sess-a", "alice")

    assert scoped_rag.active_collection == ""


#: An analysis id, as ``_finish_agent_turn`` mints one.
ANALYSIS_ID = "a" * 32


def _persist_turn_with_analysis(sm: SessionManager, session_id: str, owner: str) -> None:
    """Persist one turn that kept an analysis, in a session pinned to the collection.

    Args:
        sm (SessionManager): The session manager under test.
        session_id (str): Conversation id to write into.
        owner (str): Owning principal.
    """
    with sm._session_scope() as s:
        conv = sm._load_or_create_convo(s, session_id, owner)
        conv.collection_name = PINNED_COLLECTION
        s.commit()

    resp_mock = MagicMock()
    resp_mock.metadata = cast(dict[str, Any], {})
    resp_mock.source_nodes = cast(list[Any], [])
    sm._persist_turn(
        session_id,
        "Wer schreibt am meisten über den Hafen?",
        resp_mock,
        {"response": "user-a [1]."},
        owner=owner,
        analyses=[{**an_analysis(), "id": ANALYSIS_ID}],
    )


def test_an_analysis_is_found_only_by_its_owner(session_manager: SessionManager) -> None:
    """Another owner, another session or an unknown id finds nothing; the owner also learns the pinned collection.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    _persist_turn_with_analysis(session_manager, "session-1", "user-a")
    with session_manager._session_scope() as s:
        session_manager._load_or_create_convo(s, "session-2", "user-a")

    stored = session_manager.get_analysis("session-1", ANALYSIS_ID, "user-a")
    found = session_manager.find_analysis(ANALYSIS_ID, "user-a")

    assert stored is not None and stored["id"] == ANALYSIS_ID
    assert (stored["provenance"]["session_id"], stored["provenance"]["turn_idx"]) == ("session-1", 0)
    assert found == (stored, PINNED_COLLECTION)
    assert session_manager.get_analysis("session-1", ANALYSIS_ID, "user-b") is None
    assert session_manager.find_analysis(ANALYSIS_ID, "user-b") is None
    assert session_manager.get_analysis("session-2", ANALYSIS_ID, "user-a") is None
    assert session_manager.get_analysis("session-1", "b" * 32, "user-a") is None
    assert session_manager.find_analysis("b" * 32, "user-a") is None


def _statements(sm: SessionManager) -> list[tuple[str, Any]]:
    """Record the SQL the session manager sends from now on.

    Args:
        sm (SessionManager): The session manager under test.

    Returns:
        list[tuple[str, Any]]: Each statement and its parameters, filled in as they are sent.
    """
    sent: list[tuple[str, Any]] = []

    def _record(_conn: Any, _cursor: Any, statement: str, parameters: Any, *_rest: Any) -> None:
        sent.append((statement, parameters))

    maker = sm._SessionMaker
    assert maker is not None
    event.listen(maker.kw["bind"], "before_cursor_execute", _record)
    return sent


def test_another_owners_analysis_is_not_read_at_all(session_manager: SessionManager) -> None:
    """The owner, and for one session its id, are in the query's WHERE clause: a foreign row is never loaded.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    _persist_turn_with_analysis(session_manager, "session-1", "user-a")
    sent = _statements(session_manager)

    assert session_manager.find_analysis(ANALYSIS_ID, "user-b") is None
    assert session_manager.get_analysis("session-1", ANALYSIS_ID, "user-b") is None

    assert len(sent) == 2
    assert all("user-b" in parameters for _statement, parameters in sent)
    assert "session-1" in sent[1][1] and "session-1" not in sent[0][1]


def test_a_legacy_unowned_analysis_is_found_by_no_owner_only(session_manager: SessionManager) -> None:
    """An analysis of a session with no owner is read for ``None`` and for no one else.

    Args:
        session_manager (SessionManager): The session manager fixture.
    """
    with session_manager._session_scope() as s:
        session_manager._load_or_create_convo(s, "session-1", None)
        s.commit()
    session_manager._persist_turn(
        "session-1",
        "Wer schreibt am meisten über den Hafen?",
        MagicMock(metadata={}, source_nodes=[]),
        {"response": "Niemand."},
        analyses=[{**an_analysis(), "id": ANALYSIS_ID}],
    )

    found = session_manager.find_analysis(ANALYSIS_ID, None)

    assert found is not None and found[0]["id"] == ANALYSIS_ID
    assert session_manager.get_analysis("session-1", ANALYSIS_ID, None) == found[0]
    assert session_manager.find_analysis(ANALYSIS_ID, "user-a") is None
    assert session_manager.get_analysis("session-1", ANALYSIS_ID, "user-a") is None
