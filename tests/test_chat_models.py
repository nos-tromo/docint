from sqlalchemy import inspect
from pathlib import Path

from docint.core.state.base import _make_session_maker
from docint.core.state.conversation import Conversation
from docint.core.state.turn import Turn
from docint.core.state.citation import Citation


def test_make_session_maker_creates_tables(tmp_path: Path) -> None:
    """
    Test the creation of database tables by the session maker.

    Args:
        tmp_path (Path): The temporary path fixture.
    """
    db_path = tmp_path / "sessions.db"
    maker = _make_session_maker(f"sqlite:///{db_path}")
    session = maker()
    try:
        inspector = inspect(session.get_bind())
        tables = set(inspector.get_table_names())
        assert {"conversations", "turns", "citations"}.issubset(tables)
    finally:
        session.close()


def test_conversation_relationships(tmp_path: Path) -> None:
    """
    Test the relationships between conversation turns and citations.

    Args:
        tmp_path (Path): The temporary path fixture.
    """
    maker = _make_session_maker(f"sqlite:///{tmp_path / 'db.sqlite'}")
    session = maker()
    try:
        convo = Conversation(id="session-1", rolling_summary="Summary")
        turn = Turn(idx=0, user_text="hello", model_response="hi")
        citation = Citation(filename="doc.pdf")
        turn.citations.append(citation)
        convo.turns.append(turn)
        session.add(convo)
        session.commit()

        stored = session.get(Conversation, "session-1")
        assert stored is not None
        assert stored.turns[0].citations[0].filename == "doc.pdf"
    finally:
        session.close()
