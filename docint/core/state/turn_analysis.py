"""A turn's analyses: what its ``run_python`` runs kept, stored with the turn and deleted with it.

A row's payload is the analysis as JSON text (see :mod:`docint.agents.turn.workbench_tool`),
stamped with its id, the question it answered and where it was made. Its charts are PNG data URIs,
so a row can approach 1 MiB.
"""

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from docint.core.state.base import Base
from docint.utils.text_budget import cut_marker

#: Characters of an analysis's ``stdout`` a stored row keeps: its head and tail around a cut marker.
STORED_STDOUT_CHARS = 16_384


class TurnAnalysis(Base):
    """One analysis of a turn, at its place among the turn's analyses.

    Args:
        Base (DeclarativeBase): The declarative base class for SQLAlchemy models.
    """

    __tablename__ = "turn_analyses"
    #: A uuid4 hex, minted before the row is written.
    id: Mapped[str] = mapped_column(String, primary_key=True)
    turn_id: Mapped[int] = mapped_column(Integer, ForeignKey("turns.id"), index=True, nullable=False)
    #: From 0; the user sees it as Analysis ``position + 1``.
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    payload: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC), nullable=False)


def _stored_stdout(text: str) -> str:
    """Cut a run's output to ``STORED_STDOUT_CHARS``, its head and tail around a marker on a line of its own.

    Args:
        text (str): The output.

    Returns:
        str: The output, or its start and end around the marker, the marker included in the count.
    """
    if len(text) <= STORED_STDOUT_CHARS:
        return text
    keep = STORED_STDOUT_CHARS - len(cut_marker(len(text), len(text))) - 2
    head = (keep + 1) // 2
    tail = keep - head
    return f"{text[:head]}\n{cut_marker(len(text) - keep, len(text))}\n{text[len(text) - tail :]}"


def _provenance(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Copy an analysis's provenance.

    Args:
        payload (Mapping[str, Any]): The analysis.

    Returns:
        dict[str, Any]: Its provenance; empty when it has none.
    """
    provenance = payload.get("provenance")
    return dict(provenance) if isinstance(provenance, Mapping) else {}


def stamp_analysis(
    payload: Mapping[str, Any], *, analysis_id: str, question: str, session_id: str, turn_idx: int
) -> dict[str, Any]:
    """Make an analysis the row it is stored as: its id, the question it answered, and where it was made.

    Keys are only added, so the analysis's validator reads it as the tool kept it. Its ``stdout`` is
    cut to ``STORED_STDOUT_CHARS``, head and tail kept.

    Args:
        payload (Mapping[str, Any]): The analysis as the tool kept it.
        analysis_id (str): Its id.
        question (str): The message its turn answered.
        session_id (str): Its session.
        turn_idx (int): Its turn's index in the session.

    Returns:
        dict[str, Any]: A copy with ``id``, ``question``, ``provenance.session_id`` and ``provenance.turn_idx``.
    """
    stamped: dict[str, Any] = {**payload, "id": analysis_id, "question": question}
    stamped["provenance"] = {**_provenance(payload), "session_id": session_id, "turn_idx": turn_idx}
    stdout = stamped.get("stdout")
    if isinstance(stdout, str):
        stamped["stdout"] = _stored_stdout(stdout)
    return stamped


def served_analysis(payload: Mapping[str, Any], collection: str | None) -> dict[str, Any]:
    """Serve a stored analysis under the caller's own name for its collection.

    Args:
        payload (Mapping[str, Any]): The stored analysis.
        collection (str | None): The caller's logical collection.

    Returns:
        dict[str, Any]: A copy with ``provenance.collection`` set to it.
    """
    return {**payload, "provenance": {**_provenance(payload), "collection": collection}}


def _rows(table: Mapping[str, Any]) -> int:
    """Count a stored table's rows, as many as the run's table had.

    Args:
        table (Mapping[str, Any]): The table.

    Returns:
        int: Its ``row_count``, else the rows it keeps.
    """
    count = table.get("row_count")
    if type(count) is int:
        return count
    rows = table.get("rows")
    return len(rows) if isinstance(rows, list) else 0


def analysis_summary(payload: Mapping[str, Any], index: int) -> dict[str, Any]:
    """Describe a stored analysis for a session's history, without its data.

    Args:
        payload (Mapping[str, Any]): The stored analysis.
        index (int): Its place in the turn, from 1, which is the name the user sees.

    Returns:
        dict[str, Any]: ``id``, ``index``, ``title``, each table's name and shape and each chart's name and size.
    """
    tables = payload.get("tables")
    charts = payload.get("charts")
    return {
        "id": payload.get("id"),
        "index": index,
        "title": payload.get("title"),
        "tables": [
            {
                "name": table.get("name"),
                "columns": len(columns) if isinstance(columns := table.get("columns"), list) else 0,
                "rows": _rows(table),
            }
            for table in (tables if isinstance(tables, list) else [])
            if isinstance(table, Mapping)
        ],
        "charts": [
            {"name": chart.get("name"), "bytes": chart.get("bytes")}
            for chart in (charts if isinstance(charts, list) else [])
            if isinstance(chart, Mapping)
        ],
    }
