"""Tests for per-turn chat logging and the privacy rules it must obey.

A chat turn produced no success-path log line at any level: every statement
on the retrieval path was an error or a degradation, so a healthy backend
answering questions was indistinguishable from an idle one.

The strings here are invented; none of them corresponds to real data.
"""

from __future__ import annotations

from typing import Any

import pytest
from _pytest.logging import LogCaptureFixture

from docint.core.errors import redact_validation_errors
from docint.core.state.session_manager import _log_turn_summary

#: Stand-in for a user's question. It must never reach the log.
SECRET_QUERY = "zzsecretquestionzz what did the witness say"

#: Stand-in for the model's answer. Likewise.
SECRET_ANSWER = "zzsecretanswerzz the witness said nothing"


def _response(**overrides: Any) -> dict[str, Any]:
    """Build a normalized turn payload.

    Args:
        **overrides: Fields to override on the default payload.

    Returns:
        dict[str, Any]: A payload shaped like ``_normalize_response_data``'s.
    """
    payload: dict[str, Any] = {
        "response": SECRET_ANSWER,
        "query": SECRET_QUERY,
        "retrieval_query": SECRET_QUERY,
        "retrieval_mode": "rewrite",
        "rerank": {"applied": True, "error": None},
        "graph_debug": {"applied": True, "reason": None},
        "sources": [{"chunk_id": "c1"}, {"chunk_id": "c2"}],
    }
    payload.update(overrides)
    return payload


def test_turn_line_reports_the_shape_of_a_turn(loguru_caplog_info: LogCaptureFixture) -> None:
    """One line per answered turn, naming what retrieval actually did.

    Args:
        loguru_caplog_info (LogCaptureFixture): Bridged INFO capture.
    """
    _log_turn_summary(
        collection="u000000000000__field-notes",
        session_id="s-1",
        turn_idx=4,
        response=_response(),
        elapsed_s=4.23,
    )

    line = next(m for m in loguru_caplog_info.messages if "Turn complete" in m)
    assert "collection='u000000000000__field-notes'" in line
    assert "session=s-1 turn=4" in line
    assert "mode=rewrite" in line
    assert "sources=2 images=0" in line
    assert "rerank=applied" in line
    assert "graphrag=applied" in line
    assert "duration=4.2s" in line


def test_turn_line_never_carries_the_query_or_the_answer(loguru_caplog_info: LogCaptureFixture) -> None:
    """The load-bearing privacy rule: shapes and counts, never content.

    Args:
        loguru_caplog_info (LogCaptureFixture): Bridged INFO capture.
    """
    _log_turn_summary(
        collection="u000000000000__field-notes",
        session_id="s-1",
        turn_idx=1,
        response=_response(),
        elapsed_s=1.0,
    )

    combined = "\n".join(loguru_caplog_info.messages)
    assert "zzsecretquestionzz" not in combined
    assert "zzsecretanswerzz" not in combined


def test_rerank_none_is_not_the_same_as_not_applied(loguru_caplog_info: LogCaptureFixture) -> None:
    """The stamp is three-state; collapsing it would misreport a scoped turn.

    ``None`` means no reranker was in the loop at all — a scoped turn drops
    every ranking postprocessor. Reading that as "failed" would send an
    operator hunting a rerank outage that never happened.

    Args:
        loguru_caplog_info (LogCaptureFixture): Bridged INFO capture.
    """
    _log_turn_summary(
        collection="c",
        session_id="s",
        turn_idx=0,
        response=_response(rerank=None),
        elapsed_s=1.0,
    )
    _log_turn_summary(
        collection="c",
        session_id="s",
        turn_idx=1,
        response=_response(rerank={"applied": False, "error": "endpoint down"}),
        elapsed_s=1.0,
    )

    lines = [m for m in loguru_caplog_info.messages if "Turn complete" in m]
    assert "rerank=none" in lines[0]
    assert "rerank=failed" in lines[1]


def test_a_scoped_turn_reports_its_scope_and_why_graphrag_was_skipped(
    loguru_caplog_info: LogCaptureFixture,
) -> None:
    """A scoped turn is a different kind of answer and must read as one.

    Args:
        loguru_caplog_info (LogCaptureFixture): Bridged INFO capture.
    """
    _log_turn_summary(
        collection="c",
        session_id="s",
        turn_idx=2,
        response=_response(
            retrieval_mode="scoped",
            scoped_chunk_count=6,
            graph_debug={"applied": False, "reason": "scoped"},
        ),
        elapsed_s=2.0,
    )

    line = next(m for m in loguru_caplog_info.messages if "Turn complete" in m)
    assert "mode=scoped" in line
    assert "scoped_chunks=6" in line
    assert "graphrag=skipped:scoped" in line


def test_image_sources_are_counted_separately(loguru_caplog_info: LogCaptureFixture) -> None:
    """Images join the text hits before ranking, so the split is worth seeing.

    Args:
        loguru_caplog_info (LogCaptureFixture): Bridged INFO capture.
    """
    _log_turn_summary(
        collection="c",
        session_id="s",
        turn_idx=3,
        response=_response(
            sources=[{"chunk_id": "c1"}, {"chunk_id": "i1", "image_id": "i1"}, {"chunk_id": "i2", "image_id": "i2"}]
        ),
        elapsed_s=1.0,
    )

    assert "sources=3 images=2" in next(m for m in loguru_caplog_info.messages if "Turn complete" in m)


def test_a_malformed_payload_never_breaks_an_answered_turn() -> None:
    """The turn is already answered and persisted; describing it cannot undo that."""
    _log_turn_summary(
        collection="c",
        session_id="s",
        turn_idx=0,
        response={"sources": "not-a-list", "rerank": 7, "graph_debug": "nope"},
        elapsed_s=1.0,
    )


# ---------------------------------------------------------------------------
# the one pre-existing leak vector
# ---------------------------------------------------------------------------


def test_validation_errors_are_logged_without_the_submitted_value() -> None:
    """A malformed /query body put the question itself in the log.

    Pydantic attaches the offending ``input`` to every error, so the
    validation handler was the one path by which query text could reach a
    log line. The location and the error type are what an operator needs.
    """
    errors = [
        {
            "type": "string_too_short",
            "loc": ("body", "query"),
            "msg": "String should have at least 1 character",
            "input": SECRET_QUERY,
            "ctx": {"min_length": 1, "value": SECRET_QUERY},
        }
    ]

    redacted = redact_validation_errors(errors)

    rendered = repr(redacted)
    assert "zzsecretquestionzz" not in rendered
    assert redacted[0]["input"] == "<redacted>"
    # The diagnostic content survives.
    assert redacted[0]["type"] == "string_too_short"
    assert redacted[0]["loc"] == ("body", "query")


@pytest.mark.parametrize("error", [None, "a string", 42, {"type": "x"}])
def test_redaction_survives_an_unexpected_error_shape(error: Any) -> None:
    """A shape we did not anticipate must not crash the error handler.

    Args:
        error (Any): An entry pydantic might hand over.
    """
    assert len(redact_validation_errors([error])) == 1
