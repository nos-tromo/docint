"""Tests for the purpose docint states when its own work calls another audited service."""

from __future__ import annotations

from docint.utils.audit_forward import CHAT_REASON, forward_headers, forwarding
from docint.utils.audit_runtime import PURPOSE_HEADER, REQUEST_ID_HEADER, parse_purpose_header


def test_an_unstated_purpose_falls_back_to_the_reason_given() -> None:
    """With nothing stated, the caller's system reason goes out, else docint ingest as before."""
    assert parse_purpose_header(forward_headers(system_reason=CHAT_REASON)[PURPOSE_HEADER]) == (None, "docint chat")
    assert parse_purpose_header(forward_headers()[PURPOSE_HEADER]) == (None, "docint ingest")
    assert REQUEST_ID_HEADER not in forward_headers(system_reason=CHAT_REASON)


def test_a_stated_purpose_wins_over_any_fallback() -> None:
    """A purpose bound for the work goes out as stated, whatever fallback the caller names."""
    with forwarding(case_ref="case-0001", request_id="req-0001"):
        headers = forward_headers(system_reason=CHAT_REASON)
    assert parse_purpose_header(headers[PURPOSE_HEADER]) == ("case-0001", None)
    assert headers[REQUEST_ID_HEADER] == "req-0001"
