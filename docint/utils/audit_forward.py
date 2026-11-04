"""The purpose docint states when its own work calls another audited service.

docint ingests media through Nextext, whose API keeps a data-protection audit
trail of its own: polling a job is a ``query`` there and fetching its
artifacts a ``disclose``, and Nextext refuses both without a stated purpose
when its ``AUDIT_REQUIRE_PURPOSE`` lists them. So every call docint makes to
it states one (``X-Audit-Purpose``): the purpose stated for the docint request
or job whose work makes the call, or — when none was stated — a system purpose
saying why docint asks. docint's own ingest is a ``collect``, which needs no
purpose of its own, so the system purpose is the honest answer then. The
request id goes along too (``X-Request-ID``), so the two trails can be joined.

The work's purpose is carried in a :class:`contextvars.ContextVar`, like the
transfer meter (``docint.utils.audit_meter``): :func:`forwarding` binds it
where the work starts (``audit.metered`` for a run, the upload route for the
work an upload starts), and it follows the work into anyio worker threads and
into pool tasks submitted through ``audit_meter.in_context``.

Stdlib and the vendored audit runtime only, so every client can import it.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from docint.utils.audit_runtime import PURPOSE_HEADER, REQUEST_ID_HEADER, encode_purpose

__all__ = ["SYSTEM_REASON", "Forwarded", "forward_headers", "forwarding"]

#: The reason docint states for work whose own purpose nobody stated.
SYSTEM_REASON = "docint ingest"


@dataclass(frozen=True)
class Forwarded:
    """The purpose and correlation id of the work making outbound calls.

    Attributes:
        case_ref (str | None): The stated case reference.
        reason (str | None): The stated reason.
        request_id (str | None): The gateway's correlation id of the request
            that started the work.
    """

    case_ref: str | None = None
    reason: str | None = None
    request_id: str | None = None


_CURRENT: contextvars.ContextVar[Forwarded | None] = contextvars.ContextVar("docint_audit_forward", default=None)


@contextmanager
def forwarding(
    *, case_ref: str | None = None, reason: str | None = None, request_id: str | None = None
) -> Iterator[Forwarded]:
    """Make this the purpose outbound calls state for the work in this block.

    Args:
        case_ref (str | None): The case reference stated for the work.
        reason (str | None): The reason stated for the work.
        request_id (str | None): The request id the work belongs to.

    Yields:
        Forwarded: The bound purpose.
    """
    bound = Forwarded(case_ref=case_ref or None, reason=reason or None, request_id=request_id or None)
    token = _CURRENT.set(bound)
    try:
        yield bound
    finally:
        _CURRENT.reset(token)


def forward_headers() -> dict[str, str]:
    """Return the headers a call to another audited service carries.

    Returns:
        dict[str, str]: ``X-Audit-Purpose`` — the stated purpose of the work
            bound with :func:`forwarding`, else ``reason=`` :data:`SYSTEM_REASON`
            — and ``X-Request-ID`` when the work has one.
    """
    current = _CURRENT.get() or Forwarded()
    if current.case_ref or current.reason:
        purpose = encode_purpose(current.case_ref, current.reason)
    else:
        purpose = encode_purpose(None, SYSTEM_REASON)
    headers = {PURPOSE_HEADER: purpose}
    if current.request_id:
        headers[REQUEST_ID_HEADER] = current.request_id
    return headers
