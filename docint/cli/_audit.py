"""The command-line tools' side of the data-protection audit trail.

Every command that processes collection content writes its own audit chain
(``docint.core.audit.cli_audit``): the operator from ``AUDIT_CLI_USER``, the
purpose from ``AUDIT_CASE_REF``/``AUDIT_REASON``, and a refusal (exit code 2)
when ``AUDIT_REQUIRE_PURPOSE`` asks for a purpose the run does not state. What
each command records is listed in ``docs/audit.md``.

A command's work functions take the command's open chain, or open one when
they are called as the whole command, so a bulk run (``search-index-all``)
records every collection on one chain while a test or a script calling the
function directly is still recorded.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

from docint.core.audit import CliAudit, cli_audit


@contextmanager
def session(
    audit: CliAudit | None,
    command: str,
    action: str,
    operation: str,
    *,
    also: Sequence[str] = (),
) -> Iterator[CliAudit]:
    """Use the command's open audit chain, or open one for this call.

    Args:
        audit (CliAudit | None): The command's chain, when it opened one.
        command (str): The command, for a chain opened here.
        action (str): The work's primary action, for the purpose rule.
        operation (str): Its operation name, for a refusal's event.
        also (Sequence[str]): Further actions it may record.

    Yields:
        CliAudit: The chain to record with.
    """
    if audit is not None:
        yield audit
        return
    with cli_audit(command, action, operation, also=also) as opened:
        yield opened


def collection_criteria(typed: str, physical: str | None) -> dict[str, object]:
    """Name the physical collection a typed name resolved to, when it differs.

    The event's ``container`` is the name the operator typed; the physical,
    owner-namespaced name is what ties the run to the collection's owner.

    Args:
        typed (str): The name as entered.
        physical (str | None): The resolved physical name, when known.

    Returns:
        dict[str, object]: ``{"physical": ...}``, or nothing.
    """
    return {"physical": physical} if physical and physical != typed else {}
