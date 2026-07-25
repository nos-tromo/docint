"""Request-principal resolution dependency.

This module is the single seam the deferred auth track replaces: swap
the header read for a verified-token read and nothing downstream (data
model, ownership queries, endpoint wiring) changes. Group membership and
the admin flag are resolved here too, from the same trusted-header
contract.
"""

from dataclasses import dataclass, field

from fastapi import HTTPException, Request

from docint.utils.env_cfg import load_principal_env


@dataclass(frozen=True)
class Principal:
    """The resolved calling identity: principal name, groups, and admin scope."""

    name: str
    groups: frozenset[str] = field(default_factory=frozenset)
    is_admin: bool = False
    requested_owner: str | None = None

    @property
    def effective_owner(self) -> str:
        """The owner namespace this request operates in.

        Admins may act on another owner's collections by supplying an
        ``owner`` query param; everyone else always acts as themselves,
        so a non-admin passing ``owner`` resolves against their own
        namespace and cross-owner names 404 exactly as before.
        """
        if self.is_admin and self.requested_owner:
            return self.requested_owner
        return self.name


def resolve_principal(request: Request) -> Principal:
    """Resolve the calling principal from the configured trusted headers.

    Resolution order (spec Section 8):

    1. If the configured trusted header is present, use its value as the
       principal name. Otherwise, if a default identity is configured, use
       it (the dev/pre-auth fallback, also the migration backfill owner).
       Otherwise fail closed with HTTP 401.
    2. Groups come from the configured groups header, parsed as a
       comma-separated list; if that header is absent, the configured
       dev default groups apply. Missing or blank values yield no groups.
    3. The principal is an admin iff the configured admin group is among
       its groups.
    4. An optional ``owner`` query param is captured as ``requested_owner``
       (blank normalises to ``None``); only admins may act as another
       owner (see ``Principal.effective_owner``).

    Args:
        request (Request): The incoming FastAPI/Starlette request.

    Returns:
        Principal: The resolved calling identity.

    Raises:
        HTTPException: With status 401 when neither the trusted header
            nor a configured default identity is available.
    """
    cfg = load_principal_env()
    header_value = request.headers.get(cfg.header_name)
    if header_value:
        name = header_value
    elif cfg.default_identity:
        name = cfg.default_identity
    else:
        raise HTTPException(status_code=401, detail="Missing authenticated principal.")
    raw_groups = request.headers.get(cfg.groups_header)
    if raw_groups is None:
        raw_groups = cfg.default_groups or ""
    groups = frozenset(g.strip() for g in raw_groups.split(",") if g.strip())
    requested_owner = (request.query_params.get("owner") or "").strip() or None
    return Principal(
        name=name,
        groups=groups,
        is_admin=cfg.admin_group in groups,
        requested_owner=requested_owner,
    )
