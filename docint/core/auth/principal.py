"""Request-principal resolution dependency.

This module is the single seam the deferred auth track replaces: swap
the header read for a verified-token read and nothing downstream (data
model, ownership queries, endpoint wiring) changes. Group membership and
the admin flag are resolved here too, from the same trusted-header
contract.

Whether those headers are believed at all is decided here as well. With
``DOCINT_EDGE_SECRET`` set, only a request carrying that secret in
``X-Edge-Secret`` (the edge gateway adds it to every request it proxies)
may assert an identity. Any other caller is treated as if it had sent no
identity headers, so another container on a shared network cannot act as
a user, let alone an admin. The dev fallbacks (``DOCINT_DEFAULT_IDENTITY``,
``DOCINT_DEFAULT_GROUPS``) apply only while no edge secret is set; with one
set they are ignored, and such a caller is unauthenticated.
"""

import hmac
from dataclasses import dataclass, field
from urllib.parse import quote

from fastapi import HTTPException, Request
from loguru import logger

from docint.utils.env_cfg import PrincipalConfig, load_principal_env

#: Header carrying the edge gateway's per-app shared secret.
EDGE_SECRET_HEADER = "X-Edge-Secret"
#: The gateway's decorative display name. Never part of the identity.
DISPLAY_NAME_HEADER = "X-Auth-Name"


@dataclass(frozen=True)
class Principal:
    """The resolved calling identity: principal name, groups, and admin scope.

    ``via`` says how the name was established, in the audit trail's terms:
    ``edge`` when a trusted identity header carried it, ``system`` when it is
    the configured default identity (the dev fallback).
    """

    name: str
    groups: frozenset[str] = field(default_factory=frozenset)
    is_admin: bool = False
    requested_owner: str | None = None
    display_name: str | None = None
    via: str = "edge"

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


def check_principal_config() -> PrincipalConfig:
    """Load the principal configuration at startup and report how far it trusts callers.

    Loading it here makes an invalid ``DOCINT_EDGE_SECRET`` stop startup
    instead of failing every request. One warning is logged when the
    backend would believe identity headers from any caller that can reach
    it: no edge secret and no default identity, which is the production
    shape before the gateway's secret is rolled out. Another is logged when
    an edge secret is set beside a default identity or default groups, which
    are then ignored.

    Returns:
        PrincipalConfig: The loaded configuration.

    Raises:
        ValueError: If ``DOCINT_EDGE_SECRET`` is set but too short.
    """
    cfg = load_principal_env()
    if cfg.edge_secret is None and cfg.default_identity is None:
        logger.warning(
            "Identity headers are trusted from any caller that can reach the backend. "
            "Set DOCINT_EDGE_SECRET (see edge-plane docs/identity-contract.md)."
        )
    if cfg.edge_secret is not None and (cfg.default_identity is not None or cfg.default_groups is not None):
        logger.warning(
            "DOCINT_DEFAULT_IDENTITY and DOCINT_DEFAULT_GROUPS are ignored while DOCINT_EDGE_SECRET is set: "
            "a request without the gateway's identity is unauthenticated."
        )
    return cfg


def _carries_edge_secret(request: Request, secret: str) -> bool:
    """Whether the request's ``X-Edge-Secret`` header equals the configured secret.

    Compared as bytes in constant time. Starlette decodes header values as
    latin-1, so encoding back with latin-1 restores the bytes on the wire.

    Args:
        request (Request): The incoming request.
        secret (str): The configured ``DOCINT_EDGE_SECRET``.

    Returns:
        bool: ``True`` only when the header is present and matches.
    """
    received = request.headers.get(EDGE_SECRET_HEADER)
    if received is None:
        return False
    return hmac.compare_digest(received.encode("latin-1"), secret.encode("utf-8"))


def _warn_ignored_identity_headers(request: Request, cfg: PrincipalConfig) -> None:
    """Log that a request's identity headers were ignored, if it sent any.

    Names the method, the path without its query string, and whether the
    secret was missing or wrong. Never a header value: neither the secret
    nor the identity the caller claimed.

    Args:
        request (Request): The request whose identity headers were ignored.
        cfg (PrincipalConfig): The principal configuration in force.
    """
    identity_headers = (cfg.header_name, cfg.groups_header, DISPLAY_NAME_HEADER)
    if not any(name in request.headers for name in identity_headers):
        return
    logger.warning(
        "Ignored identity headers on {} '{}': {} {}",
        request.method,
        quote(request.scope.get("path", "")),
        EDGE_SECRET_HEADER,
        "missing" if EDGE_SECRET_HEADER not in request.headers else "does not match",
    )


def resolve_principal(request: Request) -> Principal:
    """Resolve the calling principal from the configured trusted headers.

    With ``DOCINT_EDGE_SECRET`` set, a request is trusted only when its
    ``X-Edge-Secret`` matches; unset, every request is. An untrusted
    request is resolved as if it carried no identity headers (user,
    groups, display name), and a warning names its method and path.

    Resolution order (spec Section 8):

    1. If the configured trusted header is present, use its value as the
       principal name. Otherwise, if a default identity is configured and
       no edge secret is set, use it (the dev/pre-auth fallback, also the
       migration backfill owner). Otherwise fail closed with HTTP 401.
    2. Groups come from the configured groups header, parsed as a
       comma-separated list; if that header is absent, the configured
       dev default groups apply, again only while no edge secret is set.
       Missing or blank values yield no groups.
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
        HTTPException: With status 401 when neither a trusted header nor
            a configured default identity is available.
    """
    return _resolve(request, warn=True)


def principal_name(request: Request) -> str | None:
    """Return the name ``resolve_principal`` would resolve, without refusing or logging.

    For code that only needs to know who a request is for, ahead of the
    route that authenticates it: the route still resolves, refuses and
    warns itself, so this neither raises the 401 nor logs the ignored
    identity headers a second time.

    Args:
        request (Request): The incoming request.

    Returns:
        str | None: The principal's name, ``None`` where ``resolve_principal``
            answers 401.
    """
    try:
        return _resolve(request, warn=False).name
    except HTTPException:
        return None


def _resolve(request: Request, *, warn: bool) -> Principal:
    """Resolve the calling principal (see :func:`resolve_principal`).

    Args:
        request (Request): The incoming request.
        warn (bool): Whether to log identity headers the request was not
            trusted to send.

    Returns:
        Principal: The resolved calling identity.

    Raises:
        HTTPException: With status 401 when no identity is available.
    """
    cfg = load_principal_env()
    trusted = cfg.edge_secret is None or _carries_edge_secret(request, cfg.edge_secret)
    if not trusted and warn:
        _warn_ignored_identity_headers(request, cfg)
    # The dev fallbacks hold only while no edge secret is enforced: with one, a
    # request either proves it came through the gateway or is anonymous.
    fallback_identity = cfg.default_identity if cfg.edge_secret is None else None
    fallback_groups = cfg.default_groups if cfg.edge_secret is None else None
    header_value = request.headers.get(cfg.header_name) if trusted else None
    if header_value:
        name = header_value
    elif fallback_identity:
        name = fallback_identity
    else:
        raise HTTPException(status_code=401, detail="Missing authenticated principal.")
    raw_groups = request.headers.get(cfg.groups_header) if trusted else None
    if raw_groups is None:
        raw_groups = fallback_groups or ""
    groups = frozenset(g.strip() for g in raw_groups.split(",") if g.strip())
    requested_owner = (request.query_params.get("owner") or "").strip() or None
    return Principal(
        name=name,
        groups=groups,
        is_admin=cfg.admin_group in groups,
        requested_owner=requested_owner,
        display_name=request.headers.get(DISPLAY_NAME_HEADER) if trusted else None,
        via="edge" if header_value else "system",
    )
