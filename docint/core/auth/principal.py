"""Request-principal resolution dependency.

This module is the single seam the deferred auth track replaces: swap
the header read for a verified-token read and nothing downstream (data
model, ownership queries, endpoint wiring) changes.
"""

from fastapi import HTTPException, Request

from docint.utils.env_cfg import load_principal_env


def resolve_principal(request: Request) -> str:
    """Resolve the calling principal from the configured trusted header.

    Resolution order (spec Section 8):

    1. If the configured trusted header is present, return its value.
    2. Otherwise, if a default identity is configured, return it (the
       dev/pre-auth fallback, also the migration backfill owner).
    3. Otherwise fail closed with HTTP 401.

    Args:
        request (Request): The incoming FastAPI/Starlette request.

    Returns:
        str: The resolved principal identifier.

    Raises:
        HTTPException: With status 401 when neither the trusted header
            nor a configured default identity is available.
    """
    cfg = load_principal_env()
    header_value = request.headers.get(cfg.header_name)
    if header_value:
        return header_value
    if cfg.default_identity:
        return cfg.default_identity
    raise HTTPException(status_code=401, detail="Missing authenticated principal.")
