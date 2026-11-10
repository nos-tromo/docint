"""The pseudonym that keeps one user's cached prompt prefixes apart from another's.

The shared chat backend (vllm-service) reuses the KV cache of prompt prefixes
it has seen, and a cache hit shortens the time to the first token, so without
isolation one user could time requests to learn whether another user's prompt
(a document's text, a retrieved chunk) is cached. vllm-service's router keeps
callers apart by the ``X-Cache-Scope`` header: calls with the same scope share
cached prefixes, calls with different scopes never do.

The scope is a pseudonym, not the identity: an HMAC of the user's name under a
key drawn when this process starts and never stored or sent, so nothing outside
the process can map it back to a user. A restart draws a new key, which only
costs each user one cold cache. It goes only to vllm-service
(``INFERENCE_PROVIDER=vllm``) and only on chat and completion calls.

It rides a :class:`contextvars.ContextVar`, like the purpose forwarding
(``docint.utils.audit_forward``): the request middleware
(``docint.core.auth.cache_scope``) binds it for each request's principal and
the job runner for a job's actor, and it follows the work into anyio worker
threads and into pool tasks submitted through ``audit_meter.in_context``. Work
with no user behind it binds nothing.
"""

from __future__ import annotations

import contextvars
import hashlib
import hmac
import secrets
from collections.abc import Iterator
from contextlib import contextmanager

from docint.utils.env_cfg import load_openai_env

__all__ = ["SCOPE_HEADER", "bind", "headers"]

#: The header vllm-service's router partitions the prefix cache by.
SCOPE_HEADER = "X-Cache-Scope"

_KEY = secrets.token_bytes(32)
_LABEL = b"cache-scope/v1\0"
_CURRENT: contextvars.ContextVar[str | None] = contextvars.ContextVar("docint_cache_scope", default=None)


@contextmanager
def bind(name: str | None) -> Iterator[None]:
    """Make model calls in this block carry the scope of ``name``.

    Args:
        name (str | None): The user the work runs for; ``None`` or empty for
            work with no user behind it, which then carries no scope.

    Yields:
        None: Control, with the scope in effect.
    """
    scope = hmac.new(_KEY, _LABEL + name.encode("utf-8"), hashlib.sha256).hexdigest() if name else None
    token = _CURRENT.set(scope)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def headers() -> dict[str, str]:
    """Return the headers a chat or completion call carries for cache isolation.

    Returns:
        dict[str, str]: ``{"X-Cache-Scope": <64 hex>}`` while a user is bound
            and the provider is vLLM, otherwise empty.
    """
    scope = _CURRENT.get()
    if scope is None or load_openai_env().inference_provider != "vllm":
        return {}
    return {SCOPE_HEADER: scope}
