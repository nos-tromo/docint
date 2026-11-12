"""Bind each request's prefix-cache scope (``docint.utils.cache_scope``) for its whole lifetime.

A pure ASGI middleware rather than a dependency: a sync FastAPI dependency
runs in a copied context that never reaches the endpoint, while a variable set
here, around the whole app call, is seen by the route, by its worker threads,
by a streaming response's body and by the jobs a request starts. It resolves
the principal the way the route will (``principal_name``), but never refuses a
request or logs for it: authenticating stays with the route.
"""

from __future__ import annotations

from starlette.requests import Request
from starlette.types import ASGIApp, Receive, Scope, Send

from docint.core.auth.principal import principal_name
from docint.utils import cache_scope

__all__ = ["CacheScopeMiddleware"]


class CacheScopeMiddleware:
    """Binds the request principal's cache scope around every HTTP request."""

    def __init__(self, app: ASGIApp) -> None:
        """Wrap the app.

        Args:
            app (ASGIApp): The application to wrap.
        """
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Run the request with its principal's scope bound; other scopes pass through.

        Args:
            scope (Scope): The ASGI connection scope.
            receive (Receive): The ASGI receive channel.
            send (Send): The ASGI send channel.
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        try:
            name = principal_name(Request(scope))
        except Exception:  # authenticating is the route's job; a fault here only means no scope
            name = None
        with cache_scope.bind(name):
            await self.app(scope, receive, send)
