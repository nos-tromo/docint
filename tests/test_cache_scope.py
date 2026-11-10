"""Tests for the per-user prefix-cache scope (``X-Cache-Scope``) on chat calls.

The scope is a per-process keyed pseudonym of the user a model call runs for.
vllm-service's router turns it into vLLM's ``cache_salt``, so one user's
cached prompt prefixes are never shared with another's. All names are
synthetic.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from _pytest.logging import LogCaptureFixture
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from starlette.requests import Request as StarletteRequest

from docint.core.auth.cache_scope import CacheScopeMiddleware
from docint.core.auth.principal import EDGE_SECRET_HEADER, principal_name, resolve_principal
from docint.utils import cache_scope
from docint.utils.openai_cfg import LocalOpenAI

_EDGE_SECRET = "synthetic-edge-secret-0123456789abcdef"


@pytest.fixture
def vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the test against vllm-service, the one provider that gets the header.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.setenv("INFERENCE_PROVIDER", "vllm")


def _scope_of(name: str | None) -> str | None:
    """Return the header value bound for a name.

    Args:
        name (str | None): The user name.

    Returns:
        str | None: The ``X-Cache-Scope`` value, ``None`` when none is sent.
    """
    with cache_scope.bind(name):
        return cache_scope.headers().get(cache_scope.SCOPE_HEADER)


def _request(headers: dict[str, str]) -> StarletteRequest:
    """Build a minimal request carrying the given headers.

    Args:
        headers (dict[str, str]): Header name/value pairs.

    Returns:
        StarletteRequest: The request.
    """
    raw = [(key.lower().encode("latin-1"), value.encode("latin-1")) for key, value in headers.items()]
    return StarletteRequest({"type": "http", "method": "GET", "path": "/", "query_string": b"", "headers": raw})


@pytest.mark.usefixtures("vllm")
def test_nothing_is_sent_without_a_bound_user() -> None:
    """Work bound to no user carries no scope."""
    assert cache_scope.headers() == {}
    assert _scope_of(None) is None
    assert _scope_of("") is None


@pytest.mark.usefixtures("vllm")
def test_scope_is_a_stable_pseudonym_per_user() -> None:
    """One user always gets the same 64 lowercase hex characters, another user others."""
    alice = _scope_of("alice")
    assert alice is not None
    assert len(alice) == 64
    assert set(alice) <= set("0123456789abcdef")
    assert _scope_of("alice") == alice
    assert _scope_of("bob") != alice


@pytest.mark.usefixtures("vllm")
def test_scope_does_not_reveal_the_name() -> None:
    """The value is keyed: neither the name nor its plain hash goes upstream."""
    import hashlib

    alice = _scope_of("alice")
    assert alice is not None
    assert "alice" not in alice
    assert alice != hashlib.sha256(b"alice").hexdigest()
    assert alice != hashlib.sha256(b"cache-scope/v1\0alice").hexdigest()


@pytest.mark.usefixtures("vllm")
def test_binding_no_user_clears_an_outer_scope_and_blocks_restore_it() -> None:
    """A nested block for no user sends nothing; leaving a block restores the outer scope."""
    with cache_scope.bind("alice"):
        outer = cache_scope.headers()
        with cache_scope.bind(None):
            assert cache_scope.headers() == {}
        assert cache_scope.headers() == outer
    assert cache_scope.headers() == {}


@pytest.mark.parametrize("provider", ["ollama", "openai"])
def test_nothing_is_sent_off_vllm(monkeypatch: pytest.MonkeyPatch, provider: str) -> None:
    """Only vllm-service partitions its cache by the header; nobody else gets it.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        provider (str): A provider other than vLLM.
    """
    monkeypatch.setenv("INFERENCE_PROVIDER", provider)
    assert _scope_of("alice") is None


@pytest.mark.parametrize(
    ("env", "headers"),
    [
        ({}, {"X-Auth-User": "alice"}),
        ({"DOCINT_DEFAULT_IDENTITY": "operator"}, {}),
        ({"DOCINT_DEFAULT_IDENTITY": "operator"}, {"X-Auth-User": "alice"}),
        ({"DOCINT_EDGE_SECRET": _EDGE_SECRET}, {"X-Auth-User": "alice", EDGE_SECRET_HEADER: _EDGE_SECRET}),
    ],
    ids=["header", "fallback", "header-over-fallback", "edge-secret"],
)
def test_principal_name_is_the_resolved_name(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], headers: dict[str, str]
) -> None:
    """``principal_name`` names exactly the principal ``resolve_principal`` resolves.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        env (dict[str, str]): Principal settings.
        headers (dict[str, str]): The request headers.
    """
    monkeypatch.delenv("DOCINT_DEFAULT_IDENTITY", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert principal_name(_request(headers)) == resolve_principal(_request(headers)).name


@pytest.mark.parametrize(
    "headers",
    [{}, {"X-Auth-User": "claimed-user"}, {"X-Auth-User": "claimed-user", EDGE_SECRET_HEADER: "wrong-secret"}],
    ids=["none", "no-secret", "wrong-secret"],
)
def test_principal_name_is_none_where_resolution_refuses(
    monkeypatch: pytest.MonkeyPatch, loguru_caplog: LogCaptureFixture, headers: dict[str, str]
) -> None:
    """No name where ``resolve_principal`` answers 401, and no warning logged for it.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        loguru_caplog (LogCaptureFixture): Loguru records bridged into caplog.
        headers (dict[str, str]): The request headers.
    """
    monkeypatch.setenv("DOCINT_EDGE_SECRET", _EDGE_SECRET)
    monkeypatch.setenv("DOCINT_DEFAULT_IDENTITY", "operator")
    with pytest.raises(HTTPException):
        resolve_principal(_request(headers))
    loguru_caplog.clear()

    assert principal_name(_request(headers)) is None
    assert loguru_caplog.records == []


def _app() -> FastAPI:
    """Build an app behind the middleware whose routes report the scope they see.

    Returns:
        FastAPI: The app.
    """
    app = FastAPI()
    app.add_middleware(CacheScopeMiddleware)

    @app.get("/plain")
    def plain() -> dict[str, str]:
        """Report the scope a sync route sees."""
        return cache_scope.headers()

    @app.get("/async")
    async def plain_async() -> dict[str, str]:
        """Report the scope an async route sees."""
        return cache_scope.headers()

    @app.get("/stream")
    def stream() -> StreamingResponse:
        """Report the scope seen while a response streams."""

        def body() -> Iterator[str]:
            yield json.dumps(cache_scope.headers())

        return StreamingResponse(body(), media_type="application/json")

    @app.get("/protected")
    def protected(request: Request) -> dict[str, str]:
        """Resolve the principal as a protected route does."""
        return {"name": resolve_principal(request).name}

    return app


@pytest.mark.usefixtures("vllm")
@pytest.mark.parametrize("path", ["/plain", "/async", "/stream"])
def test_middleware_binds_the_requests_user(monkeypatch: pytest.MonkeyPatch, path: str) -> None:
    """Every route, streaming included, sees the scope of the user it serves.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        path (str): The route.
    """
    monkeypatch.delenv("DOCINT_DEFAULT_IDENTITY", raising=False)
    with TestClient(_app()) as client:
        alice = client.get(path, headers={"X-Auth-User": "alice"}).json()
        bob = client.get(path, headers={"X-Auth-User": "bob"}).json()
    assert alice == {cache_scope.SCOPE_HEADER: _scope_of("alice")}
    assert bob == {cache_scope.SCOPE_HEADER: _scope_of("bob")}
    assert alice != bob


@pytest.mark.usefixtures("vllm")
def test_middleware_binds_nothing_for_an_unauthenticated_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """A request with no principal gets no scope, and the middleware refuses nothing itself.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
    """
    monkeypatch.delenv("DOCINT_DEFAULT_IDENTITY", raising=False)
    with TestClient(_app()) as client:
        assert client.get("/plain").json() == {}
        assert client.get("/protected").status_code == 401


def test_middleware_adds_no_second_ignored_headers_warning(
    monkeypatch: pytest.MonkeyPatch, loguru_caplog: LogCaptureFixture
) -> None:
    """Ignored identity headers are reported once per request: by the route, not the middleware.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch fixture.
        loguru_caplog (LogCaptureFixture): Loguru records bridged into caplog.
    """
    monkeypatch.setenv("DOCINT_EDGE_SECRET", _EDGE_SECRET)
    with TestClient(_app()) as client:
        assert client.get("/protected", headers={"X-Auth-User": "claimed-user"}).status_code == 401
    messages = [r.getMessage() for r in loguru_caplog.records if "Ignored identity headers" in r.getMessage()]
    assert messages == ["Ignored identity headers on GET '/protected': X-Edge-Secret missing"]


def test_the_api_installs_the_middleware() -> None:
    """The API app binds the scope for every request."""
    from docint.core import api

    assert any(m.cls is CacheScopeMiddleware for m in api.app.user_middleware)


def _local_openai() -> LocalOpenAI:
    """Build the LlamaIndex chat client the RAG engine uses, pointed nowhere.

    Returns:
        LocalOpenAI: The client.
    """
    return LocalOpenAI(
        model="synthetic-chat-model",
        api_key="sk-synthetic",
        api_base="http://127.0.0.1:9/v1",
        additional_kwargs={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    )


@pytest.mark.usefixtures("vllm")
def test_llamaindex_calls_carry_the_scope() -> None:
    """Every LlamaIndex chat or completion request carries the bound scope."""
    llm = _local_openai()
    with cache_scope.bind("alice"):
        kwargs = llm._get_model_kwargs(stream=True)
    assert kwargs["extra_headers"] == {cache_scope.SCOPE_HEADER: _scope_of("alice")}
    assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert llm.additional_kwargs == {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}


@pytest.mark.usefixtures("vllm")
def test_llamaindex_calls_keep_the_callers_headers() -> None:
    """A caller's own extra headers stay beside the scope."""
    with cache_scope.bind("alice"):
        kwargs = _local_openai()._get_model_kwargs(extra_headers={"X-Synthetic": "1"})
    assert kwargs["extra_headers"] == {"X-Synthetic": "1", cache_scope.SCOPE_HEADER: _scope_of("alice")}


@pytest.mark.usefixtures("vllm")
def test_llamaindex_calls_without_a_user_are_unchanged() -> None:
    """Without a bound user the request is exactly what LlamaIndex builds."""
    assert "extra_headers" not in _local_openai()._get_model_kwargs()
