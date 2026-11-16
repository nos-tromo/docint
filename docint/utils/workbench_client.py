"""Thin JSON-RPC client for workbench's MCP door, behind the chat agent's ``run_python`` tool.

docint hands a turn's evidence to workbench, which runs the model's Python on
it and returns tables (CSV) and charts (PNG). The client speaks MCP revision
``2026-07-28`` only, one JSON-RPC message per ``POST {WORKBENCH_API_BASE}/mcp``,
and holds no MCP session. Each exchange (a discovery or a call) is an
``anyio.run`` over a fresh ``httpx.AsyncClient`` that takes no proxy from the
environment, so the inline data cannot leave through one. That event loop runs
on a daemon thread of its own, and the caller waits for it at most the
exchange's deadline plus half a second. ``anyio.run`` closes its loop by
joining the thread where asyncio looks host names up, with no limit, so a slow
lookup would otherwise hold the caller past any deadline; an abandoned lookup
finishes on its own. The caller's thread never runs an event loop of the
client's, so the client may be used from any context, a running event loop
included.

Discovery (``server/discover``, then ``tools/list``, within
:data:`DISCOVERY_TIMEOUT_S` together) runs before a turn builds its tools,
single-flight. It succeeds only when workbench serves this revision and lists
``run_python`` with ``inputSchema.properties.time_limit_s.maximum``, the cap.
Each result is cached for its own ``ttlMs``. A failure offers no tool and is
tried again at most once per :data:`DISCOVERY_RETRY_S`. A call that gets no
answer, whose arguments are refused (a cap lowered since), or that is answered
``-32022`` (a revision no longer served) or ``-32602`` (an unknown tool among
others: the code decides, never the text) drops the cache. Each change between
discovered and not logs one line.

Identity reaches workbench only on ``tools/call`` and only as keyed
pseudonyms: HMAC-SHA256 under the shared ``WORKBENCH_API_KEY`` over a label, a
NUL byte and the owner's name or the session id, as 64 hex digits. The stated
purpose and request id go with it (``audit_forward``); unstated, the reason is
``docint chat``.

An answer is read as a stream and refused past :data:`MAX_ANSWER_BYTES`,
when it is not ``application/json``, or when it is compressed (the client asks
for ``identity``, so the cap counts the bytes received). workbench never
streams, so SSE is refused although MCP asks clients to accept it. A refused
run is mapped by its ``structuredContent.reason`` token, never by its text.
``run_python`` never raises; a failure is a :class:`RunFailure` whose
``token`` is one of the reason tokens, ``jsonrpc:<code>``, or one of this
client's own: ``timeout``, ``unreachable``, ``not_json``, ``http:<status>``,
``encoded``, ``answer_too_large``, ``not_json_rpc``, ``incomplete``,
``bad_reason``, ``bad_result``, ``bad_image``, ``not_configured``, ``error``.
Discovery fails with the same tokens, or ``no_modern_revision`` and
``no_capped_run_python``.

The log holds statuses, kinds and tokens, never code, data or output. A call
is no ``ai_call``: it calls no model.
"""

from __future__ import annotations

import base64
import binascii
import concurrent.futures
import functools
import hashlib
import hmac
import itertools
import json
import math
import re
import threading
import time
import urllib.parse
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

import anyio
import httpx
from loguru import logger

from docint import __version__
from docint.utils.audit_forward import CHAT_REASON, forward_headers
from docint.utils.env_cfg import WorkbenchClientConfig, load_workbench_env

__all__ = [
    "DISCOVERY_RETRY_S",
    "DISCOVERY_TIMEOUT_S",
    "MAX_ANSWER_BYTES",
    "MAX_IMAGE_BYTES",
    "META_CLIENT_CAPS",
    "META_CLIENT_INFO",
    "META_PROTOCOL",
    "META_SERVER_INFO",
    "PNG_SIGNATURE",
    "PROTOCOL_VERSION",
    "Discovery",
    "FailureKind",
    "InlineFile",
    "OutputFile",
    "RunFailure",
    "RunResult",
    "WorkbenchClient",
    "session_pseudonym",
    "user_pseudonym",
    "workbench_client",
]

#: The one MCP revision this client speaks.
PROTOCOL_VERSION = "2026-07-28"

# The ``_meta`` key names, here and nowhere else.
META_PROTOCOL = "io.modelcontextprotocol/protocolVersion"
META_CLIENT_INFO = "io.modelcontextprotocol/clientInfo"
META_CLIENT_CAPS = "io.modelcontextprotocol/clientCapabilities"
META_SERVER_INFO = "io.modelcontextprotocol/serverInfo"

#: Seconds both discovery requests get together. Never spent from a call's time.
DISCOVERY_TIMEOUT_S = 5.0
#: Seconds after a failed discovery before the next try.
DISCOVERY_RETRY_S = 60.0
#: The largest answer read; past it the answer is refused.
MAX_ANSWER_BYTES = 4 * 1024 * 1024
#: The largest image kept, decoded.
MAX_IMAGE_BYTES = 512 * 1024
#: The eight bytes every PNG starts with.
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

#: What a refused or failed run means for the turn.
FailureKind = Literal["too_much_data", "no_capacity", "code_failed", "unavailable"]

_PATH = "/mcp"
_TOOL = "run_python"
_ACCEPT = "application/json, text/event-stream"
_USER_LABEL = b"workbench-user/v1"
_SESSION_LABEL = b"workbench-session/v1"
_CONNECT_S = 3.0
_WRITE_S = 10.0
_POOL_S = 3.0
# How long past an exchange's deadline its caller waits for the exchange's thread.
_GRACE_S = 0.5
_REASON_KINDS: Mapping[str, FailureKind] = {
    "too_large": "too_much_data",
    "queue_full": "no_capacity",
    "wait_timeout": "no_capacity",
    "executor_busy": "no_capacity",
    "not_completed": "code_failed",
}
# A call without an answer may mean workbench moved; refused arguments may mean a lowered cap;
# -32022 is a revision no longer served, and -32602 covers an unknown tool (mapped by code, never text).
_DROPS_DISCOVERY = frozenset({"timeout", "unreachable", "invalid_arguments", "jsonrpc:-32022", "jsonrpc:-32602"})
# A reason is logged, so only a token's shape passes: no sentence a server might put there.
_TOKEN = re.compile(r"[a-z][a-z0-9_]{0,39}")
_JOB_ID = re.compile(r"[0-9a-f]{32}")
# The version is logged and kept in an analysis's provenance, so only a version's shape passes.
_VERSION = re.compile(r"[0-9A-Za-z.+-]{1,64}")

_T = TypeVar("_T")


@dataclass(frozen=True)
class Discovery:
    """What a successful discovery found.

    Attributes:
        cap_s (int): The largest ``time_limit_s`` workbench accepts from docint.
        server_version (str): workbench's version, or ``""`` when it named none.
    """

    cap_s: int
    server_version: str


@dataclass(frozen=True)
class InlineFile:
    """One file sent with the call, written into the job's working folder.

    Attributes:
        name (str): The file name the code opens.
        content (str): The file's text.
    """

    name: str
    content: str


@dataclass(frozen=True)
class OutputFile:
    """One file the code wrote, as workbench lists it.

    Attributes:
        name (str): The file name.
        mime (str): Its media type.
        size (int): Its size in bytes.
        returned (bool): Whether it came back, or was dropped by a cap or its type.
    """

    name: str
    mime: str
    size: int
    returned: bool


@dataclass(frozen=True)
class RunResult:
    """A run that completed, with what it returned.

    Attributes:
        job_id (str): workbench's job id, 32 hex digits.
        stdout (str): The code's standard output, as workbench bounds it.
        stderr (str): Its standard error.
        truncated (bool): Whether workbench cut either stream.
        files (tuple[OutputFile, ...]): Every file the code wrote, returned or not.
        images (tuple[tuple[str, bytes], ...]): ``(name, png)`` per returned chart.
        texts (tuple[tuple[str, str, str], ...]): ``(name, mime, text)`` per returned text file.
        hint (str | None): workbench's note on what did not come back, if any.
        server_version (str): workbench's version.
    """

    job_id: str
    stdout: str
    stderr: str
    truncated: bool
    files: tuple[OutputFile, ...]
    images: tuple[tuple[str, bytes], ...]
    texts: tuple[tuple[str, str, str], ...]
    hint: str | None
    server_version: str


@dataclass(frozen=True)
class RunFailure:
    """A run that returned nothing usable.

    Attributes:
        kind (FailureKind): What it means for the turn.
        token (str): Why, as a token (see the module docstring); safe to log.
        job_id (str | None): The job workbench named, when it named one.
        answered (bool): Whether a JSON-RPC message came back.
    """

    kind: FailureKind
    token: str
    job_id: str | None = None
    answered: bool = False


@dataclass(frozen=True)
class _Server:
    version: str
    expires_at: float


@dataclass(frozen=True)
class _Tools:
    cap_s: int
    expires_at: float


class _Unusable(Exception):
    """The door's answer, or its absence, cannot be used; ``token`` says why."""

    def __init__(self, token: str, *, answered: bool = False) -> None:
        super().__init__(token)
        self.token = token
        self.answered = answered


def _pseudonym(key: str, label: bytes, value: str) -> str:
    return hmac.new(key.encode("utf-8"), label + b"\x00" + value.encode("utf-8"), hashlib.sha256).hexdigest()


def user_pseudonym(key: str, owner: str) -> str:
    """Return the ``X-User-Id`` workbench sees for a docint owner.

    Args:
        key (str): The shared ``WORKBENCH_API_KEY``.
        owner (str): ``principal.effective_owner``.

    Returns:
        str: HMAC-SHA256 over ``workbench-user/v1``, NUL and the owner, as 64 hex digits.
    """
    return _pseudonym(key, _USER_LABEL, owner)


def session_pseudonym(key: str, session_id: str) -> str:
    """Return the ``X-Session-Id`` workbench sees for a chat session.

    Args:
        key (str): The shared ``WORKBENCH_API_KEY``.
        session_id (str): docint's session id, which a client may have chosen.

    Returns:
        str: HMAC-SHA256 over ``workbench-session/v1``, NUL and the id, as 64 hex digits.
    """
    return _pseudonym(key, _SESSION_LABEL, session_id)


def _client_meta() -> dict[str, Any]:
    return {
        META_PROTOCOL: PROTOCOL_VERSION,
        META_CLIENT_INFO: {"name": "docint", "version": __version__},
        META_CLIENT_CAPS: {},
    }


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _flat_name(value: object) -> str | None:
    if not isinstance(value, str) or value in {"", ".", ".."} or "/" in value or "\\" in value:
        return None
    return None if any(ord(char) < 32 for char in value) else value


def _job_id(value: object) -> str | None:
    return value if isinstance(value, str) and _JOB_ID.fullmatch(value) else None


def _ttl_s(result: Mapping[str, Any]) -> float:
    ttl = result.get("ttlMs")
    if isinstance(ttl, int | float) and not isinstance(ttl, bool) and math.isfinite(ttl) and ttl > 0:
        return ttl / 1000.0
    return 0.0


def _server_version(result: Mapping[str, Any]) -> str | None:
    meta = result.get("_meta")
    info = meta.get(META_SERVER_INFO) if isinstance(meta, dict) else None
    version = info.get("version") if isinstance(info, dict) else None
    return version if isinstance(version, str) and _VERSION.fullmatch(version) else None


def _cap_s(result: Mapping[str, Any]) -> int | None:
    tools = result.get("tools")
    for tool in tools if isinstance(tools, list) else []:
        if isinstance(tool, dict) and tool.get("name") == _TOOL:
            schema = tool.get("inputSchema")
            properties = schema.get("properties") if isinstance(schema, dict) else None
            limit = properties.get("time_limit_s") if isinstance(properties, dict) else None
            cap = limit.get("maximum") if isinstance(limit, dict) else None
            return cap if isinstance(cap, int) and not isinstance(cap, bool) and cap > 0 else None
    return None


def _message(raw: bytes, request_id: int) -> dict[str, Any]:
    """Return the one JSON-RPC message ``raw`` holds, answering ``request_id``.

    Args:
        raw (bytes): The answer's body.
        request_id (int): The id the request carried.

    Returns:
        dict[str, Any]: A message with either a ``result`` object or an
            ``error`` object with an integer ``code``.

    Raises:
        _Unusable: ``not_json_rpc``, when it is anything else.
    """
    try:
        message = json.loads(raw)
    except (ValueError, RecursionError):
        raise _Unusable("not_json_rpc") from None
    if (
        not isinstance(message, dict)
        or message.get("jsonrpc") != "2.0"
        or ("result" in message) == ("error" in message)
    ):
        raise _Unusable("not_json_rpc")
    if "result" in message:
        if message.get("id") != request_id or not isinstance(message["result"], dict):
            raise _Unusable("not_json_rpc")
    else:
        error = message["error"]
        if not isinstance(error, dict) or not _is_int(error.get("code")) or message.get("id") not in (request_id, None):
            raise _Unusable("not_json_rpc")
    return message


def _result(message: Mapping[str, Any]) -> dict[str, Any]:
    """Return a message's complete result.

    Args:
        message (Mapping[str, Any]): A message from :func:`_message`.

    Returns:
        dict[str, Any]: Its result.

    Raises:
        _Unusable: ``jsonrpc:<code>`` for an error, ``incomplete`` for a
            result that asks for more than this client gives.
    """
    if "error" in message:
        raise _Unusable(f"jsonrpc:{int(message['error']['code'])}", answered=True)
    result: dict[str, Any] = message["result"]
    if result.get("resultType", "complete") != "complete":
        raise _Unusable("incomplete", answered=True)
    return result


def _png(item: Mapping[str, Any]) -> bytes:
    data = item.get("data")
    if item.get("mimeType") != "image/png" or not isinstance(data, str):
        raise _Unusable("bad_image", answered=True)
    try:
        png = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise _Unusable("bad_image", answered=True) from None
    if not png.startswith(PNG_SIGNATURE) or len(png) > MAX_IMAGE_BYTES:
        raise _Unusable("bad_image", answered=True)
    return png


def _text(resource: object) -> tuple[str, str, str]:
    if not isinstance(resource, dict):
        raise _Unusable("bad_result", answered=True)
    uri, mime, text = resource.get("uri"), resource.get("mimeType"), resource.get("text")
    if not isinstance(uri, str) or not isinstance(mime, str) or not isinstance(text, str):
        raise _Unusable("bad_result", answered=True)
    try:
        name = _flat_name(urllib.parse.unquote(uri.rsplit("/", 1)[-1], errors="strict"))
    except UnicodeDecodeError:
        name = None
    if name is None:
        raise _Unusable("bad_result", answered=True)
    return name, mime, text


def _files(value: object) -> tuple[OutputFile, ...]:
    if not isinstance(value, list):
        raise _Unusable("bad_result", answered=True)
    files: list[OutputFile] = []
    for entry in value:
        if not isinstance(entry, dict):
            raise _Unusable("bad_result", answered=True)
        name, mime, size, returned = (
            _flat_name(entry.get("name")),
            entry.get("mime"),
            entry.get("bytes"),
            entry.get("returned"),
        )
        if name is None or not isinstance(mime, str) or not isinstance(returned, bool):
            raise _Unusable("bad_result", answered=True)
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise _Unusable("bad_result", answered=True)
        files.append(OutputFile(name=name, mime=mime, size=size, returned=returned))
    return tuple(files)


def _detached(work: Callable[[], Awaitable[_T]], wait_s: float) -> _T:
    """Run ``work`` in an event loop of its own on a daemon thread, and wait at most ``wait_s``.

    Args:
        work (Callable[[], Awaitable[_T]]): The exchange, bounded by its own deadline.
        wait_s (float): Seconds to wait for the thread: the deadline plus :data:`_GRACE_S`.

    Returns:
        _T: What ``work`` returned.

    Raises:
        TimeoutError: When the thread has not finished in time. It is abandoned
            and its late result dropped.
    """
    future: concurrent.futures.Future[_T] = concurrent.futures.Future()

    def exchange() -> None:
        try:
            future.set_result(anyio.run(work))
        except BaseException as exc:
            future.set_exception(exc)

    # A fresh context, not the caller's: a copy would carry the running loop's sniffio marker, and
    # anyio.run would refuse. The exchange reads no context variable; its headers are built before.
    threading.Thread(target=exchange, name="workbench-exchange", daemon=True).start()
    return future.result(timeout=wait_s)


class WorkbenchClient:
    """Discover workbench's ``run_python`` and call it, one exchange at a time."""

    def __init__(
        self,
        cfg: WorkbenchClientConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create the client.

        Args:
            cfg (WorkbenchClientConfig): The base URL and the shared key.
            transport (httpx.AsyncBaseTransport | None): Injected transport
                (tests pass an ``httpx.MockTransport``); ``None`` uses httpx's own.
            clock (Callable[[], float]): The monotonic clock the cache and the
                retry delay read.
        """
        self._cfg = cfg
        self._transport = transport
        self._clock = clock
        self._flight = threading.Lock()
        self._ids = itertools.count(1)
        self._server: _Server | None = None
        self._tools: _Tools | None = None
        self._retry_at: float | None = None
        self._up: bool | None = None

    def _timeout(self, t: float) -> httpx.Timeout:
        """Return the per-phase limits inside a deadline of ``t`` seconds.

        Args:
            t (float): The exchange's deadline, which also bounds each read.

        Returns:
            httpx.Timeout: ``t`` for reads; connect 3 s, write 10 s, pool 3 s.
        """
        return httpx.Timeout(t, connect=_CONNECT_S, write=_WRITE_S, pool=_POOL_S)

    def _http(self, t: float) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self._cfg.api_base,
            transport=self._transport,
            trust_env=False,
            timeout=self._timeout(t),
        )

    async def _post(
        self,
        client: httpx.AsyncClient,
        method: str,
        params: Mapping[str, Any],
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Send one JSON-RPC request and read its answer as a stream.

        Args:
            client (httpx.AsyncClient): The exchange's client.
            method (str): The MCP method.
            params (Mapping[str, Any]): Its parameters, without ``_meta``.
            headers (Mapping[str, str] | None): Headers beyond the common ones.

        Returns:
            dict[str, Any]: The answer, one JSON-RPC message.

        Raises:
            _Unusable: When the answer is not one ``application/json`` message
                of at most :data:`MAX_ANSWER_BYTES`.
        """
        request_id = next(self._ids)
        body = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": {**params, "_meta": _client_meta()}}
        sent = {
            "Authorization": f"Bearer {self._cfg.api_key}",
            "Content-Type": "application/json",
            "Accept": _ACCEPT,
            "Accept-Encoding": "identity",
            "MCP-Protocol-Version": PROTOCOL_VERSION,
            "Mcp-Method": method,
            **(headers or {}),
        }
        payload = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        async with client.stream("POST", _PATH, content=payload, headers=sent) as response:
            media_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            if media_type != "application/json":
                raise _Unusable(f"http:{response.status_code}" if response.is_error else "not_json")
            if response.headers.get("content-encoding", "identity").strip().lower() not in {"", "identity"}:
                raise _Unusable("encoded")
            raw = bytearray()
            async for chunk in response.aiter_bytes():
                raw += chunk
                if len(raw) > MAX_ANSWER_BYTES:
                    raise _Unusable("answer_too_large")
        return _message(bytes(raw), request_id)

    # --- discovery -------------------------------------------------------

    def discover(self) -> Discovery | None:
        """Return the cached discovery, discovering first when a result has expired.

        Concurrent callers wait for the one discovery in progress, never
        longer than its limit plus half a second; a caller that would is
        offered no tool on this turn. A failure is tried again at most once
        per :data:`DISCOVERY_RETRY_S`.

        Returns:
            Discovery | None: The cap and the version, or ``None`` when
                workbench is not usable on this turn.
        """
        if not self._cfg.enabled:
            return None
        # The holder waits at most the same, so a turn is never held past discovery's own limit.
        if not self._flight.acquire(timeout=DISCOVERY_TIMEOUT_S + _GRACE_S):
            return None
        try:
            return self._discover_held()
        finally:
            self._flight.release()

    def _discover_held(self) -> Discovery | None:
        """Discover, holding the single-flight lock.

        Returns:
            Discovery | None: As :meth:`discover`.
        """
        now = self._clock()
        server, tools = self._server, self._tools
        server = server if server is not None and now < server.expires_at else None
        tools = tools if tools is not None and now < tools.expires_at else None
        if server is not None and tools is not None:
            return Discovery(cap_s=tools.cap_s, server_version=server.version)
        if self._retry_at is not None and now < self._retry_at:
            return None
        try:
            fresh_server, fresh_tools = _detached(
                lambda: self._discover(server, tools, now), DISCOVERY_TIMEOUT_S + _GRACE_S
            )
        except _Unusable as exc:
            return self._discovery_failed(now, exc.token)
        except (TimeoutError, httpx.TimeoutException):
            return self._discovery_failed(now, "timeout")
        except httpx.TransportError:
            return self._discovery_failed(now, "unreachable")
        except Exception as exc:
            logger.warning("workbench discovery raised {}", type(exc).__name__)
            return self._discovery_failed(now, "error")
        # Only what was fetched is written back: a forget() meanwhile keeps its effect.
        if fresh_server is not server:
            self._server = fresh_server
        if fresh_tools is not tools:
            self._tools = fresh_tools
        self._retry_at = None
        found = Discovery(cap_s=fresh_tools.cap_s, server_version=fresh_server.version)
        if self._up is not True:
            self._up = True
            logger.info(
                "workbench is discovered: run_python is offered, capped at {} s (workbench '{}')",
                found.cap_s,
                found.server_version,
            )
        return found

    async def _discover(self, server: _Server | None, tools: _Tools | None, now: float) -> tuple[_Server, _Tools]:
        with anyio.fail_after(DISCOVERY_TIMEOUT_S):
            async with self._http(DISCOVERY_TIMEOUT_S) as client:
                if server is None:
                    result = _result(await self._post(client, "server/discover", {}))
                    versions = result.get("supportedVersions")
                    if not isinstance(versions, list) or PROTOCOL_VERSION not in versions:
                        raise _Unusable("no_modern_revision", answered=True)
                    server = _Server(version=_server_version(result) or "", expires_at=now + _ttl_s(result))
                if tools is None:
                    result = _result(await self._post(client, "tools/list", {}))
                    cap = _cap_s(result)
                    if cap is None:
                        raise _Unusable("no_capped_run_python", answered=True)
                    tools = _Tools(cap_s=cap, expires_at=now + _ttl_s(result))
        return server, tools

    def _discovery_failed(self, now: float, token: str) -> None:
        self._retry_at = now + DISCOVERY_RETRY_S
        if self._up is not False:
            self._up = False
            logger.warning(
                "workbench discovery failed: {}. run_python is not offered; the next try comes after {} s",
                token,
                int(DISCOVERY_RETRY_S),
            )

    def forget(self) -> None:
        """Drop both cached results, so the next turn discovers again."""
        self._server = None
        self._tools = None

    # --- run_python ------------------------------------------------------

    def run_python(
        self,
        *,
        code: str,
        data: Sequence[InlineFile],
        time_limit_s: int,
        owner: str,
        session_id: str,
        timeout_s: float,
    ) -> RunResult | RunFailure:
        """Run ``code`` at workbench on ``data``, within one deadline. Never raises.

        Args:
            code (str): The model's Python.
            data (Sequence[InlineFile]): The files written beside it.
            time_limit_s (int): The run's limit at workbench, within the cap.
            owner (str): ``principal.effective_owner``; sent as its pseudonym.
            session_id (str): The chat session; sent as its pseudonym.
            timeout_s (float): The deadline for the whole call, name lookup to last
                byte. The caller is answered within it plus half a second.

        Returns:
            RunResult | RunFailure: What the run returned, or why there is nothing.
        """
        try:
            key = self._cfg.api_key
            if key is None or not self._cfg.enabled:
                outcome: RunResult | RunFailure = RunFailure(kind="unavailable", token="not_configured")
            else:
                headers = {
                    "Mcp-Name": _TOOL,
                    "X-User-Id": user_pseudonym(key, owner),
                    "X-Session-Id": session_pseudonym(key, session_id),
                    **forward_headers(system_reason=CHAT_REASON),
                }
                arguments = {
                    "code": code,
                    "data": [{"name": item.name, "content": item.content} for item in data],
                    "time_limit_s": time_limit_s,
                }
                params = {"name": _TOOL, "arguments": arguments}
                outcome = _detached(lambda: self._call(params, headers, timeout_s), timeout_s + _GRACE_S)
        except TimeoutError:
            outcome = RunFailure(kind="unavailable", token="timeout")
        except Exception as exc:
            logger.warning("workbench run_python raised {}", type(exc).__name__)
            outcome = RunFailure(kind="unavailable", token="error")
        return self._settle(outcome)

    async def _call(
        self, params: Mapping[str, Any], headers: Mapping[str, str], timeout_s: float
    ) -> RunResult | RunFailure:
        try:
            with anyio.fail_after(timeout_s):
                async with self._http(timeout_s) as client:
                    message = await self._post(client, "tools/call", params, headers)
        except (TimeoutError, httpx.TimeoutException):
            return RunFailure(kind="unavailable", token="timeout")
        except httpx.TransportError:
            return RunFailure(kind="unavailable", token="unreachable")
        except _Unusable as exc:
            return RunFailure(kind="unavailable", token=exc.token, answered=exc.answered)
        return self._decode(message)

    def _decode(self, message: Mapping[str, Any]) -> RunResult | RunFailure:
        """Read a ``tools/call`` answer.

        Args:
            message (Mapping[str, Any]): One JSON-RPC message.

        Returns:
            RunResult | RunFailure: The outputs, or the failure the answer names.
        """
        try:
            result = _result(message)
        except _Unusable as exc:
            return RunFailure(kind="unavailable", token=exc.token, answered=True)
        structured = result.get("structuredContent")
        structured = structured if isinstance(structured, dict) else {}
        job_id = _job_id(structured.get("job_id"))
        if result.get("isError") is True:
            reason = structured.get("reason")
            token = reason if isinstance(reason, str) and _TOKEN.fullmatch(reason) else "bad_reason"
            return RunFailure(kind=_REASON_KINDS.get(token, "unavailable"), token=token, job_id=job_id, answered=True)
        try:
            return self._outputs(result, structured, job_id)
        except _Unusable as exc:
            return RunFailure(kind="unavailable", token=exc.token, job_id=job_id, answered=True)

    def _outputs(self, result: Mapping[str, Any], structured: Mapping[str, Any], job_id: str | None) -> RunResult:
        stdout, stderr = structured.get("stdout"), structured.get("stderr")
        truncated, hint = structured.get("truncated"), structured.get("hint")
        content = result.get("content")
        if job_id is None or not isinstance(stdout, str) or not isinstance(stderr, str):
            raise _Unusable("bad_result", answered=True)
        if not isinstance(truncated, bool) or not (hint is None or isinstance(hint, str)):
            raise _Unusable("bad_result", answered=True)
        if not isinstance(content, list) or not all(isinstance(item, dict) for item in content):
            raise _Unusable("bad_result", answered=True)
        files = _files(structured.get("files"))
        images = [_png(item) for item in content if item.get("type") == "image"]
        texts = tuple(_text(item.get("resource")) for item in content if item.get("type") == "resource")
        charts = [item.name for item in files if item.returned and item.name.lower().endswith(".png")]
        if len(charts) != len(images):
            raise _Unusable("bad_result", answered=True)
        cached = self._server
        version = _server_version(result) or (cached.version if cached is not None else "")
        return RunResult(
            job_id=job_id,
            stdout=stdout,
            stderr=stderr,
            truncated=truncated,
            files=files,
            images=tuple(zip(charts, images, strict=True)),
            texts=texts,
            hint=hint,
            server_version=version,
        )

    def _settle(self, outcome: RunResult | RunFailure) -> RunResult | RunFailure:
        if isinstance(outcome, RunResult):
            logger.debug(
                "workbench ran run_python as job '{}': {} files, {} charts and {} texts returned",
                outcome.job_id,
                len(outcome.files),
                len(outcome.images),
                len(outcome.texts),
            )
            return outcome
        if outcome.token in _DROPS_DISCOVERY:
            self.forget()
        answered = "answered" if outcome.answered else "no answer"
        if outcome.kind == "unavailable":
            logger.warning("workbench run_python failed: {} ({})", outcome.token, answered)
        else:
            logger.info("workbench refused run_python: {}, reason {}", outcome.kind, outcome.token)
        return outcome


@functools.cache
def workbench_client() -> WorkbenchClient | None:
    """Return the process-wide workbench client.

    Returns:
        WorkbenchClient | None: The client, or ``None`` while
            ``load_workbench_env().enabled`` is false.
    """
    cfg = load_workbench_env()
    return WorkbenchClient(cfg) if cfg.enabled else None
