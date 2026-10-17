"""Test doubles for assessment turns. Every passage and query here is invented."""

from __future__ import annotations

import contextlib
import json
import threading
import time
from collections.abc import Generator, Iterator, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from unittest.mock import MagicMock

import httpx
import openai
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole, ToolCallBlock
from llama_index.core.bridge.pydantic import Field
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle, TextNode
from typing_extensions import override

#: What the production LLM view lets through for these doubles.
_VISIBLE = frozenset({"citation_index", "filename", "page"})


def scored_node(node_id: str, text: str, **metadata: Any) -> NodeWithScore:
    """Build a retrieved passage.

    Args:
        node_id (str): The node id (the ledger's dedupe key).
        text (str): The passage text.
        **metadata (Any): Node metadata such as ``filename`` and ``page``.

    Returns:
        NodeWithScore: The passage with a neutral score.
    """
    return NodeWithScore(node=TextNode(id_=node_id, text=text, metadata=dict(metadata)), score=1.0)


POST_TEXT = "Harbour festival moved to Sunday because of the storm warning. Stalls open at ten, music from noon."


def social_post(number: int, text: str = POST_TEXT) -> NodeWithScore:
    """Build one pinned post of an invented social export, shaped like a dossier row.

    Its ``table`` and ``reference_metadata`` keys stay visible to the production
    LLM view, which renders them as a header of about 335 characters.

    Args:
        number (int): The post's position; it also varies its ids.
        text (str): The post's text.

    Returns:
        NodeWithScore: The post with a neutral score.
    """
    metadata = {
        "file_name": "example_profile/dossier.json",
        "filename": "example_profile/dossier.json",
        "file_hash": "f" * 64,
        "source": "table",
        "table": {"style": "postings", "row_index": number, "n_rows": 120},
        "reference_metadata": {
            "network": "examplenet",
            "type": "posting",
            "uuid": f"{number:024x}",
            "timestamp": "2026-03-14T09:30:00Z",
            "author": "Harbour Events",
            "author_id": "1234567890",
            "text_id": f"{9876543210 + number}",
        },
    }
    return NodeWithScore(node=TextNode(id_=f"post-{number}", text=text, metadata=metadata), score=1.0)


#: About a thousand characters, under the 1 KiB clamp the production LLM view puts on each metadata leaf.
LONG_LEAF = ("Harbour Events community calendar entry for the spring festival. " * 16)[:990]


def heavy_post(number: int) -> NodeWithScore:
    """Build a short post whose visible metadata renders a header of about 3 KB.

    Each ``reference_metadata`` leaf stays under the per-leaf clamp, so nothing
    bounds the header but the number of leaves.

    Args:
        number (int): The post's position; it also varies its id and text.

    Returns:
        NodeWithScore: The post with a neutral score.
    """
    metadata = {
        "filename": "example_profile/dossier.json",
        "reference_metadata": {
            "network": "examplenet",
            "author": LONG_LEAF,
            "text_id": f"{9876543210 + number}",
            "description": LONG_LEAF,
            "summary": LONG_LEAF,
        },
    }
    text = f"Harbour post {number}: the stalls open at ten."
    return NodeWithScore(node=TextNode(id_=f"heavy-{number}", text=text, metadata=metadata), score=1.0)


def llm_view(node: BaseNode) -> BaseNode:
    """Mirror the production LLM view: hide every key but the whitelisted ones.

    Args:
        node (BaseNode): A ledger node.

    Returns:
        BaseNode: A clone whose LLM rendering shows only the whitelisted keys.
    """
    clone = node.model_copy(deep=True)
    clone.excluded_llm_metadata_keys = [key for key in clone.metadata if key not in _VISIBLE]
    return clone


def bad_request(message: str) -> openai.BadRequestError:
    """Build the error an OpenAI-compatible endpoint raises on HTTP 400.

    Args:
        message (str): The error message.

    Returns:
        openai.BadRequestError: The error.
    """
    request = httpx.Request("POST", "http://inference.invalid/v1/chat/completions")
    return openai.BadRequestError(message, response=httpx.Response(400, request=request), body=None)


def streaming_llm(tokens: Sequence[str], *, failures: Sequence[Exception] = ()) -> MagicMock:
    """Build a chat model whose ``stream_chat`` streams ``tokens``, recording each call.

    Args:
        tokens (Sequence[str]): The answer, token by token.
        failures (Sequence[Exception]): Errors the first calls raise, one per call,
            on their first pull — where an endpoint's HTTP error surfaces.

    Returns:
        MagicMock: The model double; ``stream_chat.call_args`` holds the messages and kwargs.
    """
    llm = MagicMock()
    pending = list(failures)

    def _stream(messages: Sequence[ChatMessage], **kwargs: Any) -> Iterator[ChatResponse]:
        """Raise the next scripted failure on the first pull, or stream the answer.

        Args:
            messages (Sequence[ChatMessage]): The prompt, unused.
            **kwargs (Any): Call options, unused.

        Yields:
            ChatResponse: One response per token, the text so far included.

        Raises:
            Exception: The next scripted failure, while any remain.
        """
        if pending:
            raise pending.pop(0)
        text = ""
        for token in tokens:
            text += token
            yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=text), delta=token)

    llm.stream_chat.side_effect = _stream
    return llm


def tool_llm(script: Sequence[ToolSelection | str | Exception]) -> MagicMock:
    """Build a function-calling model double that replays one scripted reply per call.

    Args:
        script (Sequence[ToolSelection | str | Exception]): Per call: a tool
            call to return, plain text to answer with, or an error to raise.
            After the script runs out, the model calls ``finish``.

    Returns:
        MagicMock: The model double; ``chat_with_tools.call_args_list`` records every call.
    """
    replies = list(script)
    llm = MagicMock()

    def _chat_with_tools(tools: Sequence[Any], **kwargs: Any) -> ChatResponse:
        """Replay the next scripted reply as a chat response, or raise it.

        Args:
            tools (Sequence[Any]): The tools offered, unused.
            **kwargs (Any): The call's other arguments, unused.

        Returns:
            ChatResponse: A tool call, or plain text.

        Raises:
            Exception: The scripted error, when that is the next reply.
        """
        reply: ToolSelection | str | Exception = (
            replies.pop(0) if replies else ToolSelection(tool_id="", tool_name="finish", tool_kwargs={})
        )
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, ToolSelection):
            block = ToolCallBlock(
                tool_call_id=reply.tool_id or None, tool_name=reply.tool_name, tool_kwargs=reply.tool_kwargs
            )
            return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, blocks=[block]))
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=reply))

    def _tool_calls(response: ChatResponse, error_on_no_tool_call: bool = True, **kwargs: Any) -> list[ToolSelection]:
        """Read the tool calls out of a response, as the real model does.

        Args:
            response (ChatResponse): The response to read.
            error_on_no_tool_call (bool): Accepted for signature parity, unused.
            **kwargs (Any): Other arguments, unused.

        Returns:
            list[ToolSelection]: The calls; empty for a plain-text response.
        """
        return [
            ToolSelection(
                tool_id=block.tool_call_id or "",
                tool_name=block.tool_name,
                tool_kwargs=block.tool_kwargs if isinstance(block.tool_kwargs, dict) else {},
            )
            for block in response.message.blocks
            if isinstance(block, ToolCallBlock)
        ]

    llm.chat_with_tools.side_effect = _chat_with_tools
    llm.get_tool_calls_from_response.side_effect = _tool_calls
    return llm


class FixedRetriever(BaseRetriever):
    """Return the same invented hits for any query, as a reranked retriever would hand them on."""

    def __init__(self, hits: Sequence[NodeWithScore]) -> None:
        """Hold the hits.

        Args:
            hits (Sequence[NodeWithScore]): What every retrieval returns, best first.
        """
        super().__init__()
        self._hits = list(hits)

    @override
    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Return the hits.

        Args:
            query_bundle (QueryBundle): The query, unused.

        Returns:
            list[NodeWithScore]: The fixed hits.
        """
        return list(self._hits)


class RecordingPostprocessor(BaseNodePostprocessor):
    """Record the bundle each retrieval hands the postprocessor chain, and change nothing.

    Attributes:
        bundles (list[QueryBundle | None]): One bundle per pass, in order.
    """

    bundles: list[QueryBundle | None] = Field(default_factory=list)

    @override
    def _postprocess_nodes(
        self, nodes: list[NodeWithScore], query_bundle: QueryBundle | None = None
    ) -> list[NodeWithScore]:
        """Record the bundle and pass the nodes through.

        Args:
            nodes (list[NodeWithScore]): The retrieved nodes.
            query_bundle (QueryBundle | None): The bundle the engine passes on.

        Returns:
            list[NodeWithScore]: The nodes, unchanged.
        """
        self.bundles.append(query_bundle)
        return nodes


def search_call(query: str, call_id: str = "call_x") -> ToolSelection:
    """Build a scripted ``search`` tool call.

    Args:
        query (str): The search query.
        call_id (str): The tool-call id the model returns.

    Returns:
        ToolSelection: The scripted call.
    """
    return ToolSelection(tool_id=call_id, tool_name="search", tool_kwargs={"query": query})


class _LoopbackServer(ThreadingHTTPServer):
    """A loopback HTTP server that queues many connections and never waits on its handler threads."""

    request_queue_size = 64
    daemon_threads = True


#: How many pieces :func:`fragmented_endpoint` sends a reply in, and the pause before each.
REPLY_FRAGMENTS = 12
REPLY_FRAGMENT_GAP_S = 0.25


@contextlib.contextmanager
def fragmented_endpoint(body: dict[str, Any]) -> Iterator[str]:
    """Serve one canned chat completion over loopback, a fragment at a time, and yield its ``/v1`` base URL.

    Status and headers go out at once; the body follows in ``REPLY_FRAGMENTS``
    pieces ``REPLY_FRAGMENT_GAP_S`` apart. No read waits long enough for a
    per-operation timeout of a second to fire, yet the reply takes about three.

    Args:
        body (dict[str, Any]): The completion every request is answered with.

    Yields:
        str: The base URL, ending in ``/v1``.
    """
    raw = json.dumps(body).encode()
    size = -(-len(raw) // REPLY_FRAGMENTS)

    class _Fragmented(BaseHTTPRequestHandler):
        """Answers every request with the canned completion, a fragment at a time."""

        def do_POST(self) -> None:
            """Read the request, then send the reply slowly; a client that has gone is ignored."""
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                for start in range(0, len(raw), size):
                    time.sleep(REPLY_FRAGMENT_GAP_S)
                    self.wfile.write(raw[start : start + size])
                    self.wfile.flush()
            except OSError:
                return

        @override
        def log_message(self, format: str, *args: Any) -> None:
            """Keep the test output quiet.

            Args:
                format (str): The message template, unused.
                *args (Any): Its values, unused.
            """

    server = _LoopbackServer(("127.0.0.1", 0), _Fragmented)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@contextlib.contextmanager
def sse_endpoint(*streams: Sequence[dict[str, Any]]) -> Generator[tuple[str, list[dict[str, Any]]], None, None]:
    """Serve scripted chat-completion streams over loopback as server-sent events.

    The SSE variant of :func:`fragmented_endpoint`: request ``n`` is answered with stream
    ``n`` (the last one again once they run out), one ``data:`` event per chunk and then
    ``data: [DONE]``, each written and flushed in turn with no pause. Every request body
    is recorded, parsed, in arrival order.

    Args:
        *streams (Sequence[dict[str, Any]]): Each reply's chunks, in wire shape.

    Yields:
        tuple[str, list[dict[str, Any]]]: The base URL, ending in ``/v1``, and the recorded request bodies.
    """
    done = b"data: [DONE]\n\n"
    replies = [[*(f"data: {json.dumps(item)}\n\n".encode() for item in stream), done] for stream in streams]
    requests: list[dict[str, Any]] = []
    lock = threading.Lock()

    class _Events(BaseHTTPRequestHandler):
        """Answers each request with its scripted stream of events."""

        def do_POST(self) -> None:
            """Record the request, then send its stream; a client that has gone is ignored."""
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            with lock:
                requests.append(body)
                events = replies[min(len(requests), len(replies)) - 1]
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(sum(len(event) for event in events)))
                self.end_headers()
                for event in events:
                    self.wfile.write(event)
                    self.wfile.flush()
            except OSError:
                return

        @override
        def log_message(self, format: str, *args: Any) -> None:
            """Keep the test output quiet.

            Args:
                format (str): The message template, unused.
                *args (Any): Its values, unused.
            """

    server = _LoopbackServer(("127.0.0.1", 0), _Events)
    # A short poll lets shutdown return at once rather than after the default half second.
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def join_threads(name: str) -> None:
    """Wait up to ten seconds for every thread with ``name`` to end, so an abandoned call gives its slot back.

    Args:
        name (str): The thread name the work runs under.
    """
    for thread in threading.enumerate():
        if thread.name == name:
            thread.join(timeout=10)
