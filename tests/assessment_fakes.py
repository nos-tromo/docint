"""Test doubles for agent turns: passages, the LLM view, endpoint errors and loopback endpoints. All invented."""

from __future__ import annotations

import contextlib
import json
import threading
from collections.abc import Generator, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
import openai
from llama_index.core.bridge.pydantic import Field
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


class _LoopbackServer(ThreadingHTTPServer):
    """A loopback HTTP server that queues many connections and never waits on its handler threads."""

    request_queue_size = 64
    daemon_threads = True


@contextlib.contextmanager
def sse_endpoint(*streams: Sequence[dict[str, Any]]) -> Generator[tuple[str, list[dict[str, Any]]], None, None]:
    """Serve scripted chat-completion streams over loopback as server-sent events.

    Request ``n`` is answered with stream ``n`` (the last one again once they run out), one
    ``data:`` event per chunk and then ``data: [DONE]``, each written and flushed in turn with
    no pause. Every request body is recorded, parsed, in arrival order.

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
