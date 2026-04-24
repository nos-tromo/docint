"""Core RAG engine, ingestion, retrieval, and collection management."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import operator
import os
import re
import shutil
import stat
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence, cast

# isort: off
# Import env_cfg BEFORE any third-party libraries so that HF_HUB_OFFLINE and
# TRANSFORMERS_OFFLINE env vars are set before huggingface_hub caches them.
from docint.utils.env_cfg import (
    EmbeddingConfig,
    GraphRAGConfig,
    HostConfig,
    IngestionConfig,
    NERConfig,
    ModelConfig,
    OpenAIConfig,
    PathConfig,
    RetrievalConfig,
    RuntimeConfig,
    SessionConfig,
    SummaryConfig,
    load_embedding_env,
    load_graphrag_env,
    load_hate_speech_env,
    load_host_env,
    load_ingestion_env,
    load_model_env,
    load_ner_env,
    load_openai_env,
    load_path_env,
    load_retrieval_env,
    load_runtime_env,
    load_session_env,
    load_summary_env,
    resolve_hf_cache_path,
)
# isort: on

import torch
from fastembed import SparseTextEmbedding
from llama_index.core import (
    Response,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core.schema import (
    BaseNode,
    Document,
    MetadataMode,
    NodeWithScore,
    QueryBundle,
    TextNode,
)
from llama_index.core.storage.docstore.keyval_docstore import KVDocumentStore
from llama_index.core.storage.kvstore.types import BaseKVStore
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQueryMode,
)
from llama_index.llms.openai import OpenAI
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.vector_stores.qdrant import QdrantVectorStore
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models
from qdrant_client.async_qdrant_client import AsyncQdrantClient

from docint.core.ner import (
    EntityMergeMode,
    aggregate_ner_sources,
    build_entity_graph,
    build_ner_stats,
    entity_cluster_key,
    graph_neighbors,
    match_entity_text,
    normalize_entities,
    normalize_entity_merge_mode,
    search_entities,
)
from docint.core.ingest.images_service import ImageIngestionService
from docint.core.ingest.ingestion_pipeline import DocumentIngestionPipeline
from docint.core.readers.documents import CorePDFPipelineReader
from docint.core.retrieval_filters import matches_metadata_filters
from docint.core.state.session_manager import SessionManager
from docint.core.storage.sqlite_kvstore import SQLiteKVStore
from docint.core.storage.utils import qdrant_collection_exists
from docint.core.storage.sources import stage_sources_to_qdrant
from docint.utils.embed_chunking import (
    effective_budget,
    estimate_tokens,
    fits_budget,
    resplit_nodes_for_embedding,
)
from docint.utils.embedding_tokenizer import build_embedding_token_counter
from docint.utils.openai_cfg import (
    BudgetedOpenAIEmbedding,
    EmbeddingInputTooLongError,
    LocalOpenAI,
    get_openai_reasoning_effort,
)
from docint.utils.reference_metadata import REFERENCE_METADATA_FIELDS


SUMMARY_CACHE_NAMESPACE = "docint_summary_cache_v1"
SUMMARY_CACHE_PAYLOAD_KEY = "summary_payload"
SUMMARY_CACHE_REVISION_KEY = "summary_revision"
HIDDEN_COLLECTION_SUFFIXES: tuple[str, ...] = ("_images", "_dockv")
BatchSparseEncoding = tuple[list[list[int]], list[list[float]]]
EMPTY_RESPONSE_FALLBACK = (
    "I couldn't generate a grounded answer from the retrieved context. "
    "Please try rephrasing the question or ingesting more relevant documents."
)
DEFAULT_SUMMARIZE_PROMPT = (
    "Provide a concise overview of the active collection. Highlight the main "
    "topics, document types, and notable findings. Focus on text bodies, not "
    "metadata. Limit the response to 15 sentences."
)
DEFAULT_SOCIAL_SUMMARIZE_PROMPT = (
    "Summarize the social or row-based collection using only the cited posts or "
    "rows. Keep posts distinct, use metadata such as network, author, and "
    "timestamp when it helps prevent blending separate claims, and call out "
    "conflicts or uncertainty explicitly."
)
DEFAULT_RETRIEVAL_REWRITE_PROMPT = (
    "Rewrite the user's latest message into a standalone retrieval query.\n"
    "Use prior conversation only to resolve references or omitted details.\n"
    "Do not answer the question. Do not include prior assistant claims unless "
    "the user explicitly refers to them. Return only the rewritten query.\n\n"
    "Conversation context:\n{conversation_context}\n\n"
    "Latest user message:\n{user_msg}\n\n"
    "Standalone retrieval query:"
)
DEFAULT_CONVERSATION_SUMMARY_PROMPT = (
    "Summarize the recent conversation turns for future follow-up question "
    "rewriting.\n"
    "Capture only user goals, resolved references, and grounded assistant "
    "conclusions. Do not add new claims.\n\n"
)
DEFAULT_GROUNDED_TEXT_QA_PROMPT = (
    "You are answering a question from retrieved evidence.\n"
    "Use only the context snippets below.\n"
    "Treat each snippet as a distinct source chunk; do not blend claims across "
    "different chunks unless the overlap is explicit.\n"
    "If snippets conflict, say so. If evidence is insufficient, say that "
    "explicitly.\n"
    "Preserve source-specific metadata such as author, network, timestamp, page, "
    "or row when it matters.\n\n"
    "Context snippets:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Question: {query_str}\n"
    "Grounded answer:"
)
DEFAULT_GROUNDED_REFINE_PROMPT = (
    "You are refining an answer from retrieved evidence.\n"
    "Original question: {query_str}\n"
    "Current answer: {existing_answer}\n"
    "New context snippet(s):\n"
    "---------------------\n"
    "{context_msg}\n"
    "---------------------\n"
    "Update the answer only when the new evidence materially improves or corrects "
    "it. Keep source-specific claims distinct. If the new context is not useful, "
    "return the current answer unchanged.\n"
    "Refined grounded answer:"
)


class EmptyIngestionError(Exception):
    """Raised when an ingestion run produced zero documents/nodes for a fresh collection.

    Carries the collection name so callers (CLI, API) can short-circuit
    gracefully — skip ``select_collection``, emit a warning to the UI,
    avoid leaving a confusing "Ingestion failed" banner behind — instead
    of treating a soft-empty outcome as a hard failure.

    Attributes:
        collection_name (str): The name of the collection whose ingestion
            produced no content.
    """

    def __init__(self, collection_name: str, message: str | None = None) -> None:
        """Initialize the error.

        Args:
            collection_name (str): Name of the collection whose ingestion
                produced no content.
            message (str | None): Optional human-readable message; a sensible
                default referencing ``collection_name`` is used when omitted.
        """
        self.collection_name = collection_name
        super().__init__(
            message or f"No content was ingested into '{collection_name}'."
        )


class SocialSourceDiversityPostprocessor(BaseNodePostprocessor):
    """Deduplicate and diversify row-level social/table retrieval results."""

    diversity_limit: int = 2

    @classmethod
    def class_name(cls) -> str:
        """Return a stable class identifier."""
        return "SocialSourceDiversityPostprocessor"

    @staticmethod
    def _reference_metadata(node: NodeWithScore) -> dict[str, Any]:
        """Extract the reference metadata dict from a retrieved node, which may be nested under the
        "reference_metadata" key in the node's metadata or may be missing entirely.

        Args:
            node (NodeWithScore): The node from which to extract reference metadata.

        Returns:
            dict[str, Any]: The reference metadata dictionary, or an empty dictionary if not present.
        """
        metadata = getattr(node, "metadata", {}) or {}
        reference_metadata = metadata.get("reference_metadata")
        if isinstance(reference_metadata, dict):
            return reference_metadata
        return {}

    @staticmethod
    def _identity_key(node: NodeWithScore) -> str:
        """Extract a stable identity key for a retrieved node based on its content and metadata, which can be used for deduplication.

        Args:
            node (NodeWithScore): The node from which to extract an identity key.

        Returns:
            str: A string identity key that represents the content of the node, such as a text ID, a file hash and row index
                for tabular data, or a normalized text snippet. Returns an empty string if no meaningful identity can be extracted.

        """
        metadata = getattr(node, "metadata", {}) or {}
        reference_metadata = SocialSourceDiversityPostprocessor._reference_metadata(
            node
        )
        text_id = str(reference_metadata.get("text_id") or "").strip()
        if text_id:
            return f"text_id:{text_id}"

        file_hash = str(metadata.get("file_hash") or "").strip()
        table_meta = metadata.get("table") or {}
        row_index = (
            table_meta.get("row_index") if isinstance(table_meta, dict) else None
        )
        if file_hash and row_index is not None:
            return f"row:{file_hash}:{row_index}"

        text_value = str(getattr(node, "text", "") or "").strip()
        if text_value:
            normalized = re.sub(r"\s+", " ", text_value).lower()
            return f"text:{normalized[:240]}"

        return ""

    @staticmethod
    def _diversity_bucket(node: NodeWithScore) -> str:
        """Extract a diversity bucket key for a retrieved node based on its author and time, which can be used to limit near-duplicate results from the same source or time period.

        Args:
            node (NodeWithScore): The node from which to extract a diversity bucket key.

        Returns:
            str: A string representing the diversity bucket key, combining the author and time information.

        Raises:
            ValueError: If the timestamp format is invalid and cannot be parsed, which may indicate unexpected metadata structure.
        """
        metadata = getattr(node, "metadata", {}) or {}
        reference_metadata = SocialSourceDiversityPostprocessor._reference_metadata(
            node
        )
        author = str(
            reference_metadata.get("author_id")
            or reference_metadata.get("author")
            or metadata.get("author_id")
            or metadata.get("author")
            or "unknown"
        ).strip()
        timestamp_raw = str(reference_metadata.get("timestamp") or "").strip()
        time_bucket = "unknown"
        if timestamp_raw:
            try:
                parsed = datetime.fromisoformat(timestamp_raw.replace("Z", "+00:00"))
                time_bucket = parsed.astimezone(timezone.utc).strftime("%Y-%m-%dT%H")
            except ValueError:
                time_bucket = timestamp_raw[:13]
        return f"{author.lower()}::{time_bucket}"

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        """Deduplicate by post identity and cap near-duplicate author/time buckets.

        Args:
            nodes (list[NodeWithScore]): The list of retrieved nodes to postprocess.
            query_bundle (QueryBundle | None): The original query bundle that led to these retrieval results, which may be used for context but is not modified by this postprocessor.

        Returns:
            list[NodeWithScore]: The postprocessed list of nodes, where near-duplicate social or tabular sources have been limited according to the configured diversity limit.
        """
        _ = query_bundle
        seen: set[str] = set()
        bucket_counts: dict[str, int] = defaultdict(int)
        filtered: list[NodeWithScore] = []

        for node in nodes:
            identity = self._identity_key(node)
            if identity and identity in seen:
                continue
            bucket = self._diversity_bucket(node)
            if bucket_counts[bucket] >= max(1, int(self.diversity_limit)):
                continue
            if identity:
                seen.add(identity)
            bucket_counts[bucket] += 1
            filtered.append(node)

        return filtered


class ParentContextPostprocessor(BaseNodePostprocessor):
    """Promote fine-grained retrieval hits to their hierarchical parent context."""

    docstore: Any

    @classmethod
    def class_name(cls) -> str:
        """Return a stable class identifier.

        This is used to determine whether a cached postprocessor can be reused for a given pipeline configuration.

        Returns:
            str: A string identifier for this postprocessor class.
        """
        return "ParentContextPostprocessor"

    @staticmethod
    def _parent_id(node: NodeWithScore) -> str:
        """Extract the parent ID for a retrieved node, if available.

        Args:
            node (NodeWithScore): The retrieved node for which to find the parent ID.

        Returns:
            str: The parent ID if found, otherwise an empty string.
        """
        metadata = getattr(node, "metadata", {}) or {}
        parent_id = str(metadata.get("hier.parent_id") or "").strip()
        if parent_id:
            return parent_id

        raw_node = getattr(node, "node", None)
        parent = getattr(raw_node, "parent_node", None)
        if parent is not None:
            return str(getattr(parent, "node_id", "") or "").strip()
        return ""

    def _load_parent_node(self, parent_id: str) -> BaseNode | None:
        """Load the parent node from the docstore using the given parent ID.

        Args:
            parent_id (str): The ID of the parent node to load.

        Returns:
            BaseNode | None: The loaded parent node if successful, otherwise None if the parent node cannot be loaded.
        """
        if not parent_id:
            return None
        try:
            return self.docstore.get_node(parent_id, raise_error=False)
        except AttributeError:
            return self.docstore.get_document(parent_id, raise_error=False)
        except Exception as exc:
            logger.warning(
                "Failed to load parent node '{}' from docstore: {}", parent_id, exc
            )
            return None

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        """Replace fine child hits with their deduplicated parent nodes when available.

        Args:
            nodes (list[NodeWithScore]): The list of retrieved nodes to postprocess.
            query_bundle (QueryBundle | None): The original query bundle that led to these retrieval results, which may be used for context but is not modified by this postprocessor.

        Returns:
            list[NodeWithScore]: The postprocessed list of nodes, where child nodes have been replaced by their parent nodes when a parent context is available.
        """
        _ = query_bundle
        expanded: list[NodeWithScore] = []
        seen_parent_ids: set[str] = set()

        for node in nodes:
            parent_id = self._parent_id(node)
            if not parent_id:
                expanded.append(node)
                continue
            if parent_id in seen_parent_ids:
                continue

            parent_node = self._load_parent_node(parent_id)
            if parent_node is None:
                expanded.append(node)
                continue

            seen_parent_ids.add(parent_id)
            expanded.append(NodeWithScore(node=parent_node, score=node.score))

        return expanded


class VLLMRerankPostprocessor(BaseNodePostprocessor):
    """Call a vLLM-compatible rerank endpoint and map results back to nodes."""

    api_base: str
    api_key: str | None = None
    model: str
    timeout: float = 300.0
    top_n: int = 5

    @classmethod
    def class_name(cls) -> str:
        """Return a stable class identifier.

        Returns:
            str: A string identifier for this postprocessor class.
        """
        return "VLLMRerankPostprocessor"

    @staticmethod
    def _node_text(node: NodeWithScore) -> str:
        """Extract the text content from a retrieved node, trying multiple strategies to find meaningful text for reranking.

        Args:
            node (NodeWithScore): The node from which to extract text.

        Returns:
            str: The extracted text content from the node.
        """
        raw_node = getattr(node, "node", None)
        if raw_node is not None:
            try:
                content = raw_node.get_content()
            except AttributeError:
                content = getattr(raw_node, "text", "")
            if isinstance(content, str) and content.strip():
                return content
        text = getattr(node, "text", "")
        return text if isinstance(text, str) else ""

    def _fallback_nodes(self, nodes: list[NodeWithScore]) -> list[NodeWithScore]:
        """Fallback strategy to return original nodes in a stable order when vLLM reranking fails.

        Args:
            nodes (list[NodeWithScore]): The original list of nodes to return in fallback.

        Returns:
            list[NodeWithScore]: The fallback list of nodes, which is a slice of the original nodes list up to the configured
                top_n limit, ensuring at least one node is returned if available.
        """
        return nodes[: max(1, min(int(self.top_n), len(nodes)))]

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        """Rerank nodes via vLLM and degrade to the original order on failure.

        Args:
            nodes (list[NodeWithScore]): The list of nodes to rerank.
            query_bundle (QueryBundle | None): The original query bundle that led to these retrieval results, which may contain the original query string needed for reranking.

        Returns:
            list[NodeWithScore]: The reranked list of nodes, or the original order if reranking fails.

        Raises:
            ValueError: If the vLLM rerank response is malformed or does not contain usable results.
            urllib.error.HTTPError: If the HTTP request to the vLLM rerank endpoint fails with an HTTP error.
            urllib.error.URLError: If the HTTP request to the vLLM rerank endpoint fails with a URL error, such as a connection failure or timeout.
        """
        if not nodes:
            return nodes

        query_text = str(getattr(query_bundle, "query_str", "") or "").strip()
        if not query_text:
            return self._fallback_nodes(nodes)

        documents = [self._node_text(node) for node in nodes]
        request_url = f"{self.api_base.rstrip('/')}/rerank"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "query": query_text,
            "documents": documents,
            "top_n": min(max(1, int(self.top_n)), len(documents)),
        }

        try:
            request = urllib.request.Request(
                request_url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_body = json.loads(response.read().decode("utf-8"))
            results = response_body.get("results")
            if not isinstance(results, list):
                raise ValueError("vLLM rerank response did not contain a results list")

            reranked: list[NodeWithScore] = []
            seen_indices: set[int] = set()
            for result in results:
                if not isinstance(result, dict):
                    continue
                index = result.get("index")
                if not isinstance(index, int) or index < 0 or index >= len(nodes):
                    continue
                if index in seen_indices:
                    continue
                seen_indices.add(index)

                score_value = result.get("relevance_score", result.get("score"))
                score = nodes[index].score
                if isinstance(score_value, int | float):
                    score = float(score_value)
                reranked.append(NodeWithScore(node=nodes[index].node, score=score))

            if not reranked:
                raise ValueError("vLLM rerank response did not contain usable results")
            return reranked
        except (urllib.error.HTTPError, urllib.error.URLError, ValueError) as exc:
            logger.warning(
                "vLLM rerank request failed at '{}': {}. Returning original retrieval order.",
                request_url,
                exc,
            )
            return self._fallback_nodes(nodes)


class LazyRerankerPostprocessor(BaseNodePostprocessor):
    """Defer reranker materialization until the first postprocessing call.

    Accessing ``rag.reranker`` triggers the lazy-init property that loads
    bge-reranker-v2-m3 (~1 GB) or spins up the vLLM rerank client and runs
    a healthcheck. Plugging the bare ``rag.reranker`` into
    ``node_postprocessors`` at query-engine construction pays that cost
    up-front, even when the caller never intends to execute a query
    (warmup / introspection / preflight patterns). That was the root
    cause of the OOM regression chain — see commits 18a47a6 and 72e299e.

    This wrapper holds a reference to the RAG instance and delegates
    each ``_postprocess_nodes`` call through ``rag.reranker``. The real
    reranker is cached on ``rag._reranker`` by the property itself, so
    only the first query pays the load cost; construction of the query
    engine stays cheap.

    Attributes:
        rag (Any): The owning ``RAG`` instance. Typed ``Any`` because
            ``RAG`` is defined later in this module and Pydantic's
            field validation would otherwise trip on the forward
            reference.
    """

    rag: Any

    @classmethod
    def class_name(cls) -> str:
        """Return a stable class identifier.

        Returns:
            str: A string identifier for this postprocessor class, used
                by LlamaIndex when matching cached configurations.
        """
        return "LazyRerankerPostprocessor"

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        """Delegate to the real reranker, materializing it on first call.

        Args:
            nodes (list[NodeWithScore]): Retrieved nodes awaiting rerank.
            query_bundle (QueryBundle | None): The original query bundle
                forwarded to the underlying reranker unchanged.

        Returns:
            list[NodeWithScore]: Reranked (and typically top-n trimmed)
                nodes as produced by the underlying postprocessor.
        """
        return self.rag.reranker._postprocess_nodes(nodes, query_bundle)


def _vllm_service_root(api_base: str) -> str:
    """Normalize an OpenAI-compatible base URL to the vLLM service root.

    Args:
        api_base (str): OpenAI-compatible API base URL, typically ending in ``/v1``.

    Returns:
        str: The vLLM service root without the trailing ``/v1`` suffix.
    """

    normalized = api_base.rstrip("/")
    return normalized.removesuffix("/v1")


@dataclass(slots=True)
class VLLMSparseEncoder:
    """Adapter that turns vLLM pooling/tokenize responses into Qdrant sparse vectors."""

    api_base: str
    model: str
    api_key: str | None = None
    timeout: float = 300.0

    def encode_texts(self, texts: list[str]) -> BatchSparseEncoding:
        """Encode texts as sparse vectors using the configured vLLM service.

        Args:
            texts (list[str]): Input texts to encode.

        Returns:
            BatchSparseEncoding: Sparse indices and values aligned with the input order.
        """

        if not texts:
            return [], []

        score_batches = self._pool_token_scores(texts)
        sparse_indices: list[list[int]] = []
        sparse_values: list[list[float]] = []

        for text, token_scores in zip(texts, score_batches, strict=False):
            token_ids = self._tokenize(text)
            indices, values = self._build_sparse_vector(token_ids, token_scores)
            sparse_indices.append(indices)
            sparse_values.append(values)

        return sparse_indices, sparse_values

    def _headers(self) -> dict[str, str]:
        """Build the JSON request headers for vLLM requests.

        Returns:
            dict[str, str]: The headers for the vLLM request.
        """

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _request_json(self, url: str, payload: dict[str, Any]) -> Any:
        """POST JSON to a vLLM endpoint and decode the JSON response.

        Args:
            url (str): The full URL of the vLLM endpoint to which the request should be sent.
            payload (dict[str, Any]): A dictionary representing the JSON payload to be sent in the POST request.

        Returns:
            Any: The decoded JSON response from the vLLM service, which may be a dictionary, list, or other JSON structure depending on the endpoint.
        """

        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _pool_token_scores(self, texts: list[str]) -> list[list[float]]:
        """Fetch token-level sparse scores for a batch of texts.

        Args:
            texts (list[str]): A list of input texts for which to pool token scores.

        Returns:
            list[list[float]]: A list of token score lists, where each inner list corresponds to the token scores for the respective input text. The length of the outer list matches the length of the input texts, and each inner list contains float scores aligned with the tokens of the corresponding text.
        """

        request_url = f"{_vllm_service_root(self.api_base)}/pooling"
        response_body = self._request_json(
            request_url,
            {
                "model": self.model,
                "task": "token_classify",
                "input": texts,
            },
        )

        response_data = (
            response_body.get("data") if isinstance(response_body, dict) else None
        )
        if not isinstance(response_data, list):
            raise ValueError("vLLM sparse pooling response did not contain a data list")

        pooled_scores: list[list[float]] = []
        for item in response_data:
            raw_scores = item.get("data") if isinstance(item, dict) else item
            pooled_scores.append(self._coerce_token_scores(raw_scores))

        if len(pooled_scores) != len(texts):
            raise ValueError(
                "vLLM sparse pooling response count did not match the input batch size"
            )
        return pooled_scores

    def _tokenize(self, text: str) -> list[int]:
        """Tokenize a single text through the vLLM tokenizer endpoint.

        Args:
            text (str): The input text to be tokenized.

        Returns:
            list[int]: A list of token IDs corresponding to the input text.
        """

        request_url = f"{_vllm_service_root(self.api_base)}/tokenize"
        response_body = self._request_json(
            request_url,
            {
                "model": self.model,
                "prompt": text,
            },
        )
        token_ids = self._extract_token_ids(response_body)
        if not token_ids:
            raise ValueError("vLLM tokenize response did not contain token ids")
        return token_ids

    @classmethod
    def _extract_token_ids(cls, payload: Any) -> list[int]:
        """Extract token ids from a vLLM tokenize response payload.

        Args:
            payload (Any): The JSON-decoded response from the vLLM tokenize endpoint, which may have various structures but is expected to contain token ID information in one of several possible locations.

        Returns:
            list[int]: A list of token IDs extracted from the payload. If no valid token IDs can be found, an empty list is returned.
        """

        candidates: list[Any] = []
        if isinstance(payload, dict):
            candidates.extend(
                [
                    payload.get("token_ids"),
                    payload.get("tokens"),
                    payload.get("prompt_token_ids"),
                ]
            )
            data = payload.get("data")
            if isinstance(data, list) and data:
                candidates.extend(data)

        for candidate in candidates:
            if isinstance(candidate, dict):
                nested = cls._extract_token_ids(candidate)
                if nested:
                    return nested
                continue
            if (
                isinstance(candidate, list)
                and candidate
                and all(isinstance(item, int) for item in candidate)
            ):
                return [int(item) for item in candidate]
            if isinstance(candidate, list) and not candidate:
                return []

        return []

    @classmethod
    def _coerce_token_scores(cls, raw_scores: Any) -> list[float]:
        """Normalize pooled token outputs into one float score per token.

        Args:
            raw_scores (Any): The raw token scores from the vLLM sparse pooling response.

        Returns:
            list[float]: A list of float scores corresponding to each token.
        """

        if not isinstance(raw_scores, list):
            raise ValueError("vLLM sparse pooling item did not contain a score list")

        token_scores: list[float] = []
        for item in raw_scores:
            if isinstance(item, int | float):
                token_scores.append(float(item))
                continue

            if isinstance(item, list | tuple):
                numeric_values = [
                    float(value) for value in item if isinstance(value, int | float)
                ]
                if not numeric_values:
                    continue
                if len(numeric_values) == 1:
                    token_scores.append(numeric_values[0])
                else:
                    token_scores.append(max(numeric_values))
                continue

            raise ValueError("vLLM sparse pooling item contained a non-numeric score")

        return token_scores

    @staticmethod
    def _build_sparse_vector(
        token_ids: list[int],
        token_scores: list[float],
    ) -> tuple[list[int], list[float]]:
        """Aggregate token ids and scores into a Qdrant sparse vector.

        Args:
            token_ids (list[int]): A list of token IDs corresponding to the input text.
            token_scores (list[float]): A list of token scores corresponding to the input text, aligned with the token IDs.

        Returns:
            tuple[list[int], list[float]]: A tuple containing two lists: the first list is the aggregated token IDs for the sparse
                vector, and the second list is the corresponding aggregated scores for those token
                IDs. The aggregation process involves merging duplicate token IDs by taking the maximum
                score for each unique token ID, and filtering out any token IDs that are negative or
                have non-finite or non-positive scores. The resulting lists are ordered by token ID
                in ascending order. If there are no valid token IDs after filtering, both lists will be empty.
        """

        if len(token_ids) != len(token_scores):
            logger.debug(
                "vLLM sparse token length mismatch: {} token ids vs {} scores",
                len(token_ids),
                len(token_scores),
            )

        merged_scores: dict[int, float] = {}
        for token_id, score in zip(token_ids, token_scores, strict=False):
            if token_id < 0 or not math.isfinite(score) or score <= 0.0:
                continue
            existing = merged_scores.get(token_id)
            if existing is None or score > existing:
                merged_scores[token_id] = score

        ordered = sorted(merged_scores.items())
        return [token_id for token_id, _ in ordered], [score for _, score in ordered]


@dataclass(slots=True)
class RAG:
    """Represents a Retrieval-Augmented Generation (RAG) model. Handles configuration,
    initialization, and interaction with underlying components like embedding models,
    generation models, and vector stores. Provides methods to start sessions,
    retrieve information, and manage document ingestion.
    """

    # --- Constructor args ---
    qdrant_collection: str
    enable_hybrid: bool = field(default=True)

    # --- Environment config ---
    host_config: HostConfig = field(
        default_factory=load_host_env, init=False, repr=False
    )
    ingestion_config: IngestionConfig = field(
        default_factory=load_ingestion_env, init=False, repr=False
    )
    model_config: ModelConfig = field(
        default_factory=load_model_env, init=False, repr=False
    )
    ner_config: NERConfig = field(default_factory=load_ner_env, init=False, repr=False)
    openai_config: OpenAIConfig = field(
        default_factory=load_openai_env, init=False, repr=False
    )
    embedding_config: EmbeddingConfig = field(
        default_factory=load_embedding_env, init=False, repr=False
    )
    path_config: PathConfig = field(
        default_factory=load_path_env, init=False, repr=False
    )
    runtime_config: RuntimeConfig = field(
        default_factory=load_runtime_env, init=False, repr=False
    )
    graphrag_config: GraphRAGConfig = field(
        default_factory=load_graphrag_env, init=False, repr=False
    )
    retrieval_config: RetrievalConfig = field(
        default_factory=load_retrieval_env, init=False, repr=False
    )
    summary_config: SummaryConfig = field(
        default_factory=load_summary_env, init=False, repr=False
    )
    session_config: SessionConfig = field(
        default_factory=load_session_env, init=False, repr=False
    )

    # --- Models ---
    embed_model_id: str | None = field(default=None, init=False)
    sparse_model_id: str | None = field(default=None, init=False)
    rerank_model_id: str | None = field(default=None, init=False)
    text_model_id: str | None = field(default=None, init=False)

    # --- Named entity recognition ---
    ner_enabled: bool = field(default=False, init=False)
    ner_sources: list[dict[str, Any]] = field(default_factory=list, init=False)
    ner_aggregate_cache: dict[tuple[str, str], dict[str, Any]] = field(
        default_factory=dict, init=False, repr=False
    )
    ner_graph_cache: dict[tuple[str, str, int, int], dict[str, Any]] = field(
        default_factory=dict, init=False, repr=False
    )

    # --- OpenAI parameters ---
    openai_api_base: str | None = field(default=None, init=False)
    openai_api_key: str | None = field(default=None, init=False)
    openai_ctx_window: int = field(default=4096, init=False)
    openai_dimensions: int | None = field(default=None, init=False)
    openai_max_retries: int = field(default=2, init=False)
    openai_num_output: int = field(default=256, init=False)
    openai_inference_provider: str = field(default="ollama", init=False)
    openai_reuse_client: bool = field(default=True, init=False)
    openai_seed: int = field(default=42, init=False)
    openai_temperature: float = field(default=0.1, init=False)
    openai_thinking_effort: str = field(default="medium", init=False)
    openai_thinking_enabled: bool = field(default=False, init=False)
    openai_timeout: float = field(default=300.0, init=False)
    openai_top_p: float = field(default=0.0, init=False)

    # --- Embedding context budget (separate from chat LLM) ---
    embed_ctx_tokens: int = field(default=8192, init=False)
    embed_char_token_ratio: float = field(default=3.5, init=False)
    embed_ctx_safety_margin: float = field(default=0.95, init=False)
    embed_timeout_seconds: float = field(default=1800.0, init=False)
    embed_batch_size: int = field(default=16, init=False)
    embed_max_retries: int = field(default=1, init=False)
    _embed_token_counter: Callable[[str], list[int]] | None = field(
        default=None, init=False, repr=False
    )

    # --- Path setup ---
    data_dir: Path | None = field(default=None, init=False)
    hf_hub_cache: Path | None = field(default=None, init=False)

    # --- Reranking / retrieval ---
    retrieve_similarity_top_k: int = field(default=20, init=False)
    rerank_top_n: int = field(default=5, init=False)
    rerank_use_fp16: bool = field(default=False, init=False)
    chat_response_mode: str = field(default="auto", init=False)
    vector_store_query_mode: str = field(default="auto", init=False)
    hybrid_alpha: float = field(default=0.5, init=False)
    sparse_top_k: int = field(default=20, init=False)
    hybrid_top_k: int = field(default=20, init=False)
    parent_context_enabled: bool = field(default=True, init=False)
    graphrag_enabled: bool = field(default=False, init=False)
    graphrag_neighbor_hops: int = field(default=1, init=False)
    graphrag_top_k_nodes: int = field(default=100, init=False)
    graphrag_min_edge_weight: int = field(default=1, init=False)
    graphrag_max_neighbors: int = field(default=6, init=False)
    summary_coverage_target: float = field(default=0.70, init=False)
    summary_max_docs: int = field(default=30, init=False)
    summary_per_doc_top_k: int = field(default=4, init=False)
    summary_final_source_cap: int = field(default=24, init=False)
    social_summary_enabled: bool = field(default=True, init=False)
    social_summary_candidate_pool: int = field(default=48, init=False)
    social_summary_diversity_limit: int = field(default=2, init=False)

    # --- Session config ---
    session_store: str = field(default="", init=False)

    # --- Qdrant controls ---
    docstore_batch_size: int = field(default=100, init=False)
    ingest_benchmark_enabled: bool = field(default=False, init=False)
    docstore_max_retries: int = field(default=3, init=False)
    docstore_retry_backoff_seconds: float = field(default=0.25, init=False)
    docstore_retry_backoff_max_seconds: float = field(default=2.0, init=False)
    qdrant_host: str | None = field(default=None, init=False)
    _qdrant_src_dir: Path | None = field(default=None, init=False, repr=False)

    # --- Prompt config ---
    prompt_dir: Path | None = field(default=None, init=False)
    summarize_prompt_path: Path | None = field(default=None, init=False)
    summarize_social_prompt_path: Path | None = field(default=None, init=False)
    conversation_summary_prompt_path: Path | None = field(default=None, init=False)
    rewrite_retrieval_prompt_path: Path | None = field(default=None, init=False)
    grounded_text_qa_prompt_path: Path | None = field(default=None, init=False)
    grounded_refine_prompt_path: Path | None = field(default=None, init=False)
    summarize_prompt: str = field(default="", init=False)
    summarize_social_prompt: str = field(default="", init=False)
    conversation_summary_prompt: str = field(default="", init=False)
    rewrite_retrieval_prompt: str = field(default="", init=False)
    grounded_text_qa_prompt: str = field(default="", init=False)
    grounded_refine_prompt: str = field(default="", init=False)

    # --- Runtime (lazy caches / not in repr) ---
    _device: str | None = field(default=None, init=False, repr=False)
    _embed_model: BaseEmbedding | None = field(default=None, init=False, repr=False)
    _text_model: OpenAI | None = field(default=None, init=False, repr=False)
    _post_retrieval_text_model: OpenAI | None = field(
        default=None, init=False, repr=False
    )
    _reranker: BaseNodePostprocessor | None = field(
        default=None, init=False, repr=False
    )
    _qdrant_client: QdrantClient | None = field(default=None, init=False, repr=False)
    _qdrant_aclient: AsyncQdrantClient | None = field(
        default=None, init=False, repr=False
    )
    _parent_context_support_cache: dict[str, bool] = field(
        default_factory=dict, init=False, repr=False
    )
    _image_ingestion_service: ImageIngestionService | None = field(
        default=None, init=False, repr=False
    )

    # -- Ingested data ---
    dir_reader: SimpleDirectoryReader | None = field(default=None, init=False)
    docs: list[Document] = field(default_factory=list, init=False)
    nodes: list[BaseNode] = field(default_factory=list, init=False)

    # --- Built components (lazy loaded) ---
    index: VectorStoreIndex | None = field(default=None, init=False)
    query_engine: RetrieverQueryEngine | None = field(default=None, init=False)
    sessions: SessionManager | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up any necessary components.

        Raises:
            ValueError: If summarize_prompt_path is not set.
        """
        # --- Host config ---
        self.qdrant_host = self.host_config.qdrant_host

        # --- Ingestion config ---
        self.docstore_batch_size = self.ingestion_config.docstore_batch_size
        self.ingest_benchmark_enabled = self.ingestion_config.ingest_benchmark_enabled
        self.docstore_max_retries = self.ingestion_config.docstore_max_retries
        self.docstore_retry_backoff_seconds = (
            self.ingestion_config.docstore_retry_backoff_seconds
        )
        self.docstore_retry_backoff_max_seconds = (
            self.ingestion_config.docstore_retry_backoff_max_seconds
        )

        # --- Model config ---
        self.embed_model_id = self.model_config.embed_model
        self.rerank_model_id = self.model_config.rerank_model
        self.sparse_model_id = self.model_config.sparse_model
        self.text_model_id = self.model_config.text_model

        # --- OpenAI config ---
        self.openai_api_key = self.openai_config.api_key
        self.openai_api_base = self.openai_config.api_base
        self.openai_ctx_window = self.openai_config.ctx_window
        self.openai_dimensions = self.openai_config.dimensions
        self.openai_max_retries = self.openai_config.max_retries
        self.openai_num_output = self.openai_config.num_output
        self.openai_inference_provider = self.openai_config.inference_provider
        self.openai_reuse_client = self.openai_config.reuse_client
        self.openai_seed = self.openai_config.seed
        self.openai_temperature = self.openai_config.temperature
        self.openai_thinking_effort = self.openai_config.thinking_effort
        self.openai_thinking_enabled = self.openai_config.thinking_enabled
        self.openai_timeout = self.openai_config.timeout
        self.openai_top_p = self.openai_config.top_p

        # --- Embedding context budget (separate from chat LLM) ---
        self.embed_ctx_tokens = self.embedding_config.ctx_tokens
        self.embed_char_token_ratio = self.embedding_config.char_token_ratio
        self.embed_ctx_safety_margin = self.embedding_config.ctx_safety_margin
        self.embed_timeout_seconds = self.embedding_config.timeout_seconds
        self.embed_batch_size = self.embedding_config.batch_size
        self.embed_max_retries = self.embedding_config.max_retries
        logger.info(
            "Embedding context budget: {} tokens (ratio={}, margin={}); "
            "HTTP envelope: timeout={}s, batch_size={}, max_retries={}",
            self.embed_ctx_tokens,
            self.embed_char_token_ratio,
            self.embed_ctx_safety_margin,
            self.embed_timeout_seconds,
            self.embed_batch_size,
            self.embed_max_retries,
        )
        worst_case_wait = self.embed_timeout_seconds * (1 + self.embed_max_retries)
        if worst_case_wait > 3600:
            logger.warning(
                "Embedding worst-case wait is {:.0f}s (timeout={}s × (1 + "
                "max_retries={})); a single stalled batch can hang ingest for "
                "over an hour. Lower EMBED_TIMEOUT_SECONDS or EMBED_MAX_RETRIES "
                "if that is too lenient for your deployment.",
                worst_case_wait,
                self.embed_timeout_seconds,
                self.embed_max_retries,
            )

        # --- Offline embedding tokenizer (authoritative token counts) ---
        # Loaded once per RAG instance from the HF cache populated by
        # `uv run load-models`. When the snapshot is missing the counter
        # is None and the char-ratio estimator takes over; that degraded
        # state is logged loudly so operators see it in every session.
        self._embed_token_counter = build_embedding_token_counter(
            self.model_config.embed_tokenizer_repo,
            self.path_config.hf_hub_cache,
        )
        if self._embed_token_counter is None:
            logger.warning(
                "No embedding tokenizer loaded (repo={!r}, cache={}) — "
                "falling back to char/token ratio {} on a {}-token window "
                "with safety margin {}. Multilingual corpora may overflow the "
                "provider budget; run `uv run load-models` to populate the cache.",
                self.model_config.embed_tokenizer_repo,
                self.path_config.hf_hub_cache,
                self.embed_char_token_ratio,
                self.embed_ctx_tokens,
                self.embed_ctx_safety_margin,
            )
        else:
            logger.info(
                "Embedding tokenizer loaded from {} (repo={!r}) — using "
                "exact token counts for pre-embed fit checks.",
                self.path_config.hf_hub_cache,
                self.model_config.embed_tokenizer_repo,
            )

        # --- Named Entity Recognition (NER) config ---
        self.ner_enabled = self.ner_config.enabled

        # --- Path config ---
        self.path_config = self.path_config
        self.data_dir = self.path_config.data
        self.prompt_dir = self.path_config.prompts
        self._qdrant_src_dir = self.path_config.qdrant_sources
        self.hf_hub_cache = self.path_config.hf_hub_cache

        ## --- Load prompts ---
        if self.prompt_dir:
            self.summarize_prompt_path = self.prompt_dir / "summarize.txt"
            self.summarize_social_prompt_path = self.prompt_dir / "summarize_social.txt"
            self.conversation_summary_prompt_path = (
                self.prompt_dir / "conversation_summary.txt"
            )
            self.rewrite_retrieval_prompt_path = (
                self.prompt_dir / "rewrite_retrieval.txt"
            )
            self.grounded_text_qa_prompt_path = self.prompt_dir / "grounded_qa.txt"
            self.grounded_refine_prompt_path = self.prompt_dir / "grounded_refine.txt"
        if self.summarize_prompt_path is None:
            logger.error(
                "ValueError: summarize_prompt_path is not set. Cannot load summarize prompt."
            )
            raise ValueError(
                "summarize_prompt_path is not set. Cannot load summarize prompt."
            )
        self.summarize_prompt = self._load_prompt_text(
            self.summarize_prompt_path,
            default=DEFAULT_SUMMARIZE_PROMPT,
            required=True,
        )
        self.summarize_social_prompt = self._load_prompt_text(
            self.summarize_social_prompt_path,
            default=DEFAULT_SOCIAL_SUMMARIZE_PROMPT,
        )
        self.conversation_summary_prompt = self._load_prompt_text(
            self.conversation_summary_prompt_path,
            default=DEFAULT_CONVERSATION_SUMMARY_PROMPT,
        )
        self.rewrite_retrieval_prompt = self._load_prompt_text(
            self.rewrite_retrieval_prompt_path,
            default=DEFAULT_RETRIEVAL_REWRITE_PROMPT,
        )
        self.grounded_text_qa_prompt = self._load_prompt_text(
            self.grounded_text_qa_prompt_path,
            default=DEFAULT_GROUNDED_TEXT_QA_PROMPT,
        )
        self.grounded_refine_prompt = self._load_prompt_text(
            self.grounded_refine_prompt_path,
            default=DEFAULT_GROUNDED_REFINE_PROMPT,
        )

        # --- Retrieval config ---
        self.rerank_use_fp16 = self.retrieval_config.rerank_use_fp16
        self.retrieve_similarity_top_k = self.retrieval_config.retrieve_top_k
        self.chat_response_mode = self.retrieval_config.chat_response_mode
        self.vector_store_query_mode = self.retrieval_config.vector_store_query_mode
        self.hybrid_alpha = self.retrieval_config.hybrid_alpha
        self.sparse_top_k = self.retrieval_config.sparse_top_k
        self.hybrid_top_k = self.retrieval_config.hybrid_top_k
        self.parent_context_enabled = self.retrieval_config.parent_context_enabled
        self.rerank_top_n = int(self.retrieve_similarity_top_k // 4)
        self.graphrag_enabled = self.graphrag_config.enabled
        self.graphrag_neighbor_hops = self.graphrag_config.neighbor_hops
        self.graphrag_top_k_nodes = self.graphrag_config.top_k_nodes
        self.graphrag_min_edge_weight = self.graphrag_config.min_edge_weight
        self.graphrag_max_neighbors = self.graphrag_config.max_neighbors

        # --- Session config ---
        self.session_store = self.session_config.session_store
        self.sessions = SessionManager(self)

        # --- Summary config ---
        self.summary_coverage_target = self.summary_config.coverage_target
        self.summary_max_docs = self.summary_config.max_docs
        self.summary_per_doc_top_k = self.summary_config.per_doc_top_k
        self.summary_final_source_cap = self.summary_config.final_source_cap
        self.social_summary_enabled = self.summary_config.social_chunking_enabled
        self.social_summary_candidate_pool = self.summary_config.social_candidate_pool
        self.social_summary_diversity_limit = self.summary_config.social_diversity_limit

    @staticmethod
    def _load_prompt_text(
        path: Path | None,
        *,
        default: str,
        required: bool = False,
    ) -> str:
        """Load prompt text from disk, falling back to a bundled default.

        Args:
            path (Path | None): Optional filesystem path to the prompt template.
            default (str): Fallback prompt text when the file is absent.
            required (bool): Whether a missing prompt should raise an error.

        Returns:
            str: Prompt text for downstream model calls.

        Raises:
            ValueError: If ``required`` is true and no prompt path is available.
        """
        if path is None:
            if required:
                raise ValueError("Prompt path is required but missing.")
            return default
        try:
            return path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            if required:
                raise
            return default

    @property
    def session_id(self) -> str | None:
        """Get the current session ID.

        Returns:
            str | None: The current session ID.
        """
        return self.sessions.session_id if self.sessions else None

    @session_id.setter
    def session_id(self, value: str | None) -> None:
        """Set the current session ID.

        Args:
            value (str | None): The new session ID.
        """
        if self.sessions is not None:
            self.sessions.session_id = value

    @property
    def chat_engine(self) -> Any | None:
        """Get the current chat engine.

        Returns:
            Any | None: The current chat engine.
        """
        return self.sessions.chat_engine if self.sessions else None

    @chat_engine.setter
    def chat_engine(self, value: Any | None) -> None:
        """Set the current chat engine.

        Args:
            value (Any | None): The new chat engine.
        """
        if self.sessions is not None:
            self.sessions.chat_engine = value

    @property
    def chat_memory(self) -> Any | None:
        """Get the current chat memory.

        Returns:
            Any | None: The current chat memory.
        """
        return self.sessions.chat_memory if self.sessions else None

    @chat_memory.setter
    def chat_memory(self, value: Any | None) -> None:
        """Set the current chat memory.

        Args:
            value (Any | None): The new chat memory.
        """
        if self.sessions is not None:
            self.sessions.chat_memory = value

    # --- Properties (lazy loading) ---
    @property
    def qdrant_src_dir(self) -> Path:
        """Best-effort resolution of the host directory where Qdrant stores source data.
        Used only as a *fallback* when we cannot reach the Qdrant API.
        Priority: explicit field -> env var -> platform default under home.

        Returns:
            Path: The Path representing the Qdrant source host directory.

        Raises:
            ValueError: If the path configuration or the Qdrant source host directory is not set.
        """
        if self._qdrant_src_dir is None:
            if self.path_config is None:
                logger.error("ValueError: Path configuration is not set.")
                raise ValueError("Path configuration is not set.")
            env = self.path_config.qdrant_sources
            if env:
                self._qdrant_src_dir = Path(env) if not env.is_absolute() else env
            else:
                home = os.getenv("HOME") or os.getenv("USERPROFILE")
                if home:
                    self._qdrant_src_dir = (
                        Path(home) / ".qdrant" / "storage" / "sources"
                    )
        if self._qdrant_src_dir is None:
            logger.error("ValueError: Qdrant source host directory is not set.")
            raise ValueError("Qdrant source host directory is not set.")
        return self._qdrant_src_dir

    def _resolve_requested_device(self, requested_device: str) -> str | None:
        """Resolve an explicit runtime device preference.

        Args:
            requested_device (str): Normalized ``USE_DEVICE`` preference.

        Returns:
            str | None: Resolved device string, or ``None`` when auto-detection
            should be used instead.
        """
        if requested_device == "cpu":
            logger.info("Using configured CPU device for local workloads.")
            return "cpu"

        if requested_device == "mps":
            if (
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
                and torch.backends.mps.is_built()
            ):
                logger.info("Using configured MPS device for local workloads.")
                return "mps"
            logger.warning(
                "Configured device '{}' is unavailable; falling back to auto detection.",
                requested_device,
            )
            return None

        if requested_device == "cuda" or requested_device.startswith("cuda:"):
            if not torch.cuda.is_available():
                logger.warning(
                    "Configured device '{}' is unavailable; falling back to auto detection.",
                    requested_device,
                )
                return None

            if requested_device.startswith("cuda:"):
                try:
                    device_index = int(requested_device.split(":", maxsplit=1)[1])
                except ValueError:
                    logger.warning(
                        "Configured device '{}' is invalid; falling back to auto detection.",
                        requested_device,
                    )
                    return None
                if device_index < 0 or device_index >= torch.cuda.device_count():
                    logger.warning(
                        "Configured device '{}' is unavailable; falling back to auto detection.",
                        requested_device,
                    )
                    return None

            logger.info(
                "Using configured CUDA device '{}' for local workloads.",
                requested_device,
            )
            return requested_device

        logger.warning(
            "Unsupported USE_DEVICE value '{}'; falling back to auto detection.",
            requested_device,
        )
        return None

    @property
    def device(self) -> str:
        """Returns the device being used for computation.

        Returns:
            str: The device being used ("cpu", "cuda", or "mps").
        """
        if self._device is None:
            requested_device = self.runtime_config.use_device
            if requested_device != "auto":
                resolved_device = self._resolve_requested_device(requested_device)
                if resolved_device is not None:
                    self._device = resolved_device
                    return self._device

            if torch.cuda.is_available():
                self._device = "cuda"
                logger.info("Using CUDA for GPU acceleration.")
            elif (
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
                and torch.backends.mps.is_built()
            ):
                self._device = "mps"
                logger.info("Using MPS for GPU acceleration.")
            else:
                self._device = "cpu"
                logger.info("Using CPU for computation.")
        return self._device

    @property
    def embed_model(self) -> BaseEmbedding:
        """Lazily initializes and returns the embedding model.

        Returns:
            BaseEmbedding: The initialized embedding model.

        Raises:
            ValueError: If embed_model_id is None.
            FileNotFoundError: If the specified Hugging Face embedding model is not found in the local
                cache while in offline mode.
        """
        if self._embed_model is None:
            if self.embed_model_id is None:
                raise ValueError("embed_model_id cannot be None")

            logger.info("Initializing embedding model: {}", self.embed_model_id)

            embedding_kwargs: dict[str, Any] = {
                "api_base": self.openai_api_base,
                "api_key": self.openai_api_key,
                "embed_batch_size": self.embed_batch_size,
                "max_retries": self.embed_max_retries,
                "model_name": self.embed_model_id,
                "reuse_client": False,
                "timeout": self.embed_timeout_seconds,
            }
            if self.openai_dimensions is not None:
                embedding_kwargs["dimensions"] = self.openai_dimensions

            self._embed_model = BudgetedOpenAIEmbedding(
                **embedding_kwargs,
                context_window=self.embed_ctx_tokens,
            )

        return self._embed_model

    @property
    def sparse_model(self) -> str | None:
        """Returns the configured sparse model id for hybrid retrieval.

        Returns:
            str | None: The sparse model id or None if not enabled.

        Raises:
            ValueError: If the sparse model is None or not supported.
            ImportError: If fastembed is not installed when hybrid search is enabled.
        """
        if not self.enable_hybrid:
            return None

        if self.sparse_model_id is None:
            raise ValueError("sparse_model_id is None")

        if self.openai_inference_provider.lower() == "vllm":
            return self.sparse_model_id

        try:
            supported_models = SparseTextEmbedding.list_supported_models()
        except ImportError:
            raise ImportError(
                "fastembed is not installed, but hybrid search is enabled."
            )

        # Check if the configured ID is directly supported
        supported_ids = [m["model"] for m in supported_models]
        chosen = self.sparse_model_id
        if self.sparse_model_id in supported_ids:
            chosen = self.sparse_model_id
        else:
            # Check if it matches a source HF repo (mapping logic)
            for model_desc in supported_models:
                sources = model_desc.get("sources")
                if sources and sources.get("hf") == self.sparse_model_id:
                    logger.info(
                        "Mapped sparse model {} to its source {}",
                        self.sparse_model_id,
                        model_desc["model"],
                    )
                    chosen = model_desc["model"]
                    break
            else:
                logger.error(
                    "ValueError: Sparse model {} not supported. Supported: {}",
                    self.sparse_model_id,
                    supported_ids,
                )
                raise ValueError(
                    f"Sparse model {self.sparse_model_id!r} not supported. "
                    f"Supported: {supported_ids}"
                )

        # Return the canonical fastembed model name.  fastembed will resolve
        # local files via FASTEMBED_CACHE_PATH (set by env_cfg to HF_HUB_CACHE).
        return chosen

    @property
    def reranker(self) -> BaseNodePostprocessor:
        """Lazily initialize the configured reranker postprocessor.

        Returns:
            BaseNodePostprocessor: The initialized reranker.

        Raises:
            ValueError: If rerank_model_id is None.
            NotImplementedError: If FlagEmbeddingReranker fails for an unsupported operation unrelated to meta tensors.
            RuntimeError: If FlagEmbeddingReranker fails for an unsupported runtime condition unrelated to meta tensors.
        """
        if self.rerank_model_id is None:
            raise ValueError("rerank_model_id cannot be None")
        if self._reranker is None:
            provider = self.openai_inference_provider.lower()
            if provider == "vllm":
                self._reranker = VLLMRerankPostprocessor(
                    api_base=self.openai_api_base or "",
                    api_key=self.openai_api_key,
                    model=self.rerank_model_id,
                    timeout=self.openai_timeout,
                    top_n=self.rerank_top_n,
                )
                logger.info(
                    "Initializing vLLM reranker endpoint client with model: {}",
                    self.rerank_model_id,
                )
            else:
                # Resolve to local cache path for offline compatibility
                cache_dir = (
                    self.hf_hub_cache or Path.home() / ".cache" / "huggingface" / "hub"
                )
                resolved = resolve_hf_cache_path(cache_dir, self.rerank_model_id)
                resolved_model = str(resolved) if resolved else self.rerank_model_id
                if resolved:
                    logger.info("Using local reranker model path: {}", resolved_model)
                flag_reranker = FlagEmbeddingReranker(
                    top_n=self.rerank_top_n,
                    model=resolved_model,
                    use_fp16=self.rerank_use_fp16,
                )
                try:
                    flag_reranker._model.compute_score([("healthcheck", "healthcheck")])
                    self._reranker = flag_reranker
                    logger.info(
                        "Initializing FlagEmbeddingReranker with model: {} for provider {}",
                        self.rerank_model_id,
                        provider,
                    )
                except (NotImplementedError, RuntimeError) as exc:
                    if "meta tensor" not in str(exc).lower():
                        raise
                    logger.warning(
                        "FlagEmbeddingReranker failed to initialize due to a meta-tensor device transfer issue: {}. Falling back to LLMRerank.",
                        exc,
                    )
                    self._reranker = LLMRerank(
                        top_n=self.rerank_top_n,
                        llm=self.text_model,
                    )
        return self._reranker

    def _create_text_model(self, *, enable_reasoning: bool = False) -> OpenAI:
        """Helper to create an OpenAI (or compatible) model instance.

        Args:
            enable_reasoning (bool): Whether this model instance should request the
                provider reasoning/thinking mode.

        Returns:
            OpenAI: The initialized model.

        Raises:
            ValueError: If required configuration is missing.
        """
        if self.text_model_id is None:
            raise ValueError("text_model_id cannot be None")

        additional_kwargs: dict[str, Any] = {}
        reasoning_effort = get_openai_reasoning_effort(
            self.openai_config,
            enabled=enable_reasoning,
        )

        # LlamaIndex OpenAI class supports api_key, api_base, timeout, max_retries, seed, top_p
        # Use LocalOpenAI which tolerates unknown model names (e.g. paths) by falling back to default metadata
        model = LocalOpenAI(
            additional_kwargs=additional_kwargs,
            api_base=self.openai_api_base,
            api_key=self.openai_api_key,
            context_window=self.openai_ctx_window,
            max_retries=self.openai_max_retries,
            model=self.text_model_id,
            num_output=self.openai_num_output,
            reuse_client=self.openai_reuse_client,
            reasoning_effort=reasoning_effort,
            seed=self.openai_seed,
            temperature=self.openai_temperature,
            timeout=self.openai_timeout,
            top_p=self.openai_top_p,
        )

        logger.info(
            "Initializing text model: {}",
            self.text_model_id,
        )
        return model

    @property
    def text_model(self) -> OpenAI:
        """Lazily initializes and returns the generation model (OpenAI).

        Returns:
            OpenAI: The initialized generation model.
        """
        if self._text_model is None:
            self._text_model = self._create_text_model()
        return self._text_model

    @property
    def post_retrieval_text_model(self) -> OpenAI:
        """Return the model used for post-retrieval response generation.

        Grounded answer synthesis after retrieval should request provider
        reasoning/thinking. Pre-retrieval steps such as query rewriting remain
        on the default non-reasoning model.

        Returns:
            OpenAI: The post-retrieval generation model.
        """
        if get_openai_reasoning_effort(self.openai_config, enabled=True) is None:
            return self.text_model

        if self._post_retrieval_text_model is None:
            self._post_retrieval_text_model = self._create_text_model(
                enable_reasoning=True
            )
        return self._post_retrieval_text_model

    @property
    def qdrant_client(self) -> QdrantClient:
        """Lazily initializes and returns the Qdrant client.

        Returns:
            QdrantClient: The initialized Qdrant client.
        """
        if self._qdrant_client is None:
            self._qdrant_client = QdrantClient(url=self.qdrant_host)
            logger.info(
                "Qdrant client initialized: {}",
                self.qdrant_host,
            )
        return self._qdrant_client

    @property
    def qdrant_aclient(self) -> AsyncQdrantClient:
        """Lazily initializes and returns the Qdrant async client.

        Returns:
            AsyncQdrantClient: The initialized Qdrant async client.
        """
        if self._qdrant_aclient is None:
            self._qdrant_aclient = AsyncQdrantClient(url=self.qdrant_host)
            logger.info(
                "Qdrant async client initialized: {}",
                self.qdrant_host,
            )
        return self._qdrant_aclient

    # --- Build pieces ---
    def _vector_store(self) -> QdrantVectorStore:
        """Creates the vector store for document embeddings.

        Returns:
            QdrantVectorStore: The initialized vector store.

        Raises:
            ValueError: If qdrant_collection is None.
        """
        if self.qdrant_collection is None:
            logger.error("ValueError: qdrant_collection cannot be None")
            raise ValueError("qdrant_collection cannot be None")

        vector_store_kwargs: dict[str, Any] = {
            "collection_name": self.qdrant_collection,
            "client": self.qdrant_client,
            "aclient": self.qdrant_aclient,
            "enable_hybrid": self.enable_hybrid,
        }
        if self.enable_hybrid and self.openai_inference_provider.lower() == "vllm":
            sparse_encoder = VLLMSparseEncoder(
                api_base=self.openai_api_base or "",
                api_key=self.openai_api_key,
                model=self.sparse_model or "",
                timeout=self.openai_timeout,
            )
            vector_store_kwargs["sparse_doc_fn"] = sparse_encoder.encode_texts
            vector_store_kwargs["sparse_query_fn"] = sparse_encoder.encode_texts
        else:
            vector_store_kwargs["fastembed_sparse_model"] = self.sparse_model

        return QdrantVectorStore(**vector_store_kwargs)

    def _storage_context(self, vector_store: QdrantVectorStore) -> StorageContext:
        """Creates the storage context for document embeddings.

        Args:
            vector_store (QdrantVectorStore): The vector store for document embeddings.

        Returns:
            StorageContext: The created storage context.
        """
        kv_store = self._build_kv_store()
        doc_store = KVDocumentStore(
            kvstore=kv_store, batch_size=self.docstore_batch_size
        )

        return StorageContext.from_defaults(
            vector_store=vector_store,
            docstore=doc_store,
        )

    def _build_kv_store(
        self,
        collection: str | None = None,
    ) -> BaseKVStore:
        """Build a :class:`SQLiteKVStore` for the given collection.

        Args:
            collection: Optional collection name override.  When *None* the
                current ``qdrant_collection`` is used.

        Returns:
            BaseKVStore: A :class:`SQLiteKVStore` rooted at
                ``{qdrant_src_dir}/{collection}/{collection}_kv.db``.
        """
        target = str(collection or self.qdrant_collection or "").strip()
        db_path = self.qdrant_src_dir / target / f"{target}_kv.db"
        return SQLiteKVStore(
            db_path=db_path,
            batch_size=self.docstore_batch_size,
            max_retries=self.docstore_max_retries,
            retry_backoff_seconds=self.docstore_retry_backoff_seconds,
            retry_backoff_max_seconds=self.docstore_retry_backoff_max_seconds,
        )

    def _build_ingestion_pipeline(
        self, progress_callback: Callable[[str], None] | None = None
    ) -> DocumentIngestionPipeline:
        """Instantiate a document ingestion pipeline using current settings.

        Args:
            progress_callback (Callable[[str], None] | None): Optional callback for
                reporting ingestion progress.

        Returns:
            DocumentIngestionPipeline: The instantiated ingestion pipeline.

        Raises:
            ValueError: If data_dir is None.
        """
        if self.data_dir is None:
            logger.error("ValueError: data_dir cannot be None for ingestion pipeline.")
            raise ValueError("data_dir cannot be None for ingestion pipeline.")

        hate_speech_enabled = load_hate_speech_env().enabled
        use_llm_ner = self.ner_enabled and self.openai_inference_provider.lower() in {
            "openai"
        }

        shared_text_model: OpenAI | None = None
        if use_llm_ner or hate_speech_enabled:
            shared_text_model = self.text_model

        ner_model = shared_text_model if use_llm_ner else None
        hate_speech_model = shared_text_model if hate_speech_enabled else None

        if self._image_ingestion_service is None:
            self._image_ingestion_service = ImageIngestionService(device=self.device)

        return DocumentIngestionPipeline(
            data_dir=self.data_dir,
            ner_model=ner_model,
            device=self.device,
            progress_callback=progress_callback,
            hate_speech_model=hate_speech_model,
            openai_inference_provider=self.openai_inference_provider,
            target_collection=self.qdrant_collection,
            image_ingestion_service=self._image_ingestion_service,
        )

    def _retrieve_image_sources(
        self,
        query: str,
        *,
        top_k: int = 3,
        metadata_filter_rules: Sequence[Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve image matches for a text query and normalize them as sources.

        Args:
            query (str): The text query to find similar images for.
            top_k (int): The number of top matches to retrieve.
            metadata_filter_rules (Sequence[Any] | None): Optional raw request
                filters used to post-filter auxiliary image matches in memory.

        Returns:
            list[dict[str, Any]]: A list of source dictionaries representing the matched images.
        """
        if not query.strip() or not self.qdrant_collection:
            return []
        if self._image_ingestion_service is None:
            self._image_ingestion_service = ImageIngestionService(device=self.device)

        try:
            image_collection = self._image_ingestion_service._resolve_collection_name(
                self.qdrant_collection
            )
        except Exception:
            return []
        if not qdrant_collection_exists(self.qdrant_client, image_collection):
            return []

        try:
            matches = self._image_ingestion_service.query_similar_images_by_text(
                query_text=query,
                top_k=top_k,
                source_collection=self.qdrant_collection,
            )
        except Exception as exc:
            logger.warning("Image source retrieval failed: {}", exc)
            return []

        results: list[dict[str, Any]] = []
        seen: set[str] = set()
        for payload in matches:
            if metadata_filter_rules and not matches_metadata_filters(
                payload,
                metadata_filter_rules,
            ):
                continue

            image_id = str(payload.get("image_id") or "").strip()
            if image_id:
                if image_id in seen:
                    continue
                seen.add(image_id)

            description = str(payload.get("llm_description") or "").strip()
            tags_raw = payload.get("llm_tags")
            tags = [str(tag) for tag in tags_raw] if isinstance(tags_raw, list) else []
            text_value = description
            if tags:
                text_value = (
                    f"{description}\n\nTags: {', '.join(tags)}"
                    if description
                    else f"Tags: {', '.join(tags)}"
                )

            source_path = payload.get("source_path")
            file_name = (
                payload.get("file_name")
                or payload.get("filename")
                or (Path(source_path).name if isinstance(source_path, str) else None)
            )
            file_type = payload.get("mime_type") or payload.get("mimetype")
            source_kind = payload.get("source_type") or "image"
            file_hash = payload.get("source_doc_id") or payload.get("file_hash")
            page_number = payload.get("page_number")
            try:
                page_number = int(page_number) if page_number is not None else None
            except Exception:
                page_number = None

            src: dict[str, Any] = {
                "text": text_value
                or f"Image match: {image_id or file_name or 'unknown'}",
                "preview_text": (text_value or "").strip()[:280],
                "filename": file_name,
                "filetype": file_type,
                "source": source_kind,
                "score": payload.get("score"),
                "image_id": image_id or None,
                "image_collection": payload.get("image_collection"),
            }
            if file_hash:
                src["file_hash"] = file_hash
                preview_url = (
                    f"/sources/preview?collection={self.qdrant_collection}"
                    f"&file_hash={file_hash}"
                )
                src["preview_url"] = preview_url
                src["document_url"] = preview_url
            if page_number is not None:
                src["page"] = page_number
            bbox = payload.get("bbox")
            if isinstance(bbox, dict):
                src["bbox"] = bbox
            results.append(src)

        return results

    def _index(self, storage_ctx: StorageContext) -> VectorStoreIndex:
        """Creates the vector store index for document embeddings.

        Args:
            storage_ctx (StorageContext): The storage context for document embeddings.

        Returns:
            VectorStoreIndex: The created vector store index.
        """
        return VectorStoreIndex(
            nodes=self.nodes,
            storage_context=storage_ctx,
            embed_model=self.embed_model,
        )

    @staticmethod
    def _select_vector_nodes(nodes: list[BaseNode]) -> list[BaseNode]:
        """Select nodes that should be inserted into the vector store.

        Args:
            nodes (list[BaseNode]): Parsed nodes for an ingestion batch.

        Returns:
            list[BaseNode]: The subset of nodes suitable for vector indexing.
        """
        is_hierarchical = any("docint_hier_type" in n.metadata for n in nodes)
        if is_hierarchical:
            return [n for n in nodes if n.metadata.get("docint_hier_type") != "coarse"]
        return nodes

    def _resplit_vector_nodes(
        self, nodes: list[BaseNode]
    ) -> tuple[list[BaseNode], list[BaseNode]]:
        """Apply the pre-embed re-splitter to the vector-indexable nodes.

        Args:
            nodes (list[BaseNode]): Vector-indexable nodes for the current
                persistence batch.

        Returns:
            tuple[list[BaseNode], list[BaseNode]]:
                ``(vector_nodes, docstore_nodes)`` — see
                :func:`docint.utils.embed_chunking.resplit_nodes_for_embedding`.
        """
        return resplit_nodes_for_embedding(
            nodes,
            budget_tokens=self.embed_ctx_tokens,
            char_token_ratio=self.embed_char_token_ratio,
            safety_margin=self.embed_ctx_safety_margin,
            token_counter=self._embed_token_counter,
        )

    def _assert_embed_payloads_fit_budget(
        self,
        nodes_to_embed: list[BaseNode],
        texts_to_embed: list[str],
    ) -> None:
        """Guard against any embed payload slipping past the re-splitter.

        Called immediately before handing the batch to the embedding
        client. If the pre-embed re-splitter missed an input — for
        example a downstream path constructed nodes whose
        ``MetadataMode.EMBED`` rendering was not bounded — this check
        raises :class:`EmbeddingInputTooLongError` with the offending
        ``node_id`` and payload statistics instead of letting the
        provider reject the request with a cryptic 400. The detection
        is cheap (O(payload length)) and pays for itself the first time
        it surfaces a regression.

        Args:
            nodes_to_embed: Nodes whose embeddings are about to be
                requested, aligned with ``texts_to_embed``.
            texts_to_embed: The ``MetadataMode.EMBED`` rendering each
                node will be embedded as.

        Raises:
            EmbeddingInputTooLongError: When any payload exceeds the
                configured embedding budget.
        """
        budget = effective_budget(self.embed_ctx_tokens, self.embed_ctx_safety_margin)
        for node, text in zip(nodes_to_embed, texts_to_embed):
            if fits_budget(
                text,
                budget_tokens=self.embed_ctx_tokens,
                char_token_ratio=self.embed_char_token_ratio,
                safety_margin=self.embed_ctx_safety_margin,
                token_counter=self._embed_token_counter,
            ):
                continue
            payload_tokens = estimate_tokens(
                text,
                self.embed_char_token_ratio,
                token_counter=self._embed_token_counter,
            )
            counter_state = (
                "tokenizer" if self._embed_token_counter is not None else "char-ratio"
            )
            raise EmbeddingInputTooLongError(
                "Pre-embed safety net caught an oversize payload: "
                f"node_id={node.node_id} embed_payload_chars={len(text)} "
                f"estimated_tokens={payload_tokens} ({counter_state}) "
                f"budget={budget} "
                f"configured_ctx_tokens={self.embed_ctx_tokens} "
                f"safety_margin={self.embed_ctx_safety_margin} — the "
                "re-splitter missed this node; check node metadata size or "
                "raise EMBED_CTX_TOKENS."
            )

    def _prepare_vector_nodes_for_insert(
        self,
        nodes: list[BaseNode],
    ) -> tuple[list[BaseNode], list[BaseNode]]:
        """Re-split oversize nodes and attach embeddings for the vector store.

        The re-split step produces two aligned views of the input batch:
        a vector view (what the embedding call and vector store see) and
        a docstore view (what the KV store sees, including the oversize
        parent kept for retrieval-time parent-context reconstruction).
        Returning both views eliminates the previous hidden coupling
        with :meth:`_persist_node_batches`, which otherwise had to
        re-derive the docstore view by diffing the vector view against
        the caller's batch.

        Oversize inputs that cannot be reduced below the embedding
        budget raise
        :class:`docint.utils.openai_cfg.EmbeddingInputTooLongError` —
        there is no silent skip.

        Args:
            nodes (list[BaseNode]): Vector-indexable nodes for the
                current persistence batch.

        Returns:
            tuple[list[BaseNode], list[BaseNode]]:
                ``(vector_nodes, docstore_nodes)`` — the first list
                goes to the vector store (oversize parents replaced by
                sub-nodes, each with an attached embedding); the second
                goes to the docstore (oversize parents retained plus
                their sub-nodes, and every within-budget vector node).
        """
        embed_model = self._embed_model
        get_embeddings = getattr(embed_model, "get_text_embeddings_strict", None)
        if embed_model is None or not callable(get_embeddings):
            return nodes, list(nodes)

        vector_nodes, docstore_nodes = self._resplit_vector_nodes(nodes)

        nodes_to_embed: list[BaseNode] = []
        texts_to_embed: list[str] = []
        for node in vector_nodes:
            if node.embedding is not None:
                continue
            nodes_to_embed.append(node)
            texts_to_embed.append(node.get_content(metadata_mode=MetadataMode.EMBED))

        if not nodes_to_embed:
            return vector_nodes, docstore_nodes

        # Slice by ``embed_batch_size`` so each HTTP POST respects the
        # operator's per-request ceiling. The llama_index
        # ``embed_batch_size`` knob only fires inside
        # ``get_text_embedding_batch``; our strict wrapper bypasses it,
        # so the RAG layer must chunk explicitly. The safety net runs
        # per chunk — an oversize payload slipping through the
        # re-splitter raises BEFORE its chunk hits the provider, not
        # after 4 minutes of stalled batch processing. Mirrors the
        # slice idiom in :meth:`_chunk_nodes`; kept inline because we
        # need parallel slicing of ``nodes_to_embed`` and
        # ``texts_to_embed`` in lockstep.
        batch_size = max(1, self.embed_batch_size)
        for start in range(0, len(nodes_to_embed), batch_size):
            chunk_nodes = nodes_to_embed[start : start + batch_size]
            chunk_texts = texts_to_embed[start : start + batch_size]
            self._assert_embed_payloads_fit_budget(chunk_nodes, chunk_texts)
            chunk_embeddings = cast(
                list[list[float]],
                get_embeddings(chunk_texts),
            )
            for node, embedding in zip(chunk_nodes, chunk_embeddings):
                node.embedding = embedding

        return vector_nodes, docstore_nodes

    async def _aprepare_vector_nodes_for_insert(
        self,
        nodes: list[BaseNode],
    ) -> tuple[list[BaseNode], list[BaseNode]]:
        """Async variant of :meth:`_prepare_vector_nodes_for_insert`.

        Args:
            nodes (list[BaseNode]): Vector-indexable nodes for the
                current persistence batch.

        Returns:
            tuple[list[BaseNode], list[BaseNode]]:
                ``(vector_nodes, docstore_nodes)`` — the first list
                goes to the vector store (oversize parents replaced by
                sub-nodes, each with an attached embedding); the second
                goes to the docstore (oversize parents retained plus
                their sub-nodes, and every within-budget vector node).
        """
        embed_model = self._embed_model
        aget_embeddings = getattr(embed_model, "aget_text_embeddings_strict", None)
        if embed_model is None or not callable(aget_embeddings):
            return nodes, list(nodes)

        vector_nodes, docstore_nodes = self._resplit_vector_nodes(nodes)

        nodes_to_embed: list[BaseNode] = []
        texts_to_embed: list[str] = []
        for node in vector_nodes:
            if node.embedding is not None:
                continue
            nodes_to_embed.append(node)
            texts_to_embed.append(node.get_content(metadata_mode=MetadataMode.EMBED))

        if not nodes_to_embed:
            return vector_nodes, docstore_nodes

        # See the sync twin for why this chunking lives in the RAG
        # layer and not inside ``aget_text_embeddings_strict``.
        batch_size = max(1, self.embed_batch_size)
        for start in range(0, len(nodes_to_embed), batch_size):
            chunk_nodes = nodes_to_embed[start : start + batch_size]
            chunk_texts = texts_to_embed[start : start + batch_size]
            self._assert_embed_payloads_fit_budget(chunk_nodes, chunk_texts)
            chunk_embeddings = cast(
                list[list[float]],
                await aget_embeddings(chunk_texts),
            )
            for node, embedding in zip(chunk_nodes, chunk_embeddings):
                node.embedding = embedding

        return vector_nodes, docstore_nodes

    @staticmethod
    def _chunk_nodes(nodes: list[BaseNode], batch_size: int) -> list[list[BaseNode]]:
        """Split nodes into non-empty batches.

        Args:
            nodes (list[BaseNode]): Nodes to split.
            batch_size (int): Preferred maximum batch size.

        Returns:
            list[list[BaseNode]]: Node batches in input order.
        """
        if not nodes:
            return []
        effective_batch_size = max(1, int(batch_size))
        return [
            nodes[i : i + effective_batch_size]
            for i in range(0, len(nodes), effective_batch_size)
        ]

    def _docstore_batch_for_persist(
        self,
        batch: list[BaseNode],
        vector_candidates: list[BaseNode],
        docstore_nodes: list[BaseNode],
    ) -> list[BaseNode]:
        """Compose the docstore batch so sub-nodes and oversize parents both land.

        The re-split step produces a dedicated ``docstore_nodes`` view
        that already contains every vector-candidate (including oversize
        parents kept for parent-context reconstruction) and every
        newly-created sub-node. The docstore batch additionally needs
        every *non*-vector-candidate node from the original batch —
        e.g. coarse parents in hierarchical collections, which never
        reach the vector store yet still belong in the KV store.

        Args:
            batch (list[BaseNode]): Original nodes for this persistence batch.
            vector_candidates (list[BaseNode]): Nodes the vector store would
                normally embed (the pre-resplit selection).
            docstore_nodes (list[BaseNode]): Docstore view returned by
                :meth:`_prepare_vector_nodes_for_insert`. Contains
                oversize parents, their sub-nodes, and every
                within-budget vector-candidate.

        Returns:
            list[BaseNode]: Docstore batch containing non-vector-candidate
                nodes from the original batch followed by the docstore
                view from the re-split step.
        """
        candidate_ids = {id(node) for node in vector_candidates}
        non_vector_candidates = [
            node for node in batch if id(node) not in candidate_ids
        ]
        return non_vector_candidates + list(docstore_nodes)

    def _persist_node_batches(self, nodes: list[BaseNode]) -> None:
        """Persist nodes in micro-batches to reduce crash-loss windows.

        Each batch is written to the KV docstore first and to the vector
        store second.  On failure the node IDs in the affected batch are
        logged under a dedicated marker (``failed_persist_nodes`` for
        docstore failures, ``orphaned_kv_nodes`` for vector-insert
        failures) so operators can identify exactly which nodes need
        re-ingestion after a crash.  The exception is re-raised so
        ingestion aborts.

        Args:
            nodes (list[BaseNode]): Ingestion nodes to persist.

        Raises:
            RuntimeError: If the index is not initialized.
            Exception: Re-raises whatever the underlying KV or vector
                write raised, after emitting a structured log entry.
        """
        if self.index is None:
            raise RuntimeError("Index is not initialized.")

        batches = self._chunk_nodes(nodes, self.docstore_batch_size)
        for batch_no, batch in enumerate(batches, start=1):
            logger.debug(
                "Persisting node batch {}/{} ({} node(s)) to DocStore...",
                batch_no,
                len(batches),
                len(batch),
            )
            vector_candidates = self._select_vector_nodes(batch)
            (
                prepared_vector_nodes,
                prepared_docstore_nodes,
            ) = self._prepare_vector_nodes_for_insert(vector_candidates)
            persisted_batch = self._docstore_batch_for_persist(
                batch, vector_candidates, prepared_docstore_nodes
            )
            if persisted_batch:
                try:
                    self.index.docstore.add_documents(
                        persisted_batch,
                        allow_update=True,
                    )
                except Exception as exc:
                    logger.error(
                        "failed_persist_nodes | batch={}/{} collection={!r} "
                        "error={!r} node_ids={}",
                        batch_no,
                        len(batches),
                        self.qdrant_collection,
                        exc,
                        [node.node_id for node in persisted_batch],
                    )
                    raise
            if prepared_vector_nodes:
                try:
                    self.index.insert_nodes(prepared_vector_nodes)
                except Exception as exc:
                    logger.error(
                        "orphaned_kv_nodes | batch={}/{} collection={!r} "
                        "error={!r} node_ids={}",
                        batch_no,
                        len(batches),
                        self.qdrant_collection,
                        exc,
                        [node.node_id for node in prepared_vector_nodes],
                    )
                    raise

    def _log_ingest_benchmark_summary(
        self,
        *,
        mode: str,
        started_at: float,
        core_docs: int,
        core_nodes: int,
        streaming_docs: int,
        streaming_nodes: int,
        enrich_batches: int,
        persist_batches: int,
    ) -> None:
        """Log ingest benchmark counters for runtime tuning.

        Args:
            mode (str): Ingest mode label (``sync`` or ``async``).
            started_at (float): Monotonic timestamp when ingestion started.
            core_docs (int): Number of document records emitted by core PDF pipeline.
            core_nodes (int): Number of nodes persisted from core PDF pipeline.
            streaming_docs (int): Number of docs emitted by legacy streaming pipeline.
            streaming_nodes (int): Number of nodes persisted from legacy streaming pipeline.
            enrich_batches (int): Number of streaming enrichment batches processed.
            persist_batches (int): Number of docstore/vector persistence micro-batches.
        """
        elapsed_s = max(0.001, time.monotonic() - started_at)
        total_nodes = core_nodes + streaming_nodes
        total_docs = core_docs + streaming_docs
        nodes_per_second = total_nodes / elapsed_s
        logger.info(
            "Ingest benchmark ({}) | elapsed_s={:.3f} docs={} nodes={} "
            "nodes_per_s={:.2f} core_docs={} core_nodes={} streaming_docs={} "
            "streaming_nodes={} enrich_batches={} persist_batches={} "
            "ingestion_batch_size={} docstore_batch_size={}",
            mode,
            elapsed_s,
            total_docs,
            total_nodes,
            nodes_per_second,
            core_docs,
            core_nodes,
            streaming_docs,
            streaming_nodes,
            enrich_batches,
            persist_batches,
            self.ingestion_config.ingestion_batch_size,
            self.docstore_batch_size,
        )

    async def _apersist_node_batches(self, nodes: list[BaseNode]) -> None:
        """Asynchronously persist nodes in micro-batches.

        Mirrors :meth:`_persist_node_batches` — see its docstring for the
        failure-logging semantics.

        Args:
            nodes (list[BaseNode]): Ingestion nodes to persist.

        Raises:
            RuntimeError: If the index is not initialized.
            Exception: Re-raises the underlying KV or vector-write error
                after emitting a structured log entry.
        """
        if self.index is None:
            raise RuntimeError("Index is not initialized.")

        batches = self._chunk_nodes(nodes, self.docstore_batch_size)
        for batch_no, batch in enumerate(batches, start=1):
            logger.debug(
                "Persisting async node batch {}/{} ({} node(s)) to DocStore...",
                batch_no,
                len(batches),
                len(batch),
            )
            vector_candidates = self._select_vector_nodes(batch)
            (
                prepared_vector_nodes,
                prepared_docstore_nodes,
            ) = await self._aprepare_vector_nodes_for_insert(vector_candidates)
            persisted_batch = self._docstore_batch_for_persist(
                batch, vector_candidates, prepared_docstore_nodes
            )
            if persisted_batch:
                try:
                    self.index.docstore.add_documents(
                        persisted_batch,
                        allow_update=True,
                    )
                except Exception as exc:
                    logger.error(
                        "failed_persist_nodes | batch={}/{} collection={!r} "
                        "error={!r} node_ids={}",
                        batch_no,
                        len(batches),
                        self.qdrant_collection,
                        exc,
                        [node.node_id for node in persisted_batch],
                    )
                    raise
            if prepared_vector_nodes:
                try:
                    await self.index.ainsert_nodes(prepared_vector_nodes)
                except Exception as exc:
                    logger.error(
                        "orphaned_kv_nodes | batch={}/{} collection={!r} "
                        "error={!r} node_ids={}",
                        batch_no,
                        len(batches),
                        self.qdrant_collection,
                        exc,
                        [node.node_id for node in prepared_vector_nodes],
                    )
                    raise

    @staticmethod
    def _extract_file_hash(data: Any) -> str | None:
        """Best-effort extraction of a ``file_hash`` value from nested payloads.

        Args:
            data (Any): The data dictionary to search for a file hash.

        Returns:
            str | None: The extracted file hash, or None if not found.
        """

        if not isinstance(data, dict):
            return None

        candidate = data.get("file_hash")
        if isinstance(candidate, str) and candidate:
            return candidate

        origin = data.get("origin")
        if isinstance(origin, dict):
            candidate = origin.get("file_hash")
            if isinstance(candidate, str) and candidate:
                return candidate

        for key in ("metadata", "meta", "extra_info"):
            nested = data.get(key)
            if isinstance(nested, dict):
                nested_hash = RAG._extract_file_hash(nested)
                if nested_hash:
                    return nested_hash

        for value in data.values():
            if isinstance(value, dict):
                nested_hash = RAG._extract_file_hash(value)
                if nested_hash:
                    return nested_hash
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        nested_hash = RAG._extract_file_hash(item)
                        if nested_hash:
                            return nested_hash
        return None

    @staticmethod
    def _extract_payload_text(payload: dict[str, Any]) -> str:
        """Best-effort extraction of node text from a Qdrant payload.

        Args:
            payload (dict[str, Any]): Raw point payload returned by Qdrant.

        Returns:
            str: Extracted text content, or an empty string if unavailable.
        """
        for key in ("text", "chunk_text", "chunk", "content"):
            candidate = payload.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()

        node_content = payload.get("_node_content")
        node_data: dict[str, Any] | None = None
        if isinstance(node_content, dict):
            node_data = node_content
        elif isinstance(node_content, str) and node_content.strip():
            try:
                parsed = json.loads(node_content)
                if isinstance(parsed, dict):
                    node_data = parsed
            except Exception:
                node_data = None

        if isinstance(node_data, dict):
            for key in ("text", "chunk_text", "chunk", "content"):
                candidate = node_data.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()
            metadata = node_data.get("metadata")
            if isinstance(metadata, dict):
                for key in ("text", "chunk_text", "chunk", "content"):
                    candidate = metadata.get(key)
                    if isinstance(candidate, str) and candidate.strip():
                        return candidate.strip()

        return ""

    @staticmethod
    def _extract_reference_metadata(data: Any) -> dict[str, Any] | None:
        """Best-effort extraction of stable reference metadata from nested payloads.

        Args:
            data (Any): The data dictionary to search for reference metadata.

        Returns:
            dict[str, Any] | None: A dictionary containing the extracted reference metadata fields, or
        """
        if not isinstance(data, dict):
            return None

        candidate = data.get("reference_metadata")
        if isinstance(candidate, dict):
            return {
                field: candidate.get(field) if field in candidate else None
                for field in REFERENCE_METADATA_FIELDS.keys()
            }

        for key in ("origin", "metadata", "meta", "extra_info"):
            nested = data.get(key)
            extracted = RAG._extract_reference_metadata(nested)
            if extracted is not None:
                return extracted

        for value in data.values():
            if isinstance(value, dict):
                extracted = RAG._extract_reference_metadata(value)
                if extracted is not None:
                    return extracted
        return None

    @staticmethod
    def _source_from_payload(
        *,
        collection: str,
        payload: dict[str, Any],
        score: float | None = None,
        text_value: str | None = None,
    ) -> dict[str, Any]:
        """Normalize a raw metadata/payload dictionary into a source dictionary.

        Args:
            collection (str): The Qdrant collection name associated with the payload.
            payload (dict[str, Any]): The raw point payload returned by Qdrant.
            score (float | None): Optional similarity score to include in the source.
            text_value (str | None): Optional pre-extracted text value to use instead of extracting from payload.

        Returns:
            dict[str, Any]: A normalized source dictionary containing standardized fields for downstream processing.
        """
        origin = payload.get("origin") or {}
        filename = (
            origin.get("filename")
            or payload.get("file_name")
            or payload.get("filename")
            or payload.get("file_path")
        )
        filetype = (
            origin.get("filetype")
            or origin.get("mimetype")
            or payload.get("filetype")
            or payload.get("mimetype")
            or payload.get("file_type")
            or payload.get("file_format")
        )
        source_kind = (
            payload.get("source") or payload.get("source_type") or payload.get("reader")
        )
        file_hash = (
            origin.get("file_hash")
            or payload.get("file_hash")
            or RAG._extract_file_hash(payload)
        )

        page = (
            payload.get("page")
            or payload.get("page_number")
            or origin.get("page")
            or origin.get("page_number")
            or origin.get("page_no")
        )
        provenance = payload.get("provenance") or payload.get("provenances") or []
        if page is None and isinstance(provenance, list):
            for prov in provenance:
                if isinstance(prov, dict):
                    page = prov.get("page_no")
                    if page is not None:
                        break

        if page is None:
            doc_items = payload.get("doc_items")
            if isinstance(doc_items, list):
                for item in doc_items:
                    if not isinstance(item, dict):
                        continue
                    provs = item.get("prov")
                    if not isinstance(provs, list):
                        continue
                    for prov_item in provs:
                        if isinstance(prov_item, dict):
                            page = prov_item.get("page_no")
                            if page is not None:
                                break
                    if page is not None:
                        break

        try:
            page = int(page) if page is not None else None
        except Exception:
            page = None

        table_meta = payload.get("table") or {}
        row_index = table_meta.get("row_index")
        try:
            row_index = int(row_index) if row_index is not None else None
        except Exception:
            row_index = None

        resolved_text = (
            text_value if text_value is not None else RAG._extract_payload_text(payload)
        )
        preview_url: str | None = None
        if file_hash:
            preview_url = (
                f"/sources/preview?collection={collection}&file_hash={file_hash}"
            )

        src: dict[str, Any] = {
            "text": resolved_text,
            "preview_text": resolved_text[:280].strip(),
            "filename": filename,
            "filetype": filetype,
            "source": source_kind,
            "score": score,
        }
        entities = payload.get("entities") or origin.get("entities")
        relations = payload.get("relations") or origin.get("relations")
        if entities:
            src["entities"] = entities
        if relations:
            src["relations"] = relations
        if file_hash:
            src["file_hash"] = file_hash
        if preview_url:
            src["preview_url"] = preview_url
            src["document_url"] = preview_url
        if page is not None:
            src["page"] = page
        if row_index is not None:
            src["row"] = row_index
        reference_metadata = RAG._extract_reference_metadata(payload)
        if reference_metadata is not None:
            src["reference_metadata"] = reference_metadata
        if source_kind == "table":
            src["table_info"] = {
                "n_rows": table_meta.get("n_rows"),
                "n_cols": table_meta.get("n_cols"),
                "style": table_meta.get("style"),
            }
        return src

    def get_source_by_node_id(
        self,
        node_id: str,
        *,
        score: float | None = None,
    ) -> dict[str, Any] | None:
        """Resolve a stored node id back into a normalized source payload."""
        payload: dict[str, Any] | None = None
        try:
            index = self.index
            if index is not None:
                docstore = getattr(index, "storage_context", None)
                if docstore is not None:
                    docstore = getattr(docstore, "docstore", None)
                else:
                    docstore = getattr(index, "docstore", None)
                if docstore is not None:
                    for getter in ("get_node", "get", "get_document"):
                        fn = getattr(docstore, getter, None)
                        if not callable(fn):
                            continue
                        try:
                            node = fn(node_id)
                        except Exception:
                            continue
                        if node is None:
                            continue
                        payload = dict(getattr(node, "metadata", {}) or {})
                        text_value = getattr(node, "text", None)
                        if not isinstance(text_value, str) or not text_value.strip():
                            if (
                                isinstance(node, BaseNode)
                                and hasattr(node, "get_content")
                                and callable(node.get_content)
                            ):
                                content = node.get_content()
                                if isinstance(content, str) and content.strip():
                                    text_value = content
                        if isinstance(text_value, str) and text_value.strip():
                            payload.setdefault("text", text_value.strip())
                        payload.setdefault(
                            "node_id",
                            getattr(node, "node_id", None)
                            or getattr(node, "id_", None),
                        )
                        break
        except Exception:
            payload = None

        if payload is None:
            try:
                recs = self.qdrant_client.retrieve(
                    collection_name=self.qdrant_collection, ids=[node_id]
                )
                if recs:
                    candidate = getattr(recs[0], "payload", None)
                    if isinstance(candidate, dict):
                        payload = dict(candidate)
            except Exception:
                payload = None

        if payload is None:
            return None
        return self._source_from_payload(
            collection=self.qdrant_collection,
            payload=payload,
            score=score,
        )

    def _get_existing_file_hashes(self) -> set[str]:
        """Fetch file hashes already stored in the active Qdrant collection.

        Returns:
            set[str]: A set of existing file hashes.
        """

        existing: set[str] = set()

        try:
            _ = self.qdrant_client
        except Exception as exc:
            logger.warning(
                "Unable to initialize Qdrant client for hash lookup: {}", exc
            )
            return existing

        offset: Any = None
        while True:
            try:
                points, offset = self.qdrant_client.scroll(
                    collection_name=self.qdrant_collection,
                    offset=offset,
                    limit=256,
                    with_vectors=False,
                    with_payload=True,
                )
            except Exception as exc:
                # Qdrant may return a 404 when the collection does not exist;
                # treat that case as non-fatal and log at debug level to avoid
                # cluttering logs with expected messages for new collections.
                msg = str(exc)
                not_found = (
                    "Not found" in msg
                    or "doesn't exist" in msg
                    or "does not exist" in msg
                    or f"Collection `{self.qdrant_collection}`" in msg
                )
                if not_found:
                    logger.debug(
                        "Qdrant collection '{}' not found; skipping existing-hash check: {}",
                        self.qdrant_collection,
                        exc,
                    )
                else:
                    logger.warning(
                        "Failed to fetch existing hashes from collection '{}': {}",
                        self.qdrant_collection,
                        exc,
                    )
                break

            if not points:
                break

            for point in points:
                payload = getattr(point, "payload", None)
                if isinstance(payload, dict):
                    file_hash = self._extract_file_hash(payload)
                    if file_hash:
                        existing.add(file_hash)

            if offset is None:
                break

        return existing

    def create_index(self) -> None:
        """Materialize a VectorStoreIndex. If nodes are present in memory, create from nodes.
        Otherwise, load from vector store.
        """
        vector_store = self._vector_store()
        storage_ctx = self._storage_context(vector_store)

        if self.nodes:
            self.index = self._index(storage_ctx)
        else:
            # Build index with explicit storage_context so it uses the persistent docstore.
            self.index = VectorStoreIndex(
                nodes=[],
                embed_model=self.embed_model,
                storage_context=storage_ctx,
            )

    def create_query_engine(self) -> None:
        """Create the query engine with a retriever and reranker.

        Raises:
            RuntimeError: If the index is not initialized.
        """
        self.query_engine = self.build_query_engine()

    def rewrite_retrieval_query(
        self,
        *,
        user_msg: str,
        conversation_context: str = "",
    ) -> str:
        """Rewrite the latest user message into a standalone retrieval query.

        Args:
            user_msg (str): The latest user question.
            conversation_context (str): Compact prior-turn context used only for rewrite.

        Returns:
            str: Standalone retrieval query text.
        """
        if not conversation_context.strip():
            return user_msg.strip()

        prompt = self.rewrite_retrieval_prompt.format(
            conversation_context=conversation_context.strip(),
            user_msg=user_msg.strip(),
        )
        try:
            completion = self.text_model.complete(prompt)
            rewritten = str(getattr(completion, "text", "") or "").strip()
        except Exception as exc:
            logger.warning("Retrieval rewrite failed; using raw user message: {}", exc)
            return user_msg.strip()
        return rewritten or user_msg.strip()

    def _sample_collection_payloads(self, limit: int = 128) -> list[dict[str, Any]]:
        """Fetch a small payload sample from the active collection."""
        if not self.qdrant_collection:
            return []

        offset = None
        payloads: list[dict[str, Any]] = []
        remaining = max(1, int(limit))
        while remaining > 0:
            batch_size = min(remaining, 128)
            try:
                points, offset = self.qdrant_client.scroll(
                    collection_name=self.qdrant_collection,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to sample collection payloads for '{}': {}",
                    self.qdrant_collection,
                    exc,
                )
                break
            if not points:
                break
            for point in points:
                payload = getattr(point, "payload", None)
                if isinstance(payload, dict):
                    payloads.append(payload)
                    remaining -= 1
                    if remaining <= 0:
                        break
            if offset is None:
                break
        return payloads

    @staticmethod
    def _is_social_payload(payload: dict[str, Any]) -> bool:
        """Return whether a payload looks like a row-level social post."""
        if not isinstance(payload, dict):
            return False
        source_kind = str(payload.get("source") or payload.get("source_type") or "")
        if source_kind != "table":
            return False
        reference_metadata = RAG._extract_reference_metadata(payload) or {}
        if not isinstance(reference_metadata, dict):
            return False
        return any(
            str(reference_metadata.get(key) or "").strip()
            for key in ("type", "network", "author", "author_id", "text_id")
        )

    @staticmethod
    def _source_post_key(source: dict[str, Any]) -> str:
        """Build a stable social/post identity key for normalized sources."""
        reference_metadata = source.get("reference_metadata")
        if isinstance(reference_metadata, dict):
            text_id = str(reference_metadata.get("text_id") or "").strip()
            if text_id:
                return f"text_id:{text_id}"

        file_hash = str(source.get("file_hash") or "").strip()
        row_value = source.get("row")
        if file_hash and row_value is not None:
            return f"row:{file_hash}:{row_value}"

        text_value = str(source.get("text") or source.get("preview_text") or "").strip()
        if text_value:
            normalized = re.sub(r"\s+", " ", text_value).lower()
            return f"text:{normalized[:240]}"
        return ""

    @staticmethod
    def _source_diversity_bucket(source: dict[str, Any]) -> str:
        """Return a coarse author/time bucket for social summary diversity."""
        reference_metadata = source.get("reference_metadata")
        ref = reference_metadata if isinstance(reference_metadata, dict) else {}
        author = str(ref.get("author_id") or ref.get("author") or "unknown").strip()
        timestamp_raw = str(ref.get("timestamp") or "").strip()
        time_bucket = "unknown"
        if timestamp_raw:
            try:
                parsed = datetime.fromisoformat(timestamp_raw.replace("Z", "+00:00"))
                time_bucket = parsed.astimezone(timezone.utc).strftime("%Y-%m-%dT%H")
            except ValueError:
                time_bucket = timestamp_raw[:13]
        return f"{author.lower()}::{time_bucket}"

    @staticmethod
    def _coverage_unit_for_sources(sources: list[dict[str, Any]]) -> str:
        """Infer coverage unit from normalized source metadata."""
        for source in sources:
            reference_metadata = source.get("reference_metadata")
            if (
                isinstance(reference_metadata, dict)
                and str(reference_metadata.get("text_id") or "").strip()
            ):
                return "posts"
        return "chunks"

    def _infer_collection_profile(self) -> dict[str, Any]:
        """Infer whether the active collection is social/table heavy."""
        docs = self.list_documents()
        payloads = self._sample_collection_payloads(limit=96)
        social_payloads = [
            payload for payload in payloads if self._is_social_payload(payload)
        ]
        table_docs = [doc for doc in docs if "max_rows" in doc]
        is_social_table = bool(social_payloads) and (
            len(docs) <= 3
            or len(table_docs) == len(docs)
            or len(social_payloads) >= max(3, len(payloads) // 3)
        )
        coverage_unit = "documents"
        if is_social_table:
            coverage_unit = "posts"
            for payload in social_payloads:
                reference_metadata = self._extract_reference_metadata(payload) or {}
                if (
                    not isinstance(reference_metadata, dict)
                    or not str(reference_metadata.get("text_id") or "").strip()
                ):
                    coverage_unit = "chunks"
                    break
        return {
            "is_social_table": is_social_table,
            "coverage_unit": coverage_unit,
        }

    def _resolve_chat_response_mode(self) -> ResponseMode:
        """Resolve the response synthesizer mode for query/chat answers."""
        configured = str(self.chat_response_mode or "auto").strip().lower()
        if configured == "refine":
            return ResponseMode.REFINE
        if configured == "compact":
            return ResponseMode.COMPACT
        profile = self._infer_collection_profile()
        if bool(profile.get("is_social_table")):
            return ResponseMode.REFINE
        return ResponseMode.COMPACT

    def _collection_supports_parent_context(self) -> bool:
        """Return whether the active collection contains hierarchical parent/child nodes.

        Returns:
            bool: True if the collection appears to support parent/child context, else False.
        """
        collection = str(self.qdrant_collection or "").strip()
        if not collection:
            return False
        cached = self._parent_context_support_cache.get(collection)
        if cached is not None:
            return cached

        supported = False
        for payload in self._sample_collection_payloads(limit=96):
            hier_type = str(payload.get("docint_hier_type") or "").strip().lower()
            if hier_type in {"coarse", "fine"}:
                supported = True
                break
            if payload.get("hier.parent_id") or payload.get("hier.level"):
                supported = True
                break

        self._parent_context_support_cache[collection] = supported
        return supported

    @staticmethod
    def _merge_metadata_filters(
        base_filters: MetadataFilters | None,
        extra_filters: list[MetadataFilter],
    ) -> MetadataFilters | None:
        """Merge request-scoped filters with internal retrieval filters.

        Args:
            base_filters (MetadataFilters | None): The original filters provided at the query engine level, or None if no filters were provided.
            extra_filters (list[MetadataFilter]): Additional filters that must be applied for retrieval, such as parent-context scoping.

        Returns:
            MetadataFilters | None: A new MetadataFilters object that combines the base filters and extra filters with an AND condition, or None if there are no filters to apply.
        """
        if not extra_filters:
            return base_filters
        if base_filters is None:
            return MetadataFilters(
                filters=cast(list[MetadataFilter | MetadataFilters], extra_filters),
                condition=FilterCondition.AND,
            )
        return MetadataFilters(
            filters=[*base_filters.filters, *extra_filters],
            condition=FilterCondition.AND,
        )

    def _resolve_vector_store_query_mode(
        self,
        raw_mode: str | None = None,
    ) -> VectorStoreQueryMode:
        """Resolve runtime retrieval mode for the vector index retriever.

        Args:
            raw_mode (str | None): An optional retrieval mode string provided at call time, which takes precedence over config settings. Expected values
                are "auto", "default", "sparse", "hybrid", or "mmr".

        Returns:
            VectorStoreQueryMode: The resolved retrieval mode to use for the current retrieval operation.
        """
        mode_value = (
            str(raw_mode or self.vector_store_query_mode or "auto").strip().lower()
        )
        if mode_value == "auto":
            mode_value = "hybrid" if self.enable_hybrid else "default"

        mode_map = {
            "default": VectorStoreQueryMode.DEFAULT,
            "sparse": VectorStoreQueryMode.SPARSE,
            "hybrid": VectorStoreQueryMode.HYBRID,
            "mmr": VectorStoreQueryMode.MMR,
        }
        resolved = mode_map.get(mode_value, VectorStoreQueryMode.DEFAULT)
        if (
            resolved in {VectorStoreQueryMode.HYBRID, VectorStoreQueryMode.SPARSE}
            and not self.enable_hybrid
        ):
            logger.warning(
                "Retrieval mode '{}' requested without hybrid support; falling back to dense retrieval.",
                mode_value,
            )
            return VectorStoreQueryMode.DEFAULT
        return resolved

    def _resolve_runtime_retrieval_settings(
        self,
        *,
        similarity_top_k: int | None = None,
        retrieval_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Resolve retrieval settings from config plus optional call-site overrides.

        Args:
            similarity_top_k (int | None): An optional override for the number of top similar results to retrieve, which takes precedence over config settings.
            retrieval_options (dict[str, Any] | None): An optional dictionary of runtime retrieval overrides, which may include "vector_store_query_mode", "alpha",
                "sparse_top_k", "hybrid_top_k", and "parent_context_enabled". These options take precedence over config settings and are used to
                dynamically adjust retrieval behavior on a per-query basis.

        Returns:
            dict[str, Any]: A dictionary containing the resolved retrieval settings to apply for the current retrieval operation, including:
                - "similarity_top_k": The effective number of top similar results to retrieve.
                - "vector_store_query_mode": The resolved retrieval mode to use.
                - "alpha": The hybrid fusion alpha value (if applicable).
                - "sparse_top_k": The number of top sparse results to retrieve (if applicable).
                - "hybrid_top_k": The number of top hybrid results to retrieve (if applicable).
                - "parent_context_enabled": Whether parent-context expansion is enabled for this retrieval.
                - "label": A string label summarizing the retrieval mode and parent context status, useful for logging and analytics.
        """
        overrides = retrieval_options or {}
        resolved_mode = self._resolve_vector_store_query_mode(
            cast(str | None, overrides.get("vector_store_query_mode"))
        )
        effective_top_k = similarity_top_k or min(
            max(self.retrieve_similarity_top_k, self.rerank_top_n * 8),
            64,
        )
        alpha = float(overrides.get("alpha", self.hybrid_alpha))
        alpha = min(1.0, max(0.0, alpha))
        sparse_top_k = max(
            1,
            int(overrides.get("sparse_top_k", self.sparse_top_k)),
        )
        hybrid_top_k = max(
            1,
            int(overrides.get("hybrid_top_k", self.hybrid_top_k)),
        )
        parent_context_enabled = (
            bool(overrides.get("parent_context_enabled", self.parent_context_enabled))
            and self._collection_supports_parent_context()
        )

        label = resolved_mode.value
        if parent_context_enabled:
            label = f"{label}_parent"

        return {
            "similarity_top_k": effective_top_k,
            "vector_store_query_mode": resolved_mode,
            "alpha": alpha,
            "sparse_top_k": sparse_top_k,
            "hybrid_top_k": hybrid_top_k,
            "parent_context_enabled": parent_context_enabled,
            "label": label,
        }

    def _build_grounded_text_qa_template(self, *, social_table: bool) -> PromptTemplate:
        """Return the grounded QA prompt template for answer synthesis.

        Args:
            social_table (bool): Whether the active collection appears to be social/table heavy, which may require special
                instructions to preserve post-level distinctions during synthesis.

        Returns:
            PromptTemplate: The constructed prompt template to use for grounded question-answering during response synthesis.
        """
        prompt = self.grounded_text_qa_prompt
        if social_table:
            prompt = (
                f"{prompt.strip()}\n\n"
                "When the context comes from social posts or table rows, keep each "
                "post distinct and avoid merging separate authors or timestamps.\n"
            )
        return PromptTemplate(prompt)

    def _build_grounded_refine_template(self, *, social_table: bool) -> PromptTemplate:
        """Return the grounded refine prompt template for answer synthesis.

        Args:
            social_table (bool): Whether the active collection appears to be social/table heavy, which may require special
                instructions to preserve post-level distinctions during synthesis.

        Returns:
            PromptTemplate: The constructed prompt template to use for grounded question-answering during response synthesis.
        """
        prompt = self.grounded_refine_prompt
        if social_table:
            prompt = (
                f"{prompt.strip()}\n\n"
                "For row-level social evidence, preserve distinctions between "
                "different posts even when they discuss the same topic.\n"
            )
        return PromptTemplate(prompt)

    def _build_retriever(
        self,
        *,
        metadata_filters: MetadataFilters | None = None,
        similarity_top_k: int | None = None,
        vector_store_kwargs: dict[str, Any] | None = None,
        retrieval_options: dict[str, Any] | None = None,
    ) -> Any:
        """Build a retriever, optionally scoped by metadata filters.

        Args:
            metadata_filters (MetadataFilters | None): Optional request-scoped metadata filters.
            similarity_top_k (int | None): Optional override for retrieval depth.
            vector_store_kwargs (dict[str, Any] | None): Optional native vector-store query kwargs.
            retrieval_options (dict[str, Any] | None): Optional runtime overrides for retrieval mode,
                hybrid fusion, and parent-context expansion.
        """
        if self.index is None:
            logger.error("RuntimeError: Index is not initialized.")
            raise RuntimeError("Index is not initialized. Cannot create retriever.")

        retrieval_settings = self._resolve_runtime_retrieval_settings(
            similarity_top_k=similarity_top_k,
            retrieval_options=retrieval_options,
        )
        internal_filters: list[MetadataFilter] = []
        if retrieval_settings["parent_context_enabled"]:
            internal_filters.append(
                MetadataFilter(
                    key="docint_hier_type",
                    value="fine",
                    operator=FilterOperator.EQ,
                )
            )

        merged_filters = self._merge_metadata_filters(
            metadata_filters, internal_filters
        )

        retriever_kwargs: dict[str, Any] = {
            "similarity_top_k": retrieval_settings["similarity_top_k"],
            "vector_store_query_mode": retrieval_settings["vector_store_query_mode"],
        }
        if merged_filters is not None:
            retriever_kwargs["filters"] = merged_filters
        if retrieval_settings["vector_store_query_mode"] == VectorStoreQueryMode.HYBRID:
            retriever_kwargs["alpha"] = retrieval_settings["alpha"]
            retriever_kwargs["sparse_top_k"] = retrieval_settings["sparse_top_k"]
            retriever_kwargs["hybrid_top_k"] = retrieval_settings["hybrid_top_k"]
        elif (
            retrieval_settings["vector_store_query_mode"] == VectorStoreQueryMode.SPARSE
        ):
            retriever_kwargs["sparse_top_k"] = retrieval_settings["sparse_top_k"]
        if vector_store_kwargs:
            retriever_kwargs["vector_store_kwargs"] = vector_store_kwargs
        return self.index.as_retriever(**retriever_kwargs)

    def build_query_engine(
        self,
        *,
        metadata_filters: MetadataFilters | None = None,
        streaming: bool = False,
        vector_store_kwargs: dict[str, Any] | None = None,
        retrieval_options: dict[str, Any] | None = None,
    ) -> RetrieverQueryEngine:
        """Construct a query engine for the current index.

        Args:
            metadata_filters (MetadataFilters | None): Optional request-scoped metadata filters.
            streaming (bool): Whether the query engine should stream token output.
            vector_store_kwargs (dict[str, Any] | None): Optional native vector-store query kwargs.
            retrieval_options (dict[str, Any] | None): Optional runtime overrides for retrieval mode,
                hybrid fusion, and parent-context expansion.
        """
        if self.index is None:
            self.create_index()
        if self.index is None:
            logger.error("RuntimeError: Index is not initialized.")
            raise RuntimeError("Index is not initialized. Cannot create query engine.")

        profile = self._infer_collection_profile()
        retrieval_settings = self._resolve_runtime_retrieval_settings(
            retrieval_options=retrieval_options,
        )
        response_mode = self._resolve_chat_response_mode()
        node_postprocessors: list[BaseNodePostprocessor] = [
            LazyRerankerPostprocessor(rag=self)
        ]
        if retrieval_settings["parent_context_enabled"] and self.index is not None:
            node_postprocessors.append(
                ParentContextPostprocessor(docstore=self.index.docstore)
            )
        if bool(profile.get("is_social_table")):
            node_postprocessors.append(
                SocialSourceDiversityPostprocessor(
                    diversity_limit=max(1, int(self.social_summary_diversity_limit))
                )
            )

        return RetrieverQueryEngine.from_args(
            retriever=self._build_retriever(
                metadata_filters=metadata_filters,
                vector_store_kwargs=vector_store_kwargs,
                retrieval_options=retrieval_options,
            ),
            llm=self.post_retrieval_text_model,
            node_postprocessors=node_postprocessors,
            streaming=streaming,
            response_mode=response_mode,
            text_qa_template=self._build_grounded_text_qa_template(
                social_table=bool(profile.get("is_social_table"))
            ),
            refine_template=self._build_grounded_refine_template(
                social_table=bool(profile.get("is_social_table"))
            ),
        )

    def _source_from_node_with_score(self, nws: Any) -> dict[str, Any] | None:
        """Normalize one ``NodeWithScore`` item into a source dictionary.

        Args:
            nws (Any): A ``NodeWithScore``-like object.

        Returns:
            dict[str, Any] | None: Normalized source payload, or ``None``.
        """
        node = getattr(nws, "node", None)
        if node is None:
            return None

        text_value = getattr(node, "text", "") or ""
        meta = getattr(node, "metadata", {}) or {}
        return self._source_from_payload(
            collection=self.qdrant_collection,
            payload=meta,
            score=getattr(nws, "score", None),
            text_value=text_value,
        )

    @staticmethod
    def _source_backed_fallback_response(sources: Sequence[dict[str, Any]]) -> str:
        """Build a concise grounded fallback response from normalized sources.

        Args:
            sources (Sequence[dict[str, Any]]): Retrieved source payloads.

        Returns:
            str: A concise description of the matched sources.
        """
        if not sources:
            return EMPTY_RESPONSE_FALLBACK

        formatted_sources: list[str] = []
        seen: set[tuple[Any, ...]] = set()
        for source in sources:
            dedupe_key = (
                source.get("filename"),
                source.get("page"),
                source.get("row"),
                source.get("file_hash"),
                source.get("preview_text"),
                source.get("text"),
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            label = (
                source.get("filename")
                or source.get("file_hash")
                or source.get("source")
                or "source"
            )
            location_parts: list[str] = []
            page = source.get("page")
            row = source.get("row")
            if page is not None:
                location_parts.append(f"page {page}")
            if row is not None:
                location_parts.append(f"row {row}")
            if location_parts:
                label = f"{label} ({', '.join(location_parts)})"
            formatted_sources.append(str(label))

        total_sources = len(formatted_sources)
        if total_sources == 0:
            return EMPTY_RESPONSE_FALLBACK

        preview = formatted_sources[:3]
        summary = ", ".join(preview)
        if total_sources > len(preview):
            summary = f"{summary}, and {total_sources - len(preview)} more"
        return f"I found {total_sources} matching sources: {summary}."

    def _normalize_response_data(
        self,
        query: str,
        result: Any,
        reason: str | None = None,
        *,
        metadata_filters_active: bool = False,
        metadata_filter_rules: Sequence[Any] | None = None,
        retrieval_query: str | None = None,
        coverage_unit: str | None = None,
        retrieval_mode: str | None = None,
    ) -> dict[str, Any]:
        """Normalize both llama_index.core.Response and AgentChatResponse into a single payload.
        Handles:
        - response text (result.response or result.text)
        - source_nodes (list[NodeWithScore])
        - metadata differences

        Args:
            query (str): The original query string.
            result (Any): The response object from the query engine.
            reason (str | None): Optional reasoning string.
            metadata_filters_active (bool): Whether request-scoped metadata
                filters were active for the retrieval.
            metadata_filter_rules (Sequence[Any] | None): Optional raw request
                filter payloads for post-filtering auxiliary image sources.

        Returns:
            dict[str, Any]: A dictionary containing:
            - 'query': The original query string.
            - 'reasoning': The reasoning behind the response, if available.
            - 'response': The normalized response text.
            - 'sources': A list of source metadata dictionaries, each containing:
                - 'text': The text content of the source.
                - 'filename': The name of the file where the source was found.
                - 'filetype': The type of the file (e.g., PDF, CSV).
                - 'source': The source kind (e.g., "table" for TableReader).
                - 'page': Optional page number if the source is a PDF.
                - 'row': Optional row index if the source is a table.
            - 'table_info': Optional dictionary with 'n_rows' and 'n_cols' for table sources.
        """
        # --- normalize response text ---
        resp_text = None
        if hasattr(result, "response") and isinstance(result.response, str):
            resp_text = result.response
        elif hasattr(result, "text") and isinstance(result.text, str):
            resp_text = result.text
        elif hasattr(result, "message") and hasattr(result.message, "content"):
            resp_text = str(result.message.content)
        else:
            resp_text = ""

        # strip <think>…</think> (optional)
        m = re.search(
            r"<think>(.*?)</think>", resp_text, flags=re.DOTALL | re.IGNORECASE
        )
        if m:
            reason = (m.group(1).strip() if m else None) or reason
            resp_text = re.sub(
                r"<think>.*?</think>", "", resp_text, flags=re.DOTALL | re.IGNORECASE
            ).strip()

        # --- normalize source_nodes ---
        source_nodes = getattr(result, "source_nodes", None)
        if source_nodes is None and hasattr(result, "metadata"):
            # some Response variants tuck nodes under metadata
            meta = getattr(result, "metadata", {}) or {}
            source_nodes = meta.get("source_nodes")
        if not isinstance(source_nodes, list):
            source_nodes = []

        sources: list[dict[str, Any]] = []
        for nws in source_nodes:
            normalized = self._source_from_node_with_score(nws)
            if normalized is not None:
                sources.append(normalized)

        image_filter_rules = metadata_filter_rules if metadata_filters_active else None
        if (
            query.strip()
            and query != self.summarize_prompt
            and (not metadata_filters_active or bool(image_filter_rules))
        ):
            sources.extend(
                self._retrieve_image_sources(
                    query,
                    top_k=max(1, self.rerank_top_n // 2),
                    metadata_filter_rules=image_filter_rules,
                )
            )

        normalized_resp_text = str(resp_text or "").strip()
        if normalized_resp_text.lower() in {"empty response", "no response"}:
            resp_text = ""
            normalized_resp_text = ""

        if normalized_resp_text == EMPTY_RESPONSE_FALLBACK and sources:
            resp_text = self._source_backed_fallback_response(sources)
            normalized_resp_text = str(resp_text).strip()

        if not normalized_resp_text and sources:
            resp_text = self._source_backed_fallback_response(sources)
            normalized_resp_text = str(resp_text).strip()

        if not normalized_resp_text:
            resp_text = EMPTY_RESPONSE_FALLBACK

        return {
            "query": query,
            "reasoning": reason,
            "response": resp_text,
            "sources": sources,
            "retrieval_query": retrieval_query,
            "coverage_unit": coverage_unit,
            "retrieval_mode": retrieval_mode,
        }

    def _load_collection_ner_sources(
        self,
        *,
        qdrant_filter: qdrant_models.Filter | None = None,
    ) -> list[dict[str, Any]]:
        """Load NER-bearing source rows from Qdrant.

        Args:
            qdrant_filter (qdrant_models.Filter | None): Optional native Qdrant filter applied during scroll.

        Returns:
            list[dict[str, Any]]: Normalized NER source rows.
        """
        if not self.qdrant_collection:
            return []

        sources: list[dict[str, Any]] = []
        offset = None
        while True:
            try:
                points, offset = self.qdrant_client.scroll(
                    collection_name=self.qdrant_collection,
                    limit=100,
                    offset=offset,
                    scroll_filter=qdrant_filter,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as exc:
                logger.error(
                    "Failed to scroll collection '{}' for NER sources: {}",
                    self.qdrant_collection,
                    exc,
                )
                break

            if not points:
                break

            for point in points:
                payload = getattr(point, "payload", None)
                if not isinstance(payload, dict):
                    continue
                if "entities" not in payload and "relations" not in payload:
                    continue

                source = self._source_from_payload(
                    collection=self.qdrant_collection,
                    payload=payload,
                )
                source["chunk_id"] = str(
                    payload.get("node_id")
                    or payload.get("id_")
                    or str(getattr(point, "id", "") or "")
                )
                source["chunk_text"] = str(source.get("text") or "")
                sources.append(source)

            if offset is None:
                break

        return sources

    @staticmethod
    def _dedupe_source_key(source: dict[str, Any]) -> str:
        """Build a stable key for source-level deduplication.

        Args:
            source (dict[str, Any]): A normalized source dictionary.

        Returns:
            str: A string key that can be used to identify duplicate sources.
        """
        reference_metadata = source.get("reference_metadata") or {}
        if isinstance(reference_metadata, dict):
            text_id = str(reference_metadata.get("text_id") or "").strip()
            if text_id:
                return f"text_id:{text_id}"

        chunk_id = str(source.get("chunk_id") or "").strip()
        if chunk_id:
            return f"chunk:{chunk_id}"

        file_hash = str(source.get("file_hash") or "").strip()
        page = source.get("page")
        row = source.get("row")
        if file_hash and (page is not None or row is not None):
            return f"file:{file_hash}:page={page}:row={row}"

        filename = str(source.get("filename") or "").strip().lower()
        preview = str(source.get("chunk_text") or source.get("text") or "").strip()
        return f"fallback:{filename}:{preview[:160].lower()}"

    @staticmethod
    def _collect_entity_matches(
        aggregate: dict[str, Any],
        *,
        query: str,
    ) -> list[dict[str, Any]]:
        """Collect sorted entity matches for an occurrence lookup query.

        Args:
            aggregate (dict[str, Any]): Collection-wide NER aggregate payload.
            query (str): Raw user query string.

        Returns:
            list[dict[str, Any]]: Candidate entity matches sorted by rank and mentions.
        """
        entity_matches: list[dict[str, Any]] = []
        for entity in list(aggregate.get("entities") or []):
            entity_text = str(entity.get("text") or "").strip()
            if not entity_text:
                continue
            match = match_entity_text(entity_text, query)
            if match is None:
                continue
            entity_matches.append(
                {
                    "key": str(entity.get("key") or ""),
                    "text": entity_text,
                    "type": str(entity.get("type") or "Unlabeled"),
                    "mentions": int(entity.get("mentions", 0) or 0),
                    "source_count": int(entity.get("source_count", 0) or 0),
                    "best_score": entity.get("best_score"),
                    "variant_count": int(entity.get("variant_count", 0) or 0),
                    "variants": list(entity.get("variants") or []),
                    "match_rank": int(match[0]),
                    "match_alias": str(match[1]),
                }
            )

        entity_matches.sort(
            key=lambda row: (
                int(row["match_rank"]),
                -int(row["mentions"]),
                str(row["text"]).lower(),
                str(row["type"]).lower(),
            )
        )
        return entity_matches

    @staticmethod
    def _strong_entity_matches(
        entity_matches: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Return all strong top-rank entity matches for a query.

        The strongest ambiguity cases are same-surface-form collisions such as the
        same label appearing under different entity types. Narrowing to the top
        alias avoids treating substring matches like ``Migration`` vs
        ``Remigration`` as equally strong.
        """
        if not entity_matches:
            return []
        best_rank = int(entity_matches[0]["match_rank"])
        top_alias = str(entity_matches[0].get("match_alias") or "").strip().lower()
        strong_matches = [
            row
            for row in entity_matches
            if int(row["match_rank"]) == best_rank
            and str(row.get("match_alias") or "").strip().lower() == top_alias
        ]
        return strong_matches or [
            row for row in entity_matches if int(row["match_rank"]) == best_rank
        ]

    @staticmethod
    def _entity_candidate_payload(entity_match: dict[str, Any]) -> dict[str, Any]:
        """Normalize an entity match row for API/UI disambiguation payloads."""
        return {
            "key": str(entity_match.get("key") or ""),
            "text": str(entity_match.get("text") or ""),
            "type": str(entity_match.get("type") or "Unlabeled"),
            "mentions": int(entity_match.get("mentions", 0) or 0),
            "source_count": int(entity_match.get("source_count", 0) or 0),
            "best_score": entity_match.get("best_score"),
            "variant_count": int(entity_match.get("variant_count", 0) or 0),
            "variants": list(entity_match.get("variants") or []),
            "match_rank": int(entity_match.get("match_rank", 99) or 99),
            "match_alias": str(entity_match.get("match_alias") or ""),
        }

    def _build_entity_occurrence_group(
        self,
        *,
        sources: list[dict[str, Any]],
        matched_entity: dict[str, Any],
        limit: int,
        entity_merge_mode: EntityMergeMode,
    ) -> dict[str, Any]:
        """Build grouped occurrence results for one matched entity.

        Args:
            sources (list[dict[str, Any]]): Candidate NER-bearing source rows.
            matched_entity (dict[str, Any]): Selected entity match metadata.
            limit (int): Maximum number of source rows retained for the entity.

        Returns:
            dict[str, Any]: Group payload containing entity metadata and sources.
        """
        matched_key = str(matched_entity.get("key") or "")
        matched_text = str(matched_entity["text"])
        matched_type = str(matched_entity["type"])
        occurrence_sources: list[dict[str, Any]] = []
        seen_keys: set[str] = set()
        merge_mode = normalize_entity_merge_mode(entity_merge_mode)

        for source in sources:
            mention_rows: list[dict[str, Any]] = []
            for entity in normalize_entities(source.get("entities")):
                entity_text = str(entity.get("text") or "").strip()
                entity_type = str(entity.get("type") or "Unlabeled")
                if entity_type.lower() != matched_type.lower():
                    continue
                source_key = entity_cluster_key(
                    entity_text,
                    entity_type,
                    entity_merge_mode=merge_mode,
                )
                if matched_key and source_key != matched_key:
                    continue
                if not matched_key and entity_text.lower() != matched_text.lower():
                    continue
                mention_rows.append(
                    {
                        "text": entity_text,
                        "type": entity_type,
                        "score": entity.get("score"),
                    }
                )

            if not mention_rows:
                continue

            source_row = dict(source)
            source_row["matched_entity"] = {
                "key": matched_key,
                "text": matched_text,
                "type": matched_type,
                "variant_count": int(matched_entity.get("variant_count", 0) or 0),
                "variants": list(matched_entity.get("variants") or []),
                "match_alias": str(matched_entity.get("match_alias") or ""),
            }
            source_row["matched_mentions"] = mention_rows
            source_row["occurrence_count"] = len(mention_rows)

            dedupe_key = self._dedupe_source_key(source_row)
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            occurrence_sources.append(source_row)

        occurrence_sources.sort(
            key=lambda source: (
                str(source.get("filename") or "").lower(),
                int(source.get("page") or 0),
                int(source.get("row") or 0),
                str(source.get("chunk_id") or "").lower(),
            )
        )

        limited_sources = occurrence_sources[: max(1, int(limit))]
        return {
            "entity": self._entity_candidate_payload(matched_entity),
            "sources": limited_sources,
            "chunk_count": len(occurrence_sources),
            "document_count": len(
                {
                    str(source.get("file_hash") or source.get("filename") or "")
                    for source in occurrence_sources
                    if str(source.get("file_hash") or source.get("filename") or "")
                }
            ),
            "truncated": len(occurrence_sources) > len(limited_sources),
        }

    @staticmethod
    def _flatten_occurrence_groups(
        groups: list[dict[str, Any]],
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Flatten grouped occurrence results into one deduplicated source list."""
        flattened: list[dict[str, Any]] = []
        seen_keys: set[str] = set()
        for group in groups:
            for source in list(group.get("sources") or []):
                if not isinstance(source, dict):
                    continue
                dedupe_key = RAG._dedupe_source_key(source)
                if dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)
                flattened.append(source)
                if len(flattened) >= max(1, int(limit)):
                    return flattened
        return flattened

    def run_entity_occurrence_query(
        self,
        prompt: str,
        *,
        qdrant_filter: qdrant_models.Filter | None = None,
        limit: int = 100,
        refresh: bool = False,
        entity_merge_mode: EntityMergeMode = "orthographic",
    ) -> dict[str, Any]:
        """Return mention-level source rows for the best matching entity.

        Args:
            prompt (str): Raw user query used to identify the target entity.
            qdrant_filter (qdrant_models.Filter | None): Optional native Qdrant filter to constrain candidate rows.
            limit (int): Maximum number of occurrence rows to return.
            refresh (bool): Whether to bypass cached NER rows when no native filter is used.

        Returns:
            dict[str, Any]: A response payload aligned with the normal query path.
        """
        query = str(prompt or "").strip()
        if not query:
            logger.error("ValueError: Query prompt cannot be empty.")
            raise ValueError("Query prompt cannot be empty.")

        sources = (
            self._load_collection_ner_sources(qdrant_filter=qdrant_filter)
            if qdrant_filter is not None
            else self.get_collection_ner(refresh=refresh)
        )
        merge_mode = normalize_entity_merge_mode(entity_merge_mode)
        aggregate = aggregate_ner_sources(sources, entity_merge_mode=merge_mode)
        entity_matches = self._collect_entity_matches(aggregate, query=query)

        if not entity_matches:
            return {
                "query": query,
                "reasoning": None,
                "response": (
                    f"I couldn't find a named-entity match for '{query}' in the "
                    "active collection."
                ),
                "sources": [],
                "retrieval_query": query,
                "coverage_unit": "entity_mentions",
                "retrieval_mode": "entity_occurrence",
                "vector_query_mode": "entity_occurrence",
                "retrieval_profile": "entity_occurrence",
                "parent_context_enabled": False,
            }

        strong_matches = self._strong_entity_matches(entity_matches)
        if len(strong_matches) > 1:
            return {
                "query": query,
                "reasoning": None,
                "response": (
                    f"Your query matches multiple entities equally well. Choose one "
                    f"candidate below or switch to multi-entity occurrence mode to see "
                    f"all strong matches for '{query}'."
                ),
                "sources": [],
                "retrieval_query": query,
                "coverage_unit": "entity_mentions",
                "retrieval_mode": "entity_occurrence_ambiguous",
                "vector_query_mode": "entity_occurrence",
                "retrieval_profile": "entity_occurrence_ambiguous",
                "parent_context_enabled": False,
                "entity_match_candidates": [
                    self._entity_candidate_payload(match) for match in strong_matches
                ],
                "entity_match_groups": [],
            }

        matched_entity = strong_matches[0]
        group = self._build_entity_occurrence_group(
            sources=sources,
            matched_entity=matched_entity,
            limit=limit,
            entity_merge_mode=merge_mode,
        )

        response_text = (
            f"Found {matched_entity['mentions']} occurrence(s) of '{matched_entity['text']}' "
            f"across {int(group['chunk_count'])} chunk(s) in {int(group['document_count'])} "
            "document(s)."
        )
        if bool(group.get("truncated")):
            response_text += (
                f" Showing the first {len(list(group.get('sources') or []))} chunk(s); refine with "
                "metadata filters to narrow the result set."
            )

        return {
            "query": query,
            "reasoning": None,
            "response": response_text,
            "sources": list(group.get("sources") or []),
            "retrieval_query": query,
            "coverage_unit": "entity_mentions",
            "retrieval_mode": "entity_occurrence",
            "vector_query_mode": "entity_occurrence",
            "retrieval_profile": "entity_occurrence",
            "parent_context_enabled": False,
            "entity_match_candidates": [self._entity_candidate_payload(matched_entity)],
            "entity_match_groups": [group],
        }

    def run_multi_entity_occurrence_query(
        self,
        prompt: str,
        *,
        qdrant_filter: qdrant_models.Filter | None = None,
        limit: int = 100,
        refresh: bool = False,
        entity_merge_mode: EntityMergeMode = "orthographic",
    ) -> dict[str, Any]:
        """Return grouped occurrence results for all strong entity matches.

        Args:
            prompt (str): Raw user query used to identify the target entities.
            qdrant_filter (qdrant_models.Filter | None): Optional native Qdrant filter to constrain candidate rows.
            limit (int): Maximum number of source rows retained across all groups.
            refresh (bool): Whether to bypass cached NER rows when no native filter is used.

        Returns:
            dict[str, Any]: Grouped occurrence payload.
        """
        query = str(prompt or "").strip()
        if not query:
            logger.error("ValueError: Query prompt cannot be empty.")
            raise ValueError("Query prompt cannot be empty.")

        sources = (
            self._load_collection_ner_sources(qdrant_filter=qdrant_filter)
            if qdrant_filter is not None
            else self.get_collection_ner(refresh=refresh)
        )
        merge_mode = normalize_entity_merge_mode(entity_merge_mode)
        aggregate = aggregate_ner_sources(sources, entity_merge_mode=merge_mode)
        entity_matches = self._collect_entity_matches(aggregate, query=query)

        if not entity_matches:
            return {
                "query": query,
                "reasoning": None,
                "response": (
                    f"I couldn't find a named-entity match for '{query}' in the "
                    "active collection."
                ),
                "sources": [],
                "retrieval_query": query,
                "coverage_unit": "entity_mentions",
                "retrieval_mode": "entity_occurrence_multi",
                "vector_query_mode": "entity_occurrence_multi",
                "retrieval_profile": "entity_occurrence_multi",
                "parent_context_enabled": False,
                "entity_match_candidates": [],
                "entity_match_groups": [],
            }

        strong_matches = self._strong_entity_matches(entity_matches)
        per_group_limit = max(1, int(limit))
        groups = [
            self._build_entity_occurrence_group(
                sources=sources,
                matched_entity=match,
                limit=per_group_limit,
                entity_merge_mode=merge_mode,
            )
            for match in strong_matches
        ]
        groups = [group for group in groups if list(group.get("sources") or [])]

        flattened_sources = self._flatten_occurrence_groups(
            groups, limit=per_group_limit
        )
        total_chunks = sum(int(group.get("chunk_count", 0) or 0) for group in groups)
        total_documents = len(
            {
                str(source.get("file_hash") or source.get("filename") or "")
                for source in flattened_sources
                if str(source.get("file_hash") or source.get("filename") or "")
            }
        )
        response_text = (
            f"Found {len(groups)} equally strong entity match(es) for '{query}', "
            f"covering {total_chunks} chunk(s) across {total_documents} document(s)."
        )

        return {
            "query": query,
            "reasoning": None,
            "response": response_text,
            "sources": flattened_sources,
            "retrieval_query": query,
            "coverage_unit": "entity_mentions",
            "retrieval_mode": "entity_occurrence_multi",
            "vector_query_mode": "entity_occurrence_multi",
            "retrieval_profile": "entity_occurrence_multi",
            "parent_context_enabled": False,
            "entity_match_candidates": [
                self._entity_candidate_payload(match) for match in strong_matches
            ],
            "entity_match_groups": groups,
        }

    # --- Collection discovery / selection ---
    def list_collections(self) -> list[str]:
        """Return user-selectable collection names via the Qdrant API.

        Collections whose names end with any suffix in
        :data:`HIDDEN_COLLECTION_SUFFIXES` are auxiliary / internal
        (e.g. ``_images`` image-embedding companions, ``_dockv``
        docstore side-effects) and are excluded so they never surface
        in the UI selector or pass :meth:`select_collection` validation.

        Returns:
            list[str]: Sorted list of user-selectable collection names.
        """
        try:
            resp = self.qdrant_client.get_collections()
            names = [
                c.name
                for c in getattr(resp, "collections", []) or []
                if not c.name.endswith(HIDDEN_COLLECTION_SUFFIXES)
            ]
            if names:
                return sorted(names)
            return []
        except Exception as e:
            logger.warning(
                "Qdrant API list_collections failed: {}",
                e,
            )
            raise e

    def delete_collection(self, name: str) -> None:
        """Delete a collection by name from Qdrant and clean up source files.

        The primary Qdrant collection is deleted first.  If that delete
        fails, the method raises immediately — the SQLite KV file
        (nested under ``{qdrant_src_dir}/{name}/``) and the source
        directory are **not** touched, so the caller can diagnose and
        retry without losing ground truth.  Failures deleting the
        supplementary ``{name}_images`` collection are logged and
        swallowed because they are not load-bearing.

        Args:
            name: Name of the collection to delete.

        Raises:
            ValueError: If the name is empty.
            Exception: If the primary Qdrant collection delete fails.
        """
        if not name or not name.strip():
            raise ValueError("Collection name cannot be empty")
        target = name.strip()
        self._invalidate_ner_cache(target)
        self._bump_summary_revision(target, allow_create=False)

        # The primary collection is the only one whose failure is fatal.
        # `{target}_images` is supplementary metadata whose absence is tolerated.
        secondary_collections: list[str] = []
        if not target.endswith("_images"):
            secondary_collections.append(f"{target}_images")

        # 1. Delete the primary Qdrant collection — fail-fast on error so
        #    we don't proceed to destroy the SQLite KV file / source dir.
        try:
            self.qdrant_client.delete_collection(target)
            logger.info("Deleted collection '{}' from Qdrant.", target)
        except Exception:
            logger.error(
                "Failed to delete primary Qdrant collection '{}'; aborting "
                "delete_collection to preserve KV store and source files.",
                target,
            )
            raise

        # 1b. Best-effort delete of supplementary collections.
        for collection_name in secondary_collections:
            try:
                self.qdrant_client.delete_collection(collection_name)
                logger.info("Deleted collection '{}' from Qdrant.", collection_name)
            except Exception as e:
                logger.warning(
                    "Failed to delete supplementary collection '{}': {}",
                    collection_name,
                    e,
                )

        # 2. Cleanup source files (this also removes the nested SQLite KV db).
        for collection_name in [target, *secondary_collections]:
            try:
                src_path = self.qdrant_src_dir / collection_name
                if src_path.exists():

                    def on_error(func: Callable, path: str, _exc_info: Any) -> None:
                        """Error handler for shutil.rmtree.

                        Attempts to fix permissions/flags and retry operation.

                        Args:
                            func (Callable): The function that raised the exception.
                            path (str): The path name passed to function.
                            _exc_info (Any): The exception information returned by sys.exc_info().
                        """
                        try:
                            # 1. Try adding write permission
                            os.chmod(path, stat.S_IWUSR | stat.S_IREAD)

                            # 2. Try clearing flags (macOS/BSD specific)
                            if sys.platform == "darwin":
                                try:
                                    # Clear all file flags (uchg, etc.)
                                    os.chflags(path, 0)
                                except (AttributeError, OSError):
                                    pass

                            # 3. Retry the failed operation
                            func(path)
                        except Exception as e:
                            logger.warning("Failed to force delete {}: {}", path, e)

                    shutil.rmtree(path=src_path, onerror=on_error)
                    logger.info(
                        "Deleted source directory for collection '{}'.",
                        collection_name,
                    )
            except Exception as e:
                logger.error(
                    "Failed to delete source directory for collection '{}': {}",
                    collection_name,
                    e,
                )

    def verify_collection(
        self,
        collection: str | None = None,
        *,
        repair: bool = False,
    ) -> dict[str, Any]:
        """Report cross-store consistency between Qdrant and the KV docstore.

        Scans the Qdrant vector collection for node IDs (point IDs are the
        LlamaIndex node IDs) and the SQLite KV docstore for persisted nodes,
        then categorises any drift:

        * ``kv_orphans``: non-coarse nodes present in the KV store but
          missing from Qdrant — unintended drift from a crashed ingestion
          or an external Qdrant wipe.
        * ``qdrant_orphans``: points present in Qdrant but missing from
          the KV store — retrieval will fail to hydrate these nodes.
        * ``expected_coarse_only``: coarse hierarchical parents correctly
          absent from Qdrant (informational only, not drift).
        * ``missing_parent_ids``: ``hier.parent_id`` values referenced by
          fine nodes that do not resolve in the KV store — broken
          hierarchical retrieval.

        Args:
            collection: Collection name (defaults to the active one).
            repair: When ``True``, delete every id in ``kv_orphans`` from
                the KV docstore.  ``qdrant_orphans`` and
                ``missing_parent_ids`` are left untouched — repairing
                them requires re-ingestion.

        Returns:
            A dict with keys ``collection``, ``qdrant_count``,
            ``kv_count``, ``kv_orphans``, ``qdrant_orphans``,
            ``expected_coarse_only``, ``missing_parent_ids`` and
            ``repaired_ids``.

        Raises:
            ValueError: If no collection is specified and none is active.
        """
        target = str(collection or self.qdrant_collection or "").strip()
        if not target:
            raise ValueError("No collection specified and none active.")

        # 1. Scan Qdrant for node IDs (point IDs are the LI node IDs).
        qdrant_ids: set[str] = set()
        if qdrant_collection_exists(self.qdrant_client, target):
            offset: Any = None
            while True:
                try:
                    points, offset = self.qdrant_client.scroll(
                        collection_name=target,
                        offset=offset,
                        limit=256,
                        with_vectors=False,
                        with_payload=False,
                    )
                except Exception as exc:
                    logger.warning(
                        "verify_collection: Qdrant scroll failed for '{}': {}",
                        target,
                        exc,
                    )
                    break
                for point in points:
                    pid = str(getattr(point, "id", "") or "")
                    if pid:
                        qdrant_ids.add(pid)
                if offset is None:
                    break
        else:
            logger.warning(
                "verify_collection: Qdrant collection '{}' does not exist.",
                target,
            )

        # 2. Scan the KV docstore for node IDs.  Skip if the SQLite file
        #    does not exist — constructing the store would create an
        #    empty one, which would mask the real drift.
        db_path = self.qdrant_src_dir / target / f"{target}_kv.db"
        kv_docs: dict[str, Any] = {}
        doc_store: KVDocumentStore | None = None
        if db_path.exists():
            kv_store = self._build_kv_store(collection=target)
            doc_store = KVDocumentStore(
                kvstore=kv_store, batch_size=self.docstore_batch_size
            )
            kv_docs = doc_store.docs
        else:
            logger.warning(
                "verify_collection: KV store file '{}' does not exist.",
                db_path,
            )

        kv_ids: set[str] = set(kv_docs.keys())

        # 3. Partition drift by whether each KV node is coarse.
        kv_orphans: list[str] = []
        expected_coarse_only: list[str] = []
        for node_id in sorted(kv_ids - qdrant_ids):
            node = kv_docs[node_id]
            meta = getattr(node, "metadata", {}) or {}
            if meta.get("docint_hier_type") == "coarse":
                expected_coarse_only.append(node_id)
            else:
                kv_orphans.append(node_id)

        qdrant_orphans = sorted(qdrant_ids - kv_ids)

        # 4. Walk fine nodes and flag any hier.parent_id that does not
        #    resolve in the KV store.
        missing_parents: set[str] = set()
        for node in kv_docs.values():
            meta = getattr(node, "metadata", {}) or {}
            parent_id = meta.get("hier.parent_id")
            if parent_id and parent_id not in kv_ids:
                missing_parents.add(str(parent_id))

        # 5. Optionally delete kv_orphans.
        repaired: list[str] = []
        if repair and kv_orphans and doc_store is not None:
            for node_id in kv_orphans:
                try:
                    doc_store.delete_document(node_id, raise_error=False)
                    repaired.append(node_id)
                except Exception as exc:
                    logger.warning(
                        "verify_collection: failed to repair orphan '{}': {}",
                        node_id,
                        exc,
                    )

        return {
            "collection": target,
            "qdrant_count": len(qdrant_ids),
            "kv_count": len(kv_ids),
            "kv_orphans": kv_orphans,
            "qdrant_orphans": qdrant_orphans,
            "expected_coarse_only": expected_coarse_only,
            "missing_parent_ids": sorted(missing_parents),
            "repaired_ids": repaired,
        }

    def select_collection(self, name: str) -> None:
        """Switch active collection, ensuring it already exists.

        Args:
            name (str): Name of the collection to select.

        Raises:
            ValueError: If the name is empty or the collection does not exist.
        """
        if not name or not name.strip():
            logger.error("ValueError: Collection name cannot be empty.")
            raise ValueError("Collection name cannot be empty.")
        name = name.strip()
        if name not in self.list_collections():
            logger.error("ValueError: Collection '{}' does not exist.", name)
            raise ValueError(f"Collection '{name}' does not exist.")

        previous_collection = self.qdrant_collection
        self.qdrant_collection = name
        self._parent_context_support_cache.pop(previous_collection, None)
        self._parent_context_support_cache.pop(name, None)

        # Reset any state tied to the previously selected collection so that
        # future queries do not use stale indexes or conversations.
        self.docs.clear()
        self.nodes.clear()
        self.index = None
        self.query_engine = None
        self._image_ingestion_service = None
        self.reset_session_state()
        self._invalidate_ner_cache(previous_collection)
        self._invalidate_ner_cache(name)

    def _prepare_sources_dir(self, data_dir: Path) -> Path:
        """Ensure source files live under qdrant_src_dir/<collection> for preview and persistence.

        If the provided data_dir is already under that path, it is returned as-is.
        Otherwise, files/directories are copied into the target.

        Args:
            data_dir (Path): The original data directory.

        Returns:
            Path: The path to the staged sources directory.
        """
        if not self.qdrant_collection:
            return data_dir
        return stage_sources_to_qdrant(
            data_dir, self.qdrant_collection, self.qdrant_src_dir
        )

    def _finalize_empty_ingestion(
        self,
        collection: str,
        progress_callback: Callable[[str], None] | None,
    ) -> None:
        """Clean up after an ingestion that produced no content.

        Removes the orphan SQLite KV files for *collection* (the main
        ``<collection>_kv.db`` plus its ``-wal`` / ``-shm`` siblings) and
        best-effort deletes the ``<collection>_images`` companion Qdrant
        collection if it happens to exist. The user's uploaded source
        files under ``qdrant_src_dir / collection`` are intentionally
        retained so they can be inspected or retried.

        Emits a ``"warning:"``-prefixed progress message which the API SSE
        layer maps to a ``warning`` event, and a ``loguru`` warning log line.

        Args:
            collection (str): Name of the collection that produced no
                content.
            progress_callback (Callable[[str], None] | None): Optional
                callback for surfacing the warning to the UI.
        """
        warning_msg = (
            f"warning: No content was ingested for collection "
            f"'{collection}'. All source files were empty or contained "
            "no usable data (e.g., silent audio). Source files are kept "
            "on disk for inspection."
        )
        if progress_callback is not None:
            try:
                progress_callback(warning_msg)
            except Exception as exc:
                logger.warning("Empty-ingestion progress callback failed: {}", exc)
        logger.warning(
            "No documents produced during ingestion of '{}'; cleaning up "
            "empty KV store and companion collections.",
            collection,
        )

        db_path = self.qdrant_src_dir / collection / f"{collection}_kv.db"
        for suffix in ("", "-wal", "-shm"):
            candidate = db_path.with_name(db_path.name + suffix) if suffix else db_path
            if candidate.exists():
                try:
                    candidate.unlink()
                except OSError as exc:
                    logger.warning(
                        "Failed to remove orphan KV file '{}': {}",
                        candidate,
                        exc,
                    )

        images_collection = f"{collection}_images"
        if qdrant_collection_exists(self.qdrant_client, images_collection):
            try:
                self.qdrant_client.delete_collection(images_collection)
                logger.info(
                    "Deleted empty companion collection '{}' from Qdrant.",
                    images_collection,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to delete companion collection '{}': {}",
                    images_collection,
                    exc,
                )

    # --- Public API ---
    def ingest_docs(
        self,
        data_dir: str | Path,
        *,
        build_query_engine: bool = True,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        """Ingest documents from the specified directory into the Qdrant collection.

        Args:
            data_dir (str | Path): The directory containing the documents to ingest.
            build_query_engine (bool): Whether to eagerly build the query engine after
                ingestion. Disable when running headless ingestion jobs to avoid
                loading large reranker/generation models. Defaults to True.
            progress_callback (Callable[[str], None] | None): Optional callback for
                reporting ingestion progress.

        Raises:
            EmptyIngestionError: When no documents/nodes were produced and the
                target collection did not previously exist. Triggers cleanup of
                the orphan SQLite KV files; uploaded source files are kept.
        """
        prepared_dir = self._prepare_sources_dir(
            Path(data_dir) if isinstance(data_dir, str) else data_dir
        )
        self.data_dir = prepared_dir
        ingest_started_at = time.monotonic()
        core_docs = 0
        core_nodes = 0
        streaming_docs = 0
        streaming_nodes = 0
        enrich_batches = 0
        persist_batches = 0

        # Initialize index (load existing or create new wrapper)
        vector_store = self._vector_store()
        storage_ctx = self._storage_context(vector_store)

        # Build index with explicit storage_context so it uses the persistent docstore.
        embed_model = self.embed_model
        self.index = VectorStoreIndex(
            nodes=[],
            embed_model=embed_model,
            storage_context=storage_ctx,
        )

        pipeline = self._build_ingestion_pipeline(progress_callback=progress_callback)
        existing_hashes = self._get_existing_file_hashes()
        processed_hashes = set(existing_hashes)
        image_ingestion_service = getattr(pipeline, "image_ingestion_service", None)
        core_pdf_reader = CorePDFPipelineReader(
            data_dir=prepared_dir,
            entity_extractor=pipeline.entity_extractor,
            ner_max_workers=pipeline.ner_max_workers,
            source_collection=self.qdrant_collection,
            image_ingestion_service=image_ingestion_service,
        )

        for docs, nodes, file_hash in core_pdf_reader.build(
            existing_hashes=processed_hashes, progress_callback=progress_callback
        ):
            core_docs += len(docs)
            if nodes:
                self._persist_node_batches(nodes)
                core_nodes += len(nodes)
                persist_batches += len(
                    self._chunk_nodes(nodes, self.docstore_batch_size)
                )
                processed_hashes.add(file_hash)

        # PDFs are owned by the core pipeline reader and should not be
        # re-processed by the legacy ingestion path.
        processed_hashes.update(core_pdf_reader.discovered_hashes)

        # Process batches from the pipeline generator, persisting nodes as soon
        # as each enrichment micro-batch completes when supported.
        if hasattr(pipeline, "build_streaming") and callable(
            getattr(pipeline, "build_streaming")
        ):
            for docs, nodes, completed_hashes in pipeline.build_streaming(
                processed_hashes
            ):
                if docs:
                    streaming_docs += len(docs)
                if nodes:
                    self._persist_node_batches(nodes)
                    streaming_nodes += len(nodes)
                    persist_batches += len(
                        self._chunk_nodes(nodes, self.docstore_batch_size)
                    )
                    enrich_batches += 1
                if completed_hashes:
                    processed_hashes.update(completed_hashes)
        else:
            for docs, nodes in pipeline.build(processed_hashes):
                if docs:
                    streaming_docs += len(docs)
                if nodes:
                    self._persist_node_batches(nodes)
                    streaming_nodes += len(nodes)
                    persist_batches += len(
                        self._chunk_nodes(nodes, self.docstore_batch_size)
                    )

        total_docs = core_docs + streaming_docs
        total_nodes = core_nodes + streaming_nodes
        if (
            total_docs == 0
            and total_nodes == 0
            and not qdrant_collection_exists(self.qdrant_client, self.qdrant_collection)
        ):
            if self.ingest_benchmark_enabled:
                self._log_ingest_benchmark_summary(
                    mode="sync",
                    started_at=ingest_started_at,
                    core_docs=core_docs,
                    core_nodes=core_nodes,
                    streaming_docs=streaming_docs,
                    streaming_nodes=streaming_nodes,
                    enrich_batches=enrich_batches,
                    persist_batches=persist_batches,
                )
            self._finalize_empty_ingestion(self.qdrant_collection, progress_callback)
            raise EmptyIngestionError(self.qdrant_collection)

        self.dir_reader = pipeline.dir_reader
        # Clear memory-heavy lists as they are persisted in the vector store
        self.docs = []
        self.nodes = []

        if build_query_engine:
            self.create_query_engine()
        else:
            # Ensure downstream callers recreate a fresh query engine as needed.
            self.query_engine = None

        self.reset_session_state()
        self._invalidate_ner_cache(self.qdrant_collection)

        eff_k = None
        if self.query_engine is not None and hasattr(self.query_engine, "retriever"):
            try:
                eff_k = getattr(self.query_engine.retriever, "similarity_top_k", None)
            except Exception:
                eff_k = None

        if self.query_engine is not None:
            logger.info(
                "Effective retrieval k={} | top_n={} | embed_device={} | rerank_device={}",
                eff_k,
                self.rerank_top_n,
                self.device,
                self.device,
            )
        if self.ingest_benchmark_enabled:
            self._log_ingest_benchmark_summary(
                mode="sync",
                started_at=ingest_started_at,
                core_docs=core_docs,
                core_nodes=core_nodes,
                streaming_docs=streaming_docs,
                streaming_nodes=streaming_nodes,
                enrich_batches=enrich_batches,
                persist_batches=persist_batches,
            )
        self._bump_summary_revision(self.qdrant_collection)
        logger.info("Documents ingested successfully.")

    async def asingest_docs(
        self,
        data_dir: str | Path,
        *,
        build_query_engine: bool = True,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        """Asynchronously ingest documents from the specified directory into the Qdrant collection.

        Args:
            data_dir (str | Path): The directory containing the documents to ingest.
            build_query_engine (bool): Whether to build the query engine immediately
                after ingestion. Defaults to True.
            progress_callback (Callable[[str], None] | None): Optional callback for
                reporting ingestion progress.

        Raises:
            RuntimeError: If the index is not initialized for async ingestion.
            EmptyIngestionError: When no documents/nodes were produced and the
                target collection did not previously exist. Triggers cleanup of
                the orphan SQLite KV files; uploaded source files are kept.
        """
        prepared_dir = self._prepare_sources_dir(
            Path(data_dir) if isinstance(data_dir, str) else data_dir
        )
        self.data_dir = prepared_dir
        ingest_started_at = time.monotonic()
        core_docs = 0
        core_nodes = 0
        streaming_docs = 0
        streaming_nodes = 0
        enrich_batches = 0
        persist_batches = 0
        # Initialize index
        vector_store = self._vector_store()
        storage_ctx = self._storage_context(vector_store)
        embed_model = self.embed_model
        self.index = VectorStoreIndex(
            nodes=[],
            embed_model=embed_model,
            storage_context=storage_ctx,
        )

        pipeline = self._build_ingestion_pipeline(progress_callback=progress_callback)
        existing_hashes = self._get_existing_file_hashes()
        processed_hashes = set(existing_hashes)
        image_ingestion_service = getattr(pipeline, "image_ingestion_service", None)
        core_pdf_reader = CorePDFPipelineReader(
            data_dir=prepared_dir,
            entity_extractor=pipeline.entity_extractor,
            ner_max_workers=pipeline.ner_max_workers,
            source_collection=self.qdrant_collection,
            image_ingestion_service=image_ingestion_service,
        )

        for docs, nodes, file_hash in core_pdf_reader.build(
            existing_hashes=processed_hashes, progress_callback=progress_callback
        ):
            core_docs += len(docs)
            if nodes:
                await self._apersist_node_batches(nodes)
                core_nodes += len(nodes)
                persist_batches += len(
                    self._chunk_nodes(nodes, self.docstore_batch_size)
                )
                processed_hashes.add(file_hash)

        processed_hashes.update(core_pdf_reader.discovered_hashes)

        # Process batches, persisting nodes as soon as each enrichment
        # micro-batch completes when supported.
        if hasattr(pipeline, "build_streaming") and callable(
            getattr(pipeline, "build_streaming")
        ):
            for docs, nodes, completed_hashes in pipeline.build_streaming(
                processed_hashes
            ):
                if docs:
                    streaming_docs += len(docs)
                if nodes:
                    await self._apersist_node_batches(nodes)
                    streaming_nodes += len(nodes)
                    persist_batches += len(
                        self._chunk_nodes(nodes, self.docstore_batch_size)
                    )
                    enrich_batches += 1
                if completed_hashes:
                    processed_hashes.update(completed_hashes)
        else:
            for docs, nodes in pipeline.build(processed_hashes):
                if docs:
                    streaming_docs += len(docs)
                if nodes:
                    await self._apersist_node_batches(nodes)
                    streaming_nodes += len(nodes)
                    persist_batches += len(
                        self._chunk_nodes(nodes, self.docstore_batch_size)
                    )

        total_docs = core_docs + streaming_docs
        total_nodes = core_nodes + streaming_nodes
        if (
            total_docs == 0
            and total_nodes == 0
            and not qdrant_collection_exists(self.qdrant_client, self.qdrant_collection)
        ):
            if self.ingest_benchmark_enabled:
                self._log_ingest_benchmark_summary(
                    mode="async",
                    started_at=ingest_started_at,
                    core_docs=core_docs,
                    core_nodes=core_nodes,
                    streaming_docs=streaming_docs,
                    streaming_nodes=streaming_nodes,
                    enrich_batches=enrich_batches,
                    persist_batches=persist_batches,
                )
            self._finalize_empty_ingestion(self.qdrant_collection, progress_callback)
            raise EmptyIngestionError(self.qdrant_collection)

        self.dir_reader = pipeline.dir_reader
        self.docs = []
        self.nodes = []

        if build_query_engine:
            self.create_query_engine()
        else:
            self.query_engine = None

        self.reset_session_state()
        self._invalidate_ner_cache(self.qdrant_collection)

        eff_k = None
        if self.query_engine is not None and hasattr(self.query_engine, "retriever"):
            try:
                eff_k = getattr(self.query_engine.retriever, "similarity_top_k", None)
            except Exception:
                eff_k = None

        if self.query_engine is not None:
            logger.info(
                "Effective retrieval k={} | top_n={} | embed_device={} | rerank_device={}",
                eff_k,
                self.rerank_top_n,
                self.device,
                self.device,
            )
        if self.ingest_benchmark_enabled:
            self._log_ingest_benchmark_summary(
                mode="async",
                started_at=ingest_started_at,
                core_docs=core_docs,
                core_nodes=core_nodes,
                streaming_docs=streaming_docs,
                streaming_nodes=streaming_nodes,
                enrich_batches=enrich_batches,
                persist_batches=persist_batches,
            )
        self._bump_summary_revision(self.qdrant_collection)
        logger.info("Documents ingested successfully.")

    def run_query(
        self,
        prompt: str,
        *,
        metadata_filters: MetadataFilters | None = None,
        metadata_filter_rules: Sequence[Any] | None = None,
        vector_store_kwargs: dict[str, Any] | None = None,
        retrieval_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run a query against the Qdrant collection.

        Args:
            prompt (str): The query prompt.
            metadata_filters (MetadataFilters | None): Optional request-scoped
                metadata filters.
            metadata_filter_rules (Sequence[Any] | None): Optional raw request
                filter payloads for post-filtering auxiliary image sources.
            vector_store_kwargs (dict[str, Any] | None): Optional native
                vector-store query kwargs.
            retrieval_options (dict[str, Any] | None): Optional runtime
                retrieval overrides.

        Returns:
            dict[str, Any]: The query results.

        Raises:
            ValueError: If the prompt is empty.
            RuntimeError: If the query engine is not initialized.
            TypeError: If the response is not of the expected type.
        """
        if not prompt.strip():
            logger.error("ValueError: Query prompt cannot be empty.")
            raise ValueError("Query prompt cannot be empty.")
        engine = (
            self.build_query_engine(
                metadata_filters=metadata_filters,
                vector_store_kwargs=vector_store_kwargs,
                retrieval_options=retrieval_options,
            )
            if metadata_filters is not None or vector_store_kwargs or retrieval_options
            else self.query_engine
        )
        if engine is None:
            # Post-ingest eager warmup was intentionally removed to avoid
            # OOM on CPU (see commits 18a47a6 / 72e299e), so the default
            # query engine can legitimately still be None here after an
            # ingest + collection-select sequence. Build it lazily.
            # ``build_query_engine`` is typed non-Optional and raises on
            # its own failure modes, so no second None guard is needed.
            logger.debug("Query engine not initialized; building lazily for run_query.")
            self.query_engine = self.build_query_engine()
            engine = self.query_engine
        try:
            result = engine.query(prompt)
        except ValueError as exc:
            if "context size" in str(exc):
                logger.error(
                    "Context window overflow (configured {}): {}",
                    self.openai_ctx_window,
                    exc,
                )
                raise ValueError(
                    f"The query and retrieved context exceed the configured "
                    f"context window ({self.openai_ctx_window} tokens). "
                    f"Increase OPENAI_CTX_WINDOW to match your model's "
                    f"actual context length or reduce the retrieval top-k."
                ) from exc
            raise
        if not isinstance(result, Response):
            logger.error("TypeError: Expected Response, got {}.", type(result).__name__)
            raise TypeError(f"Expected Response, got {type(result).__name__}")
        normalized = self._normalize_response_data(
            prompt,
            result,
            metadata_filters_active=(
                metadata_filters is not None or bool(vector_store_kwargs)
            ),
            metadata_filter_rules=metadata_filter_rules,
        )
        retrieval_settings = self._resolve_runtime_retrieval_settings(
            retrieval_options=retrieval_options,
        )
        normalized["vector_query_mode"] = retrieval_settings[
            "vector_store_query_mode"
        ].value
        normalized["retrieval_profile"] = retrieval_settings["label"]
        normalized["parent_context_enabled"] = retrieval_settings[
            "parent_context_enabled"
        ]
        return normalized

    async def run_query_async(
        self,
        prompt: str,
        *,
        metadata_filters: MetadataFilters | None = None,
        metadata_filter_rules: Sequence[Any] | None = None,
        vector_store_kwargs: dict[str, Any] | None = None,
        retrieval_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run a query against the Qdrant collection asynchronously.

        Args:
            prompt (str): The query prompt.
            metadata_filters (MetadataFilters | None): Optional request-scoped
                metadata filters.
            metadata_filter_rules (Sequence[Any] | None): Optional raw request
                filter payloads for post-filtering auxiliary image sources.
            vector_store_kwargs (dict[str, Any] | None): Optional native
                vector-store query kwargs.
            retrieval_options (dict[str, Any] | None): Optional runtime
                retrieval overrides.

        Returns:
            dict[str, Any]: The query results.

        Raises:
            ValueError: If the prompt is empty.
            RuntimeError: If the query engine is not initialized.
            TypeError: If the response is not of the expected type.
        """
        if not prompt.strip():
            logger.error("ValueError: Query prompt cannot be empty.")
            raise ValueError("Query prompt cannot be empty.")
        engine = (
            self.build_query_engine(
                metadata_filters=metadata_filters,
                vector_store_kwargs=vector_store_kwargs,
                retrieval_options=retrieval_options,
            )
            if metadata_filters is not None or vector_store_kwargs or retrieval_options
            else self.query_engine
        )
        if engine is None:
            # See run_query for rationale: post-ingest warmup was
            # removed, so the default engine can be None on first use.
            # ``build_query_engine`` raises on its own failure modes;
            # no second None guard is needed here.
            logger.debug(
                "Query engine not initialized; building lazily for run_query_async."
            )
            self.query_engine = self.build_query_engine()
            engine = self.query_engine
        try:
            result = await engine.aquery(prompt)
        except ValueError as exc:
            if "context size" in str(exc):
                logger.error(
                    "Context window overflow (configured {}): {}",
                    self.openai_ctx_window,
                    exc,
                )
                raise ValueError(
                    f"The query and retrieved context exceed the configured "
                    f"context window ({self.openai_ctx_window} tokens). "
                    f"Increase OPENAI_CTX_WINDOW to match your model's "
                    f"actual context length or reduce the retrieval top-k."
                ) from exc
            raise
        if not isinstance(result, Response):
            logger.error("TypeError: Expected Response, got {}.", type(result).__name__)
            raise TypeError(f"Expected Response, got {type(result).__name__}")
        normalized = self._normalize_response_data(
            prompt,
            result,
            metadata_filters_active=(
                metadata_filters is not None or bool(vector_store_kwargs)
            ),
            metadata_filter_rules=metadata_filter_rules,
        )
        retrieval_settings = self._resolve_runtime_retrieval_settings(
            retrieval_options=retrieval_options,
        )
        normalized["vector_query_mode"] = retrieval_settings[
            "vector_store_query_mode"
        ].value
        normalized["retrieval_profile"] = retrieval_settings["label"]
        normalized["parent_context_enabled"] = retrieval_settings[
            "parent_context_enabled"
        ]
        return normalized

    # --- Session integration ---
    def init_session_store(self, db_url: str) -> None:
        """Initialize the relational session store via SessionManager.

        Args:
            db_url (str): The database URL for the session store.
        """
        if self.sessions is None:
            self.sessions = SessionManager(self)
        self.sessions.init_session_store(db_url=db_url)

    def reset_session_state(self) -> None:
        """Clear cached chat state so future sessions start fresh."""
        if self.sessions is not None:
            self.sessions.reset_runtime()

    def _invalidate_ner_cache(self, collection: str | None = None) -> None:
        """Invalidate cached NER payloads for one or all collections.

        Args:
            collection (str | None): Optional collection name. If omitted, clears all NER caches.
        """
        if collection is None:
            self.ner_sources = []
            self.ner_aggregate_cache.clear()
            self.ner_graph_cache.clear()
            self._parent_context_support_cache.clear()
            return

        stale_aggregate_keys = [
            key for key in self.ner_aggregate_cache if key[0] == collection
        ]
        for aggregate_key in stale_aggregate_keys:
            self.ner_aggregate_cache.pop(aggregate_key, None)
        stale_graph_keys: list[tuple[str, str, int, int]] = [
            key for key in self.ner_graph_cache if key[0] == collection
        ]
        for graph_key in stale_graph_keys:
            self.ner_graph_cache.pop(graph_key, None)

        if collection == self.qdrant_collection:
            self.ner_sources = []
        self._parent_context_support_cache.pop(collection, None)

    def export_session(
        self, session_id: str | None = None, out_dir: str | Path = "session"
    ) -> Path:
        """Delegate session export to SessionManager.

        Args:
            session_id (str | None): The session ID to export. If None, exports the
                current session.
            out_dir (str | Path): The output directory for the exported session.

        Returns:
            Path: The path to the exported session file.
        """
        if self.sessions is None:
            self.sessions = SessionManager(self)
        return self.sessions.export_session(session_id=session_id, out_dir=out_dir)

    def start_session(self, session_id: str | None = None) -> str:
        """Start or resume a chat session through SessionManager.

        Args:
            session_id (str | None): The session ID to start or resume. If None,
                a new session is created.
        """
        if self.sessions is None:
            self.sessions = SessionManager(self)
        return self.sessions.start_session(session_id)

    def chat(
        self,
        user_msg: str,
        *,
        metadata_filters: MetadataFilters | None = None,
        metadata_filters_active: bool = False,
        metadata_filter_rules: Sequence[Any] | None = None,
        vector_store_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Proxy chat turns to SessionManager.

        Args:
            user_msg (str): The user's chat message.
            metadata_filters (MetadataFilters | None): Optional request-scoped
                metadata filters.
            metadata_filters_active (bool): Whether request-scoped metadata
                filters were active for the retrieval.
            metadata_filter_rules (Sequence[Any] | None): Optional raw request
                filter payloads for post-filtering auxiliary image sources.
            vector_store_kwargs (dict[str, Any] | None): Optional native
                vector-store query kwargs.

        Returns:
            dict[str, Any]: The chat response data.
        """
        if self.sessions is None:
            self.sessions = SessionManager(self)
        return self.sessions.chat(
            user_msg,
            metadata_filters=metadata_filters,
            metadata_filters_active=metadata_filters_active,
            metadata_filter_rules=metadata_filter_rules,
            vector_store_kwargs=vector_store_kwargs,
        )

    def stream_chat(
        self,
        user_msg: str,
        *,
        metadata_filters: MetadataFilters | None = None,
        metadata_filters_active: bool = False,
        metadata_filter_rules: Sequence[Any] | None = None,
        vector_store_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Proxy stream chat turns to SessionManager.

        Args:
            user_msg (str): The user's chat message.
            metadata_filters (MetadataFilters | None): Optional request-scoped
                metadata filters.
            metadata_filters_active (bool): Whether request-scoped metadata
                filters were active for the retrieval.
            metadata_filter_rules (Sequence[Any] | None): Optional raw request
                filter payloads for post-filtering auxiliary image sources.
            vector_store_kwargs (dict[str, Any] | None): Optional native
                vector-store query kwargs.

        Returns:
            Any: A generator yielding response chunks.
        """
        if self.sessions is None:
            self.sessions = SessionManager(self)
        return self.sessions.stream_chat(
            user_msg,
            metadata_filters=metadata_filters,
            metadata_filters_active=metadata_filters_active,
            metadata_filter_rules=metadata_filter_rules,
            vector_store_kwargs=vector_store_kwargs,
        )

    def expand_query_with_graph_with_debug(
        self, query: str
    ) -> tuple[str, dict[str, Any]]:
        """Optionally expand a query and return GraphRAG debug metadata.

        Args:
            query (str): Original retrieval query.

        Returns:
            tuple[str, dict[str, Any]]: A tuple of ``(expanded_query, debug_payload)``.
        """
        debug: dict[str, Any] = {
            "enabled": bool(self.graphrag_enabled),
            "applied": False,
            "original_query": query,
            "expanded_query": query,
            "anchor_entities": [],
            "neighbor_entities": [],
        }

        if not query.strip():
            debug["reason"] = "empty_query"
            return query, debug
        if not self.qdrant_collection:
            debug["reason"] = "no_collection_selected"
            return query, debug
        if not self.graphrag_enabled:
            debug["reason"] = "graphrag_disabled"
            return query, debug

        try:
            aggregate = self._get_collection_ner_aggregate(refresh=False)
            entities = list(aggregate.get("entities") or [])
            anchors = []
            for ent in entities:
                text = str(ent.get("text") or "").strip()
                if not text:
                    continue
                match = match_entity_text(text, query)
                if match is None:
                    continue
                anchors.append((match[0], ent))
            anchors.sort(
                key=lambda item: (
                    int(item[0]),
                    -int(item[1].get("mentions", 0) or 0),
                    str(item[1].get("text") or "").lower(),
                )
            )
            if not anchors:
                debug["reason"] = "no_anchor_entities"
                return query, debug

            selected_anchors = [ent for _, ent in anchors[:2]]
            anchor_texts = [
                str(ent.get("text") or "").strip() for ent in selected_anchors
            ]
            debug["anchor_entities"] = [txt for txt in anchor_texts if txt]
            anchor_text_set = set(debug["anchor_entities"])
            neighbor_texts: list[str] = []
            seen: set[str] = set()
            for ent in selected_anchors:
                neighborhood = self.get_collection_ner_graph_neighbors(
                    entity=str(ent.get("text") or ""),
                    hops=self.graphrag_neighbor_hops,
                    top_k_nodes=self.graphrag_top_k_nodes,
                    min_edge_weight=self.graphrag_min_edge_weight,
                    refresh=False,
                )
                for nbr in neighborhood.get("neighbors") or []:
                    text = str(nbr.get("text") or "").strip()
                    if (
                        not text
                        or text in anchor_text_set
                        or text.lower() in seen
                        or len(neighbor_texts) >= self.graphrag_max_neighbors
                    ):
                        continue
                    seen.add(text.lower())
                    neighbor_texts.append(text)
                if len(neighbor_texts) >= self.graphrag_max_neighbors:
                    break

            debug["neighbor_entities"] = neighbor_texts
            if not neighbor_texts:
                debug["reason"] = "no_neighbors_found"
                return query, debug

            related = ", ".join(neighbor_texts)
            expanded = f"{query}\n\nRelated entities for retrieval: {related}"
            debug["applied"] = True
            debug["expanded_query"] = expanded
            return expanded, debug
        except Exception as exc:
            logger.warning("Graph query expansion skipped: {}", exc)
            debug["reason"] = f"error:{type(exc).__name__}"
            return query, debug

    def expand_query_with_graph(self, query: str) -> str:
        """Optionally expand a query using graph-neighbor entities.

        Args:
            query (str): Original retrieval query.

        Returns:
            str: Expanded query when graph expansion is enabled and applicable,
            otherwise the original query.
        """
        expanded_query, _ = self.expand_query_with_graph_with_debug(query)
        return expanded_query

    def _summary_document_targets(self) -> list[dict[str, Any]]:
        """Return capped document targets ordered by descending node count."""
        documents = self.list_documents()
        documents.sort(
            key=lambda item: (
                -int(item.get("node_count", 0) or 0),
                str(item.get("filename") or "").lower(),
            )
        )
        return documents[: self.summary_max_docs]

    def _summary_source_matches_document(
        self,
        source: dict[str, Any],
        *,
        filename: str,
        file_hash: str | None,
    ) -> bool:
        """Check whether a normalized source belongs to a document target."""
        if file_hash and str(source.get("file_hash") or "") == file_hash:
            return True

        src_filename = str(source.get("filename") or "").strip()
        if not src_filename:
            return False

        if src_filename == filename:
            return True

        try:
            return Path(src_filename).name == Path(filename).name
        except Exception:
            return src_filename.lower() == filename.lower()

    def _summary_document_filters(
        self, *, filename: str, file_hash: str | None
    ) -> MetadataFilters:
        """Build metadata filters that scope retrieval to one document."""
        filters: list[MetadataFilter | MetadataFilters] = [
            MetadataFilter(key="filename", value=filename, operator=FilterOperator.EQ),
            MetadataFilter(key="file_name", value=filename, operator=FilterOperator.EQ),
            MetadataFilter(key="file_path", value=filename, operator=FilterOperator.EQ),
        ]
        if file_hash:
            filters.append(
                MetadataFilter(
                    key="file_hash", value=file_hash, operator=FilterOperator.EQ
                )
            )
        return MetadataFilters(filters=filters, condition=FilterCondition.OR)

    def _summary_payload_fallback_nodes(
        self, *, filename: str, file_hash: str | None
    ) -> list[NodeWithScore]:
        """Build synthetic summary nodes by scrolling stored payloads.

        This fallback bypasses query embeddings entirely, so summary generation
        can still proceed when the embedding backend returns invalid values.

        Args:
            filename (str): Target document filename.
            file_hash (str | None): Optional file hash for precise scoping.

        Returns:
            list[NodeWithScore]: Synthetic node hits derived from stored payloads.
        """
        if not self.qdrant_collection:
            return []

        top_k = max(1, self.summary_per_doc_top_k)
        matched_nodes: list[NodeWithScore] = []
        offset = None
        scroll_filter = None
        if file_hash:
            scroll_filter = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="file_hash",
                        match=qdrant_models.MatchValue(value=file_hash),
                    )
                ]
            )

        while len(matched_nodes) < top_k:
            try:
                points, offset = self.qdrant_client.scroll(
                    collection_name=self.qdrant_collection,
                    limit=128,
                    offset=offset,
                    scroll_filter=scroll_filter,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as exc:
                logger.warning(
                    "Payload fallback summary retrieval failed for '{}': {}",
                    filename,
                    exc,
                )
                break

            if not points:
                break

            for index, point in enumerate(points, start=len(matched_nodes) + 1):
                payload = getattr(point, "payload", None)
                if not isinstance(payload, dict):
                    continue
                source = self._source_from_payload(
                    collection=self.qdrant_collection,
                    payload=payload,
                )
                if not self._summary_source_matches_document(
                    source,
                    filename=filename,
                    file_hash=file_hash,
                ):
                    continue

                node_id = str(
                    payload.get("node_id")
                    or payload.get("id_")
                    or getattr(point, "id", "")
                    or f"summary-fallback-{index}"
                )
                text_value = str(source.get("text") or source.get("preview_text") or "")
                synthetic_node = TextNode(
                    text=text_value,
                    id_=node_id,
                    metadata=dict(payload),
                )
                matched_nodes.append(NodeWithScore(node=synthetic_node, score=0.0))
                if len(matched_nodes) >= top_k:
                    break

            if offset is None:
                break

        return matched_nodes

    def _retrieve_summary_nodes_for_document(
        self, *, filename: str, file_hash: str | None
    ) -> list[Any]:
        """Retrieve top evidence nodes for a single document."""
        if self.index is None:
            self.create_index()
        if self.index is None:
            return []

        query = (
            f"Extract factual highlights and notable findings from '{filename}'. "
            "Focus on substantive content."
        )
        top_k = max(1, self.summary_per_doc_top_k)
        try:
            retriever = self.index.as_retriever(
                similarity_top_k=top_k,
                filters=self._summary_document_filters(
                    filename=filename, file_hash=file_hash
                ),
            )
            nodes = retriever.retrieve(query)
            if isinstance(nodes, list):
                return nodes[:top_k]
        except Exception as exc:
            logger.warning(
                "Filtered summary retrieval failed for '{}': {}",
                filename,
                exc,
            )

        try:
            fallback_retriever = self.index.as_retriever(similarity_top_k=top_k * 2)
            fallback_nodes = fallback_retriever.retrieve(query)
            if not isinstance(fallback_nodes, list):
                return []
            matched_nodes: list[Any] = []
            for nws in fallback_nodes:
                source = self._source_from_node_with_score(nws)
                if source is None:
                    continue
                if self._summary_source_matches_document(
                    source, filename=filename, file_hash=file_hash
                ):
                    matched_nodes.append(nws)
                if len(matched_nodes) >= top_k:
                    break
            return matched_nodes
        except Exception as exc:
            logger.warning(
                "Fallback summary retrieval failed for '{}': {}", filename, exc
            )
            return self._summary_payload_fallback_nodes(
                filename=filename,
                file_hash=file_hash,
            )

    def _summary_source_key(self, source: dict[str, Any]) -> str:
        """Build a deterministic deduplication key for summary sources.

        Args:
            source (dict[str, Any]): A normalized source dictionary.

        Returns:
            str: A string key that uniquely identifies the source for deduplication purposes.
        """
        reference_metadata = source.get("reference_metadata")
        text_id = ""
        if isinstance(reference_metadata, dict):
            text_id = str(reference_metadata.get("text_id") or "").strip()
        if text_id:
            return f"text_id||{text_id}"
        return "||".join(
            [
                str(source.get("file_hash") or ""),
                str(source.get("filename") or ""),
                str(source.get("page") or ""),
                str(source.get("row") or ""),
                str(source.get("preview_text") or source.get("text") or ""),
            ]
        )

    def _summary_document_brief(
        self, *, filename: str, sources: list[dict[str, Any]]
    ) -> str:
        """Build one compact, evidence-first brief for a document.

        Args:
            filename (str): The name of the document.
            sources (list[dict[str, Any]]): A list of normalized source dictionaries associated with the document.

        Returns:
            str: A formatted string brief that includes key points and evidence snippets from the sources.
        """
        snippets: list[str] = []
        for source in sources:
            raw_text = str(
                source.get("preview_text") or source.get("text") or ""
            ).strip()
            if not raw_text:
                continue
            compact = re.sub(r"\s+", " ", raw_text)
            snippets.append(compact[:240])
            if len(snippets) >= 2:
                break

        if not snippets:
            return f"- Document: {filename}\n  - Evidence: (none)"

        evidence_lines = "\n".join(
            f"  - Evidence {idx}: {snippet}"
            for idx, snippet in enumerate(snippets, start=1)
        )
        key_points = " ; ".join(snippets)
        return f"- Document: {filename}\n  - Key points: {key_points}\n{evidence_lines}"

    def _merge_summary_sources(
        self, per_doc_sources: dict[str, list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        """Merge per-document evidence and guarantee broad document coverage.

        Args:
            per_doc_sources (dict[str, list[dict[str, Any]]]): A dictionary mapping document identifiers to lists of source dictionaries.

        Returns:
            list[dict[str, Any]]: A merged list of source dictionaries that prioritizes at least one source per document and fills remaining slots with additional evidence.
        """
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()

        # Ensure at least one source per covered document.
        for sources in per_doc_sources.values():
            if not sources:
                continue
            key = self._summary_source_key(sources[0])
            if key in seen:
                continue
            seen.add(key)
            merged.append(sources[0])
            if len(merged) >= self.summary_final_source_cap:
                return merged

        # Fill remaining slots with additional evidence snippets.
        for sources in per_doc_sources.values():
            for source in sources[1:]:
                key = self._summary_source_key(source)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(source)
                if len(merged) >= self.summary_final_source_cap:
                    return merged
        return merged

    def _build_summary_synthesis_prompt(
        self,
        *,
        briefs: list[str],
        diagnostics: dict[str, Any],
        style_prompt: str,
    ) -> str:
        """Build the final synthesis prompt from per-document evidence briefs.

        Args:
            briefs (list[str]): A list of evidence briefs for each document.
            diagnostics (dict[str, Any]): A dictionary containing diagnostic information such as coverage ratio and uncovered documents.

        Returns:
            str: A formatted string representing the final synthesis prompt.
        """
        coverage_ratio = float(diagnostics.get("coverage_ratio", 0.0) or 0.0)
        coverage_target = float(diagnostics.get("coverage_target", 0.0) or 0.0)
        coverage_unit = str(diagnostics.get("coverage_unit") or "documents")
        uncovered = diagnostics.get("uncovered_documents") or []
        uncovered_text = ", ".join(str(item) for item in uncovered) or "(none)"
        evidence_block = "\n\n".join(briefs) if briefs else "(no evidence extracted)"
        return (
            "You are producing a grounded collection summary.\n"
            "Use only the evidence briefs below. If evidence is insufficient, state that explicitly.\n"
            "Include cross-document themes, notable differences or outliers, and concrete findings.\n"
            "Do not introduce claims unsupported by the evidence briefs.\n\n"
            f"Coverage unit: {coverage_unit}\n"
            f"Coverage ratio: {coverage_ratio:.2f}\n"
            f"Coverage target: {coverage_target:.2f}\n"
            f"Uncovered documents: {uncovered_text}\n\n"
            f"Style instructions:\n{style_prompt.strip()}\n\n"
            f"Evidence briefs:\n{evidence_block}\n"
        )

    def _prepare_document_summary_context(self) -> dict[str, Any]:
        """Prepare document-level summary context for standard collections."""
        targets = self._summary_document_targets()
        target_filenames = [
            str(doc.get("filename") or "") for doc in targets if doc.get("filename")
        ]
        per_doc_sources: dict[str, list[dict[str, Any]]] = {}
        briefs: list[str] = []
        candidate_count = 0
        deduped_count = 0

        for doc in targets:
            filename = str(doc.get("filename") or "").strip()
            if not filename:
                continue
            file_hash_raw = doc.get("file_hash")
            file_hash = str(file_hash_raw).strip() if file_hash_raw else None
            nodes = self._retrieve_summary_nodes_for_document(
                filename=filename,
                file_hash=file_hash,
            )
            candidate_count += len(nodes)
            normalized_sources: list[dict[str, Any]] = []
            seen_doc_keys: set[str] = set()
            for nws in nodes:
                source = self._source_from_node_with_score(nws)
                if source is None:
                    continue
                if not self._summary_source_matches_document(
                    source, filename=filename, file_hash=file_hash
                ):
                    continue
                key = self._summary_source_key(source)
                if key in seen_doc_keys:
                    continue
                seen_doc_keys.add(key)
                normalized_sources.append(source)
                if len(normalized_sources) >= self.summary_per_doc_top_k:
                    break

            if normalized_sources:
                deduped_count += len(normalized_sources)
                per_doc_sources[filename] = normalized_sources
                briefs.append(
                    self._summary_document_brief(
                        filename=filename,
                        sources=normalized_sources,
                    )
                )

        covered_filenames = list(per_doc_sources.keys())
        uncovered = [
            filename for filename in target_filenames if filename not in per_doc_sources
        ]
        total_documents = len(target_filenames)
        covered_documents = len(covered_filenames)
        coverage_ratio = (
            covered_documents / total_documents if total_documents > 0 else 0.0
        )
        diagnostics = {
            "total_documents": total_documents,
            "covered_documents": covered_documents,
            "coverage_ratio": round(coverage_ratio, 4),
            "uncovered_documents": uncovered,
            "coverage_target": self.summary_coverage_target,
            "coverage_unit": "documents",
            "candidate_count": candidate_count,
            "deduped_count": deduped_count,
            "sampled_count": len(self._merge_summary_sources(per_doc_sources)),
        }
        merged_sources = self._merge_summary_sources(per_doc_sources)

        return {
            "synthesis_prompt": self._build_summary_synthesis_prompt(
                briefs=briefs,
                diagnostics=diagnostics,
                style_prompt=self.summarize_prompt,
            ),
            "sources": merged_sources,
            "summary_diagnostics": diagnostics,
        }

    def _retrieve_social_summary_nodes(self) -> list[Any]:
        """Retrieve a larger candidate pool for social/table-heavy summaries."""
        if self.index is None:
            self.create_index()
        if self.index is None:
            return []

        query = (
            "Identify representative posts or rows, recurring themes, concrete "
            "claims, disagreements, and notable outliers across this collection."
        )
        top_k = max(
            1,
            int(
                max(
                    self.social_summary_candidate_pool,
                    self.summary_final_source_cap,
                )
            ),
        )
        try:
            retriever = self.index.as_retriever(similarity_top_k=top_k)
            nodes = retriever.retrieve(query)
            if isinstance(nodes, list):
                return nodes[:top_k]
        except Exception as exc:
            logger.warning("Social summary retrieval failed: {}", exc)
        return []

    def _select_social_summary_sources(
        self,
        candidates: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Deduplicate and diversify candidate social/table sources."""
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for source in candidates:
            identity = self._source_post_key(source)
            if identity and identity in seen:
                continue
            if identity:
                seen.add(identity)
            deduped.append(source)

        bucket_counts: dict[str, int] = defaultdict(int)
        sampled: list[dict[str, Any]] = []
        limit = max(1, int(self.social_summary_diversity_limit))
        for source in deduped:
            bucket = self._source_diversity_bucket(source)
            if bucket_counts[bucket] >= limit:
                continue
            bucket_counts[bucket] += 1
            sampled.append(source)
            if len(sampled) >= self.summary_final_source_cap:
                break
        return deduped, sampled

    def _count_social_coverage_units(self, coverage_unit: str) -> int:
        """Count total social coverage units across the active collection."""
        if not self.qdrant_collection:
            return 0
        seen: set[str] = set()
        offset = None
        while True:
            try:
                points, offset = self.qdrant_client.scroll(
                    collection_name=self.qdrant_collection,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to count social coverage units for '{}': {}",
                    self.qdrant_collection,
                    exc,
                )
                break
            if not points:
                break
            for point in points:
                payload = getattr(point, "payload", None)
                if not isinstance(payload, dict) or not self._is_social_payload(
                    payload
                ):
                    continue
                source = self._source_from_payload(
                    collection=self.qdrant_collection,
                    payload=payload,
                )
                if coverage_unit == "posts":
                    key = self._source_post_key(source)
                else:
                    key = self._summary_source_key(source)
                if key:
                    seen.add(key)
            if offset is None:
                break
        return len(seen)

    def _build_social_summary_briefs(
        self,
        sources: list[dict[str, Any]],
    ) -> list[str]:
        """Build source-preserving evidence briefs for row-level social summaries."""
        briefs: list[str] = []
        for index, source in enumerate(sources, start=1):
            reference_metadata = source.get("reference_metadata")
            ref = reference_metadata if isinstance(reference_metadata, dict) else {}
            metadata_bits = [
                f"network={ref.get('network')}" if ref.get("network") else "",
                f"type={ref.get('type')}" if ref.get("type") else "",
                f"author={ref.get('author') or ref.get('author_id')}"
                if (ref.get("author") or ref.get("author_id"))
                else "",
                f"timestamp={ref.get('timestamp')}" if ref.get("timestamp") else "",
                f"row={source.get('row')}" if source.get("row") is not None else "",
            ]
            metadata_line = (
                ", ".join(bit for bit in metadata_bits if bit) or "metadata=n/a"
            )
            raw_text = str(
                source.get("text") or source.get("preview_text") or ""
            ).strip()
            compact = re.sub(r"\s+", " ", raw_text)[:280] if raw_text else "(no text)"
            briefs.append(f"- Source {index}: {metadata_line}\n  - Evidence: {compact}")
        return briefs

    def _prepare_social_summary_context(self) -> dict[str, Any]:
        """Prepare chunk/post-level summary context for social/table collections."""
        candidate_nodes = self._retrieve_social_summary_nodes()
        candidate_sources: list[dict[str, Any]] = []
        for nws in candidate_nodes:
            source = self._source_from_node_with_score(nws)
            if source is None:
                continue
            if str(source.get("source") or "") != "table":
                continue
            candidate_sources.append(source)

        if not candidate_sources:
            for payload in self._sample_collection_payloads(
                limit=self.social_summary_candidate_pool
            ):
                if not self._is_social_payload(payload):
                    continue
                candidate_sources.append(
                    self._source_from_payload(
                        collection=self.qdrant_collection,
                        payload=payload,
                    )
                )

        deduped_sources, sampled_sources = self._select_social_summary_sources(
            candidate_sources
        )
        coverage_unit = self._coverage_unit_for_sources(
            sampled_sources or deduped_sources or candidate_sources
        )
        total_units = self._count_social_coverage_units(coverage_unit)
        covered_units = len(sampled_sources)
        coverage_ratio = covered_units / total_units if total_units > 0 else 0.0
        diagnostics = {
            "total_documents": total_units,
            "covered_documents": covered_units,
            "coverage_ratio": round(coverage_ratio, 4),
            "uncovered_documents": [],
            "coverage_target": self.summary_coverage_target,
            "coverage_unit": coverage_unit,
            "candidate_count": len(candidate_sources),
            "deduped_count": len(deduped_sources),
            "sampled_count": len(sampled_sources),
        }

        briefs = self._build_social_summary_briefs(sampled_sources)
        return {
            "synthesis_prompt": self._build_summary_synthesis_prompt(
                briefs=briefs,
                diagnostics=diagnostics,
                style_prompt=self.summarize_social_prompt,
            ),
            "sources": sampled_sources,
            "summary_diagnostics": diagnostics,
        }

    def _prepare_collection_summary_context(self) -> dict[str, Any]:
        """Prepare summary context for the active collection."""
        profile = self._infer_collection_profile()
        if self.social_summary_enabled and bool(profile.get("is_social_table")):
            return self._prepare_social_summary_context()
        return self._prepare_document_summary_context()

    def _summary_kv_store(
        self,
        collection: str | None = None,
        *,
        allow_create: bool = True,
    ) -> BaseKVStore | None:
        """Return the per-collection KV store used by summary cache operations.

        Args:
            collection: Optional collection name override.
            allow_create: When ``False``, return ``None`` unless the
                collection's SQLite KV database already exists on disk.
                This prevents summary reads from spuriously creating an
                empty database for a collection that was never ingested.

        Returns:
            BaseKVStore | None: A KV store instance when available, else None.
        """
        target = str(collection or self.qdrant_collection or "").strip()
        if not target:
            return None

        if not allow_create:
            db_path = self.qdrant_src_dir / target / f"{target}_kv.db"
            if not db_path.exists():
                return None

        try:
            return self._build_kv_store(collection=target)
        except Exception as exc:
            logger.warning(
                "Failed to initialize summary cache KV store for '{}': {}",
                target,
                exc,
            )
            return None

    def _summary_prompt_fingerprint(self) -> str:
        """Build a stable fingerprint for summarize prompt and summary knobs.

        Returns:
            str: SHA-256 fingerprint used for cache validation.
        """
        payload = {
            "summarize_prompt": self.summarize_prompt,
            "summarize_social_prompt": self.summarize_social_prompt,
            "summary_coverage_target": self.summary_coverage_target,
            "summary_max_docs": self.summary_max_docs,
            "summary_per_doc_top_k": self.summary_per_doc_top_k,
            "summary_final_source_cap": self.summary_final_source_cap,
            "social_summary_enabled": self.social_summary_enabled,
            "social_summary_candidate_pool": self.social_summary_candidate_pool,
            "social_summary_diversity_limit": self.social_summary_diversity_limit,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _get_summary_revision(
        self,
        collection: str | None = None,
        *,
        allow_create: bool = True,
    ) -> int:
        """Load the current summary revision for a collection.

        Args:
            collection (str | None): Optional collection name override.
            allow_create (bool): Whether creating the dockv collection is allowed.

        Returns:
            int: Monotonic revision value; defaults to 0 when unavailable.
        """
        kv_store = self._summary_kv_store(
            collection=collection, allow_create=allow_create
        )
        if kv_store is None:
            return 0

        try:
            stored = kv_store.get(
                SUMMARY_CACHE_REVISION_KEY,
                collection=SUMMARY_CACHE_NAMESPACE,
            )
        except Exception as exc:
            logger.warning("Failed to load summary revision: {}", exc)
            return 0

        if not isinstance(stored, dict):
            return 0
        try:
            revision = int(stored.get("revision", 0))
        except (TypeError, ValueError):
            return 0
        return max(0, revision)

    def _bump_summary_revision(
        self,
        collection: str | None = None,
        *,
        allow_create: bool = True,
    ) -> int:
        """Increment and persist summary revision for a collection.

        Args:
            collection (str | None): Optional collection name override.
            allow_create (bool): Whether creating the dockv collection is allowed.

        Returns:
            int: The updated revision.
        """
        kv_store = self._summary_kv_store(
            collection=collection, allow_create=allow_create
        )
        if kv_store is None:
            return 0

        current_revision = self._get_summary_revision(
            collection=collection,
            allow_create=allow_create,
        )
        next_revision = current_revision + 1
        try:
            kv_store.put(
                SUMMARY_CACHE_REVISION_KEY,
                {"revision": next_revision},
                collection=SUMMARY_CACHE_NAMESPACE,
            )
        except Exception as exc:
            logger.warning("Failed to persist summary revision bump: {}", exc)
            return current_revision
        return next_revision

    def _load_cached_collection_summary(
        self, *, refresh: bool
    ) -> dict[str, Any] | None:
        """Load a cached summary if revision and prompt fingerprint still match.

        Args:
            refresh (bool): If ``True``, bypass cache lookup.

        Returns:
            dict[str, Any] | None: Cached summary payload or None when stale/missing.
        """
        if refresh:
            return None

        kv_store = self._summary_kv_store()
        if kv_store is None:
            return None

        try:
            payload = kv_store.get(
                SUMMARY_CACHE_PAYLOAD_KEY,
                collection=SUMMARY_CACHE_NAMESPACE,
            )
        except Exception as exc:
            logger.warning("Failed to load cached collection summary: {}", exc)
            return None

        if not isinstance(payload, dict):
            return None

        try:
            cached_revision = int(payload.get("revision", -1))
        except (TypeError, ValueError):
            return None
        current_revision = self._get_summary_revision()
        if cached_revision != current_revision:
            return None

        expected_fingerprint = self._summary_prompt_fingerprint()
        cached_fingerprint = str(payload.get("prompt_fingerprint") or "")
        if cached_fingerprint != expected_fingerprint:
            return None

        sources = payload.get("sources")
        if not isinstance(sources, list):
            sources = []
        summary_diagnostics = payload.get("summary_diagnostics")
        if not isinstance(summary_diagnostics, dict):
            summary_diagnostics = {}

        return {
            "query": self.summarize_prompt,
            "reasoning": None,
            "response": str(payload.get("response") or ""),
            "sources": sources,
            "summary_diagnostics": summary_diagnostics,
        }

    def _store_cached_collection_summary(self, payload: dict[str, Any]) -> None:
        """Persist a collection summary payload in the dockv summary namespace.

        Args:
            payload (dict[str, Any]): Summary payload to cache.
        """
        kv_store = self._summary_kv_store()
        if kv_store is None:
            return

        revision = self._get_summary_revision()
        prompt_fingerprint = self._summary_prompt_fingerprint()
        sources = payload.get("sources")
        if not isinstance(sources, list):
            sources = []
        summary_diagnostics = payload.get("summary_diagnostics")
        if not isinstance(summary_diagnostics, dict):
            summary_diagnostics = {}

        cache_payload = {
            "revision": revision,
            "prompt_fingerprint": prompt_fingerprint,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "response": str(payload.get("response") or ""),
            "sources": sources,
            "summary_diagnostics": summary_diagnostics,
        }
        try:
            kv_store.put(
                SUMMARY_CACHE_PAYLOAD_KEY,
                cache_payload,
                collection=SUMMARY_CACHE_NAMESPACE,
            )
            kv_store.put(
                SUMMARY_CACHE_REVISION_KEY,
                {"revision": revision},
                collection=SUMMARY_CACHE_NAMESPACE,
            )
        except Exception as exc:
            logger.warning("Failed to store cached collection summary: {}", exc)

    def summarize_collection(self, refresh: bool = False) -> dict[str, Any]:
        """Generate a coverage-aware summary for the selected collection.

        Args:
            refresh (bool): If ``True``, bypass cached summary payloads.

        Returns:
            dict[str, Any]: Summary payload with normalized sources and diagnostics.

        Raises:
            ValueError: If no collection is selected.
        """
        if not self.qdrant_collection:
            raise ValueError("No collection selected.")

        cached_payload = self._load_cached_collection_summary(refresh=refresh)
        if cached_payload is not None:
            return cached_payload

        context = self._prepare_collection_summary_context()
        diagnostics = context["summary_diagnostics"]
        covered_documents = int(diagnostics.get("covered_documents", 0) or 0)
        total_documents = int(diagnostics.get("total_documents", 0) or 0)

        if total_documents == 0:
            summary_text = "No documents available in the selected collection."
        elif covered_documents == 0:
            summary_text = (
                "Unable to extract grounded evidence from the selected collection."
            )
        else:
            completion = self.post_retrieval_text_model.complete(
                context["synthesis_prompt"]
            )
            summary_text = str(getattr(completion, "text", "") or "").strip()

        payload = {
            "query": self.summarize_prompt,
            "reasoning": None,
            "response": summary_text,
            "sources": context["sources"],
            "summary_diagnostics": diagnostics,
        }
        self._store_cached_collection_summary(payload)
        return payload

    def stream_summarize_collection(self, refresh: bool = False) -> Any:
        """Generate a streaming summary of the currently selected collection.

        Args:
            refresh (bool): If ``True``, bypass cached summary payloads.

        Yields:
            str | dict: Chunks of text, followed by a dict with metadata.

        Raises:
            ValueError: If no collection is selected.
        """
        if not self.qdrant_collection:
            raise ValueError("No collection selected.")

        cached_payload = self._load_cached_collection_summary(refresh=refresh)
        if cached_payload is not None:
            full_text = str(cached_payload.get("response") or "")
            if full_text:
                yield full_text
            yield cached_payload
            return

        context = self._prepare_collection_summary_context()
        diagnostics = context["summary_diagnostics"]
        covered_documents = int(diagnostics.get("covered_documents", 0) or 0)
        total_documents = int(diagnostics.get("total_documents", 0) or 0)

        if total_documents == 0:
            full_text = "No documents available in the selected collection."
            payload = {
                "query": self.summarize_prompt,
                "reasoning": None,
                "response": full_text,
                "sources": context["sources"],
                "summary_diagnostics": diagnostics,
            }
            self._store_cached_collection_summary(payload)
            yield full_text
            yield payload
            return

        if covered_documents == 0:
            full_text = (
                "Unable to extract grounded evidence from the selected collection."
            )
            payload = {
                "query": self.summarize_prompt,
                "reasoning": None,
                "response": full_text,
                "sources": context["sources"],
                "summary_diagnostics": diagnostics,
            }
            self._store_cached_collection_summary(payload)
            yield full_text
            yield payload
            return

        full_text = ""
        running_text = ""
        for chunk in self.post_retrieval_text_model.stream_complete(
            context["synthesis_prompt"]
        ):
            delta = getattr(chunk, "delta", None)
            if isinstance(delta, str) and delta:
                token = delta
                running_text += token
            else:
                text_value = str(getattr(chunk, "text", "") or "")
                if text_value.startswith(running_text):
                    token = text_value[len(running_text) :]
                else:
                    token = text_value
                running_text = text_value
            if not token:
                continue
            full_text += token
            yield token

        payload = {
            "query": self.summarize_prompt,
            "reasoning": None,
            "response": full_text,
            "sources": context["sources"],
            "summary_diagnostics": diagnostics,
        }
        self._store_cached_collection_summary(payload)
        yield payload

    def list_documents(self) -> list[dict[str, Any]]:
        """List all documents in the current collection by scanning all points.

        Returns:
            list[dict[str, Any]]: A list of document metadata dictionaries.
        """
        if not self.qdrant_collection:
            return []

        docs_map: dict[str, dict[str, Any]] = {}
        offset = None

        while True:
            try:
                points, offset = self.qdrant_client.scroll(
                    collection_name=self.qdrant_collection,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as exc:
                logger.error(
                    "Failed to scroll collection '{}': {}", self.qdrant_collection, exc
                )
                break

            if not points:
                break

            for point in points:
                payload = getattr(point, "payload", {}) or {}
                origin = payload.get("origin") or {}
                filename = (
                    origin.get("filename")
                    or payload.get("file_name")
                    or payload.get("filename")
                    or payload.get("file_path")
                )
                if not filename:
                    continue

                if filename not in docs_map:
                    docs_map[filename] = {
                        "filename": filename,
                        "mimetype": (
                            payload.get("filetype")
                            or payload.get("mimetype")
                            or payload.get("file_type")
                            or origin.get("filetype")
                            or payload.get("file_format")
                            or origin.get("mimetype")
                        ),
                        "file_hash": payload.get("file_hash")
                        or origin.get("file_hash"),
                        "node_count": 0,
                        "pages": set(),
                        "max_rows": 0,
                        "max_duration": 0.0,
                        "entity_types": set(),
                    }

                entry = docs_map[filename]
                entry["node_count"] += 1

                # Extract entities from payload
                ents = payload.get("entities") or []
                if isinstance(ents, list):
                    for e in ents:
                        if isinstance(e, dict):
                            t = e.get("type", e.get("label"))
                            if t:
                                entry["entity_types"].add(t)

                page = (
                    payload.get("page")
                    or payload.get("page_number")
                    or origin.get("page_no")
                )

                # Try getting page from doc_items (Docling structure)
                if page is None:
                    doc_items = payload.get("doc_items")
                    if isinstance(doc_items, list):
                        for item in doc_items:
                            if isinstance(item, dict):
                                prov = item.get("prov")
                                if isinstance(prov, list):
                                    for p in prov:
                                        if isinstance(p, dict) and "page_no" in p:
                                            page = p["page_no"]
                                            break
                            if page is not None:
                                break

                if page is not None:
                    try:
                        entry["pages"].add(int(page))
                    except (ValueError, TypeError):
                        entry["pages"].add(page)

                # Table rows logic
                table_info = payload.get("table")
                if isinstance(table_info, dict):
                    rows = table_info.get("n_rows")
                    if isinstance(rows, (int, float)):
                        entry["max_rows"] = max(entry["max_rows"], int(rows))

                # Transcript duration logic (Nextext segment end timestamps).
                end_sec = payload.get("end_seconds") or (
                    payload.get("extra_metadata") or {}
                ).get("end_seconds")
                if isinstance(end_sec, (int, float)):
                    entry["max_duration"] = max(entry["max_duration"], float(end_sec))

            if offset is None:
                break

        results = []
        for _, data in docs_map.items():
            data["page_count"] = len(data.pop("pages"))
            data["entity_types"] = sorted(list(data.get("entity_types", set())))
            if "entity_types" in data and isinstance(data["entity_types"], set):
                # Fallback if get didn't return set but pop of set or something (redundant with line above but safer)
                pass

            if data["max_rows"] == 0:
                del data["max_rows"]
            if data["max_duration"] == 0.0:
                del data["max_duration"]
            results.append(data)

        return sorted(results, key=lambda x: str(x["filename"]))

    def get_collection_ner(self, refresh: bool = False) -> list[dict[str, Any]]:
        """Fetch all nodes from the current collection and return their NER metadata.

        Args:
            refresh (bool): If ``True``, bypass in-memory NER cache and re-fetch from Qdrant.

        Returns:
            list[dict[str, Any]]: A list of source metadata dictionaries containing NER data.
        """
        if not self.qdrant_collection:
            return []

        if refresh:
            self._invalidate_ner_cache(self.qdrant_collection)

        if self.ner_sources and not refresh:
            return self.ner_sources

        self.ner_sources = self._load_collection_ner_sources()
        return self.ner_sources

    def get_collection_hate_speech(self) -> list[dict[str, Any]]:
        """Return flagged hate-speech chunks from the selected collection.

        Returns:
            list[dict[str, Any]]: A list of dictionaries containing metadata about hate-speech
            findings, such as chunk ID, text, category, confidence, reason, source reference,
            and page number.
        """
        if not self.qdrant_collection:
            return []

        findings: list[dict[str, Any]] = []
        offset = None
        while True:
            try:
                points, offset = self.qdrant_client.scroll(
                    collection_name=self.qdrant_collection,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as exc:
                logger.error(
                    "Failed to fetch hate-speech rows from '{}': {}",
                    self.qdrant_collection,
                    exc,
                )
                break

            if not points:
                break

            for point in points:
                payload = getattr(point, "payload", None)
                if not isinstance(payload, dict):
                    continue

                detection = payload.get("hate_speech")
                if not isinstance(detection, dict) or not bool(
                    detection.get("hate_speech")
                ):
                    continue

                source = self._source_from_payload(
                    collection=self.qdrant_collection,
                    payload=payload,
                    text_value=str(
                        detection.get("chunk_text")
                        or self._extract_payload_text(payload)
                        or ""
                    ),
                )
                source["chunk_id"] = str(
                    detection.get("chunk_id")
                    or payload.get("node_id")
                    or payload.get("id_")
                    or str(getattr(point, "id", "") or "")
                )
                source["chunk_text"] = str(source.get("text") or "")
                source["category"] = str(detection.get("category") or "none")
                source["confidence"] = str(detection.get("confidence") or "low")
                source["reason"] = str(detection.get("reason") or "")
                source["source_ref"] = str(
                    detection.get("source_ref")
                    or source.get("filename")
                    or payload.get("file_path")
                    or ""
                )
                findings.append(source)

            if offset is None:
                break

        findings.sort(key=operator.itemgetter("source_ref", "chunk_id"))
        return findings

    def _get_collection_ner_aggregate(
        self,
        *,
        refresh: bool = False,
        entity_merge_mode: EntityMergeMode = "orthographic",
    ) -> dict[str, Any]:
        """Return cached aggregate NER payload for the active collection.

        Args:
            refresh (bool): If ``True``, recompute aggregate from fresh collection NER rows.
            entity_merge_mode (EntityMergeMode): Entity clustering mode used for derived views.

        Returns:
            dict[str, Any]: Aggregation dictionary for stats/search/graph operations.
        """
        merge_mode = normalize_entity_merge_mode(entity_merge_mode)
        if not self.qdrant_collection:
            return aggregate_ner_sources([], entity_merge_mode=merge_mode)
        if refresh:
            self._invalidate_ner_cache(self.qdrant_collection)

        collection = self.qdrant_collection
        cache_key = (collection, merge_mode)
        if cache_key in self.ner_aggregate_cache:
            return self.ner_aggregate_cache[cache_key]

        sources = self.get_collection_ner(refresh=refresh)
        aggregate = aggregate_ner_sources(sources, entity_merge_mode=merge_mode)
        self.ner_aggregate_cache[cache_key] = aggregate
        return aggregate

    def get_collection_ner_stats(
        self,
        *,
        top_k: int = 15,
        min_mentions: int = 2,
        entity_type: str | None = None,
        include_relations: bool = True,
        refresh: bool = False,
        entity_merge_mode: EntityMergeMode = "orthographic",
    ) -> dict[str, Any]:
        """Return collection-wide NER statistics for dashboard and analysis views.

        Args:
            top_k (int): Maximum number of top entities/relations to include.
            min_mentions (int): Minimum mention count for ranked outputs.
            entity_type (str | None): Optional case-insensitive entity-type filter.
            include_relations (bool): Whether relation aggregates are included.
            refresh (bool): If ``True``, recompute from fresh collection data.

        Returns:
            dict[str, Any]: NER stats payload.
        """
        aggregate = self._get_collection_ner_aggregate(
            refresh=refresh,
            entity_merge_mode=entity_merge_mode,
        )
        return build_ner_stats(
            aggregate,
            top_k=max(1, int(top_k)),
            min_mentions=max(1, int(min_mentions)),
            entity_type=entity_type,
            include_relations=bool(include_relations),
        )

    def search_collection_ner_entities(
        self,
        *,
        q: str = "",
        entity_type: str | None = None,
        limit: int = 100,
        refresh: bool = False,
        entity_merge_mode: EntityMergeMode = "orthographic",
    ) -> list[dict[str, Any]]:
        """Search canonicalized entities across the selected collection.

        Args:
            q (str): Case-insensitive text query applied to entity names.
            entity_type (str | None): Optional case-insensitive type filter.
            limit (int): Maximum number of entities to return.
            refresh (bool): If ``True``, recompute from fresh collection data.

        Returns:
            list[dict[str, Any]]: Search result rows sorted by mention frequency.
        """
        aggregate = self._get_collection_ner_aggregate(
            refresh=refresh,
            entity_merge_mode=entity_merge_mode,
        )
        return search_entities(
            aggregate,
            q=q,
            entity_type=entity_type,
            limit=max(1, int(limit)),
        )

    def get_collection_ner_graph(
        self,
        *,
        top_k_nodes: int = 100,
        min_edge_weight: int = 1,
        refresh: bool = False,
        entity_merge_mode: EntityMergeMode = "orthographic",
    ) -> dict[str, Any]:
        """Build a derived NER graph for the selected collection.

        Args:
            top_k_nodes (int): Maximum number of highest-mention entity nodes to include.
            min_edge_weight (int): Minimum edge weight threshold.
            refresh (bool): If ``True``, recompute graph from fresh collection data.

        Returns:
            dict[str, Any]: Graph payload containing ``nodes``, ``edges``, and ``meta``.
        """
        if not self.qdrant_collection:
            return {
                "nodes": [],
                "edges": [],
                "meta": {"node_count": 0, "edge_count": 0},
            }
        if refresh:
            self._invalidate_ner_cache(self.qdrant_collection)

        merge_mode = normalize_entity_merge_mode(entity_merge_mode)
        cache_key = (
            self.qdrant_collection,
            merge_mode,
            max(1, int(top_k_nodes)),
            max(1, int(min_edge_weight)),
        )
        if cache_key in self.ner_graph_cache:
            return self.ner_graph_cache[cache_key]

        aggregate = self._get_collection_ner_aggregate(
            refresh=refresh,
            entity_merge_mode=merge_mode,
        )
        graph = build_entity_graph(
            aggregate,
            top_k_nodes=max(1, int(top_k_nodes)),
            min_edge_weight=max(1, int(min_edge_weight)),
        )
        self.ner_graph_cache[cache_key] = graph
        return graph

    def get_collection_ner_graph_neighbors(
        self,
        *,
        entity: str,
        hops: int = 1,
        top_k_nodes: int = 100,
        min_edge_weight: int = 1,
        refresh: bool = False,
        entity_merge_mode: EntityMergeMode = "orthographic",
    ) -> dict[str, Any]:
        """Return a local graph neighborhood around a specific entity.

        Args:
            entity (str): Entity text or canonical node id.
            hops (int): Number of graph hops to traverse.
            top_k_nodes (int): Graph node cap used to build the base graph.
            min_edge_weight (int): Graph edge threshold used to build the base graph.
            refresh (bool): If ``True``, recompute graph from fresh collection data.

        Returns:
            dict[str, Any]: Neighborhood payload with ``center`` and ``neighbors``.
        """
        graph = self.get_collection_ner_graph(
            top_k_nodes=top_k_nodes,
            min_edge_weight=min_edge_weight,
            refresh=refresh,
            entity_merge_mode=entity_merge_mode,
        )
        return graph_neighbors(graph, entity=entity, hops=max(1, int(hops)))

    def unload_models(self) -> None:
        """Unload models to free up memory.

        Releases the lazily-loaded embed / text / post-retrieval-text /
        reranker / image-ingestion services, invalidates the NER cache,
        and drops the captured ``dir_reader`` handle so Python's
        ref-count collector can reclaim it immediately instead of
        waiting for a later cycle. Pure null-and-``gc.collect``
        semantics; no platform-specific allocator tricks are invoked.
        """
        self._embed_model = None
        self._text_model = None
        self._post_retrieval_text_model = None
        self._reranker = None
        self._image_ingestion_service = None
        self._invalidate_ner_cache()

        self.dir_reader = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Models unloaded and memory cleared.")
