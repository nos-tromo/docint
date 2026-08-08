"""Core RAG engine, ingestion, retrieval, and collection management."""

from __future__ import annotations

import copy
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
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from typing_extensions import override

if TYPE_CHECKING:
    from docint.agents.types import PriorTurn

# isort: off
# Import env_cfg BEFORE any third-party libraries so that HF_HUB_OFFLINE and
# TRANSFORMERS_OFFLINE env vars are set before huggingface_hub caches them.
from docint.utils.env_cfg import (
    EmbeddingConfig,
    GraphRAGConfig,
    EmbedClientConfig,
    HostConfig,
    IngestionConfig,
    NERConfig,
    ModelConfig,
    OpenAIConfig,
    PathConfig,
    RerankClientConfig,
    RetrievalConfig,
    SessionConfig,
    SparseClientConfig,
    SummaryConfig,
    load_embed_client_env,
    load_embedding_env,
    load_graphrag_env,
    load_hate_speech_env,
    load_host_env,
    load_ingestion_env,
    load_language_env,
    load_model_env,
    load_ner_env,
    load_openai_env,
    load_path_env,
    load_principal_env,
    load_rerank_client_env,
    load_resolution_env,
    load_retrieval_env,
    load_session_env,
    load_sparse_client_env,
    load_summary_env,
    resolve_enable_hybrid,
)
from docint.utils.translate_client import translate as translate_text
# isort: on

from llama_index.core import (
    Response,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import BaseSynthesizer, CompactAndRefine, Refine
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core.retrievers import BaseRetriever
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
from llama_index.vector_stores.qdrant import QdrantVectorStore
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models

# Names re-exported for test monkey-patching. pyrefly treats these as
# private re-exports without an explicit ``__all__``,
# so list every test-reachable third-party symbol here.
__all__ = [
    "RAG",
    "EmptyIngestionError",
    "QueryBundle",
    "ResponseMode",
    "RetrieverQueryEngine",
    "VectorStoreQueryMode",
    "logger",
    "qdrant_models",
    "urllib",
]
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from docint.core.collection_overview import summarize_document_types
from docint.core.entities.resolution import (
    ResolutionSummary,
    SurfaceMention,
    normalize_surface,
    resolve_collection,
)
from docint.core.entities.store import EntityStore
from docint.core.ingest.images_service import ImageIngestionService
from docint.core.ingest.ingestion_pipeline import DocumentIngestionPipeline
from docint.core.ingest.streaming_executor import overlapped
from docint.core.ner import (
    EntityMergeMode,
    aggregate_ner_sources,
    build_entity_graph,
    build_ner_stats,
    graph_neighbors,
    match_entity_text,
    normalize_entities,
    normalize_entity_merge_mode,
    search_entities,
)
from docint.core.readers.documents import CorePDFPipelineReader
from docint.core.retrieval_filters import matches_metadata_filters, merge_qdrant_filters
from docint.core.search.fulltext import build_search_filter, parse_keywords
from docint.core.search.index import (
    ensure_search_index,
    image_companion_name,
    search_index_status,
    write_search_text,
)
from docint.core.state.collection_owner_manager import CollectionOwnerManager
from docint.core.state.report_manager import ReportManager
from docint.core.state.session_manager import SessionManager
from docint.core.storage.ingest_manifest import (
    IngestManifest,
    NullIngestManifest,
)
from docint.core.storage.scroll import iter_scroll
from docint.core.storage.sources import stage_sources_to_qdrant
from docint.core.storage.sqlite_kvstore import SQLiteKVStore
from docint.core.storage.utils import build_quantization_config, qdrant_collection_exists
from docint.core.summary.tree import MapCache, TreeSummarizer, UnitChunk
from docint.core.summary.units import MapUnit, partition_units, payload_text
from docint.utils.batching import chunk_nodes
from docint.utils.cursor import decode_cursor, encode_cursor
from docint.utils.embed_chunking import (
    effective_budget,
    estimate_tokens,
    fits_budget,
    resplit_nodes_for_embedding,
)
from docint.utils.embedding_tokenizer import build_embedding_token_counter
from docint.utils.llm_sanitize import strip_reasoning
from docint.utils.openai_cfg import (
    BudgetedOpenAIEmbedding,
    EmbeddingInputTooLongError,
    LocalOpenAI,
    get_openai_reasoning_effort,
)
from docint.utils.reference_metadata import REFERENCE_METADATA_FIELDS
from docint.utils.retry import (
    aretry_with_backoff,
    is_transient_qdrant_error,
    retry_with_backoff,
)

SUMMARY_CACHE_NAMESPACE = "docint_summary_cache_v1"
SUMMARY_CACHE_PAYLOAD_KEY = "summary_payload"
SUMMARY_CACHE_REVISION_KEY = "summary_revision"
# KV namespace for the tree summarizer's per-unit map cache (one entry per
# MapUnit.unit_key), distinct from SUMMARY_CACHE_NAMESPACE which holds the
# single final synthesized payload.
SUMMARY_MAP_CACHE_NAMESPACE = "docint_summary_map_cache_v1"
HIDDEN_COLLECTION_SUFFIXES: tuple[str, ...] = ("_images", "_dockv", "_entities")

# Marks a retrieved node as coming from the image lane. Set on the node
# metadata by ``RAG._retrieve_image_nodes`` and read by
# ``ImageRelevanceFloorPostprocessor``, which applies a relevance floor to
# image captions only. Prefixed like the other internal markers so it is
# excluded from the prompt (it is absent from ``LLM_VISIBLE_METADATA_KEYS``)
# and ignored by ``_source_from_payload``.
IMAGE_LANE_METADATA_KEY = "docint_image_lane"

# Fallback for ``IMAGE_RERANK_MIN_SCORE`` when the image service (which owns
# the loaded config) has not been constructed yet. Mirrors the default in
# ``env_cfg.load_image_ingestion_config``.
DEFAULT_IMAGE_RERANK_MIN_SCORE = 0.05

# Fallback for ``IMAGE_RETRIEVE_TOP_K`` under the same condition — how many
# CLIP candidates the image lane draws before the rerank ranks them against
# the text hits.
DEFAULT_IMAGE_RETRIEVE_TOP_K = 5

# Fallback tie-break preamble used when the locale prompt file is absent.
# The canonical templates live in ``prompts/{en,de}/entity_tiebreak.txt``.
DEFAULT_ENTITY_TIEBREAK_PROMPT = (
    "You are resolving an entity reference to a canonical entity.\n"
    "Reply with ONLY the id of the candidate that refers to the same "
    "real-world entity as the surface form, or NONE if none of them do."
)

# Confidence values stored on hate-speech findings, ordered weakest → strongest.
_HATE_SPEECH_CONFIDENCE_ORDER: dict[str, int] = {"low": 0, "medium": 1, "high": 2}


def _compact_entity_form(text: str) -> str:
    r"""Build the compact lookup form used by the frontend entity matcher.

    Mirrors the compact-lookup form the SPA's entity matcher uses:
    keep unicode letters and digits, drop everything else, lowercase the result.
    Python's :meth:`str.isalnum` is unicode-aware so this matches the
    ``[^\p{L}\p{N}]+`` regex used in the SPA.
    """
    return "".join(ch for ch in str(text or "").lower() if ch.isalnum())


def _filter_sources_by_entity(
    sources: list[dict[str, Any]],
    *,
    target_text: str,
    target_type: str,
) -> list[dict[str, Any]]:
    """Return only sources whose entity list contains the target entity.

    Mirrors ``sourceContainsEntity`` from the SPA: matches on lowercase text
    or compact-lookup form within the same entity type, so callers see the
    same finding set the UI computed client-side before pagination existed.
    """
    target_text_lower = str(target_text or "").strip().lower()
    if not target_text_lower:
        return list(sources)

    target_compact = _compact_entity_form(target_text_lower)
    target_type_lower = str(target_type or "").strip().lower()

    matches: list[dict[str, Any]] = []
    for source in sources:
        candidates = source.get("entities") or []
        if not isinstance(candidates, list):
            continue
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            cand_text = str(cand.get("text") or "").strip().lower()
            if not cand_text:
                continue
            cand_type = str(cand.get("type") or "").strip().lower()
            if target_type_lower and cand_type and cand_type != target_type_lower:
                continue
            if cand_text == target_text_lower:
                matches.append(source)
                break
            if target_compact and _compact_entity_form(cand_text) == target_compact:
                matches.append(source)
                break
    return matches


def _filter_sources_by_surfaces(
    sources: list[dict[str, Any]],
    *,
    surfaces: set[str],
    target_type: str,
) -> list[dict[str, Any]]:
    """Return sources whose entity list contains any of several surface forms.

    The multi-surface variant of :func:`_filter_sources_by_entity`, used for
    resolved drill-down: a canonical entity's siblings (e.g. ``"US"`` plus
    ``"United States"``) are all accepted, so no mention rows are lost. Matching
    is case-insensitive on text or compact-lookup form within the same type.

    Args:
        sources (list[dict[str, Any]]): Candidate source rows.
        surfaces (set[str]): Accepted surface forms (any case).
        target_type (str): Entity type/label constraint (empty = unconstrained).

    Returns:
        list[dict[str, Any]]: Sources mentioning at least one accepted surface.
    """
    surface_set = {str(s or "").strip().lower() for s in surfaces if str(s or "").strip()}
    if not surface_set:
        return list(sources)
    compact_set = {_compact_entity_form(s) for s in surface_set}
    compact_set.discard("")
    target_type_lower = str(target_type or "").strip().lower()

    matches: list[dict[str, Any]] = []
    for source in sources:
        candidates = source.get("entities") or []
        if not isinstance(candidates, list):
            continue
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            cand_text = str(cand.get("text") or "").strip().lower()
            if not cand_text:
                continue
            cand_type = str(cand.get("type") or "").strip().lower()
            if target_type_lower and cand_type and cand_type != target_type_lower:
                continue
            if cand_text in surface_set or _compact_entity_form(cand_text) in compact_set:
                matches.append(source)
                break
    return matches


def _filter_hate_speech(
    findings: list[dict[str, Any]],
    *,
    category: str | None,
    min_confidence: str | None,
) -> list[dict[str, Any]]:
    """Apply category and confidence filters to hate-speech findings."""
    category_lower = (category or "").strip().lower() or None
    min_rank = _HATE_SPEECH_CONFIDENCE_ORDER.get((min_confidence or "").strip().lower())

    filtered: list[dict[str, Any]] = []
    for row in findings:
        if category_lower and str(row.get("category") or "").strip().lower() != category_lower:
            continue
        if min_rank is not None:
            row_rank = _HATE_SPEECH_CONFIDENCE_ORDER.get(
                str(row.get("confidence") or "").strip().lower(),
                _HATE_SPEECH_CONFIDENCE_ORDER["low"],
            )
            if row_rank < min_rank:
                continue
        filtered.append(row)
    return filtered


# Pinned to LlamaIndex's QdrantVectorStore DEFAULT_DENSE_VECTOR_NAME /
# DEFAULT_SPARSE_VECTOR_NAME so a collection we pre-create has the same
# named-vector schema the runtime QdrantVectorStore will later upsert
# into. We replicate the schema here rather than instantiating
# QdrantVectorStore because that class only creates the Qdrant collection
# lazily, from its ``add()`` method, the first time nodes are written —
# building one in ``create_collection_if_missing`` would not pre-create
# anything for the UI to show before ingestion runs (see that method).
# Sparse encoding is a remote HTTP call now (``RemoteSparseEncoder``),
# not a local fastembed model, so there is no local model-loading cost
# to weigh either way.
QDRANT_DENSE_VECTOR_NAME = "text-dense"
QDRANT_SPARSE_VECTOR_NAME = "text-sparse-new"
# Startup reachability probe against Qdrant's /readyz. Kept short: the probe
# only exists to make a mis-wired deployment visible in the logs immediately,
# and a DNS/connect failure resolves well within this bound.
QDRANT_PROBE_TIMEOUT_S = 5.0


def _quantization_matches(
    current: object,
    target: qdrant_models.TurboQuantization,
) -> bool:
    """Return whether *current* already equals the target TurboQuant config.

    A live server may report ``always_ram=False`` where the target carries
    ``None`` (unset), so those two are treated as equal.

    Args:
        current: The collection's live quantization config.
        target: The TurboQuant config resolved from the environment.

    Returns:
        True when *current* is a TurboQuant config with the same bit width
        and effective RAM-pinning as *target*.
    """
    if not isinstance(current, qdrant_models.TurboQuantization):
        return False
    return current.turbo.bits == target.turbo.bits and bool(current.turbo.always_ram) == bool(target.turbo.always_ram)


# Metadata keys that stay visible to the chat LLM when the synthesizer
# renders ``node.get_content(MetadataMode.LLM)``. Everything *not* in this
# set is added to each emitted node's ``excluded_llm_metadata_keys`` so
# the prompt only carries the matched text plus a tiny set of grounding
# hints. Downstream consumers — citation rendering
# (``_source_from_node_with_score``), the UI analysis section, and
# graph-building — read ``node.metadata`` directly as a dict and are
# unaffected; this whitelist only gates prompt assembly.
#
# Each key is kept because it provides a *locator* the LLM cannot infer
# from ``node.text`` alone. Mapping by source type:
#
# * PDFs / page-level pipeline: ``filename``, ``origin``, ``page``,
#   ``page_number``.
# * Markdown / plain text: ``filename``, ``origin``.
# * Nextext transcripts: ``start_ts``, ``end_ts``, ``speaker``,
#   ``sentence_index`` (position within the transcript).
# * CSV / XLSX / Parquet tables: ``table`` (nested dict holding
#   ``row_index`` / ``original_row_index``, along with a short list of
#   column names — bounded to ~hundreds of characters per hit).
# * Social / reference-mapped tables: ``reference_metadata`` (compact
#   structured locator: ``type``, ``uuid``, ``author``, ``network``, etc.
#   Designed to be prompt-friendly by construction).
# * All readers: ``docint_doc_kind`` — labels the source shape
#   (``transcript_segment``, ``table_row``, ...) so the LLM frames
#   citations correctly.
# * All readers: ``citation_index`` — the snippet's number in this
#   answer's evidence set, stamped by
#   :class:`CitationNumberingPostprocessor` so the answer's "source 3"
#   and the chat window's third card are the same chunk.
#
# Explicitly excluded via absence: ``entities`` / ``relations`` (NER
# output, can exceed 60 KB), per-column row dumps in ``tables.py``
# (``metadata[col] = row_dict.get(col, "")`` — redundant with
# ``node.text``), ``file_hash``, ``hier.*``, ``embedding_split``,
# ``split_part_*``, ``whisper_task``, ``source_file_hash``,
# ``llm_description`` / ``llm_tags`` (image caption / tagging output —
# bulky), pipeline artefacts, and our own
# ``parent_context_windowed`` / ``parent_full_chars`` / ``window_chars``
# debug markers. Any new metadata key added by a future reader defaults
# to excluded — safer than accidentally reintroducing the overflow.
LLM_VISIBLE_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "citation_index",
        "filename",
        "origin",
        "page",
        "page_number",
        "start_ts",
        "end_ts",
        "speaker",
        "sentence_index",
        "table",
        "reference_metadata",
        "docint_doc_kind",
    }
)

# Sub-keys permitted inside the whitelisted ``origin`` dict when it
# reaches the chat LLM. Readers today only populate ``filename`` /
# ``mimetype`` / ``filetype`` / ``page_number`` / ``file_hash`` here,
# but a future reader could add deployment-internal identifiers such as
# absolute ``file_path`` strings with usernames or tenant IDs. Filtering
# on emission means such additions cannot silently leak into the LLM
# prompt (or, via the provider, into external log storage).
LLM_VISIBLE_ORIGIN_SUBKEYS: frozenset[str] = frozenset({"filename", "mimetype", "filetype", "page_number"})

# Hard ceiling for any single metadata string that reaches the LLM.
# Legitimate locators (filenames, timestamps, speaker names, ISO
# timestamps, short IDs) fit well under 1 KB; anything larger is almost
# certainly a bulky payload that slipped past the whitelist (e.g. a
# column named ``description`` that was added to ``reference_mapping``
# by a social-table profile and carries row prose) or an attacker-
# controlled prompt-injection payload embedded in an ingested document
# field. Clamping plus control-character stripping reduces both risks.
LLM_METADATA_VALUE_MAX_CHARS: int = 1024


def _sanitize_metadata_value_for_llm(value: Any) -> Any:
    r"""Clamp length and strip control characters from a metadata leaf.

    Recurses through ``dict`` / ``list`` / ``tuple`` so nested locators
    (``origin``, ``table``, ``reference_metadata``) are scrubbed in
    place. Non-string leaves (int, float, bool, None) pass through
    unchanged. Strings longer than :data:`LLM_METADATA_VALUE_MAX_CHARS`
    are truncated with a visible ``… [truncated]`` marker so the LLM
    does not silently see fabricated context; newline / carriage-return
    / tab runs collapse to a single space so attacker-controlled
    formatting cannot forge "```", headers, or fake chat-role lines
    inside ``{metadata_str}\\n\\n{content}``.

    Args:
        value: A metadata value of any JSON-compatible type.

    Returns:
        The scrubbed value. Containers are rebuilt as plain ``dict`` /
        ``list`` instances; strings may be shortened.
    """
    if isinstance(value, str):
        cleaned = re.sub(r"[\r\n\t]+", " ", value)
        if len(cleaned) > LLM_METADATA_VALUE_MAX_CHARS:
            cleaned = cleaned[:LLM_METADATA_VALUE_MAX_CHARS] + "… [truncated]"
        return cleaned
    if isinstance(value, dict):
        return {k: _sanitize_metadata_value_for_llm(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_metadata_value_for_llm(item) for item in value]
    return value


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
DEFAULT_RETRIEVAL_REWRITE_PROMPT = (
    "Rewrite the user's latest message into a standalone retrieval query "
    "suitable for vector search.\n\n"
    "Rules:\n"
    "- Resolve all pronouns and ellipses using the conversation context.\n"
    "- If the user is asking to elaborate, clarify, or expand on a previous "
    'answer ("tell me more", "please elaborate", "I didn\'t understand X", '
    '"what do you mean by ..."), EXTRACT the specific entities, names, '
    "organizations, claims, or topics they are referring to from the most "
    "recent assistant turn and inline them in the rewritten query. Example: "
    '"Tell me more about the UN references" becomes a query that names the '
    "specific UN bodies/resolutions/claims the prior assistant turn "
    "mentioned.\n"
    "- Do not answer the question; only produce search terms.\n"
    "- Do not invent facts that are absent from both the user message and the "
    "conversation context.\n"
    "- Return ONLY the rewritten query, no preamble.\n\n"
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
    "You are answering a question from retrieved evidence in an ongoing "
    "conversation.\n\n"
    "Conversation continuity:\n"
    "{prior_turn_context}\n\n"
    "Retrieved context snippets:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Current question: {query_str}\n\n"
    "Instructions:\n"
    "- Treat each retrieved snippet as a distinct source chunk; do not blend "
    "claims across chunks unless the overlap is explicit.\n"
    "- Each snippet is labelled with a `citation_index`. Refer to a snippet "
    "only by that number, in square brackets (e.g. [3]); never number the "
    "snippets yourself.\n"
    "- If the current question asks to elaborate on, clarify, or expand on the "
    "prior assistant turn, restate and expand the specific claims from that "
    "prior turn that the user is asking about. Use the retrieved snippets to "
    "corroborate or add supporting detail. The prior assistant turn was itself "
    "sourced; you may quote and elaborate on it.\n"
    "- If snippets conflict, say so explicitly.\n"
    "- Only respond that evidence is insufficient when BOTH the retrieved "
    "snippets AND the prior assistant turn lack the requested information. In "
    "that case, name the specific aspect that is missing.\n"
    "- Preserve source-specific metadata such as author, network, timestamp, "
    "page, or row when it matters.\n\n"
    "Grounded answer:"
)
DEFAULT_GROUNDED_REFINE_PROMPT = (
    "You are refining an answer from retrieved evidence in an ongoing "
    "conversation.\n\n"
    "Conversation continuity:\n"
    "{prior_turn_context}\n\n"
    "Original question: {query_str}\n"
    "Current answer: {existing_answer}\n"
    "New context snippet(s):\n"
    "---------------------\n"
    "{context_msg}\n"
    "---------------------\n"
    "Update the answer only when the new evidence materially improves or "
    "corrects it. Keep source-specific claims distinct. If the original "
    "question is an elaboration of the prior assistant turn, you may quote "
    "and expand on the prior turn's claims when the new snippets corroborate "
    "them. If the new context is not useful, return the current answer "
    "unchanged. Each snippet is labelled with a `citation_index`; refer to a "
    "snippet only by that number, in square brackets (e.g. [3]), and leave "
    "the numbers already used in the current answer as they are.\n"
    "Refined grounded answer:"
)
DEFAULT_GROUNDED_COLLECTION_SUMMARY_PROMPT = (
    "You are producing a grounded collection summary.\n"
    "Use only the evidence briefs below. If evidence is insufficient, state that "
    "explicitly.\n"
    "Include cross-document themes, notable differences or outliers, and concrete "
    "findings.\n"
    "Do not introduce claims unsupported by the evidence briefs.\n\n"
    "Coverage unit: {coverage_unit}\n"
    "Coverage ratio: {coverage_ratio}\n"
    "Coverage target: {coverage_target}\n"
    "Uncovered documents: {uncovered_text}\n\n"
    "Style instructions:\n{style_prompt}\n\n"
    "Evidence briefs:\n{evidence_block}\n"
)
DEFAULT_SUMMARY_MAP_PROMPT = (
    "You are summarizing one unit of source material from a larger document "
    "collection.\n"
    "Unit: {label}\n\n"
    "Write a dense, factual summary of the numbered excerpts below (5-10 "
    "sentences). Capture the main topics, concrete claims and findings, named "
    "people, organizations and places, dates, and notable outliers or "
    "disagreements. Do not invent facts and do not editorialize. If the "
    "excerpts are unintelligible, say so.\n\n"
    "After the summary, output one final line exactly in this form, naming "
    "the one or two excerpt numbers that best represent this unit:\n"
    "EVIDENCE_INDICES: 1,2\n\n"
    "Excerpts:\n{chunk_block}"
)
DEFAULT_SUMMARY_FOLD_PROMPT = (
    "You are combining partial summaries of one document collection into a "
    "single intermediate summary. Preserve concrete facts, named entities, "
    "dates, recurring themes, disagreements, and outliers. Do not invent "
    "facts and do not generalize away specifics. Keep the result under 20 "
    "sentences.\n\n"
    "Partial summaries:\n{summaries_block}"
)


class _KVMapCache:
    """Adapts a collection's ``SQLiteKVStore`` to the tree summarizer's :class:`MapCache` protocol.

    Wraps ``kv_store`` so per-unit map results persist under
    ``SUMMARY_MAP_CACHE_NAMESPACE``, keyed by ``unit_key``. The stored
    validator concatenates the caller-supplied unit fingerprint with a
    ``validator_suffix`` fixed at construction (the summary prompt
    fingerprint plus the chat model id), so a prompt or model change
    invalidates every entry without touching the unit fingerprints
    themselves — :mod:`docint.core.summary.tree` keeps passing only the
    unit's own fingerprint.

    As a side effect, every resolved unit (cache hit via :meth:`get`, or a
    fresh map result recorded via :meth:`put`) is recorded in
    :attr:`covered_keys`. :meth:`RAG.build_tree_summary` reads this set from
    *inside* the synthesis-prompt closure — the one place that needs to know
    which units are covered before :class:`TreeSummarizer.build` returns.
    """

    def __init__(self, kv_store: BaseKVStore | None, *, validator_suffix: str) -> None:
        """Configure the adapter.

        Args:
            kv_store: The collection's KV store, or ``None`` when persistence
                is unavailable — the adapter then degrades to an in-memory,
                always-miss cache so callers (and :attr:`covered_keys`
                tracking) do not need a separate code path.
            validator_suffix: ``"{prompt_fingerprint}|{model_name}"``,
                appended to every unit fingerprint to form the full
                validator string.
        """
        self._kv_store = kv_store
        self._validator_suffix = validator_suffix
        self.covered_keys: set[str] = set()

    def get(self, unit_key: str, validator: str) -> dict[str, Any] | None:
        """Return a cached map result for ``unit_key`` iff its validator matches.

        Args:
            unit_key: The unit's stable identity.
            validator: The unit's content fingerprint (not yet combined with
                the prompt/model suffix).

        Returns:
            dict[str, Any] | None: ``{"summary": str, "evidence_ids": list[str]}``
            on a hit, else ``None`` (including on any storage exception).
        """
        if self._kv_store is None:
            return None
        try:
            entry = self._kv_store.get(unit_key, collection=SUMMARY_MAP_CACHE_NAMESPACE)
        except Exception as exc:
            logger.warning("Map cache get failed for unit '{}': {}", unit_key, exc)
            return None
        if not isinstance(entry, dict):
            return None
        expected = f"{validator}|{self._validator_suffix}"
        if str(entry.get("validator") or "") != expected:
            return None
        self.covered_keys.add(unit_key)
        return {
            "summary": str(entry.get("summary") or ""),
            "evidence_ids": list(entry.get("evidence_ids") or []),
        }

    def put(self, unit_key: str, validator: str, entry: dict[str, Any]) -> None:
        """Store a fresh map result for ``unit_key``.

        Args:
            unit_key: The unit's stable identity.
            validator: The unit's content fingerprint (not yet combined with
                the prompt/model suffix).
            entry: ``{"summary": str, "evidence_ids": list[str]}``.
        """
        self.covered_keys.add(unit_key)
        if self._kv_store is None:
            return
        try:
            self._kv_store.put(
                unit_key,
                {
                    "validator": f"{validator}|{self._validator_suffix}",
                    "summary": entry.get("summary"),
                    "evidence_ids": entry.get("evidence_ids") or [],
                },
                collection=SUMMARY_MAP_CACHE_NAMESPACE,
            )
        except Exception as exc:
            logger.warning("Map cache put failed for unit '{}': {}", unit_key, exc)

    def all_keys(self) -> list[str]:
        """Return every unit key currently persisted in the map cache.

        Returns:
            list[str]: Cached unit keys, or ``[]`` when persistence is
            unavailable or the lookup fails.
        """
        if self._kv_store is None:
            return []
        try:
            return list(self._kv_store.get_all(collection=SUMMARY_MAP_CACHE_NAMESPACE).keys())
        except Exception as exc:
            logger.warning("Map cache all_keys() failed: {}", exc)
            return []

    def delete(self, key: str) -> None:
        """Remove one entry from the persistent map cache.

        Args:
            key: The unit key to remove.
        """
        if self._kv_store is None:
            return
        try:
            self._kv_store.delete(key, collection=SUMMARY_MAP_CACHE_NAMESPACE)
        except Exception as exc:
            logger.warning("Map cache delete failed for unit '{}': {}", key, exc)


def _extract_node_file_hashes(nodes: list[BaseNode]) -> set[str]:
    """Collect unique ``file_hash`` values from a list of ingestion nodes.

    Used by :meth:`RAG.ingest_docs` and :meth:`RAG.asingest_docs` to
    drive per-file manifest hooks without requiring the streaming
    pipeline to surface in-flight file hashes through its yield shape.
    Each ingestion node carries its source document's ``file_hash`` in
    metadata via :meth:`DocumentIngestionPipeline._ensure_file_hashes`.

    Args:
        nodes: Ingestion nodes about to be persisted.

    Returns:
        Set of unique non-empty file-hash strings observed in
        ``node.metadata['file_hash']``.
    """
    hashes: set[str] = set()
    for node in nodes:
        metadata = getattr(node, "metadata", None) or {}
        value = metadata.get("file_hash")
        if isinstance(value, str) and value.strip():
            hashes.add(value.strip())
    return hashes


def _attach_posting_group(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tag each source with its posting UUID group key when linkable.

    Reads the posting UUID from a top-level ``posting_uuid`` or from
    ``reference_metadata.uuid``/``reference_metadata.posting_uuid`` and writes
    it as ``posting_group`` so the UI can render a post and its media as one
    entity. Sources without a link are left untouched.

    Args:
        sources (list[dict[str, Any]]): Normalized source dicts.

    Returns:
        list[dict[str, Any]]: The same list, mutated with ``posting_group``.
    """
    for source in sources:
        group = str(source.get("posting_uuid") or "").strip()
        if not group:
            reference_metadata = source.get("reference_metadata")
            if isinstance(reference_metadata, dict):
                group = str(reference_metadata.get("posting_uuid") or reference_metadata.get("uuid") or "").strip()
        if group:
            source["posting_group"] = group
    return sources


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
        super().__init__(message or f"No content was ingested into '{collection_name}'.")


class SocialSourceDiversityPostprocessor(BaseNodePostprocessor):
    """Deduplicate and diversify row-level social/table retrieval results."""

    diversity_limit: int = 2

    @override
    @classmethod
    def class_name(cls) -> str:
        """Return a stable class identifier."""
        return "SocialSourceDiversityPostprocessor"

    @staticmethod
    def _reference_metadata(node: NodeWithScore) -> dict[str, Any]:
        """Extract the reference metadata dict from a retrieved node.

        May be nested under ``reference_metadata`` in the node's metadata, or missing entirely.

        Args:
            node (NodeWithScore): The node from which to extract reference metadata.

        Returns:
            dict[str, Any]: The reference metadata dictionary, or an empty dict if not present.
        """
        metadata = getattr(node, "metadata", {}) or {}
        reference_metadata = metadata.get("reference_metadata")
        if isinstance(reference_metadata, dict):
            return reference_metadata
        return {}

    @staticmethod
    def _identity_key(node: NodeWithScore) -> str:
        """Extract a stable identity key for deduplication from a retrieved node.

        Args:
            node (NodeWithScore): The node from which to extract an identity key.

        Returns:
            str: An identity key derived from text ID, file-hash + row-index (tabular), or a
                normalized text snippet. Empty string if no meaningful identity is available.
        """
        metadata = getattr(node, "metadata", {}) or {}
        reference_metadata = SocialSourceDiversityPostprocessor._reference_metadata(node)
        text_id = str(reference_metadata.get("text_id") or "").strip()
        if text_id:
            return f"text_id:{text_id}"

        file_hash = str(metadata.get("file_hash") or "").strip()
        table_meta = metadata.get("table") or {}
        row_index = table_meta.get("row_index") if isinstance(table_meta, dict) else None
        if file_hash and row_index is not None:
            return f"row:{file_hash}:{row_index}"

        text_value = str(getattr(node, "text", "") or "").strip()
        if text_value:
            normalized = re.sub(r"\s+", " ", text_value).lower()
            return f"text:{normalized[:240]}"

        return ""

    @staticmethod
    def _diversity_bucket(node: NodeWithScore) -> str:
        """Extract an author+time diversity bucket key for limiting near-duplicate results.

        Args:
            node (NodeWithScore): The node from which to extract a diversity bucket key.

        Returns:
            str: A bucket key combining lowercased author and an hour-resolution time bucket.

        Raises:
            ValueError: If the timestamp can't be parsed (indicates unexpected metadata shape).
        """
        metadata = getattr(node, "metadata", {}) or {}
        reference_metadata = SocialSourceDiversityPostprocessor._reference_metadata(node)
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
                time_bucket = parsed.astimezone(UTC).strftime("%Y-%m-%dT%H")
            except ValueError:
                time_bucket = timestamp_raw[:13]
        return f"{author.lower()}::{time_bucket}"

    @override
    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        """Deduplicate by post identity and cap near-duplicate author/time buckets.

        Args:
            nodes (list[NodeWithScore]): The list of retrieved nodes to postprocess.
            query_bundle (QueryBundle | None): Original query bundle for context; not modified.

        Returns:
            list[NodeWithScore]: Nodes after duplicate and bucket-limit filtering.
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


class LinkFollowingPostprocessor(BaseNodePostprocessor):
    """Expand each retrieved post to include its linked media (and vice versa).

    For every hit, resolves the posting UUID (``reference_metadata.uuid`` or a
    top-level ``posting_uuid``) and appends the post's sibling artifacts —
    transcript segments and image/keyframe captions — so the generator sees a
    post and its media as one evidence block. Bounded by ``max_per_post`` and
    deduplicated by node id; triggering is bidirectional (a media hit pulls in
    its post's siblings too).

    Attributes:
        rag: A :class:`RAG` instance exposing ``_fetch_posting_entity_nodes``.
        max_per_post: Maximum number of sibling nodes to append per posting UUID.
    """

    rag: Any
    max_per_post: int = 12

    @override
    @classmethod
    def class_name(cls) -> str:
        """Return a stable class identifier."""
        return "LinkFollowingPostprocessor"

    @staticmethod
    def _posting_uuid(node: NodeWithScore) -> str:
        """Extract the posting UUID link key from a node's metadata.

        Args:
            node (NodeWithScore): The node from which to extract the posting UUID.

        Returns:
            str: The posting UUID if found, otherwise an empty string.
        """
        metadata = getattr(node, "metadata", {}) or {}
        direct = str(metadata.get("posting_uuid") or "").strip()
        if direct:
            return direct
        reference_metadata = metadata.get("reference_metadata")
        if isinstance(reference_metadata, dict):
            return str(reference_metadata.get("posting_uuid") or reference_metadata.get("uuid") or "").strip()
        return ""

    @override
    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        """Append linked sibling artifacts for each retrieved posting.

        Args:
            nodes (list[NodeWithScore]): Retrieved nodes.
            query_bundle (QueryBundle | None): Unused.

        Returns:
            list[NodeWithScore]: Original nodes plus bounded, deduped siblings.
        """
        _ = query_bundle
        present: set[str] = {n.node.node_id for n in nodes}
        additions: list[NodeWithScore] = []
        seen_posts: set[str] = set()
        for node in nodes:
            posting_uuid = self._posting_uuid(node)
            if not posting_uuid or posting_uuid in seen_posts:
                continue
            seen_posts.add(posting_uuid)
            try:
                siblings = self.rag._fetch_posting_entity_nodes(posting_uuid, exclude_node_ids=present)
            except Exception as exc:
                logger.warning("Link-following expansion failed for {}: {}", posting_uuid, exc)
                continue
            for sibling in siblings[: self.max_per_post]:
                sid = sibling.node.node_id
                if sid not in present:
                    present.add(sid)
                    additions.append(sibling)
        return nodes + additions


class ParentContextPostprocessor(BaseNodePostprocessor):
    """Promote fine-grained retrieval hits to their hierarchical parent context.

    When the :class:`~docint.utils.embed_chunking.resplit_nodes_for_embedding`
    path keeps an oversize coarse parent in the docstore, a naive expansion
    would splice the full parent into the chat prompt and can overflow the
    chat context budget. This postprocessor enforces a budget at query time
    via a greedy packer:

    - Hits are iterated in score order (already sorted by the reranker).
    - A parent that fits the remaining budget is emitted verbatim (status quo).
    - A parent that does not fit is emitted as a **windowed slice** centred
      on the matched sub-node text, keeping the parent ``node_id`` so
      citations (keyed on ``node_id``) still resolve.
    - If the sub-node text cannot be located in the parent (e.g.
      whitespace normalization drifted between ingest and query), the
      postprocessor falls back to emitting the sub-node itself rather than
      guessing.

    Legacy callers that construct the postprocessor without the new
    ``usable_tokens`` budget get the pre-budget behavior (emit the full
    parent, no bound) so existing call sites continue to work.

    Attributes:
        docstore: Any docstore object exposing ``get_node(node_id,
            raise_error=False)`` (or ``get_document`` as fallback).
        usable_tokens: Total chat-budget tokens available across all hits
            in a single query. ``0`` disables the packer / windowing
            entirely (legacy behavior).
        per_hit_floor: Minimum window size in tokens when the packer has
            to truncate the last hit. Guards against emitting a
            near-empty window at the tail of the budget.
        char_token_ratio: Characters per token used by the
            :func:`~docint.utils.embed_chunking.estimate_tokens` fallback
            estimator. Matches the embed-side default so the chat-side
            estimate stays consistent.
    """

    docstore: Any
    usable_tokens: int = 0
    per_hit_floor: int = 256
    char_token_ratio: float = 3.5
    budget_enforced: bool = False

    @override
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
            return cast(BaseNode | None, self.docstore.get_node(parent_id, raise_error=False))
        except AttributeError:
            return cast(BaseNode | None, self.docstore.get_document(parent_id, raise_error=False))
        except Exception as exc:
            logger.warning("Failed to load parent node '{}' from docstore: {}", parent_id, exc)
            return None

    @staticmethod
    def _find_match_offset(parent_text: str, sub_text: str) -> int:
        r"""Return the offset of *sub_text* inside *parent_text*, or ``-1``.

        First tries an exact substring search. On miss, whitespace-
        normalizes both strings (``\\s+`` → single space, stripped) while
        tracking each normalized character's original parent offset, so a
        normalized hit can be mapped back to the original parent for
        accurate slicing. Returns ``-1`` if even the normalized search
        misses — callers treat that as a signal to fall back to the
        sub-node rather than guessing.

        Args:
            parent_text: The full parent text loaded from the docstore.
            sub_text: The matched sub-node text whose position inside
                *parent_text* anchors the window.

        Returns:
            The character offset in *parent_text* where *sub_text* (or
            its normalized equivalent) begins, or ``-1`` when the
            sub-node cannot be located.
        """
        if not sub_text or not parent_text:
            return -1
        offset = parent_text.find(sub_text)
        if offset >= 0:
            return offset

        norm_sub = re.sub(r"\s+", " ", sub_text).strip()
        if not norm_sub:
            return -1

        # Build a parallel normalized parent + index map back to the
        # original offsets so we return an accurate slice boundary.
        norm_chars: list[str] = []
        idx_map: list[int] = []
        prev_space = True
        for i, ch in enumerate(parent_text):
            if ch.isspace():
                if not prev_space:
                    norm_chars.append(" ")
                    idx_map.append(i)
                    prev_space = True
            else:
                norm_chars.append(ch)
                idx_map.append(i)
                prev_space = False
        if norm_chars and norm_chars[-1] == " ":
            norm_chars.pop()
            idx_map.pop()
        norm_parent = "".join(norm_chars)

        norm_offset = norm_parent.find(norm_sub)
        if norm_offset < 0:
            return -1
        return idx_map[norm_offset]

    @staticmethod
    def _snap_to_whitespace(text: str, start: int, end: int, scan: int = 80) -> tuple[int, int]:
        """Expand or shrink a window's edges to the nearest whitespace boundary.

        Readable slices should not split words mid-token. The scan is
        bounded so a window that lands deep inside a space-less stretch
        (e.g. base64 blob) still returns in reasonable time.

        Args:
            text: The text the window is being sliced from.
            start: Current start offset (inclusive).
            end: Current end offset (exclusive).
            scan: Maximum characters to scan forward / backward when
                hunting for whitespace.

        Returns:
            A ``(start, end)`` pair snapped to whitespace boundaries.
        """
        if start > 0:
            scan_end = min(len(text), start + scan)
            for i in range(start, scan_end):
                if text[i].isspace():
                    start = i + 1
                    break
        if end < len(text):
            scan_start = max(0, end - scan)
            for i in range(end, scan_start, -1):
                if text[i - 1].isspace():
                    end = i - 1
                    break
        return start, end

    def _window_parent_text(self, parent_text: str, sub_text: str, budget_chars: int) -> str | None:
        """Return a ~``budget_chars``-sized window of *parent_text* around *sub_text*.

        When the sub-node text itself already exceeds ``budget_chars`` —
        possible because sub-nodes are sized to the *embedding* context
        which can exceed the *chat* context — the window starts at the
        sub-node and truncates to the budget so we never exceed it.

        Returns ``None`` when the sub-node text cannot be located in the
        parent (the caller falls back to emitting the sub-node).

        Args:
            parent_text: The full parent text.
            sub_text: The matched sub-node text.
            budget_chars: The character ceiling for the returned window.

        Returns:
            The windowed parent slice, or ``None`` on location miss.
        """
        if budget_chars <= 0 or not parent_text or not sub_text:
            return None
        if len(sub_text) >= budget_chars:
            return sub_text[:budget_chars]

        offset = self._find_match_offset(parent_text, sub_text)
        if offset < 0:
            return None

        sub_end = offset + len(sub_text)
        half = max(0, (budget_chars - len(sub_text)) // 2)
        start = max(0, offset - half)
        end = min(len(parent_text), sub_end + half)
        start, end = self._snap_to_whitespace(parent_text, start, end)
        return parent_text[start:end]

    @staticmethod
    def _emit_with_llm_exclusion(source: BaseNode, ner_fallback: BaseNode | None = None) -> TextNode:
        r"""Clone *source* into a ``TextNode`` whose LLM view hides noisy metadata.

        The synthesiser splices each emitted node into the chat prompt
        via ``get_content(MetadataMode.LLM)``, which renders
        ``"{metadata_str}\\n\\n{content}"``. Unbounded metadata (NER
        entities, per-column row dumps, PDF pipeline artefacts,
        reference-metadata blocks) can balloon the prompt well past
        ``OPENAI_CTX_WINDOW``. Cloning into a new :class:`TextNode` and
        populating ``excluded_llm_metadata_keys`` with every key absent
        from :data:`LLM_VISIBLE_METADATA_KEYS` keeps the LLM view
        minimal while leaving the original ``node.metadata`` dict intact
        for every other consumer (citations, analysis section,
        frontend sources panel).

        The clone is necessary because the docstore-loaded parent may be
        cached across queries; mutating its ``excluded_llm_metadata_keys``
        in place would silently leak this postprocessor's policy into
        unrelated code paths.

        Args:
            source: The node to emit — a docstore-loaded parent, a
                windowed ``TextNode``, or a retrieved sub-node.
            ner_fallback: The retrieved sub-node whose ``entities`` /
                ``relations`` backfill the clone when *source* carries
                none. Parents ingested while PDF NER skipped coarse nodes
                sit entity-less in the docstore; without the carry,
                promoting such a parent strips the entity pills off the
                answer's sources. A parent that has its own NER metadata
                (fresh ingests mirror the full children union) always
                wins over the single matched child's slice.

        Returns:
            A fresh ``TextNode`` preserving the original ``node_id`` and
            metadata dict but hiding non-whitelisted keys from LLM
            serialisation.
        """
        # ``deepcopy`` — we are about to narrow nested dicts (``origin``)
        # and clamp string values on the clone. The docstore may hand us a
        # cached parent node, and mutating its nested dicts in place would
        # leak policy into every future query that touches the same parent.
        metadata = copy.deepcopy(dict(source.metadata or {}))

        if ner_fallback is not None:
            fallback_meta = ner_fallback.metadata or {}
            for key in ("entities", "relations"):
                if not metadata.get(key) and fallback_meta.get(key):
                    metadata[key] = copy.deepcopy(fallback_meta[key])

        # Narrow ``origin`` to a known-safe sub-key set so a future reader
        # that adds deployment-internal paths / tenant IDs / usernames to
        # the nested dict cannot silently leak them into the chat prompt
        # (and therefore into upstream LLM-provider logs).
        raw_origin = metadata.get("origin")
        if isinstance(raw_origin, dict):
            metadata["origin"] = {k: v for k, v in raw_origin.items() if k in LLM_VISIBLE_ORIGIN_SUBKEYS}

        # Clamp + scrub whitelisted metadata values. Protects against a
        # bulky string sliding in via a social-table ``reference_mapping``
        # pointing at a long prose column, and against prompt-injection
        # payloads hidden in ingested content that use newline-heavy
        # formatting to fake structured LLM instructions.
        metadata = {k: _sanitize_metadata_value_for_llm(v) for k, v in metadata.items()}

        # Sorted for deterministic log / test output; llama_index does not
        # require a specific ordering for ``excluded_llm_metadata_keys``.
        excluded = sorted(k for k in metadata if k not in LLM_VISIBLE_METADATA_KEYS)
        # ``MetadataMode.NONE`` is explicit rather than relying on the
        # BaseNode default so we never accidentally pick up a metadata-
        # prefixed string if a future LlamaIndex release changes the
        # default rendering for a ``BaseNode`` subclass.
        clone = TextNode(
            id_=source.node_id,
            text=source.get_content(metadata_mode=MetadataMode.NONE),
            metadata=metadata,
            excluded_llm_metadata_keys=excluded,
            excluded_embed_metadata_keys=list(getattr(source, "excluded_embed_metadata_keys", []) or []),
        )
        return clone

    @override
    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        """Expand reranked hits to their parents with a budget-aware packer.

        Args:
            nodes: Retrieved hits in score order after reranking.
            query_bundle: Unused; required by the postprocessor contract.

        Returns:
            A list of ``NodeWithScore`` whose combined ``node.text``
            token count never exceeds ``self.usable_tokens`` when that
            bound is set. Hits without a parent link, with a missing
            parent, or whose sub-node text cannot be located in the
            loaded parent pass through unchanged so retrieval always
            returns something citable.
        """
        _ = query_bundle
        expanded: list[NodeWithScore] = []
        seen_parent_ids: set[str] = set()
        remaining_tokens = int(self.usable_tokens)
        budget_enforced = bool(self.budget_enforced)
        windowed_hits = 0
        budget_exhausted_hits = 0
        parent_hits = 0

        def _emit(
            base_node: BaseNode,
            score: float | None,
            ner_fallback: BaseNode | None = None,
        ) -> NodeWithScore:
            """Wrap *base_node* in a ``NodeWithScore`` with LLM-view exclusions applied.

            Hides non-whitelisted metadata from the chat prompt while
            preserving the original ``metadata`` dict for citation /
            analysis consumers. See :meth:`_emit_with_llm_exclusion`.

            Args:
                base_node: The ``BaseNode`` to emit (retrieved sub-node,
                    docstore-loaded parent, or windowed clone).
                score: The retrieval score to attach; may be ``None``
                    when the caller has no score.
                ner_fallback: The retrieved sub-node backfilling
                    ``entities`` / ``relations`` when *base_node* carries
                    none (see :meth:`_emit_with_llm_exclusion`).

            Returns:
                A ``NodeWithScore`` whose underlying node hides noisy
                metadata from ``get_content(MetadataMode.LLM)``.
            """
            return NodeWithScore(
                node=self._emit_with_llm_exclusion(base_node, ner_fallback=ner_fallback),
                score=score,
            )

        def _llm_tokens(base_node: BaseNode) -> int:
            """Estimate the token cost of *base_node*'s rendered LLM payload.

            The synthesiser calls ``get_content(MetadataMode.LLM)`` for
            every node, so that rendered string — not ``node.text`` —
            is what the budget must debit against. Runs against the
            post-exclusion clone so the count matches what the prompt
            actually carries.

            Args:
                base_node: The node whose LLM-payload token cost is
                    being estimated.

            Returns:
                Estimated token count for the node's LLM-mode rendering.
            """
            clone = self._emit_with_llm_exclusion(base_node)
            llm_payload = clone.get_content(metadata_mode=MetadataMode.LLM)
            return estimate_tokens(llm_payload, self.char_token_ratio)

        for node in nodes:
            parent_id = self._parent_id(node)
            if not parent_id:
                expanded.append(_emit(node.node, node.score))
                continue
            if parent_id in seen_parent_ids:
                continue

            parent_node = self._load_parent_node(parent_id)
            if parent_node is None:
                expanded.append(_emit(node.node, node.score))
                continue

            seen_parent_ids.add(parent_id)
            parent_hits += 1
            parent_text = parent_node.get_content()

            # Legacy / unbounded path.
            if not budget_enforced:
                expanded.append(_emit(parent_node, node.score, ner_fallback=node.node))
                continue

            # Budget exhausted by earlier hits — emit the sub-node so the hit
            # still contributes to retrieval count and citations without
            # inflating the prompt past ``usable_tokens``. Using
            # ``per_hit_floor`` here would silently exceed the budget by
            # ``per_hit_floor * remaining_hits`` tokens.
            if remaining_tokens <= 0:
                budget_exhausted_hits += 1
                expanded.append(_emit(node.node, node.score))
                continue

            parent_tokens = _llm_tokens(parent_node)
            if parent_tokens <= remaining_tokens:
                expanded.append(_emit(parent_node, node.score, ner_fallback=node.node))
                remaining_tokens -= parent_tokens
                continue

            # Doesn't fit — window around the matched sub-node. ``per_hit_floor``
            # is the MINIMUM window size: even when little budget is left, we
            # window at least ``per_hit_floor`` tokens so the slice is useful.
            # This is the one path that may overshoot ``usable_tokens`` by up
            # to ``per_hit_floor`` tokens; the ``remaining_tokens <= 0``
            # short-circuit above caps the overshoot at a single hit.
            budget_tokens = max(self.per_hit_floor, remaining_tokens)
            budget_chars = max(1, int(budget_tokens * self.char_token_ratio))
            sub_text = getattr(getattr(node, "node", None), "text", "") or getattr(node, "text", "") or ""
            windowed_text = self._window_parent_text(parent_text, sub_text, budget_chars)

            if windowed_text is None:
                logger.debug(
                    "parent_context_fallback_subnode: parent_id={} sub_node_id={} sub_chars={} parent_chars={}",
                    parent_id,
                    getattr(getattr(node, "node", None), "node_id", ""),
                    len(sub_text),
                    len(parent_text),
                )
                expanded.append(_emit(node.node, node.score))
                continue

            windowed_node = TextNode(
                id_=parent_node.node_id,
                text=windowed_text,
                metadata={
                    **dict(parent_node.metadata or {}),
                    "parent_context_windowed": True,
                    "parent_full_chars": len(parent_text),
                    "window_chars": len(windowed_text),
                },
            )
            windowed_emit = _emit(windowed_node, node.score, ner_fallback=node.node)
            # Render once — the LLM payload is what both the log line and
            # the budget debit need, so compute it here and reuse.
            windowed_llm_payload = windowed_emit.node.get_content(metadata_mode=MetadataMode.LLM)
            windowed_hits += 1
            logger.info(
                "parent_context_windowed: parent_id={} parent_full_chars={} "
                "window_chars={} budget_tokens={} llm_payload_chars={}",
                parent_id,
                len(parent_text),
                len(windowed_text),
                budget_tokens,
                len(windowed_llm_payload),
            )
            # Debit the LLM-rendered payload tokens (not just the raw
            # windowed text), so the budget matches what the prompt
            # carries after ``MetadataMode.LLM`` rendering.
            remaining_tokens = max(
                0,
                remaining_tokens - estimate_tokens(windowed_llm_payload, self.char_token_ratio),
            )
            expanded.append(windowed_emit)

        if budget_enforced and parent_hits > 0 and (windowed_hits > 0 or budget_exhausted_hits > 0):
            logger.info(
                "parent_context_summary: windowed_hits={}/{} budget_exhausted_hits={} usable_tokens={}",
                windowed_hits,
                parent_hits,
                budget_exhausted_hits,
                self.usable_tokens,
            )

        return expanded


class VLLMRerankPostprocessor(BaseNodePostprocessor):
    """Call a vLLM-compatible rerank endpoint and map results back to nodes."""

    api_base: str
    api_key: str | None = None
    model: str
    timeout: float = 300.0
    top_n: int = 5

    @override
    @classmethod
    def class_name(cls) -> str:
        """Return a stable class identifier.

        Returns:
            str: A string identifier for this postprocessor class.
        """
        return "VLLMRerankPostprocessor"

    @staticmethod
    def _node_text(node: NodeWithScore) -> str:
        """Extract text content from a retrieved node, trying multiple strategies for reranking.

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
        """Fallback strategy returning original nodes in stable order when vLLM reranking fails.

        Args:
            nodes (list[NodeWithScore]): The original list of nodes to return in fallback.

        Returns:
            list[NodeWithScore]: A slice of ``nodes`` up to ``top_n`` (at least one if available).
        """
        return nodes[: max(1, min(int(self.top_n), len(nodes)))]

    @override
    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        """Rerank nodes via vLLM and degrade to the original order on failure.

        Args:
            nodes (list[NodeWithScore]): The list of nodes to rerank.
            query_bundle (QueryBundle | None): Original query bundle; may carry the query string
                needed for reranking.

        Returns:
            list[NodeWithScore]: Reranked nodes, or the original order if reranking fails.

        Raises:
            ValueError: If the vLLM rerank response is malformed or has no usable results.
            urllib.error.HTTPError: HTTP error from the vLLM rerank endpoint.
            urllib.error.URLError: URL error (e.g. connection failure or timeout).
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

    @override
    @classmethod
    def class_name(cls) -> str:
        """Return a stable class identifier.

        Returns:
            str: A string identifier for this postprocessor class, used
                by LlamaIndex when matching cached configurations.
        """
        return "LazyRerankerPostprocessor"

    @override
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
        return cast(list[NodeWithScore], self.rag.reranker._postprocess_nodes(nodes, query_bundle))


def _as_qdrant_point_id(node_id: str) -> str | int:
    """Restore a point id's native type for a Qdrant lookup.

    Qdrant ids are unsigned integers or UUIDs, and nothing else. Ids travel to
    the SPA and back through JSON as strings, so an integer id returns as
    ``"1"`` — which Qdrant rejects outright, failing the whole retrieve. A
    scope built from search hits would then answer from no evidence at all
    while reporting every chunk missing. All-digit is unambiguous here
    precisely because the id domain is only those two shapes.

    Args:
        node_id (str): Point id as it came back over the wire.

    Returns:
        str | int: The id in the type Qdrant expects.
    """
    return int(node_id) if node_id.isdigit() else node_id


#: Characters of chunk text a search hit carries inline. Beyond this the
#: panel fetches the whole chunk on demand (``GET /search/chunk``).
_SEARCH_PREVIEW_CHARS = 600


class _ScopedRetriever(BaseRetriever):
    """Return exactly the chunks a session's scope names, in stable order.

    Used when an investigator has hand-picked evidence from the search panel:
    there is nothing to rank, so this bypasses the vector query entirely and
    fetches the points by id. Swapping the retriever — rather than hand-building
    a prompt — keeps citation numbering, source normalization, the report
    controls and Inspector links working unchanged, because everything
    downstream is driven by the node set.
    """

    def __init__(self, *, rag: RAG, node_ids: Sequence[str]) -> None:
        """Initialize the retriever.

        Args:
            rag (RAG): Owning engine, for the Qdrant client and collection.
            node_ids (Sequence[str]): Point ids to answer from.
        """
        super().__init__()
        self._rag = rag
        self._node_ids = [str(entry) for entry in node_ids]
        #: Scoped ids Qdrant no longer has — surfaced so a stale scope is
        #: reported rather than silently narrowing the evidence.
        self.missing = 0

    def _fetch(self, collection: str, node_ids: Sequence[str]) -> dict[str, Any]:
        """Retrieve points by id from one collection.

        Args:
            collection (str): Collection to read from.
            node_ids (Sequence[str]): Point ids to fetch.

        Returns:
            dict[str, Any]: ``{point_id: point}`` for whatever was found; an
                outage yields an empty mapping so the caller reports those ids
                missing rather than raising mid-answer.
        """
        try:
            points = self._rag.qdrant_client.retrieve(
                collection_name=collection,
                ids=[_as_qdrant_point_id(node_id) for node_id in node_ids],
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            logger.warning("Scoped retrieve failed for {}: {}", collection, exc)
            return {}
        return {str(getattr(point, "id", "")): point for point in points}

    @override
    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Fetch the scoped points and rebuild them as nodes.

        Args:
            query_bundle (QueryBundle): Ignored — the node set is fixed by the
                scope, not by the question.

        Returns:
            list[NodeWithScore]: The scoped nodes, in the scope's own order
                (not Qdrant's return order, which is unspecified).
        """
        if not self._node_ids:
            self.missing = 0
            return []

        collection = self._rag.qdrant_collection
        by_id = self._fetch(collection, self._node_ids)
        # A scoped image hit's chunk lives in the companion, not here. Looking
        # only in the main collection would resolve it to nothing and report it
        # missing — a scoped answer built on part of the selected evidence.
        unresolved = [node_id for node_id in self._node_ids if node_id not in by_id]
        if unresolved:
            companion = image_companion_name(collection)
            if self._rag._collection_exists(companion):
                by_id.update(self._fetch(companion, unresolved))
        nodes: list[NodeWithScore] = []
        for node_id in self._node_ids:
            point = by_id.get(node_id)
            if point is None:
                continue
            payload = dict(getattr(point, "payload", {}) or {})
            # The same text that made this chunk searchable is the text
            # that resolves it; otherwise an image could be found but
            # not scoped.
            text = RAG._extract_indexable_text(payload)
            if not text:
                continue
            nodes.append(NodeWithScore(node=TextNode(id_=node_id, text=text, metadata=payload), score=None))

        self.missing = len(self._node_ids) - len(nodes)
        return nodes


class MultimodalRetriever(BaseRetriever):
    """Retrieve text chunks and image captions as one evidence set.

    The image lane used to run after generation, appending its matches to a
    source list the answer had already been written from — so an image could
    never be cited, never be numbered, and never take a slot from a weaker
    text chunk. Fusing here, at the retriever, is what makes an image an
    ordinary source: everything downstream (rerank, parent context, diversity,
    numbering, synthesis) sees one list and cannot tell the lanes apart.

    The image lane is called with the user's *original* query. It translates
    for CLIP itself (``RAG._image_query_for_clip``) because only the CLIP text
    tower needs English; the reranker downstream is cross-lingual.

    Attributes:
        text_retriever: The vector/hybrid retriever over the main collection.
        image_lane: Callable taking the query and returning image-caption
            nodes. Its failures are absorbed — an image-lane outage degrades
            the answer to text-only rather than failing the query.
    """

    def __init__(
        self,
        *,
        text_retriever: Any,
        image_lane: Callable[[str], list[NodeWithScore]],
    ) -> None:
        """Initialize the fused retriever.

        Args:
            text_retriever (Any): Retriever over the main collection.
            image_lane (Callable[[str], list[NodeWithScore]]): Image-caption
                node producer.
        """
        self.text_retriever = text_retriever
        self.image_lane = image_lane
        super().__init__()

    @override
    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Retrieve both lanes for one query.

        Args:
            query_bundle (QueryBundle): The query being answered.

        Returns:
            list[NodeWithScore]: Text hits followed by image-caption hits. The
                order carries no ranking intent — the reranker downstream
                scores both lanes on one scale.
        """
        nodes: list[NodeWithScore] = list(self.text_retriever.retrieve(query_bundle))
        try:
            image_nodes = self.image_lane(query_bundle.query_str)
        except Exception as exc:
            logger.warning("Image lane retrieval failed: {}. Answering from text sources only.", exc)
            return nodes

        # A standalone image file lives in both collections: ``ImageReader``
        # writes its caption into the main collection as the document's text,
        # and ``ImageIngestionService`` writes the CLIP point into the
        # ``_images`` companion. Retrieving both would spend two numbered
        # source slots on one piece of evidence. The main-collection node wins
        # -- it is the one with a docstore entry and parent links.
        already_retrieved = {
            str(node.node.metadata.get("image_id") or "").strip()
            for node in nodes
            if node.node.metadata.get("image_id")
        }
        nodes.extend(
            node
            for node in image_nodes
            if str(node.node.metadata.get("image_id") or "").strip() not in already_retrieved
        )
        return nodes


class ImageRelevanceFloorPostprocessor(BaseNodePostprocessor):
    """Drop image sources the reranker judged irrelevant.

    Runs immediately after the reranker, so it sees comparable scores. The
    floor exists because the top-n cut alone cannot protect a sparse
    collection: with few text chunks competing, a merely-nearest image would
    take a slot and read as evidence. It sits on the *reranker* score, never
    on raw CLIP cosine — measured on a live collection, an unrelated query and
    a matching one both land in a ~0.20-0.30 CLIP band, while reranker scores
    separate by ~30x (see ``IMAGE_RERANK_MIN_SCORE`` in the docs).

    Text nodes pass through untouched: their relevance is the reranker's and
    the top-n cut's business, not this floor's.

    Attributes:
        min_score: The reranker score an image caption must reach.
    """

    min_score: float = 0.05

    @override
    @classmethod
    def class_name(cls) -> str:
        """Return a stable class identifier."""
        return "ImageRelevanceFloorPostprocessor"

    @override
    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        """Apply the floor to image-lane nodes only.

        Args:
            nodes (list[NodeWithScore]): Reranked nodes from both lanes.
            query_bundle (QueryBundle | None): Unused; required by the
                postprocessor contract.

        Returns:
            list[NodeWithScore]: The input minus sub-floor image nodes.
        """
        if not any(node.node.metadata.get(IMAGE_LANE_METADATA_KEY) for node in nodes):
            return nodes
        # ``VLLMRerankPostprocessor`` swallows its own transport errors and
        # returns the nodes untouched, so a wholly unscored set means the
        # rerank degraded -- not that nothing is relevant. Gating on that would
        # blank the image lane for as long as the endpoint is down.
        if all(node.score is None for node in nodes):
            logger.warning("Rerank returned no scores; surfacing image sources ungated.")
            return nodes

        kept: list[NodeWithScore] = []
        for node in nodes:
            if not node.node.metadata.get(IMAGE_LANE_METADATA_KEY):
                kept.append(node)
                continue
            if node.score is not None and float(node.score) >= self.min_score:
                kept.append(node)
        return kept


class CitationNumberingPostprocessor(BaseNodePostprocessor):
    """Number the final snippet set so the answer and the UI agree.

    Answers refer to "source 1", "source 2". Without this the generator is
    counting the snippets itself — the prompt carries no numbers — so its
    ordinals are pinned to nothing a reader can click. The frontend dedupes
    its citation list and the backend appends image sources after generation,
    so list position and prompt position drift apart.

    Stamping ``citation_index`` here, as the *last* postprocessor, numbers
    exactly the node set the synthesizer packs into ``context_str``: after
    reranking has trimmed, parent-context has expanded, and social diversity
    and link-following have added and dropped nodes. The same number then
    rides the normalized source out to the chat window
    (:meth:`RAG._source_from_payload`), so nothing downstream ever recomputes
    it from a position.

    ``citation_index`` is whitelisted in :data:`LLM_VISIBLE_METADATA_KEYS` so
    the synthesizer's ``get_content(MetadataMode.LLM)`` rendering carries it
    into the prompt.
    """

    @override
    @classmethod
    def class_name(cls) -> str:
        """Return a stable class identifier.

        Returns:
            str: A string identifier for this postprocessor class, used
                by LlamaIndex when matching cached configurations.
        """
        return "CitationNumberingPostprocessor"

    @override
    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        """Stamp each node with its 1-based position in the synthesized set.

        The incoming nodes are left untouched. ``ParentContextPostprocessor``
        emits nodes loaded from the docstore, which caches them across
        queries; writing a number into one in place would leak this query's
        numbering into every later query that retrieves the same parent.

        Args:
            nodes (list[NodeWithScore]): The final node set, in the order the
                synthesizer will render it.
            query_bundle (QueryBundle | None): Unused; required by the
                postprocessor contract.

        Returns:
            list[NodeWithScore]: Clones carrying ``citation_index``, in the
                incoming order.
        """
        numbered: list[NodeWithScore] = []
        for index, scored in enumerate(nodes, start=1):
            clone = scored.node.model_copy(deep=True)
            clone.metadata["citation_index"] = index
            # A node that reached us with an exclusion list (the LLM-visible
            # whitelist applied by ``ParentContextPostprocessor``) predates
            # the key, but a future postprocessor ordering could hand us one
            # that names it. Numbering the prompt is the whole point, so it
            # must never be excluded from the LLM's view.
            clone.excluded_llm_metadata_keys = [k for k in clone.excluded_llm_metadata_keys if k != "citation_index"]
            numbered.append(NodeWithScore(node=clone, score=scored.score))
        return numbered


class _StreamingRefineMixin:
    """Restore true token streaming to llama-index's ``Refine`` synthesizers.

    llama-index 0.14's ``Refine._update_response`` streaming path routes
    through ``DefaultRefineProgram.stream_call``, which consumes the entire
    LLM token stream before yielding one complete answer (and
    ``_get_attribute_from_object_generator`` drains whatever a program yields
    before emitting), so ``response_gen`` produces the whole answer as a
    single chunk only after generation finishes. Chat answers then appear all
    at once with time-to-first-token equal to full generation, while the
    summary path — which calls ``stream_complete`` directly — streams.

    In the plain-text case (no structured answer filtering, no output class)
    this mixin hands the LLM's live token generator straight back to the
    refine loop, which already supports ``Generator`` responses: a further
    refine pass materializes it via ``get_response_text`` and the final one is
    returned as ``response_gen``.
    """

    def _update_response(
        self,
        program: Any,
        program_kwargs: dict[str, Any],
        response_kwargs: dict[str, Any],
    ) -> Any:
        """Stream plain-text answers directly; defer to upstream otherwise.

        Args:
            program (Any): The refine program built for the current prompt.
            program_kwargs (dict[str, Any]): Prompt variables for this chunk.
            response_kwargs (dict[str, Any]): Extra LLM kwargs from the caller.

        Returns:
            Any: A live token generator in the plain-text streaming case,
                otherwise whatever the upstream implementation produces.
        """
        prompt = getattr(program, "_prompt", None)
        plain_text_streaming = (
            self._streaming  # type: ignore[attr-defined]
            and not self._structured_answer_filtering  # type: ignore[attr-defined]
            and self._output_cls is None  # type: ignore[attr-defined]
            and prompt is not None
        )
        if not plain_text_streaming:
            return super()._update_response(program, program_kwargs, response_kwargs)  # type: ignore[misc]
        return self._llm.stream(prompt, **program_kwargs, **response_kwargs)  # type: ignore[attr-defined]


class StreamingRefine(_StreamingRefineMixin, Refine):
    """``Refine`` that streams plain-text answers token by token."""


class StreamingCompactAndRefine(_StreamingRefineMixin, CompactAndRefine):
    """``CompactAndRefine`` that streams plain-text answers token by token."""


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
class RemoteSparseEncoder:
    """Adapter that turns remote pooling/tokenize responses into Qdrant sparse vectors.

    Speaks the vLLM pooling protocol — ``POST {root}/pooling`` with
    ``task="token_classify"`` plus ``POST {root}/tokenize`` — against
    either the full vllm-service router (which exposes both as LiteLLM
    pass-throughs to the ``embed`` backend) or the standalone
    ``embed-only`` CPU container. The wire format is frozen: production
    collections were ingested with it.

    Both backends return one ``/pooling`` score per *inner* token — they
    drop the BOS/EOS positions server-side (vLLM's ``BOSEOSFilter``;
    verified element-wise in vllm-service#75) — while ``/tokenize``
    returns the full id list including the specials. The encoder strips
    the boundary ids with the same conditional semantics before pairing
    (see ``_strip_boundary_ids``); the ids default to bge-m3's XLM-R
    values and ``-1`` disables either strip, mirroring ``BOSEOSFilter``.
    """

    api_base: str
    model: str
    api_key: str | None = None
    timeout: float = 300.0
    bos_token_id: int = 0
    eos_token_id: int = 2

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
            token_ids = self._strip_boundary_ids(self._tokenize(text))
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
            Any: The decoded JSON response (dict, list, or other JSON shape per endpoint).
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
            list[list[float]]: One token-score list per input text, in input order. Each inner
                list's scores align with the tokens of the corresponding text.
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

        response_data = response_body.get("data") if isinstance(response_body, dict) else None
        if not isinstance(response_data, list):
            raise ValueError("vLLM sparse pooling response did not contain a data list")

        pooled_scores: list[list[float]] = []
        for item in response_data:
            raw_scores = item.get("data") if isinstance(item, dict) else item
            pooled_scores.append(self._coerce_token_scores(raw_scores))

        if len(pooled_scores) != len(texts):
            raise ValueError("vLLM sparse pooling response count did not match the input batch size")
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

    def _strip_boundary_ids(self, token_ids: list[int]) -> list[int]:
        """Drop the BOS/EOS boundary ids the pooling backend never scores.

        Mirrors vLLM's ``BOSEOSFilter`` exactly: the first id is dropped
        iff it equals ``bos_token_id``, the last iff it equals
        ``eos_token_id`` — never an unconditional slice. After this the
        id list aligns one-to-one with the ``/pooling`` scores; the
        boundary weights the backends discard are NOT reliably zero on
        the model side (``<s>`` measured at 0.11-0.24), so pairing
        without the strip shifts every score onto the previous token's
        id (docint#410).

        Args:
            token_ids (list[int]): Full ``/tokenize`` id list, specials included.

        Returns:
            list[int]: The ids the pooling response actually scores.
        """
        if token_ids and token_ids[0] == self.bos_token_id:
            token_ids = token_ids[1:]
        if token_ids and token_ids[-1] == self.eos_token_id:
            token_ids = token_ids[:-1]
        return token_ids

    @classmethod
    def _extract_token_ids(cls, payload: Any) -> list[int]:
        """Extract token ids from a vLLM tokenize response payload.

        Args:
            payload (Any): JSON-decoded response from the vLLM tokenize endpoint; expected to
                carry token IDs at one of several possible locations.

        Returns:
            list[int]: Extracted token IDs, or an empty list if none can be found.
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
            if isinstance(candidate, list) and candidate and all(isinstance(item, int) for item in candidate):
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

            if isinstance(item, list | tuple):  # pyrefly: ignore[implicit-any-type-argument]  # bare list/tuple in isinstance; type params unknowable at this call site
                numeric_values = [float(value) for value in item if isinstance(value, int | float)]
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
            token_ids (list[int]): Token IDs from the input text.
            token_scores (list[float]): Scores aligned with ``token_ids``.

        Returns:
            tuple[list[int], list[float]]: Aggregated (ids, scores) for the sparse vector. Duplicate
                token IDs are merged by max-score; negative IDs and non-finite or non-positive scores
                are filtered out; results are sorted by token ID ascending. Both lists are empty if
                nothing survives filtering.

        Raises:
            ValueError: When ids and scores differ in length. Zipping
                misaligned lists writes an off-by-one sparse vector into
                the collection silently; sparse encoding is not
                fail-soft, so this refuses instead (docint#410 — the
                mismatch was previously a debug log).
        """
        if len(token_ids) != len(token_scores):
            raise ValueError(
                f"vLLM sparse token length mismatch after boundary strip: "
                f"{len(token_ids)} token ids vs {len(token_scores)} scores"
            )

        merged_scores: dict[int, float] = {}
        for token_id, score in zip(token_ids, token_scores, strict=True):
            if token_id < 0 or not math.isfinite(score) or score <= 0.0:
                continue
            existing = merged_scores.get(token_id)
            if existing is None or score > existing:
                merged_scores[token_id] = score

        ordered = sorted(merged_scores.items())
        return [token_id for token_id, _ in ordered], [score for _, score in ordered]


# Per-request active physical collection. Backs the ``RAG.qdrant_collection``
# property so concurrent requests for different collections never clobber a
# shared field (the WS2 multi-tenant fix). Unset (``None``) outside a request
# scope, where the engine falls back to its process default; bound for the
# duration of a request by :meth:`RAG.collection_scope`. ContextVars copy into
# anyio worker threads via ``copy_context()``, so the override survives the
# ``to_thread.run_sync`` hops used by the streaming endpoints.
_active_collection: ContextVar[str | None] = ContextVar("docint_active_collection", default=None)

# Upper bound on the per-collection retrieval-handle caches (``index`` /
# ``query_engine``). Each cached handle pins a SQLite docstore connection, so an
# unbounded cache would leak file descriptors on a host with many collections.
# Evicted handles are dropped from the cache only (not closed): an in-flight
# request keeps its own reference and the OS reclaims the fd once it is gone.
_RETRIEVAL_HANDLE_CACHE_MAX = 32


# ``slots=True`` is intentionally NOT used here. ``qdrant_collection`` is a
# normal field for the type checker / generated ``__init__`` signature, but its
# reads and writes are intercepted by a property attached after the class body
# (see ``_rag_qdrant_collection_get`` / ``_set`` at the end of the module).
# Under ``slots`` the field's slot descriptor would capture the constructor
# value and the post-class property would never see it; without slots the
# property cleanly shadows the (never-stored) field.
@dataclass
class RAG:
    """Retrieval-Augmented Generation engine.

    Handles configuration, initialization, and interaction with embedding models, generation
    models, and vector stores. Provides methods to start sessions, retrieve information, and
    manage document ingestion.

    Collection statelessness:
        ``qdrant_collection`` reads/writes go through a property (attached after
        the class body) layered over a per-request
        :class:`contextvars.ContextVar` (see :meth:`collection_scope`) with a
        process-default fallback (``_collection_default``). The derived
        ``index`` and ``query_engine`` are likewise per-collection: properties
        backed by caches keyed on the active physical name, holding read-only
        handles that are safe to share across threads. Concurrent requests for
        different collections are therefore isolated without any global mutation.
    """

    # --- Constructor args ---
    # Declared as a plain field so ``RAG(qdrant_collection=...)`` type-checks and
    # internal ``self.qdrant_collection`` reads resolve to ``str``; the value is
    # never actually stored on this attribute -- the post-class property routes
    # writes into ``_collection_default`` (or the request ContextVar). MUST stay
    # the first field so its init assignment runs before ``_collection_default``.
    qdrant_collection: str
    enable_hybrid: bool = field(default_factory=resolve_enable_hybrid)

    # --- Environment config ---
    host_config: HostConfig = field(default_factory=load_host_env, init=False, repr=False)
    ingestion_config: IngestionConfig = field(default_factory=load_ingestion_env, init=False, repr=False)
    model_config: ModelConfig = field(default_factory=load_model_env, init=False, repr=False)
    ner_config: NERConfig = field(default_factory=load_ner_env, init=False, repr=False)
    openai_config: OpenAIConfig = field(default_factory=load_openai_env, init=False, repr=False)
    embedding_config: EmbeddingConfig = field(default_factory=load_embedding_env, init=False, repr=False)
    path_config: PathConfig = field(default_factory=load_path_env, init=False, repr=False)
    graphrag_config: GraphRAGConfig = field(default_factory=load_graphrag_env, init=False, repr=False)
    retrieval_config: RetrievalConfig = field(default_factory=load_retrieval_env, init=False, repr=False)
    summary_config: SummaryConfig = field(default_factory=load_summary_env, init=False, repr=False)
    session_config: SessionConfig = field(default_factory=load_session_env, init=False, repr=False)

    # --- Models ---
    embed_model_id: str | None = field(default=None, init=False)
    sparse_model_id: str | None = field(default=None, init=False)
    rerank_model_id: str | None = field(default=None, init=False)
    text_model_id: str | None = field(default=None, init=False)

    # --- Named entity recognition ---
    ner_enabled: bool = field(default=False, init=False)
    # Keyed by physical collection name, like the pagination caches below.
    # A single un-keyed list here served whichever collection populated it
    # first to every later request (and every tenant) — see the regression
    # tests in tests/test_ner_sources_collection_scoping.py.
    _ner_sources_cache: dict[str, list[dict[str, Any]]] = field(default_factory=dict, init=False, repr=False)
    ner_aggregate_cache: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict, init=False, repr=False)
    ner_graph_cache: dict[tuple[str, str, int, int], dict[str, Any]] = field(
        default_factory=dict, init=False, repr=False
    )

    # --- Pagination caches (server-side slicing for HTTP endpoints) ---
    _documents_cache: dict[str, list[dict[str, Any]]] = field(default_factory=dict, init=False, repr=False)
    _hate_speech_cache: dict[str, list[dict[str, Any]]] = field(default_factory=dict, init=False, repr=False)
    # Per-collection resolved-entity index ({alias_to_id, canonical, case_normalize}
    # or None), memoized so paginated drill-down pages don't re-scroll _entities.
    _resolved_index_cache: dict[str, dict[str, Any] | None] = field(default_factory=dict, init=False, repr=False)

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
    _embed_token_counter: Callable[[str], list[int]] | None = field(default=None, init=False, repr=False)

    # --- Path setup ---
    data_dir: Path | None = field(default=None, init=False)
    hf_hub_cache: Path | None = field(default=None, init=False)

    # --- Reranking / retrieval ---
    retrieve_similarity_top_k: int = field(default=20, init=False)
    rerank_top_n: int = field(default=5, init=False)
    chat_response_mode: str = field(default="auto", init=False)
    vector_store_query_mode: str = field(default="auto", init=False)
    hybrid_alpha: float = field(default=0.5, init=False)
    sparse_top_k: int = field(default=20, init=False)
    hybrid_top_k: int = field(default=20, init=False)
    parent_context_enabled: bool = field(default=True, init=False)
    parent_context_safety_margin: float = field(default=0.95, init=False)
    social_diversity_limit: int = field(default=2, init=False)
    graphrag_enabled: bool = field(default=False, init=False)
    graphrag_neighbor_hops: int = field(default=1, init=False)
    graphrag_top_k_nodes: int = field(default=100, init=False)
    graphrag_min_edge_weight: int = field(default=1, init=False)
    graphrag_max_neighbors: int = field(default=6, init=False)
    summary_coverage_target: float = field(default=0.70, init=False)
    summary_final_source_cap: int = field(default=24, init=False)

    # --- Session config ---
    session_store: str = field(default="", init=False)

    # --- Qdrant controls ---
    docstore_batch_size: int = field(default=100, init=False)
    ingest_benchmark_enabled: bool = field(default=False, init=False)
    ingest_fail_fast: bool = field(default=False, init=False)
    ingest_manifest_enabled: bool = field(default=True, init=False)
    ingest_pipeline_overlap_enabled: bool = field(default=False, init=False)
    ingest_queue_max_size: int = field(default=4, init=False)
    docstore_max_retries: int = field(default=3, init=False)
    docstore_retry_backoff_seconds: float = field(default=0.25, init=False)
    docstore_retry_backoff_max_seconds: float = field(default=2.0, init=False)
    qdrant_host: str | None = field(default=None, init=False)
    _qdrant_src_dir: Path | None = field(default=None, init=False, repr=False)

    # --- Prompt config ---
    language_code: str = field(default="en", init=False)
    prompt_dir: Path | None = field(default=None, init=False)
    summarize_prompt_path: Path | None = field(default=None, init=False)
    conversation_summary_prompt_path: Path | None = field(default=None, init=False)
    rewrite_retrieval_prompt_path: Path | None = field(default=None, init=False)
    grounded_text_qa_prompt_path: Path | None = field(default=None, init=False)
    grounded_refine_prompt_path: Path | None = field(default=None, init=False)
    grounded_collection_summary_prompt_path: Path | None = field(default=None, init=False)
    summary_map_prompt_path: Path | None = field(default=None, init=False)
    summary_fold_prompt_path: Path | None = field(default=None, init=False)
    summarize_prompt: str = field(default="", init=False)
    conversation_summary_prompt: str = field(default="", init=False)
    rewrite_retrieval_prompt: str = field(default="", init=False)
    grounded_text_qa_prompt: str = field(default="", init=False)
    grounded_refine_prompt: str = field(default="", init=False)
    grounded_collection_summary_prompt: str = field(default="", init=False)
    summary_map_prompt: str = field(default="", init=False)
    summary_fold_prompt: str = field(default="", init=False)

    # --- Runtime (lazy caches / not in repr) ---
    _embed_model: BaseEmbedding | None = field(default=None, init=False, repr=False)
    _text_model: OpenAI | None = field(default=None, init=False, repr=False)
    _post_retrieval_text_model: OpenAI | None = field(default=None, init=False, repr=False)
    _reranker: BaseNodePostprocessor | None = field(default=None, init=False, repr=False)
    sparse_client_config: SparseClientConfig | None = field(default=None, init=False, repr=False)
    embed_client_config: EmbedClientConfig | None = field(default=None, init=False, repr=False)
    _qdrant_client: QdrantClient | None = field(default=None, init=False, repr=False)
    _qdrant_aclient: AsyncQdrantClient | None = field(default=None, init=False, repr=False)
    _parent_context_support_cache: dict[str, bool] = field(default_factory=dict, init=False, repr=False)
    _image_ingestion_service: ImageIngestionService | None = field(default=None, init=False, repr=False)

    # -- Ingested data ---
    dir_reader: SimpleDirectoryReader | None = field(default=None, init=False)
    docs: list[Document] = field(default_factory=list, init=False)
    nodes: list[BaseNode] = field(default_factory=list, init=False)

    # --- Built components (lazy loaded, per physical collection) ---
    # ``index`` / ``query_engine`` are exposed as properties backed by these
    # per-collection caches so concurrent requests on different collections do
    # not share a single handle. Handles are read-only (a Qdrant vector store +
    # HTTP clients) and therefore safe to share across threads.
    # Process-default active collection (the property's fallback when no request
    # scope is bound). MUST keep a simple ``default`` (never ``default_factory``)
    # so the generated ``__init__`` does not re-assign it and clobber the value
    # the ``qdrant_collection`` field's init assignment routes here.
    _collection_default: str = field(default="", init=False, repr=False)
    _index_cache: OrderedDict[str, VectorStoreIndex] = field(default_factory=OrderedDict, init=False, repr=False)
    _query_engine_cache: OrderedDict[str, RetrieverQueryEngine] = field(
        default_factory=OrderedDict, init=False, repr=False
    )
    _retrieval_cache_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    sessions: SessionManager | None = field(default=None, init=False)
    reports: ReportManager | None = field(default=None, init=False)
    collection_owners: CollectionOwnerManager | None = field(default=None, init=False)
    _collection_backfill_done: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up any necessary components.

        The constructor's ``qdrant_collection`` argument has, by this point,
        already been routed into ``_collection_default`` by the property setter
        (the generated ``__init__`` assigns the first field before calling this
        hook). Per-request overrides are applied via :meth:`collection_scope`;
        reads fall back to the default when no request scope is active (CLI /
        single-collection usage).

        Raises:
            ValueError: If summarize_prompt_path is not set.
        """
        # --- Host config ---
        self.qdrant_host = self.host_config.qdrant_host

        # --- Ingestion config ---
        self.docstore_batch_size = self.ingestion_config.docstore_batch_size
        self.ingest_benchmark_enabled = self.ingestion_config.ingest_benchmark_enabled
        self.ingest_fail_fast = self.ingestion_config.ingest_fail_fast
        self.ingest_manifest_enabled = self.ingestion_config.ingest_manifest_enabled
        self.ingest_pipeline_overlap_enabled = self.ingestion_config.ingest_pipeline_overlap_enabled
        self.ingest_queue_max_size = self.ingestion_config.ingest_queue_max_size
        self.docstore_max_retries = self.ingestion_config.docstore_max_retries
        self.docstore_retry_backoff_seconds = self.ingestion_config.docstore_retry_backoff_seconds
        self.docstore_retry_backoff_max_seconds = self.ingestion_config.docstore_retry_backoff_max_seconds

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

        # --- Sparse encoder client config (remote on every provider) ---
        self.sparse_client_config = load_sparse_client_env(
            default_api_base=self.openai_api_base or "",
            default_api_key=self.openai_api_key,
            default_timeout=self.openai_timeout,
        )

        # --- Dense embedding client config (remote on every provider) ---
        self.embed_client_config = load_embed_client_env(
            default_api_base=self.openai_api_base or "",
            default_api_key=self.openai_api_key,
        )

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
                "Embedding worst-case wait is {:.0f}s (timeout={}s * (1 + "
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
                "Embedding tokenizer loaded from {} (repo={!r}) — using exact token counts for pre-embed fit checks.",
                self.path_config.hf_hub_cache,
                self.model_config.embed_tokenizer_repo,
            )

        # --- Named Entity Recognition (NER) config ---
        self.ner_enabled = self.ner_config.enabled

        # --- Path config ---
        self.path_config = self.path_config
        self.data_dir = self.path_config.data
        self.language_code = load_language_env().code
        self.prompt_dir = self.path_config.prompts / self.language_code
        self._qdrant_src_dir = self.path_config.qdrant_sources
        self.hf_hub_cache = self.path_config.hf_hub_cache

        ## --- Load prompts ---
        if self.prompt_dir:
            self.summarize_prompt_path = self.prompt_dir / "summarize.txt"
            self.conversation_summary_prompt_path = self.prompt_dir / "conversation_summary.txt"
            self.rewrite_retrieval_prompt_path = self.prompt_dir / "rewrite_retrieval.txt"
            self.grounded_text_qa_prompt_path = self.prompt_dir / "grounded_qa.txt"
            self.grounded_refine_prompt_path = self.prompt_dir / "grounded_refine.txt"
            self.grounded_collection_summary_prompt_path = self.prompt_dir / "grounded_collection_summary.txt"
            self.summary_map_prompt_path = self.prompt_dir / "summary_map.txt"
            self.summary_fold_prompt_path = self.prompt_dir / "summary_fold.txt"
        if self.summarize_prompt_path is None:
            logger.error("ValueError: summarize_prompt_path is not set. Cannot load summarize prompt.")
            raise ValueError("summarize_prompt_path is not set. Cannot load summarize prompt.")
        self.summarize_prompt = self._load_prompt_text(
            self.summarize_prompt_path,
            default=DEFAULT_SUMMARIZE_PROMPT,
            required=True,
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
        self.grounded_collection_summary_prompt = self._load_prompt_text(
            self.grounded_collection_summary_prompt_path,
            default=DEFAULT_GROUNDED_COLLECTION_SUMMARY_PROMPT,
        )
        self.summary_map_prompt = self._load_prompt_text(
            self.summary_map_prompt_path,
            default=DEFAULT_SUMMARY_MAP_PROMPT,
        )
        self.summary_fold_prompt = self._load_prompt_text(
            self.summary_fold_prompt_path,
            default=DEFAULT_SUMMARY_FOLD_PROMPT,
        )

        # --- Retrieval config ---
        self.retrieve_similarity_top_k = self.retrieval_config.retrieve_top_k
        self.chat_response_mode = self.retrieval_config.chat_response_mode
        self.vector_store_query_mode = self.retrieval_config.vector_store_query_mode
        self.hybrid_alpha = self.retrieval_config.hybrid_alpha
        self.sparse_top_k = self.retrieval_config.sparse_top_k
        self.hybrid_top_k = self.retrieval_config.hybrid_top_k
        self.parent_context_enabled = self.retrieval_config.parent_context_enabled
        self.parent_context_safety_margin = self.retrieval_config.parent_context_safety_margin
        self.social_diversity_limit = self.retrieval_config.social_diversity_limit
        self.rerank_top_n = int(self.retrieve_similarity_top_k // 4)
        self.graphrag_enabled = self.graphrag_config.enabled
        self.graphrag_neighbor_hops = self.graphrag_config.neighbor_hops
        self.graphrag_top_k_nodes = self.graphrag_config.top_k_nodes
        self.graphrag_min_edge_weight = self.graphrag_config.min_edge_weight
        self.graphrag_max_neighbors = self.graphrag_config.max_neighbors

        # --- Session config ---
        self.session_store = self.session_config.session_store
        self.sessions = SessionManager(self)
        self.reports = ReportManager(self)

        # --- Summary config ---
        self.summary_coverage_target = self.summary_config.coverage_target
        self.summary_final_source_cap = self.summary_config.final_source_cap

    # --- Active collection (stateless, per-request) ---
    # NOTE: ``qdrant_collection`` is exposed as a property, but it is attached
    # *after* the class body (see ``_rag_qdrant_collection_get`` / ``_set``
    # below) rather than declared here. A same-named property in the body would
    # hide the ``InitVar`` field from the dataclass-generated ``__init__``
    # signature (and from the type checker), breaking ``RAG(qdrant_collection=
    # ...)`` at every call site. Reads resolve to the per-request ContextVar
    # override when a ``collection_scope`` is active, else the process default.

    @contextmanager
    def collection_scope(self, physical: str) -> Iterator[None]:
        """Bind ``physical`` as the active collection for the enclosed block.

        Sets the per-request :data:`_active_collection` ContextVar so every
        ``self.qdrant_collection`` read (and the derived ``index`` /
        ``query_engine`` caches) resolves to ``physical`` for the duration of
        the block, then restores the previous value. The binding is scoped to
        the current context, so concurrent requests on different threads — and
        the anyio worker threads they spawn via ``to_thread.run_sync`` — each
        observe only their own collection.

        Args:
            physical (str): The physical (owner-namespaced) Qdrant collection
                name to make active.

        Yields:
            None: Control returns to the caller with the scope active.
        """
        token = _active_collection.set(physical)
        try:
            yield
        finally:
            _active_collection.reset(token)

    def _cache_get(self, cache: OrderedDict[str, Any], key: str) -> Any:
        """Return a cached handle for ``key``, marking it most-recently-used.

        Args:
            cache (OrderedDict[str, Any]): The bounded LRU cache to read.
            key (str): The active physical collection name.

        Returns:
            Any: The cached handle, or ``None`` when absent.
        """
        with self._retrieval_cache_lock:
            value = cache.get(key)
            if value is not None:
                cache.move_to_end(key)
            return value

    def _cache_put(self, cache: OrderedDict[str, Any], key: str, value: Any) -> None:
        """Store (or evict) a cached handle for ``key`` under the cache bound.

        A ``None`` value evicts the entry (preserving the previous reset
        semantics of ``self.index = None``). Otherwise the entry is inserted as
        most-recently-used and the oldest entries are dropped until the cache is
        within :data:`_RETRIEVAL_HANDLE_CACHE_MAX`.

        Args:
            cache (OrderedDict[str, Any]): The bounded LRU cache to mutate.
            key (str): The active physical collection name.
            value (Any): The handle to cache, or ``None`` to evict.
        """
        with self._retrieval_cache_lock:
            if value is None:
                cache.pop(key, None)
                return
            cache[key] = value
            cache.move_to_end(key)
            while len(cache) > _RETRIEVAL_HANDLE_CACHE_MAX:
                cache.popitem(last=False)

    @property
    def index(self) -> VectorStoreIndex | None:
        """Return the cached :class:`VectorStoreIndex` for the active collection.

        Returns:
            VectorStoreIndex | None: The per-collection index, or ``None`` when
            one has not been built yet.
        """
        return cast("VectorStoreIndex | None", self._cache_get(self._index_cache, self.qdrant_collection))

    @index.setter
    def index(self, value: VectorStoreIndex | None) -> None:
        """Cache (or evict) the index for the active collection.

        Args:
            value (VectorStoreIndex | None): The index to cache; ``None`` evicts
                the active collection's entry.
        """
        self._cache_put(self._index_cache, self.qdrant_collection, value)

    @property
    def query_engine(self) -> RetrieverQueryEngine | None:
        """Return the cached query engine for the active collection.

        Returns:
            RetrieverQueryEngine | None: The per-collection query engine, or
            ``None`` when one has not been built yet.
        """
        return cast("RetrieverQueryEngine | None", self._cache_get(self._query_engine_cache, self.qdrant_collection))

    @query_engine.setter
    def query_engine(self, value: RetrieverQueryEngine | None) -> None:
        """Cache (or evict) the query engine for the active collection.

        Args:
            value (RetrieverQueryEngine | None): The engine to cache; ``None``
                evicts the active collection's entry.
        """
        self._cache_put(self._query_engine_cache, self.qdrant_collection, value)

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
                    self._qdrant_src_dir = Path(home) / ".qdrant" / "storage" / "sources"
        if self._qdrant_src_dir is None:
            logger.error("ValueError: Qdrant source host directory is not set.")
            raise ValueError("Qdrant source host directory is not set.")
        return self._qdrant_src_dir

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
                "api_base": self.embed_client_config.api_base if self.embed_client_config else self.openai_api_base,
                "api_key": self.embed_client_config.api_key if self.embed_client_config else self.openai_api_key,
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
        """Return the configured sparse model id for hybrid retrieval.

        The id is passed through to the remote encoder as the ``model``
        field; docint no longer resolves it against a local support list,
        because it no longer runs a local sparse model.

        Returns:
            str | None: The sparse model id, or None when hybrid is off.

        Raises:
            ValueError: If hybrid is enabled but no sparse model is set.
        """
        if not self.enable_hybrid:
            return None
        if self.sparse_model_id is None:
            raise ValueError("sparse_model_id is None")
        return self.sparse_model_id

    @property
    def reranker(self) -> BaseNodePostprocessor:
        """Lazily initialize the remote rerank-endpoint postprocessor.

        Reranking is always remote — the consumer points at either the
        full vllm-service stack (LiteLLM router forwards ``/v1/rerank``
        to the vLLM rerank backend) or the standalone ``rerank-only``
        deployment (``http://rerank-cpu:8000``; same Jina-shape
        contract). Override the endpoint independently of chat/embed
        with ``RERANK_API_BASE`` / ``RERANK_API_KEY``.

        Transport failures degrade gracefully — when ``/rerank`` is
        unreachable or returns an unexpected payload,
        :class:`VLLMRerankPostprocessor._postprocess_nodes` catches the
        error and returns the original retrieval order (top_n nodes
        unranked).

        Returns:
            BaseNodePostprocessor: A configured
            :class:`VLLMRerankPostprocessor`.

        Raises:
            ValueError: If ``rerank_model_id`` is ``None``.
        """
        if self.rerank_model_id is None:
            raise ValueError("rerank_model_id cannot be None")
        if self._reranker is None:
            rerank_cfg: RerankClientConfig = load_rerank_client_env(
                default_api_base=self.openai_api_base or "",
                default_api_key=self.openai_api_key,
                default_timeout=self.openai_timeout,
            )
            self._reranker = VLLMRerankPostprocessor(
                api_base=rerank_cfg.api_base,
                api_key=rerank_cfg.api_key,
                model=self.rerank_model_id,
                timeout=rerank_cfg.timeout,
                top_n=self.rerank_top_n,
            )
            logger.info(
                "Initializing remote reranker endpoint client at {} with model: {}",
                rerank_cfg.api_base,
                self.rerank_model_id,
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
            self._post_retrieval_text_model = self._create_text_model(enable_reasoning=True)
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

    def _build_sparse_encoder(self) -> RemoteSparseEncoder:
        """Construct the remote sparse encoder from the resolved config.

        Single construction point, shared by the vector-store wiring and
        the pre-ingest probe, so the two can never drift apart in how
        they resolve the endpoint.

        Returns:
            RemoteSparseEncoder: Encoder bound to the configured endpoint.
        """
        sparse_config = self.sparse_client_config
        return RemoteSparseEncoder(
            api_base=sparse_config.api_base if sparse_config else self.openai_api_base or "",
            api_key=sparse_config.api_key if sparse_config else self.openai_api_key,
            model=self.sparse_model or "",
            timeout=sparse_config.timeout if sparse_config else self.openai_timeout,
        )

    def probe_qdrant(self) -> bool:
        """Check that Qdrant answers at the configured host.

        Qdrant is otherwise contacted lazily, so a mis-wired deployment
        (backend not on data-net, data-plane stack down) would surface only
        at the first ingest or query. Called once at application startup to
        log a loud, actionable error instead. Never raises: Qdrant may
        legitimately come up after the backend, and the SQLite-backed
        endpoints still work without it.

        Returns:
            bool: ``True`` when Qdrant's readiness endpoint answered with a
                2xx status, ``False`` otherwise.
        """
        if not self.qdrant_host:
            logger.error("QDRANT_HOST is unset; ingest and query will fail until it is configured.")
            return False
        url = f"{self.qdrant_host.rstrip('/')}/readyz"
        try:
            with urllib.request.urlopen(url, timeout=QDRANT_PROBE_TIMEOUT_S) as response:
                status = int(response.status)
        except Exception as exc:
            logger.error(
                "Qdrant is unreachable at {}: {}. Check that the data-plane stack is running "
                "and that this container shares the data-net network with a container "
                "aliased 'qdrant' (or point QDRANT_HOST elsewhere). Ingest and query will "
                "fail until it is reachable.",
                self.qdrant_host,
                exc,
            )
            return False
        if not 200 <= status < 300:
            logger.error(
                "Qdrant readiness probe at {} returned HTTP {}; ingest and query may fail.",
                url,
                status,
            )
            return False
        logger.info("Qdrant reachable at {}", self.qdrant_host)
        return True

    def reconcile_quantization(self) -> int:
        """Best-effort upgrade of existing collections to the configured quantization.

        Runs once at application startup, after :meth:`probe_qdrant`. Add-only:
        when quantization is disabled (``QDRANT_QUANTIZATION=none``) this is a
        no-op — it never strips quantization from a collection, and a
        deliberately configured non-TurboQuant family is never overwritten.
        Collections without dense vector params are skipped. Every failure is
        logged and swallowed; startup must never block on this.

        Returns:
            int: Number of collections whose quantization config was updated.
        """
        target = build_quantization_config()
        if target is None:
            return 0
        updated = 0
        try:
            collections = self.qdrant_client.get_collections().collections
        except Exception as exc:
            logger.warning("Quantization reconcile skipped; could not list collections: {}", exc)
            return 0
        for entry in collections:
            name = getattr(entry, "name", None)
            if not name:
                continue
            try:
                info = self.qdrant_client.get_collection(name)
                params = getattr(getattr(info, "config", None), "params", None)
                if not getattr(params, "vectors", None):
                    continue
                current = getattr(info.config, "quantization_config", None)
                if current is not None:
                    if not isinstance(current, qdrant_models.TurboQuantization):
                        # A different quantization family was configured
                        # deliberately; do not overwrite it from a startup pass.
                        continue
                    if _quantization_matches(current, target):
                        continue
                self.qdrant_client.update_collection(collection_name=name, quantization_config=target)
                updated += 1
                logger.info("Enabled TurboQuant quantization on collection '{}'.", name)
            except Exception as exc:
                logger.warning("Quantization reconcile failed for collection '{}': {}", name, exc)
        return updated

    def probe_sparse_endpoint(self) -> None:
        """Verify the sparse endpoint answers before an ingest run starts.

        Sparse encoding is not fail-soft: a transport failure partway
        through an ingest would write dense-only points into a hybrid
        collection and corrupt it. Probing once up front converts that
        into a clean, actionable job failure.

        No-op when hybrid retrieval is disabled.

        Raises:
            RuntimeError: When hybrid is enabled and the configured sparse
                endpoint cannot be reached.
        """
        if not self.enable_hybrid:
            return

        encoder = self._build_sparse_encoder()
        try:
            encoder.encode_texts(["ping"])
        except Exception as exc:
            base = self.sparse_client_config.api_base if self.sparse_client_config else "<unset>"
            logger.error("Sparse endpoint probe failed against {}: {}", base, exc)
            raise RuntimeError(
                f"Hybrid retrieval is enabled but the sparse endpoint at {base} is unreachable: {exc}. "
                "Point SPARSE_API_BASE at a reachable sparse service (the embed-only shape listens on "
                "http://embed-only:8000), or set ENABLE_HYBRID=false to ingest dense-only."
            ) from exc

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
        if self.enable_hybrid:
            sparse_encoder = self._build_sparse_encoder()
            vector_store_kwargs["sparse_doc_fn"] = sparse_encoder.encode_texts
            vector_store_kwargs["sparse_query_fn"] = sparse_encoder.encode_texts

        return QdrantVectorStore(**vector_store_kwargs)

    def _storage_context(self, vector_store: QdrantVectorStore) -> StorageContext:
        """Creates the storage context for document embeddings.

        Args:
            vector_store (QdrantVectorStore): The vector store for document embeddings.

        Returns:
            StorageContext: The created storage context.
        """
        kv_store = self._build_kv_store()
        doc_store = KVDocumentStore(kvstore=kv_store, batch_size=self.docstore_batch_size)

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

    def _build_ingest_manifest(self, collection: str | None = None) -> IngestManifest | NullIngestManifest:
        """Build the per-collection ingestion manifest.

        Returns a :class:`NullIngestManifest` no-op stub when
        ``INGEST_MANIFEST_ENABLED`` is false, so callers can use the
        manifest unconditionally without None-checks. Otherwise the
        manifest lives at
        ``{qdrant_src_dir}/{collection}/{collection}_ingest_manifest.db``,
        sharing the same parent directory as the SQLite KV store but
        a separate file (different access patterns: frequent updates
        vs. blob KV).

        Args:
            collection: Optional collection name override.  When
                *None* the current ``qdrant_collection`` is used.

        Returns:
            IngestManifest | NullIngestManifest: The manifest instance.
        """
        if not self.ingest_manifest_enabled:
            return NullIngestManifest()
        target = str(collection or self.qdrant_collection or "").strip()
        if not target:
            return NullIngestManifest()
        db_path = self.qdrant_src_dir / target / f"{target}_ingest_manifest.db"
        return IngestManifest(
            db_path=db_path,
            max_retries=self.docstore_max_retries,
            retry_backoff_seconds=self.docstore_retry_backoff_seconds,
            retry_backoff_max_seconds=self.docstore_retry_backoff_max_seconds,
        )

    def _build_ingestion_pipeline(
        self,
        progress_callback: Callable[[str], None] | None = None,
        *,
        ner: bool | None = None,
        hate_speech: bool | None = None,
    ) -> DocumentIngestionPipeline:
        """Instantiate a document ingestion pipeline using current settings.

        Args:
            progress_callback (Callable[[str], None] | None): Optional callback for
                reporting ingestion progress.
            ner (bool | None): Per-request NER override; ``None`` keeps the
                env default (``NER_ENABLED``).
            hate_speech (bool | None): Per-request hate-speech override;
                ``None`` keeps the env default (``ENABLE_HATE_SPEECH_DETECTION``).

        Returns:
            DocumentIngestionPipeline: The instantiated ingestion pipeline.

        Raises:
            ValueError: If data_dir is None.
        """
        if self.data_dir is None:
            logger.error("ValueError: data_dir cannot be None for ingestion pipeline.")
            raise ValueError("data_dir cannot be None for ingestion pipeline.")

        hate_speech_enabled = load_hate_speech_env().enabled if hate_speech is None else hate_speech
        ner_enabled = self.ner_enabled if ner is None else ner
        use_llm_ner = ner_enabled and self.openai_inference_provider.lower() in {"openai"}

        shared_text_model: OpenAI | None = None
        if use_llm_ner or hate_speech_enabled:
            shared_text_model = self.text_model

        ner_model = shared_text_model if use_llm_ner else None
        hate_speech_model = shared_text_model if hate_speech_enabled else None

        if self._image_ingestion_service is None:
            self._image_ingestion_service = ImageIngestionService()

        return DocumentIngestionPipeline(
            data_dir=self.data_dir,
            ner_model=ner_model,
            progress_callback=progress_callback,
            hate_speech_model=hate_speech_model,
            ner_override=ner,
            hate_speech_override=hate_speech,
            openai_inference_provider=self.openai_inference_provider,
            target_collection=self.qdrant_collection,
            image_ingestion_service=self._image_ingestion_service,
        )

    def _image_collection_name(self) -> str:
        """Return the ``_images`` companion collection name for the active collection."""
        if self._image_ingestion_service is None:
            self._image_ingestion_service = ImageIngestionService()
        return self._image_ingestion_service._resolve_collection_name(self.qdrant_collection)

    def _fetch_posting_entity_nodes(self, posting_uuid: str, *, exclude_node_ids: set[str]) -> list[NodeWithScore]:
        """Fetch a posting's sibling artifacts across both collections.

        Gathers transcript/text nodes (main collection) and image/keyframe
        caption nodes (``_images``) whose payload carries ``posting_uuid``, so a
        post and its media surface together. Fail-soft: scroll errors yield
        whatever was collected so far.

        Args:
            posting_uuid (str): The posting UUID link key.
            exclude_node_ids (set[str]): Node ids already present in the result
                set (skip to avoid duplicates).

        Returns:
            list[NodeWithScore]: Sibling nodes with score ``None``.
        """
        if not posting_uuid:
            return []
        # Main collection: OR on posting_uuid OR reference_metadata.uuid so that
        # posting TEXT nodes (which carry the link only as reference_metadata.uuid)
        # are collected even when retrieval started from a media/transcript hit.
        main_flt = qdrant_models.Filter(
            should=[
                qdrant_models.FieldCondition(key="posting_uuid", match=qdrant_models.MatchValue(value=posting_uuid)),
                qdrant_models.FieldCondition(
                    key="reference_metadata.uuid", match=qdrant_models.MatchValue(value=posting_uuid)
                ),
            ]
        )
        # Images companion: top-level posting_uuid is always populated on image payloads.
        images_flt = qdrant_models.Filter(
            must=[qdrant_models.FieldCondition(key="posting_uuid", match=qdrant_models.MatchValue(value=posting_uuid))]
        )
        collected: list[NodeWithScore] = []
        image_collection = self._image_collection_name()
        targets = [(self.qdrant_collection, "text"), (image_collection, "image")]
        for collection_name, kind in targets:
            if not collection_name or not qdrant_collection_exists(self.qdrant_client, collection_name):
                continue
            flt = images_flt if collection_name == image_collection else main_flt
            try:
                points, _ = self.qdrant_client.scroll(
                    collection_name=collection_name,
                    scroll_filter=flt,
                    limit=64,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as exc:
                logger.warning("Link-following scroll failed for {}: {}", collection_name, exc)
                continue
            for point in points:
                point_id = str(getattr(point, "id", ""))
                if not point_id or point_id in exclude_node_ids:
                    continue
                payload = dict(getattr(point, "payload", {}) or {})
                if kind == "image":
                    text = str(payload.get("llm_description") or "").strip()
                    tags = payload.get("llm_tags")
                    if isinstance(tags, list) and tags:
                        text = f"{text}\n\nTags: {', '.join(str(t) for t in tags)}".strip()
                else:
                    text = str(payload.get("text") or "").strip()
                    if not text:
                        raw = payload.get("_node_content")
                        if isinstance(raw, str) and raw:
                            try:
                                parsed = json.loads(raw)
                                text = str(parsed.get("text") or "").strip() if isinstance(parsed, dict) else ""
                            except (json.JSONDecodeError, ValueError):
                                text = ""
                if not text:
                    continue
                exclude_node_ids.add(point_id)
                collected.append(NodeWithScore(node=TextNode(id_=point_id, text=text, metadata=payload), score=None))
        return collected

    def _image_query_for_clip(self, query: str) -> str:
        """Render ``query`` in the only language the CLIP text tower understands.

        The deployed CLIP checkpoint (``openai/clip-vit-base-patch32``) has an
        English-only text tower, so a German query embeds near-degenerately and
        the nearest images come back essentially at random. Translating first
        restores the ranking. English deployments skip the round-trip entirely,
        and a translation outage degrades to the untranslated query rather than
        dropping the image lane.

        Args:
            query (str): The user's original query.

        Returns:
            str: The query to embed with CLIP.
        """
        if (self.language_code or "en").lower().startswith("en"):
            return query
        result = translate_text(query, target_lang="en")
        translated = (result.translation or "").strip() if result.ok else ""
        if not translated:
            logger.warning("Image query translation unavailable; embedding the untranslated query.")
            return query
        return translated

    def _image_config_value(self, name: str, default: Any) -> Any:
        """Read one image-lane setting, tolerating an unbuilt image service.

        The service owns the loaded ``ImageIngestionConfig``, but it is
        constructed lazily on first retrieval — and the query engine is built
        before that.

        Args:
            name (str): Config field name.
            default (Any): Value to use when the service is not built yet.

        Returns:
            Any: The configured value, or ``default``.
        """
        config = getattr(self._image_ingestion_service, "img_ingestion_config", None)
        if config is None:
            return default
        return getattr(config, name, default)

    def _image_relevance_floor(self) -> float:
        """Return the reranker score an image caption must reach to surface.

        Returns:
            float: ``IMAGE_RERANK_MIN_SCORE``, or its default when the image
            service has not been constructed yet.
        """
        return float(self._image_config_value("rerank_min_score", DEFAULT_IMAGE_RERANK_MIN_SCORE))

    def _retrieve_image_nodes(
        self,
        query: str,
        *,
        top_k: int,
        metadata_filter_rules: Sequence[Any] | None = None,
    ) -> list[NodeWithScore]:
        """Retrieve image captions as retrieval nodes.

        This is the image half of :class:`MultimodalRetriever`. CLIP generates
        candidates from an English rendering of the query (its text tower is
        English-only); the caption becomes the node's body, so the reranker
        downstream can score images and text chunks on one scale and the
        generator can cite an image like any other source.

        Relevance is *not* decided here — that is
        :class:`ImageRelevanceFloorPostprocessor`'s job, after the rerank.

        Args:
            query (str): The user's original query, untranslated.
            top_k (int): How many CLIP candidates to draw.
            metadata_filter_rules (Sequence[Any] | None): Optional raw request
                filters, applied in memory (the companion collection is not
                queried through the vector-store filter path).

        Returns:
            list[NodeWithScore]: Caption nodes carrying the image payload as
            metadata, scored with raw CLIP similarity. Empty on any outage —
            the lane degrades, it does not fail the query.
        """
        if not query.strip() or not self.qdrant_collection:
            return []
        if self._image_ingestion_service is None:
            self._image_ingestion_service = ImageIngestionService()

        try:
            image_collection = self._image_ingestion_service._resolve_collection_name(self.qdrant_collection)
        except Exception:
            return []
        if not qdrant_collection_exists(self.qdrant_client, image_collection):
            return []

        try:
            matches = self._image_ingestion_service.query_similar_images_by_text(
                query_text=self._image_query_for_clip(query),
                top_k=top_k,
                source_collection=self.qdrant_collection,
            )
        except Exception as exc:
            logger.warning("Image source retrieval failed: {}", exc)
            return []

        nodes: list[NodeWithScore] = []
        seen: set[str] = set()
        for payload in matches:
            if metadata_filter_rules and not matches_metadata_filters(payload, metadata_filter_rules):
                continue

            image_id = str(payload.get("image_id") or "").strip()
            if image_id:
                if image_id in seen:
                    continue
                seen.add(image_id)

            caption = RAG._image_caption_text(payload)
            if not caption:
                # An uncaptioned image carries no evidence a reader or a
                # reranker could judge, so it is not a source.
                continue

            nodes.append(self._image_caption_node(payload, caption))
        return nodes

    def _image_caption_node(self, payload: dict[str, Any], caption: str) -> NodeWithScore:
        """Wrap one image payload as a retrieval node.

        Args:
            payload (dict[str, Any]): The `_images` companion point payload.
            caption (str): The assembled caption body.

        Returns:
            NodeWithScore: The caption node, marked as image-lane and scored
            with the raw CLIP similarity.
        """
        raw_score = payload.get("score")
        metadata = {key: value for key, value in payload.items() if key != "score"}
        metadata[IMAGE_LANE_METADATA_KEY] = True
        # Label the shape for the prompt the way every reader does, and give
        # the model the file name -- the image payload spells it differently
        # (``source_path`` / ``file_name``) than the whitelist expects.
        metadata.setdefault("docint_doc_kind", "image")
        filename = RAG._source_from_payload(collection=self.qdrant_collection, payload=payload).get("filename")
        if filename:
            metadata.setdefault("filename", filename)

        node = TextNode(text=caption, metadata=metadata)
        node_id = str(payload.get("node_id") or payload.get("image_id") or "").strip()
        if node_id:
            node.id_ = node_id
        # The payload is the citation record, not prompt material: an image
        # point carries ingest paths and caption/tag artefacts. Hide everything
        # the chat prompt has no use for, independently of whether
        # ``ParentContextPostprocessor`` (which applies the same whitelist) is
        # enabled for this collection.
        node.excluded_llm_metadata_keys = sorted(k for k in metadata if k not in LLM_VISIBLE_METADATA_KEYS)
        return NodeWithScore(node=node, score=float(raw_score) if isinstance(raw_score, (int, float)) else None)

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

    def _resplit_vector_nodes(self, nodes: list[BaseNode]) -> tuple[list[BaseNode], list[BaseNode]]:
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
        for node, text in zip(nodes_to_embed, texts_to_embed, strict=False):
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
            counter_state = "tokenizer" if self._embed_token_counter is not None else "char-ratio"
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
        # after 4 minutes of stalled batch processing. The slicing is
        # kept inline because we need parallel slicing of
        # ``nodes_to_embed`` and ``texts_to_embed`` in lockstep, which
        # the generic :func:`chunk_nodes` helper cannot express.
        batch_size = max(1, self.embed_batch_size)
        for start in range(0, len(nodes_to_embed), batch_size):
            embed_batch = nodes_to_embed[start : start + batch_size]
            chunk_texts = texts_to_embed[start : start + batch_size]
            self._assert_embed_payloads_fit_budget(embed_batch, chunk_texts)
            chunk_embeddings = cast(
                list[list[float]],
                get_embeddings(chunk_texts),
            )
            for node, embedding in zip(embed_batch, chunk_embeddings, strict=False):
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
            embed_batch = nodes_to_embed[start : start + batch_size]
            chunk_texts = texts_to_embed[start : start + batch_size]
            self._assert_embed_payloads_fit_budget(embed_batch, chunk_texts)
            chunk_embeddings = cast(
                list[list[float]],
                await aget_embeddings(chunk_texts),  # pyrefly: ignore[not-async]  # getattr'd coroutine fn
            )
            for node, embedding in zip(embed_batch, chunk_embeddings, strict=False):
                node.embedding = embedding

        return vector_nodes, docstore_nodes

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
        non_vector_candidates = [node for node in batch if id(node) not in candidate_ids]
        return non_vector_candidates + list(docstore_nodes)

    def _write_search_text(self, nodes: list[BaseNode]) -> None:
        """Write each persisted node's text to its Qdrant point for search.

        Called after a successful insert, keyed by ``node_id`` — llama-index
        uses the node id as the Qdrant point id. Deliberately a payload-only
        write rather than node metadata: metadata is rendered into the
        embedding input and serialized into ``_node_content``, so stamping it
        there would embed every chunk's text twice and store a third copy of
        it. Fail-soft — search degrades to "needs a backfill", ingestion does
        not fail.

        Args:
            nodes (list[BaseNode]): Nodes just written to the vector store.
        """
        if not self.qdrant_collection or not nodes:
            return
        try:
            texts = {
                node.node_id: text
                for node in nodes
                if (text := node.get_content(metadata_mode=MetadataMode.NONE).strip())
            }
            if not texts:
                return
            write_search_text(self.qdrant_client, self.qdrant_collection, texts)
        except Exception as exc:
            logger.warning(
                "search_text write skipped for {} node(s) in {}: {} — run `make search-index` to backfill.",
                len(nodes),
                self.qdrant_collection,
                exc,
            )

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

        batches = chunk_nodes(nodes, self.docstore_batch_size)
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
            persisted_batch = self._docstore_batch_for_persist(batch, vector_candidates, prepared_docstore_nodes)
            if persisted_batch:
                try:
                    self.index.docstore.add_documents(
                        persisted_batch,
                        allow_update=True,
                    )
                except Exception as exc:
                    logger.error(
                        "failed_persist_nodes | batch={}/{} collection={!r} error={!r} node_ids={}",
                        batch_no,
                        len(batches),
                        self.qdrant_collection,
                        exc,
                        [node.node_id for node in persisted_batch],
                    )
                    raise
            if prepared_vector_nodes:
                index = self.index
                # Retry safety invariant: ``_prepare_vector_nodes_for_insert``
                # has already attached an embedding to every vector node, so
                # llama-index's ``insert_nodes`` will not re-embed on a
                # retry attempt — the retry simply replays the Qdrant
                # upsert with the same point payloads.
                # B023: the lambda is consumed synchronously by
                # retry_with_backoff within this loop iteration, so the
                # late-binding of ``index`` / ``prepared_vector_nodes`` is
                # not actually a bug — bind them as defaults to silence.
                try:
                    retry_with_backoff(
                        "qdrant_insert_nodes",
                        lambda idx=index, nodes=prepared_vector_nodes: idx.insert_nodes(nodes),  # type: ignore[misc]
                        max_retries=self.docstore_max_retries,
                        initial_backoff=self.docstore_retry_backoff_seconds,
                        max_backoff=self.docstore_retry_backoff_max_seconds,
                        is_retryable=is_transient_qdrant_error,
                    )
                except Exception as exc:
                    logger.error(
                        "orphaned_kv_nodes | batch={}/{} collection={!r} error={!r} max_attempts={} node_ids={}",
                        batch_no,
                        len(batches),
                        self.qdrant_collection,
                        exc,
                        self.docstore_max_retries + 1,
                        [node.node_id for node in prepared_vector_nodes],
                    )
                    raise
                self._write_search_text(prepared_vector_nodes)

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

        batches = chunk_nodes(nodes, self.docstore_batch_size)
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
            persisted_batch = self._docstore_batch_for_persist(batch, vector_candidates, prepared_docstore_nodes)
            if persisted_batch:
                try:
                    self.index.docstore.add_documents(
                        persisted_batch,
                        allow_update=True,
                    )
                except Exception as exc:
                    logger.error(
                        "failed_persist_nodes | batch={}/{} collection={!r} error={!r} node_ids={}",
                        batch_no,
                        len(batches),
                        self.qdrant_collection,
                        exc,
                        [node.node_id for node in persisted_batch],
                    )
                    raise
            if prepared_vector_nodes:
                index = self.index

                # B023: see sync twin — closure consumed synchronously inside
                # aretry_with_backoff within this loop iteration; default-bind
                # the loop vars to silence the late-binding warning.
                async def _do_ainsert(
                    idx: Any = index,
                    nodes: Any = prepared_vector_nodes,
                ) -> None:
                    # See sync twin for the retry-safety invariant —
                    # nodes are pre-embedded by
                    # ``_aprepare_vector_nodes_for_insert`` so a retry
                    # replays the upsert without re-embedding.
                    await idx.ainsert_nodes(nodes)

                try:
                    await aretry_with_backoff(
                        "qdrant_ainsert_nodes",
                        _do_ainsert,
                        max_retries=self.docstore_max_retries,
                        initial_backoff=self.docstore_retry_backoff_seconds,
                        max_backoff=self.docstore_retry_backoff_max_seconds,
                        is_retryable=is_transient_qdrant_error,
                    )
                except Exception as exc:
                    logger.error(
                        "orphaned_kv_nodes | batch={}/{} collection={!r} error={!r} max_attempts={} node_ids={}",
                        batch_no,
                        len(batches),
                        self.qdrant_collection,
                        exc,
                        self.docstore_max_retries + 1,
                        [node.node_id for node in prepared_vector_nodes],
                    )
                    raise
                self._write_search_text(prepared_vector_nodes)

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
        node_id: str | None = None,
    ) -> dict[str, Any]:
        """Normalize a raw metadata/payload dictionary into a source dictionary.

        Args:
            collection (str): The Qdrant collection name associated with the payload.
            payload (dict[str, Any]): The raw point payload returned by Qdrant.
            score (float | None): Optional similarity score to include in the source.
            text_value (str | None): Optional pre-extracted text value to use instead of extracting from payload.
            node_id (str | None): Optional id of the node the payload came from.
                Becomes the source's ``id``, which is what makes a citation
                traceable back to one specific chunk.

        Returns:
            dict[str, Any]: A normalized source dictionary containing standardized fields for downstream processing.
        """
        origin = payload.get("origin") or {}
        # ``source_path`` is the image companion's own filename key; the
        # `_images` payload carries neither ``file_name`` nor ``file_path``.
        source_path = payload.get("source_path")
        filename = (
            origin.get("filename")
            or payload.get("file_name")
            or payload.get("filename")
            or payload.get("file_path")
            or (Path(source_path).name if isinstance(source_path, str) and source_path else None)
        )
        filetype = (
            origin.get("filetype")
            or origin.get("mimetype")
            or payload.get("filetype")
            or payload.get("mimetype")
            or payload.get("mime_type")
            or payload.get("file_type")
            or payload.get("file_format")
        )
        source_kind = payload.get("source") or payload.get("source_type") or payload.get("reader")
        # ``source_doc_id`` is the image companion's link back to the file the
        # image was extracted from; it is what makes the preview link resolve.
        file_hash = (
            origin.get("file_hash")
            or payload.get("file_hash")
            or payload.get("source_doc_id")
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
        if row_index is None and payload.get("docint_doc_kind") == "transcript_segment":
            # Transcript segments have no table.row_index but do carry a
            # sequential ``sentence_index``. Surface it as ``row`` so the
            # citation/dropdown header shows ``row=<index>`` instead of the
            # default ``row=None`` placeholder.
            row_index = payload.get("sentence_index")
        try:
            row_index = int(row_index) if row_index is not None else None
        except Exception:
            row_index = None

        resolved_text = text_value if text_value is not None else RAG._extract_payload_text(payload)
        if not resolved_text:
            # An image's evidence is its stored caption and tags -- there is no
            # chunk text on an `_images` point.
            resolved_text = RAG._image_caption_text(payload)
        preview_url: str | None = None
        if file_hash:
            preview_url = f"/sources/preview?collection={collection}&file_hash={file_hash}"

        src: dict[str, Any] = {
            "text": resolved_text,
            "preview_text": resolved_text[:280].strip(),
            "filename": filename,
            "filetype": filetype,
            "source": source_kind,
            "score": score,
        }
        # Citation identity. Without it every exported source renders as
        # "Chunk-ID: n/a" and two chunks of the same page are
        # indistinguishable, so an answer cannot be traced to its evidence.
        # An image's durable chunk identity is its content hash (``image_id``);
        # it plays the same role a reader-minted ``chunk_id`` does for text.
        chunk_id = payload.get("chunk_id") or payload.get("image_id")
        identity = node_id or payload.get("node_id") or chunk_id
        if identity:
            src["id"] = str(identity)
        if chunk_id:
            src["chunk_id"] = str(chunk_id)
        # The number the generator saw for this snippet
        # (``CitationNumberingPostprocessor``). Absent for sources that never
        # reached the prompt — image matches are retrieved after generation —
        # so the UI leaves those cards unnumbered rather than implying the
        # answer could have cited them.
        citation_index = payload.get("citation_index")
        if isinstance(citation_index, int):
            src["citation_index"] = citation_index
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
        posting_uuid = payload.get("posting_uuid")
        if posting_uuid:
            src["posting_uuid"] = posting_uuid
        # Image-only locators: which point in the `_images` companion this came
        # from, and where inside the page it sits.
        image_id = payload.get("image_id")
        if image_id:
            src["image_id"] = str(image_id)
        image_collection = payload.get("image_collection")
        if image_collection:
            src["image_collection"] = image_collection
        bbox = payload.get("bbox")
        if isinstance(bbox, dict):
            src["bbox"] = bbox
        return src

    @staticmethod
    def _extract_indexable_text(payload: dict[str, Any]) -> str:
        """Return the text a point should be searchable by.

        Document chunks keep their stored text. Image points fall back to
        their caption and tags: those live in the payload, and depending only
        on the serialized node would mark an image "without text" — silently
        unsearchable, and recorded as processed so it never retries.

        Args:
            payload (dict[str, Any]): A Qdrant point payload.

        Returns:
            str: The indexable text, or an empty string when there is none.
        """
        return RAG._extract_payload_text(payload) or RAG._image_caption_text(payload)

    @staticmethod
    def _image_caption_text(payload: dict[str, Any]) -> str:
        """Assemble an image's evidence body from its caption and tags.

        Args:
            payload (dict[str, Any]): An `_images` companion point payload.

        Returns:
            str: The caption, the caption plus its tags, the tags alone, or an
            empty string when the image carries neither.
        """
        description = str(payload.get("llm_description") or "").strip()
        tags_raw = payload.get("llm_tags")
        tags: list[str] = [str(tag) for tag in tags_raw] if isinstance(tags_raw, list) else []
        if not tags:
            return description
        tag_line = f"Tags: {', '.join(tags)}"
        return f"{description}\n\n{tag_line}" if description else tag_line

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
                            getattr(node, "node_id", None) or getattr(node, "id_", None),
                        )
                        break
        except Exception:
            payload = None

        if payload is None:
            try:
                recs = self.qdrant_client.retrieve(collection_name=self.qdrant_collection, ids=[node_id])
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
            logger.warning("Unable to initialize Qdrant client for hash lookup: {}", exc)
            return existing

        # Decide the missing-collection case via the API contract instead of
        # provoking a scroll failure and matching its message text — client or
        # server rewordings must not change control flow (issue #419).
        if not qdrant_collection_exists(self.qdrant_client, self.qdrant_collection):
            logger.debug(
                "Qdrant collection '{}' not found; skipping existing-hash check",
                self.qdrant_collection,
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
                # The collection can still vanish between the existence
                # pre-check above and this scroll (concurrent delete); a 404
                # is identified by its status code, never by message wording.
                not_found = isinstance(exc, UnexpectedResponse) and exc.status_code == 404
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
        """Materialize a VectorStoreIndex.

        If nodes are present in memory, create from nodes; otherwise, load from vector store.
        Also best-effort-indexes the ``posting_uuid`` payload field on the main collection
        so link-following scrolls run against an index rather than a full scan at scale.
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

        # Best-effort: index posting_uuid on the main collection so that
        # _fetch_posting_entity_nodes scrolls use an index rather than a full scan.
        # Idempotent (Qdrant ignores duplicate index creation) and fail-soft.
        try:
            self.qdrant_client.create_payload_index(
                collection_name=self.qdrant_collection,
                field_name="posting_uuid",
                field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
            )
        except Exception as idx_exc:
            logger.debug("posting_uuid index on {} skipped: {}", self.qdrant_collection, idx_exc)

        # Full-text search needs a lowercase prefix index on `search_text`.
        # Idempotent and fail-soft, like the posting_uuid index above.
        ensure_search_index(self.qdrant_client, self.qdrant_collection)

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

    def _infer_collection_profile(self) -> dict[str, Any]:
        """Infer whether the active collection is social/table heavy."""
        docs = self.list_documents()
        payloads = self._sample_collection_payloads(limit=96)
        social_payloads = [payload for payload in payloads if self._is_social_payload(payload)]
        table_docs = [doc for doc in docs if "max_rows" in doc]
        is_social_table = bool(social_payloads) and (
            len(docs) <= 3 or len(table_docs) == len(docs) or len(social_payloads) >= max(3, len(payloads) // 3)
        )
        coverage_unit = "documents"
        if is_social_table:
            coverage_unit = "posts"
            for payload in social_payloads:
                reference_metadata = self._extract_reference_metadata(payload) or {}
                if not isinstance(reference_metadata, dict) or not str(reference_metadata.get("text_id") or "").strip():
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
            base_filters (MetadataFilters | None): Original filters from the query engine, or None.
            extra_filters (list[MetadataFilter]): Additional filters that must be applied for
                retrieval, such as parent-context scoping.

        Returns:
            MetadataFilters | None: ``base_filters`` AND ``extra_filters`` combined, or None if
                neither produces any filter.
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
            raw_mode (str | None): Optional retrieval mode string from the call site (takes
                precedence over config). One of "auto", "default", "sparse", "hybrid", or "mmr".

        Returns:
            VectorStoreQueryMode: The resolved retrieval mode for this retrieval operation.
        """
        mode_value = str(raw_mode or self.vector_store_query_mode or "auto").strip().lower()
        if mode_value == "auto":
            mode_value = "hybrid" if self.enable_hybrid else "default"

        mode_map = {
            "default": VectorStoreQueryMode.DEFAULT,
            "sparse": VectorStoreQueryMode.SPARSE,
            "hybrid": VectorStoreQueryMode.HYBRID,
            "mmr": VectorStoreQueryMode.MMR,
        }
        resolved = mode_map.get(mode_value, VectorStoreQueryMode.DEFAULT)
        if resolved in {VectorStoreQueryMode.HYBRID, VectorStoreQueryMode.SPARSE} and not self.enable_hybrid:
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
            similarity_top_k (int | None): Optional override for the number of top similar results
                (takes precedence over config).
            retrieval_options (dict[str, Any] | None): Optional runtime overrides; recognized keys
                are ``vector_store_query_mode``, ``alpha``, ``sparse_top_k``, ``hybrid_top_k``, and
                ``parent_context_enabled``. Take precedence over config; used to adjust retrieval
                behavior per query.

        Returns:
            dict[str, Any]: Resolved settings for this retrieval, with keys:
                - ``similarity_top_k``: effective number of top similar results.
                - ``vector_store_query_mode``: resolved retrieval mode.
                - ``alpha``: hybrid fusion alpha (if applicable).
                - ``sparse_top_k``: number of top sparse results (if applicable).
                - ``hybrid_top_k``: number of top hybrid results (if applicable).
                - ``parent_context_enabled``: whether parent-context expansion is on.
                - ``label``: short label summarizing mode + parent-context status, for logs.
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

        The template carries a ``{prior_turn_context}`` placeholder that the
        chat loop binds per-turn via
        :meth:`llama_index.core.PromptTemplate.partial_format` when the caller
        passes a ``PriorTurn`` (see
        :meth:`docint.core.state.session_manager.SessionManager.chat`). On the
        default path the placeholder is partial-formatted with the sentinel
        below so the rendered prompt is well-formed even without a prior turn.

        Args:
            social_table (bool): Whether the active collection is social/table-heavy and needs
                instructions to preserve post-level distinctions during synthesis.

        Returns:
            PromptTemplate: Prompt template for grounded QA during response synthesis.
        """
        prompt = self.grounded_text_qa_prompt
        if social_table:
            prompt = (
                f"{prompt.strip()}\n\n"
                "When the context comes from social posts or table rows, keep each "
                "post distinct and avoid merging separate authors or timestamps.\n"
            )
        return PromptTemplate(prompt).partial_format(prior_turn_context="(no prior turn)")

    def _build_grounded_refine_template(self, *, social_table: bool) -> PromptTemplate:
        """Return the grounded refine prompt template for answer synthesis.

        Carries the same ``{prior_turn_context}`` placeholder as
        :meth:`_build_grounded_text_qa_template`; see that method for the
        partial-format contract.

        Args:
            social_table (bool): Whether the active collection is social/table-heavy and needs
                instructions to preserve post-level distinctions during synthesis.

        Returns:
            PromptTemplate: Prompt template for grounded refinement during response synthesis.
        """
        prompt = self.grounded_refine_prompt
        if social_table:
            prompt = (
                f"{prompt.strip()}\n\n"
                "For row-level social evidence, preserve distinctions between "
                "different posts even when they discuss the same topic.\n"
            )
        return PromptTemplate(prompt).partial_format(prior_turn_context="(no prior turn)")

    def _compute_parent_context_budget(self, *, social_table: bool) -> tuple[int, int]:
        """Compute the per-query chat budget available to parent-context expansion.

        The synthesizer splices ``node.text`` directly into
        ``{context_str}`` via the grounded templates
        (:meth:`_build_grounded_text_qa_template`,
        :meth:`_build_grounded_refine_template`). Both templates render
        at ~150-200 tokens with an empty ``context_str``; the refine
        path additionally includes the prior ``existing_answer``. We
        estimate both and subtract the larger one, plus
        ``openai_num_output`` and a small slack, from
        ``openai_ctx_window * safety_margin``.

        Args:
            social_table: Forwarded to the grounded template builders
                so the estimate reflects the template variant that
                will actually render at synthesis time.

        Returns:
            A ``(usable_tokens, per_hit_floor)`` pair. ``usable_tokens``
            is the total chat budget available to the postprocessor's
            greedy packer; ``per_hit_floor`` is the minimum window size
            when the packer has to truncate the last hit. Both are
            ``0`` when the context window is misconfigured (``<= 0``)
            so the postprocessor degrades to legacy unbounded behavior
            rather than refusing every query.
        """
        ctx_window = int(self.openai_ctx_window)
        if ctx_window <= 0:
            return 0, 0

        # Use the raw template strings (.template) rather than .format() —
        # different template variants use different placeholder names and
        # the static prose dominates the token count either way. Add a
        # fixed allowance for the rendered query + prior existing_answer
        # the refine path will splice in.
        qa_raw = getattr(
            self._build_grounded_text_qa_template(social_table=social_table),
            "template",
            "",
        )
        refine_raw = getattr(
            self._build_grounded_refine_template(social_table=social_table),
            "template",
            "",
        )
        # Cross-model proxy: ``embed_char_token_ratio`` is calibrated
        # against the embedding tokenizer, not the chat model's. The 64-
        # token slack below absorbs the drift and we'd rather over-estimate
        # the template overhead (shrinking the usable budget slightly) than
        # under-estimate it (overflowing the chat context).
        chat_ratio_proxy = float(self.embed_char_token_ratio or 3.5)
        # 400 tokens: user query + refine's prior ``existing_answer``.
        # 400 tokens: ``prior_turn_context`` block (orchestrator-supplied
        # prior assistant turn, bound via PromptTemplate.partial_format at
        # chat time). Sized for a typical grounded answer; we'd rather
        # over-reserve than overflow the chat context.
        query_answer_allowance_tokens = 800
        template_tokens = (
            max(
                estimate_tokens(qa_raw, chat_ratio_proxy),
                estimate_tokens(refine_raw, chat_ratio_proxy),
            )
            + query_answer_allowance_tokens
        )
        safety_margin = float(self.parent_context_safety_margin or 0.95)
        reserved = template_tokens + int(self.openai_num_output) + 64  # 64 = slack
        usable = max(0, int(ctx_window * safety_margin) - reserved)
        per_hit_floor = max(256, usable // max(1, int(self.rerank_top_n)))
        return usable, per_hit_floor

    def _build_retriever(
        self,
        *,
        metadata_filters: MetadataFilters | None = None,
        similarity_top_k: int | None = None,
        vector_store_kwargs: dict[str, Any] | None = None,
        retrieval_options: dict[str, Any] | None = None,
        metadata_filter_rules: Sequence[Any] | None = None,
        metadata_filters_active: bool = False,
    ) -> Any:
        """Build a retriever, optionally scoped by metadata filters.

        Args:
            metadata_filters (MetadataFilters | None): Optional request-scoped metadata filters.
            similarity_top_k (int | None): Optional override for retrieval depth.
            vector_store_kwargs (dict[str, Any] | None): Optional native vector-store query kwargs.
            retrieval_options (dict[str, Any] | None): Optional runtime overrides for retrieval mode,
                hybrid fusion, and parent-context expansion.
            metadata_filter_rules (Sequence[Any] | None): Raw request filters,
                used to post-filter image candidates in memory.
            metadata_filters_active (bool): Whether this request carries
                metadata filters at all. Together with ``metadata_filter_rules``
                it decides whether the image lane can run — see
                :meth:`_build_image_lane`.
        """
        if self.index is None:
            logger.error("RuntimeError: Index is not initialized.")
            raise RuntimeError("Index is not initialized. Cannot create retriever.")

        retrieval_settings = self._resolve_runtime_retrieval_settings(
            similarity_top_k=similarity_top_k,
            retrieval_options=retrieval_options,
        )
        # The internal condition is expressed twice — once per filter
        # representation — because either may be the one that executes:
        # QdrantVectorStore.query uses ``qdrant_filters`` *instead of* the
        # LlamaIndex filters when both are supplied.
        internal_filters: list[MetadataFilter] = []
        internal_conditions: list[qdrant_models.FieldCondition] = []
        if retrieval_settings["parent_context_enabled"]:
            internal_filters.append(
                MetadataFilter(
                    key="docint_hier_type",
                    value="fine",
                    operator=FilterOperator.EQ,
                )
            )
            internal_conditions.append(
                qdrant_models.FieldCondition(
                    key="docint_hier_type",
                    match=qdrant_models.MatchValue(value="fine"),
                )
            )

        merged_filters = self._merge_metadata_filters(metadata_filters, internal_filters)

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
        elif retrieval_settings["vector_store_query_mode"] == VectorStoreQueryMode.SPARSE:
            retriever_kwargs["sparse_top_k"] = retrieval_settings["sparse_top_k"]
        if vector_store_kwargs:
            # Copy before mutating: the caller owns this dict and may reuse it
            # across the text and image lanes of the same request.
            native_kwargs = dict(vector_store_kwargs)
            if internal_conditions and native_kwargs.get("qdrant_filters") is not None:
                native_kwargs["qdrant_filters"] = merge_qdrant_filters(
                    native_kwargs["qdrant_filters"],
                    internal_conditions,
                )
            retriever_kwargs["vector_store_kwargs"] = native_kwargs
        text_retriever = self.index.as_retriever(**retriever_kwargs)
        image_lane = self._build_image_lane(
            metadata_filter_rules=metadata_filter_rules,
            metadata_filters_active=metadata_filters_active,
        )
        if image_lane is None:
            return text_retriever
        return MultimodalRetriever(text_retriever=text_retriever, image_lane=image_lane)

    def _build_image_lane(
        self,
        *,
        metadata_filter_rules: Sequence[Any] | None,
        metadata_filters_active: bool,
    ) -> Callable[[str], list[NodeWithScore]] | None:
        """Build the image half of the retriever, or decline to.

        The lane stands down when the request carries metadata filters that
        did not reach the runtime as raw rules: image candidates are filtered
        in memory, so without the rules the only honest options are unfiltered
        images or none, and none is the safe one.

        Args:
            metadata_filter_rules (Sequence[Any] | None): Raw request filters.
            metadata_filters_active (bool): Whether filters are in play.

        Returns:
            Callable[[str], list[NodeWithScore]] | None: The lane, or ``None``
            when images must not participate in this request.
        """
        image_filter_rules = metadata_filter_rules if metadata_filters_active else None
        if metadata_filters_active and not image_filter_rules:
            return None

        top_k = max(1, int(self._image_config_value("retrieve_top_k", DEFAULT_IMAGE_RETRIEVE_TOP_K)))

        def _lane(query: str) -> list[NodeWithScore]:
            """Retrieve image caption nodes for ``query``.

            Args:
                query (str): The user's original query.

            Returns:
                list[NodeWithScore]: Caption nodes from the image companion.
            """
            return self._retrieve_image_nodes(
                query,
                top_k=top_k,
                metadata_filter_rules=image_filter_rules,
            )

        return _lane

    def build_query_engine(
        self,
        *,
        metadata_filters: MetadataFilters | None = None,
        streaming: bool = False,
        vector_store_kwargs: dict[str, Any] | None = None,
        retrieval_options: dict[str, Any] | None = None,
        metadata_filter_rules: Sequence[Any] | None = None,
        metadata_filters_active: bool = False,
        scoped_node_ids: Sequence[str] | None = None,
    ) -> RetrieverQueryEngine:
        """Construct a query engine for the current index.

        Args:
            metadata_filters (MetadataFilters | None): Optional request-scoped metadata filters.
            streaming (bool): Whether the query engine should stream token output.
            vector_store_kwargs (dict[str, Any] | None): Optional native vector-store query kwargs.
            retrieval_options (dict[str, Any] | None): Optional runtime overrides for retrieval mode,
                hybrid fusion, and parent-context expansion.
            metadata_filter_rules (Sequence[Any] | None): Raw request filters,
                threaded through to the image lane, which post-filters its
                candidates in memory.
            metadata_filters_active (bool): Whether this request carries
                metadata filters at all.
            scoped_node_ids (Sequence[str] | None): When set, answer from
                exactly these chunks instead of retrieving. Selects the
                scoped engine, which drops every ranking postprocessor.
        """
        if self.index is None:
            self.create_index()
        if self.index is None:
            logger.error("RuntimeError: Index is not initialized.")
            raise RuntimeError("Index is not initialized. Cannot create query engine.")

        if scoped_node_ids:
            # A hand-picked set has nothing to rank, and every ranking
            # postprocessor adds, drops or reorders nodes — parent-context
            # expansion and link-following would silently widen the evidence,
            # the diversity cap and relevance floor would silently narrow it,
            # and reranking would spend an inference call reordering a set the
            # user already chose. Only citation numbering, which merely
            # numbers, survives.
            return RetrieverQueryEngine.from_args(
                retriever=_ScopedRetriever(rag=self, node_ids=scoped_node_ids),
                llm=self.post_retrieval_text_model,
                node_postprocessors=[CitationNumberingPostprocessor()],
                response_synthesizer=self._build_response_synthesizer(
                    streaming=streaming,
                    social_table=bool(self._infer_collection_profile().get("is_social_table")),
                ),
            )

        profile = self._infer_collection_profile()
        retrieval_settings = self._resolve_runtime_retrieval_settings(
            retrieval_options=retrieval_options,
        )
        node_postprocessors: list[BaseNodePostprocessor] = [
            LazyRerankerPostprocessor(rag=self),
            # Directly after the rerank, where image captions and text chunks
            # first carry comparable scores.
            ImageRelevanceFloorPostprocessor(min_score=self._image_relevance_floor()),
        ]
        if retrieval_settings["parent_context_enabled"] and self.index is not None:
            usable_tokens, per_hit_floor = self._compute_parent_context_budget(
                social_table=bool(profile.get("is_social_table")),
            )
            node_postprocessors.append(
                ParentContextPostprocessor(
                    docstore=self.index.docstore,
                    usable_tokens=usable_tokens,
                    per_hit_floor=per_hit_floor,
                    char_token_ratio=float(self.embed_char_token_ratio or 3.5),
                    # Always enforce the budget when we're building from a
                    # configured RAG instance. A ``usable_tokens`` value of
                    # ``0`` that somehow slips through (e.g. a misconfigured
                    # ``OPENAI_CTX_WINDOW`` smaller than the template
                    # overhead) then collapses safely to emitting sub-nodes
                    # rather than silently reverting to unbounded expansion.
                    budget_enforced=True,
                )
            )
        if bool(profile.get("is_social_table")):
            # Configurable via SOCIAL_SOURCE_DIVERSITY_LIMIT (RetrievalConfig)
            # — this knob used to be sourced from the (now-deleted) sampling
            # summarizer's social config, but this call site is on the
            # chat/retrieval path, not the summarizer.
            node_postprocessors.append(SocialSourceDiversityPostprocessor(diversity_limit=self.social_diversity_limit))
            node_postprocessors.append(LinkFollowingPostprocessor(rag=self))
        # Last: numbers the node set as the synthesizer will actually see it,
        # after every postprocessor above has added, dropped or reordered.
        node_postprocessors.append(CitationNumberingPostprocessor())

        return RetrieverQueryEngine.from_args(
            retriever=self._build_retriever(
                metadata_filters=metadata_filters,
                vector_store_kwargs=vector_store_kwargs,
                retrieval_options=retrieval_options,
                metadata_filter_rules=metadata_filter_rules,
                metadata_filters_active=metadata_filters_active,
            ),
            llm=self.post_retrieval_text_model,
            node_postprocessors=node_postprocessors,
            response_synthesizer=self._build_response_synthesizer(
                streaming=streaming,
                social_table=bool(profile.get("is_social_table")),
            ),
        )

    def _build_response_synthesizer(self, *, streaming: bool, social_table: bool) -> BaseSynthesizer:
        """Build the chat/query response synthesizer for the resolved mode.

        Constructed explicitly (rather than through ``from_args``'s
        ``response_mode``/``streaming`` knobs) so the docint streaming
        subclasses are used — upstream ``Refine`` buffers the whole answer in
        streaming mode (see :class:`_StreamingRefineMixin`). ``prompt_helper``
        is left to :class:`BaseSynthesizer`, which resolves it from the LLM
        metadata exactly like llama-index's synthesizer factory.

        Args:
            streaming (bool): Whether the synthesizer should stream tokens.
            social_table (bool): Whether the active collection is a social
                table, selecting the social prompt variants.

        Returns:
            BaseSynthesizer: A :class:`StreamingCompactAndRefine` for compact
                mode, else a :class:`StreamingRefine`.
        """
        response_mode = self._resolve_chat_response_mode()
        synthesizer_cls = StreamingCompactAndRefine if response_mode == ResponseMode.COMPACT else StreamingRefine
        return synthesizer_cls(
            llm=self.post_retrieval_text_model,
            streaming=streaming,
            text_qa_template=self._build_grounded_text_qa_template(social_table=social_table),
            refine_template=self._build_grounded_refine_template(social_table=social_table),
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
        node_id = getattr(node, "node_id", None) or getattr(node, "id_", None)
        return self._source_from_payload(
            collection=self.qdrant_collection,
            payload=meta,
            score=getattr(nws, "score", None),
            text_value=text_value,
            node_id=str(node_id) if node_id else None,
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

            label = source.get("filename") or source.get("file_hash") or source.get("source") or "source"
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
        """Normalize llama_index.core.Response and AgentChatResponse into a single payload.

        Handles:
        - response text (result.response or result.text)
        - source_nodes (list[NodeWithScore])
        - metadata differences.

        Args:
            query (str): The original query string.
            result (Any): The response object from the query engine.
            reason (str | None): Optional reasoning string.
            metadata_filters_active (bool): Whether request-scoped metadata
                filters were active for the retrieval.
            metadata_filter_rules (Sequence[Any] | None): Optional raw request
                filter payloads for post-filtering auxiliary image sources.
            retrieval_query (str | None): The query string actually used for retrieval (may
                differ from ``query`` after rewriting).
            coverage_unit (str | None): Unit label for coverage reporting (e.g. "rows",
                "chunks") used in the response metadata.
            retrieval_mode (str | None): Effective retrieval mode for this call (e.g. "hybrid",
                "sparse"); recorded in the response metadata for diagnostics.

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

        resp_text, captured = strip_reasoning(resp_text or "")
        reason = captured or reason

        # --- normalize source_nodes ---
        source_nodes = getattr(result, "source_nodes", None)
        if source_nodes is None and hasattr(result, "metadata"):
            # some Response variants tuck nodes under metadata
            meta = getattr(result, "metadata", {}) or {}
            source_nodes = meta.get("source_nodes")
        if not isinstance(source_nodes, list):
            source_nodes = cast(list[Any], [])

        sources: list[dict[str, Any]] = []
        for nws in source_nodes:
            normalized = self._source_from_node_with_score(nws)
            if normalized is not None:
                sources.append(normalized)

        # Images are not appended here. They retrieve through
        # ``MultimodalRetriever``, so they are already in ``source_nodes`` --
        # reranked against the text chunks, seen by the generator, and numbered
        # like any other source.
        sources = _attach_posting_group(sources)

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
        for page in iter_scroll(
            self.qdrant_client,
            collection_name=self.qdrant_collection,
            scroll_filter=qdrant_filter,
            page_size=100,
            error_context="NER sources",
        ):
            for point in page:
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
                    payload.get("node_id") or payload.get("id_") or str(getattr(point, "id", "") or "")
                )
                source["chunk_text"] = str(source.get("text") or "")
                sources.append(source)

        return sources

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
        # The `{target}_images` / `{target}_entities` companions are
        # supplementary metadata whose absence is tolerated.
        secondary_collections: list[str] = []
        if not target.endswith(HIDDEN_COLLECTION_SUFFIXES):
            secondary_collections.append(f"{target}_images")
            if qdrant_collection_exists(self.qdrant_client, f"{target}_entities"):
                secondary_collections.append(f"{target}_entities")

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

        # 1a. Always evict the deleted collection's cached retrieval handles so
        #     a later re-ingest of the same (owner, logical) -> same physical
        #     name does not reuse an index/query engine bound to the now-deleted
        #     Qdrant collection and its removed SQLite docstore. This is keyed by
        #     ``target`` directly, independent of the (possibly unscoped) active
        #     collection — the API delete path runs without a request scope.
        with self._retrieval_cache_lock:
            self._index_cache.pop(target, None)
            self._query_engine_cache.pop(target, None)

        # 1b. If the deleted collection happens to be the active one, also clear
        #     the rest of the per-collection runtime so the next query does not
        #     point at a tombstone.
        if target == self.qdrant_collection:
            self.qdrant_collection = ""
            self.docs.clear()
            self.nodes.clear()
            self.index = None
            self.query_engine = None
            self._image_ingestion_service = None
            self.reset_session_state()

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

                    def on_error(func: Callable[..., Any], path: str, _exc_info: Any) -> None:
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
            doc_store = KVDocumentStore(kvstore=kv_store, batch_size=self.docstore_batch_size)
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
        return stage_sources_to_qdrant(data_dir, self.qdrant_collection, self.qdrant_src_dir)

    def create_collection_if_missing(self) -> None:
        """Materialize the target Qdrant collection upfront if it does not yet exist.

        Ensures the active ``qdrant_collection`` is visible in Qdrant as soon
        as an ingest request begins, so the user can select it from the UI
        even when subsequent embedding batches fail to persist any nodes.

        Implementation note: this calls ``qdrant_client.create_collection``
        directly with the same dense + sparse named-vector schema that
        :class:`QdrantVectorStore` writes — replicating LlamaIndex's
        defaults rather than instantiating a ``QdrantVectorStore`` to do
        it. That class only creates the Qdrant collection lazily, from
        its ``add()`` method, the first time nodes are written, so
        constructing one here would not actually pre-create anything —
        hence the manual schema replication. NER (GLiNER) and sparse
        encoding are both remote services now (no in-process model
        weights), so this is purely about *when* the collection becomes
        visible, not about avoiding a local model load.

        When ``openai_dimensions`` is configured the vector size is taken
        from there; otherwise a single embed probe determines it. Probe
        failures (e.g., the embedding endpoint is unreachable) propagate
        unchanged so the API layer can surface a meaningful error rather
        than masking it as a silent zero-node ingest.

        Returns:
            None.

        Raises:
            ValueError: If ``qdrant_collection`` is unset.
        """
        if not self.qdrant_collection:
            raise ValueError("qdrant_collection must be set to create a collection")
        if qdrant_collection_exists(self.qdrant_client, self.qdrant_collection):
            return

        if self.openai_dimensions is not None:
            vector_size = int(self.openai_dimensions)
        else:
            probe_vector = self.embed_model.get_text_embedding("ping")
            vector_size = len(probe_vector)

        dense_params = qdrant_models.VectorParams(
            size=vector_size,
            distance=qdrant_models.Distance.COSINE,
        )

        if self.enable_hybrid:
            # Mirrors QdrantVectorStore: dense vector named "text-dense",
            # sparse vector named "text-sparse-new". No IDF modifier: sparse
            # encoding is now a remote call (RemoteSparseEncoder) and
            # fastembed is absent by design (tests/test_rag_sparse_gate.py
            # asserts it stays uninstalled), so qdrant_client's own
            # IDF_EMBEDDING_MODELS set — sourced from fastembed's model
            # registry — degrades to empty. No sparse model, including
            # bge-m3, can ever match it.
            modifier = None
            sparse_params = qdrant_models.SparseVectorParams(
                index=qdrant_models.SparseIndexParams(),
                modifier=modifier,
            )
            self.qdrant_client.create_collection(
                collection_name=self.qdrant_collection,
                vectors_config={QDRANT_DENSE_VECTOR_NAME: dense_params},
                sparse_vectors_config={QDRANT_SPARSE_VECTOR_NAME: sparse_params},
                quantization_config=build_quantization_config(),
            )
        else:
            self.qdrant_client.create_collection(
                collection_name=self.qdrant_collection,
                vectors_config=dense_params,
                quantization_config=build_quantization_config(),
            )
        logger.info(
            "Pre-created Qdrant collection '{}' (vector_size={}, hybrid={}).",
            self.qdrant_collection,
            vector_size,
            self.enable_hybrid,
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
            "No documents produced during ingestion of '{}'; cleaning up empty KV store and companion collections.",
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

        for companion in (f"{collection}_images", f"{collection}_entities"):
            if qdrant_collection_exists(self.qdrant_client, companion):
                try:
                    self.qdrant_client.delete_collection(companion)
                    logger.info(
                        "Deleted empty companion collection '{}' from Qdrant.",
                        companion,
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to delete companion collection '{}': {}",
                        companion,
                        exc,
                    )

    # --- Public API ---
    def ingest_docs(
        self,
        data_dir: str | Path,
        *,
        build_query_engine: bool = True,
        progress_callback: Callable[[str], None] | None = None,
        ner: bool | None = None,
        hate_speech: bool | None = None,
    ) -> None:
        """Ingest documents from the specified directory into the Qdrant collection.

        Args:
            data_dir (str | Path): The directory containing the documents to ingest.
            build_query_engine (bool): Whether to eagerly build the query engine after
                ingestion. Disable when running headless ingestion jobs to avoid
                loading large reranker/generation models. Defaults to True.
            progress_callback (Callable[[str], None] | None): Optional callback for
                reporting ingestion progress.
            ner (bool | None): Per-request NER override; ``None`` keeps the
                env default.
            hate_speech (bool | None): Per-request hate-speech override;
                ``None`` keeps the env default.

        Raises:
            EmptyIngestionError: When no documents/nodes were produced and the
                target collection did not previously exist. Triggers cleanup of
                the orphan SQLite KV files; uploaded source files are kept.
        """
        # Make the target Qdrant collection visible to the user before any
        # batch work begins, so it remains selectable even if every embedding
        # batch later fails. A probe failure here surfaces the embedding
        # outage immediately instead of masking it as a zero-node "success".
        self.create_collection_if_missing()

        # Fail before any file preparation, node parsing, or batch work: a
        # sparse transport failure discovered mid-run would already have
        # written dense-only points into a hybrid collection.
        self.probe_sparse_endpoint()

        prepared_dir = self._prepare_sources_dir(Path(data_dir) if isinstance(data_dir, str) else data_dir)
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
        # Ingestion builds the index directly rather than through
        # create_index(), so the payload index has to be ensured here too or a
        # freshly ingested collection gets search_text with no index over it —
        # searchable, but silently case-sensitive on non-ASCII text.
        ensure_search_index(self.qdrant_client, self.qdrant_collection)

        pipeline = self._build_ingestion_pipeline(progress_callback=progress_callback, ner=ner, hate_speech=hate_speech)
        manifest = self._build_ingest_manifest()
        manifest_completed = manifest.completed_files(self.qdrant_collection)
        existing_hashes = self._get_existing_file_hashes() | manifest_completed
        processed_hashes = set(existing_hashes)
        manifest_started: set[str] = set()
        manifest_in_flight: set[str] = set()
        image_ingestion_service = getattr(pipeline, "image_ingestion_service", None)
        core_pdf_reader = CorePDFPipelineReader(
            data_dir=prepared_dir,
            entity_extractor=pipeline.entity_extractor,
            ner_max_workers=pipeline.ner_max_workers,
            source_collection=self.qdrant_collection,
            image_ingestion_service=image_ingestion_service,
            hierarchical_node_parser=getattr(pipeline, "hierarchical_node_parser", None),
        )

        ingest_failures: list[tuple[set[str], str]] = []

        def _handle_batch_failure(in_flight: set[str], exc: BaseException) -> None:
            """Centralised handling for a per-batch persistence failure.

            Reraises immediately when ``ingest_fail_fast`` is true (CI
            mode); otherwise logs the failure with a structured marker,
            marks every in-flight file hash failed in the manifest, and
            records the failure for the end-of-run summary.
            """
            if self.ingest_fail_fast:
                raise exc
            failed_for_batch = set(in_flight)
            for fh in failed_for_batch:
                manifest.mark_failed(self.qdrant_collection, fh, repr(exc))
            ingest_failures.append((failed_for_batch, repr(exc)))
            logger.error(
                "failed_ingest_batch | collection={!r} file_hashes={} error={!r}",
                self.qdrant_collection,
                sorted(failed_for_batch),
                exc,
            )

        try:
            for docs, nodes, file_hash in core_pdf_reader.build(
                existing_hashes=processed_hashes, progress_callback=progress_callback
            ):
                core_docs += len(docs)
                if file_hash and file_hash not in manifest_started:
                    manifest.mark_started(self.qdrant_collection, file_hash)
                    manifest_started.add(file_hash)
                    manifest_in_flight.add(file_hash)
                if nodes:
                    try:
                        self._persist_node_batches(nodes)
                    except Exception as exc:
                        per_batch = {file_hash} if file_hash else set(manifest_in_flight)
                        _handle_batch_failure(per_batch, exc)
                        manifest_in_flight -= per_batch
                        continue
                    core_nodes += len(nodes)
                    persist_batches += len(chunk_nodes(nodes, self.docstore_batch_size))
                    processed_hashes.add(file_hash)
                    if file_hash:
                        manifest.mark_completed(self.qdrant_collection, file_hash)
                        manifest_in_flight.discard(file_hash)

            # PDFs are owned by the core pipeline reader and should not be
            # re-processed by the legacy ingestion path.
            processed_hashes.update(core_pdf_reader.discovered_hashes)

            # Process batches from the pipeline generator, persisting nodes as
            # soon as each enrichment micro-batch completes when supported.
            if hasattr(pipeline, "build_streaming") and callable(pipeline.build_streaming):
                if self.ingest_pipeline_overlap_enabled:
                    streaming_iter: Iterable[tuple[list[Document], list[BaseNode], set[str]]] = overlapped(
                        lambda: pipeline.build_streaming(processed_hashes),
                        queue_max_size=self.ingest_queue_max_size,
                    )
                else:
                    streaming_iter = pipeline.build_streaming(processed_hashes)
                for docs, nodes, completed_hashes in streaming_iter:
                    if docs:
                        streaming_docs += len(docs)
                    batch_hashes = _extract_node_file_hashes(nodes)
                    for fh in batch_hashes - manifest_started:
                        manifest.mark_started(self.qdrant_collection, fh)
                        manifest_started.add(fh)
                        manifest_in_flight.add(fh)
                    if nodes:
                        try:
                            self._persist_node_batches(nodes)
                        except Exception as exc:
                            _handle_batch_failure(batch_hashes, exc)
                            manifest_in_flight -= batch_hashes
                            continue
                        streaming_nodes += len(nodes)
                        persist_batches += len(chunk_nodes(nodes, self.docstore_batch_size))
                        enrich_batches += 1
                    if completed_hashes:
                        processed_hashes.update(completed_hashes)
                        for fh in completed_hashes:
                            manifest.mark_completed(self.qdrant_collection, fh)
                            manifest_in_flight.discard(fh)
            else:
                for docs, nodes in pipeline.build(processed_hashes):
                    if docs:
                        streaming_docs += len(docs)
                    if nodes:
                        self._persist_node_batches(nodes)
                        streaming_nodes += len(nodes)
                        persist_batches += len(chunk_nodes(nodes, self.docstore_batch_size))
        except Exception as exc:
            # Generator-level exception or fail-fast escape: mark every
            # remaining in-flight file failed and abort.
            for fh in manifest_in_flight:
                manifest.mark_failed(self.qdrant_collection, fh, repr(exc))
            raise
        finally:
            manifest.close()
            if ingest_failures:
                aggregated = sorted({fh for hashes, _ in ingest_failures for fh in hashes})
                logger.warning(
                    "Ingest finished with {} failed batch(es) "
                    "(skip-and-continue): collection={!r} failed_file_hashes={}",
                    len(ingest_failures),
                    self.qdrant_collection,
                    aggregated,
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
                "Effective retrieval k={} | top_n={} (embed/rerank served remotely)",
                eff_k,
                self.rerank_top_n,
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
        ner: bool | None = None,
        hate_speech: bool | None = None,
    ) -> None:
        """Asynchronously ingest documents from the specified directory into the Qdrant collection.

        Args:
            data_dir (str | Path): The directory containing the documents to ingest.
            build_query_engine (bool): Whether to build the query engine immediately
                after ingestion. Defaults to True.
            progress_callback (Callable[[str], None] | None): Optional callback for
                reporting ingestion progress.
            ner (bool | None): Per-request NER override; ``None`` keeps the
                env default.
            hate_speech (bool | None): Per-request hate-speech override;
                ``None`` keeps the env default.

        Raises:
            RuntimeError: If the index is not initialized for async ingestion.
            EmptyIngestionError: When no documents/nodes were produced and the
                target collection did not previously exist. Triggers cleanup of
                the orphan SQLite KV files; uploaded source files are kept.

        Warning:
            Unlike :meth:`ingest_docs`, this method does **not** call
            :meth:`probe_sparse_endpoint` before writing to a hybrid
            collection — it currently has zero production callers (tests
            only), so the missing probe has never mattered in practice.
            It carries the same corruption risk described there: a sparse
            endpoint that fails partway through would write dense-only
            points into a hybrid collection. Wiring this method to a real
            caller MUST add a ``self.probe_sparse_endpoint()`` call first.
        """
        # See ingest_docs: pre-create the Qdrant collection so it stays
        # selectable in the UI even when every embedding batch fails.
        self.create_collection_if_missing()

        prepared_dir = self._prepare_sources_dir(Path(data_dir) if isinstance(data_dir, str) else data_dir)
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
        # Ingestion builds the index directly rather than through
        # create_index(), so the payload index has to be ensured here too or a
        # freshly ingested collection gets search_text with no index over it —
        # searchable, but silently case-sensitive on non-ASCII text.
        ensure_search_index(self.qdrant_client, self.qdrant_collection)

        pipeline = self._build_ingestion_pipeline(progress_callback=progress_callback, ner=ner, hate_speech=hate_speech)
        manifest = self._build_ingest_manifest()
        manifest_completed = manifest.completed_files(self.qdrant_collection)
        existing_hashes = self._get_existing_file_hashes() | manifest_completed
        processed_hashes = set(existing_hashes)
        manifest_started: set[str] = set()
        manifest_in_flight: set[str] = set()
        image_ingestion_service = getattr(pipeline, "image_ingestion_service", None)
        core_pdf_reader = CorePDFPipelineReader(
            data_dir=prepared_dir,
            entity_extractor=pipeline.entity_extractor,
            ner_max_workers=pipeline.ner_max_workers,
            source_collection=self.qdrant_collection,
            image_ingestion_service=image_ingestion_service,
            hierarchical_node_parser=getattr(pipeline, "hierarchical_node_parser", None),
        )

        ingest_failures: list[tuple[set[str], str]] = []

        def _handle_batch_failure(in_flight: set[str], exc: BaseException) -> None:
            """Async ingest variant of :func:`_handle_batch_failure` (sync twin)."""
            if self.ingest_fail_fast:
                raise exc
            failed_for_batch = set(in_flight)
            for fh in failed_for_batch:
                manifest.mark_failed(self.qdrant_collection, fh, repr(exc))
            ingest_failures.append((failed_for_batch, repr(exc)))
            logger.error(
                "failed_ingest_batch | collection={!r} file_hashes={} error={!r}",
                self.qdrant_collection,
                sorted(failed_for_batch),
                exc,
            )

        try:
            for docs, nodes, file_hash in core_pdf_reader.build(
                existing_hashes=processed_hashes, progress_callback=progress_callback
            ):
                core_docs += len(docs)
                if file_hash and file_hash not in manifest_started:
                    manifest.mark_started(self.qdrant_collection, file_hash)
                    manifest_started.add(file_hash)
                    manifest_in_flight.add(file_hash)
                if nodes:
                    try:
                        await self._apersist_node_batches(nodes)
                    except Exception as exc:
                        per_batch = {file_hash} if file_hash else set(manifest_in_flight)
                        _handle_batch_failure(per_batch, exc)
                        manifest_in_flight -= per_batch
                        continue
                    core_nodes += len(nodes)
                    persist_batches += len(chunk_nodes(nodes, self.docstore_batch_size))
                    processed_hashes.add(file_hash)
                    if file_hash:
                        manifest.mark_completed(self.qdrant_collection, file_hash)
                        manifest_in_flight.discard(file_hash)

            processed_hashes.update(core_pdf_reader.discovered_hashes)

            # Process batches, persisting nodes as soon as each enrichment
            # micro-batch completes when supported.
            if hasattr(pipeline, "build_streaming") and callable(pipeline.build_streaming):
                for docs, nodes, completed_hashes in pipeline.build_streaming(processed_hashes):
                    if docs:
                        streaming_docs += len(docs)
                    batch_hashes = _extract_node_file_hashes(nodes)
                    for fh in batch_hashes - manifest_started:
                        manifest.mark_started(self.qdrant_collection, fh)
                        manifest_started.add(fh)
                        manifest_in_flight.add(fh)
                    if nodes:
                        try:
                            await self._apersist_node_batches(nodes)
                        except Exception as exc:
                            _handle_batch_failure(batch_hashes, exc)
                            manifest_in_flight -= batch_hashes
                            continue
                        streaming_nodes += len(nodes)
                        persist_batches += len(chunk_nodes(nodes, self.docstore_batch_size))
                        enrich_batches += 1
                    if completed_hashes:
                        processed_hashes.update(completed_hashes)
                        for fh in completed_hashes:
                            manifest.mark_completed(self.qdrant_collection, fh)
                            manifest_in_flight.discard(fh)
            else:
                for docs, nodes in pipeline.build(processed_hashes):
                    if docs:
                        streaming_docs += len(docs)
                    if nodes:
                        await self._apersist_node_batches(nodes)
                        streaming_nodes += len(nodes)
                        persist_batches += len(chunk_nodes(nodes, self.docstore_batch_size))
        except Exception as exc:
            for fh in manifest_in_flight:
                manifest.mark_failed(self.qdrant_collection, fh, repr(exc))
            raise
        finally:
            manifest.close()
            if ingest_failures:
                aggregated = sorted({fh for hashes, _ in ingest_failures for fh in hashes})
                logger.warning(
                    "Async ingest finished with {} failed batch(es) "
                    "(skip-and-continue): collection={!r} failed_file_hashes={}",
                    len(ingest_failures),
                    self.qdrant_collection,
                    aggregated,
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
                "Effective retrieval k={} | top_n={} (embed/rerank served remotely)",
                eff_k,
                self.rerank_top_n,
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
                metadata_filter_rules=metadata_filter_rules,
                metadata_filters_active=(metadata_filters is not None or bool(vector_store_kwargs)),
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
            engine = self.build_query_engine()
            self.query_engine = engine
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
            metadata_filters_active=(metadata_filters is not None or bool(vector_store_kwargs)),
            metadata_filter_rules=metadata_filter_rules,
        )
        retrieval_settings = self._resolve_runtime_retrieval_settings(
            retrieval_options=retrieval_options,
        )
        normalized["vector_query_mode"] = retrieval_settings["vector_store_query_mode"].value
        normalized["retrieval_profile"] = retrieval_settings["label"]
        normalized["parent_context_enabled"] = retrieval_settings["parent_context_enabled"]
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
                metadata_filter_rules=metadata_filter_rules,
                metadata_filters_active=(metadata_filters is not None or bool(vector_store_kwargs)),
            )
            if metadata_filters is not None or vector_store_kwargs or retrieval_options
            else self.query_engine
        )
        if engine is None:
            # See run_query for rationale: post-ingest warmup was
            # removed, so the default engine can be None on first use.
            # ``build_query_engine`` raises on its own failure modes;
            # no second None guard is needed here.
            logger.debug("Query engine not initialized; building lazily for run_query_async.")
            engine = self.build_query_engine()
            self.query_engine = engine
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
            metadata_filters_active=(metadata_filters is not None or bool(vector_store_kwargs)),
            metadata_filter_rules=metadata_filter_rules,
        )
        retrieval_settings = self._resolve_runtime_retrieval_settings(
            retrieval_options=retrieval_options,
        )
        normalized["vector_query_mode"] = retrieval_settings["vector_store_query_mode"].value
        normalized["retrieval_profile"] = retrieval_settings["label"]
        normalized["parent_context_enabled"] = retrieval_settings["parent_context_enabled"]
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
        """Invalidate cached NER, document, and hate-speech payloads.

        Args:
            collection (str | None): Optional collection name. If omitted, clears
                all per-collection caches across the instance.
        """
        if collection is None:
            self._ner_sources_cache.clear()
            self.ner_aggregate_cache.clear()
            self.ner_graph_cache.clear()
            self._parent_context_support_cache.clear()
            self._documents_cache.clear()
            self._hate_speech_cache.clear()
            self._resolved_index_cache.clear()
            return

        # Snapshot keys with list() before filtering. Collections are now
        # per-request/concurrent, so a parallel NER request may write these
        # caches while we scan; iterating the dict live would raise "dictionary
        # changed size during iteration". list(dict) materializes the keys
        # atomically under the GIL, and pop(..., None) tolerates a racing delete.
        stale_aggregate_keys = [key for key in list(self.ner_aggregate_cache) if key[0] == collection]
        for aggregate_key in stale_aggregate_keys:
            self.ner_aggregate_cache.pop(aggregate_key, None)
        stale_graph_keys: list[tuple[str, str, int, int]] = [
            key for key in list(self.ner_graph_cache) if key[0] == collection
        ]
        for graph_key in stale_graph_keys:
            self.ner_graph_cache.pop(graph_key, None)

        self._ner_sources_cache.pop(collection, None)
        self._parent_context_support_cache.pop(collection, None)
        self._documents_cache.pop(collection, None)
        self._hate_speech_cache.pop(collection, None)
        self._resolved_index_cache.pop(collection, None)

    def ensure_session_manager(self) -> SessionManager:
        """Ensure the SessionManager is initialized and return it.

        Use when a session-store helper (``list_sessions``,
        ``get_session_history``, ``delete_session``, ``export_session``, ...)
        is needed without starting a chat turn or building the query engine.
        ``start_session`` builds on top of this and additionally lazily
        constructs the query engine.

        Returns:
            SessionManager: The initialized session manager for this RAG instance.
        """
        if self.sessions is None:
            self.sessions = SessionManager(self)
        return self.sessions

    def ensure_report_manager(self) -> ReportManager:
        """Ensure the ReportManager is initialized and return it.

        Parallels :meth:`ensure_session_manager`. Used by the report API
        endpoints, which need owner-scoped report CRUD without starting a chat
        turn or building the query engine.

        Returns:
            ReportManager: The initialized report manager for this RAG instance.
        """
        if self.reports is None:
            self.reports = ReportManager(self)
        return self.reports

    def ensure_collection_owner_manager(self) -> CollectionOwnerManager:
        """Ensure the CollectionOwnerManager is initialized and return it.

        Parallels :meth:`ensure_report_manager`. On first successful use it
        backfills any pre-existing Qdrant collections (created before ownership
        shipped) to the configured default identity, so the current operator
        keeps access to legacy data. The backfill is best-effort and retried on
        subsequent calls until it succeeds once (e.g. if Qdrant was briefly
        unreachable at startup).

        Returns:
            CollectionOwnerManager: The initialized manager for this RAG instance.
        """
        if self.collection_owners is None:
            self.collection_owners = CollectionOwnerManager(self)
        if not self._collection_backfill_done:
            try:
                default_owner = load_principal_env().default_identity
                if default_owner:
                    self.collection_owners.backfill_legacy(self.list_collections(), default_owner)
                self._collection_backfill_done = True
            except Exception as exc:
                logger.warning("Collection ownership backfill skipped (will retry): {}", exc)
        return self.collection_owners

    def export_session(self, session_id: str | None = None, out_dir: str | Path = "session") -> Path:
        """Delegate session export to SessionManager.

        Args:
            session_id (str | None): The session ID to export. If None, exports the
                current session.
            out_dir (str | Path): The output directory for the exported session.

        Returns:
            Path: The path to the exported session file.
        """
        return self.ensure_session_manager().export_session(session_id=session_id, out_dir=out_dir)

    def start_session(self, session_id: str | None = None, owner: str | None = None) -> str:
        """Start or resume a chat session through SessionManager.

        ``SessionManager.start_session`` lazily builds the query engine when
        ``select_collection`` has reset it (see :meth:`run_query` for the same
        pattern), so this method only needs to ensure the SessionManager is
        wired up and delegate.

        Args:
            session_id (str | None): The session ID to start or resume. If None,
                a new session is created.
            owner (str | None): The owning principal for persisted sessions.

        Returns:
            str: The ID of the started or resumed session.
        """
        return self.ensure_session_manager().start_session(session_id, owner=owner)

    def chat(
        self,
        user_msg: str,
        *,
        session_id: str | None = None,
        owner: str | None = None,
        metadata_filters: MetadataFilters | None = None,
        metadata_filters_active: bool = False,
        metadata_filter_rules: Sequence[Any] | None = None,
        vector_store_kwargs: dict[str, Any] | None = None,
        prior_turn: PriorTurn | None = None,
        skip_query_rewrite: bool | None = None,
        scoped_node_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Proxy chat turns to SessionManager.

        Args:
            user_msg (str): The user's chat message.
            session_id (str | None): The conversation to append the turn to,
                threaded explicitly per request (mints a new one when ``None``).
            owner (str | None): The principal that owns the session.
            metadata_filters (MetadataFilters | None): Optional request-scoped
                metadata filters.
            metadata_filters_active (bool): Whether request-scoped metadata
                filters were active for the retrieval.
            metadata_filter_rules (Sequence[Any] | None): Optional raw request
                filter payloads for post-filtering auxiliary image sources.
            vector_store_kwargs (dict[str, Any] | None): Optional native
                vector-store query kwargs.
            prior_turn (PriorTurn | None): Prior user/assistant exchange. See
                :meth:`docint.core.state.session_manager.SessionManager.chat`
                for semantics.
            skip_query_rewrite (bool | None): Forwarded to
                :meth:`docint.core.state.session_manager.SessionManager.chat`;
                see there for semantics.
            scoped_node_ids (Sequence[str] | None): Hand-picked chunk ids the
                turn must answer from. When given, the session answers only
                from them and skips vector retrieval entirely.

        Returns:
            dict[str, Any]: The chat response data.
        """
        return self.ensure_session_manager().chat(
            user_msg,
            session_id=session_id,
            owner=owner,
            metadata_filters=metadata_filters,
            metadata_filters_active=metadata_filters_active,
            metadata_filter_rules=metadata_filter_rules,
            vector_store_kwargs=vector_store_kwargs,
            prior_turn=prior_turn,
            skip_query_rewrite=skip_query_rewrite,
            scoped_node_ids=scoped_node_ids,
        )

    def stream_chat(
        self,
        user_msg: str,
        *,
        session_id: str | None = None,
        owner: str | None = None,
        metadata_filters: MetadataFilters | None = None,
        metadata_filters_active: bool = False,
        metadata_filter_rules: Sequence[Any] | None = None,
        vector_store_kwargs: dict[str, Any] | None = None,
        prior_turn: PriorTurn | None = None,
        skip_query_rewrite: bool | None = None,
        scoped_node_ids: Sequence[str] | None = None,
    ) -> Any:
        """Proxy stream chat turns to SessionManager.

        Args:
            user_msg (str): The user's chat message.
            session_id (str | None): The conversation to append the turn to,
                threaded explicitly per request (mints a new one when ``None``).
            owner (str | None): The principal that owns the session.
            metadata_filters (MetadataFilters | None): Optional request-scoped
                metadata filters.
            metadata_filters_active (bool): Whether request-scoped metadata
                filters were active for the retrieval.
            metadata_filter_rules (Sequence[Any] | None): Optional raw request
                filter payloads for post-filtering auxiliary image sources.
            vector_store_kwargs (dict[str, Any] | None): Optional native
                vector-store query kwargs.
            prior_turn (PriorTurn | None): Prior user/assistant exchange. See
                :meth:`docint.core.state.session_manager.SessionManager.chat`
                for semantics.
            skip_query_rewrite (bool | None): Forwarded to
                :meth:`docint.core.state.session_manager.SessionManager.chat`;
                see there for semantics.
            scoped_node_ids (Sequence[str] | None): Hand-picked chunk ids the
                turn must answer from. When given, the session answers only
                from them and skips vector retrieval entirely.

        Returns:
            Any: A generator yielding response chunks.
        """
        return self.ensure_session_manager().stream_chat(
            user_msg,
            session_id=session_id,
            owner=owner,
            metadata_filters=metadata_filters,
            metadata_filters_active=metadata_filters_active,
            metadata_filter_rules=metadata_filter_rules,
            vector_store_kwargs=vector_store_kwargs,
            prior_turn=prior_turn,
            skip_query_rewrite=skip_query_rewrite,
            scoped_node_ids=scoped_node_ids,
        )

    def expand_query_with_graph_with_debug(self, query: str) -> tuple[str, dict[str, Any]]:
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
            anchor_texts = [str(ent.get("text") or "").strip() for ent in selected_anchors]
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

    def _summary_image_nodes_for_document(self, *, file_hash: str | None, top_k: int) -> list[NodeWithScore]:
        """Collect a document's stored images as summary evidence.

        A PDF's figures and a clip's keyframes live in the `_images` companion,
        which the summary's retrieval never touched — so a multimodal document
        was summarized as if it were text-only. They are scrolled by the
        document's own hash rather than retrieved by similarity: the summary
        asks "what is in this document", not "what matches this query".

        Args:
            file_hash (str | None): The parent document's content hash. Without
                it nothing ties an image to the document, and the lane declines.
            top_k (int): Cap on images returned, so a figure-heavy document
                cannot crowd out its own text evidence.

        Returns:
            list[NodeWithScore]: Caption nodes, scored ``None`` (unranked
            evidence, not query matches). Empty on any outage.
        """
        if not file_hash or not self.qdrant_collection or top_k <= 0:
            return []
        if self._image_ingestion_service is None:
            self._image_ingestion_service = ImageIngestionService()
        try:
            image_collection = self._image_ingestion_service._resolve_collection_name(self.qdrant_collection)
        except Exception:
            return []
        if not image_collection or not qdrant_collection_exists(self.qdrant_client, image_collection):
            return []

        try:
            points, _ = self.qdrant_client.scroll(
                collection_name=image_collection,
                scroll_filter=qdrant_models.Filter(
                    should=[
                        qdrant_models.FieldCondition(
                            key="source_doc_id", match=qdrant_models.MatchValue(value=file_hash)
                        ),
                        qdrant_models.FieldCondition(key="file_hash", match=qdrant_models.MatchValue(value=file_hash)),
                    ]
                ),
                limit=max(1, top_k),
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            logger.warning("Summary image scroll failed for '{}': {}", image_collection, exc)
            return []

        nodes: list[NodeWithScore] = []
        for point in points or []:
            payload = dict(getattr(point, "payload", {}) or {})
            caption = RAG._image_caption_text(payload)
            if not caption:
                continue
            point_id = str(getattr(point, "id", "") or "")
            if point_id:
                payload.setdefault("node_id", point_id)
            scored = self._image_caption_node(payload, caption)
            nodes.append(NodeWithScore(node=scored.node, score=None))
            if len(nodes) >= top_k:
                break
        return nodes

    def _iter_collection_points(self) -> Iterator[tuple[str, dict[str, Any]]]:
        """Scroll the active collection, yielding every point's id and payload.

        The tree summarizer's :func:`~docint.core.summary.units.partition_units`
        consumes this directly to discover map units, so every point in the
        collection — not just the ones a similarity query would surface — is
        represented once.

        Scroll errors are raised rather than warned (``iter_scroll``'s
        fail-soft default, which is right for the fail-soft NER/hate-speech
        aggregators). Here a mid-scroll Qdrant blip would shrink the universe
        silently: ``partition_units`` would see only the pages that arrived,
        every one of them would map, and the build would report
        ``coverage_ratio: 1.0`` before caching a fraction of the collection as
        the complete summary. Failing the summary job is the tested,
        recoverable outcome.

        Yields:
            tuple[str, dict[str, Any]]: ``(point_id, payload)`` pairs for
            every point with a dict payload in the active collection.

        Raises:
            Exception: Whatever the Qdrant client raises mid-scroll.
        """
        if not self.qdrant_collection:
            return
        for page in iter_scroll(
            self.qdrant_client,
            collection_name=self.qdrant_collection,
            page_size=256,
            with_payload=True,
            with_vectors=False,
            on_error="raise",
            error_context="tree summary points",
        ):
            for point in page:
                payload = getattr(point, "payload", None)
                if not isinstance(payload, dict):
                    continue
                yield str(getattr(point, "id", "")), payload

    def _fetch_unit_chunks(self, unit: MapUnit) -> list[UnitChunk]:
        """Fetch a map unit's member chunks, in the unit's reading order.

        Args:
            unit: The unit to fetch chunks for.

        Returns:
            list[UnitChunk]: One chunk per member point whose extracted text
            is non-empty, in ``unit.member_ids`` order. For a document unit
            keyed by content hash (``doc:{file_hash}``, as opposed to the
            hash-less ``doc:name:{filename}`` fallback), the document's
            stored figures/keyframes are appended as extra chunks — a
            document's images are evidence too (see
            :meth:`_summary_image_nodes_for_document`).
        """
        if not unit.member_ids or not self.qdrant_collection:
            return []
        try:
            points = self.qdrant_client.retrieve(
                collection_name=self.qdrant_collection,
                ids=list(unit.member_ids),
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            logger.warning("Tree summary chunk fetch failed for unit '{}': {}", unit.unit_key, exc)
            return []

        payload_by_id: dict[str, Any] = {}
        for point in points or []:
            point_id = str(getattr(point, "id", "") or "")
            if point_id:
                payload_by_id[point_id] = getattr(point, "payload", None)

        chunks: list[UnitChunk] = []
        for member_id in unit.member_ids:
            payload = payload_by_id.get(member_id)
            if not isinstance(payload, dict):
                continue
            text = payload_text(payload)
            if not text:
                continue
            chunks.append(UnitChunk(chunk_id=member_id, text=text))

        if unit.kind == "document" and unit.unit_key.startswith("doc:") and not unit.unit_key.startswith("doc:name:"):
            file_hash = unit.unit_key[len("doc:") :]
            for nws in self._summary_image_nodes_for_document(file_hash=file_hash, top_k=4):
                image_text = nws.node.get_content()
                if image_text:
                    chunks.append(UnitChunk(chunk_id=str(nws.node.node_id), text=image_text))

        return chunks

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
            diagnostics (dict[str, Any]): Diagnostic info (coverage ratio, uncovered documents).
            style_prompt (str): Instructions for the synthesis style/voice; appended to the
                final prompt to shape the model's response.

        Returns:
            str: A formatted string representing the final synthesis prompt.
        """
        coverage_ratio = float(diagnostics.get("coverage_ratio", 0.0) or 0.0)
        coverage_target = float(diagnostics.get("coverage_target", 0.0) or 0.0)
        coverage_unit = str(diagnostics.get("coverage_unit") or "documents")
        uncovered = diagnostics.get("uncovered_documents") or []
        uncovered_text = ", ".join(str(item) for item in uncovered) or "(none)"
        evidence_block = "\n\n".join(briefs) if briefs else "(no evidence extracted)"
        return self.grounded_collection_summary_prompt.format(
            coverage_unit=coverage_unit,
            coverage_ratio=f"{coverage_ratio:.2f}",
            coverage_target=f"{coverage_target:.2f}",
            uncovered_text=uncovered_text,
            style_prompt=style_prompt.strip(),
            evidence_block=evidence_block,
        )

    @staticmethod
    def _number_summary_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Stamp 1-based ``citation_index`` on summary sources in display order.

        Chat sources are numbered by :class:`CitationNumberingPostprocessor`
        inside the query engine's postprocessor chain; the summary paths
        retrieve through bare retrievers and build their source dicts by hand,
        so the field never exists and the SPA's shared citation badge (which is
        conditional on it) stays hidden. Numbers are assigned (and overwritten)
        in list order so they always match what the panel displays.

        Args:
            sources (list[dict[str, Any]]): Summary source dicts, in display order.

        Returns:
            list[dict[str, Any]]: The same list, with ``citation_index`` stamped.
        """
        for index, source in enumerate(sources, start=1):
            if isinstance(source, dict):
                source["citation_index"] = index
        return sources

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
            "summary_map_prompt": self.summary_map_prompt,
            "summary_fold_prompt": self.summary_fold_prompt,
            "summary_coverage_target": self.summary_coverage_target,
            "summary_final_source_cap": self.summary_final_source_cap,
            "summary_map_window_tokens": self.summary_config.map_window_tokens,
            "summary_reduce_fanin": self.summary_config.reduce_fanin,
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
        kv_store = self._summary_kv_store(collection=collection, allow_create=allow_create)
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
        kv_store = self._summary_kv_store(collection=collection, allow_create=allow_create)
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

    def _load_cached_collection_summary(self, *, refresh: bool) -> dict[str, Any] | None:
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
            sources = cast(list[dict[str, Any]], [])
        # Payloads cached before summary numbering shipped lack the field;
        # stamping at read time spares a forced refresh.
        self._number_summary_sources(sources)
        summary_diagnostics = payload.get("summary_diagnostics")
        if not isinstance(summary_diagnostics, dict):
            summary_diagnostics = cast(dict[str, Any], {})

        return {
            "query": self.summarize_prompt,
            "reasoning": None,
            "response": str(payload.get("response") or ""),
            "sources": sources,
            "summary_diagnostics": summary_diagnostics,
        }

    def _store_cached_collection_summary(
        self,
        payload: dict[str, Any],
        *,
        expected_revision: int | None = None,
    ) -> None:
        """Persist a collection summary payload in the dockv summary namespace.

        Summary and ingest jobs for one collection run concurrently by design
        (``create_if_idle`` keys idleness on ``(owner, physical, kind)``), so
        reading the revision at *write* time would stamp a build that started
        at revision R with whatever revision an ingest bumped it to
        mid-build — publishing a stale summary as current, and overwriting the
        newer summary that ingest just cached. Callers therefore capture the
        revision when the build starts and pass it as ``expected_revision``;
        this is a compare-and-set, skipping the write when the collection
        moved underneath the build. The revision key is always re-put with the
        *current* value, never the captured one, so the counter cannot roll
        backwards.

        Args:
            payload (dict[str, Any]): Summary payload to cache.
            expected_revision (int | None): Revision observed when the build
                started. ``None`` skips the check and stamps the current
                revision (for callers with no build window to protect).
        """
        kv_store = self._summary_kv_store()
        if kv_store is None:
            return

        revision = self._get_summary_revision()
        if expected_revision is not None and expected_revision != revision:
            logger.info(
                "Skipping summary cache write for '{}': collection changed during the build (revision {} -> {}).",
                self.qdrant_collection,
                expected_revision,
                revision,
            )
            return
        prompt_fingerprint = self._summary_prompt_fingerprint()
        sources = payload.get("sources")
        if not isinstance(sources, list):
            sources = cast(list[dict[str, Any]], [])
        summary_diagnostics = payload.get("summary_diagnostics")
        if not isinstance(summary_diagnostics, dict):
            summary_diagnostics = cast(dict[str, Any], {})

        cache_payload = {
            "revision": revision,
            "prompt_fingerprint": prompt_fingerprint,
            "generated_at": datetime.now(UTC).isoformat(),
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

    def build_tree_summary(self, progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
        """Build the map-reduce ("tree") summary for the currently scoped collection.

        Partitions every point in the active collection into map units (one
        per document, or one per social/table author-hour bucket — see
        :func:`~docint.core.summary.units.partition_units`), summarizes each
        unit independently through :class:`~docint.core.summary.tree.TreeSummarizer`
        (reusing a unit's cached map result whenever its content and the
        active prompts/model are unchanged), and folds the per-unit
        summaries into one final synthesis call.

        Every build that *completes* is persisted via
        :meth:`_store_cached_collection_summary` and prunes the map cache of
        units that no longer exist — including a build the LLM-call cap cut
        short (``summary_diagnostics.partial`` is ``True``, and the SPA shows
        an explicit notice) and a build over an empty collection. Only a build
        that fails mid-way propagates its exception and caches nothing.
        Refusing to cache a completed-but-partial build would make it
        unreachable: ``POST /summarize`` answers 200 solely from this cache,
        so the client's post-completion refetch would miss, silently queue
        another full build, and report a failure — forever, for an empty
        collection.

        The summary revision is captured *before* the build and handed to the
        cache write as a compare-and-set, so a concurrent ingest that lands
        mid-build cannot have this (now stale) summary stamped as current.

        Args:
            progress: Optional callback invoked ``(processed, total)`` after
                each unit resolves (cache hit, mapped, or failed) — useful
                for surfacing progress on a long build.

        Returns:
            dict[str, Any]: ``{"query", "reasoning", "response", "sources",
            "summary_diagnostics"}``.

        Raises:
            ValueError: If no collection is selected.
        """
        if not self.qdrant_collection:
            raise ValueError("No collection selected.")

        # Captured before any work: an ingest completing mid-build bumps this,
        # and the cache write must notice rather than stamp a stale summary.
        build_revision = self._get_summary_revision()

        units = partition_units(self._iter_collection_points())
        total_units = len(units)
        kinds = {unit.kind for unit in units}
        if kinds == {"social_bucket"}:
            coverage_unit = "posts"
        elif not kinds or kinds == {"document"}:
            # An empty collection has no units to derive a kind from; default
            # to "documents" — the conventional unit for an empty document
            # collection — rather than the meaningless "units".
            coverage_unit = "documents"
        else:
            coverage_unit = "units"

        if total_units == 0:
            empty_payload: dict[str, Any] = {
                "query": self.summarize_prompt,
                "reasoning": None,
                "response": "No documents available in the selected collection.",
                "sources": [],
                "summary_diagnostics": {
                    "total_documents": 0,
                    "covered_documents": 0,
                    "coverage_ratio": 0.0,
                    "uncovered_documents": [],
                    "coverage_target": self.summary_coverage_target,
                    "coverage_unit": coverage_unit,
                    "candidate_count": 0,
                    "deduped_count": 0,
                    "sampled_count": 0,
                    "partial": False,
                    "llm_calls": 0,
                },
            }
            # Cached like any other completed build: without this an empty
            # collection can never answer 200, so every /summarize call queues
            # a fresh job that finds nothing and reports failure, forever.
            self._store_cached_collection_summary(empty_payload, expected_revision=build_revision)
            return empty_payload

        kv_store = self._summary_kv_store()
        validator_suffix = f"{self._summary_prompt_fingerprint()}|{self.text_model_id}"
        cache_adapter = _KVMapCache(kv_store, validator_suffix=validator_suffix)
        cache: MapCache = cache_adapter
        covered_keys = cache_adapter.covered_keys

        def _complete(prompt: str) -> str:
            """Invoke the chat model and coerce its response to text."""
            return str(getattr(self.post_retrieval_text_model.complete(prompt), "text", "") or "")

        def _build_synthesis_prompt(briefs: list[str], diag: dict[str, Any]) -> str:
            """Render the final synthesis prompt from the tree's live diagnostics."""
            covered = int(diag.get("covered_units", 0) or 0)
            total = int(diag.get("total_units", total_units) or total_units)
            coverage_ratio = covered / total if total else 0.0
            # Approximate: `covered_keys` only tracks map-cache get/put
            # activity, so a unit that was mapped but deliberately not
            # cached (a cap-truncated result — see `_KVMapCache`'s
            # docstring) reads as uncovered here even though it produced a
            # result. This runs *during* the build, before `TreeSummarizer`
            # returns a `TreeSummaryResult` with per-unit results, so
            # `covered_keys` is the only signal available at this point.
            # The post-build diagnostics below recompute this from
            # `result.unit_results` and are the authoritative value served
            # to the client; this closure only feeds the LLM's synthesis
            # prompt, not `summary_diagnostics`.
            uncovered_labels = [unit.label for unit in units if unit.unit_key not in covered_keys][:20]
            return self._build_summary_synthesis_prompt(
                briefs=briefs,
                diagnostics={
                    "coverage_unit": coverage_unit,
                    "coverage_ratio": coverage_ratio,
                    "coverage_target": self.summary_coverage_target,
                    "uncovered_documents": uncovered_labels,
                },
                style_prompt=self.summarize_prompt,
            )

        summarizer = TreeSummarizer(
            complete=_complete,
            fetch_chunks=self._fetch_unit_chunks,
            map_prompt=self.summary_map_prompt,
            fold_prompt=self.summary_fold_prompt,
            build_synthesis_prompt=_build_synthesis_prompt,
            cache=cache,
            window_chars=self.summary_config.map_window_tokens * 4,
            reduce_fanin=self.summary_config.reduce_fanin,
            max_llm_calls=self.summary_config.max_llm_calls,
            progress=progress,
        )
        result = summarizer.build(units)

        covered_units = result.covered_units
        coverage_ratio = covered_units / total_units if total_units else 0.0
        # Authoritative post-build source: a unit is covered iff it produced
        # a `UnitMapResult` (cache hit, mapped, or cap-truncated), regardless
        # of whether that result was written to the map cache. `covered_keys`
        # (used above, mid-build, where `result` does not yet exist) instead
        # tracks map-cache get/put activity, which a deliberately-uncached
        # truncated unit never joins — that unit would otherwise be counted
        # in both `covered_units` (it has a `UnitMapResult`) and
        # `uncovered_labels` (`_KVMapCache` never saw it), making the
        # coverage banner self-contradictory.
        covered_result_keys = {unit_result.unit_key for unit_result in result.unit_results}
        uncovered_labels = [unit.label for unit in units if unit.unit_key not in covered_result_keys][:20]

        if covered_units == 0:
            response_text = "Unable to extract grounded evidence from the selected collection."
        else:
            response_text = result.response

        sources: list[dict[str, Any]] = []
        if covered_units:
            evidence_ids: list[str] = []
            for unit_result in result.unit_results[: self.summary_final_source_cap]:
                for evidence_id in unit_result.evidence_ids:
                    if evidence_id not in evidence_ids:
                        evidence_ids.append(evidence_id)

            # Qdrant's retrieve() does not promise the response is ordered
            # like the requested ids (the same reason _fetch_unit_chunks
            # re-orders by member_ids above) — so results are indexed by id
            # first and then emitted by walking evidence_ids in covered-unit
            # order, not in whatever order retrieve() happened to return them.
            payload_by_id: dict[str, Any] = {}
            for batch in chunk_nodes(evidence_ids, 200):
                try:
                    points = self.qdrant_client.retrieve(
                        collection_name=self.qdrant_collection,
                        ids=batch,
                        with_payload=True,
                        with_vectors=False,
                    )
                except Exception as exc:
                    logger.warning(
                        "Tree summary evidence retrieve failed for '{}': {}",
                        self.qdrant_collection,
                        exc,
                    )
                    continue
                for point in points or []:
                    point_id = str(getattr(point, "id", "") or "")
                    if point_id:
                        payload_by_id[point_id] = getattr(point, "payload", None)

            for evidence_id in evidence_ids:
                payload = payload_by_id.get(evidence_id)
                if not isinstance(payload, dict):
                    continue
                sources.append(
                    self._source_from_payload(
                        collection=self.qdrant_collection,
                        payload=payload,
                    )
                )
                if len(sources) >= self.summary_final_source_cap:
                    break
            self._number_summary_sources(sources)

        diagnostics = {
            "total_documents": total_units,
            "covered_documents": covered_units,
            "coverage_ratio": round(coverage_ratio, 4),
            "uncovered_documents": uncovered_labels,
            "coverage_target": self.summary_coverage_target,
            "coverage_unit": coverage_unit,
            "candidate_count": total_units,
            "deduped_count": covered_units,
            "sampled_count": len(sources),
            # A zero-covered build over a non-empty collection (the
            # `total_units == 0` case already returned above) produces the
            # bare "unable to extract grounded evidence" response_text above
            # with no per-unit diagnostics to explain it. Without `partial`,
            # `CoverageBanner` has nothing to flag and the non-answer looks
            # like a normal, complete summary until Refresh or the next
            # revision bump.
            "partial": result.partial or covered_units == 0,
            "llm_calls": result.llm_calls,
        }

        payload = {
            "query": self.summarize_prompt,
            "reasoning": None,
            "response": response_text,
            "sources": sources,
            "summary_diagnostics": diagnostics,
        }

        # This build completed, so it is cacheable — partial or not. The
        # honesty requirement is met by `partial` travelling with the payload
        # (through the cache, `SummaryDiagnosticsOut`, and the SPA's coverage
        # banner), not by withholding the result: a summary that is never
        # cached is never served, because `/summarize` answers 200 only from
        # here. A build that fails mid-way never reaches this line.
        self._store_cached_collection_summary(payload, expected_revision=build_revision)
        current_keys = {unit.unit_key for unit in units}
        for stale_key in cache_adapter.all_keys():
            if stale_key not in current_keys:
                cache_adapter.delete(stale_key)

        return payload

    def cached_collection_summary(self) -> dict[str, Any] | None:
        """Return the currently cached final summary for the active collection, if any.

        Thin public wrapper over :meth:`_load_cached_collection_summary`
        that never bypasses the cache — it is a pure read, distinct from
        :meth:`build_tree_summary` (which computes and stores a fresh one).

        Returns:
            dict[str, Any] | None: The cached payload, or ``None`` when
            nothing is cached or the cached entry is stale.

        Raises:
            ValueError: If no collection is selected.
        """
        if not self.qdrant_collection:
            raise ValueError("No collection selected.")
        return self._load_cached_collection_summary(refresh=False)

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
                logger.error("Failed to scroll collection '{}': {}", self.qdrant_collection, exc)
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
                        "file_hash": payload.get("file_hash") or origin.get("file_hash"),
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

                page = payload.get("page") or payload.get("page_number") or origin.get("page_no")

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
                end_sec = payload.get("end_seconds") or (payload.get("extra_metadata") or {}).get("end_seconds")
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

    def iter_documents(
        self,
        *,
        cursor: str | None = None,
        limit: int = 50,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Return one paginated slice of document records.

        The first call after a cache miss runs :meth:`list_documents` to build
        the per-collection result; subsequent calls slice the cached list. The
        cache is invalidated by :meth:`_invalidate_ner_cache`, so collection
        switches and ingest operations refresh the data automatically.

        Args:
            cursor (str | None): Opaque cursor token from a previous call.
            limit (int): Records per page (clamped to ``[1, 500]``).

        Returns:
            tuple[list[dict[str, Any]], str | None]: The page of records and
            the next cursor token, or ``None`` if there are no further pages.
        """
        if not self.qdrant_collection:
            return [], None

        decoded = decode_cursor(cursor)
        raw_offset = decoded.get("o")
        offset = int(raw_offset) if raw_offset is not None else 0
        page_size = max(1, min(int(limit), 500))

        cached = self._documents_cache.get(self.qdrant_collection)
        if cached is None:
            cached = self.list_documents()
            self._documents_cache[self.qdrant_collection] = cached

        end = offset + page_size
        page = cached[offset:end]
        next_cursor = encode_cursor(end) if end < len(cached) else None
        return page, next_cursor

    def get_document_count(self) -> int:
        """Return the number of unique documents in the active collection.

        Hits :attr:`_documents_cache` first; on a cache miss runs the full
        :meth:`list_documents` scan and stores the result so subsequent
        callers (e.g. the dashboard KPI and the paginated inspector) share
        the same materialized list. Returns ``0`` when no collection is
        selected.
        """
        if not self.qdrant_collection:
            return 0
        cached = self._documents_cache.get(self.qdrant_collection)
        if cached is None:
            cached = self.list_documents()
            self._documents_cache[self.qdrant_collection] = cached
        return len(cached)

    def get_document_summary(self) -> dict[str, Any]:
        """Return collection-wide document aggregates for the Inspector KPI strip.

        Aggregates the *whole* document list (document/node totals, file-type and
        entity-type breakdown) so the Inspector's summary cards reflect the
        entire collection rather than only the pages the user has scrolled in —
        the fix for the file-type counts undercounting a large, lazily-paginated
        collection. Shares :attr:`_documents_cache` with :meth:`get_document_count`
        and the paginated inspector, so it is O(1) after the first scroll and is
        refreshed by the same ingest/collection-switch cache invalidation.

        Returns:
            dict[str, Any]: ``document_count``, ``node_count``, ``file_types``
            (``[{label, count}]``) and ``entity_types`` (sorted). Zeroed/empty
            when no collection is selected.
        """
        if not self.qdrant_collection:
            return {"document_count": 0, "node_count": 0, "file_types": [], "entity_types": []}
        cached = self._documents_cache.get(self.qdrant_collection)
        if cached is None:
            cached = self.list_documents()
            self._documents_cache[self.qdrant_collection] = cached
        return summarize_document_types(cached)

    def measure_scope(self, chunk_ids: Sequence[str]) -> dict[str, Any]:
        """Measure a candidate scope against the chat context budget.

        Scoped answering splices the chosen chunks straight into the prompt, so
        the selection is bounded by the model's context window rather than by a
        top-k. Reuses the same ``usable_tokens`` figure the parent-context
        packer works from, so the two cannot drift apart.

        Args:
            chunk_ids (Sequence[str]): Candidate Qdrant point ids.

        Returns:
            dict[str, Any]: ``chunks``, ``est_tokens``, ``usable_tokens``,
                ``missing`` (scoped ids Qdrant no longer has) and ``fits``.
        """
        usable_tokens, _ = self._compute_parent_context_budget(
            social_table=bool(self._infer_collection_profile().get("is_social_table")),
        )
        retriever = _ScopedRetriever(rag=self, node_ids=chunk_ids)
        nodes = retriever.retrieve("")
        est = sum(estimate_tokens(node.node.get_content() or "", self.embed_char_token_ratio) for node in nodes)
        return {
            "chunks": len(list(chunk_ids)),
            "est_tokens": est,
            "usable_tokens": usable_tokens,
            "missing": retriever.missing,
            "fits": est <= usable_tokens,
        }

    def get_chunk_text(self, chunk_id: str) -> str | None:
        """Return one chunk's full text, for expanding a search hit.

        Search hits carry only a capped ``preview`` — returning full text for
        every hit would inflate each search by an order of magnitude for
        something most hits never need — so the whole chunk is fetched on
        demand.

        Args:
            chunk_id (str): Qdrant point id.

        Returns:
            str | None: The chunk text, or ``None`` when the point is gone
                (re-ingestion mints new ids) or carries no text.
        """
        node_id = str(chunk_id or "").strip()
        if not node_id:
            return None
        # Image hits live in the companion, so both lanes are tried before
        # calling a chunk gone.
        lanes = [self.qdrant_collection]
        companion = image_companion_name(self.qdrant_collection)
        if self._collection_exists(companion):
            lanes.append(companion)

        for lane in lanes:
            try:
                points = self.qdrant_client.retrieve(
                    collection_name=lane,
                    ids=[_as_qdrant_point_id(node_id)],
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as exc:
                logger.warning("Chunk fetch failed for {} in {}: {}", node_id, lane, exc)
                continue
            for point in points:
                text = RAG._extract_indexable_text(dict(getattr(point, "payload", {}) or {}))
                if text:
                    return text
        return None

    def search_fulltext(
        self,
        query: str,
        *,
        base_filter: qdrant_models.Filter | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """Return chunks containing every keyword in ``query``.

        Pure local lookup: one native Qdrant scroll, no embedding call and no
        inference. Keywords are ANDed and order-independent; matching is
        case-insensitive and prefix-based via the ``search_text`` index.

        Args:
            query (str): Raw query text; whitespace-separated keywords.
            base_filter (qdrant_models.Filter | None): Caller's metadata filter,
                ANDed with the keyword conditions.
            limit (int): Hits per page, clamped to ``[1, 500]``.
            cursor (str | None): Opaque page cursor from a previous call.

        Returns:
            dict[str, Any]: ``status`` — ``"ok"`` when every point in the
                collection is indexed, ``"partial"`` when some are not (a
                backfill is running or was interrupted, so the result set is
                incomplete), ``"not_indexed"`` when none are — plus ``hits``,
                ``total``, ``next_cursor`` and ``index_status`` counts.

        Raises:
            KeywordTooShortError: When a keyword cannot be indexed.
        """
        collection = self.qdrant_collection
        status = search_index_status(self.qdrant_client, collection)
        empty: dict[str, Any] = {
            "status": "ok" if status.get("complete") else "partial",
            "hits": [],
            "total": 0,
            "next_cursor": None,
            "index_status": status,
        }
        # Both conditions are required. The field alone is not a working
        # search: un-indexed MatchText case-folds ASCII only, so a lowercase
        # German query silently misses its title-case match. "not_indexed" is
        # both the honest label and the one that points at the remedy.
        if not status.get("with_search_text") or not status.get("indexed"):
            return {**empty, "status": "not_indexed"}

        keywords = parse_keywords(query)
        search_filter = build_search_filter(keywords, base_filter=base_filter)
        if search_filter is None:
            return empty

        page_size = max(1, min(int(limit), 500))
        companion = image_companion_name(collection)
        has_images = self._collection_exists(companion)

        # Lanes run in sequence — documents, then images — and a page fills
        # across the boundary. Hard match is a filter, not a ranker, so there
        # is no meaningful interleaving to preserve; what matters is that a
        # short final text page does not end the results and strand every
        # image hit behind a cursor nobody follows.
        cursor_state = decode_cursor(cursor)
        lane = str(cursor_state.get("lane") or "text")
        offset = cursor_state.get("o")

        hits: list[dict[str, Any]] = []
        next_cursor: str | None = None

        if lane == "text":
            points, next_offset = self._scroll_search_lane(collection, search_filter, page_size, offset)
            hits.extend(self._search_hits(collection, points, kind="text"))
            if next_offset is not None:
                next_cursor = encode_cursor(next_offset, extra={"lane": "text"})
            else:
                lane, offset = "image", None

        if lane == "image" and has_images and next_cursor is None and len(hits) < page_size:
            points, next_offset = self._scroll_search_lane(companion, search_filter, page_size - len(hits), offset)
            hits.extend(self._search_hits(companion, points, kind="image"))
            if next_offset is not None:
                next_cursor = encode_cursor(next_offset, extra={"lane": "image"})

        total = self._search_total(collection, search_filter)
        if has_images:
            total += self._search_total(companion, search_filter)

        # "partial" is a distinct status rather than a nested field so a caller
        # cannot miss incomplete coverage by ignoring ``index_status``. A search
        # run while the backfill is still walking the collection returns only
        # what has been written so far.
        return {
            "status": "ok" if status.get("complete") else "partial",
            "hits": hits,
            "total": total,
            "next_cursor": next_cursor,
            "index_status": status,
        }

    def _collection_exists(self, name: str) -> bool:
        """Return whether a collection exists, treating an outage as absent.

        Args:
            name (str): Physical collection name.

        Returns:
            bool: ``True`` when Qdrant reports the collection.
        """
        try:
            return bool(self.qdrant_client.collection_exists(collection_name=name))
        except Exception as exc:
            logger.debug("collection_exists failed for {}: {}", name, exc)
            return False

    def _scroll_search_lane(
        self,
        name: str,
        search_filter: qdrant_models.Filter,
        limit: int,
        offset: Any,
    ) -> tuple[list[Any], Any]:
        """Scroll one search lane.

        Args:
            name (str): Collection to scroll.
            search_filter (qdrant_models.Filter): Compiled keyword filter.
            limit (int): Maximum points to return.
            offset (Any): Scroll offset from the cursor, or ``None``.

        Returns:
            tuple[list[Any], Any]: The page and the next offset.
        """
        points, next_offset = self.qdrant_client.scroll(
            collection_name=name,
            scroll_filter=search_filter,
            limit=max(1, limit),
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        return list(points), next_offset

    def _search_total(self, name: str, search_filter: qdrant_models.Filter) -> int:
        """Return the exact number of matches in one lane.

        Args:
            name (str): Collection to count.
            search_filter (qdrant_models.Filter): Compiled keyword filter.

        Returns:
            int: Match count, or ``0`` when the count is unavailable.
        """
        try:
            return int(
                self.qdrant_client.count(
                    collection_name=name,
                    count_filter=search_filter,
                    exact=True,
                ).count
            )
        except Exception as exc:
            logger.debug("search count unavailable for {}: {}", name, exc)
            return 0

    def _search_hits(self, name: str, points: list[Any], *, kind: str) -> list[dict[str, Any]]:
        """Normalize a scrolled page into search hits.

        Runs through the same ``_source_from_payload`` the retrieval path uses,
        which already understands the ``_images`` payload shape, so an image
        hit carries the same citation identity as a document chunk.

        Args:
            name (str): Collection the points came from.
            points (list[Any]): Scrolled Qdrant points.
            kind (str): ``"text"`` or ``"image"``, so the panel can tell them
                apart — an image hit's body is a caption, not document prose.

        Returns:
            list[dict[str, Any]]: Normalized hits.
        """
        hits: list[dict[str, Any]] = []
        for point in points:
            payload = getattr(point, "payload", None)
            if not isinstance(payload, dict):
                continue
            node_id = str(getattr(point, "id", "") or "")
            source = self._source_from_payload(collection=name, payload=payload, node_id=node_id)
            text = str(source.get("text") or "")
            hits.append(
                {
                    "id": node_id,
                    "kind": kind,
                    "chunk_id": source.get("chunk_id"),
                    # Carried so a hit can deep-link into the Inspector's
                    # source preview, which keys on the document hash.
                    "file_hash": source.get("file_hash"),
                    "filename": source.get("filename"),
                    "page": source.get("page"),
                    "row": source.get("row"),
                    "preview": text[:_SEARCH_PREVIEW_CHARS].strip(),
                    # Lets the panel offer "expand" only where there is more to
                    # read; without it every hit invites a round-trip that
                    # returns the text already on screen.
                    "truncated": len(text) > _SEARCH_PREVIEW_CHARS,
                    "entity_types": sorted(
                        {
                            str(entity.get("type") or "Unlabeled")
                            for entity in normalize_entities(payload.get("entities"))
                        }
                    ),
                    "est_tokens": estimate_tokens(text, self.embed_char_token_ratio),
                }
            )
        return hits

    def get_collection_ner(self, refresh: bool = False) -> list[dict[str, Any]]:
        """Fetch all nodes from the current collection and return their NER metadata.

        Args:
            refresh (bool): If ``True``, bypass in-memory NER cache and re-fetch from Qdrant.

        Returns:
            list[dict[str, Any]]: A list of source metadata dictionaries containing NER data.
        """
        collection = self.qdrant_collection
        if not collection:
            return []

        if refresh:
            self._invalidate_ner_cache(collection)

        cached = self._ner_sources_cache.get(collection)
        if cached is not None:
            return cached

        sources = self._load_collection_ner_sources()
        self._ner_sources_cache[collection] = sources
        return sources

    def iter_collection_ner_sources(
        self,
        *,
        cursor: str | None = None,
        limit: int = 50,
        entity_text: str | None = None,
        entity_type: str | None = None,
        entity_key: str | None = None,
        entity_merge_mode: EntityMergeMode = "orthographic",
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Return one paginated slice of NER-bearing sources.

        Sources come from :meth:`get_collection_ner` (cached on
        ``self._ner_sources_cache`` per collection), so the first call after a cache miss pays the
        Qdrant scroll cost and subsequent calls only filter and slice.

        Entity filtering mirrors the SPA's ``sourceContainsEntity``: the same
        match-by-text-or-compact-form rules apply, scoped to the same entity
        type when both sides specify one. In ``"resolved"`` mode the filter
        expands the requested surface to every sibling alias of its canonical
        entity, so a chunk mentioning only an alias (e.g. "United States" for
        the canonical "US") is still returned — no mention rows are lost.

        Args:
            cursor (str | None): Opaque cursor token from a previous call.
            limit (int): Records per page (clamped to ``[1, 500]``).
            entity_text (str | None): Entity surface form to filter by. When
                set without ``entity_type``, type is not constrained.
            entity_type (str | None): Entity type/label (e.g. ``"PERSON"``).
            entity_key (str | None): Raw ``"<text>::<type>"`` key, accepted as
                a shorthand for ``entity_text``/``entity_type``. The SPA's
                ``Analysis.tsx`` ``keyOf`` produces this format.
            entity_merge_mode (EntityMergeMode): Clustering mode; ``"resolved"``
                expands to the canonical entity's sibling aliases.

        Returns:
            tuple[list[dict[str, Any]], str | None]: The page of source rows
            and the next cursor token, or ``None`` when exhausted.
        """
        if not self.qdrant_collection:
            return [], None

        decoded = decode_cursor(cursor)
        raw_offset = decoded.get("o")
        offset = int(raw_offset) if raw_offset is not None else 0
        page_size = max(1, min(int(limit), 500))

        sources = self.get_collection_ner()

        if entity_key and not (entity_text or entity_type):
            if "::" in entity_key:
                entity_text, entity_type = entity_key.split("::", 1)
            else:
                entity_text = entity_key

        if entity_text:
            alias_surfaces = (
                self._resolved_alias_surfaces(entity_text, entity_type or "")
                if normalize_entity_merge_mode(entity_merge_mode) == "resolved"
                else None
            )
            if alias_surfaces is not None:
                filtered = _filter_sources_by_surfaces(
                    sources,
                    surfaces=alias_surfaces,
                    target_type=entity_type or "",
                )
            else:
                filtered = _filter_sources_by_entity(
                    sources,
                    target_text=entity_text,
                    target_type=entity_type or "",
                )
        else:
            filtered = list(sources)

        end = offset + page_size
        page = filtered[offset:end]
        next_cursor = encode_cursor(end) if end < len(filtered) else None
        return page, next_cursor

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
        for page in iter_scroll(
            self.qdrant_client,
            collection_name=self.qdrant_collection,
            page_size=100,
            error_context="hate-speech rows",
        ):
            for point in page:
                payload = getattr(point, "payload", None)
                if not isinstance(payload, dict):
                    continue

                detection = payload.get("hate_speech")
                if not isinstance(detection, dict) or not bool(detection.get("hate_speech")):
                    continue

                source = self._source_from_payload(
                    collection=self.qdrant_collection,
                    payload=payload,
                    text_value=str(detection.get("chunk_text") or self._extract_payload_text(payload) or ""),
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
                    detection.get("source_ref") or source.get("filename") or payload.get("file_path") or ""
                )
                findings.append(source)

        findings.sort(key=operator.itemgetter("source_ref", "chunk_id"))
        return findings

    def iter_hate_speech(
        self,
        *,
        cursor: str | None = None,
        limit: int = 50,
        category: str | None = None,
        min_confidence: str | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Return one paginated slice of hate-speech findings.

        Filtering accepts a ``category`` (case-insensitive equality) and a
        ``min_confidence`` ordering — rows below the requested confidence
        level (``low`` < ``medium`` < ``high``) are dropped. The cache is
        invalidated alongside other per-collection caches via
        :meth:`_invalidate_ner_cache`.

        Args:
            cursor (str | None): Opaque cursor token from a previous call.
            limit (int): Records per page (clamped to ``[1, 500]``).
            category (str | None): If set, only return findings with this
                category.
            min_confidence (str | None): Confidence threshold; matches are
                returned in the original sort order from
                :meth:`get_collection_hate_speech`.

        Returns:
            tuple[list[dict[str, Any]], str | None]: The page of findings
            and the next cursor token, or ``None`` when exhausted.
        """
        if not self.qdrant_collection:
            return [], None

        decoded = decode_cursor(cursor)
        raw_offset = decoded.get("o")
        offset = int(raw_offset) if raw_offset is not None else 0
        page_size = max(1, min(int(limit), 500))

        cached = self._hate_speech_cache.get(self.qdrant_collection)
        if cached is None:
            cached = self.get_collection_hate_speech()
            self._hate_speech_cache[self.qdrant_collection] = cached

        filtered = _filter_hate_speech(cached, category=category, min_confidence=min_confidence)
        end = offset + page_size
        page = filtered[offset:end]
        next_cursor = encode_cursor(end) if end < len(filtered) else None
        return page, next_cursor

    def _entities_collection(self) -> str:
        """Return the hidden companion collection name for resolved entities.

        Returns:
            str: ``{active_collection}_entities`` (empty when no collection
            is selected).
        """
        return f"{self.qdrant_collection}_entities" if self.qdrant_collection else ""

    def _embed_surfaces(self, texts: list[str]) -> list[list[float]]:
        """Embed surface forms in batches that respect the embed budget.

        Reuses the embedding model and per-request batch size already
        configured on the RAG instance (see
        :meth:`_prepare_vector_nodes_for_insert` for the same slicing).

        Args:
            texts (list[str]): Surface forms to embed.

        Returns:
            list[list[float]]: Embedding vectors aligned to ``texts``.
        """
        embed_model = self.embed_model
        get_embeddings = getattr(embed_model, "get_text_embeddings_strict", None)
        if not callable(get_embeddings):
            return [embed_model.get_text_embedding(text) for text in texts]
        vectors: list[list[float]] = []
        batch_size = max(1, self.embed_batch_size)
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            vectors.extend(cast(list[list[float]], get_embeddings(chunk)))
        return vectors

    def _entity_vector_dim(self) -> int:
        """Resolve the embedding dimension for the entity vector store.

        Returns:
            int: Configured ``openai_dimensions`` when set, otherwise a single
            embed probe of the active model.
        """
        if self.openai_dimensions is not None:
            return int(self.openai_dimensions)
        return len(self.embed_model.get_text_embedding("ping"))

    def _load_resolved_index(self) -> dict[str, Any] | None:
        """Load the durable entity index for ``"resolved"`` aggregation.

        Returns:
            dict[str, Any] | None: ``{"alias_to_id", "canonical",
            "case_normalize"}`` for :func:`aggregate_ner_sources`, or ``None``
            when no entities have been resolved for the active collection.
        """
        base = self.qdrant_collection
        if base and base in self._resolved_index_cache:
            return self._resolved_index_cache[base]

        collection = self._entities_collection()
        if not collection or not qdrant_collection_exists(self.qdrant_client, collection):
            if base:
                self._resolved_index_cache[base] = None
            return None
        cfg = load_resolution_env()
        store = EntityStore(
            self.qdrant_client,
            collection=collection,
            dim=int(self.openai_dimensions or 1),
            embed_model=self.embed_model_id,
        )
        alias_to_id, canonical = store.load_alias_index(case_normalize=cfg.case_normalize)
        index: dict[str, Any] = {
            "alias_to_id": alias_to_id,
            "canonical": canonical,
            "case_normalize": cfg.case_normalize,
        }
        if base:
            self._resolved_index_cache[base] = index
        return index

    def _resolved_alias_surfaces(self, entity_text: str, entity_type: str) -> set[str] | None:
        """Return all normalized surface forms of the resolved entity for a surface.

        Looks up the canonical entity that ``(entity_text, entity_type)`` resolves
        to and returns every surface (aliases + canonical) attached to it, so the
        drill-down can match sibling-alias chunks (e.g. "United States" when the
        canonical is "US"). Normalized (casefolded) forms, matched case-insensitively
        against source mentions.

        Args:
            entity_text (str): A surface (canonical or alias) of the target entity.
            entity_type (str): The entity type/label.

        Returns:
            set[str] | None: Normalized sibling surfaces, or ``None`` when the
            collection is unresolved or the surface maps to no canonical entity.
        """
        resolved = self._load_resolved_index()
        if not resolved:
            return None
        alias_to_id: dict[tuple[str, str], str] = resolved.get("alias_to_id") or {}
        case_normalize = bool(resolved.get("case_normalize", True))
        norm = normalize_surface(entity_text, case_normalize=case_normalize)
        entity_id = alias_to_id.get((norm, str(entity_type or "").lower()))
        if entity_id is None:
            return None
        return {surface for (surface, _type), eid in alias_to_id.items() if eid == entity_id}

    def resolve_entities(
        self,
        *,
        progress_callback: Callable[[str], None] | None = None,
    ) -> ResolutionSummary:
        """Resolve the active collection's entities into durable canonicals.

        Mirrors chorus's batch ``resolve`` stage: collapse extracted entities to
        unique ``(surface, type)`` pairs, embed them once, and resolve each
        most-mentioned-first into the hidden ``{collection}_entities`` store
        (exact alias -> type-blocked vector match -> conservative LLM tie-break
        -> mint). Idempotent: surfaces already resolved on a prior run are
        skipped. Invalidates the NER cache so the ``"resolved"`` aggregate
        recomputes against the updated store.

        Args:
            progress_callback (Callable[[str], None] | None): Optional sink for
                human-readable progress messages.

        Returns:
            ResolutionSummary: Counts of minted/attached/skipped surfaces.

        Raises:
            ValueError: If no collection is selected.
        """
        if not self.qdrant_collection:
            raise ValueError("qdrant_collection must be set to resolve entities")

        def _emit(message: str) -> None:
            """Forward a progress message to the callback when present."""
            if progress_callback is not None:
                progress_callback(message)
            logger.info(message)

        sources = self._load_collection_ner_sources()
        mention_counts: dict[tuple[str, str], int] = defaultdict(int)
        for src in sources:
            for ent in normalize_entities(src.get("entities")):
                mention_counts[(str(ent["text"]), str(ent["type"]))] += 1

        surfaces = [
            SurfaceMention(surface=surface, entity_type=entity_type, mentions=mentions)
            for (surface, entity_type), mentions in mention_counts.items()
        ]
        if not surfaces:
            _emit(f"No entities to resolve in collection '{self.qdrant_collection}'.")
            return ResolutionSummary(processed=0, minted=0, attached=0, skipped=0, entities_touched=0)

        _emit(f"Resolving {len(surfaces)} distinct entity surfaces in '{self.qdrant_collection}'.")
        cfg = load_resolution_env()
        store = EntityStore(
            self.qdrant_client,
            collection=self._entities_collection(),
            dim=self._entity_vector_dim(),
            embed_model=self.embed_model_id,
        )
        store.ensure_collection()

        prompt_header = DEFAULT_ENTITY_TIEBREAK_PROMPT
        if self.prompt_dir:
            prompt_header = self._load_prompt_text(
                self.prompt_dir / "entity_tiebreak.txt",
                default=DEFAULT_ENTITY_TIEBREAK_PROMPT,
            )

        def _chat(prompt: str) -> str:
            """Run the configured chat model for one tie-break prompt."""
            completion = self.text_model.complete(prompt)
            return str(getattr(completion, "text", "") or "").strip()

        summary = resolve_collection(
            store,
            surfaces,
            embed_fn=self._embed_surfaces,
            chat_fn=_chat,
            prompt_header=prompt_header,
            cfg=cfg,
        )

        self._bump_summary_revision(self.qdrant_collection)
        self._invalidate_ner_cache(self.qdrant_collection)
        _emit(
            f"Entity resolution complete for '{self.qdrant_collection}': "
            f"{summary.minted} minted, {summary.attached} attached, {summary.skipped} skipped."
        )
        return summary

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

        resolved_index = self._load_resolved_index() if merge_mode == "resolved" else None
        sources = self.get_collection_ner(refresh=refresh)
        aggregate = aggregate_ner_sources(sources, entity_merge_mode=merge_mode, resolved_index=resolved_index)
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
            entity_merge_mode (EntityMergeMode): Entity clustering mode used for derived views.

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
            entity_merge_mode (EntityMergeMode): Entity clustering mode used for derived views.

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
            entity_merge_mode (EntityMergeMode): Entity clustering mode used for derived views.

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
            entity_merge_mode (EntityMergeMode): Entity clustering mode used for derived views.

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
        logger.info("Models unloaded and memory cleared.")


# --- ``RAG.qdrant_collection`` property (attached post-class) ---
# Defined here, not in the class body, so the dataclass-generated ``__init__``
# (and the type checker) keep seeing ``qdrant_collection`` as the ``InitVar``
# constructor parameter. A property of the same name inside the body would
# shadow that field and break ``RAG(qdrant_collection=...)`` everywhere.


def _rag_qdrant_collection_get(self: RAG) -> str:
    """Return the active physical collection for the current request.

    Resolves to the per-request override bound by :meth:`RAG.collection_scope`
    when one is active, otherwise the process default seeded at construction.
    Reads are therefore isolated across concurrent requests/threads.

    Args:
        self (RAG): The RAG instance (bound by the property).

    Returns:
        str: The active physical Qdrant collection name.
    """
    override = _active_collection.get()
    return override if override is not None else self._collection_default


def _rag_qdrant_collection_set(self: RAG, value: str) -> None:
    """Set the active collection for the current scope.

    Within a :meth:`RAG.collection_scope` the per-request override is updated
    (keeping the change request-local); outside any scope the process default is
    updated -- the CLI / single-collection path, and pre-existing tests that
    assign ``rag.qdrant_collection`` directly.

    Args:
        self (RAG): The RAG instance (bound by the property).
        value (str): The collection name to make active.
    """
    if _active_collection.get() is not None:
        _active_collection.set(value)
    else:
        self._collection_default = value


RAG.qdrant_collection = property(_rag_qdrant_collection_get, _rag_qdrant_collection_set)  # type: ignore[assignment]
