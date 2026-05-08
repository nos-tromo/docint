# Retrieval and agents

This document describes the read path of Docint: from a user question to
a grounded answer, including the agent orchestration layer, the RAG
engine, and the postprocessing stages that shape the final result.

## Agent orchestration

All agent code lives under `docint/agents/`. The entry point is
`AgentOrchestrator` (`docint/agents/orchestrator.py:18`), which wires up
four optional agents behind a single `handle_turn()` method.

### Stages

1. **Understanding** — `docint/agents/understanding.py`
   - `SimpleUnderstandingAgent` — heuristic keyword-based intent
     detection with confidence between `0.6` and `0.8`. Covers intents
     `qa`, `ner`, `table`, and `summary`.
   - `ContextualUnderstandingAgent` — LLM-backed; prompts the configured
     text model to produce an `IntentAnalysis` with a rewritten query
     and extracted entities. Falls back to `qa` if the LLM errors.
   - The FastAPI app automatically upgrades to the contextual agent
     when a text model is available (`docint/core/api.py:74`).

2. **Clarification policy** — `docint/agents/policies.py`
   - `ClarificationPolicy.evaluate()` reads the intent analysis and
     decides whether to ask the user for clarification. Defaults:
     clarify if confidence is below the policy threshold or required
     entities are missing, and cap the number of clarifications per
     session to `2`.

3. **Clarification agent** — `docint/agents/clarify.py`
   - Builds the user-facing clarification message from the analysis and
     any missing required fields.

4. **Retrieval agent** — `docint/agents/retrieval.py`
   - `RAGRetrievalAgent.retrieve()` dispatches by intent:
     - `ner` / `extract` → `RAG.get_collection_ner()` with entity and
       page filters.
     - `table` → placeholder (defaults to RAG pass-through).
     - Default → `RAG.chat()` (session mode) or `RAG.run_query()`
       (stateless). If an LLM-rewritten query is available, it is used
       in place of the raw user message.

5. **Response agent (optional)** — `docint/agents/generation.py`
   - `PassthroughResponseAgent` is the default no-op.
   - `ResultValidationResponseAgent`
     (`docint/agents/generation.py:50`) re-checks the generated answer
     against the returned sources using the configured LLM. When the
     LLM disagrees, it sets `validation_mismatch=true` and attaches a
     `validation_reason`. Gated by
     [`RESPONSE_VALIDATION_ENABLED`](configuration.md#response-validation--responsevalidationconfig).

### Data shapes

`docint/agents/types.py` defines the dataclasses shared across stages:

- `Turn` — the user's message, session id, and any attached metadata.
- `IntentAnalysis` — `intent`, `confidence`, `entities`,
  `rewritten_query`, `needs_clarification`.
- `ClarificationRequest` — `needed`, `message`, `reason`.
- `RetrievalRequest` / `RetrievalResult` — the retrieval payload plus
  its response, sources, and diagnostics.
- `OrchestratorResult` — the final envelope returned by
  `handle_turn()`, containing either a clarification or a retrieval.
- `TurnContext` (`docint/agents/context.py`) — per-turn context with
  session id, metadata, and a clarification counter.

### Tools

`docint/agents/tools.py` wraps the tool surface the orchestrator can
expose to LLM-backed understanding agents (e.g. for structured
intent analysis). New tools should be added here and registered with
the contextual understanding agent.

## RAG engine

`docint/core/rag.py` is the workhorse. The `RAG` class encapsulates:

- **Qdrant client management** — `list_collections()`,
  `select_collection()`, `delete_collection()`.
- **Index construction** — `create_index()` builds a
  `VectorStoreIndex` backed by the `QdrantKVStore` docstore
  (`docint/core/storage/docstore.py`).
- **Query engine construction** — `create_query_engine()` attaches the
  reranker, postprocessors, and response synthesiser.
- **Stateless query** — `run_query(prompt, metadata_filters=...,
  vector_store_kwargs=...)` returns a `dict` with `response`,
  `sources`, `retrieval_query`, `coverage_unit`, and `retrieval_mode`.
- **Session chat** — `chat(...)` proxies to
  `SessionManager.chat()` (`docint/core/state/session_manager.py`).
- **Streaming chat** — `stream_chat(...)` yields token-level events
  for SSE streaming.
- **NER queries** — `get_collection_ner()`,
  `run_entity_occurrence_query()`,
  `run_multi_entity_occurrence_query()`.
- **Summarisation** — `summarize_collection()` drives
  `/summarize` with the knobs from
  [`SummaryConfig`](configuration.md#summarisation--summaryconfig).

## Retrieval modes

`RetrievalConfig` (see [configuration.md](configuration.md#retrieval--retrievalconfig))
controls the vector store query mode used by the query engine:

| Mode | Description |
|---|---|
| `auto`    | Default — the engine picks dense or hybrid based on collection capability. |
| `default` | Pure dense retrieval. |
| `sparse`  | Sparse-only retrieval using the BM25-style sparse model. |
| `hybrid`  | Dense + sparse fusion with `RETRIEVAL_HYBRID_ALPHA` as the weight. |
| `mmr`     | Maximal marginal relevance — dense with redundancy penalty. |

Top-K values are split across retrieval types:

- `RETRIEVE_TOP_K` — dense top-K.
- `RETRIEVAL_SPARSE_TOP_K` — sparse top-K.
- `RETRIEVAL_HYBRID_TOP_K` — final depth after fusion.

## Metadata filters

Metadata filters come in on the `/query` payload as
`MetadataFilterIn` objects and are translated by
`docint/core/retrieval_filters.py`:

- `build_metadata_filters()` produces LlamaIndex-native filter objects
  applied at the query engine level.
- `build_qdrant_filter()` produces a Qdrant-native filter that is
  passed through `vector_store_kwargs` so Qdrant short-circuits the
  candidate list server-side.

Supported operators (`MetadataFilterIn.operator`): `eq`, `neq`, `gt`,
`gte`, `lt`, `lte`, `in`, `contains`, `mime_match`, `date_after`,
`date_on_or_after`, `date_before`, `date_on_or_before`.

## Reranking

Candidates retrieved from Qdrant are reranked with
`FlagEmbeddingReranker` (BGE cross-encoder). Gates:

- `RERANK_MODEL` — model identifier (default
  `BAAI/bge-reranker-v2-m3`).
- `RERANK_USE_FP16` — flip to `true` to use FP16 weights.

For LLM-backed rerankers, an alternative `LLMRerank` variant can be
swapped in via the query engine construction path.

## Parent-context expansion

`ParentContextExpansionPostprocessor` (implemented in `rag.py`) takes a
fine-grained chunk hit and pulls in its coarse parent node for more
context in the final prompt. It is enabled by
`PARENT_CONTEXT_RETRIEVAL_ENABLED=true` and requires
`HIERARCHICAL_CHUNKING_ENABLED=true` at ingest time.

## Source diversity

`SocialSourceDiversityPostprocessor` caps the number of consecutive
chunks returned from the same social/table row source, defaulting to
`2` per bucket via `SUMMARY_SOCIAL_DIVERSITY_LIMIT`. It keeps
summaries and row-heavy collections balanced across documents.

`CustomImageSourcePostprocessor` filters and reranks image-vector
matches when the active collection carries an image sibling
(`{collection}_images`).

## Graph-assisted retrieval

When `GRAPHRAG_ENABLED=true`, `RAG` builds an entity graph from the
ingested NER metadata (`docint/core/ner.py`) and exposes
`expand_query_with_graph_with_debug()`. Before dispatching a query to
the retriever, the engine:

1. Extracts entity mentions from the user question.
2. Walks `GRAPHRAG_NEIGHBOR_HOPS` hops through the graph, keeping edges
   whose weight is at least `GRAPHRAG_MIN_EDGE_WEIGHT`.
3. Picks up to `GRAPHRAG_MAX_NEIGHBORS` neighbour entities, capped by
   `GRAPHRAG_TOP_K_NODES`.
4. Appends the neighbour names to the retrieval query so that Qdrant
   sees a richer embedding.

A `graph_debug` payload is returned in the `QueryOut` for the
stateless path so the UI can visualise what was added.

## Response validation

When `RESPONSE_VALIDATION_ENABLED=true`, the orchestrator attaches a
`ResultValidationResponseAgent`. If no sources were retrieved, validation
short-circuits with `validation_mismatch=true` and a reason of "Answer
produced without retrieved sources." Otherwise, it asks the text LLM two
questions:

1. Does the answer match the retrieved sources?
2. Do the sources actually contain the answer?

When building the validation prompt, the agent includes retrieval context
such as the retrieval query (if rewritten), the detected intent, and the
tool used. If the LLM answers "no" to either question, `validation_mismatch=true`
is set on the `QueryOut` / `AgentChatOut` payload and `validation_reason`
carries the LLM's explanation. The frontend surfaces this as a warning banner.

## Sessions and citations

For session-aware retrieval, `SessionManager.chat()`
(`docint/core/state/session_manager.py`) handles:

- Loading or creating a `Conversation` row from SQLite
  (`docint/core/state/conversation.py`).
- Condensing the user message using a rolling summary plus the last
  few turns.
- Calling `RAG.run_query()` with the condensed question.
- Persisting a `Turn` row (`docint/core/state/turn.py`) and any
  `Citation` rows (`docint/core/state/citation.py`).

The session store URL is resolved by
[`SessionConfig`](configuration.md#sessions--sessionconfig).

## End-to-end trace

A typical `/agent/chat` request touches, in order:

1. `docint/core/api.py:1070` — validates the `AgentChatIn` payload.
2. `AgentOrchestrator.handle_turn()`
   (`docint/agents/orchestrator.py:47`).
3. `ContextualUnderstandingAgent.analyze()`
   (`docint/agents/understanding.py`) — produces an
   `IntentAnalysis`.
4. `ClarificationPolicy.evaluate()` (`docint/agents/policies.py`) —
   decides on clarification.
5. `RAGRetrievalAgent.retrieve()` → `RAG.chat()`
   (`docint/core/rag.py`).
6. `SessionManager.chat()` → `RAG.run_query()` → Qdrant + reranker +
   postprocessors.
7. `ResultValidationResponseAgent.finalize()` —
   groundedness check.
8. `AgentChatOut` marshalled and returned to the caller.

For the ingestion side of the story, see
[ingestion.md](ingestion.md).
