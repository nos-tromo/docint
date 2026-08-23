# Retrieval and agents

This document describes the read path of Docint: from a user question to
a grounded answer, including the agent orchestration layer, the RAG
engine, and the postprocessing stages that shape the final result.

## Agent orchestration

All agent code lives under `docint/agents/`. The entry point is
`AgentOrchestrator` (`docint/agents/orchestrator.py:70`), which wires up
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
     when a text model is available (`docint/core/api.py:181`).

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
     (`docint/agents/generation.py:75`) re-checks the generated answer
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
  `VectorStoreIndex` backed by the `SQLiteKVStore` docstore
  (`docint/core/storage/sqlite_kvstore.py`).
- **Query engine construction** — `create_query_engine()` attaches the
  reranker, postprocessors, and response synthesiser.
- **Stateless query** — `run_query(prompt, metadata_filters=...,
  vector_store_kwargs=...)` returns a `dict` with `response`,
  `sources`, `retrieval_query`, `coverage_unit`, and `retrieval_mode`.
- **Session chat** — `chat(...)` proxies to
  `SessionManager.chat()` (`docint/core/state/session_manager.py`).
- **Streaming chat** — `stream_chat(...)` yields token-level events
  for SSE streaming.
- **NER queries** — `get_collection_ner()`.
- **Summarisation** — `build_tree_summary()` drives
  `/summarize` with the knobs from
  [`SummaryConfig`](configuration.md#summarisation--summaryconfig).

## Retrieval modes

`RetrievalConfig` (see [configuration.md](configuration.md#retrieval--retrievalconfig))
controls the vector store query mode used by the query engine:

| Mode | Description |
|---|---|
| `auto`    | Default — the engine picks dense or hybrid based on collection capability. |
| `default` | Pure dense retrieval. |
| `sparse`  | Sparse-only retrieval using the learned sparse model. |
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

A filter targets either a single `field` or several `fields`; when several are
given the rule matches if any of them matches. The SPA uses this to apply one
date bound to both `reference_metadata.timestamp` and
`reference_metadata.posting_timestamp`.

Date, `contains`, and non-numeric range bounds are carried **only** by
`build_qdrant_filter()`. `build_metadata_filters()` deliberately compiles them
to nothing: `QdrantVectorStore` turns a date bound into `Range(gte=<ISO
string>)` whose bounds are floats, and raises `NotImplementedError` for
`FilterOperator.CONTAINS`. Since `vector_store_kwargs["qdrant_filters"]`
overrides the LlamaIndex filters inside `QdrantVectorStore.query`, the native
filter is the one that executes.

## Scoped answering

When a session carries a search scope (`PUT /sessions/{id}/scope`), the read
path changes shape: `build_query_engine(scoped_node_ids=...)` swaps in
`_ScopedRetriever`, which fetches exactly those points by id and returns them in
the scope's own order. There is no vector query, no rerank and no inference
beyond generation.

Swapping the *retriever* rather than hand-building a prompt is deliberate:
citation numbering, source normalization, the report-builder controls and the
Inspector links are all driven by the node set, so they keep working unchanged.

The scoped engine **drops every ranking postprocessor**. Parent-context
expansion and link-following would silently widen a hand-picked set; the social
diversity cap and the image relevance floor would silently narrow it; reranking
would spend an inference call reordering a set the user already chose. Only
`CitationNumberingPostprocessor`, which merely numbers, survives.

`RAG.measure_scope()` sizes a candidate selection against the same
`usable_tokens` figure the parent-context packer works from, so the budget the
API enforces and the budget retrieval assumes cannot drift apart.

## Reranking

Candidates retrieved from Qdrant are reranked by
`VLLMRerankPostprocessor` (`rag.py`). Reranking is **always a remote
call** on every provider — there is no local fallback model. The
postprocessor POSTs to `{RERANK_API_BASE}/rerank` in the Jina shape
(`{model, query, documents, top_n}`) and maps the returned order back
onto the nodes. Gates:

- `RERANK_MODEL` — model identifier (default
  `BAAI/bge-reranker-v2-m3`, `env_cfg.py:1108`).
- `RERANK_API_BASE` / `RERANK_API_KEY` / `RERANK_TIMEOUT` — endpoint,
  Bearer token and per-request timeout; each inherits the matching
  `OPENAI_*` setting when unset. See
  [configuration.md](configuration.md) for the full table.

Reranking is **fail-soft**: a transport failure degrades to the
original retrieval order rather than failing the query, and the outcome
is stamped on each returned node (`docint_rerank_applied`,
`docint_rerank_error`) so `/query` responses report whether the rerank
actually ran. The reranker is wrapped in `LazyRerankerPostprocessor`,
which defers client construction and its healthcheck to the first
query — building a query engine for warmup or introspection does not
pay that cost.

## Parent-context expansion

`ParentContextExpansionPostprocessor` (implemented in `rag.py`) takes a
fine-grained chunk hit and pulls in its coarse parent node for more
context in the final prompt. It is enabled by
`PARENT_CONTEXT_RETRIEVAL_ENABLED=true` and requires
`HIERARCHICAL_CHUNKING_ENABLED=true` at ingest time.

## Source diversity

`SocialSourceDiversityPostprocessor` caps the number of consecutive
chunks returned from the same social/table row source, defaulting to
`2` per bucket (`SOCIAL_SOURCE_DIVERSITY_LIMIT`, clamped to a minimum
of `1`). It keeps row-heavy collections balanced across documents in
the chat/retrieval path.

`CustomImageSourcePostprocessor` filters and reranks image-vector
matches when the active collection carries an image sibling
(`{collection}_images`).

## Image retrieval lane

Images are ordinary sources. A stored image — a standalone file, a figure
embedded in a PDF, a video keyframe — is retrieved by CLIP, ranked against the
text chunks by the same reranker on the same scale, shown to the model as part
of the evidence, numbered like any other citation, and quotable in the
collection summary alongside its document's text.

What the model sees of an image is what was stored for it at ingest time: its
caption and tags, and — where a document OCR model is configured — the text
printed *inside* it (see
[ingestion.md](ingestion.md#images--imagespy)). No pixels are sent at query
time, and no vision call happens on the chat path.

Settings that shape the lane:

- `IMAGE_RETRIEVE_TOP_K` (default `5`) — how many CLIP candidates enter the
  ranking. They then compete with text chunks for the answer's source slots;
  a query with no relevant imagery spends none of them.
- `IMAGE_OCR_ENABLED` (on when `OCR_MODEL` is set) — read the text inside
  images. `KEYFRAME_OCR_ENABLED` (default off) extends that to video
  keyframes, where usually only slides carry text.
- `IMAGE_RERANK_MIN_SCORE` (default `0.05`) — the reranker score an image
  caption must reach. The floor sits on the reranker, never on raw CLIP
  similarity, which is not comparable across queries: an unrelated query and a
  matching one both land in the same narrow CLIP band. Raise it if unrelated
  images still appear; lower it if relevant ones are missing.

If the rerank endpoint is down, images surface ungated rather than vanishing —
a degraded ranking is more useful than a silently emptied lane. Full defaults
and rationale:
[configuration.md](configuration.md#image-ingestion--imageingestionconfig).

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
tool used.

The source bodies in that prompt share one character budget
(`RESPONSE_VALIDATION_SOURCE_BUDGET_CHARS`, default 48000) allocated
shortest-first, with each source's unused share passed on to the longer
ones. This matters for correctness, not just cost: the generator answers
from whole chunks, so a validator shown only the head of each chunk reports
everything drawn from further in as a hallucination. Sources that still do
not fit are cut with an inline `[... N of M characters not shown ...]`
marker, and a localized note (`prompts/{en,de}/response_validator_truncation.txt`)
tells the validator to judge only visible contradictions.

If the LLM answers "no" to either question, `validation_mismatch=true`
is set on the `QueryOut` / `AgentChatOut` payload and `validation_reason`
carries the LLM's explanation. The frontend surfaces this as a warning banner.

## Corrective retry

A rejected answer that is *also* weak — empty, under 40 characters, or
carrying a known refusal phrase (`is_weak_answer` in
`agents/orchestrator.py`) — gets one automatic second attempt when
`CORRECTIVE_RETRY_ENABLED=true`. `QueryReformulationAgent`
(`agents/reformulation.py`, prompt `prompts/{en,de}/reformulate_retrieval.txt`)
turns the original question, the failed retrieval query, and the validator's
reason into a fresh retrieval query; the turn is re-retrieved, re-generated,
and re-validated against the **original** question. The cap is one attempt and
it is structural — there is no loop.

Both chat paths run it:

- **`/agent/chat`** — inside `AgentOrchestrator.handle_turn`, between
  validation and the existing weak-answer clarification fallback. A retry that
  is still weak and mismatched falls through to that same fallback. The
  response carries `retried` / `retry_query`.
- **`/stream_query`** — after the first answer has streamed and been
  validated. The retry is announced with its own SSE frame
  (`{"retry": {"query": ...}}`) *before* the replacement streams, so the SPA
  discards the rejected answer on arrival rather than swapping it silently at
  the end; the final envelope carries `retried` / `retry_query`, and both are
  persisted on the turn so a reloaded session still names the retry.

The gate is deliberately narrow. A mismatched but substantive answer is kept —
discarding it to chase a better one trades a real answer for a coin flip. A
scoped turn is skipped entirely, since it runs no retrieval for a new query to
change. And the second attempt overwrites the first attempt's turn
(`replace_turn_idx`) instead of appending, so one user message stays one turn.

Failures degrade rather than escalate: no reformulation available, or a second
pass that dies mid-stream, leaves the first answer as the delivered result —
a retry never turns a delivered answer into an error.

The cost is up to three extra LLM round-trips (reformulate, regenerate,
re-validate) inside the one request, so a turn that triggers a retry is
noticeably slower than one that does not.

## Citation numbering

Answers refer to their evidence by number ("source 3"), and the chat window's
source cards carry the matching number. The number is assigned server-side
before generation: the last node postprocessor stamps `citation_index` onto
the snippet set the synthesizer renders, so the model reads its number instead
of counting, and the same value rides the source payload out to the SPA.

The list can have gaps: the SPA drops broken-preview duplicates from the card
list, and the surviving cards keep their original numbers rather than closing
the gap, because renumbering would break the link to the answer.

Conversations replayed from the session DB take their numbers from citation
row order, which is the order the generator saw. Answers written before this
feature still contain hand-counted ordinals that may not line up.

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

1. `docint/core/api.py:3734` — validates the `AgentChatIn` payload.
2. `AgentOrchestrator.handle_turn()`
   (`docint/agents/orchestrator.py:106`).
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
