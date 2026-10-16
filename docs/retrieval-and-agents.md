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
[ingestion.md](ingestion.md#images--imagespy)). Under the `all` and
`documents` targets no pixels are sent at query time and no vision call
happens on the chat path; the `visual` target is the exception and attaches
the stored thumbnails itself (see [Retrieval targets](#retrieval-targets)).

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
  images still appear; lower it if relevant ones are missing. It applies to the
  `all` target only — see the visual target below.

If the rerank endpoint is down, images surface ungated rather than vanishing —
a degraded ranking is more useful than a silently emptied lane. Full defaults
and rationale:
[configuration.md](configuration.md#image-ingestion--imageingestionconfig).

## Retrieval targets

A turn carries a **retrieval target** saying which evidence may answer it,
sent as `retrieval_target` on `/query` and `/stream_query` and echoed back on
the response and the final SSE frame. It is a separate field from
`retrieval_mode`, which is session routing on the request (`session` /
`stateless`) and a different vocabulary on the response (`scoped`,
`agent`, `rewrite_*`).

| Target | Retrieves | Chain |
|---|---|---|
| `all` (default) | Text chunks and image captions, fused | The full postprocessor chain |
| `documents` | Text chunks only | The full chain, image lane never built |
| `visual` | The `_images` companion only | Rerank, numbering |

A pinned scope outranks the target: hand-picked chunks are hand-picked
whatever the target says.

**The visual target applies no relevance floor.** `IMAGE_RERANK_MIN_SCORE`
protects a *text* answer from a merely-nearest image, and its threshold sits on
a cross-encoder score answering "does this passage answer this question". A
question about the imagery — "what is shown in the documents?", the shape this
target invites — scores the correct image in the same band as an unrelated one,
so the floor emptied the evidence set and the turn answered from nothing. Here
the rerank's top-n already bounds the set, and the model is shown the pixels, so
it can say that none of the images shows the thing asked about.

**The visual target retrieves in two lanes.** CLIP similarity finds imagery
whose caption never names the thing asked about, and a keyword pass over the
companion's indexed caption/tag/OCR text finds the literal terms CLIP's
English-only text tower is weak at. The keyword pass runs on the *original*
query (the stored text is in the corpus's own language) and requires half its
keywords rather than all of them — a question carries more words than a
one-line caption can hold. The two rankings are fused by reciprocal rank, and
either lane failing degrades the turn rather than ending it: a dead CLIP
endpoint leaves keyword-only, a rejected scroll leaves CLIP-only.

**It answers from the pictures, not only their captions.** After ranking, the
stored 768px thumbnails of the top sources (`VISUAL_ANSWER_MAX_IMAGES`,
default 6) are attached to the same synthesis call as image parts, with a
legend tying each to its citation number. The context window is shortened by
what they cost, so the captions still fit beside them. This is the only place
in docint where a chat turn sends pixels. When the thumbnails cannot be
fetched the answer is written from the captions alone and reports
`visual: {"images_attached": 0}` — a degraded turn the SPA flags rather than
presenting as an ordinary one.

**A model that refuses the pictures degrades the same way.** The inference
endpoint caps how many images one prompt may carry (vLLM's
`--limit-mm-per-prompt`, `CHAT_LIMIT_MM_PER_PROMPT` on the vllm-service chat
container) and knows nothing of `VISUAL_ANSWER_MAX_IMAGES`, so the two can
disagree on any deployment. When the call is refused, the turn is answered
from the captions and reports `images_attached: 0` rather than failing. Set
the two knobs to agree — raise the endpoint's limit to answer from several
pictures at once, or lower `VISUAL_ANSWER_MAX_IMAGES` to what it accepts, so
the model actually sees the evidence it is citing.

**Its postprocessor chain is shorter, and deliberately.** Parent-context
expansion reads a docstore that holds no companion nodes; the social
diversity cap would collapse a clip's consecutive keyframes, which is the
opposite of what a "when does X appear" question needs; and link-following
would pull posting prose into a set that is meant to be imagery.

**Filters reach the companion collection itself.** Request filters used to
apply in memory, after the top-k cut, so a clip or time-range rule usually
left no imagery at all; the compiled Qdrant filter is now handed to the
companion, which applies it before ranking. This fixes the `all` target too.
The SPA's visual presets (kind of imagery, clip name, time range) compile to
`source_type`, `source_file` and `keyframe_time_sec` rules, and the
companion's indexes for those keys are created at ingest and backported on
the first visual query.

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
change, and so is an assessment turn (see [Assessment turns](#assessment-turns)).
And the second attempt overwrites the first attempt's turn
(`replace_turn_idx`) instead of appending, so one user message stays one turn.

Failures degrade rather than escalate: no reformulation available, or a second
pass that dies mid-stream, leaves the first answer as the delivered result —
a retry never turns a delivered answer into an error.

The cost is up to three extra LLM round-trips (reformulate, regenerate,
re-validate) inside the one request, so a turn that triggers a retry is
noticeably slower than one that does not.

## Assessment turns

A question that asks for a judgement — "what could be improved in this
paper?", "is the method convincing?", "which option is better?" — is not a
lookup. Its answer is written nowhere in the collection, and the passages that
best match its wording are rarely the ones a judgement rests on: asked what
could be improved, plain retrieval returns the paper's own "improvements over
prior work". Such turns are answered differently, and say so.

**Classification.** `/stream_query` and `/query` classify every turn once,
before retrieval, with one short call on the plain chat model
(`agents/assessment/classify.py`, prompt `prompts/{en,de}/assessment_classify.txt`).
Any failure — no model, no JSON, a transport error — classifies as a lookup,
which is the behaviour without this feature. No classifier call is made when
`ASSESSMENT_ENABLED=false`, when the resolved context window is below 8192
tokens (`agents/assessment/gate.py` logs one warning naming
`OPENAI_CTX_WINDOW`), on the `visual` target, or on a corrective retry.
Classification and research make a single attempt per call, whatever
`OPENAI_MAX_RETRIES` says, and their 15-second and two-minute deadlines hold
however a reply arrives: one that trickles in, each piece within the client's
timeout, is abandoned at the deadline all the same. Lookups and the answer keep
the retries.

**Evidence.** An assessment seeds its evidence with the passages a lookup
would have used — the turn's own engine, filters, rerank and parent-context
chain — then lets the plain model search further through a
`search(query)` tool until it calls `finish()` (`agents/assessment/research.py`,
prompt `assessment_research.txt`, `tool_choice="required"`). The seed and every
search keep every child the reranker returned. A section still yields one
passage: the whole section, a window of it, a cut of its first child when that
child outgrows the window, or the first child itself once the parent-context
budget is spent or the child cannot be located. That passage carries every child
retrieved under the section, each appended after a `…` line when the passage
does not show it whole. Lookups are unchanged. Research stops at
`ASSESSMENT_MAX_STEPS`, at the context budget or a two-minute research budget,
on a repeated query or keyword check, when the evidence ledger is full, when the
model replies without calling a tool, or after two searches in a row that add
nothing new (a keyword check with a known count always adds a fact; see
**Keyword checks**).
The research budget covers the model calls as well as the searches, and model
calls run like searches: each on its own thread, abandoned when the budget runs
out, and the turn reports `time_budget`; a reply that lands after the budget,
even a `finish`, is not acted on. Research searches and research model calls
each share a per-process bound of eight: an abandoned call holds its slot until
it ends, and when no slot frees within five seconds (or the time left, if less)
research stops with `research_failed`; a slot that frees only as the time runs
out ends research with `time_budget`, the call never started. The classifier
has eight slots of its own, so research never uses up what every turn needs:
with more than eight classifications in flight in one process, a further turn
waits up to a second for a slot and is answered as a lookup.
Research admits a passage only while every passage stays legible in the answer,
headers included; a result that would crowd the others is refused, and the
ledger counts as full once not even a bare 300-character passage would fit.
Every match an entry holds keeps room in the answer: each costs 250 characters
(200 of text and a cut marker) in its window, each further window 55, and an
extension whose matches would no longer fit is refused. Matches that touch or
overlap are counted apart, each keeping its own region.
A new window of a section the ledger already holds is not dropped. Text the entry
lacks extends it after a `…` line, a window that repeats it adds only a match
whose kept text the entry lacks, and the entry keeps its number; a window with a
located match never merges with one without, so whichever comes second adds
nothing. When the answer has to shorten such an entry, the cut is
shared across its windows rather than taken from the end. Shortening keeps the
text around each passage's retrieved match, so a section promoted from a matched
chunk is cut around that chunk rather than to its opening. A match longer than
200 characters, typically a whole retrieved child chunk, keeps only its first
200 when it has to be cut. The research model's digest of a passage shows only
text the answer is sure to keep: the first 200 characters of each match,
top-ranked first, up to 400 in all, or the passage's opening when it has no
match.
Each step reaches the SPA as an `{"agent": {...}}` frame. A pinned scope is
never widened: a scoped assessment answers from its chunks with no research.
Evidence too large to show legibly in one answer — typically a large pinned
scope — is answered as a lookup instead.

**Keyword checks.** When the collection's keyword index is ready, research also
gets `keyword(words, field)` (`agents/assessment/keyword.py`, described in prose
by `assessment_research_keyword.txt`). Every word must occur in the same chunk,
in any order, matched as the Search panel matches a word — case-insensitive, at
the start of a word — but several words are not checked as a phrase. A check
reports the exact number of matching indexed chunks within the turn's filters
(coarse parents excluded), then the most relevant new ones, retrieved through the
turn's retriever narrowed to the matches: the same rerank, image floor and
parent context as a search, without the social diversity cap and link-following,
so every passage it adds is a match. A text hit's matches are moved to just
before its first keyword, with up to 80 characters of left context, so the
digest and the answer show the word rather than the chunk's opening. `field` is
`text`, or on a social collection `author`, `network` or `uuid`, each offered
only when every key it searches carries its own index kind; the tool schema
limits the model to those. The image companion joins a check on the `all`
target only for fields an image carries and only when it carries the same
indexes; otherwise its chunks, like chunks the text index does not cover yet,
are reported as unchecked, since a zero is only as good as what was checked.
The tool is withheld, and research keeps `search` and `finish`, when the
request's filters cannot be compiled for Qdrant, when an image companion in
scope could not be counted, when no field is ready, and on any fault. Deciding
reads coverage and payload schemas only, and creates no index. A check runs on
the search slots under the research budget and fails as a search does; a count
that failed is reported unavailable, never zero. Input over 100 characters, no
word of two or more characters, or several values for `uuid` get an error reply.
A repeated check (same field, same words in any order) ends research like a
repeated query, but a search for the same words is no repeat. A check with a
known count is a fact even when it adds no passage, so only one whose count
failed and that added nothing counts toward the two idle rounds.

**Answer.** One streamed call on the answer model (the reasoning toggle
applies) writes two parts separated by a `[[ASSESSMENT]]` line
(`assessment_answer.txt`): what the sources say, every claim cited, then the
model's own assessment, which may draw on general knowledge and must say so.
The backend replaces the sentinel with a localized label (`ui_strings`
`assessment_heading` / `assessment_disclaimer`) in the stream and in the
persisted text, so the boundary survives into history, transcripts and report
exports without any of them knowing the protocol. Sources are the evidence
ledger, numbered 1..N in the order it was found; an entry is shortened to fit
the context rather than dropped, because replay numbers citations by position.
If the final estimate still does not fit, the latest research additions are
left out, never the seed, so the numbering stays contiguous, and the turn
reports `context_budget`; it does the same when what remains, a dense seed
included, can only be sent cut below its legible minimum. If the prompt
still overflows the model's context, it is rebuilt once with less evidence and
compacted metadata headers, each passage's citation line kept whole (a rebuild
that cuts passages below their minimum is reported the same way); a second
overflow, or compact headers that alone fill the room, ends the turn with
`context_overflow` on `/stream_query` (`/query` answers HTTP 500, as for a lookup).
Research's keyword checks with a known count reach the prompt after the
snippets, one line each (`- "words" (field): N`, with `, unchecked: M` on a
partial index). They are counts, not sources: Part 1 stays snippet-only, and
Part 2 may report a zero as no passage containing those words, never as proof
the idea is missing. The block exists only once research ends, so it is
measured then: it spends what the prior turn and question left of their reserve
first, and beyond that the latest research additions are left out as at any fit.

**Validation.** Only the source-backed part is validated, against the sources it
cites (at most six; the first six when it cites none), with a note
(`response_validator_assessment.txt`) that puts the
judgement out of scope. An answer that did not separate the two parts is
labelled as a whole and not validated. Assessments never take the corrective
retry: their evidence was gathered on purpose, not by one query that a
reformulation could improve.

**Degraded turns** say why. `agent.reason` is `tools_unavailable` (the endpoint
rejected tool calls or answered without one; the answer rests on the seed
passages), `research_failed`, `context_budget` (the context window ended
research early or never let it start, the last search lost a passage to the
room, passages were left out at a fit, or the evidence was sent below its
legible minimum; a refusal that a later search follows is not reported, and an
earlier reason is kept), `time_budget` or `no_evidence`;
`agent.answer` is `unsplit` or `empty`. These are protocol tokens; the SPA maps
each to catalog copy.

**Follow-ups.** A lookup asked after an assessment receives only the
source-backed part of that answer as its prior turn, so the model's judgement is
never restated as sourced fact. The cut is made where the prior turn is bound to
the generation prompt, so the legacy `/agent/*` endpoints, which build theirs
from stored history, get it too. The prior turn is read by
`SessionManager.get_prior_turn`, without rehydrating any citation. An assessment
asked after one sees both parts of it, each labelled: the source-backed part,
whose `[n]` refer to the previous turn's sources, and the model's own
assessment. They share the 4,000-character prior-answer budget shortest-first,
so a long source-backed part cannot push the judgement out. Research and the
answer also see the rewrite a lookup answers from, on a labelled line under the
user's own words, with reasoning and citation numbers dropped and cut at 1,000
characters. The line is left out when the rewrite adds no words, and on
stateless turns, which have no history. It is a reading aid, never a source.

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
