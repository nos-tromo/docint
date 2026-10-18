# Retrieval and agents

This document describes the read path of Docint: from a user question to
a grounded answer, including the agent orchestration layer, the RAG
engine, and the postprocessing stages that shape the final result.

## Agent orchestration

All agent code lives under `docint/agents/`. The entry point is
`AgentOrchestrator` (`docint/agents/orchestrator.py:70`), which wires up
four optional agents behind a single `handle_turn()` method. It serves the
`/agent/chat*` endpoints. Chat turns on `/query` and `/stream_query` run as
[agent turns](#agent-turns) (`docint/agents/turn/`) when agent turns are on and
available, and as plain lookups otherwise.

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
- **Stateless query** — `run_query(prompt, expanded_query=...,
  metadata_filters=..., vector_store_kwargs=...)` retrieves with
  `expanded_query` (the prompt itself when omitted), answers `prompt`, and
  returns a `dict` with `response`, `sources`, `retrieval_query`,
  `coverage_unit`, and `retrieval_mode`.
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

An agent turn fetches the pinned chunks through the same scoped engine and
answers from them in one round that offers no tools (see
[Pinned scope](#pinned-scope)).

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

Reranking is **fail-soft**: a transport failure degrades to the order
the retriever returned rather than failing the query. With the image lane
running that is the text and image hits fused by rank (see
[Image retrieval lane](#image-retrieval-lane)), so the cut to the top-n
keeps each lane's best hits. The outcome is stamped on each returned node
(`docint_rerank_applied`, `docint_rerank_error`) so `/query` responses
report whether the rerank actually ran; a query with no text to rank
against is stamped the same way, with the error `empty_query`. Each failed
call (unreachable endpoint, HTTP error, malformed payload) is also counted
in `docint_rerank_failures_total`, exported on `GET /metrics`, and logged
as a warning. The image re-score makes its own rerank call, which counts and
logs the same way when it fails but leaves a turn's `rerank` field alone.
The reranker is wrapped in `LazyRerankerPostprocessor`, which defers client
construction and its healthcheck to the first query — building a query
engine for warmup or introspection does not pay that cost.

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
  matching one both land in the same narrow CLIP band. A reranker asks whether
  a caption *answers* the query, so a question about the imagery scores even a
  relevant image far below a caption-shaped query: 0.0044–0.0065 measured,
  against the floor's 0.05. That is why the floor alone drops such images, and
  why they get the second look below. Raise the floor if unrelated images still
  appear. It applies to the `all` target only — see the visual target below.
- `IMAGE_RESCORE_ENABLED` (default `true`), `IMAGE_RESCORE_MIN_SCORE` (default
  `0.02`) and `IMAGE_RESCORE_TIMEOUT` (default `10`, seconds) — the second look
  for a sub-floor image, described below.

**The lanes reach the rerank fused.** Text and image hits are handed to the
rerank as one list, fused by reciprocal rank over each lane's own order: each
lane's best hit leads, and a tie goes to the text hit. Rank decides because the
raw scores — text similarity and CLIP cosine — are not on one scale. A healthy
reranker re-scores the whole list and ignores that order. It matters when the
rerank fails: stacking one lane behind the other would let the cut to the top-n
remove a whole modality.

**If the rerank endpoint is down, the floor stands down.** A failed rerank hands
the nodes back with their retrieval scores, and an image's CLIP cosine is not on
the reranker's scale, so measuring it against `IMAGE_RERANK_MIN_SCORE` says
nothing about relevance: CLIP cosines sit in a ~0.20–0.30 band whatever the
query, so every one would clear the default floor, and a floor raised above the
band would drop them all. The stamp on the nodes (`docint_rerank_applied=false`)
tells the floor so, and so does a set with no scores at all. No image is gated
or re-scored then; the set is only cut to the top-n, in its fused order. The cut
is what matters here: with the text hits stacked ahead of the images it removes
the images, and in the fused order it leaves each lane its best hits instead of
the answer shipping text alone. The response reports the turn as
`rerank.applied: false`.

**A sub-floor image inside the cut gets one second look.** While the rerank is
healthy, an image under the floor that would still land inside the top-n —
counting every node kept, or already waiting for this look, ahead of it — is
re-scored against a description of the query instead of the query itself. The
chat model (`TEXT_MODEL`) writes the description in `RESPONSE_LANGUAGE`: one
line listing the visible cues the question needs (and the kind of shot, indoors
or outdoors, when it matters), each with its usual alternatives rather than a
guessed answer, and with people described by what is visible, never by name
(`prompts/{en,de}/image_rescore_description.txt`). The candidates' captions are
then reranked once against it, and an image reaching `IMAGE_RESCORE_MIN_SCORE`
is kept at the position and with the score it already had. An image the cut
would remove anyway is never a candidate, so a turn with none makes no extra
call, and the `visual` target, which applies no floor, never re-scores.

The description is a chat call that runs before the answer's first token, so it
has its own deadline, `IMAGE_RESCORE_TIMEOUT`, and a single attempt rather than
the chat client's timeout and retries. It also asks for no reasoning whatever
`OPENAI_ENABLE_THINKING` says (no `reasoning_effort`, and on
`INFERENCE_PROVIDER=vllm` `chat_template_kwargs.enable_thinking` sent as
`false`), because a model that thought first could spend the whole deadline.
Like the rest of the lane it fails soft: no description, a failed rerank
(counted as described under [Reranking](#reranking)) or any other error drops
the sub-floor images exactly as the floor alone would, and never fails the
turn. Successful descriptions are cached per query, language and model, so an
identical query costs only the second rerank. At `LOG_LEVEL=DEBUG` each
re-score that completes logs one counts-only line, `Image re-score |
candidates=… rescued=… elapsed_ms=…`, never the query or the description.

Measured with a local reranker (eight descriptions against nine captions), the
true matches that cleared `0.02` scored 0.039–0.86 against their description,
the true matches that did not 0.0003–0.0119, and the strongest non-match 0.0072.
The re-score rescues some question-phrased misses and admitted no measured
non-match; a true match under the re-score floor still drops, as it would
without the re-score. In a live replay the model-written description rescued
the right keyframe for person and weather questions (0.03–0.18) while
unrelated captions stayed below 0.02. The figures come from small runs on a
local reranker; raise `IMAGE_RESCORE_MIN_SCORE` if unrelated images are
rescued.

Full defaults and rationale:
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

Each stored turn records the target it answered under, and the assistant entry
in `GET /sessions/{id}/history` returns it, so a reloaded chat can still say what
drew each answer. A scoped turn records none, for the reason above, and turns
stored before the field existed have none either; the SPA shows nothing for
either.

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
what they cost, so the captions still fit beside them. The visual target is the
only one on which a chat turn sends pixels; an agent turn attaches them round by
round instead (see [Visual target](#visual-target)). When the thumbnails cannot be
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

The appended names serve retrieval alone. A session lookup retrieves with the
rewritten, expanded query but is asked the user's own words, with the
history-resolved rewrite on a labelled line beneath them when it adds any
(`question_block` in `docint/agents/turn/history.py`). A follow-up such as
"Are those all?" is therefore answered as asked, and the neighbour names never
reach the generation prompt. A stateless query has no rewrite: it retrieves with
the expanded question and is asked the question alone. The `retrieval_query` and
`graph_debug` the response reports do not change.

An agent turn runs no turn-level expansion. Each query its model writes is
expanded for that search alone, and its `graph_debug` reports the skip with the
reason `agent`.

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

An agent turn is validated by its answer mode (see
[Answer, validation and storage](#answer-validation-and-storage)).

## Corrective retry

A rejected answer that is *also* weak — empty, under 40 characters, or a
refusal (`is_weak_answer` in `agents/orchestrator.py`) — gets one automatic
second attempt when `CORRECTIVE_RETRY_ENABLED=true`. A statement about the
evidence itself ("evidence is insufficient", "keine Belege") makes an answer a
refusal wherever it appears. A missing-detail statement ("does not mention",
"geht nicht hervor", "keine Angaben") does so only when the answer is nothing
but such statements: it cites no source, and setting those sentences aside
leaves under 40 characters. `QueryReformulationAgent`
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
  persisted on the turn so a reloaded session still names the retry. The
  second pass retrieves with the reformulation, which it never rewrites again,
  and is asked the user's own words, the reformulation on a labelled line
  beneath them.

The gate is deliberately narrow. A mismatched but substantive answer is kept —
discarding it to chase a better one trades a real answer for a coin flip. That
includes an answer that flags the one detail it lacks ("the log does not
mention the captain's name"). A
scoped turn is skipped entirely, since it runs no retrieval for a new query to
change, and so is an agent turn: it wrote its own searches, and a plain second
pass would overwrite its stored row. A turn an agent turn handed to the plain
path because tools were refused retries as any plain lookup does, and keeps the
trail that says why.
And the second attempt overwrites the first attempt's turn
(`replace_turn_idx`) instead of appending, so one user message stays one turn.

Failures degrade rather than escalate: no reformulation available, or a second
pass that dies mid-stream, leaves the first answer as the delivered result —
a retry never turns a delivered answer into an error.

The cost is up to three extra LLM round-trips (reformulate, regenerate,
re-validate) inside the one request, so a turn that triggers a retry is
noticeably slower than one that does not.

## Agent turns

Every chat turn on `/query` and `/stream_query`, session or stateless, is
answered by an agent loop when the deployment can run one
(`docint/agents/turn/`). The model reads the conversation and the message,
calls tools to search the collection, reads every reply, and goes again until
it answers. Its last reply, the one that calls no tool, is the answer, streamed
as it is written. The plain path described above answers instead when agent
turns are off or cannot run, and a turn falls back to it as long as nothing has
been shown (see [Fallback to the plain path](#fallback-to-the-plain-path)). The
legacy `/agent/chat*` endpoints keep the orchestrator's plain path.

### When a turn is an agent turn

The API asks for an agent turn (`_agent_turn_allowed` in `core/api.py`) when
all of these hold:

- a text model is configured;
- `AGENT_TURNS_ENABLED` is on, which is the default;
- the resolved context window is at least 8192 tokens
  (`agents/assessment/gate.py`). Below that, agent turns stay off, with one
  warning per window size naming `OPENAI_CTX_WINDOW`;
- no rejection of tools is cached for the request's reasoning flag (see
  [When the endpoint refuses tools](#when-the-endpoint-refuses-tools)).

Every retrieval target may run one. If the check itself fails, the turn is
answered on the plain path. The corrective retry's second pass is always a
plain lookup.

`RAG.agent_turn` then builds the turn. It measures four parts:

- the system prompt: `agent_system.txt`, filled with the target's tools guide
  (`agent_tools_{all,documents,visual}.txt`), the assessment rule while
  `ASSESSMENT_ENABLED` is on, and the conversation's summary block;
- the offered tool schemas;
- the recent exchanges;
- the message.

The room for tool replies is what these leave of the window, less a 5% margin
and the answer's reserve: 1024 tokens, or 3072 when reasoning is on. With less
than 1500 tokens of room no turn is built, and the plain path answers. One
warning gives the numbers.

### The loop

Each round is one streamed model call, offered the turn's tools with
`tool_choice: "auto"`.

- **A round that calls tools** has every call answered, as a `tool` message
  under the call's id, and the model goes again. Arguments that are not a
  JSON object are echoed back as `{}`, so a chat template that parses earlier
  calls never meets them.
- **A round that calls none** is the answer.
- **A round's text is provisional.** It is held until 120 visible characters
  have arrived with no tool call begun, then streamed. If the round still ends
  in calls, or breaks off, a `note` step tells the client to withdraw what was
  streamed. The note carries that text, cut to 500 characters, and the next
  tokens begin a fresh answer.
- **The first round is held whole.** None of its text streams before it ends.
  Text it wrote before its calls becomes its note without ever showing as
  answer text, and an answer goes out once it is complete. A first round that
  breaks off has shown nothing, so its failure is handled as one before any
  output (see [Fallback to the plain path](#fallback-to-the-plain-path)).
- **A first round that looks nothing up is asked again.** If the first round
  calls no tool, its reply is dropped unseen, its reasoning with it. The search
  note (`agent_search_first.txt`) is joined to the user's message, and the
  round runs once more. The note asks the model to look up what its answer
  relies on, including any phrase or name the user just gave, and to cite the
  passages. Its cost counts against the room for tool replies.
- **The second try runs under the first round's rules.** It is no tool round,
  but it is a model call. The limits are checked before it, so a turn whose
  first round used up the time or the room answers at once. It waits for a model
  slot as the first round does. It fails as the first round does, except that a
  refusal of it never reads as the endpoint taking no tools. It streams as later
  rounds do; if it calls no tool either, its reply is the answer. A pinned scope
  offers no tools and is never asked again. One INFO line, with no text, says
  the round was asked again.
- **Reasoning is never streamed as the answer.** Think blocks, and a reasoning
  field the endpoint sends apart from the content, are kept for the response's
  `reasoning` field, every round's in order.
- **Every round runs on the request's model.** The reasoning toggle picks it,
  and it makes one attempt per call, whatever `OPENAI_MAX_RETRIES` says,
  because the loop's time budget and slots assume one.
- **Model calls share a bound.** A process runs at most eight agent-turn model
  calls at once. A tool round waits up to five seconds for a slot, never longer
  than the time left. The answer of a turn that has shown something waits up
  to the time left, and at least five seconds.
- **Calls run at once.** A round's calls run together, up to
  `AGENT_MAX_CALLS_PER_ROUND`. They share the process-wide eight search slots,
  run in a copy of the request's context, and are numbered in call order.
  Calls past the limit are answered "not run".

When a limit is reached, the model answers in one more round. That round
sends the same conversation, with the final note (`agent_final.txt`) joined to
the last user message, so a strict chat template never sees two user messages
in a row. The tools stay declared, with `tool_choice: "none"`.

An endpoint that ignores `tool_choice` may still answer that round with tool
calls and no text. The round is then sent once more with no tools declared,
neither `tools` nor `tool_choice`, and its reply is the answer. Only if that
reply is empty too is the answer left empty. One warning, with the number of
calls, says the round was sent again.

### Tools offered

| Target | Tools |
|---|---|
| `documents` | `search`, `read` |
| `visual` | `search_images`, `keyframes`, `read` |
| `all` | `search`, `search_images`, `keyframes`, `read` |

- **`search(query)`** retrieves through the turn's own engine, so the request's
  filters, rerank and parent-context expansion apply as on a lookup. Every
  retrieved child of a section is kept (`EveryChildQuery`). Each query is
  expanded with graph neighbours for retrieval alone; the reply and the trail
  show the model's own words.
- **`search_images(description)`** runs the visual target's chain over the
  `{collection}_images` companion, under the request's filters: CLIP and the
  keyword lane fused, then reranked against the description, with no relevance
  floor. The description names visible cues, never a guessed answer.
- **`keyframes(file, index)`** shows one clip's frames in time order, labelled
  `#n @ mm:ss`. A clip of up to eight frames shows them all; a longer one shows
  eight evenly spaced, its first and last among them, with the total. `file` is
  the clip's file name or the number of a passage from the clip, and `index`
  opens one frame. A file name that two clips share is answered with a way to
  name each, never with a merged list.
- **`read(n)`** shows passage `[n]` again, as much of it as one reply holds.

A reply numbers only what it shows:

- a passage new to the turn is shown under the number the answer cites it by
  (its `citation_index:` line);
- a passage shown before is named by its number ("Also matched, shown
  before: [1]");
- a passage that grew is named with a note that `read(n)` shows its new text;
- matching passages that did not fit are counted, with a hint to search again
  with narrower words.

A call made before is answered from memory and never run again; a `read` only
while its passage is unchanged. A call to a tool the turn does not offer, or
with arguments that will not do, gets an error reply saying what to send
instead.

A reply holds at most 3000 characters, and each passage's text at most 900;
with reasoning on, 2000 and 700. A header shows whole up to 500 characters;
past that it keeps its citation line and 240 characters of the rest. The model
reads passages through an agent view. That view cuts metadata strings to 200
characters, and leads a social row's header with its file and who wrote it.

### Evidence and sources

The turn's evidence is everything its tool replies showed, numbered in the
order it was found. The windows of one section are one passage, which keeps its
number as it grows. The evidence holds at most 64 passages, or as many as a
larger pinned scope holds. The answer's sources are this evidence in number
order, so the model's `[n]` is the number on the source card.

The trail (`agent.actions`) holds one `{action, detail, found}` per note, and
one per call its tool answered. A call that was not answered gets its reply but
no step: one skipped, refused for room or time, failed, or carrying an argument
error.

### Limits and reasons

`agent.reason` names the first cause that cut a turn short or left evidence out.
The one exception is `answer_cut`: a cut answer matters more than a limit or a
failure before it, so it replaces whatever reason stood.

| `agent.reason` | When |
|---|---|
| `round_budget` | The turn ran `AGENT_MAX_ROUNDS` tool rounds (default 6). |
| `time_budget` | `AGENT_TIME_BUDGET_S` ran out (default 120 s). Limits are checked between rounds, so an answer already streaming is never cut, and a search that outlives the time left gets a "no reply in time" reply. |
| `context_budget` | No room is left for one more full reply. A round whose own calls do not fit the room left is withdrawn, its shown text withdrawn by its note. The reason is also set when an overflow is retried once as the answer with every tool reply halved. |
| `idle` | Two rounds in a row added no passage. |
| `research_failed` | Evidence may be missing; the answer is still written. A call could not be prepared, a search failed or found no search slot, or a later round failed or found no model slot. |
| `answer_cut` | The endpoint cut the answer at its length limit (`finish_reason: "length"`), so it may be incomplete. The answer is kept as written. This covers every round whose text is the answer: one answering by choice, the forced answer and its resend without tools, and a pinned round. |
| `tools_unavailable` | The endpoint refused tools, and the plain path answered (see below). |
| `no_evidence` | Found only on turns stored by the earlier assessment pipeline. |

These are protocol tokens; the SPA maps each to its own copy.

### Fallback to the plain path

Until the turn has shown a step or text, these failures hand the message to the
plain path, which answers as it would without agent turns:

- the budget leaves under 1500 tokens for tool replies;
- the first call, tools declared, is refused with HTTP 400 or 422 that is not
  an overflow. The plain answer then reports
  `agent: {"steps": 0, "actions": [], "reason": "tools_unavailable", "answer": "plain"}`,
  and the stored turn keeps it;
- no model slot comes free in time;
- a pinned scope cannot be shown legibly in the room;
- the prompt overflows the context window.

Any other failure before anything was shown, such as a server error, a timeout
or a first round breaking off mid-reply, is the turn's error, as it would be on
a plain lookup. A refusal of a later call, the first round's second try
included, is such a failure too: only the turn's first call can show that the
endpoint takes no tools.

Once a step or text is out, a failed tool call or tool round no longer ends the
turn: the model answers from what it found, with `research_failed`. What does
end it is a failure of the answer itself: the answer round failing, finding no
model slot, or overflowing once its tool replies were already cut. Its resend
without tools ends the turn in the same ways, and on any overflow. The turn
then sends one `error` frame and stores nothing. The code is `context_overflow`
for an overflow, and `generation_failed` otherwise.

### When the endpoint refuses tools

A turn's refusal can come from its own content: a message, a picture, or a
schema its chat template trips. Switching agent turns off on that would let
one request turn them off for every user. So the turn's refusal starts a probe
on the same model client, before anything is cached. The probe carries nothing
of any user or collection:

- a fixed system line and the message `ping`;
- one tool, `noop`, with no parameters, and `tool_choice: "auto"`;
- streamed, one attempt, with a 10-second timeout.

| Probe | What follows |
|---|---|
| Refused (HTTP 400 or 422, not an overflow) | The rejection is cached: for 10 minutes, turns with that reasoning flag are plain lookups. One warning names the flag and the requirement. |
| Accepted | Nothing is cached. One warning, naming the flag, says that something in that turn's own request was refused. |
| Inconclusive (a timeout, a 5xx, a 429, an unreachable endpoint) | Nothing is cached. One warning, naming the flag, says the probe could not tell. |

The cache and the probe are kept per reasoning flag, since the toggle picks
the client and an endpoint may refuse tools in one mode only. At most one probe
runs at a time per flag. A refusal that arrives during a probe, or while a
rejection is cached, probes nothing. Either way, the turn that hit the refusal
has already fallen back. The session layer also logs each refusal at INFO, with
its HTTP status.

### Pinned scope

A turn with a pinned scope answers from the pinned passages alone, in one
round that offers no tools. That round sends neither `tools` nor
`tool_choice`, and its refusal is the turn's own error, never a rejection of
tools. The passages are rendered into the user message between fence lines,
after the scope note (`agent_scope.txt`) and before the user's words, within the
turn's room. Pinned evidence is never dropped: a scope the room cannot show
legibly is answered on the plain path's scoped engine instead. The trail opens
with a `seed` step (step 0) that counts the pinned passages. The response
reports `retrieval_mode: "scoped"` and `scoped_chunk_count`. A scope outranks
the target, so no picture is shown.

### Visual target

On the `visual` target the loop shows the model the pictures themselves. After
each tool round, the passages the round admitted that are pictures have their
stored thumbnails fetched, in one call. They are shown in one user message after
the round's tool replies: a legend (`agent_image_legend.txt`) tying each
picture to its citation number, then the images.

- A turn shows at most `VISUAL_ANSWER_MAX_IMAGES` pictures, and no more than
  its room holds. A picture costs about 600 tokens, plus its legend line.
- A picture is shown once per turn.
- A round that carries pictures and fails before its first event is taken as
  refusing them. Every picture is taken back, and the round is sent once more
  without them.

The response reports `visual: {"images_attached": n}`, where `0` means the
model saw no pixels. Only the visual target shows pictures; turns on `all` and
`documents` see captions.

### Answer, validation and storage

While `ASSESSMENT_ENABLED` is on (the default), the system prompt carries the
assessment rule. A question that asks for a judgement is then answered in two
parts: what the sources say, a line holding only `[[ASSESSMENT]]`, then the
model's own assessment, which may draw on general knowledge and must say so.

- **An assessment** is a reply carrying the sentinel: `answer_mode:
  "assessment"`, `agent.answer` `split` (or `empty` for a reply of nothing but
  the sentinel), and `evidence_chars`, the length of the source-backed part. The
  backend swaps the sentinel for a localized label (`ui_strings`
  `assessment_heading` / `assessment_disclaimer`) in the stream and in the
  stored text. History, transcripts and report exports then show the boundary
  without knowing the protocol.
- **A lookup** is any other reply: `answer_mode: "lookup"`, `agent.answer:
  "plain"`.

With the switch off, every agent turn answers as a lookup.

Validation follows the mode. An assessment is validated on its source-backed
part only, against the sources that part cites, with a note
(`response_validator_assessment.txt`) that puts the judgement out of scope. An
answer without a separate source-backed part is labelled but not validated. An
agent lookup is validated against the sources it cites, with the user's
previous message shown ahead of the question. Either way the validator sees at
most six sources, the first six when the answer cites none. Agent turns never
take the corrective retry.

An agent turn is stored with its `answer_mode` and its trail. It reports
`retrieval_mode: "agent"` (or `"scoped"`), `retrieval_query: null` and a
`graph_debug` skipped with the reason `agent`. Its `Turn complete | …` log line
adds `answer=`, `steps=`, `outcome=` and `reason=`: counts and protocol tokens,
never a query, an argument or document text.

### Conversation history

An agent turn reads the conversation before its message, built before the turn
is stored (`agents/turn/history.py`). It holds:

- the last two exchanges, as real messages, or one when reasoning is on;
- the user's earlier questions, newest first;
- the newest part of the rolling summary.

All of it fits within 1300 tokens, or 750 with reasoning. Citation numbers are
dropped from earlier answers, since they number earlier sources. An earlier
assessment shows its two parts, each labelled. A stateless turn reads none.
The model writes its own searches, so an agent turn runs no history rewrite and
no turn-level graph expansion.

A plain lookup that follows an assessment (a fallback, or any turn while agent
turns are off) receives only the source-backed part of that answer as its prior
turn. The model's judgement is never restated as sourced fact.

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
