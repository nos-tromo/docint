# Chat full-text search panel — design

**Date:** 2026-08-07
**Status:** Approved, not yet implemented
**Supersedes:** the chat entity-occurrence query modes (`entity_occurrence`,
`entity_occurrence_multi`), which are removed by this work.

## Problem

The chat sidebar offers two "entity occurrence" query modes that were meant to
be a search index next to the chat window. They were built as *entity* search —
matching the query against the NER aggregate — and they do not deliver correct
results.

Three defects were confirmed while investigating:

1. **A natural-language query silently selects the wrong entity.** Every entity
   whose text appears anywhere in the query ties at rank 1
   (`core/ner.py::match_entity_text`, the `text_lower in query_lower` branch),
   and the tiebreak inside a rank is raw mention count
   (`core/rag.py::_collect_entity_matches`). The most *frequent* entity in the
   collection therefore wins, not the one the user asked about. Reproduced with
   a synthetic aggregate: a six-mention filler noun beat the two-mention place
   name the query was about.

2. **No disambiguation is offered in that case.** `_strong_entity_matches`
   narrows rank-1 hits to a single top alias, which suppresses the candidate
   panel that exists for exactly this situation.

3. **Chat entity search can never use resolved entities.**
   `_load_resolved_index()` is wired only into `get_collection_ner_stats` (the
   Analysis page). `run_entity_occurrence_query` hardcodes
   `entity_merge_mode="orthographic"` and passes no resolved index, and
   `entity_cluster_key` treats `"resolved"` exactly like orthographic anyway. So
   after entity resolution has run, Analysis merges an acronym with its expanded
   form and chat does not.

The intent was never entity search. It was raw full-text search over chunk
text — "needle in a haystack" — with entity type shown as an annotation where
one exists.

## Goals

- Raw full-text keyword search across every chunk of the caller's collection.
- Multiple keywords are ANDed, order-independent: two keywords list only chunks
  where both appear.
- Case-insensitive, including non-ASCII text.
- Prefix matching, so a query for the head of a German compound finds the
  compound (`Partei` finds `Parteitag`).
- Search lives in a panel beside the chat, not as a chat query mode: the hit
  list persists while the user asks questions about it.
- Selected hits **scope** the chat — subsequent questions are answered only from
  those chunks. The scope is sticky and pinned to the chat session.

## Non-goals

- Mid-word matching (`tag` finding `Parteitag`). Prefix only; see "Rejected
  alternatives".
- Ranking search hits by relevance. Hard match is a filter, not a ranker.
- A search box on the Analysis entity list. Deferred to a follow-up; see
  "Deferred work".
- Any use of sparse embeddings in the search path; see "Rejected alternatives".

## Verified platform behaviour

These were measured against a live Qdrant, not assumed. They are the load-bearing
facts behind the design and are pinned by an integration test (see "Testing").

| Behaviour | Result |
| --- | --- |
| `MatchText` without a payload index | Works — falls back to a scan |
| Un-indexed `MatchText`, ASCII case folding | Works (`berlin` matches `Berlin`) |
| Un-indexed `MatchText`, **non-ASCII** case folding | **Fails** — a title-case German token matched, the same token lowercased matched nothing |
| `TextIndexParams(tokenizer=PREFIX, lowercase=True)` | All cases match, including non-ASCII |
| Prefix matching under that index | `Partei` matches both `Partei` and `Parteitag` |
| Mid-word under that index | `tag` does not match `Parteitag` — as intended |
| Multiple `must` conditions | AND, order-independent |

The non-ASCII case-folding result is the reason the payload index is mandatory
rather than a performance optimisation: without it, case-insensitive search is
simply untrue for German content.

## Architecture

### New module: `docint/core/search/`

Kept out of `core/rag.py`, which is already ~8.7k lines.

#### `index.py` — owns the payload field and its index

- `SEARCH_TEXT_FIELD = "search_text"`
- `ensure_search_index(client, collection)` — idempotent `create_payload_index`
  with `TextIndexParams(type="text", tokenizer=PREFIX, lowercase=True,
  min_token_len=2, max_token_len=30)`. Best-effort and fail-soft, mirroring the
  existing `posting_uuid` index creation in `RAG.create_index`.
- `write_search_text(client, collection, texts, *, batch_size, wait)` — writes
  many distinct per-point payloads in **one** request via
  `batch_update_points`. Qdrant's `set_payload` applies a single payload to
  many points, so distinct per-point texts would otherwise cost one request
  each; this is what makes both the ingest hook and the backfill affordable.
  Point ids keep their own type — Qdrant ids are unsigned ints or UUIDs, and
  coercing an int id to a string targets a point that does not exist, so the
  write silently lands nowhere.
- `backfill_search_text(client, collection, *, extract_text, batch_size,
  force=False, progress=None)` — scroll the collection, extract each point's
  text, write it back through `write_search_text`. Payload-only: no
  re-embedding, no inference, no model download, so it is airgap-safe. Returns
  a `BackfillSummary(scanned, written, skipped, empty)`. Points that already
  carry `search_text` are skipped unless `force`. A point with no extractable
  text is marked with an **empty value** rather than left alone — otherwise it
  counts as missing forever, every search reports `partial`, and the warning
  stops meaning anything. Transient Qdrant write failures are retried, so one
  blip cannot leave a collection permanently half-indexed. `extract_text` is **injected**
  (the caller passes `RAG._extract_payload_text`) so this package never imports
  `core/rag.py` — it stays unit-testable without a RAG instance and cannot
  create a circular import, mirroring how `core/jobs.py` takes its `runner`.
- `search_index_status(client, collection)` — whether the payload index exists
  plus exact coverage counts (`total`, `with_search_text`, `missing`,
  `complete`). Counted, never sampled: see the `"partial"` note under the
  Search API section for why.

#### `fulltext.py` — pure, dependency-injected, like `core/retrieval_filters.py`

- `parse_keywords(raw) -> list[str]` — whitespace split, deduplicated, empties
  dropped.
- `build_search_filter(keywords, *, base_filter)` — one
  `FieldCondition(SEARCH_TEXT_FIELD, MatchText(keyword))` per keyword in `must`,
  AND-merged with the metadata filter supplied by the panel, so the filters
  constrain the search.

### Why a dedicated `search_text` field

The chunk text has no payload field of its own today. Across every existing
collection the text lives only inside `_node_content`, a JSON string that also
carries filenames, entity names, JSON keys and point UUIDs.

Indexing `_node_content` directly was rejected: a *prefix* index over that field
would store every prefix of every UUID, so the index would be dominated by
fragments that can never be searched for, and a query like `chunk` would match
every point in the collection.

`search_text` duplicates the chunk text in the payload. Because the text is
already duplicated inside `_node_content`, this adds one more copy of the text
bytes, not a doubling of collection storage.

### Ingest wiring

**Corrected during implementation (2026-08-08).** An earlier draft of this
section had `search_text` stamped into `node.metadata` with both exclusion lists
set. It is written **payload-only** instead, via `batch_update_points`, straight
after a successful insert and keyed by `node_id` — llama-index uses the node id
as the Qdrant point id.

Node metadata is the wrong home for it on four counts:

- Metadata is rendered into the embedding input unless excluded. A missed or
  reset exclusion would put every chunk's text into its own embedding twice,
  silently degrading retrieval in a way nobody would notice for months.
- Metadata is serialized *into* `_node_content`, so stamping there stores the
  text three times per chunk (node text, metadata copy, payload key) rather
  than twice.
- Retrieved nodes rebuild metadata from `_node_content`, so a missed
  `excluded_llm_metadata_keys` would push the full chunk text into the prompt a
  second time.
- The ingest hook and the backfill call the same `write_search_text()`, so the
  migration path is exercised by every ingest instead of rotting until someone
  runs it.

The cost is one extra Qdrant round-trip per persistence batch, beside an
embedding call in that same batch. The write is fail-soft end to end: a
surprising node shape or a Qdrant outage degrades search to "needs a backfill"
rather than failing ingestion.

### Hierarchical duplicates

**Corrected during implementation (2026-08-08).** An earlier draft restricted
search to `docint_hier_type == "fine"`. Search **excludes `"coarse"`** instead.

With `HIERARCHICAL_CHUNKING_ENABLED`, a coarse parent and its fine child both
contain the keyword, so a naive search returns both for a single logical hit.
But verified against a live Qdrant with points tagged `fine`, tagged `coarse`,
and untagged: requiring `"fine"` returns *only* the tagged one, so a collection
ingested **without** hierarchical chunking — which tags nothing — would return
zero hits for every search. Excluding `"coarse"` keeps the fine node and the
untagged node while still dropping the duplicate parent.
`core/rag.py:3605-3607` already uses exclude-coarse for exactly this reason.

### Data flow

```
panel keywords ──┐
                 ├─► build_search_filter ──► Qdrant scroll
metadata filters ┘        (MatchText AND, prefix, lowercase, fine nodes only)
                                    │
                                    ▼
                       hits: chunk_id, filename, page/row,
                       preview text, entity types, est_tokens
```

No embeddings and no inference call appear anywhere in this path.

### Search API

`POST /search` — POST rather than GET because the body carries
`metadata_filters`, mirroring `QueryIn`.

```jsonc
// request
{
  "collection": "<logical name>",   // owner-gated, resolved per request
  "query": "keyword1 keyword2",     // whitespace-split into ANDed keywords
  "metadata_filters": [],           // same MetadataFilterIn shape as /query
  "limit": 50,                      // 1..500
  "cursor": null                    // opaque, from a previous page
}

// response
{
  "status": "ok",                   // or "not_indexed"
  "hits": [
    {
      "chunk_id": "…",
      "id": "…",                    // Qdrant point id — what the scope stores
      "filename": "…",
      "page": 3,                    // or "row" for tabular sources
      "preview": "…",
      "entity_types": ["LOC"],      // from the chunk's own NER payload, may be []
      "est_tokens": 412
    }
  ],
  "total": 14,                      // exact, via Qdrant count with the same filter
  "next_cursor": null,
  "index_status": { "indexed": true, "total": 724, "with_search_text": 724, "missing": 0, "complete": true }
}
```

Owner-gated through `_require_owned_collection` and bound per request with
`RAG.collection_scope`, like every other collection-scoped endpoint.

- **Pagination** uses the opaque-cursor idiom already established by
  `GET /collections/ner/sources`: `limit` defaults to 50 and is clamped to
  `[1, 500]`, and `next_cursor` is `null` on the last page. Because a scope can
  span pages, selection state is held by the panel and keyed by point id, not by
  page position.
- **`total`** is exact and comes from a Qdrant `count` with the same filter. The
  document count shown beside it is derived from the hits loaded so far and is
  rendered as `6+ docs` while `next_cursor` is non-null, rather than implying a
  collection-wide figure it does not have.
- **`status: "not_indexed"`** is returned when the collection carries no
  `search_text` at all, with `hits` empty. The panel renders the migration
  prompt for this state. An empty `hits` list with `status: "ok"` therefore
  means "genuinely no matches" and nothing else.
- **`status: "partial"`** is a third state, added during implementation after
  review. Coverage is **counted exactly** (two Qdrant `count` calls), not
  sampled. A head sample cannot distinguish a finished backfill from one that
  has written only its first page — the backfill walks the collection from the
  same offset a sample would — so a search issued mid-migration would have
  reported plain success while silently omitting every chunk not yet written.
  `index_status` carries `total` / `with_search_text` / `missing` / `complete`,
  and `partial` is a distinct *status* rather than only a nested field so a
  client cannot miss incomplete coverage by ignoring `index_status`. The two
  counts are also cheaper than the payload-bearing scroll pages a sample needed.
- **Keywords shorter than `min_token_len` (2 characters) are not indexable.**
  They are rejected with a clear message naming the offending keyword rather
  than silently contributing an `must` condition that can never match.

### Migration

Payload-index creation is automatic and idempotent. The backfill for existing
collections ships as an explicit one-off command: a `search-index` CLI entry
point plus a `make search-index` target modelled on the existing `make resolve`
(one-off backend container, so it reaches the `qdrant` network alias), taking
`COLLECTION=` or prompting for it.

Searching a collection that has not been backfilled returns an explicit
"not search-indexed yet" state, never an empty result list — an empty list must
never be able to mean "the migration has not run".

## Session-pinned scope

### Persistence

Two nullable columns on the existing `conversations` table:

```python
scope_chunk_ids = Column(Text, nullable=True)      # JSON list of Qdrant point ids
scope_set_at    = Column(DateTime, nullable=True)
```

Added by an `_ensure_conversation_scope_columns(engine)` helper following the
idiom already established in `core/state/base.py`: `ALTER TABLE … ADD COLUMN`,
fail-soft with a warning, so an existing sessions DB upgrades in place.
`SessionManager` gains owner-scoped `get_scope` / `set_scope` / `clear_scope`,
threaded per request like the rest of that class.

### API

- `PUT /sessions/{id}/scope` `{chunk_ids: [...]}` — owner-gated, validates the
  token budget, returns the accepted scope and its measured cost.
- `DELETE /sessions/{id}/scope`
- The scope is included in `GET /sessions/{id}` so a reload restores it.

### Token budget guard

Scoped answering splices the selected chunks straight into the prompt, so the
selection is bounded by the chat context window rather than by a top-k.

Each search hit carries `est_tokens`, computed with the existing
`estimate_tokens(text, ratio)`, so the panel sums the selection client-side and
displays remaining capacity live without a round-trip per checkbox. `PUT`
re-validates server-side against `_compute_parent_context_budget()`'s
`usable_tokens`.

An oversized scope is **refused**, with the measured numbers returned so the UI
can explain the shortfall. It is never silently truncated: dropping part of an
investigator's evidence without saying so is the worst available failure mode.

### Answering

When a session has a scope, `RAG.chat` takes a different path: no retriever, no
vector query, no rerank. It fetches the points by id, normalises them through the
**same** `_source_from_payload` used by the retrieval path — so citations, the
source panel, "Add to report" and Inspector links keep working unchanged — packs
them into `context_str` in stable order, and synthesises with the existing
grounded templates. `retrieval_mode` is reported as `scoped`.

### Drift

Re-ingestion mints new point ids, so a scope can outlive its chunks. When
`retrieve` returns fewer ids than requested, the difference is surfaced as
`scope_missing: N` rather than quietly answering from the remainder. The panel
reports it and prompts a re-search.

### Selection semantics

The scope *is* the selection, and it is sticky. Running a new search does not
clear it, and hits already in scope come back pre-checked. A
"Scoped to N chunks · clear" banner sits in the chat column so a scoped answer is
never a surprise.

Scope is per session: a new chat starts unscoped. No process-global scope state
is introduced, consistent with the multi-tenant rule in `CLAUDE.md` against
process-global active-collection or session state.

Chat turns record only `scoped_chunk_count`, not a copy of the scope, so report
exports can state that an answer was scoped without duplicating evidence the
report builder already snapshots.

## UI

The right column of the Chat route becomes one collapsible panel holding search
(primary) and the metadata filters (secondary, since filters constrain search).

```
expanded                                          collapsed
┌─ Chat ──────────────────┬─‹─┬─ Search ────────┐  ┌─ Chat ──────────────────────┬─›─┐
│ Scoped to 14 chunks ·   │   │ [keyword kw2  ]🔍│  │ Scoped to 14 chunks · clear │ 14│
│ clear                   │ r │ 14 hits · 6 docs │  │                             │  2│
│                         │ a │ ~12.4k / 22k     │  │  > question…                │   │
│  > question…            │ i │ ┌──────────────┐ │  │                             │   │
│                         │ l │ │[x] file p.3  │ │  │  answer…                    │   │
│  answer…                │   │ │  …preview…   │ │  │                             │   │
│                         │   │ │[x] file p.7  │ │  │                             │   │
│                         │   │ │[ ] file r.12 │ │  │                             │   │
│ [ask…]          [Send]  │   │ └── scrolls ──┘ │  │ [ask…]              [Send]  │   │
│                         │   │ > Filters (2)   │  │                             │   │
└─────────────────────────┴───┴─────────────────┘  └─────────────────────────────┴───┘
```

- **Filters are a disclosure at the foot of the column**, not a tab. They are set
  occasionally and search is used constantly, so filters must not hold half the
  vertical space permanently. Collapsed they are a one-line `Filters (2)`
  summary; expanded they overlay the result list.
- **Collapse affordance** is a slim chevron rail on the panel edge, not a header
  hamburger: `text-muted-foreground` at rest, full contrast on hover/focus,
  `aria-expanded`, persisted in the existing `stores/chatUi.ts` store, with an
  animated `grid-template-columns` transition so the chat reflows rather than
  popping. Collapsed, the rail keeps two badges — hit count and active-filter
  count — because a panel that silently filters or scopes while hidden is a trap.
- **Hits** show filename plus page/row, a preview with the matched keywords
  highlighted (client-side, replicating prefix + lowercase matching), entity
  type badges where the chunk carries NER metadata, a checkbox that writes the
  scope, and a link into the Inspector.
- All controls use the `@infra/ui` primitives (`Input`, `Select`, `Button`,
  `Badge`, `Card`). The current filter panel hand-rolls raw `<input>`/`<select>`
  with `bg-muted` on a `bg-muted` card, giving the controls no contrast at all.

## Removals

- `core/rag.py`: `run_entity_occurrence_query`,
  `run_multi_entity_occurrence_query`, `_build_entity_occurrence_group`,
  `_collect_entity_matches`, `_strong_entity_matches`,
  `_entity_candidate_payload`, `_flatten_occurrence_groups`.
- `core/api.py`: the `entity_occurrence*` branches in `/query` and
  `/stream_query`.
- Frontend: `components/chat/EntityCandidatesPanel.tsx`, the query-mode dropdown,
  and the `chat.mode_*` i18n keys in both `en` and `de`.
- The corresponding tests in `tests/test_rag_unit.py` and `tests/test_api.py`.

`core/ner.py::match_entity_text` **stays** — the deferred Analysis search box
still needs it.

With both entity options gone, `query_mode` has one legal value left, so
`QueryIn.query_mode` is dropped from the API and the modes block in the panel
shrinks to the Retrieval selector alone.

## Prerequisite: metadata-filter fixes

The metadata filter panel is separately broken, and the search panel inherits its
payload, so these land first and independently:

1. The SPA emits `date_gte` / `date_lte`, which are not in
   `MetadataFilterIn.operator`'s `Literal` — every request carrying a date filter
   is rejected with 422. The API's operators are `date_on_or_after` /
   `date_on_or_before`.
2. The SPA filters dates on a field named `date` and hate speech on
   `hate_speech_flagged`. The real keys are `reference_metadata.timestamp`
   (plus `reference_metadata.posting_timestamp` for media artifacts linked to a
   posting) and `hate_speech.hate_speech`.
3. Matching either timestamp key requires a rule that ORs across several fields.
   `MetadataFilterIn` gains `fields: list[str]`, compiled to a nested OR group in
   `build_metadata_filters`, a nested `Filter(should=[...])` in
   `build_qdrant_filter`, and an any-field match in `matches_metadata_filters`.
4. `_compile_rule` emits filter shapes the Qdrant vector store cannot compile:
   `date_*` operators produce `Range(gte=<ISO string>)` where a float is
   required; `contains` produces `FilterOperator.CONTAINS`, which
   `QdrantVectorStore` raises `NotImplementedError` for; `gt`/`gte`/`lt`/`lte`
   with a non-numeric value hit the same `Range` validation error. All three are
   already expressible by the native `build_qdrant_filter`, so `_compile_rule`
   returns `None` and logs, exactly as it already does for booleans.
5. `vector_store_kwargs["qdrant_filters"]` *overrides* the LlamaIndex
   `MetadataFilters` inside `QdrantVectorStore.query`, so the internal
   `docint_hier_type == "fine"` parent-context filter is silently dropped
   whenever any user filter is active. `PARENT_CONTEXT_RETRIEVAL_ENABLED`
   defaults to `True`, so this is the default path. The internal condition is
   merged into the native filter instead.

## Testing

- Pure unit tests for `parse_keywords`, `build_search_filter`, and budget
  refusal.
- Fake-client tests for backfill idempotence and the `force` path.
- The scope-column migration tested against a pre-existing DB, following the
  existing owner-column migration test.
- An **opt-in integration test against a live Qdrant, skipped when unreachable**,
  pinning the behaviours in "Verified platform behaviour": prefix matching,
  non-ASCII case folding, mid-word non-matching, and AND across keywords. This
  earns its keep — the non-ASCII case-folding assumption was wrong at the start
  of this design and only a real server exposed it. A mocked client would have
  confirmed the bug.
- Frontend tests for the panel: keyword highlighting, scope checkbox writes,
  budget display, collapse persistence, and the "not search-indexed yet" state.

## Implementation sequencing

This is too large for one plan. It decomposes into three, each independently
shippable and independently useful:

**Plan 1 — metadata-filter fixes.** Everything under "Prerequisite" above.
Touches `core/retrieval_filters.py`, `core/api.py`, `core/rag.py`,
`stores/chatFilters.ts`. Ships on its own: it repairs a panel that is broken
today, with or without search.

**Plan 2 — search backend.** `core/search/`, the ingest stamping, the payload
index, the backfill command, and `POST /search`. No UI. Verifiable end to end
against a live collection through the API alone.

**Plan 3 — scope and panel.** The `conversations` columns and their migration,
the scope endpoints, scoped answering in `RAG.chat`, the right-column UI, and the
entity-occurrence removals. Depends on both preceding plans.

The removals land in Plan 3 rather than earlier: the entity modes stay usable,
however imperfectly, until there is a replacement in the UI to switch to.

## Rejected alternatives

**Sparse embeddings for search.** `bge-m3` sparse is a ranker, not a filter: it
scores chunks by weighted lexical overlap and returns a top-k, so a chunk
containing only one of two keywords is still returned, merely lower. AND
semantics cannot be expressed. It also tokenises into XLM-R subwords, so a hard
match is not guaranteed, and it costs a round-trip to the sparse endpoint per
query. Qdrant's native `MatchText` delivers the requested semantics exactly, with
no inference call. Sparse remains available for ranking a filtered set if
relevance ordering is ever wanted.

**Indexing `_node_content` instead of a dedicated field.** See "Why a dedicated
`search_text` field".

**Scroll-and-filter in Python with no index.** Correct, and no schema change, but
a full scan of every chunk with a JSON parse per point on every search. Fine at
a few hundred points, not at production scale.

**Full substring matching.** Would let `tag` find `Parteitag`, but Qdrant cannot
index for it, so any query without a usable token prefix degrades to a
collection scan — on exactly the vague queries that already return the most hits.
German compounds are head-final, so the discriminating fragment is normally the
start of the word; prefix covers the realistic case and keeps every query
index-backed.

**Search as a chat query mode.** Results would become chat turns and scroll away,
and a result list could not stay open while asking questions about it. A panel
keeps the hit list and the conversation visible together, which is the point.

**One-shot or per-tab scope.** An investigator asking several questions about the
same working set should not reselect for each one, and a scope that vanishes on
reload cannot be reported honestly in an export. Session-pinned scope is
consistent with collections already being pinned to sessions.

## Deferred work

- **Analysis entity search box.** A keyword box filtering the Analysis entity
  list, case-insensitive hard match. Note that `GET /collections/ner/search`
  already exists (`q`, `entity_type`, `limit`, `entity_merge_mode`) and is
  entirely unused by the frontend — but it routes `q` through
  `match_entity_text`, the fuzzy matcher responsible for defect 1 above, so it
  needs a hard-match path before it can back the box. Wiring it would also lift
  the current client-side `top_k: 500` ceiling on which entities are findable.
- **Resolved-entity merging in chat.** Defect 3 above is not addressed by this
  work; the search path is raw text and does not consult the entity index at all.
  If entity-aware search returns later, it must load the resolved index.
