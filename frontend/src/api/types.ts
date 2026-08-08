export interface Source {
  // Node id of the chunk the source came from (docint/core/rag.py::
  // _source_from_payload). Optional because payloads that carry no node id
  // and no chunk id — image matches, synthesized rows — have no identity.
  id?: string
  // Reader-assigned chunk id, present for readers that mint one (page-level
  // PDF chunks). Preferred over `id` when displaying a citation identity.
  chunk_id?: string
  // The number the generator saw this snippet under, stamped server-side
  // (docint/core/rag.py::CitationNumberingPostprocessor) so the answer's
  // "source 3" and this card are the same chunk. Never derive it from the
  // card's position: the list is deduped and image matches are appended
  // after generation. Absent for sources that never reached the prompt.
  citation_index?: number
  file_hash?: string
  filename: string
  filetype?: string | null
  source?: string | null
  page?: number | null
  row?: number | null
  score?: number | null
  text?: string
  preview_text?: string
  reference_metadata?: ReferenceMetadata
  entities?: Entity[]
  relations?: Relation[]
  ner?: { entities?: Entity[]; relations?: Relation[] }
}

export interface Entity {
  text: string
  type: string
  count?: number
  variants?: string[]
}

export interface Relation {
  subject: string
  predicate: string
  object: string
  count?: number
}

export interface ValidationFields {
  validation_checked?: boolean
  validation_mismatch?: boolean
  validation_reason?: string | null
}

export interface ChatFinalEvent extends ValidationFields {
  status?: 'answer' | 'clarification'
  answer?: string
  message?: string
  sources: Source[]
  session_id: string
  intent?: string
  confidence?: number
  tool_used?: string
  reason?: string
  graph_debug?: unknown
  retrieval_query?: string
  coverage_unit?: string
  /** `'scoped'` when the turn answered from a hand-picked chunk set. */
  retrieval_mode?: string
  /** How many chunks that scope held. */
  scoped_chunk_count?: number
}

/** A rule targets either one `field` or several `fields`; with several, it
 *  matches when any of them matches. The API rejects a rule naming neither. */
export interface MetadataFilter {
  field?: string
  fields?: string[]
  operator: string
  value?: unknown
  values?: unknown[]
}

export type RetrievalMode = 'stateless' | 'session'

/** One chunk matching every keyword of a full-text search (`POST /search`). */
export interface SearchHit {
  /** Qdrant point id — the value a scope is written with. */
  id: string
  chunk_id?: string | null
  /** Document hash — lets a hit deep-link into the Inspector's source preview. */
  file_hash?: string | null
  filename?: string | null
  page?: number | null
  row?: number | null
  preview: string
  entity_types: string[]
  est_tokens: number
}

/** How much of the collection carries the `search_text` field. */
export interface SearchIndexStatus {
  indexed: boolean
  total: number
  with_search_text: number
  missing: number
  complete: boolean
}

/**
 * A search response.
 *
 * `status` is load-bearing and must never be flattened into "no results":
 * `not_indexed` means the collection was never backfilled (`make search-index`),
 * `partial` means a backfill is incomplete so the hit list is short by an
 * unknown amount, and only `ok` with an empty `hits` means "no matches".
 */
export interface SearchResult {
  status: 'ok' | 'partial' | 'not_indexed'
  hits: SearchHit[]
  total: number
  next_cursor: string | null
  index_status: SearchIndexStatus
}

export interface SearchRequest {
  question: string
  collection?: string
  metadata_filters?: MetadataFilter[]
  limit?: number
  cursor?: string
}

/** A session's pinned scope plus what it costs against the chat budget. */
export interface ScopeResult {
  chunk_ids: string[]
  est_tokens: number
  usable_tokens: number
  /** Scoped chunks Qdrant no longer has (a re-ingest mints new point ids). */
  missing: number
}

export interface ChatRequest {
  question: string
  session_id?: string
  // Caller's logical collection (`useUiStore.selectedCollection`). The WS2
  // backend is stateless per request: `/query` and `/stream_query` owner-gate
  // and scope on this field, so concurrent users on different collections no
  // longer clobber a shared server-side "active collection". Omitted only when
  // nothing is selected yet.
  collection?: string
  metadata_filters?: MetadataFilter[]
  retrieval_mode?: RetrievalMode
}

export interface SessionSummary {
  id: string
  title?: string | null
  created_at: string
  collection?: string | null
}

export interface SessionMessage extends Partial<ValidationFields> {
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  reasoning?: string
}

export interface DocumentRecord {
  filename: string
  file_hash: string
  mimetype?: string
  page_count?: number
  row_count?: number
  node_count?: number
  entity_types?: string[]
}

export interface FileTypeCount {
  label: string
  count: number
}

/**
 * Collection-wide document aggregates for the Inspector KPI strip, served by
 * `GET /collections/documents/summary`. Computed over the whole collection, so
 * the file-type / entity-type breakdown is accurate regardless of how many
 * document pages the paginated table has loaded.
 */
export interface DocumentsSummary {
  document_count: number
  node_count: number
  file_types: FileTypeCount[]
  entity_types: string[]
}

export interface NerVariant {
  text: string
  type?: string
  mentions?: number
  score?: number | null
}

export interface NerEntityRow {
  text: string
  type: string
  mentions: number
  best_score?: number | null
  source_count?: number
  variant_count?: number
  variants?: NerVariant[]
}

export type ReferenceMetadata = Record<string, unknown> & {
  network?: string | null
  type?: string | null
  uuid?: string | null
  posting_uuid?: string | null
  posting_id?: string | null
  media_id?: string | null
  url?: string | null
  posting_network?: string | null
  posting_author?: string | null
  posting_author_id?: string | null
  posting_vanity?: string | null
  posting_timestamp?: string | null
  posting_url?: string | null
  posting_text?: string | null
  timestamp?: string | null
  author?: string | null
  author_id?: string | null
  vanity?: string | null
  text?: string | null
  text_id?: string | null
  parent_text?: string | null
  anchor_text?: string | null
  speaker?: string | null
  language?: string | null
  detected_language?: string | null
  source_file?: string | null
}

export interface NerEntityMention {
  text: string
  type: string
  score?: number | null
  key?: string
}

export interface NerSourceRow {
  chunk_id?: string
  chunk_text?: string
  text?: string
  filename?: string
  filetype?: string | null
  source?: string | null
  file_hash?: string
  page?: number | null
  row?: number | null
  score?: number | null
  preview_url?: string | null
  document_url?: string | null
  reference_metadata?: ReferenceMetadata
  entities?: NerEntityMention[]
  relations?: Array<{ head?: string; label?: string; tail?: string }>
}

export interface NerTypeRow {
  type: string
  mentions: number
  unique_entities: number
}

export interface NerRelationRow {
  head: string
  label: string
  tail: string
  mentions: number
}

export interface NerDocumentRow {
  filename: string
  entity_mentions: number
  unique_entities: number
  ie_source_count: number
  entity_density: number
}

export type EntityMergeMode = 'orthographic' | 'exact' | 'resolved'

/**
 * The entity-merge mode the UI requests. The backend's `entity_merge_mode`
 * parameter also accepts 'exact'/'orthographic' (see `docint/core/ner.py`),
 * but the frontend has no control for switching modes — it always pins to
 * the durable, canonical-entity grouping.
 */
export const ENTITY_MERGE_MODE: EntityMergeMode = 'resolved'

export interface NerStats {
  totals: {
    unique_entities: number
    entity_mentions: number
    unique_relations: number
  }
  top_entities: NerEntityRow[]
  entity_types: NerTypeRow[]
  top_relations: NerRelationRow[]
  documents: NerDocumentRow[]
}

export interface NerGraphNode {
  // Cluster key (e.g. `compact::type` or `ent::<id>`); not the `text::type`
  // selection key. Map a node to an entity for drill-down via `text`/`type`.
  id: string
  text: string
  type: string
  mentions: number
}

export interface NerGraphEdge {
  source: string
  target: string
  label: string
  // "relation" (extracted head→tail) or "cooccurrence" (entities in one chunk).
  kind: string
  weight: number
}

export interface NerGraph {
  nodes: NerGraphNode[]
  edges: NerGraphEdge[]
  meta: { node_count: number; edge_count: number }
}

export interface HateSpeechRow {
  chunk_id?: string
  filename?: string
  page?: number | null
  page_label?: string | null
  row?: number | null
  file_hash?: string
  chunk_text?: string
  text?: string
  category?: string
  confidence?: string
  reason?: string
  source_ref?: string
  reference_metadata?: ReferenceMetadata
}

// --- Report builder ---
export type ArtifactType = 'chat_answer' | 'entity_finding' | 'hate_speech_finding' | 'summary'

export type ReportExportFormat = 'md' | 'html' | 'pdf' | 'json' | 'zip'

/** A frozen artifact snapshot; its shape varies by `artifact_type`. */
export type ReportSnapshot = Record<string, unknown>

export interface ReportItemInput {
  artifact_type: ArtifactType
  dedupe_key: string
  snapshot: ReportSnapshot
  note?: string | null
}

export interface ReportItem {
  id: number
  artifact_type: ArtifactType
  dedupe_key: string
  position: number
  note: string | null
  snapshot: ReportSnapshot
  created_at: string | null
}

export interface ReportSummary {
  id: number
  title: string
  collection_name: string | null
  operator: string | null
  reference_number: string | null
  show_toc: boolean
  show_collection_overview: boolean
  session_id: string | null
  created_at: string | null
  updated_at: string | null
  item_count: number
}

export interface Report extends ReportSummary {
  items: ReportItem[]
  collection_overview: CollectionOverviewSnapshot | null
}

export interface CollectionOverviewDocument {
  filename: string
  mimetype?: string | null
  type_label: string
  page_count: number
  row_count: number | null
  node_count: number
  file_hash: string
}

export interface CollectionOverviewSnapshot {
  collection: string
  captured_at: string
  document_count: number
  node_count: number
  file_types: { label: string; count: number }[]
  entity_types: string[]
  documents: CollectionOverviewDocument[]
}

export interface IngestEvent {
  event:
    | 'start'
    | 'upload_progress'
    | 'file_saved'
    | 'ingestion_started'
    | 'ingestion_progress'
    | 'ingestion_complete'
    | 'warning'
    | 'error'
    // The owner-multiplexed `/ingest/jobs/events` stream is multiplexed
    // across job kinds, not just ingest runs (`jobs.py::KIND_EVENTS`) — a
    // summary-rebuild job's frames land in the same store as an ingest job's
    // (see `useIngestJobStream.ts`, which applies no kind filter), so this
    // union must cover both lifecycles.
    | 'summary_started'
    | 'summary_progress'
    | 'summary_completed'
  data: Record<string, unknown>
  /**
   * Client-side wall-clock time (ms since epoch) at which this event was
   * received from the SSE stream, stamped once by `streamIngestUploadBatched`.
   *
   * The ingest elapsed timer is derived from this rather than read from the
   * wall clock inside `deriveIngestStatus`, so that derivation stays a pure
   * function of its inputs. Re-deriving status on every incoming event must
   * not move `startedAt` (otherwise the timer resets on each batch).
   */
  receivedAt?: number
}

export interface SummaryDiagnostics {
  total_documents: number
  covered_documents: number
  coverage_ratio: number
  uncovered_documents: string[]
  coverage_target: number
  candidate_count: number
  deduped_count: number
  sampled_count: number
  /** True when the build hit `SUMMARY_MAX_LLM_CALLS` and covers only part of
   *  the collection. Absent on payloads cached before the flag shipped. */
  partial?: boolean
}

export interface SummaryResponse extends ValidationFields {
  summary: string
  sources: Source[]
  summary_diagnostics?: SummaryDiagnostics
}

/**
 * A queued-build acknowledgement from `POST /summarize` (202, or a 409's
 * adopted in-flight job). Carries only the job id — progress arrives
 * separately on the owner-multiplexed `GET /ingest/jobs/events` stream,
 * tagged with this same `job_id`.
 */
export interface SummaryJobQueued {
  job_id: string
}

/**
 * `POST /summarize`'s full result shape: a cache hit answers the summary
 * directly (200), a miss queues a background build and answers just the
 * `job_id` (202). Callers must discriminate on `'summary' in result` rather
 * than casting — see `SummaryPanel.tsx`.
 */
export type SummarizeResult = SummaryResponse | SummaryJobQueued

/**
 * Server-owned ingest job snapshot, served by the `/ingest/jobs*` endpoints.
 * `collection` is the caller's logical name — the physical owner-namespaced
 * name is never echoed to the client.
 */
export interface IngestJobSnapshot {
  job_id: string
  collection: string
  status: 'queued' | 'running' | 'completed' | 'failed'
  message: string | null
  error: string | null
  empty: boolean
  resolution: Record<string, number> | null
  created_at: string
  started_at: string | null
  finished_at: string | null
}

export interface AppConfig {
  graph_top_k: number
  graph_max_top_k: number
  collection_timeout: number
  /**
   * Per-request upload ceiling in bytes, mirroring the nginx
   * `client_max_body_size` the frontend enforces. The Ingest view splits a
   * large file selection into batches that each stay under this (times a
   * safety margin) so no single POST is rejected with 413.
   */
  max_upload_bytes: number
  /** Active UI/response locale, driven by `RESPONSE_LANGUAGE` on the backend. */
  language: 'en' | 'de'
}
