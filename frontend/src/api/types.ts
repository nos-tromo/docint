export interface Source {
  // Backend (docint/core/rag.py::_source_from_payload) does not emit a
  // stable per-source id, so this is optional and only present for
  // sources that flow through node-id-aware code paths.
  id?: string
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
  entity_match_candidates?: unknown[]
  entity_match_groups?: unknown[]
}

export interface MetadataFilter {
  field: string
  operator: string
  value?: unknown
  values?: unknown[]
}

export type QueryMode = 'answer' | 'entity_occurrence' | 'entity_occurrence_multi'
export type RetrievalMode = 'stateless' | 'session'

export interface ChatRequest {
  question: string
  session_id?: string
  metadata_filters?: MetadataFilter[]
  retrieval_mode?: RetrievalMode
  query_mode?: QueryMode
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
  session_id: string | null
  created_at: string | null
  updated_at: string | null
  item_count: number
}

export interface Report extends ReportSummary {
  items: ReportItem[]
}

export interface IngestEvent {
  event:
    | 'start'
    | 'upload_progress'
    | 'file_saved'
    | 'ingestion_started'
    | 'ingestion_progress'
    | 'ingestion_complete'
    | 'error'
  data: Record<string, unknown>
  /**
   * Client-side wall-clock time (ms since epoch) at which this event was
   * received from the SSE stream, stamped once by `streamIngestUpload`.
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
}

export interface SummaryResponse extends ValidationFields {
  summary: string
  sources: Source[]
  summary_diagnostics?: SummaryDiagnostics
}
