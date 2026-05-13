export interface Source {
  id: string
  file_hash: string
  filename: string
  page_label?: string | null
  row_label?: string | null
  score: number
  text?: string
  reference_metadata?: Record<string, unknown>
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
  validation_status?: 'ok' | 'warning' | 'failed' | string
  validation_message?: string | null
  validation_details?: Record<string, unknown> | null
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
  session_id: string
  title?: string | null
  created_at: string
  updated_at: string
  message_count: number
}

export interface SessionMessage {
  role: 'user' | 'assistant'
  content: string
  citations?: Source[]
  created_at: string
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

export interface NerStats {
  totals: { entities: number; relations: number; documents: number }
  top_entities: Array<{ text: string; type: string; count: number }>
  entity_types: Array<{ type: string; count: number }>
  top_relations: Array<{ subject: string; predicate: string; object: string; count: number }>
  documents: Array<{ filename: string; entity_count: number }>
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
