import { apiDelete, apiGet, apiPost, url } from './client'
import type { DocumentRecord, EntityMergeMode, HateSpeechRow, NerSourceRow, NerStats } from './types'

export interface Page<T> {
  items: T[]
  next_cursor: string | null
}

export const listCollections = () => apiGet<string[]>('/collections/list')

export const selectCollection = (name: string) =>
  apiPost<{ ok: boolean; name: string }>('/collections/select', { name })

export const deleteCollection = (name: string) =>
  apiDelete<{ ok: boolean }>(`/collections/${encodeURIComponent(name)}`)

export const listDocuments = () =>
  apiGet<{ documents: DocumentRecord[] }>('/collections/documents')

export const getDocumentsPage = (params: { cursor?: string | null; limit?: number }) =>
  apiGet<Page<DocumentRecord>>('/collections/documents', {
    cursor: params.cursor ?? undefined,
    limit: params.limit ?? 50
  })

export const getDocumentsCount = () =>
  apiGet<{ count: number }>('/collections/documents/count')

export const getNerStats = (params: {
  top_k?: number
  min_mentions?: number
  entity_type?: string
  include_relations?: boolean
  entity_merge_mode?: EntityMergeMode
}) => apiGet<NerStats>('/collections/ner/stats', params)

export const getNerSourcesPage = (params: {
  cursor?: string | null
  limit?: number
  entity_key?: string
  entity_text?: string
  entity_type?: string
}) =>
  apiGet<Page<NerSourceRow>>('/collections/ner/sources', {
    cursor: params.cursor ?? undefined,
    limit: params.limit ?? 50,
    entity_key: params.entity_key,
    entity_text: params.entity_text,
    entity_type: params.entity_type
  })

export const warmCollectionNer = () =>
  apiPost<{ ok: boolean }>('/collections/ner/warm')

export const getHateSpeech = () =>
  apiGet<{ results: HateSpeechRow[] }>('/collections/hate-speech')

export const getHateSpeechPage = (params: {
  cursor?: string | null
  limit?: number
  category?: string
  min_confidence?: string
}) =>
  apiGet<Page<HateSpeechRow>>('/collections/hate-speech', {
    cursor: params.cursor ?? undefined,
    limit: params.limit ?? 50,
    category: params.category,
    min_confidence: params.min_confidence
  })

export const getIeStats = (collection: string) =>
  apiGet<unknown>(`/collections/${encodeURIComponent(collection)}/ie-stats`)

export type CsvExportKind = 'documents' | 'entities' | 'ner-sources' | 'hate-speech'

export interface CsvExportParams {
  entity_key?: string
  entity_text?: string
  entity_type?: string
  category?: string
  min_confidence?: string
  top_k?: number
  min_mentions?: number
}

/**
 * Build an absolute URL pointing at one of the streaming CSV export
 * endpoints. Use this as the ``href`` of a download anchor so the browser
 * handles the streaming response natively.
 */
export function csvExportHref(
  collection: string,
  kind: CsvExportKind,
  params: CsvExportParams = {}
): string {
  const qs = Object.entries(params)
    .filter(([, v]) => v !== undefined && v !== null && v !== '')
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`)
    .join('&')
  const suffix = qs ? `?${qs}` : ''
  return url(`/collections/${encodeURIComponent(collection)}/export/${kind}.csv${suffix}`)
}
