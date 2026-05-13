import { apiDelete, apiGet, apiPost } from './client'
import type { DocumentRecord, NerStats } from './types'

export const listCollections = () => apiGet<string[]>('/collections/list')

export const selectCollection = (name: string) =>
  apiPost<{ ok: boolean; name: string }>('/collections/select', { name })

export const deleteCollection = (name: string) =>
  apiDelete<{ ok: boolean }>(`/collections/${encodeURIComponent(name)}`)

export const listDocuments = () =>
  apiGet<{ documents: DocumentRecord[] }>('/collections/documents')

export const getNerStats = (params: {
  top_k?: number
  min_mentions?: number
  entity_type?: string
  include_relations?: boolean
  entity_merge_mode?: 'orthographic' | 'exact'
}) => apiGet<NerStats>('/collections/ner/stats', params)

export const getNer = (refresh?: boolean) =>
  apiGet<{ entities: unknown[]; relations: unknown[] }>('/collections/ner', { refresh })

export const getHateSpeech = () =>
  apiGet<{ results: unknown[] }>('/collections/hate-speech')

export const getIeStats = (collection: string) =>
  apiGet<unknown>(`/collections/${encodeURIComponent(collection)}/ie-stats`)
