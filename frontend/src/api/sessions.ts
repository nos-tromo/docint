import { apiDelete, apiGet } from './client'
import type { Analysis, SessionMessage, SessionSummary, SessionTopicFocus } from './types'

export const listSessions = (collection?: string) =>
  apiGet<{ sessions: SessionSummary[] }>('/sessions/list', { collection })

/** A session's history, served only through the collection the session is pinned to. */
export const getSessionHistory = (id: string, collection: string) =>
  apiGet<{ messages: SessionMessage[] }>(`/sessions/${encodeURIComponent(id)}/history`, { collection })

/** One analysis of a session, served only through the collection the session is pinned to. */
export const getSessionAnalysis = (sessionId: string, analysisId: string, collection: string) =>
  apiGet<Analysis>(`/sessions/${encodeURIComponent(sessionId)}/analyses/${encodeURIComponent(analysisId)}`, {
    collection
  })

export const deleteSession = (id: string) =>
  apiDelete<{ ok: boolean }>(`/sessions/${encodeURIComponent(id)}`)

/**
 * The topic a session was opened with through "Ask in chat", or null for none.
 * Served, like the history, only through the collection the session is pinned to.
 */
export const getSessionTopic = (id: string, collection: string) =>
  apiGet<{ topic_focus: SessionTopicFocus | null }>(`/sessions/${encodeURIComponent(id)}/topic`, {
    collection
  })

/** Remove a session's topic; the session and its turns stay. */
export const clearSessionTopic = (id: string) =>
  apiDelete<{ ok: boolean }>(`/sessions/${encodeURIComponent(id)}/topic`)

/** API path of the ZIP of every source a session cites, for `downloadFile`. */
export const sessionSourcesPath = (id: string) => `/sessions/${encodeURIComponent(id)}/sources.zip`
