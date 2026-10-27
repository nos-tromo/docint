import { apiDelete, apiGet } from './client'
import type { SessionMessage, SessionSummary, SessionTopicFocus } from './types'

export const listSessions = (collection?: string) =>
  apiGet<{ sessions: SessionSummary[] }>('/sessions/list', { collection })

/** A session's history, served only through the collection the session is pinned to. */
export const getSessionHistory = (id: string, collection: string) =>
  apiGet<{ messages: SessionMessage[] }>(`/sessions/${encodeURIComponent(id)}/history`, { collection })

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
