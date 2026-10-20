import { apiDelete, apiGet } from './client'
import type { SessionMessage, SessionSummary } from './types'

export const listSessions = (collection?: string) =>
  apiGet<{ sessions: SessionSummary[] }>('/sessions/list', { collection })

/** A session's history, served only through the collection the session is pinned to. */
export const getSessionHistory = (id: string, collection: string) =>
  apiGet<{ messages: SessionMessage[] }>(`/sessions/${encodeURIComponent(id)}/history`, { collection })

export const deleteSession = (id: string) =>
  apiDelete<{ ok: boolean }>(`/sessions/${encodeURIComponent(id)}`)
