import { apiDelete, apiGet } from './client'
import type { SessionMessage, SessionSummary } from './types'

export const listSessions = () =>
  apiGet<{ sessions: SessionSummary[] }>('/sessions/list')

export const getSessionHistory = (id: string) =>
  apiGet<{ messages: SessionMessage[] }>(`/sessions/${encodeURIComponent(id)}/history`)

export const deleteSession = (id: string) =>
  apiDelete<{ ok: boolean }>(`/sessions/${encodeURIComponent(id)}`)
