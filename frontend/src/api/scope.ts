import { apiDelete, apiPut } from './client'
import type { ScopeResult } from './types'

/**
 * Pin a session's answers to a hand-picked set of chunks.
 *
 * The backend refuses (422) a selection larger than the chat context window
 * rather than truncating it — silently dropping part of an investigator's
 * evidence would produce an answer that looks complete and is not. A 422 is
 * therefore terminal: surface it, never retry.
 *
 * @param sessionId - The session to scope.
 * @param chunkIds - Qdrant point ids to answer from.
 * @param collection - Caller's logical collection, needed to measure the cost.
 * @returns The stored scope and its measured cost.
 */
export const setScope = (sessionId: string, chunkIds: string[], collection?: string) =>
  apiPut<ScopeResult>(
    `/sessions/${encodeURIComponent(sessionId)}/scope` +
      (collection ? `?collection=${encodeURIComponent(collection)}` : ''),
    { chunk_ids: chunkIds }
  )

/**
 * Return a session to normal retrieval.
 *
 * @param sessionId - The session to unscope.
 * @returns An empty scope.
 */
export const clearScope = (sessionId: string) =>
  apiDelete<ScopeResult>(`/sessions/${encodeURIComponent(sessionId)}/scope`)
