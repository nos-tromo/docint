import { QueryClient } from '@tanstack/react-query'
import { subscribePurpose } from '@infra/ui'
import { ApiError, PURPOSE_SCOPE, isPurposeRefusal } from './client'

const MAX_RETRIES = 1

/**
 * Retry policy for queries: never retry a deterministic client error (4xx),
 * otherwise allow up to MAX_RETRIES for transient failures (network, 5xx).
 *
 * A 401 from `/sessions/list` (missing principal) is deterministic, so the
 * sidebar should surface its actionable message immediately rather than after
 * a wasted refetch.
 */
export function retryPolicy(failureCount: number, error: unknown): boolean {
  if (error instanceof ApiError && error.status >= 400 && error.status < 500) {
    return false
  }
  return failureCount < MAX_RETRIES
}

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 30_000, retry: retryPolicy, refetchOnWindowFocus: false }
  }
})

/**
 * Refetch the queries refused for want of a purpose as soon as one is stated.
 *
 * A `428` the user answered in the purpose dialog is resent there and then
 * (`withPurpose`); one whose dialog they dismissed leaves its query in error,
 * and nothing would ask again until a reload. Stating a purpose in the bar
 * later is that ask. A query already fetching again is left to finish.
 *
 * @param client - The app's query client.
 * @returns A function that stops listening.
 */
export function refetchRefusedOnPurpose(client: QueryClient): () => void {
  return subscribePurpose(PURPOSE_SCOPE, (purpose) => {
    if (!purpose) return
    void client.refetchQueries(
      { type: 'active', predicate: (query) => query.state.status === 'error' && isPurposeRefusal(query.state.error) },
      { cancelRefetch: false }
    )
  })
}
