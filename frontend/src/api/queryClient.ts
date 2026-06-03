import { QueryClient } from '@tanstack/react-query'
import { ApiError } from './client'

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
