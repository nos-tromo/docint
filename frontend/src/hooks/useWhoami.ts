import { useQuery } from '@tanstack/react-query'
import { getWhoami } from '@/api/config'

/**
 * The signed-in principal, for the AppHeader's user slot. Fetched once per
 * session — like the version, the resolved identity is constant for the
 * lifetime of the trusted-header session. Loading and error states both
 * resolve to `undefined` so the header simply omits the user block rather
 * than showing a stale or error placeholder.
 */
export function useWhoami() {
  return useQuery({
    queryKey: ['whoami'],
    queryFn: getWhoami,
    staleTime: Infinity,
    gcTime: Infinity,
    retry: false
  })
}
