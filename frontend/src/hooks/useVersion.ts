import { useQuery } from '@tanstack/react-query'
import { getVersion } from '@/api/config'

/**
 * The running app release version, for the AppHeader's version slot. Fetched
 * once per session — the backend's version is constant for the process
 * lifetime.
 */
export function useVersion() {
  return useQuery({
    queryKey: ['version'],
    queryFn: getVersion,
    staleTime: Infinity,
    gcTime: Infinity
  })
}
