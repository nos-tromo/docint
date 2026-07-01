import { useQuery } from '@tanstack/react-query'
import { getConfig } from '@/api/config'

/**
 * Deploy-time frontend configuration (graph node default + ceiling, collection
 * timeout). The backend reads these from env at process start, so the value is
 * constant for the session: fetch once and never refetch.
 */
export function useConfig() {
  return useQuery({
    queryKey: ['app-config'],
    queryFn: getConfig,
    staleTime: Infinity,
    gcTime: Infinity
  })
}
