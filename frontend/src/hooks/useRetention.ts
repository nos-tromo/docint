import { useQuery } from '@tanstack/react-query'
import { getCollectionsRetention } from '@/api/collections'
import { collectionsKey } from '@/hooks/useCollections'
import { useConfig } from '@/hooks/useConfig'

/** Under the collections key, so every invalidation of the listing refreshes the dates too. */
export const retentionKey = [...collectionsKey, 'retention'] as const

/** The deployment's retention window, or `null` while retention is off (or unknown). */
export function useRetentionWindow(): string | null {
  const { data } = useConfig()
  const window = data?.collection_retention
  return window && window !== 'off' ? window : null
}

/**
 * Every listed collection's deletion date. Asks nothing while retention is off,
 * so a deployment that never enabled it never sees a deadline anywhere.
 */
export function useCollectionsRetention() {
  const window = useRetentionWindow()
  return useQuery({ queryKey: retentionKey, queryFn: getCollectionsRetention, enabled: window !== null })
}
