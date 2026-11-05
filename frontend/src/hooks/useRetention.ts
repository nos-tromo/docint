import { useQuery } from '@tanstack/react-query'
import { getRetentionNotice } from '@/api/collections'
import { collectionsKey } from '@/hooks/useCollections'
import { useConfig } from '@/hooks/useConfig'

/** Under the collections key, so every invalidation of the listing refreshes the dates too. */
export const retentionKey = [...collectionsKey, 'retention'] as const

export interface RetentionPeriod {
  days: number
  noticeDays: number
}

/** The deployment's retention period, or `null` while retention is off (or unknown). */
export function useRetentionPeriod(): RetentionPeriod | null {
  const { data } = useConfig()
  const days = data?.document_retention_days
  if (typeof days !== 'number') return null
  return { days, noticeDays: data?.retention_notice_days ?? 30 }
}

/**
 * What falls due within the notice period. Asks nothing while retention is off,
 * so a deployment that never enabled it never sees a deadline anywhere.
 */
export function useRetentionNotice() {
  const period = useRetentionPeriod()
  return useQuery({ queryKey: retentionKey, queryFn: getRetentionNotice, enabled: period !== null })
}
