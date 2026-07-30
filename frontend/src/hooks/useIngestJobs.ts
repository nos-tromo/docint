import { useQuery } from '@tanstack/react-query'
import { listIngestJobs } from '@/api/jobs'

export const ingestJobsKey = ['ingest-jobs'] as const

/**
 * The caller's server-side ingest jobs.
 *
 * Fetched once on load so a persisted `activeJobId` can be validated: a job the
 * server no longer knows (backend restarted) renders as interrupted rather
 * than as a run that never reports progress.
 */
export function useIngestJobs() {
  return useQuery({ queryKey: ingestJobsKey, queryFn: listIngestJobs, staleTime: 30_000 })
}
