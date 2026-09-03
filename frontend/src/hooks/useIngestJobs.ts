import { useQuery } from '@tanstack/react-query'
import { listIngestJobs } from '@/api/jobs'
import type { IngestJobSnapshot } from '@/api/types'

export const ingestJobsKey = ['ingest-jobs'] as const

/**
 * The caller's ingest jobs, newest first, as the server knows them.
 *
 * This is the discovery half of the Ingest screen: the SSE stream carries
 * progress but a job waiting on a worker slot emits no frames at all
 * (`docint/core/jobs.py`), and a job queued in another tab or before this
 * page loaded has no local record either. Only the list finds those.
 *
 * Summary and extract jobs share the registry and the stream but have their
 * own surfaces, so they are filtered out here. A snapshot from a backend
 * that predates `kind` is an ingest job.
 */
export function useIngestJobs() {
  return useQuery({
    queryKey: ingestJobsKey,
    queryFn: listIngestJobs,
    select: (data: { jobs: IngestJobSnapshot[] }) =>
      data.jobs.filter((job) => (job.kind ?? 'ingest') === 'ingest')
  })
}
