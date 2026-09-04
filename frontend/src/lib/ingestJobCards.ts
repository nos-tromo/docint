import type { IngestJobSnapshot } from '@/api/types'
import type { TrackedJob } from '@/stores/ingestRun'

/** One card on the Ingest screen: a job, and whatever is known about it. */
export interface IngestJobCardEntry {
  jobId: string
  collection: string
  /** The server's own snapshot, when the job was found in the jobs list. */
  listItem?: IngestJobSnapshot
}

/**
 * Merge the jobs this browser queued with the jobs the server lists.
 *
 * Neither source is complete on its own: the server list misses a job for the
 * moment between finalizing it and the list refetching, and misses one it has
 * forgotten entirely (an interrupted run — which is exactly the job worth
 * offering a re-run for); the tracked list misses jobs queued from another
 * tab or before this browser ever loaded the page.
 *
 * Ordering is newest-first: locally tracked jobs the server has not listed
 * yet are the freshest, so they lead, followed by the server's own order.
 *
 * @param tracked - Jobs this browser queued, newest first.
 * @param listed - The server's jobs, newest first.
 * @returns One entry per distinct job id.
 */
export function mergeJobCards(
  tracked: TrackedJob[],
  listed: IngestJobSnapshot[]
): IngestJobCardEntry[] {
  const byId = new Map(listed.map((job) => [job.job_id, job]))
  const untracked: IngestJobCardEntry[] = []
  for (const job of tracked) {
    if (byId.has(job.job_id)) continue
    untracked.push({ jobId: job.job_id, collection: job.collection })
  }
  const trackedNames = new Map(tracked.map((job) => [job.job_id, job.collection]))
  const fromServer = listed.map((job) => ({
    jobId: job.job_id,
    collection: job.collection || (trackedNames.get(job.job_id) ?? ''),
    listItem: job
  }))
  return [...untracked, ...fromServer]
}

/**
 * Whether a card's job has finished, and can therefore be dismissed.
 *
 * Two sources, because either can be ahead of the other: the live log's
 * terminal frame arrives before the list is refetched, and after a reload the
 * list carries a terminal status for a job whose frames aged out of the
 * replay.
 *
 * @param entry - The card.
 * @param terminal - Ids whose log reached a terminal frame.
 * @returns True when the job is complete or failed.
 */
export function isFinishedCard(
  entry: IngestJobCardEntry,
  terminal: Record<string, true>
): boolean {
  if (terminal[entry.jobId]) return true
  const status = entry.listItem?.status
  return status === 'completed' || status === 'failed'
}
