import { useQuery } from '@tanstack/react-query'
import { getStagedBatch, type PreprocessStage } from '@/api/jobs'
import type { IngestPhase } from '@/lib/ingestStatus'

/** How often to re-ask while a run is live. */
export const PREPROCESS_POLL_INTERVAL_MS = 5_000

/** The phases during which the server may still be preprocessing this batch. */
const LIVE_PHASES: ReadonlySet<IngestPhase> = new Set<IngestPhase>([
  'uploading',
  'queued',
  'processing'
])

/**
 * The query key every view of one collection's preprocessing shares.
 *
 * One key, so the upload card, the job card and the staged card are served by
 * a single poll — and so the hand-over at finalize renders from cache rather
 * than blanking while a fresh request goes out.
 *
 * @param collection - The logical collection name.
 * @returns The react-query key.
 */
export const preprocessQueryKey = (collection: string) =>
  ['ingest-staged', collection, { entries: false }] as const

/**
 * Poll what the preprocessing pool has done for a collection.
 *
 * The heavy per-file stages start as each file is saved, long before a job
 * exists to report them: a PDF's pages, an image's caption, a clip's
 * transcript and keyframes. There is no job id to hang an SSE frame on and no
 * client attached to the upload after a reload, so this is pulled rather than
 * pushed — one small request every few seconds, against work measured in
 * minutes per file.
 *
 * @param collection - The logical collection, or empty while none is chosen.
 * @param phase - The run's phase; polling stops once it is no longer live.
 * @returns The stages in render order, or `undefined` before the first answer.
 */
export function usePreprocessProgress(
  collection: string | undefined,
  phase: IngestPhase
): PreprocessStage[] | undefined {
  const name = collection?.trim() ?? ''
  const live = name.length > 0 && LIVE_PHASES.has(phase)

  const { data } = useQuery({
    queryKey: preprocessQueryKey(name),
    queryFn: () => getStagedBatch(name, { includeEntries: false }),
    enabled: live,
    retry: false,
    // Never a drain check: the pool empties between two files landing, and
    // the job's own prefetch fills it again after finalize. The phase is the
    // only honest end of this work.
    refetchInterval: live ? PREPROCESS_POLL_INTERVAL_MS : false,
    select: (staged) => staged?.preprocess?.stages ?? []
  })

  return data
}
