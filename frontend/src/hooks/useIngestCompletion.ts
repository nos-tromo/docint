import { useEffect } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { selectCollection } from '@/api/collections'
import { collectionsKey } from '@/hooks/useCollections'
import { ingestJobsKey } from '@/hooks/useIngestJobs'
import { useIngestRunStore } from '@/stores/ingestRun'
import { useUiStore } from '@/stores/ui'
import type { IngestEvent } from '@/api/types'

/**
 * Run the post-ingest side effect once per job: select the collection it
 * filled and refresh the owned-collections list.
 *
 * Guarded by the run store's `handledJobIds` — a *persisted* list, not a
 * component ref — so this fires once per job id no matter how many times the
 * terminal frame is observed: within a mount (a reconnect replay re-delivers
 * it in a new event-log array) and across mounts (navigating away and back,
 * or a reload, while the job's log lives on in the module-level job store).
 *
 * With several jobs finishing in a row each one selects its own collection,
 * so the last to finish is the one left selected.
 *
 * @param jobId - The job this card renders.
 * @param events - That job's event log.
 * @param collection - Fallback collection name when the terminal frame
 *   carries none.
 */
export function useIngestCompletion(
  jobId: string,
  events: IngestEvent[],
  collection: string
): void {
  const qc = useQueryClient()
  const setSelected = useUiStore((s) => s.setSelectedCollection)
  useEffect(() => {
    const last = events[events.length - 1]
    if (!last || last.event !== 'ingestion_complete') return
    // Read non-reactively: the guard is only consulted at fire time, and
    // depending on it would re-run this effect on every job's completion.
    const run = useIngestRunStore.getState()
    if (run.handledJobIds.includes(jobId)) return
    const data = last.data as { collection?: unknown }
    const name = typeof data.collection === 'string' ? data.collection : collection
    // Mark handled synchronously, before the async work below, so a
    // re-render triggered by that very write (or any other update while the
    // work is in flight) can't slip past the guard a second time.
    run.markJobHandled(jobId)
    // The job's own snapshot is now stale — it still reads as running.
    void qc.invalidateQueries({ queryKey: ingestJobsKey })
    if (!name) return
    void (async () => {
      await selectCollection(name)
      // Refresh the owned-collections list BEFORE selecting: the Sidebar's
      // reconcile effect clears any active collection not present in that
      // cached list, so selecting a brand-new collection while the list is
      // stale would immediately snap the selection back to null. Awaiting the
      // refetch first ensures the new name is in the list before we select it.
      await qc.invalidateQueries({ queryKey: collectionsKey })
      setSelected(name)
    })()
  }, [events, jobId, collection, qc, setSelected])
}
