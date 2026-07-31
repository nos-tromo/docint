import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { mergeFiles } from '@infra/ui'
import { streamIngestUploadBatched, type BatchFailure } from '@/api/ingest'
import { createIngestJob } from '@/api/jobs'
import type { IngestEvent } from '@/api/types'
import type { Strings } from '@/i18n'

type Translate = (key: keyof Strings, vars?: Record<string, string | number>) => string

/**
 * The ingest run controller: owns a run from picked files through to a queued
 * server-side job.
 *
 * Lives in a module singleton rather than the `Ingest` route component on
 * purpose. The upload leg is a sequence of fetches that outlives an unmounted
 * component, so a component-local reducer means navigating away mid-upload
 * keeps the transfer running while its progress vanishes into a dead reducer.
 * Here, leaving and returning shows the run exactly where it is.
 *
 * Persisted: the form (collection, enrichment toggles) and `activeJobId`, so a
 * reload re-attaches. Not persisted: `files` (`File` handles cannot survive a
 * reload, and a reloaded page could not act on them anyway) and the transient
 * run state.
 */
export interface IngestRunState {
  collection: string
  ner: boolean
  hate: boolean
  activeJobId: string | null
  /**
   * The collection `activeJobId` was queued against, captured at queue time.
   * Read by the interrupted-run "Run again" flow instead of the live
   * `collection` field, which the user can edit between the interruption and
   * the click — otherwise a re-run can finalize a different collection than
   * the one that was actually interrupted.
   */
  activeJobCollection: string | null
  /**
   * Id of the job whose terminal `ingestion_complete` has already triggered
   * the view's post-ingest collection-select side effect. Persisted (like
   * `activeJobId`) rather than kept in component state, so navigating away
   * and back — or a reload — doesn't repeat the effect for a job that was
   * already handled in an earlier mount.
   */
  handledJobId: string | null
  files: File[]
  uploadEvents: IngestEvent[]
  failedFiles: string[]
  warnings: string[]
  uploading: boolean
  error: string | null
  setCollection: (v: string) => void
  setNer: (v: boolean) => void
  setHate: (v: boolean) => void
  addFiles: (v: File[]) => void
  removeFile: (i: number) => void
  clearFiles: () => void
  /**
   * Upload the picked files, then queue a server-side ingest job over the
   * collection they were staged into.
   *
   * No-ops (without touching `error`) when there is no collection, no files,
   * or a run is already in flight — re-entrant callers (e.g. a double click)
   * are safe.
   *
   * @param limitBytes - The server's per-request upload ceiling, forwarded to
   *   `streamIngestUploadBatched`.
   * @param t - Translate function for user-facing error copy.
   */
  start: (limitBytes: number, t: Translate) => Promise<void>
  /**
   * Mark `jobId` as having already triggered the post-ingest side effect.
   * Called once, synchronously, the moment the view observes that job's
   * terminal `ingestion_complete` frame.
   */
  markJobHandled: (jobId: string) => void
  /**
   * Adopt `jobId` as the active run without touching the upload leg — used
   * to re-queue an interrupted run (the staged files are already on the
   * server; `POST /ingest/finalize` re-ingests over them directly) or to
   * adopt a 409-in-flight job id.
   */
  adoptJob: (jobId: string) => void
  dismissActive: () => void
  reset: () => void
}

const transient = {
  files: [] as File[],
  uploadEvents: [] as IngestEvent[],
  failedFiles: [] as string[],
  warnings: [] as string[],
  uploading: false,
  error: null as string | null
}

export const useIngestRunStore = create<IngestRunState>()(
  persist(
    (set, get) => ({
      collection: '',
      ner: false,
      hate: false,
      activeJobId: null,
      activeJobCollection: null,
      handledJobId: null,
      ...transient,
      setCollection: (collection) => set({ collection }),
      setNer: (ner) => set({ ner }),
      setHate: (hate) => set({ hate }),
      addFiles: (v) => set((s) => ({ files: mergeFiles(s.files, v) })),
      removeFile: (i) => set((s) => ({ files: s.files.filter((_, idx) => idx !== i) })),
      clearFiles: () => set({ files: [] }),
      markJobHandled: (jobId) => set({ handledJobId: jobId }),
      adoptJob: (jobId) =>
        set({ activeJobId: jobId, handledJobId: null, uploadEvents: [], failedFiles: [], error: null }),
      dismissActive: () =>
        set({
          activeJobId: null,
          activeJobCollection: null,
          handledJobId: null,
          uploadEvents: [],
          failedFiles: []
        }),
      reset: () =>
        set({
          collection: '',
          ner: false,
          hate: false,
          activeJobId: null,
          activeJobCollection: null,
          handledJobId: null,
          ...transient
        }),
      start: async (limitBytes, t) => {
        const { collection, files, ner, hate, uploading } = get()
        if (!collection || files.length === 0 || uploading) return
        set({ uploading: true, error: null, warnings: [], uploadEvents: [], failedFiles: [] })

        let anySaved = false
        let failures: BatchFailure[] = []
        let lastEvent: IngestEvent | null = null
        try {
          const stream = streamIngestUploadBatched(collection, files, limitBytes, undefined, t)
          let next = await stream.next()
          while (!next.done) {
            const ev = next.value
            lastEvent = ev
            set((s) => ({ uploadEvents: [...s.uploadEvents, ev] }))
            if (ev.event === 'warning') {
              const message = (ev.data as { message?: unknown }).message
              if (typeof message === 'string') set((s) => ({ warnings: [...s.warnings, message] }))
            }
            next = await stream.next()
          }
          anySaved = next.value.anySaved
          failures = next.value.failures
        } catch {
          // Files stay in state so the user can retry `start()` without
          // re-picking — see the store's persistence/retry contract above.
          set({ uploading: false, error: t('ingest.failed_default') })
          return
        }

        if (!anySaved) {
          // The generator's own terminal `error` event (already appended to
          // `uploadEvents` above) already picked the more actionable message
          // — e.g. distinguishing "every file is over the size limit" from a
          // generic rejection. Reuse it instead of recomputing a duplicate,
          // less-specific message here, which is how the two drifted apart.
          const terminalMessage =
            lastEvent?.event === 'error' && typeof lastEvent.data.message === 'string'
              ? lastEvent.data.message
              : t('ingest.upload_failed_rejected', { count: failures.length })
          set({ uploading: false, error: terminalMessage })
          return
        }

        try {
          const { job_id } = await createIngestJob({
            collection,
            hybrid: true,
            ner,
            hate_speech: hate
          })
          set({
            activeJobId: job_id,
            activeJobCollection: collection,
            uploading: false,
            files: [],
            failedFiles: failures.flatMap((f) => f.files)
          })
        } catch {
          // The batches already staged server-side; only the finalize call
          // failed. Files stay in state (they are not cleared until the job
          // is actually queued below) so the user can hit start() again.
          set({ uploading: false, error: t('ingest.failed_default') })
        }
      }
    }),
    {
      name: 'docint-ingest-run',
      // `File` handles cannot be serialized, and a reloaded page could not act
      // on them anyway — the user must re-pick. Everything else about the form
      // is worth restoring, and `activeJobId` is what makes reattach possible.
      partialize: (s) => ({
        collection: s.collection,
        ner: s.ner,
        hate: s.hate,
        activeJobId: s.activeJobId,
        activeJobCollection: s.activeJobCollection,
        handledJobId: s.handledJobId
      }),
      version: 1
    }
  )
)
