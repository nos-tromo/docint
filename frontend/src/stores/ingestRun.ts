import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { mergeFiles } from '@infra/ui'
import { streamIngestUploadBatched, type BatchFailure } from '@/api/ingest'
import { createIngestJob, getStagedBatch } from '@/api/jobs'
import { applyIngestEvent, emptyStatus, type IngestStatus } from '@/lib/ingestStatus'
import type { IngestEvent } from '@/api/types'
import type { Strings } from '@/i18n'

type Translate = (key: keyof Strings, vars?: Record<string, string | number>) => string

/** The slice of the run that survives a reload. */
interface PersistedIngestRun {
  collection: string
  ner: boolean
  hate: boolean
  trackedJobs: TrackedJob[]
  handledJobIds: string[]
}

/** A job this browser queued, and the collection it was queued against. */
export interface TrackedJob {
  job_id: string
  collection: string
}

/**
 * How many handled-job ids to keep. Matches the backend's own
 * `MAX_TERMINAL_JOBS_PER_OWNER`, so an id is only ever forgotten once the
 * server has forgotten the job too.
 */
const MAX_HANDLED_JOBS = 50

/** The upload's own name for a file — the multipart filename it is sent as. */
const uploadName = (f: File): string => f.webkitRelativePath || f.name

/**
 * Drop the files the server already holds, byte for byte.
 *
 * This is what makes an interrupted upload finishable. A reloaded page cannot
 * re-read the user's files, so the only recovery is picking the folder again
 * — bearable only if the 18 GB that already arrived is not sent a second
 * time. Size is compared as well as name, so a file the user has since
 * replaced is still uploaded.
 *
 * @param files - The picked selection.
 * @param staged - What `GET /ingest/staged` reports for the collection.
 * @returns Only the files the server does not already have.
 */
export function filesNotYetStaged(files: File[], staged: { name: string; bytes: number }[]): File[] {
  // Keys are only ever compared whole, never split, and the size is always a
  // decimal number — so no two distinct (name, size) pairs can join to the
  // same string, whatever the name contains. The separator is printable on
  // purpose: this line held a raw NUL for four commits, which reads as a
  // space in every terminal and turned the whole file binary to git.
  const have = new Set(staged.map((e) => `${e.name}|${e.bytes}`))
  return files.filter((f) => !have.has(`${uploadName(f)}|${f.size}`))
}

/**
 * The ingest run controller: owns a run from picked files through to a queued
 * server-side job, and remembers every job this browser queued.
 *
 * Lives in a module singleton rather than the `Ingest` route component on
 * purpose. The upload leg is a sequence of fetches that outlives an unmounted
 * component, so a component-local reducer means navigating away mid-upload
 * keeps the transfer running while its progress vanishes into a dead reducer.
 * Here, leaving and returning shows the run exactly where it is.
 *
 * Persisted: the form (collection, enrichment toggles), `trackedJobs` and
 * `handledJobIds`, so a reload re-attaches to every run in flight. Not
 * persisted: `files` (`File` handles cannot survive a reload, and a reloaded
 * page could not act on them anyway) and the transient run state.
 */
export interface IngestRunState {
  collection: string
  ner: boolean
  hate: boolean
  /**
   * Jobs this browser queued, newest first. Persisted, and kept even after a
   * job finishes: it is the only way to tell a job the server has forgotten
   * (an interrupted run, worth offering a re-run for) from one that never
   * existed. The server's own list is the other source the Ingest screen
   * merges in — that one also carries jobs queued from another tab.
   */
  trackedJobs: TrackedJob[]
  /**
   * Ids whose terminal `ingestion_complete` already triggered the view's
   * post-ingest collection-select side effect. Persisted (like `trackedJobs`)
   * rather than kept in component state, so navigating away and back — or a
   * reload — doesn't repeat the effect for a job already handled in an
   * earlier mount. Bounded to {@link MAX_HANDLED_JOBS}, newest last.
   */
  handledJobIds: string[]
  /**
   * The upload leg's finished status, filed under the job that leg produced,
   * so each job's card can still report what it saved. Transient: a reload
   * loses it and the card falls back to the server's own progress.
   */
  uploadStatusByJob: Record<string, IngestStatus>
  files: File[]
  /**
   * The status of the upload currently in flight; moved to
   * `uploadStatusByJob` the moment that upload's job is queued.
   *
   * A folded status rather than the events that produced it: the upload leg
   * emits one frame per megabyte plus one per file, so keeping the array and
   * re-reducing it per frame was quadratic — tens of thousands of frames for
   * a folder-sized batch, which is enough to hang the tab.
   */
  uploadStatus: IngestStatus
  failedFiles: string[]
  /**
   * How many of the picked files the server already held, and so were not
   * sent again. Non-zero means this run finished an earlier, interrupted
   * upload rather than starting a fresh one.
   */
  alreadyStaged: number
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
   * Files the server already holds are not sent again, so re-picking a folder
   * finishes an interrupted upload instead of repeating it; when it holds all
   * of them, the job is queued without uploading anything.
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
   * Track `jobId` without touching the upload leg — used to re-queue an
   * interrupted run (the staged files are already on the server; `POST
   * /ingest/finalize` re-ingests over them directly) or to adopt a
   * 409-in-flight job id. Re-tracking a known id only refreshes its position.
   *
   * @param jobId - The server's job id.
   * @param collection - The collection it was queued against.
   * @param uploadStatus - The upload leg that produced it, if any.
   */
  trackJob: (jobId: string, collection: string, uploadStatus?: IngestStatus) => void
  /** Forget a job: drop it from the list along with its upload status. */
  untrackJob: (jobId: string) => void
  reset: () => void
}

const transient = {
  files: [] as File[],
  uploadStatus: emptyStatus(),
  failedFiles: [] as string[],
  alreadyStaged: 0,
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
      trackedJobs: [],
      handledJobIds: [],
      uploadStatusByJob: {},
      ...transient,
      setCollection: (collection) => set({ collection }),
      setNer: (ner) => set({ ner }),
      setHate: (hate) => set({ hate }),
      addFiles: (v) => set((s) => ({ files: mergeFiles(s.files, v) })),
      removeFile: (i) => set((s) => ({ files: s.files.filter((_, idx) => idx !== i) })),
      clearFiles: () => set({ files: [] }),
      markJobHandled: (jobId) =>
        set((s) =>
          s.handledJobIds.includes(jobId)
            ? s
            : { handledJobIds: [...s.handledJobIds, jobId].slice(-MAX_HANDLED_JOBS) }
        ),
      trackJob: (jobId, collection, uploadStatus) =>
        set((s) => ({
          trackedJobs: [
            { job_id: jobId, collection },
            ...s.trackedJobs.filter((j) => j.job_id !== jobId)
          ],
          uploadStatusByJob: uploadStatus
            ? { ...s.uploadStatusByJob, [jobId]: uploadStatus }
            : s.uploadStatusByJob
        })),
      untrackJob: (jobId) =>
        set((s) => {
          const uploadStatusByJob = { ...s.uploadStatusByJob }
          delete uploadStatusByJob[jobId]
          return {
            trackedJobs: s.trackedJobs.filter((j) => j.job_id !== jobId),
            uploadStatusByJob
          }
        }),
      reset: () =>
        set({
          collection: '',
          ner: false,
          hate: false,
          trackedJobs: [],
          handledJobIds: [],
          uploadStatusByJob: {},
          ...transient
        }),
      start: async (limitBytes, t) => {
        const { collection, files, ner, hate, uploading } = get()
        if (!collection || files.length === 0 || uploading) return
        set({
          uploading: true,
          error: null,
          warnings: [],
          uploadStatus: emptyStatus(),
          failedFiles: [],
          alreadyStaged: 0
        })

        // Ask what the server already holds and send only the rest. An
        // interrupted upload is finished this way rather than repeated: the
        // user re-picks the folder, and every file that arrived the first
        // time costs nothing. Fail-soft — if the question cannot be
        // answered, upload everything, which is what always happened before.
        // No initialiser, for the same reason as `anySaved` below: both
        // branches assign, so `no-useless-assignment` reports one if given.
        let pending: File[]
        try {
          const staged = await getStagedBatch(collection)
          pending = filesNotYetStaged(files, staged.entries)
        } catch {
          pending = files
        }
        const alreadyStaged = files.length - pending.length
        set({ alreadyStaged })

        if (pending.length === 0) {
          // Everything is already on the server — the exact state a reload
          // mid-upload leaves behind once the transfer finished but finalize
          // never ran. Queue the job over the staged batch directly.
          try {
            const { job_id } = await createIngestJob({ collection, ner, hate_speech: hate })
            get().trackJob(job_id, collection)
            set({ uploading: false, uploadStatus: emptyStatus(), files: [], failedFiles: [] })
          } catch {
            set({ uploading: false, error: t('ingest.failed_default') })
          }
          return
        }

        // Built once, up front: the per-file upload bar needs each picked
        // file's size, and rebuilding this map per frame would undo the point
        // of folding each frame in constant time.
        const fileSizes: Record<string, number> = {}
        for (const f of pending) fileSizes[uploadName(f)] = f.size

        // Both are assigned from the generator's return value below. The catch
        // returns, so nothing downstream can observe an initial value — and
        // `no-useless-assignment` (new in eslint:recommended under ESLint 10)
        // reports one if given.
        let anySaved: boolean
        let failures: BatchFailure[]
        let lastEvent: IngestEvent | null = null
        try {
          const stream = streamIngestUploadBatched(collection, pending, limitBytes, undefined, t)
          let next = await stream.next()
          while (!next.done) {
            const ev = next.value
            lastEvent = ev
            set((s) => ({ uploadStatus: applyIngestEvent(s.uploadStatus, ev, fileSizes) }))
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
          // The generator's own terminal `error` event (already folded into
          // `uploadStatus` above) already picked the more actionable message
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
          // Measured from the synthetic `start` event's own stamp — the exact
          // instant `deriveIngestStatus` anchors the card's timer to — so the
          // duration the server ends up logging and echoing back covers the
          // upload leg the user was already watching tick.
          const uploadStatus = get().uploadStatus
          const runStartedAt = uploadStatus.startedAt
          const { job_id } = await createIngestJob({
            collection,
            ner,
            hate_speech: hate,
            upload_elapsed_ms:
              runStartedAt === undefined ? undefined : Date.now() - runStartedAt
          })
          // The upload leg belongs to the job it produced from here on, so
          // `uploadStatus` is free to describe the *next* upload. Without the
          // handover a second run in the same tab would fold the previous
          // run's log into its own card.
          get().trackJob(job_id, collection, uploadStatus)
          set({
            uploading: false,
            uploadStatus: emptyStatus(),
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
      // is worth restoring, and `trackedJobs` is what makes reattach possible.
      partialize: (s) => ({
        collection: s.collection,
        ner: s.ner,
        hate: s.hate,
        trackedJobs: s.trackedJobs,
        handledJobIds: s.handledJobIds
      }),
      version: 2,
      // v1 tracked exactly one job in three scalars. Carry it over so a reload
      // across the upgrade still re-attaches to the run that was in flight.
      migrate: (persisted, version): PersistedIngestRun => {
        if (version >= 2) return persisted as PersistedIngestRun
        const old = (persisted ?? {}) as Record<string, unknown>
        const jobId = typeof old.activeJobId === 'string' ? old.activeJobId : null
        const collection = typeof old.collection === 'string' ? old.collection : ''
        const jobCollection =
          typeof old.activeJobCollection === 'string' ? old.activeJobCollection : collection
        const handled = typeof old.handledJobId === 'string' ? [old.handledJobId] : []
        return {
          collection,
          ner: old.ner === true,
          hate: old.hate === true,
          trackedJobs: jobId ? [{ job_id: jobId, collection: jobCollection }] : [],
          handledJobIds: handled
        }
      }
    }
  )
)
