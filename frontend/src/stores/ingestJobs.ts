import { create } from 'zustand'
import { appendCollapsedEvent } from '@/lib/ingestStatus'
import type { IngestEvent } from '@/api/types'

/**
 * Live per-job ingest event logs, keyed by `job_id`.
 *
 * Deliberately **not** persisted: the backend replays each job's collapsed
 * history when the stream reconnects, so persisting would only risk showing a
 * stale run that the server has since finished or forgotten.
 */
export interface IngestJobsState {
  events: Record<string, IngestEvent[]>
  /** True once the SSE stream exhausted its reconnect budget. */
  streamLost: boolean
  /** Bumped to force `useIngestJobStream` to re-subscribe after giving up. */
  retryNonce: number
  appendEvent: (jobId: string, ev: IngestEvent) => void
  setStreamLost: (v: boolean) => void
  dropJob: (jobId: string) => void
  /** Clear the lost-stream flag and bump `retryNonce` so the stream hook's
   *  effect re-runs and opens a fresh connection. */
  retryStream: () => void
  clear: () => void
}

export const useIngestJobsStore = create<IngestJobsState>((set) => ({
  events: {},
  streamLost: false,
  retryNonce: 0,
  appendEvent: (jobId, ev) =>
    set((s) => ({
      events: {
        ...s.events,
        // `ingestion_started` is the first frame of every replay, so it marks
        // the start of a fresh fold: reset rather than append. Without this,
        // each reconnect would re-append the job's warnings and terminal
        // frame (only *progress* frames collapse), so a flaky connection
        // would visibly duplicate warnings in the event log. Within a single
        // connection a job emits `ingestion_started` exactly once, so this
        // costs nothing in the normal case.
        [jobId]:
          ev.event === 'ingestion_started' ? [ev] : appendCollapsedEvent(s.events[jobId] ?? [], ev)
      }
    })),
  setStreamLost: (streamLost) => set({ streamLost }),
  dropJob: (jobId) =>
    set((s) => {
      if (!(jobId in s.events)) return s // stable reference -> no needless re-render
      const events = { ...s.events }
      delete events[jobId]
      return { events }
    }),
  retryStream: () => set((s) => ({ streamLost: false, retryNonce: s.retryNonce + 1 })),
  clear: () => set({ events: {}, streamLost: false, retryNonce: 0 })
}))

/**
 * Whether any tracked job is still running — i.e. it has started and has not
 * yet produced a terminal frame. Drives the sidebar badge.
 */
export const selectHasRunningJob = (s: IngestJobsState): boolean =>
  Object.values(s.events).some(
    (events) =>
      events.length > 0 &&
      !events.some((e) => e.event === 'ingestion_complete' || e.event === 'error')
  )

/** Stable selector for one job's log; returns a frozen empty array when absent. */
const EMPTY: IngestEvent[] = []
export const selectJobEvents =
  (jobId: string | null) =>
  (s: IngestJobsState): IngestEvent[] =>
    jobId ? (s.events[jobId] ?? EMPTY) : EMPTY
