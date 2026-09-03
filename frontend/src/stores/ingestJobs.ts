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
  /**
   * Ids whose log has reached a terminal frame. Derived from `events`, but
   * kept as its own map so a consumer that only cares *whether* a job is
   * finished — the Ingest screen's "Clear finished" — can subscribe to
   * something that changes once per run rather than once per progress frame.
   */
  terminal: Record<string, true>
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

/**
 * SSE event names that open a run, across every job kind this store can hold:
 * an ingest job starts with `ingestion_started`, a summary job with
 * `summary_started`, an extract job with `extract_started`. The backend
 * replays a job's collapsed history from its started frame on every
 * reconnect, so all of them must reset the local log.
 */
const STARTED_EVENTS: ReadonlySet<IngestEvent['event']> = new Set([
  'ingestion_started',
  'summary_started',
  'extract_started'
])

/**
 * Fold one frame into the terminal-job map, returning the *same* map when
 * nothing changed.
 *
 * @param terminal - The current map.
 * @param jobId - The job the frame belongs to.
 * @param ev - The frame.
 * @returns The next map, or `terminal` itself when unchanged.
 */
function nextTerminal(
  terminal: Record<string, true>,
  jobId: string,
  ev: IngestEvent
): Record<string, true> {
  if (TERMINAL_EVENTS.has(ev.event)) {
    return terminal[jobId] ? terminal : { ...terminal, [jobId]: true }
  }
  if (STARTED_EVENTS.has(ev.event) && terminal[jobId]) {
    const next = { ...terminal }
    delete next[jobId]
    return next
  }
  return terminal
}

export const useIngestJobsStore = create<IngestJobsState>((set) => ({
  events: {},
  terminal: {},
  streamLost: false,
  retryNonce: 0,
  appendEvent: (jobId, ev) =>
    set((s) => ({
      // Only rebuilt when the flag actually changes, so a run of progress
      // frames leaves the reference — and every subscriber — untouched.
      terminal: nextTerminal(s.terminal, jobId, ev),
      events: {
        ...s.events,
        // A job's *started* frame is the first frame of every replay, so it
        // marks the start of a fresh fold: reset rather than append. Without
        // this, each reconnect would re-append the job's warnings and terminal
        // frame (only *progress* frames collapse), so a flaky connection
        // would visibly duplicate warnings in the event log. Both job kinds
        // multiplex through this store, so a summary job's `summary_started`
        // has to reset too — keying on `ingestion_started` alone left a
        // mid-build summary reconnect re-appending its replayed history.
        // Within a single connection a job emits its started frame exactly
        // once, so this costs nothing in the normal case.
        [jobId]: STARTED_EVENTS.has(ev.event) ? [ev] : appendCollapsedEvent(s.events[jobId] ?? [], ev)
      }
    })),
  setStreamLost: (streamLost) => set({ streamLost }),
  dropJob: (jobId) =>
    set((s) => {
      if (!(jobId in s.events) && !(jobId in s.terminal)) return s // stable reference -> no needless re-render
      const events = { ...s.events }
      delete events[jobId]
      const terminal = { ...s.terminal }
      delete terminal[jobId]
      return { events, terminal }
    }),
  retryStream: () => set((s) => ({ streamLost: false, retryNonce: s.retryNonce + 1 })),
  clear: () => set({ events: {}, terminal: {}, streamLost: false, retryNonce: 0 })
}))

/**
 * SSE event names, across every job kind this store can hold, that end a
 * run. Mirrors the backend's `jobs.py::TERMINAL_EVENTS` — an ingest job
 * terminates on `ingestion_complete`, a summary job on `summary_completed`,
 * an extract job on `extract_completed`, any kind on `error`. The stream is multiplexed across kinds with no
 * kind filter (`useIngestJobStream.ts`), so both must be recognized here or
 * a completed summary job would look permanently "running" to the selector
 * below and leave the sidebar badge stuck on.
 */
const TERMINAL_EVENTS: ReadonlySet<IngestEvent['event']> = new Set([
  'ingestion_complete',
  'summary_completed',
  'extract_completed',
  'error'
])

/**
 * Whether any tracked job is still running — i.e. it has started and has not
 * yet produced a terminal frame. Drives the sidebar badge.
 */
export const selectHasRunningJob = (s: IngestJobsState): boolean =>
  Object.values(s.events).some(
    (events) => events.length > 0 && !events.some((e) => TERMINAL_EVENTS.has(e.event))
  )

/** Stable selector for one job's log; returns a frozen empty array when absent. */
const EMPTY: IngestEvent[] = []
export const selectJobEvents =
  (jobId: string | null) =>
  (s: IngestJobsState): IngestEvent[] =>
    jobId ? (s.events[jobId] ?? EMPTY) : EMPTY
