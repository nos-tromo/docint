import { useEffect } from 'react'
import { streamSseGet } from '@/api/sse'
import { INGEST_JOB_EVENTS_PATH } from '@/api/jobs'
import { useIngestJobsStore } from '@/stores/ingestJobs'
import type { IngestEvent } from '@/api/types'

// Deliberately short: each *consecutive* failed attempt still carries its own
// real connection latency (DNS/TCP/TLS, or the browser's own fetch timeout on
// a truly dead network), so this delay is a minor addend to the total
// giveup time, not the dominant cost. Keeping it short also keeps the test
// suite's wall-clock cost for the reconnect-budget tests bounded.
const RECONNECT_DELAY_MS = 300
/** Consecutive reconnect attempts before the stream is declared lost. */
export const MAX_RECONNECTS = 5

/**
 * Subscribe to the owner-multiplexed ingest job stream and fan its frames into
 * {@link useIngestJobsStore}, keyed by `job_id`.
 *
 * Mounted once for the whole session (in `Shell`), this opens exactly one
 * connection carrying every job the caller owns. The backend replays each
 * job's collapsed history on connect, and the store's fold is idempotent over
 * a replay, so reconnecting is always safe.
 *
 * Reconnects up to {@link MAX_RECONNECTS} consecutive times; any received
 * event resets the budget. On exhaustion the store's `streamLost` flag lets
 * the UI offer a manual reconnect rather than freezing on the last frame.
 */
export function useIngestJobStream(): void {
  useEffect(() => {
    const controller = new AbortController()
    let cancelled = false
    let reconnects = 0
    // Non-reactive access: this hook is a producer and must not re-render on
    // its own writes. The actions are stable.
    const { appendEvent, setStreamLost } = useIngestJobsStore.getState()

    async function run(): Promise<void> {
      while (!cancelled && reconnects <= MAX_RECONNECTS) {
        try {
          for await (const frame of streamSseGet(INGEST_JOB_EVENTS_PATH, controller.signal)) {
            reconnects = 0 // a real event resets the consecutive-failure budget
            const data = (frame.data ?? {}) as Record<string, unknown>
            const jobId = typeof data.job_id === 'string' ? data.job_id : null
            if (!jobId) continue
            setStreamLost(false)
            appendEvent(jobId, {
              event: frame.event as IngestEvent['event'],
              data,
              receivedAt: Date.now()
            })
          }
        } catch {
          if (cancelled || controller.signal.aborted) return
        }
        if (cancelled) return
        reconnects += 1
        if (reconnects > MAX_RECONNECTS) {
          setStreamLost(true)
          return
        }
        await new Promise((resolve) => setTimeout(resolve, RECONNECT_DELAY_MS))
      }
    }

    void run()
    return () => {
      cancelled = true
      controller.abort()
    }
  }, [])
}
