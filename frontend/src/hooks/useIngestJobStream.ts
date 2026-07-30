import { useEffect } from 'react'
import { streamSseGet } from '@/api/sse'
import { INGEST_JOB_EVENTS_PATH } from '@/api/jobs'
import { useIngestJobsStore } from '@/stores/ingestJobs'
import type { IngestEvent } from '@/api/types'

/**
 * Reconnect timing, exposed as a mutable config object rather than plain
 * constants so tests can shrink `reconnectDelayMs` for the reconnect-budget
 * tests without touching the production value. `reconnectDelayMs` (1500ms)
 * times `maxReconnects` (5) is the give-up window a real user rides out
 * silently — e.g. a rolling backend deploy or container restart — before
 * `streamLost` flips and they need to reconnect by hand; narrowing it would
 * be a real change in production resilience, not just a test-speed tweak.
 * Fake timers were considered for the tests instead, but this hook's retry
 * loop is built on an async-generator SSE read loop, and fake timers do not
 * compose cleanly with pending microtask chains like that — this seam avoids
 * the risk entirely while keeping the tests on real timers.
 */
export const ingestStreamConfig = {
  reconnectDelayMs: 1500,
  /** Consecutive reconnect attempts before the stream is declared lost. */
  maxReconnects: 5
}

/**
 * Subscribe to the owner-multiplexed ingest job stream and fan its frames into
 * {@link useIngestJobsStore}, keyed by `job_id`.
 *
 * Mounted once for the whole session (in `Shell`), this opens exactly one
 * connection carrying every job the caller owns. The backend replays each
 * job's collapsed history on connect, and the store's fold is idempotent over
 * a replay, so reconnecting is always safe.
 *
 * Reconnects up to {@link ingestStreamConfig}'s `maxReconnects` consecutive
 * times; any received event resets the budget. On exhaustion the store's
 * `streamLost` flag lets the UI offer a manual reconnect rather than
 * freezing on the last frame.
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
      while (!cancelled && reconnects <= ingestStreamConfig.maxReconnects) {
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
        if (reconnects > ingestStreamConfig.maxReconnects) {
          setStreamLost(true)
          return
        }
        await new Promise((resolve) => setTimeout(resolve, ingestStreamConfig.reconnectDelayMs))
      }
    }

    void run()
    return () => {
      cancelled = true
      controller.abort()
    }
  }, [])
}
