import { renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useIngestJobsStore } from '@/stores/ingestJobs'
import { useIngestJobStream, ingestStreamConfig } from './useIngestJobStream'

const sse = (event: string, data: unknown) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`

function streamOf(frames: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder()
  return new ReadableStream({
    start(controller) {
      for (const f of frames) controller.enqueue(encoder.encode(f))
      controller.close()
    }
  })
}

// Real timers throughout — this hook's retry loop is built on an
// async-generator SSE read loop, and fake timers do not compose cleanly with
// pending microtask chains like that. Instead we shrink the production
// delay via the exported config seam for just the reconnect-heavy tests
// below, and restore it afterwards so the production value (asserted
// nowhere in this file, but relied on by real users riding out a backend
// blip) is never left mutated for another test file.
const PRODUCTION_RECONNECT_DELAY_MS = ingestStreamConfig.reconnectDelayMs

beforeEach(() => {
  useIngestJobsStore.getState().clear()
  vi.unstubAllGlobals()
  ingestStreamConfig.reconnectDelayMs = 15
})

afterEach(() => {
  ingestStreamConfig.reconnectDelayMs = PRODUCTION_RECONNECT_DELAY_MS
})

describe('useIngestJobStream', () => {
  it('routes frames into the store by job_id', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(
        async () =>
          new Response(
            streamOf([
              sse('ingestion_started', { job_id: 'a', collection: 'mydocs' }),
              sse('ingestion_progress', { job_id: 'a', message: 'working' }),
              sse('ingestion_started', { job_id: 'b', collection: 'other' })
            ]),
            { status: 200, headers: { 'Content-Type': 'text/event-stream' } }
          )
      )
    )

    renderHook(() => useIngestJobStream())

    await waitFor(() => {
      expect(useIngestJobsStore.getState().events['a']).toHaveLength(2)
      expect(useIngestJobsStore.getState().events['b']).toHaveLength(1)
    })
  })

  it('is idempotent across a replayed history', async () => {
    // The backend replays a job's collapsed history on every (re)connect, so
    // the second connection re-delivers frames the store already folded. The
    // fold must converge, not duplicate.
    //
    // Two *same-kind* progress frames land within each connection's replay
    // (not just one) so this test actually exercises the collapse branch of
    // the fold, not just the ingestion_started reset — with only one
    // progress frame per connection, the collapse branch never has an
    // in-connection predecessor to fold against, and the test would pass
    // identically whether or not collapsing worked at all.
    const frames = [
      sse('ingestion_started', { job_id: 'a', collection: 'mydocs' }),
      sse('ingestion_progress', { job_id: 'a', message: 'Extracting entities: 3/9 chunks processed' }),
      sse('ingestion_progress', { job_id: 'a', message: 'Extracting entities: 4/9 chunks processed' })
    ]
    let connections = 0
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        connections += 1
        // First connection ends after the history (simulating a dropped
        // stream); the hook reconnects and receives the same replay again.
        return new Response(streamOf(frames), {
          status: 200,
          headers: { 'Content-Type': 'text/event-stream' }
        })
      })
    )

    renderHook(() => useIngestJobStream())

    await waitFor(() => expect(connections).toBeGreaterThanOrEqual(2), { timeout: 5000 })
    // Two deliveries of the same three frames must still fold to two
    // entries: one `ingestion_started`, one collapsed progress entry
    // carrying the newer (4/9) message — not four entries (broken reset)
    // and not three (broken collapse).
    expect(useIngestJobsStore.getState().events['a']).toHaveLength(2)
    expect(
      (useIngestJobsStore.getState().events['a'][1].data as { message: string }).message
    ).toContain('4/9')
  }, 10000)

  it('flags streamLost after exhausting the reconnect budget', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(null, { status: 500 })))

    renderHook(() => useIngestJobStream())

    await waitFor(() => expect(useIngestJobsStore.getState().streamLost).toBe(true), {
      timeout: 15000
    })
  }, 20000)
})
