import { renderHook, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { useIngestJobsStore } from '@/stores/ingestJobs'
import { useIngestJobStream } from './useIngestJobStream'

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

beforeEach(() => {
  useIngestJobsStore.getState().clear()
  vi.unstubAllGlobals()
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
    // fold must converge, not duplicate: the collapse rule means a repeated
    // progress frame of the same kind replaces rather than appends.
    const frames = [
      sse('ingestion_started', { job_id: 'a', collection: 'mydocs' }),
      sse('ingestion_progress', { job_id: 'a', message: 'Extracting entities: 3/9 chunks processed' })
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
    // Two deliveries of the same two frames must still fold to two entries:
    // one `ingestion_started`, one collapsed progress entry.
    expect(useIngestJobsStore.getState().events['a']).toHaveLength(2)
    expect(
      (useIngestJobsStore.getState().events['a'][1].data as { message: string }).message
    ).toContain('3/9')
  }, 10000)

  it('flags streamLost after exhausting the reconnect budget', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(null, { status: 500 })))

    renderHook(() => useIngestJobStream())

    await waitFor(() => expect(useIngestJobsStore.getState().streamLost).toBe(true), {
      timeout: 15000
    })
  }, 20000)
})
