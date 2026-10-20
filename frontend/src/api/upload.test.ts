import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { streamUpload, UploadHttpError, UploadStalledError, UPLOAD_STALL_MS } from './upload'
import type { SseEvent } from './sse'
import { FakeXhr, installFakeXhr } from '@/test/fakeXhr'

beforeEach(() => {
  installFakeXhr()
})

afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

/** Drain an upload stream; starting it opens the request synchronously. */
async function drain(gen: AsyncGenerator<SseEvent, void, unknown>, out: SseEvent[] = []): Promise<SseEvent[]> {
  for await (const ev of gen) out.push(ev)
  return out
}

describe('streamUpload', () => {
  it('posts FormData and yields parsed SSE frames, even split across chunks', async () => {
    const fd = new FormData()
    fd.append('collection', 'c1')
    const events = drain(streamUpload('/ingest/upload', fd))
    const xhr = FakeXhr.instances[0]

    expect(xhr.method).toBe('POST')
    expect(xhr.body).toBe(fd)
    xhr.receive('event: start\ndata: {"collection":')
    xhr.receive('"c1"}\n\nevent: upload_complete\ndata: {"files_saved":1}\n\n')
    xhr.finish()

    expect(await events).toEqual([
      { event: 'start', data: { collection: 'c1' } },
      { event: 'upload_complete', data: { files_saved: 1 } }
    ])
  })

  it('throws a typed UploadHttpError carrying the status on a non-OK response', async () => {
    const events = drain(streamUpload('/ingest/upload', new FormData())).catch((err: unknown) => err)
    const xhr = FakeXhr.instances[0]
    xhr.receive('<html>413 Request Entity Too Large</html>', 413)
    xhr.finish()

    // 413 (nginx client_max_body_size) must be distinguishable from a generic
    // failure so callers can show a size-specific message.
    const caught = await events
    expect(caught).toBeInstanceOf(UploadHttpError)
    expect((caught as UploadHttpError).status).toBe(413)
  })

  it('yields the frames that arrived before a network failure, then throws', async () => {
    const seen: SseEvent[] = []
    const events = drain(streamUpload('/ingest/upload', new FormData()), seen).catch((err: unknown) => err)
    const xhr = FakeXhr.instances[0]
    xhr.receive('event: file_saved\ndata: {"filename":"a"}\n\n')
    xhr.fail()

    expect(await events).toBeInstanceOf(TypeError)
    expect(seen.map((e) => e.event)).toEqual(['file_saved'])
  })

  it('aborts with UploadStalledError once nothing is sent or received for UPLOAD_STALL_MS', async () => {
    vi.useFakeTimers()
    const events = drain(streamUpload('/ingest/upload', new FormData())).catch((err: unknown) => err)
    const xhr = FakeXhr.instances[0]

    // Body bytes leaving and frames arriving both count as progress.
    await vi.advanceTimersByTimeAsync(UPLOAD_STALL_MS - 1)
    xhr.sent()
    await vi.advanceTimersByTimeAsync(UPLOAD_STALL_MS - 1)
    xhr.receive('event: start\ndata: {}\n\n')
    await vi.advanceTimersByTimeAsync(UPLOAD_STALL_MS - 1)
    expect(xhr.aborted).toBe(false)

    await vi.advanceTimersByTimeAsync(1)
    expect(xhr.aborted).toBe(true)
    expect(await events).toBeInstanceOf(UploadStalledError)
  })

  it('aborts the request when the signal aborts', async () => {
    const controller = new AbortController()
    const events = drain(streamUpload('/ingest/upload', new FormData(), controller.signal)).catch(
      (err: unknown) => err
    )
    controller.abort()

    expect(FakeXhr.instances[0].aborted).toBe(true)
    expect((await events as Error).name).toBe('AbortError')
  })
})
