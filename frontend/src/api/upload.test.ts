import { describe, it, expect, vi, afterEach } from 'vitest'
import { streamUpload } from './upload'

afterEach(() => vi.restoreAllMocks())

function bodyFromString(s: string): ReadableStream<Uint8Array> {
  const enc = new TextEncoder()
  return new ReadableStream({
    start(c) {
      c.enqueue(enc.encode(s))
      c.close()
    }
  })
}

describe('streamUpload', () => {
  it('posts FormData and yields parsed SSE frames', async () => {
    const frames =
      'event: start\ndata: {"collection":"c1"}\n\n' +
      'event: ingestion_complete\ndata: {"collection":"c1","data_dir":"/x"}\n\n'
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: true, body: bodyFromString(frames) })
    )

    const fd = new FormData()
    fd.append('collection', 'c1')

    const events: Array<{ event: string }> = []
    for await (const ev of streamUpload('/ingest/upload', fd)) {
      events.push({ event: ev.event })
    }

    expect(events).toEqual([{ event: 'start' }, { event: 'ingestion_complete' }])
    const call = (fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0]
    expect(call[1].method).toBe('POST')
    expect(call[1].body).toBe(fd)
  })
})
