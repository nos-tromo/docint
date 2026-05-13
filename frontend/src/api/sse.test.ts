import { describe, it, expect, vi, afterEach } from 'vitest'
import { streamSse } from './sse'

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

describe('streamSse', () => {
  it('parses event/data frames into objects', async () => {
    const frames =
      'event: token\n' +
      'data: {"token":"hello"}\n\n' +
      'event: token\n' +
      'data: {"token":" world"}\n\n' +
      'event: done\n' +
      'data: {"answer":"hello world","sources":[]}\n\n'
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: true, body: bodyFromString(frames) })
    )

    const events: Array<{ event: string; data: unknown }> = []
    for await (const ev of streamSse('/x', { foo: 'bar' })) events.push(ev)

    expect(events).toEqual([
      { event: 'token', data: { token: 'hello' } },
      { event: 'token', data: { token: ' world' } },
      { event: 'done', data: { answer: 'hello world', sources: [] } }
    ])
  })

  it('handles frames split across chunks', async () => {
    const enc = new TextEncoder()
    const stream = new ReadableStream({
      start(c) {
        c.enqueue(enc.encode('event: token\nda'))
        c.enqueue(enc.encode('ta: {"token":"hi"}\n\n'))
        c.close()
      }
    })
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, body: stream }))

    const events: Array<{ event: string; data: unknown }> = []
    for await (const ev of streamSse('/x')) events.push(ev)
    expect(events).toEqual([{ event: 'token', data: { token: 'hi' } }])
  })
})
