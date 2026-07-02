import { describe, it, expect, vi, afterEach } from 'vitest'
import { streamQuery } from './chat'

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

function lastCall() {
  return (fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0]
}

describe('streamQuery', () => {
  it('POSTs /stream_query with the selected collection in the body', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        body: bodyFromString('data: {"response":"ok","sources":[],"session_id":"s"}\n\n')
      })
    )

    // The generator is lazy — consume it so the underlying fetch fires.
    for await (const _ of streamQuery({ question: 'hi', collection: 'docs' })) {
      void _
    }

    const call = lastCall()
    expect(String(call[0])).toContain('/stream_query')
    expect(call[1].method).toBe('POST')
    expect(JSON.parse(call[1].body)).toMatchObject({ question: 'hi', collection: 'docs' })
  })

  it('omits collection from the body when none is selected', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        body: bodyFromString('data: {"response":"ok","sources":[],"session_id":"s"}\n\n')
      })
    )

    for await (const _ of streamQuery({ question: 'hi' })) {
      void _
    }

    expect(JSON.parse(lastCall()[1].body)).not.toHaveProperty('collection')
  })
})
