import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { streamSse } from './sse'
import { setOwnerParam } from './client'

function emptyStream(): ReadableStream<Uint8Array> {
  return new ReadableStream({
    start(c) {
      c.close()
    }
  })
}

describe('streamSse owner query param', () => {
  const fetchMock = vi.fn(async (..._args: unknown[]) => ({ ok: true, body: emptyStream() }) as unknown as Response)

  beforeEach(() => {
    vi.stubGlobal('fetch', fetchMock)
    fetchMock.mockClear()
  })

  afterEach(() => {
    setOwnerParam(null)
    vi.unstubAllGlobals()
  })

  it('appends owner to the request URL when an admin has a foreign collection selected', async () => {
    setOwnerParam('jane.doe')
    await streamSse('/stream_query', { question: 'hi' }).next()
    expect(String(fetchMock.mock.calls[0][0])).toContain('owner=jane.doe')
  })

  it('adds nothing when no owner is set', async () => {
    await streamSse('/stream_query', { question: 'hi' }).next()
    expect(String(fetchMock.mock.calls[0][0])).not.toContain('owner=')
  })
})
