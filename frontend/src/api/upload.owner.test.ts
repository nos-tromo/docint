import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { streamUpload } from './upload'
import { setOwnerParam } from './client'

function emptyStream(): ReadableStream<Uint8Array> {
  return new ReadableStream({
    start(c) {
      c.close()
    }
  })
}

describe('streamUpload owner query param', () => {
  const fetchMock = vi.fn(
    async (..._args: unknown[]) => ({ ok: true, status: 200, body: emptyStream() }) as unknown as Response
  )

  beforeEach(() => {
    vi.stubGlobal('fetch', fetchMock)
    fetchMock.mockClear()
  })

  afterEach(() => {
    setOwnerParam(null)
    vi.unstubAllGlobals()
  })

  it('appends owner to the upload URL — otherwise an admin ingest into a foreign collection silently registers a new one under the admin', async () => {
    setOwnerParam('jane.doe')
    await streamUpload('/ingest/upload', new FormData()).next()
    expect(String(fetchMock.mock.calls[0][0])).toContain('owner=jane.doe')
  })

  it('adds nothing when no owner is set', async () => {
    await streamUpload('/ingest/upload', new FormData()).next()
    expect(String(fetchMock.mock.calls[0][0])).not.toContain('owner=')
  })
})
