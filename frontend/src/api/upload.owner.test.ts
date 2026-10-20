import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { streamUpload } from './upload'
import { setOwnerParam } from './client'
import { FakeXhr, installFakeXhr } from '@/test/fakeXhr'

/** Open an upload, complete it empty, and return the URL it was sent to. */
async function uploadUrl(owner?: string | null): Promise<string> {
  const next = streamUpload('/ingest/upload', new FormData(), undefined, owner).next()
  const xhr = FakeXhr.instances[0]
  xhr.finish()
  await next
  return xhr.url
}

describe('streamUpload owner query param', () => {
  beforeEach(() => {
    installFakeXhr()
  })

  afterEach(() => {
    setOwnerParam(null)
    vi.unstubAllGlobals()
  })

  it('appends owner to the upload URL — otherwise an admin ingest into a foreign collection silently registers a new one under the admin', async () => {
    setOwnerParam('jane.doe')
    expect(await uploadUrl()).toContain('owner=jane.doe')
  })

  it('adds nothing when no owner is set', async () => {
    expect(await uploadUrl()).not.toContain('owner=')
  })

  it('sends the owner it was given, not the one selected now', async () => {
    setOwnerParam('bob')
    const sent = await uploadUrl('alice')
    expect(sent).toContain('owner=alice')
    expect(sent).not.toContain('owner=bob')
  })
})
