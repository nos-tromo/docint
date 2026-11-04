import { afterEach, describe, expect, it, vi } from 'vitest'
import { setOwnerParam } from './client'
import { fetchFile } from './files'
import { sourcePreviewPath, sourceThumbnailPath } from './ingest'

describe('source preview owner query param', () => {
  afterEach(() => {
    setOwnerParam(null)
    vi.unstubAllGlobals()
  })

  async function fetchedUrl(path: string): Promise<string> {
    const fetchMock = vi.fn(async (..._args: unknown[]) => new Response('x', { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)
    await fetchFile(path)
    return String(fetchMock.mock.calls[0][0])
  }

  it('reaches a foreign owner namespace for an admin, the owner added once', async () => {
    setOwnerParam('jane.doe')
    expect(sourcePreviewPath('alpha', 'hash1')).not.toContain('owner=')
    expect(await fetchedUrl(sourcePreviewPath('alpha', 'hash1'))).toBe(
      '/sources/preview?collection=alpha&file_hash=hash1&owner=jane.doe'
    )
    expect(await fetchedUrl(sourceThumbnailPath('alpha', 'img1'))).toBe(
      '/sources/thumbnail?collection=alpha&image_id=img1&owner=jane.doe'
    )
  })

  it('adds nothing when no owner is set', async () => {
    expect(await fetchedUrl(sourcePreviewPath('alpha', 'hash1'))).not.toContain('owner=')
  })
})
