import { afterEach, describe, expect, it, vi } from 'vitest'

// A source preview is fetched through apiFetch, which must put the sub-path
// base in front of it like every other API call — a root-anchored request
// 404s at the edge gateway under /docint/.
describe('source previews under a sub-path base', () => {
  afterEach(() => {
    vi.unstubAllEnvs()
    vi.unstubAllGlobals()
    vi.resetModules()
  })

  async function previewUrl(base: string): Promise<string> {
    vi.stubEnv('BASE_URL', base)
    vi.resetModules()
    const fetchMock = vi.fn(async (..._args: unknown[]) => new Response('text', { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)
    const { fetchFile } = await import('./files')
    const { sourcePreviewPath } = await import('./ingest')
    await fetchFile(sourcePreviewPath('alpha', 'hash1'))
    return String(fetchMock.mock.calls[0][0])
  }

  it('prefixes the /docint base like every other API call', async () => {
    expect(await previewUrl('/docint/')).toBe('/docint/sources/preview?collection=alpha&file_hash=hash1')
  })

  it('stays root-anchored at root base', async () => {
    expect(await previewUrl('/')).toBe('/sources/preview?collection=alpha&file_hash=hash1')
  })
})
