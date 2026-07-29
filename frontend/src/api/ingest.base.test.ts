import { afterEach, describe, expect, it, vi } from 'vitest'

// sourcePreviewUrl is consumed as a plain <a href> (chat citations, entity
// findings), so unlike fetch-based calls it must carry the sub-path base
// itself — a root-anchored href 404s at the edge gateway under /docint/.
describe('sourcePreviewUrl under a sub-path base', () => {
  afterEach(() => {
    vi.unstubAllEnvs()
    vi.resetModules()
  })

  async function loadWithBase(base: string) {
    vi.stubEnv('BASE_URL', base)
    vi.resetModules()
    return import('./ingest')
  }

  it('prefixes the /docint base like every other API call', async () => {
    const { sourcePreviewUrl } = await loadWithBase('/docint/')
    expect(sourcePreviewUrl('alpha', 'hash1')).toBe(
      '/docint/sources/preview?collection=alpha&file_hash=hash1'
    )
  })

  it('stays root-anchored at root base', async () => {
    const { sourcePreviewUrl } = await loadWithBase('/')
    expect(sourcePreviewUrl('alpha', 'hash1')).toBe(
      '/sources/preview?collection=alpha&file_hash=hash1'
    )
  })
})
