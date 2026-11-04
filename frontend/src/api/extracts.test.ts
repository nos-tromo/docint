import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { ApiError, setOwnerParam } from './client'
import {
  createExtract,
  deleteExtract,
  extractDownloadPath,
  listExtracts,
  sourceExtractPath
} from './extracts'

const originalFetch = globalThis.fetch

/** Stub `fetch` with one canned response, JSON body included. */
function stubFetch(body: unknown, init: { ok?: boolean; status?: number } = {}) {
  const spy = vi.fn(async (input: RequestInfo | URL, requestInit?: RequestInit) => {
    void input
    void requestInit
    return {
      ok: init.ok ?? true,
      status: init.status ?? 200,
      text: async () => JSON.stringify(body),
      json: async () => body
    }
  })
  globalThis.fetch = spy as unknown as typeof fetch
  return spy
}

describe('extract API', () => {
  beforeEach(() => setOwnerParam(null))
  afterEach(() => {
    globalThis.fetch = originalFetch
    setOwnerParam(null)
  })

  it('queues a build and returns its job id', async () => {
    stubFetch({ job_id: 'j1' })
    await expect(createExtract('mydocs')).resolves.toEqual({ job_id: 'j1', adopted: false })
  })

  it('files a queued build under the active report, omitting what is unset', async () => {
    const spy = stubFetch({ job_id: 'j1' })
    await createExtract('mydocs', undefined, { reference_number: 'AZ-12', operator: '' })
    expect(JSON.parse(String(spy.mock.calls[0][1]?.body))).toEqual({ reference_number: 'AZ-12' })
  })

  it('carries the appendix fields on a single-source download', () => {
    const path = sourceExtractPath('mydocs', 'a1b2', 'pdf', { reference_number: 'AZ-12', operator: '' })
    expect(path).toBe('/collections/mydocs/sources/a1b2/extract.pdf?reference_number=AZ-12')
  })

  it('adopts the in-flight job a 409 names', async () => {
    stubFetch({ detail: { message: 'busy', job_id: 'j9' } }, { ok: false, status: 409 })
    await expect(createExtract('mydocs')).resolves.toEqual({ job_id: 'j9', adopted: true })
  })

  it('rethrows a 409 that names no job', async () => {
    stubFetch({ detail: 'busy' }, { ok: false, status: 409 })
    await expect(createExtract('mydocs')).rejects.toBeInstanceOf(ApiError)
  })

  it('sends the target when one source is asked for', async () => {
    const spy = stubFetch({ job_id: 'j1' })
    await createExtract('mydocs', 'abc123')
    const init = spy.mock.calls[0][1] as RequestInit
    expect(JSON.parse(String(init.body))).toEqual({ target: 'abc123' })
  })

  it('lists and deletes through the collection-scoped paths', async () => {
    const spy = stubFetch({ extracts: [] })
    await listExtracts('my docs')
    await deleteExtract('my docs', '20260102-030405-deadbeef')
    expect(String(spy.mock.calls[0][0])).toContain('/collections/my%20docs/extracts')
    expect(String(spy.mock.calls[1][0])).toContain('/extracts/20260102-030405-deadbeef')
  })

  it('builds download paths that encode their parts', () => {
    expect(extractDownloadPath('my docs', '20260102-030405-deadbeef')).toBe(
      '/collections/my%20docs/extracts/20260102-030405-deadbeef/download'
    )
    expect(sourceExtractPath('mydocs', 'a/b', 'md')).toBe('/collections/mydocs/sources/a%2Fb/extract.md')
  })

  it('leaves the admin owner context to the wrapper, which adds it once', () => {
    setOwnerParam('other-operator')
    expect(extractDownloadPath('mydocs', '20260102-030405-deadbeef')).not.toContain('owner=')
    expect(sourceExtractPath('mydocs', 'abc', 'zip')).not.toContain('owner=')
  })
})
