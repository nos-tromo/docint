import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { ApiError, setOwnerParam } from './client'
import {
  createExtract,
  deleteExtract,
  extractDownloadHref,
  listExtracts,
  sourceExtractHref
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

  it('builds download hrefs that encode their parts', () => {
    expect(extractDownloadHref('my docs', '20260102-030405-deadbeef')).toContain(
      '/collections/my%20docs/extracts/20260102-030405-deadbeef/download'
    )
    expect(sourceExtractHref('mydocs', 'a/b', 'md')).toContain('/sources/a%2Fb/extract.md')
  })

  it('carries the admin owner context on every href', () => {
    setOwnerParam('other-operator')
    expect(extractDownloadHref('mydocs', '20260102-030405-deadbeef')).toContain(
      '?owner=other-operator'
    )
    expect(sourceExtractHref('mydocs', 'abc', 'zip')).toContain('?owner=other-operator')
  })
})
