import { describe, it, expect, vi, afterEach } from 'vitest'
import { apiGet, apiPost, apiDelete, ApiError, apiBase } from './client'

afterEach(() => {
  vi.restoreAllMocks()
})

describe('apiBase', () => {
  it('uses an explicit VITE_API_BASE_URL override verbatim (trailing slash trimmed)', () => {
    expect(apiBase('http://elsewhere/', '/docint/')).toBe('http://elsewhere')
  })
  it('derives from BASE_URL when no override is set', () => {
    expect(apiBase(undefined, '/docint/')).toBe('/docint')
  })
  it('is empty (same-origin root) at root BASE_URL with no override', () => {
    expect(apiBase(undefined, '/')).toBe('')
  })
})

function mockFetch(body: unknown, init: { status?: number; ok?: boolean } = {}) {
  const status = init.status ?? 200
  const ok = init.ok ?? status < 400
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
    ok,
    status,
    json: async () => body,
    text: async () => JSON.stringify(body)
  }))
}

describe('client', () => {
  it('apiGet returns parsed JSON', async () => {
    mockFetch({ hello: 'world' })
    expect(await apiGet<{ hello: string }>('/x')).toEqual({ hello: 'world' })
  })

  it('apiPost sends JSON body', async () => {
    mockFetch({ ok: true })
    await apiPost('/x', { a: 1 })
    const call = (fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0]
    expect(call[1].method).toBe('POST')
    expect(call[1].headers['Content-Type']).toBe('application/json')
    expect(call[1].body).toBe('{"a":1}')
  })

  it('apiDelete uses DELETE method', async () => {
    mockFetch({ ok: true })
    await apiDelete('/x')
    const call = (fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0]
    expect(call[1].method).toBe('DELETE')
  })

  it('throws ApiError on non-2xx', async () => {
    mockFetch({ detail: 'bad' }, { status: 400, ok: false })
    await expect(apiGet('/x')).rejects.toBeInstanceOf(ApiError)
  })
})
