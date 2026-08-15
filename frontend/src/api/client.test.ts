import { describe, it, expect, vi, afterEach } from 'vitest'
import { apiGet, apiGetOrNull, apiPost, apiDelete, ApiError, apiBase } from './client'

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

describe('apiGetOrNull', () => {
  it('returns null on 204 without touching the empty body', async () => {
    // A 204 carries no body at all, so `handle`'s unconditional res.json()
    // would reject. That is the whole reason this entry point exists.
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        status: 204,
        json: async () => {
          throw new SyntaxError('Unexpected end of JSON input')
        }
      })
    )
    await expect(apiGetOrNull('/summarize')).resolves.toBeNull()
  })

  it('returns the parsed body on 200', async () => {
    mockFetch({ summary: 'text' })
    await expect(apiGetOrNull<{ summary: string }>('/summarize')).resolves.toEqual({ summary: 'text' })
  })

  it('still throws on 404 — absent is not the same as forbidden', async () => {
    mockFetch({ detail: 'nope' }, { status: 404 })
    await expect(apiGetOrNull('/summarize')).rejects.toBeInstanceOf(ApiError)
  })
})
