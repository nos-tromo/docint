import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { apiGet, apiPost, setOwnerParam } from './client'

describe('owner query param plumbing', () => {
  const fetchMock = vi.fn(async (..._args: unknown[]) => new Response(JSON.stringify({ ok: true })))

  beforeEach(() => {
    vi.stubGlobal('fetch', fetchMock)
    fetchMock.mockClear()
  })

  afterEach(() => {
    setOwnerParam(null)
    vi.unstubAllGlobals()
  })

  it('appends owner to GET requests, composing with existing params', async () => {
    setOwnerParam('jane.doe')
    await apiGet('/collections/documents', { collection: 'alpha' })
    const called = String(fetchMock.mock.calls[0][0])
    expect(called).toContain('collection=alpha')
    expect(called).toContain('owner=jane.doe')
  })

  it('appends owner to POST requests as a query param', async () => {
    setOwnerParam('jane.doe')
    await apiPost('/collections/select', { name: 'alpha' })
    expect(String(fetchMock.mock.calls[0][0])).toContain('?owner=jane.doe')
  })

  it('adds nothing when no owner is set', async () => {
    await apiGet('/collections/list')
    expect(String(fetchMock.mock.calls[0][0])).not.toContain('owner=')
  })

  it('sends the owner a caller pinned, whichever one is selected now', async () => {
    setOwnerParam('bob')
    await apiGet('/ingest/staged', { collection: 'alpha' }, 'alice')
    await apiPost('/ingest/finalize', { collection: 'alpha' }, 'alice')
    const [get, post] = fetchMock.mock.calls.map((call) => String(call[0]))
    expect(get).toContain('owner=alice')
    expect(get).not.toContain('owner=bob')
    expect(post).toContain('?owner=alice')
  })

  it('sends no owner when a caller pinned none', async () => {
    setOwnerParam('bob')
    await apiGet('/ingest/staged', { collection: 'alpha' }, null)
    expect(String(fetchMock.mock.calls[0][0])).not.toContain('owner=')
  })
})
