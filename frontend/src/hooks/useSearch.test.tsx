import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactNode } from 'react'
import { useChunkText, useScope, useSearch } from './useSearch'
import { ApiError } from '@/api/client'
import { useUiStore } from '@/stores/ui'
import { useChatFiltersStore } from '@/stores/chatFilters'

afterEach(() => vi.restoreAllMocks())

beforeEach(() => {
  useUiStore.setState({ selectedCollection: null })
  useChatFiltersStore.getState().reset()
})

function mockFetch(body: unknown, status = 200) {
  const fn = vi.fn().mockResolvedValue({
    ok: status < 400,
    status,
    json: async () => body,
    text: async () => JSON.stringify(body)
  })
  vi.stubGlobal('fetch', fn)
  return fn
}

function wrapper({ children }: { children: ReactNode }) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return <QueryClientProvider client={qc}>{children}</QueryClientProvider>
}

const INDEX_STATUS = {
  indexed: true,
  total: 10,
  with_search_text: 10,
  missing: 0,
  complete: true
}

describe('useSearch', () => {
  it('returns the hits for a successful search', async () => {
    mockFetch({
      status: 'ok',
      hits: [
        {
          id: 'p1',
          filename: 'alpha.pdf',
          page: 3,
          preview: 'a matching chunk',
          entity_types: ['PERSON'],
          est_tokens: 120
        }
      ],
      total: 1,
      next_cursor: null,
      index_status: INDEX_STATUS
    })
    useUiStore.setState({ selectedCollection: 'docs' })

    const { result } = renderHook(() => useSearch('alpha'), { wrapper })

    await waitFor(() => expect(result.current.data).toBeDefined())
    expect(result.current.data?.status).toBe('ok')
    expect(result.current.data?.hits).toHaveLength(1)
    expect(result.current.data?.hits[0].id).toBe('p1')
  })

  it('surfaces not_indexed as its own status rather than as an empty result', async () => {
    // Collapsing this into "no matches" would tell an investigator the
    // evidence is absent when the collection was merely never backfilled.
    mockFetch({
      status: 'not_indexed',
      hits: [],
      total: 0,
      next_cursor: null,
      index_status: { ...INDEX_STATUS, with_search_text: 0, complete: false }
    })
    useUiStore.setState({ selectedCollection: 'docs' })

    const { result } = renderHook(() => useSearch('alpha'), { wrapper })

    await waitFor(() => expect(result.current.data).toBeDefined())
    expect(result.current.data?.status).toBe('not_indexed')
    expect(result.current.data?.hits).toEqual([])
  })

  it('exposes how many chunks are missing on a partial response', async () => {
    mockFetch({
      status: 'partial',
      hits: [
        { id: 'p1', filename: 'alpha.pdf', preview: 'x', entity_types: [], est_tokens: 4 }
      ],
      total: 1,
      next_cursor: null,
      index_status: { ...INDEX_STATUS, with_search_text: 6, missing: 4, complete: false }
    })
    useUiStore.setState({ selectedCollection: 'docs' })

    const { result } = renderHook(() => useSearch('alpha'), { wrapper })

    await waitFor(() => expect(result.current.data).toBeDefined())
    expect(result.current.data?.status).toBe('partial')
    expect(result.current.data?.index_status.missing).toBe(4)
    // The hits still come back — a partial index is not an empty one.
    expect(result.current.data?.hits).toHaveLength(1)
  })

  it('sends the collection and the panel filters in the request body', async () => {
    const fetchMock = mockFetch({
      status: 'ok',
      hits: [],
      total: 0,
      next_cursor: null,
      index_status: INDEX_STATUS
    })
    useUiStore.setState({ selectedCollection: 'docs' })
    useChatFiltersStore.getState().setFilterEnabled(true)
    useChatFiltersStore.getState().setMimePattern('application/pdf')

    renderHook(() => useSearch('alpha'), { wrapper })

    await waitFor(() => {
      const call = fetchMock.mock.calls.find(([u]) => String(u).includes('/search'))
      expect(call).toBeDefined()
      expect(JSON.parse(call![1].body)).toMatchObject({
        question: 'alpha',
        collection: 'docs',
        metadata_filters: [{ field: 'mimetype', operator: 'mime_match', value: 'application/pdf' }]
      })
    })
  })

  it('does not search before a collection is selected', async () => {
    const fetchMock = mockFetch({})

    renderHook(() => useSearch('alpha'), { wrapper })

    await new Promise((r) => setTimeout(r, 20))
    expect(fetchMock.mock.calls.some(([u]) => String(u).includes('/search'))).toBe(false)
  })

  it('does not search on a blank query — that would scan the whole collection', async () => {
    const fetchMock = mockFetch({})
    useUiStore.setState({ selectedCollection: 'docs' })

    renderHook(() => useSearch('   '), { wrapper })

    await new Promise((r) => setTimeout(r, 20))
    expect(fetchMock.mock.calls.some(([u]) => String(u).includes('/search'))).toBe(false)
  })
})

describe('useChunkText', () => {
  it('fetches nothing until the hit is expanded', async () => {
    const fetchMock = mockFetch({ id: 'p1', text: 'the whole chunk' })
    useUiStore.setState({ selectedCollection: 'docs' })

    renderHook(() => useChunkText('p1', false), { wrapper })

    await new Promise((r) => setTimeout(r, 20))
    expect(fetchMock.mock.calls.some(([u]) => String(u).includes('/search/chunk'))).toBe(false)
  })

  it('fetches the chunk text once expanded, scoped to the collection', async () => {
    const fetchMock = mockFetch({ id: 'p1', text: 'the whole chunk' })
    useUiStore.setState({ selectedCollection: 'docs' })

    const { result } = renderHook(() => useChunkText('p1', true), { wrapper })

    await waitFor(() => expect(result.current.data).toBeDefined())
    expect(result.current.data?.text).toBe('the whole chunk')
    const call = fetchMock.mock.calls.find(([u]) => String(u).includes('/search/chunk'))
    expect(String(call![0])).toContain('id=p1')
    expect(String(call![0])).toContain('collection=docs')
  })

  it('surfaces a 404 as an error, never as empty text', async () => {
    // A re-ingest mints new point ids, so a hit can outlive its chunk. An
    // empty string here would render as an empty chunk instead of a gone one.
    mockFetch({ detail: 'Not found.' }, 404)
    useUiStore.setState({ selectedCollection: 'docs' })

    const { result } = renderHook(() => useChunkText('gone', true), { wrapper })

    await waitFor(() => expect(result.current.isError).toBe(true))
    expect(result.current.data).toBeUndefined()
    expect((result.current.error as ApiError).status).toBe(404)
  })
})

describe('useScope', () => {
  it('PUTs the selected chunk ids to the session scope', async () => {
    const fetchMock = mockFetch({
      chunk_ids: ['p1', 'p2'],
      est_tokens: 240,
      usable_tokens: 22000,
      missing: 0
    })
    useUiStore.setState({ selectedCollection: 'docs' })

    const { result } = renderHook(() => useScope('sess-1'), { wrapper })
    const stored = await result.current.set.mutateAsync(['p1', 'p2'])

    expect(stored.usable_tokens).toBe(22000)
    const call = fetchMock.mock.calls.find(([u]) => String(u).includes('/scope'))
    expect(call).toBeDefined()
    expect(call![1].method).toBe('PUT')
    expect(String(call![0])).toContain('collection=docs')
    expect(JSON.parse(call![1].body)).toEqual({ chunk_ids: ['p1', 'p2'] })
  })

  it('DELETEs to clear the scope', async () => {
    const fetchMock = mockFetch({ chunk_ids: [], est_tokens: 0, usable_tokens: 0, missing: 0 })

    const { result } = renderHook(() => useScope('sess-1'), { wrapper })
    await result.current.clear.mutateAsync()

    const call = fetchMock.mock.calls.find(([u]) => String(u).includes('/scope'))
    expect(call![1].method).toBe('DELETE')
  })
})
