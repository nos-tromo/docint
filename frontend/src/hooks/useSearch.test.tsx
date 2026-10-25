import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactNode } from 'react'
import { useChunkText, useScope, useSearch } from './useSearch'
import { ApiError } from '@/api/client'
import { useUiStore } from '@/stores/ui'
import { useChatFiltersStore } from '@/stores/chatFilters'
import { useTopicFocusStore } from '@/stores/topicFocus'

afterEach(() => vi.restoreAllMocks())

beforeEach(() => {
  useUiStore.setState({ selectedCollection: null })
  useChatFiltersStore.getState().reset()
  useTopicFocusStore.setState({ focus: null })
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

  it('sends the chosen field in the request body', async () => {
    const fetchMock = mockFetch({
      status: 'ok',
      hits: [],
      total: 0,
      next_cursor: null,
      index_status: INDEX_STATUS
    })
    useUiStore.setState({ selectedCollection: 'docs' })

    const { result } = renderHook(() => useSearch('mar', 'author'), { wrapper })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    const call = fetchMock.mock.calls.find(([u]) => String(u).includes('/search'))
    expect(JSON.parse(call![1].body)).toMatchObject({ question: 'mar', field: 'author' })
  })

  it('defaults to the text field', async () => {
    const fetchMock = mockFetch({
      status: 'ok',
      hits: [],
      total: 0,
      next_cursor: null,
      index_status: INDEX_STATUS
    })
    useUiStore.setState({ selectedCollection: 'docs' })

    renderHook(() => useSearch('alpha'), { wrapper })

    await waitFor(() => {
      const call = fetchMock.mock.calls.find(([u]) => String(u).includes('/search'))
      expect(JSON.parse(call![1].body)).toMatchObject({ field: 'text' })
    })
  })

  it('appends the topic rule to a search and its key', async () => {
    const run = '20260102-030405-a1b2c3'
    const respond = (body: unknown) => ({
      ok: true,
      status: 200,
      json: async () => body,
      text: async () => JSON.stringify(body)
    })
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      void init
      const u = String(input)
      if (u.includes('/config')) return respond({ topics_enabled: true })
      if (u.includes('/topics')) {
        return respond({
          run_id: run,
          created_at: '2026-01-02T03:04:05+00:00',
          updated_at: '2026-01-02T03:04:05+00:00',
          embed_model: 'embed-small',
          embed_model_changed: false,
          coverage: { total: 10, assigned: 8, unassigned: 2, pending: 0 },
          topics: []
        })
      }
      return respond({ status: 'ok', hits: [], total: 0, next_cursor: null, index_status: INDEX_STATUS })
    })
    vi.stubGlobal('fetch', fetchMock)
    useUiStore.setState({ selectedCollection: 'docs', selectedOwner: null })
    useChatFiltersStore.getState().setFilterEnabled(true)
    useChatFiltersStore.getState().setMimePattern('application/pdf')
    useTopicFocusStore.setState({ focus: { collection: 'docs', owner: null, runId: run, topicId: 3, label: 'Harbour cranes' } })
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const withClient = ({ children }: { children: ReactNode }) => (
      <QueryClientProvider client={qc}>{children}</QueryClientProvider>
    )

    renderHook(() => useSearch('alpha'), { wrapper: withClient })

    const rules = [
      { field: 'mimetype', operator: 'mime_match', value: 'application/pdf' },
      { field: 'docint_topic', operator: 'eq', value: '20260102-030405-a1b2c3:3' }
    ]
    await waitFor(() => {
      const searches = fetchMock.mock.calls.filter(([u]) => String(u).includes('/search'))
      expect(searches.length).toBeGreaterThan(0)
      expect(JSON.parse(String(searches.at(-1)![1]!.body)).metadata_filters).toEqual(rules)
    })
    // Keyed by the rule too, so dropping the focus re-runs the search instead of showing its hits.
    const keys = qc.getQueryCache().findAll({ queryKey: ['search'] }).map((query) => query.queryKey)
    expect(keys).toContainEqual(['search', { collection: 'docs', owner: null }, 'alpha', 'text', rules])
  })

  /**
   * Stub the flag, a topics run held back until released, and the search.
   *
   * @returns The fetch mock and the release for the run read.
   */
  function stubHeldRun() {
    const respond = (body: unknown) => ({
      ok: true,
      status: 200,
      json: async () => body,
      text: async () => JSON.stringify(body)
    })
    let release: () => void = () => {}
    const held = new Promise<void>((resolve) => {
      release = resolve
    })
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      void init
      const u = String(input)
      if (u.includes('/config')) return respond({ topics_enabled: true })
      if (u.includes('/topics')) {
        await held
        return respond({
          run_id: '20260102-030405-a1b2c3',
          created_at: '2026-01-02T03:04:05+00:00',
          updated_at: '2026-01-02T03:04:05+00:00',
          embed_model: 'embed-small',
          embed_model_changed: false,
          coverage: { total: 10, assigned: 8, unassigned: 2, pending: 0 },
          topics: []
        })
      }
      return respond({ status: 'ok', hits: [], total: 0, next_cursor: null, index_status: INDEX_STATUS })
    })
    vi.stubGlobal('fetch', fetchMock)
    return { fetchMock, release }
  }

  it('reads no topics run without a focus', async () => {
    const { fetchMock, release } = stubHeldRun()
    release()
    useUiStore.setState({ selectedCollection: 'docs', selectedOwner: null })

    const { result } = renderHook(() => useSearch('alpha'), { wrapper })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    await new Promise((resolve) => setTimeout(resolve, 30))
    expect(fetchMock.mock.calls.filter(([u]) => String(u).includes('/topics'))).toEqual([])
  })

  it('searches without the rule when the topics run cannot be read', async () => {
    const respond = (body: unknown, status = 200) => ({
      ok: status < 400,
      status,
      json: async () => body,
      text: async () => JSON.stringify(body)
    })
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      void init
      const u = String(input)
      if (u.includes('/config')) return respond({ topics_enabled: true })
      if (u.includes('/topics')) return respond({ detail: 'unavailable' }, 503)
      return respond({ status: 'ok', hits: [], total: 0, next_cursor: null, index_status: INDEX_STATUS })
    })
    vi.stubGlobal('fetch', fetchMock)
    useUiStore.setState({ selectedCollection: 'docs', selectedOwner: null })
    useTopicFocusStore.setState({
      focus: { collection: 'docs', owner: null, runId: '20260102-030405-a1b2c3', topicId: 3, label: 'Harbour cranes' }
    })

    const { result } = renderHook(() => useSearch('alpha'), { wrapper })

    // A failed read never settles into a run; waiting on it would hold the search for good.
    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    const searches = fetchMock.mock.calls.filter(([u]) => String(u).includes('/search'))
    expect(searches).toHaveLength(1)
    expect(JSON.parse(String(searches[0][1]!.body)).metadata_filters).toEqual([])
  })

  it('sends one search, carrying the rule, once the topics run is read', async () => {
    const { fetchMock, release } = stubHeldRun()
    useUiStore.setState({ selectedCollection: 'docs', selectedOwner: null })
    useTopicFocusStore.setState({
      focus: { collection: 'docs', owner: null, runId: '20260102-030405-a1b2c3', topicId: 3, label: 'Harbour cranes' }
    })
    const searches = () => fetchMock.mock.calls.filter(([u]) => String(u).includes('/search'))

    const { result } = renderHook(() => useSearch('alpha'), { wrapper })

    await waitFor(() => expect(fetchMock.mock.calls.some(([u]) => String(u).includes('/topics'))).toBe(true))
    await new Promise((resolve) => setTimeout(resolve, 30))
    // Whether the rule applies is not known yet; a search sent now would be one without it.
    expect(searches()).toEqual([])

    release()

    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(searches()).toHaveLength(1)
    expect(JSON.parse(String(searches()[0][1]!.body)).metadata_filters).toEqual([
      { field: 'docint_topic', operator: 'eq', value: '20260102-030405-a1b2c3:3' }
    ])
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

