import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { SearchPanel, formatTokens, queryKeywords } from './SearchPanel'
import { useUiStore } from '@/stores/ui'
import { useChatFiltersStore } from '@/stores/chatFilters'
import { useSearchUiStore } from '@/stores/searchUi'
import type { SearchResult } from '@/api/types'

const SESSION = 'sess-1'

const INDEX_STATUS = {
  indexed: true,
  total: 10,
  with_search_text: 10,
  missing: 0,
  complete: true
}

const HIT = {
  id: 'p1',
  chunk_id: 'c1',
  filename: 'alpha.pdf',
  page: 3,
  row: null,
  preview: 'Der Parteitag beschloss den Tagesordnungspunkt.',
  entity_types: ['PERSON'],
  est_tokens: 1200
}

const SCOPE_OK = { chunk_ids: ['p1'], est_tokens: 1200, usable_tokens: 22000, missing: 0 }

/** Route by URL so one mock serves both /search and the scope endpoints. */
function mockApi(search: SearchResult, scope: { body: unknown; status?: number } = { body: SCOPE_OK }) {
  const fn = vi.fn((req: RequestInfo | URL, init?: RequestInit) => {
    const u = typeof req === 'string' ? req : String(req)
    if (u.includes('/scope')) {
      const status = scope.status ?? 200
      return Promise.resolve({
        ok: status < 400,
        status,
        json: async () => scope.body,
        text: async () => JSON.stringify(scope.body)
      })
    }
    void init
    return Promise.resolve({
      ok: true,
      status: 200,
      json: async () => search,
      text: async () => JSON.stringify(search)
    })
  })
  vi.stubGlobal('fetch', fn)
  return fn
}

function renderPanel(sessionId: string | null = SESSION) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter>
        <SearchPanel sessionId={sessionId} />
      </MemoryRouter>
    </QueryClientProvider>
  )
}

beforeEach(() => {
  useUiStore.setState({ selectedCollection: 'docs' })
  useChatFiltersStore.getState().reset()
  useSearchUiStore.setState({
    drafts: {},
    queries: { [SESSION]: 'Partei', new: 'Partei' },
    scopes: {},
    filtersOpen: false
  })
})

afterEach(() => vi.restoreAllMocks())

describe('SearchPanel result states', () => {
  it('renders the hits of an ok response', async () => {
    mockApi({
      status: 'ok',
      hits: [HIT],
      total: 1,
      next_cursor: null,
      index_status: INDEX_STATUS
    })

    renderPanel()

    expect(await screen.findByText(/alpha\.pdf/)).toBeInTheDocument()
    expect(screen.getByTestId('search-summary')).toHaveTextContent('1 hits')
    expect(screen.queryByTestId('search-no-matches')).toBeNull()
  })

  it('prompts to build the index on not_indexed, and never says "no matches"', async () => {
    // A collection that was never backfilled has an answerable question and no
    // way to answer it; "no matches" would be a lie about the evidence.
    mockApi({
      status: 'not_indexed',
      hits: [],
      total: 0,
      next_cursor: null,
      index_status: { ...INDEX_STATUS, with_search_text: 0, complete: false }
    })

    renderPanel()

    const banner = await screen.findByTestId('search-not-indexed')
    expect(banner).toHaveTextContent(/make search-index/i)
    expect(screen.queryByTestId('search-no-matches')).toBeNull()
    expect(screen.queryByTestId('search-partial')).toBeNull()
  })

  it('renders the partial warning together with the hits it did find', async () => {
    mockApi({
      status: 'partial',
      hits: [HIT],
      total: 1,
      next_cursor: null,
      index_status: { ...INDEX_STATUS, with_search_text: 6, missing: 4, complete: false }
    })

    renderPanel()

    const warning = await screen.findByTestId('search-partial')
    expect(warning).toHaveTextContent(/4 chunks are not indexed/i)
    // The hits must still be shown — a partial index is not an empty one.
    expect(screen.getByText(/alpha\.pdf/)).toBeInTheDocument()
    expect(screen.queryByTestId('search-no-matches')).toBeNull()
  })

  it('says "no matches" only for an ok response with no hits', async () => {
    mockApi({
      status: 'ok',
      hits: [],
      total: 0,
      next_cursor: null,
      index_status: INDEX_STATUS
    })

    renderPanel()

    expect(await screen.findByTestId('search-no-matches')).toBeInTheDocument()
    expect(screen.queryByTestId('search-not-indexed')).toBeNull()
    expect(screen.queryByTestId('search-partial')).toBeNull()
  })
})

describe('SearchPanel scope selection', () => {
  it('writes the scope when a hit is checked', async () => {
    const fetchMock = mockApi({
      status: 'ok',
      hits: [HIT],
      total: 1,
      next_cursor: null,
      index_status: INDEX_STATUS
    })

    renderPanel()

    const checkbox = await screen.findByRole('checkbox', { name: /alpha\.pdf/i })
    await userEvent.click(checkbox)

    await waitFor(() => {
      const call = fetchMock.mock.calls.find(([u]) => String(u).includes('/scope'))
      expect(call).toBeDefined()
      expect(call![1]!.method).toBe('PUT')
      expect(JSON.parse(String(call![1]!.body))).toEqual({ chunk_ids: ['p1'] })
    })
    await waitFor(() => {
      expect(useSearchUiStore.getState().scopes[SESSION]?.tokens).toEqual({ p1: 1200 })
    })
  })

  it('shows remaining capacity live once the budget is known', async () => {
    mockApi({
      status: 'ok',
      hits: [HIT],
      total: 1,
      next_cursor: null,
      index_status: INDEX_STATUS
    })

    renderPanel()

    await userEvent.click(await screen.findByRole('checkbox', { name: /alpha\.pdf/i }))

    // Before the PUT lands only the local sum is known; after it, the meter
    // shows the capacity the backend measured.
    await waitFor(() => {
      expect(screen.getByTestId('token-meter')).toHaveTextContent('≈1.2k / 22.0k tokens')
    })
  })

  it('rolls the selection back and explains when the scope exceeds the budget', async () => {
    // 422 is terminal: the selection cannot fit, so keeping it checked would
    // claim an evidence set the next answer will not actually use.
    mockApi(
      {
        status: 'ok',
        hits: [HIT],
        total: 1,
        next_cursor: null,
        index_status: INDEX_STATUS
      },
      { body: { detail: 'Invalid request.' }, status: 422 }
    )

    renderPanel()

    await userEvent.click(await screen.findByRole('checkbox', { name: /alpha\.pdf/i }))

    expect(await screen.findByText(/larger than the answer can hold/i)).toBeInTheDocument()
    await waitFor(() => {
      expect(useSearchUiStore.getState().scopes[SESSION]?.tokens).toEqual({})
    })
    // The raw backend body is never rendered.
    expect(screen.queryByText(/Invalid request/)).toBeNull()
  })

  it('holds the selection locally and says so before the chat has an id', async () => {
    const fetchMock = mockApi({
      status: 'ok',
      hits: [HIT],
      total: 1,
      next_cursor: null,
      index_status: INDEX_STATUS
    })

    renderPanel(null)

    await userEvent.click(await screen.findByRole('checkbox', { name: /alpha\.pdf/i }))

    expect(await screen.findByText(/apply once this chat has started/i)).toBeInTheDocument()
    expect(fetchMock.mock.calls.some(([u]) => String(u).includes('/scope'))).toBe(false)
  })
})

describe('SearchPanel hit rendering', () => {
  it('highlights whole words the keyword prefixes, and nothing mid-word', async () => {
    useSearchUiStore.setState({ queries: { [SESSION]: 'Partei' } })
    mockApi({
      status: 'ok',
      hits: [HIT],
      total: 1,
      next_cursor: null,
      index_status: INDEX_STATUS
    })

    const { container } = renderPanel()

    await screen.findByText(/alpha\.pdf/)
    const marked = [...container.querySelectorAll('mark')].map((el) => el.textContent)
    // 'Partei' is a prefix of 'Parteitag' (which the index matches) and of
    // nothing else in the preview — mid-word matching is not a thing here.
    expect(marked).toEqual(['Parteitag'])
  })

  it('renders entity types as badges', async () => {
    mockApi({
      status: 'ok',
      hits: [HIT],
      total: 1,
      next_cursor: null,
      index_status: INDEX_STATUS
    })

    renderPanel()

    expect(await screen.findByText('PERSON')).toBeInTheDocument()
  })

  it('links each hit into the Inspector', async () => {
    mockApi({
      status: 'ok',
      hits: [HIT],
      total: 1,
      next_cursor: null,
      index_status: INDEX_STATUS
    })

    renderPanel()

    const link = await screen.findByRole('link', { name: /documents/i })
    expect(link).toHaveAttribute('href', '/inspector')
  })
})

describe('SearchPanel helpers', () => {
  it('compacts token counts for the meter', () => {
    expect(formatTokens(940)).toBe('940')
    expect(formatTokens(12400)).toBe('12.4k')
  })

  it('splits a query into the keywords that must all match', () => {
    expect(queryKeywords('  Partei  Tag ')).toEqual(['Partei', 'Tag'])
    expect(queryKeywords('   ')).toEqual([])
  })
})
