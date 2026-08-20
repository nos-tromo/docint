import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { SearchPanel, formatTokens, queryKeywords } from './SearchPanel'
import { useUiStore } from '@/stores/ui'
import { useChatFiltersStore } from '@/stores/chatFilters'
import { useSearchUiStore } from '@/stores/searchUi'
import type { AggregateResult, SearchResult } from '@/api/types'

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

const HIT2 = {
  id: 'p2',
  chunk_id: 'c2',
  filename: 'beta.pdf',
  page: 7,
  row: null,
  preview: 'Zweiter Abschnitt zum Parteitag.',
  entity_types: [],
  est_tokens: 1200
}

/** The text only the expanded view shows — never part of a hit's preview. */
const FULL_TEXT = 'Der Parteitag beschloss den Tagesordnungspunkt. Danach folgte die Aussprache.'

const SCOPE_OK = { chunk_ids: ['p1'], est_tokens: 1200, usable_tokens: 22000, missing: 0 }

const CHUNK_OK = { body: { id: 'p1', text: FULL_TEXT } }

const AGGREGATE_OK: AggregateResult = {
  status: 'ok',
  group_by: 'author',
  total: 5,
  unassigned: 0,
  groups: [{ value: 'acme_news', count: 5, samples: [HIT] }],
  limit: 100,
  index_status: INDEX_STATUS
}

/** Two groups, each with a distinct sample — enough to prove mark-all in
 *  Social mode collects samples across every group, not just the first. */
const AGGREGATE_TWO: AggregateResult = {
  status: 'ok',
  group_by: 'author',
  total: 2,
  unassigned: 0,
  groups: [
    { value: 'acme_news', count: 1, samples: [HIT] },
    { value: 'beta_daily', count: 1, samples: [HIT2] }
  ],
  limit: 100,
  index_status: INDEX_STATUS
}

/** Route by URL so one mock serves /search, /search/aggregate, /search/chunk
 *  and the scope endpoints. */
function mockApi(
  search: SearchResult,
  scope: { body: unknown; status?: number } = { body: SCOPE_OK },
  chunk: { body: unknown; status?: number } = CHUNK_OK,
  aggregate: { body: unknown; status?: number } = { body: AGGREGATE_OK }
) {
  const fn = vi.fn((req: RequestInfo | URL, init?: RequestInit) => {
    const u = typeof req === 'string' ? req : String(req)
    // Checked before '/scope' and the generic '/search' fallthrough — and
    // before '/search/chunk' would matter too, though the two never collide.
    if (u.includes('/search/aggregate')) {
      const status = aggregate.status ?? 200
      return Promise.resolve({
        ok: status < 400,
        status,
        json: async () => aggregate.body,
        text: async () => JSON.stringify(aggregate.body)
      })
    }
    if (u.includes('/search/chunk')) {
      const status = chunk.status ?? 200
      return Promise.resolve({
        ok: status < 400,
        status,
        json: async () => chunk.body,
        text: async () => JSON.stringify(chunk.body)
      })
    }
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
    filtersOpen: false,
    mode: 'hits',
    groupBy: 'author'
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
  it('writes the scope when a hit tile is clicked', async () => {
    const fetchMock = mockApi({
      status: 'ok',
      hits: [HIT],
      total: 1,
      next_cursor: null,
      index_status: INDEX_STATUS
    })

    renderPanel()

    const tile = await screen.findByRole('button', { name: /alpha\.pdf/i })
    await userEvent.click(tile)

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

  it('scopes from anywhere on the tile, not just its heading', async () => {
    // The whole card is the control now: an investigator skimming previews
    // should not have to hunt for a 16px box to pin what they just read.
    mockApi({
      status: 'ok',
      hits: [HIT],
      total: 1,
      next_cursor: null,
      index_status: INDEX_STATUS
    })

    renderPanel()

    await userEvent.click(await screen.findByTestId('hit-preview'))

    await waitFor(() => {
      expect(useSearchUiStore.getState().scopes[SESSION]?.tokens).toEqual({ p1: 1200 })
    })
    expect(screen.getByRole('button', { name: /stop answering from/i })).toHaveAttribute(
      'aria-pressed',
      'true'
    )
  })

  it('scopes from the keyboard, with both Enter and Space', async () => {
    // A div carrying role="button" gets neither for free, and Space would
    // otherwise scroll the panel instead of selecting.
    mockApi({
      status: 'ok',
      hits: [HIT],
      total: 1,
      next_cursor: null,
      index_status: INDEX_STATUS
    })

    renderPanel()

    const tile = await screen.findByRole('button', { name: /alpha\.pdf/i })
    tile.focus()
    await userEvent.keyboard('{Enter}')
    await waitFor(() => {
      expect(useSearchUiStore.getState().scopes[SESSION]?.tokens).toEqual({ p1: 1200 })
    })

    await userEvent.keyboard(' ')
    await waitFor(() => {
      expect(useSearchUiStore.getState().scopes[SESSION]?.tokens).toEqual({})
    })
  })

  it('does not re-scope when the click merely ended a text selection', async () => {
    // Dragging across a preview to copy a passage finishes with a click on the
    // tile. Quoting evidence must not silently change what the next answer is
    // allowed to draw on.
    mockApi({
      status: 'ok',
      hits: [HIT],
      total: 1,
      next_cursor: null,
      index_status: INDEX_STATUS
    })

    renderPanel()

    const preview = await screen.findByTestId('hit-preview')
    vi.spyOn(window, 'getSelection').mockReturnValue({ isCollapsed: false } as Selection)
    await userEvent.click(preview)

    expect(useSearchUiStore.getState().scopes[SESSION]?.tokens ?? {}).toEqual({})
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

    await userEvent.click(await screen.findByRole('button', { name: /alpha\.pdf/i }))

    // Before the PUT lands only the local sum is known; after it, the meter
    // shows the capacity the backend measured.
    await waitFor(() => {
      expect(screen.getByTestId('token-meter')).toHaveTextContent('≈1.2k / 22.0k tokens')
    })
  })

  it('renders the token meter outside the truncating summary line even before the active mode has data', async () => {
    // A selection can go live before the active mode's query has resolved —
    // right after a mode switch, or on first render with a persisted
    // selection and no query submitted. The meter is a sibling of the counts
    // `<p>`, not text inside it, so it must not depend on that `<p>` having
    // anything to say.
    useSearchUiStore.setState({
      queries: {},
      scopes: { [SESSION]: { tokens: { p1: 1200 }, usableTokens: 22000, missing: 0 } }
    })
    mockApi({
      status: 'ok',
      hits: [HIT],
      total: 1,
      next_cursor: null,
      index_status: INDEX_STATUS
    })

    renderPanel()

    const meter = await screen.findByTestId('token-meter')
    expect(meter).toHaveTextContent('≈1.2k / 22.0k tokens')
    expect(screen.getByTestId('search-summary')).toHaveTextContent('')
    expect(screen.getByTestId('search-summary-row')).toContainElement(meter)
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

    await userEvent.click(await screen.findByRole('button', { name: /alpha\.pdf/i }))

    expect(await screen.findByText(/larger than the answer can hold/i)).toBeInTheDocument()
    await waitFor(() => {
      expect(useSearchUiStore.getState().scopes[SESSION]?.tokens).toEqual({})
    })
    // The raw backend body is never rendered.
    expect(screen.queryByText(/Invalid request/)).toBeNull()
  })

  it('holds the selection locally, silently, before the chat has an id', async () => {
    // No session to write to yet — Chat flushes the selection when the backend
    // mints one on the first turn. That is not worth a notice: it behaves
    // exactly like a selection made afterwards, and the line it used to print
    // pushed the whole hit list down the moment anything was picked.
    const fetchMock = mockApi({
      status: 'ok',
      hits: [HIT],
      total: 1,
      next_cursor: null,
      index_status: INDEX_STATUS
    })

    renderPanel(null)

    await userEvent.click(await screen.findByRole('button', { name: /alpha\.pdf/i }))

    await waitFor(() => {
      expect(useSearchUiStore.getState().scopes.new?.tokens).toEqual({ p1: 1200 })
    })
    expect(screen.getByTestId('token-meter')).toHaveTextContent('≈1.2k tokens')
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

  it('marks an image hit so a caption is not mistaken for document prose', async () => {
    // Image hits come from the `_images` companion: their body is a caption
    // and tags, so they read differently from a document chunk.
    mockApi({
      status: 'ok',
      hits: [{ ...HIT, id: 'img1', kind: 'image' as const }],
      total: 1,
      next_cursor: null,
      index_status: INDEX_STATUS
    })
    renderPanel()

    expect(await screen.findByText('Image')).toBeInTheDocument()
  })
})


describe('SearchPanel hit expansion', () => {
  const okResult: SearchResult = {
    status: 'ok',
    hits: [HIT],
    total: 1,
    next_cursor: null,
    index_status: INDEX_STATUS
  }

  it('fetches and renders the full chunk on expand, and hides it again on collapse', async () => {
    const fetchMock = mockApi(okResult)

    renderPanel()

    // Nothing is fetched for a hit nobody opened.
    await screen.findByText(/alpha\.pdf/)
    expect(fetchMock.mock.calls.some(([u]) => String(u).includes('/search/chunk'))).toBe(false)

    await userEvent.click(screen.getByRole('button', { name: /show full chunk/i }))

    const body = await screen.findByTestId('hit-full-text')
    expect(body).toHaveTextContent(/Danach folgte die Aussprache/)
    const call = fetchMock.mock.calls.find(([u]) => String(u).includes('/search/chunk'))
    expect(String(call![0])).toContain('id=p1')

    await userEvent.click(screen.getByRole('button', { name: /hide full chunk/i }))

    expect(screen.queryByTestId('hit-full-text')).toBeNull()
    expect(screen.getByTestId('hit-preview')).toHaveTextContent(/Der Parteitag beschloss/)
    expect(screen.queryByText(/Danach folgte die Aussprache/)).toBeNull()
  })

  it('caches the fetched text, so re-expanding costs no second request', async () => {
    const fetchMock = mockApi(okResult)

    renderPanel()

    await screen.findByText(/alpha\.pdf/)
    await userEvent.click(screen.getByRole('button', { name: /show full chunk/i }))
    await screen.findByTestId('hit-full-text')
    await userEvent.click(screen.getByRole('button', { name: /hide full chunk/i }))
    await userEvent.click(screen.getByRole('button', { name: /show full chunk/i }))

    await screen.findByTestId('hit-full-text')
    const chunkCalls = fetchMock.mock.calls.filter(([u]) => String(u).includes('/search/chunk'))
    expect(chunkCalls).toHaveLength(1)
  })

  it('shows a loading state while the chunk text is in flight', async () => {
    let release: (() => void) | null = null
    const gate = new Promise<void>((resolve) => {
      release = () => resolve()
    })
    vi.stubGlobal(
      'fetch',
      vi.fn(async (req: RequestInfo | URL) => {
        const u = String(req)
        const body: unknown = u.includes('/search/chunk') ? { id: 'p1', text: FULL_TEXT } : okResult
        if (u.includes('/search/chunk')) await gate
        return {
          ok: true,
          status: 200,
          json: async () => body,
          text: async () => JSON.stringify(body)
        }
      })
    )

    renderPanel()

    await screen.findByText(/alpha\.pdf/)
    await userEvent.click(screen.getByRole('button', { name: /show full chunk/i }))

    expect(await screen.findByTestId('hit-loading')).toBeInTheDocument()
    release!()
    expect(await screen.findByTestId('hit-full-text')).toHaveTextContent(/Danach folgte/)
    expect(screen.queryByTestId('hit-loading')).toBeNull()
  })

  it('says the chunk no longer exists on a 404 — not that it is empty', async () => {
    // Re-ingestion mints new point ids, so a hit can outlive its chunk. An
    // empty body here would read as an empty chunk, which is a different claim.
    mockApi(okResult, { body: SCOPE_OK }, { body: { detail: 'Not found.' }, status: 404 })

    renderPanel()

    await screen.findByText(/alpha\.pdf/)
    await userEvent.click(screen.getByRole('button', { name: /show full chunk/i }))

    expect(await screen.findByTestId('hit-chunk-error')).toHaveTextContent(/no longer exists/i)
    // No empty expanded body pretending to be the chunk; the preview stands.
    expect(screen.queryByTestId('hit-full-text')).toBeNull()
    expect(screen.getByTestId('hit-preview')).toHaveTextContent(/Der Parteitag beschloss/)
    // The raw backend body is never rendered.
    expect(screen.queryByText(/Not found/)).toBeNull()
  })

  it('highlights the searched keywords in the expanded text too', async () => {
    mockApi(okResult)

    const { container } = renderPanel()

    await screen.findByText(/alpha\.pdf/)
    await userEvent.click(screen.getByRole('button', { name: /show full chunk/i }))
    await screen.findByTestId('hit-full-text')

    const marked = [...container.querySelectorAll('mark')].map((el) => el.textContent)
    expect(marked).toEqual(['Parteitag'])
  })

  it('leaves the selection alone — expanding and scoping are separate controls', async () => {
    const fetchMock = mockApi(okResult)

    renderPanel()

    await screen.findByText(/alpha\.pdf/)
    await userEvent.click(screen.getByRole('button', { name: /show full chunk/i }))
    await screen.findByTestId('hit-full-text')

    expect(screen.getByRole('button', { name: /alpha\.pdf/i })).toHaveAttribute(
      'aria-pressed',
      'false'
    )
    expect(useSearchUiStore.getState().scopes[SESSION]?.tokens ?? {}).toEqual({})
    expect(fetchMock.mock.calls.some(([u]) => String(u).includes('/sessions'))).toBe(false)
  })
})

describe('SearchPanel bulk selection', () => {
  const twoHits: SearchResult = {
    status: 'ok',
    hits: [HIT, HIT2],
    total: 42,
    next_cursor: 'next',
    index_status: INDEX_STATUS
  }

  it('projects what selecting all would cost before anything is committed', async () => {
    const fetchMock = mockApi(twoHits)

    renderPanel()

    await screen.findByText(/alpha\.pdf/)
    // 1200 + 1200 est_tokens, offered before the click rather than as a 422
    // after it. The control names the loaded slice, not the 42 matches behind
    // it — it is an icon now, so both the promise and the cost live in its
    // accessible name and tooltip.
    const selectAll = screen.getByRole('button', { name: /select all 2 loaded/i })
    expect(selectAll).toHaveAttribute(
      'title',
      expect.stringMatching(/not every match.*≈2\.4k tokens if selected/is)
    )
    expect(fetchMock.mock.calls.some(([u]) => String(u).includes('/scope'))).toBe(false)
  })

  it('adds every loaded hit to the scope', async () => {
    const fetchMock = mockApi(twoHits)

    renderPanel()

    await screen.findByText(/alpha\.pdf/)
    await userEvent.click(screen.getByRole('button', { name: /select all 2 loaded/i }))

    await waitFor(() => {
      const call = fetchMock.mock.calls.find(([u]) => String(u).includes('/scope'))
      expect(call).toBeDefined()
      expect(JSON.parse(String(call![1]!.body))).toEqual({ chunk_ids: ['p1', 'p2'] })
    })
    await waitFor(() => {
      expect(useSearchUiStore.getState().scopes[SESSION]?.tokens).toEqual({ p1: 1200, p2: 1200 })
    })
  })

  it('clears the whole selection in one go', async () => {
    useSearchUiStore.setState({
      scopes: { [SESSION]: { tokens: { p1: 1200, p2: 1200 }, usableTokens: 22000, missing: 0 } }
    })
    const fetchMock = mockApi(twoHits)

    renderPanel()

    await screen.findByText(/alpha\.pdf/)
    await userEvent.click(screen.getByRole('button', { name: /clear selection/i }))

    await waitFor(() => {
      expect(useSearchUiStore.getState().scopes[SESSION]?.tokens).toEqual({})
    })
    const call = fetchMock.mock.calls.find(([u]) => String(u).includes('/scope'))
    expect(JSON.parse(String(call![1]!.body))).toEqual({ chunk_ids: [] })
  })

  it('warns when selecting all would not fit the measured budget', async () => {
    useSearchUiStore.setState({
      scopes: { [SESSION]: { tokens: {}, usableTokens: 1000, missing: 0 } }
    })
    mockApi(twoHits)

    renderPanel()

    await screen.findByText(/alpha\.pdf/)
    expect(screen.getByTestId('select-all-over-budget')).toHaveTextContent(/would exceed/i)
  })

  it('rolls a refused select-all back, like a single tile', async () => {
    mockApi(twoHits, { body: { detail: 'Invalid request.' }, status: 422 })

    renderPanel()

    await screen.findByText(/alpha\.pdf/)
    await userEvent.click(screen.getByRole('button', { name: /select all 2 loaded/i }))

    expect(await screen.findByText(/larger than the answer can hold/i)).toBeInTheDocument()
    await waitFor(() => {
      expect(useSearchUiStore.getState().scopes[SESSION]?.tokens).toEqual({})
    })
    expect(screen.queryByText(/Invalid request/)).toBeNull()
  })

  it('keeps Clear reachable when a selection outlives a zero-hit search', async () => {
    // The row follows the selection, not the hits: chunks picked earlier are
    // still scoping the chat even though nothing on screen matches now.
    useSearchUiStore.setState({
      scopes: { [SESSION]: { tokens: { p1: 1200 }, usableTokens: 22000, missing: 0 } }
    })
    mockApi({
      status: 'ok',
      hits: [],
      total: 0,
      next_cursor: null,
      index_status: INDEX_STATUS
    })

    renderPanel()

    await screen.findByTestId('search-no-matches')
    await userEvent.click(screen.getByRole('button', { name: /clear selection/i }))

    await waitFor(() => {
      expect(useSearchUiStore.getState().scopes[SESSION]?.tokens).toEqual({})
    })
  })

  it('disables the bulk control when there is nothing loaded and nothing picked', async () => {
    mockApi({
      status: 'ok',
      hits: [],
      total: 0,
      next_cursor: null,
      index_status: INDEX_STATUS
    })

    renderPanel()

    await screen.findByTestId('search-no-matches')
    // Nothing to select and nothing to let go of — the one state where the
    // toggle has no direction to point in.
    expect(screen.getByRole('button', { name: /select all 0 loaded/i })).toBeDisabled()
  })

  it('flips to clearing once every loaded hit is picked, and back again', async () => {
    mockApi(twoHits)

    renderPanel()

    // Off: the control offers the selection.
    const select = await screen.findByRole('button', { name: /select all 2 loaded/i })
    expect(select).toHaveAttribute('aria-pressed', 'false')
    await userEvent.click(select)

    // On: the same control now offers to undo it. One button, two directions —
    // the whole point of collapsing the pair.
    const clear = await screen.findByRole('button', { name: /clear selection/i })
    expect(clear).toHaveAttribute('aria-pressed', 'true')
    await userEvent.click(clear)

    await waitFor(() => {
      expect(useSearchUiStore.getState().scopes[SESSION]?.tokens).toEqual({})
    })
    expect(await screen.findByRole('button', { name: /select all 2 loaded/i })).toBeInTheDocument()
  })

  it('still offers selecting when only some loaded hits are picked', async () => {
    // Half-selected is not selected: the toggle must not read as "on" and
    // strand the operator one click from losing what they picked.
    useSearchUiStore.setState({
      scopes: { [SESSION]: { tokens: { p1: 1200 }, usableTokens: 22000, missing: 0 } }
    })
    mockApi(twoHits)

    renderPanel()

    expect(await screen.findByRole('button', { name: /select all 2 loaded/i })).toBeInTheDocument()
  })
})

describe('SearchPanel scope', () => {
  it('carries no chat-retrieval settings — those belong to the chat', async () => {
    // The metadata filters and the retrieval mode narrow what any answer
    // retrieves against, not what the keyword index returns; they live beside
    // the Chat heading. Search owns the query and the hits, nothing else.
    mockApi({
      status: 'ok',
      hits: [HIT],
      total: 1,
      next_cursor: null,
      index_status: INDEX_STATUS
    })

    renderPanel()

    await screen.findByText(/alpha\.pdf/)
    expect(screen.queryByRole('button', { name: /filters/i })).toBeNull()
    expect(screen.queryByRole('button', { name: /retrieval/i })).toBeNull()
  })
})

describe('SearchPanel groups mode', () => {
  const hitsResult: SearchResult = {
    status: 'ok',
    hits: [HIT],
    total: 1,
    next_cursor: null,
    index_status: INDEX_STATUS
  }

  it('switching to Groups fetches the aggregate with the default group-by', async () => {
    const fetchMock = mockApi(hitsResult)

    renderPanel()

    await screen.findByText(/alpha\.pdf/)
    await userEvent.click(screen.getByRole('button', { name: 'Social' }))

    await waitFor(() => {
      const call = fetchMock.mock.calls.find(([u]) => String(u).includes('/search/aggregate'))
      expect(call).toBeDefined()
      expect(JSON.parse(String(call![1]!.body))).toMatchObject({ group_by: 'author' })
    })
  })

  it('choosing a different group-by field re-fetches the aggregate', async () => {
    const fetchMock = mockApi(hitsResult)

    renderPanel()

    await screen.findByText(/alpha\.pdf/)
    await userEvent.click(screen.getByRole('button', { name: 'Social' }))
    await waitFor(() => {
      expect(fetchMock.mock.calls.some(([u]) => String(u).includes('/search/aggregate'))).toBe(true)
    })

    const trigger = await screen.findByRole('combobox', { name: /group by/i })
    await userEvent.click(trigger)
    await userEvent.click(await screen.findByRole('option', { name: 'Network' }))

    await waitFor(() => {
      const call = fetchMock.mock.calls
        .filter(([u]) => String(u).includes('/search/aggregate'))
        .at(-1)
      expect(call).toBeDefined()
      expect(JSON.parse(String(call![1]!.body))).toMatchObject({ group_by: 'network' })
    })
  })

  it('shows the not_indexed banner in Groups mode too', async () => {
    mockApi(hitsResult, { body: SCOPE_OK }, CHUNK_OK, {
      body: {
        status: 'not_indexed',
        group_by: 'author',
        total: 0,
        unassigned: 0,
        groups: [],
        limit: 100,
        index_status: { ...INDEX_STATUS, with_search_text: 0, complete: false }
      }
    })

    renderPanel()

    await screen.findByText(/alpha\.pdf/)
    await userEvent.click(screen.getByRole('button', { name: 'Social' }))

    expect(await screen.findByTestId('search-not-indexed')).toBeInTheDocument()
    expect(screen.queryByTestId('search-no-groups')).toBeNull()
  })

  it('shows the capped notice once the group list reaches the effective limit', async () => {
    const capped: AggregateResult = {
      status: 'ok',
      group_by: 'author',
      total: 2,
      unassigned: 0,
      groups: [
        { value: 'acme_news', count: 1, samples: [] },
        { value: 'beta_daily', count: 1, samples: [] }
      ],
      limit: 2,
      index_status: INDEX_STATUS
    }
    mockApi(hitsResult, { body: SCOPE_OK }, CHUNK_OK, { body: capped })

    renderPanel()

    await screen.findByText(/alpha\.pdf/)
    await userEvent.click(screen.getByRole('button', { name: 'Social' }))

    const summary = await screen.findByTestId('search-summary')
    expect(summary.textContent).toMatch(/Showing the 2 largest results/)
  })

  it('omits the capped notice while the group list is under the effective limit', async () => {
    mockApi(hitsResult, { body: SCOPE_OK }, CHUNK_OK, { body: AGGREGATE_OK })

    renderPanel()

    await screen.findByText(/alpha\.pdf/)
    await userEvent.click(screen.getByRole('button', { name: 'Social' }))

    const summary = await screen.findByTestId('search-summary')
    expect(summary.textContent).not.toMatch(/largest results/)
  })

  it('shows a CSV export link with the current collection, query and group-by', async () => {
    mockApi(hitsResult, { body: SCOPE_OK }, CHUNK_OK, { body: AGGREGATE_OK })

    renderPanel()

    await screen.findByText(/alpha\.pdf/)
    await userEvent.click(screen.getByRole('button', { name: 'Social' }))

    const link = await screen.findByRole('link', { name: 'Export CSV' })
    const href = link.getAttribute('href') ?? ''
    expect(href).toContain('/search/export.csv')
    expect(href).toContain('collection=docs')
    expect(href).toContain('group_by=author')
    expect(href).toContain('question=Partei')
    // Nothing pinned yet, so the export is unmarked — only a live selection
    // adds marked_ids (see the Hits-mode export test below).
    expect(href).not.toContain('marked_ids')
  })

  it('shows a CSV export link in Hits mode with no group_by, gaining marked_ids once a hit is selected pre-session', async () => {
    // The same endpoint and href builder serve both lanes now; Hits must
    // never send group_by, and must pick up marked_ids reactively — but only
    // pre-session (no sessionId yet), which is the one case marked_ids earns
    // its place at all (see the with-a-session negative test below).
    mockApi({
      status: 'ok',
      hits: [HIT],
      total: 1,
      next_cursor: null,
      index_status: INDEX_STATUS
    })

    renderPanel(null)

    await screen.findByText(/alpha\.pdf/)

    const hrefBefore = screen.getByRole('link', { name: 'Export CSV' }).getAttribute('href') ?? ''
    expect(hrefBefore).toContain('/search/export.csv')
    expect(hrefBefore).toContain('collection=docs')
    expect(hrefBefore).toContain('question=Partei')
    expect(hrefBefore).not.toContain('group_by')
    expect(hrefBefore).not.toContain('marked_ids')

    await userEvent.click(await screen.findByRole('button', { name: /alpha\.pdf/i }))

    await waitFor(() => {
      const href = screen.getByRole('link', { name: 'Export CSV' }).getAttribute('href') ?? ''
      expect(href).toContain('marked_ids=p1')
    })
  })

  it('omits marked_ids from the CSV export once a session exists, even with a live selection', async () => {
    // Once a session id exists, commitScope has already written the same
    // selection server-side by the time this link can be clicked, so
    // marked_ids is redundant — and a scope has no count cap, only a token
    // budget, so a large selection serialized into the URL could overflow
    // the gateway's header limit. The link must fall back to the stored
    // session scope (session_id alone) instead.
    const fetchMock = mockApi({
      status: 'ok',
      hits: [HIT],
      total: 1,
      next_cursor: null,
      index_status: INDEX_STATUS
    })

    renderPanel(SESSION)

    await userEvent.click(await screen.findByRole('button', { name: /alpha\.pdf/i }))
    await waitFor(() => {
      expect(fetchMock.mock.calls.some(([u]) => String(u).includes('/scope'))).toBe(true)
    })

    const href = screen.getByRole('link', { name: 'Export CSV' }).getAttribute('href') ?? ''
    expect(href).toContain('session_id=' + SESSION)
    expect(href).not.toContain('marked_ids')
  })

  it('omits the CSV export link when there are no groups', async () => {
    mockApi(hitsResult, { body: SCOPE_OK }, CHUNK_OK, {
      body: { ...AGGREGATE_OK, groups: [] }
    })

    renderPanel()

    await screen.findByText(/alpha\.pdf/)
    await userEvent.click(screen.getByRole('button', { name: 'Social' }))

    await screen.findByTestId('search-summary')
    expect(screen.queryByRole('link', { name: 'Export CSV' })).toBeNull()
  })

  it('omits the CSV export link in Hits mode when the query matches nothing', async () => {
    // Mirrors the Groups-mode case above: a zero-row result set is not worth
    // a CSV, in either mode.
    mockApi({
      status: 'ok',
      hits: [],
      total: 0,
      next_cursor: null,
      index_status: INDEX_STATUS
    })

    renderPanel()

    await screen.findByTestId('search-no-matches')
    expect(screen.queryByRole('link', { name: 'Export CSV' })).toBeNull()
  })

  it('omits the CSV export link in Hits mode while the query is unsubmitted', async () => {
    // Hits with a blank query and hits present can't co-occur (useSearch
    // stays disabled until the query is non-empty), so this pins the
    // query.trim() half of the gate via a selection that is already live —
    // the summary row still renders, but with no query submitted there is
    // nothing to export.
    // A keyword-less hits export would 422 the same way a keyword-less
    // search does, so the link is hidden rather than offered and refused.
    useSearchUiStore.setState({
      queries: {},
      scopes: { [SESSION]: { tokens: { p1: 1200 }, usableTokens: 22000, missing: 0 } }
    })
    mockApi({
      status: 'ok',
      hits: [],
      total: 0,
      next_cursor: null,
      index_status: INDEX_STATUS
    })

    renderPanel()

    await screen.findByTestId('search-summary-row')
    expect(screen.queryByRole('link', { name: 'Export CSV' })).toBeNull()
  })

  it("pins one of a group's sample chunks into the scope", async () => {
    const fetchMock = mockApi(hitsResult, { body: SCOPE_OK }, CHUNK_OK, { body: AGGREGATE_OK })

    renderPanel()

    await screen.findByText(/alpha\.pdf/)
    await userEvent.click(screen.getByRole('button', { name: 'Social' }))

    const disclosure = await screen.findByRole('button', { name: /show sample chunks/i })
    await userEvent.click(disclosure)

    const tile = await screen.findByRole('button', { name: /alpha\.pdf/i })
    await userEvent.click(tile)

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

  it('shows the token meter inline in Social mode once a sample is pinned, beside a working mark-all', async () => {
    // The summary row is unified across modes now: Social gets the same
    // mark-all control Hits has, sized over whatever aggregate samples are
    // loaded, and the token meter renders inline in the same row either way
    // — as its own flex child beside the counts `<p>`, not text inside it
    // (see the search-summary-row layout comment in SearchPanel.tsx).
    mockApi(hitsResult, { body: SCOPE_OK }, CHUNK_OK, { body: AGGREGATE_OK })

    renderPanel()

    await screen.findByText(/alpha\.pdf/)
    await userEvent.click(screen.getByRole('button', { name: 'Social' }))

    const disclosure = await screen.findByRole('button', { name: /show sample chunks/i })
    await userEvent.click(disclosure)

    const tile = await screen.findByRole('button', { name: /alpha\.pdf/i })
    await userEvent.click(tile)

    await waitFor(() => {
      const meter = screen.getByTestId('token-meter')
      expect(meter).toHaveTextContent('≈1.2k / 22.0k tokens')
      expect(screen.getByTestId('search-summary-row')).toContainElement(meter)
    })
    // The one loaded sample is now selected, so the bulk control flips to
    // "clear" — Social's mark-all behaves exactly like Hits' does.
    expect(screen.getByRole('button', { name: /clear selection/i })).toBeInTheDocument()
  })

  it('mark-all in Social pins every loaded sample across groups, not just the first', async () => {
    const fetchMock = mockApi(hitsResult, { body: SCOPE_OK }, CHUNK_OK, { body: AGGREGATE_TWO })

    renderPanel()

    await screen.findByText(/alpha\.pdf/)
    await userEvent.click(screen.getByRole('button', { name: 'Social' }))

    const selectAll = await screen.findByRole('button', { name: /select all 2 loaded/i })
    await userEvent.click(selectAll)

    await waitFor(() => {
      const call = fetchMock.mock.calls.find(([u]) => String(u).includes('/scope'))
      expect(call).toBeDefined()
      expect(JSON.parse(String(call![1]!.body))).toEqual({ chunk_ids: ['p1', 'p2'] })
    })
    await waitFor(() => {
      expect(useSearchUiStore.getState().scopes[SESSION]?.tokens).toEqual({ p1: 1200, p2: 1200 })
    })
  })

  it('warns in Social mode too when the loaded samples exceed the measured budget', async () => {
    useSearchUiStore.setState({
      scopes: { [SESSION]: { tokens: {}, usableTokens: 1000, missing: 0 } }
    })
    mockApi(hitsResult, { body: SCOPE_OK }, CHUNK_OK, { body: AGGREGATE_TWO })

    renderPanel()

    await screen.findByText(/alpha\.pdf/)
    await userEvent.click(screen.getByRole('button', { name: 'Social' }))

    expect(await screen.findByTestId('select-all-over-budget')).toHaveTextContent(/would exceed/i)
  })

  it('keeps one summary row present across both modes', async () => {
    mockApi(hitsResult, { body: SCOPE_OK }, CHUNK_OK, { body: AGGREGATE_OK })

    renderPanel()

    await screen.findByText(/alpha\.pdf/)
    expect(screen.getByTestId('search-summary-row')).toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: 'Social' }))

    expect(await screen.findByTestId('search-summary-row')).toBeInTheDocument()
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
