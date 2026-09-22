import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, waitFor, within } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Dashboard } from './Dashboard'
import { useUiStore } from '@/stores/ui'

function renderDashboard() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={qc}>
      <Dashboard />
    </QueryClientProvider>
  )
}

// Only /collections/list drives the Backend indicator (useCollections.isError);
// every other dashboard query gets a benign empty payload.
function mockFetch(collectionsReachable: boolean) {
  return vi.fn(async (input: RequestInfo | URL) => {
    const path = typeof input === 'string' ? input : input.toString()
    if (path.includes('/collections/list')) {
      return collectionsReachable
        ? { ok: true, status: 200, json: async () => [] }
        : { ok: false, status: 500, json: async () => ({}), text: async () => '' }
    }
    return {
      ok: true,
      status: 200,
      json: async () => ({ sessions: [], documents: [], top_entities: [] })
    }
  })
}

beforeEach(() => {
  useUiStore.setState({ selectedCollection: null })
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('Dashboard backend status indicator', () => {
  it('shows a green dot when the backend is reachable', async () => {
    vi.stubGlobal('fetch', mockFetch(true))
    renderDashboard()

    const dot = await screen.findByTestId('backend-status-dot')
    expect(dot).toHaveClass('bg-primary')
    expect(screen.getByText('online')).toBeInTheDocument()
  })

  it('shows a red dot when the backend is unreachable', async () => {
    vi.stubGlobal('fetch', mockFetch(false))
    renderDashboard()

    await waitFor(() => {
      expect(screen.getByTestId('backend-status-dot')).toHaveClass('bg-red-400')
    })
    expect(screen.getByText('offline')).toBeInTheDocument()
  })
})

describe('Dashboard entity merge mode is fixed to resolved', () => {
  it('does not render a merge-mode control', async () => {
    useUiStore.setState({ selectedCollection: 'alpha' })
    vi.stubGlobal('fetch', mockFetch(true))
    renderDashboard()

    await screen.findByTestId('backend-status-dot')
    expect(screen.queryByRole('group', { name: /merge/i })).not.toBeInTheDocument()
  })
})

describe('Dashboard sessions panel scoping', () => {
  it('prompts to select a collection in the recent-sessions panel when none is active', async () => {
    vi.stubGlobal('fetch', mockFetch(true))
    renderDashboard()

    expect(
      await screen.findByText(/select a collection to see its chats/i)
    ).toBeInTheDocument()
  })

  it("lists the active collection's recent sessions without the physical-name label", async () => {
    useUiStore.setState({ selectedCollection: 'alpha' })
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const path = typeof input === 'string' ? input : input.toString()
        if (path.includes('/collections/list')) {
          return { ok: true, status: 200, json: async () => ['alpha'] }
        }
        if (path.includes('/sessions/list')) {
          return {
            ok: true,
            status: 200,
            json: async () => ({
              sessions: [
                { id: 'sess-1', title: 'First chat', created_at: '2026-01-01', collection: 'u123__alpha' }
              ]
            })
          }
        }
        return { ok: true, status: 200, json: async () => ({ documents: [], top_entities: [] }) }
      })
    )
    renderDashboard()

    expect(await screen.findByText('First chat')).toBeInTheDocument()
    // The panel is scoped to the active collection, so no per-row collection
    // label — and never the owner-namespaced physical name.
    expect(screen.queryByText('u123__alpha')).not.toBeInTheDocument()
    expect(
      screen.queryByText(/select a collection to see its chats/i)
    ).not.toBeInTheDocument()
  })
})

interface RetentionRow {
  name: string
  owner: string | null
  last_activity_at: string | null
  expires_at: string | null
  warning: boolean
}

function row(name: string, expires_at: string | null, warning = false, owner: string | null = null): RetentionRow {
  return { name, owner, last_activity_at: '2026-03-02T09:14:00Z', expires_at, warning }
}

/** A backend with retention set to `window` whose deadline listing is `rows`. */
function retentionFetch(window: string, rows: RetentionRow[]) {
  return vi.fn(async (input: RequestInfo | URL) => {
    const path = typeof input === 'string' ? input : input.toString()
    const body = path.includes('/collections/retention')
      ? { window, collections: rows }
      : path.includes('/collections/list')
        ? ['projekt-alpha']
        : path.includes('/config')
          ? { language: 'en', collection_retention: window }
          : { sessions: [], documents: [], top_entities: [] }
    return { ok: true, status: 200, json: async () => body }
  })
}

describe('Dashboard automatic deletion card', () => {
  it('stays hidden, and asks for no deadlines, while retention is off', async () => {
    const fetchMock = retentionFetch('off', [row('projekt-alpha', null)])
    vi.stubGlobal('fetch', fetchMock)
    renderDashboard()

    await screen.findByTestId('backend-status-dot')
    await waitFor(() => expect(fetchMock.mock.calls.some((c) => String(c[0]).includes('/config'))).toBe(true))
    expect(screen.queryByRole('heading', { name: /automatic deletion/i })).not.toBeInTheDocument()
    expect(fetchMock.mock.calls.some((c) => String(c[0]).includes('/collections/retention'))).toBe(false)
  })

  it('lists each collection with its deletion date and flags the imminent ones', async () => {
    vi.stubGlobal(
      'fetch',
      retentionFetch('6m', [row('projekt-alpha', '2026-10-21T06:45:00Z', true), row('laufend', '2027-03-19T08:00:00Z')])
    )
    renderDashboard()

    const card = await screen.findByRole('region', { name: /automatic deletion/i })
    expect(card).toHaveTextContent(/6 months without activity/i)
    const items = within(card).getAllByRole('listitem')
    expect(items.map((li) => li.textContent)).toEqual([
      expect.stringContaining('projekt-alpha'),
      expect.stringContaining('laufend')
    ])
    expect(items[0]).toHaveTextContent('Oct 21, 2026')
    expect(within(items[0]).getByRole('img', { name: /within 30 days/i })).toBeInTheDocument()
    expect(within(items[1]).queryByRole('img')).not.toBeInTheDocument()
  })

  it("names the owner of another user's collection", async () => {
    vi.stubGlobal('fetch', retentionFetch('12m', [row('shared-notes', '2027-01-05T10:00:00Z', false, 'a.beispiel')]))
    renderDashboard()

    expect(await screen.findByText(/shared-notes \(owner: a\.beispiel\)/)).toBeInTheDocument()
  })

  it('says so when there is no collection yet', async () => {
    vi.stubGlobal('fetch', retentionFetch('6m', []))
    renderDashboard()

    const card = await screen.findByRole('region', { name: /automatic deletion/i })
    expect(card).toHaveTextContent(/no collections yet/i)
    expect(within(card).queryByRole('list')).not.toBeInTheDocument()
  })

  it('says when a collection is not scheduled', async () => {
    vi.stubGlobal('fetch', retentionFetch('6m', [row('ohne-aktivitaet', null)]))
    renderDashboard()

    expect(await screen.findByText(/not scheduled/i)).toBeInTheDocument()
  })

  it('shows the soonest few and counts the rest', async () => {
    const rows = Array.from({ length: 10 }, (_, i) => row(`sammlung-${i}`, `2027-0${(i % 9) + 1}-01T00:00:00Z`))
    vi.stubGlobal('fetch', retentionFetch('6m', rows))
    renderDashboard()

    expect(await screen.findByText('sammlung-7')).toBeInTheDocument()
    expect(screen.queryByText('sammlung-8')).not.toBeInTheDocument()
    expect(screen.getByText(/2 more/)).toBeInTheDocument()
  })
})
