import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
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
