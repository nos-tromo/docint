import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, useLocation } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Sidebar } from './Sidebar'
import { useUiStore } from '@/stores/ui'

function mockFetch(map: Record<string, unknown>) {
  return vi.fn().mockImplementation(async (input: RequestInfo | URL) => {
    const path = typeof input === 'string' ? input : input.toString()
    for (const [pattern, body] of Object.entries(map)) {
      if (path.includes(pattern)) {
        return {
          ok: true,
          status: 200,
          json: async () => body,
          text: async () => JSON.stringify(body)
        }
      }
    }
    return { ok: true, status: 200, json: async () => null, text: async () => 'null' }
  })
}

function renderSidebar() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter>
        <Sidebar />
      </MemoryRouter>
    </QueryClientProvider>
  )
}

function LocationProbe() {
  const location = useLocation()
  return <div data-testid="location-probe">{location.pathname}</div>
}

function renderSidebarAt(initialPath: string) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={[initialPath]}>
        <Sidebar />
        <LocationProbe />
      </MemoryRouter>
    </QueryClientProvider>
  )
}

beforeEach(() => {
  useUiStore.setState({ selectedCollection: null, currentSessionId: null, previewModal: null })
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('Sidebar collection selection', () => {
  it('does not auto-select on mount even when a collection exists', async () => {
    const fetchMock = mockFetch({
      '/collections/list': ['kept', 'other'],
      '/sessions/list': { sessions: [] }
    })
    vi.stubGlobal('fetch', fetchMock)

    renderSidebar()

    await waitFor(() => {
      const calls = fetchMock.mock.calls.map((c) => String(c[0]))
      expect(calls.some((u) => u.endsWith('/collections/list'))).toBe(true)
    })
    const calls = fetchMock.mock.calls.map((c) => String(c[0]))
    expect(calls.some((u) => u.endsWith('/collections/select'))).toBe(false)
    expect(useUiStore.getState().selectedCollection).toBeNull()
  })

  it('shows the no-active-collection hint when nothing is selected', async () => {
    const fetchMock = mockFetch({
      '/collections/list': ['a'],
      '/sessions/list': { sessions: [] }
    })
    vi.stubGlobal('fetch', fetchMock)

    renderSidebar()

    expect(screen.getByText(/no active collection/i)).toBeInTheDocument()
    expect(screen.queryByTestId('active-collection')).not.toBeInTheDocument()
  })

  it('posts to /collections/select and shows the Active badge after picking one', async () => {
    const fetchMock = mockFetch({
      '/collections/list': ['alpha', 'beta'],
      '/sessions/list': { sessions: [] },
      '/collections/select': { ok: true, name: 'alpha' }
    })
    vi.stubGlobal('fetch', fetchMock)

    renderSidebar()

    const select = await screen.findByLabelText(/select collection/i)
    await screen.findByRole('option', { name: 'alpha' })
    await userEvent.selectOptions(select, 'alpha')

    await waitFor(() => {
      expect(useUiStore.getState().selectedCollection).toBe('alpha')
    })
    const selectCall = fetchMock.mock.calls.find((c) =>
      String(c[0]).endsWith('/collections/select')
    )!
    expect(JSON.parse(selectCall[1].body)).toEqual({ name: 'alpha' })

    const badge = await screen.findByTestId('active-collection')
    expect(badge).toHaveTextContent(/active/i)
    expect(badge).toHaveTextContent('alpha')
  })

  it('shows an actionable error when the sessions list requires a principal', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation(async (input: RequestInfo | URL) => {
        const path = typeof input === 'string' ? input : input.toString()
        if (path.includes('/collections/list')) {
          return {
            ok: true,
            status: 200,
            json: async () => ['alpha'],
            text: async () => '["alpha"]'
          }
        }
        if (path.includes('/sessions/list')) {
          return {
            ok: false,
            status: 401,
            json: async () => ({ detail: 'Missing authenticated principal.' }),
            text: async () => '{"detail":"Missing authenticated principal."}'
          }
        }
        return { ok: true, status: 200, json: async () => null, text: async () => 'null' }
      })
    )
    useUiStore.setState({ selectedCollection: 'alpha' })

    renderSidebar()

    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent(/DOCINT_DEFAULT_IDENTITY/i)
    expect(alert).toHaveTextContent(/authenticated user/i)
  })

  it("lists the active collection's sessions and scopes the request", async () => {
    const fetchMock = mockFetch({
      '/collections/list': ['alpha'],
      '/sessions/list': {
        sessions: [{ id: 's1', created_at: '2026-01-01', title: 'First chat', collection: 'alpha' }]
      }
    })
    vi.stubGlobal('fetch', fetchMock)
    useUiStore.setState({ selectedCollection: 'alpha' })

    renderSidebar()

    expect(await screen.findByText('First chat')).toBeInTheDocument()
    await waitFor(() => {
      const call = fetchMock.mock.calls
        .map((c) => String(c[0]))
        .find((u) => u.includes('/sessions/list'))
      expect(call).toContain('collection=alpha')
    })
  })

  it('prompts to select a collection when none is active and skips the sessions fetch', async () => {
    const fetchMock = mockFetch({ '/collections/list': ['alpha'] })
    vi.stubGlobal('fetch', fetchMock)

    renderSidebar()

    expect(await screen.findByText(/select a collection to see its chats/i)).toBeInTheDocument()
    const calls = fetchMock.mock.calls.map((c) => String(c[0]))
    expect(calls.some((u) => u.includes('/sessions/list'))).toBe(false)
  })

  it('clears the open chat when switching collections', async () => {
    const fetchMock = mockFetch({
      '/collections/list': ['alpha', 'beta'],
      '/sessions/list': { sessions: [] },
      '/collections/select': { ok: true, name: 'beta' }
    })
    vi.stubGlobal('fetch', fetchMock)
    useUiStore.setState({ selectedCollection: 'alpha', currentSessionId: 'sess-old' })

    renderSidebar()

    const select = await screen.findByLabelText(/select collection/i)
    await screen.findByRole('option', { name: 'beta' })
    await userEvent.selectOptions(select, 'beta')

    await waitFor(() => {
      expect(useUiStore.getState().selectedCollection).toBe('beta')
    })
    expect(useUiStore.getState().currentSessionId).toBeNull()
  })

  it('clears selection and the open chat after deleting the active collection', async () => {
    const fetchMock = mockFetch({
      '/collections/list': ['alpha'],
      '/sessions/list': { sessions: [] }
    })
    vi.stubGlobal('fetch', fetchMock)
    vi.stubGlobal('confirm', () => true)
    useUiStore.setState({ selectedCollection: 'alpha', currentSessionId: 'sess-old' })

    renderSidebar()

    const del = await screen.findByLabelText(/delete collection alpha/i)
    await userEvent.click(del)

    await waitFor(() => {
      expect(useUiStore.getState().selectedCollection).toBeNull()
    })
    expect(useUiStore.getState().currentSessionId).toBeNull()
  })
})

describe('Sidebar keeps the current section when switching collections', () => {
  it('stays on the current section instead of jumping to chat', async () => {
    const fetchMock = mockFetch({
      '/collections/list': ['alpha', 'beta'],
      '/sessions/list': { sessions: [] },
      '/collections/select': { ok: true, name: 'beta' }
    })
    vi.stubGlobal('fetch', fetchMock)
    useUiStore.setState({ selectedCollection: 'alpha' })

    renderSidebarAt('/analysis')

    const select = await screen.findByLabelText(/select collection/i)
    await screen.findByRole('option', { name: 'beta' })
    await userEvent.selectOptions(select, 'beta')

    await waitFor(() => {
      expect(useUiStore.getState().selectedCollection).toBe('beta')
    })
    expect(screen.getByTestId('location-probe').textContent).toBe('/analysis')
  })

  it('drops to a fresh chat when switching collections while viewing a pinned session', async () => {
    const fetchMock = mockFetch({
      '/collections/list': ['alpha', 'beta'],
      '/sessions/list': { sessions: [] },
      '/collections/select': { ok: true, name: 'beta' }
    })
    vi.stubGlobal('fetch', fetchMock)
    useUiStore.setState({ selectedCollection: 'alpha', currentSessionId: 'sess-old' })

    renderSidebarAt('/chat/sess-old')

    const select = await screen.findByLabelText(/select collection/i)
    await screen.findByRole('option', { name: 'beta' })
    await userEvent.selectOptions(select, 'beta')

    await waitFor(() => {
      expect(useUiStore.getState().selectedCollection).toBe('beta')
    })
    // A session is pinned to the collection it was created under, so the stale
    // session sub-route is dropped — but the user stays within the chat section.
    expect(screen.getByTestId('location-probe').textContent).toBe('/chat')
  })
})
