import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, useLocation } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Sidebar } from './Sidebar'
import { useUiStore } from '@/stores/ui'
import { useChatUiStore } from '@/stores/chatUi'
import { useIngestJobsStore } from '@/stores/ingestJobs'

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
  useIngestJobsStore.getState().clear()
  useChatUiStore.setState({ drafts: {} })
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
      expect(calls.some((u) => u.includes('/collections/list'))).toBe(true)
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

  it('rolls back to the previous selection and session when /collections/select fails', async () => {
    const fetchMock = vi.fn().mockImplementation(async (input: RequestInfo | URL) => {
      const path = typeof input === 'string' ? input : input.toString()
      if (path.includes('/collections/select')) {
        return { ok: false, status: 404, json: async () => ({ detail: 'not found' }), text: async () => '{}' }
      }
      if (path.includes('/collections/list')) {
        return { ok: true, status: 200, json: async () => ['alpha', 'beta'], text: async () => '["alpha","beta"]' }
      }
      if (path.includes('/sessions/list')) {
        return { ok: true, status: 200, json: async () => ({ sessions: [] }), text: async () => '{"sessions":[]}' }
      }
      return { ok: true, status: 200, json: async () => null, text: async () => 'null' }
    })
    vi.stubGlobal('fetch', fetchMock)
    useUiStore.setState({ selectedCollection: 'alpha', selectedOwner: null, currentSessionId: 'sess-old' })

    renderSidebar()

    const select = await screen.findByLabelText(/select collection/i)
    await screen.findByRole('option', { name: 'beta' })
    await userEvent.selectOptions(select, 'beta')

    await waitFor(() => {
      const selectCall = fetchMock.mock.calls.find((c) => String(c[0]).includes('/collections/select'))
      expect(selectCall).toBeDefined()
    })
    await waitFor(() => {
      expect(useUiStore.getState().selectedCollection).toBe('alpha')
    })
    expect(useUiStore.getState().currentSessionId).toBe('sess-old')
  })

  it('surfaces an error and keeps the selection when deleting a collection fails', async () => {
    const fetchMock = vi.fn().mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = typeof input === 'string' ? input : input.toString()
      if (init?.method === 'DELETE' && path.includes('/collections/alpha')) {
        return {
          ok: false,
          status: 500,
          json: async () => ({ detail: 'Request failed.' }),
          text: async () => '{"detail":"Request failed."}'
        }
      }
      if (path.includes('/collections/list')) {
        return { ok: true, status: 200, json: async () => ['alpha'], text: async () => '["alpha"]' }
      }
      if (path.includes('/sessions/list')) {
        return { ok: true, status: 200, json: async () => ({ sessions: [] }), text: async () => '{"sessions":[]}' }
      }
      return { ok: true, status: 200, json: async () => null, text: async () => 'null' }
    })
    vi.stubGlobal('fetch', fetchMock)
    vi.stubGlobal('confirm', () => true)
    useUiStore.setState({ selectedCollection: 'alpha', currentSessionId: 'sess-old' })

    renderSidebar()

    const del = await screen.findByLabelText(/delete collection alpha/i)
    await userEvent.click(del)

    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent(/alpha/)
    expect(alert).toHaveTextContent(/could not be deleted|konnte nicht gelöscht/i)
    // The collection still exists server-side — the selection must survive.
    expect(useUiStore.getState().selectedCollection).toBe('alpha')
    expect(useUiStore.getState().currentSessionId).toBe('sess-old')
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

describe('Sidebar chat draft pruning', () => {
  it('prunes the chat draft when a session is deleted', async () => {
    const fetchMock = mockFetch({
      '/collections/list': ['alpha'],
      '/sessions/list': {
        sessions: [{ id: 's1', created_at: '2026-01-01', title: 'First chat', collection: 'alpha' }]
      },
      '/sessions/s1': { ok: true }
    })
    vi.stubGlobal('fetch', fetchMock)
    vi.stubGlobal('confirm', () => true)
    useUiStore.setState({ selectedCollection: 'alpha' })
    useChatUiStore.getState().setDraft('s1', 'half typed question')

    renderSidebar()

    const del = await screen.findByLabelText(/delete session/i)
    await userEvent.click(del)

    await waitFor(() => {
      expect(useChatUiStore.getState().drafts['s1']).toBeUndefined()
    })
  })

  it('prunes the drafts of a deleted collection\'s sessions', async () => {
    const fetchMock = mockFetch({
      '/collections/list': ['alpha'],
      '/sessions/list': {
        sessions: [
          { id: 's1', created_at: '2026-01-01', title: 'First chat', collection: 'alpha' },
          { id: 's2', created_at: '2026-01-02', title: 'Second chat', collection: 'alpha' }
        ]
      }
    })
    vi.stubGlobal('fetch', fetchMock)
    vi.stubGlobal('confirm', () => true)
    useUiStore.setState({ selectedCollection: 'alpha', currentSessionId: 's1' })
    useChatUiStore.setState({
      drafts: { s1: 'half typed question', s2: 'another draft' }
    })

    renderSidebar()
    await screen.findByText('First chat')

    const del = await screen.findByLabelText(/delete collection alpha/i)
    await userEvent.click(del)

    await waitFor(() => {
      expect(useChatUiStore.getState().drafts['s1']).toBeUndefined()
      expect(useChatUiStore.getState().drafts['s2']).toBeUndefined()
    })
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
    // The router applies the navigation on its own render tick, after the
    // store update above — poll for it instead of asserting synchronously.
    await waitFor(() => {
      expect(screen.getByTestId('location-probe').textContent).toBe('/chat')
    })
  })
})

describe('Sidebar new chat', () => {
  it('drops the open session and lands in chat from any section', async () => {
    const fetchMock = mockFetch({
      '/collections/list': ['alpha'],
      '/sessions/list': { sessions: [{ id: 's-1', title: 'Prior chat' }] }
    })
    vi.stubGlobal('fetch', fetchMock)
    useUiStore.setState({ selectedCollection: 'alpha', currentSessionId: 's-1' })

    renderSidebarAt('/analysis')

    await userEvent.click(await screen.findByRole('button', { name: /new chat/i }))

    expect(useUiStore.getState().currentSessionId).toBeNull()
    await waitFor(() => {
      expect(screen.getByTestId('location-probe').textContent).toBe('/chat')
    })
  })

  it('draws the control rather than spelling it, so it renders the same everywhere', async () => {
    const fetchMock = mockFetch({
      '/collections/list': [],
      '/sessions/list': { sessions: [] }
    })
    vi.stubGlobal('fetch', fetchMock)

    renderSidebar()

    // A "+" typed as text renders from whatever font the OS falls back to, and
    // in a control carrying no text of its own that drawing is the affordance.
    const button = await screen.findByRole('button', { name: /new chat/i })
    expect(button.querySelector('svg')).toBeInTheDocument()
    expect(button).toHaveTextContent('')
  })
})

describe('Sidebar layout', () => {
  it('leads with the collection every section below it is scoped to', async () => {
    const fetchMock = mockFetch({
      '/collections/list': ['alpha'],
      '/sessions/list': { sessions: [] }
    })
    vi.stubGlobal('fetch', fetchMock)
    useUiStore.setState({ selectedCollection: 'alpha' })

    renderSidebar()

    const collection = await screen.findByTestId('active-collection')
    const nav = document.querySelector('nav')!
    // Node.DOCUMENT_POSITION_FOLLOWING — the nav comes after the collection.
    expect(collection.compareDocumentPosition(nav) & 4).toBeTruthy()
  })

  it('gives the session list the one heading, since its rows are drawn like the nav rows', async () => {
    const fetchMock = mockFetch({
      '/collections/list': ['alpha'],
      '/sessions/list': { sessions: [] }
    })
    vi.stubGlobal('fetch', fetchMock)
    useUiStore.setState({ selectedCollection: 'alpha' })

    renderSidebar()

    expect(await screen.findByText(/sessions/i)).toBeInTheDocument()
    // The collection leads the panel and names itself; a heading over it only
    // repeated the row beneath, which the nav above manages without.
    expect(screen.queryByText(/^collection$/i)).not.toBeInTheDocument()
  })
})

describe('Sidebar navigation', () => {
  it('orders sections dashboard, ingest, inspector, chat, analysis, report', () => {
    const fetchMock = mockFetch({
      '/collections/list': [],
      '/sessions/list': { sessions: [] }
    })
    vi.stubGlobal('fetch', fetchMock)

    renderSidebar()
    // Assert the full <nav> contents (not just a prefix slice) so a 7th nav
    // entry appended later would fail this test instead of passing silently.
    const hrefs = Array.from(document.querySelectorAll('nav a')).map((a) => a.getAttribute('href'))
    expect(hrefs).toEqual(['/', '/ingest', '/inspector', '/chat', '/analysis', '/report'])
  })
})

describe('Sidebar chat nav link', () => {
  it('points the Chat nav at the open session', async () => {
    const fetchMock = mockFetch({
      '/collections/list': [],
      '/sessions/list': { sessions: [] }
    })
    vi.stubGlobal('fetch', fetchMock)
    useUiStore.setState({ currentSessionId: 's-1' })

    renderSidebar()

    expect(screen.getByRole('link', { name: /chat/i })).toHaveAttribute(
      'href',
      expect.stringContaining('/chat/s-1')
    )
  })

  it('points the Chat nav at a fresh chat when no session is open', async () => {
    const fetchMock = mockFetch({
      '/collections/list': [],
      '/sessions/list': { sessions: [] }
    })
    vi.stubGlobal('fetch', fetchMock)
    useUiStore.setState({ currentSessionId: null })

    renderSidebar()

    const href = screen.getByRole('link', { name: /chat/i }).getAttribute('href')
    expect(href).toMatch(/\/chat$/)
  })
})

describe('Sidebar ingest job badge', () => {
  it('badges the Ingest nav entry while a job is running', async () => {
    const fetchMock = mockFetch({
      '/collections/list': [],
      '/sessions/list': { sessions: [] }
    })
    vi.stubGlobal('fetch', fetchMock)

    useIngestJobsStore.getState().appendEvent('job-1', {
      event: 'ingestion_started',
      data: { job_id: 'job-1', collection: 'mydocs' },
      receivedAt: Date.now()
    })

    renderSidebar()
    expect(await screen.findByLabelText(/ingestion running|verarbeitung läuft/i)).toBeInTheDocument()
  })

  it('drops the badge once the job completes', async () => {
    const fetchMock = mockFetch({
      '/collections/list': [],
      '/sessions/list': { sessions: [] }
    })
    vi.stubGlobal('fetch', fetchMock)

    const { appendEvent } = useIngestJobsStore.getState()
    appendEvent('job-1', {
      event: 'ingestion_started',
      data: { job_id: 'job-1', collection: 'mydocs' },
      receivedAt: Date.now()
    })
    appendEvent('job-1', {
      event: 'ingestion_complete',
      data: { job_id: 'job-1', collection: 'mydocs' },
      receivedAt: Date.now()
    })

    renderSidebar()
    await waitFor(() => {
      const calls = fetchMock.mock.calls.map((c) => String(c[0]))
      expect(calls.some((u) => u.includes('/collections/list'))).toBe(true)
    })
    expect(screen.queryByLabelText(/ingestion running|verarbeitung läuft/i)).not.toBeInTheDocument()
  })
})
