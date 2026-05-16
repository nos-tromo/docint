import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
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
})
