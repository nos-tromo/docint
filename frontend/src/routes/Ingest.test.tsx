import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Ingest } from './Ingest'
import { Sidebar } from '@/layout/Sidebar'
import { useUiStore } from '@/stores/ui'

// The ingest stream is mocked to "create" the new collection: consuming it
// flips the shared collections list to include `gamma`, then emits the terminal
// event — mirroring the backend, where the collection exists only after ingest.
const h = vi.hoisted(() => ({ collections: [] as string[] }))
vi.mock('@/api/ingest', () => ({
  streamIngestUploadBatched: () =>
    (async function* () {
      h.collections = ['gamma']
      yield { event: 'ingestion_complete', data: {} }
    })()
}))

function jsonRes(body: unknown) {
  return { ok: true, status: 200, json: async () => body, text: async () => JSON.stringify(body) }
}

beforeEach(() => {
  h.collections = []
  useUiStore.setState({ selectedCollection: null, currentSessionId: null, previewModal: null })
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const u = typeof input === 'string' ? input : input.toString()
      if (u.includes('/collections/list')) return jsonRes(h.collections)
      if (u.includes('/collections/select')) return jsonRes({ ok: true, name: 'gamma' })
      if (u.includes('/config'))
        return jsonRes({ graph_top_k: 0, graph_max_top_k: 0, collection_timeout: 0, max_upload_bytes: 1024 * 1024 })
      if (u.includes('/sessions/list')) return jsonRes({ sessions: [] })
      return jsonRes(null)
    })
  )
})

afterEach(() => {
  vi.restoreAllMocks()
})

function renderIngestAndSidebar() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter>
        <Ingest />
        <Sidebar />
      </MemoryRouter>
    </QueryClientProvider>
  )
}

describe('Ingest → collection auto-selection', () => {
  it('keeps the new collection selected after ingesting into it', async () => {
    const { container } = renderIngestAndSidebar()

    await userEvent.type(screen.getByPlaceholderText('my-collection'), 'gamma')
    // The Dropzone's file input is hidden (Tailwind `hidden` = display:none), so
    // set files via fireEvent.change rather than userEvent.upload (which enforces
    // visibility).
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement
    fireEvent.change(fileInput, { target: { files: [new File(['x'], 'a.txt', { type: 'text/plain' })] } })

    await userEvent.click(screen.getByRole('button', { name: /^ingest$/i }))

    await waitFor(() => {
      expect(useUiStore.getState().selectedCollection).toBe('gamma')
    })
  })
})
