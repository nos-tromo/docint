import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactNode } from 'react'
import { ExtractsPanel } from './ExtractsPanel'
import { useIngestJobsStore } from '@/stores/ingestJobs'
import { useUiStore } from '@/stores/ui'
import type { ExtractRecord, IngestEvent } from '@/api/types'

const RECORD: ExtractRecord = {
  extract_id: '20260102-030405-deadbeef',
  collection: 'mydocs',
  filename: 'mydocs-extract-20260102-0304.zip',
  created_at: '2026-01-02T03:04:05+00:00',
  size: 2048,
  target: null,
  counts: { documents: 3, media: 1, figures: 12 },
  pdf_skipped: false
}

vi.mock('@/api/extracts', async () => {
  const actual = await vi.importActual<typeof import('@/api/extracts')>('@/api/extracts')
  return {
    ...actual,
    listExtracts: vi.fn(async () => ({ extracts: [RECORD] })),
    createExtract: vi.fn(async () => ({ job_id: 'j1', adopted: false })),
    deleteExtract: vi.fn(async () => ({ ok: true }))
  }
})

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>
}

function frame(event: IngestEvent['event'], data: Record<string, unknown> = {}): IngestEvent {
  return { event, data } as IngestEvent
}

describe('ExtractsPanel', () => {
  beforeEach(() => {
    useUiStore.setState({ selectedCollection: 'mydocs' })
    useIngestJobsStore.getState().clear()
  })
  afterEach(() => useIngestJobsStore.getState().clear())

  it('lists a stored bundle with its size and contents', async () => {
    render(<ExtractsPanel />, { wrapper })
    expect(await screen.findByText('mydocs-extract-20260102-0304.zip')).toBeInTheDocument()
    expect(screen.getByText(/2\.0 KB/)).toBeInTheDocument()
    expect(screen.getByText(/3 documents/)).toBeInTheDocument()
    expect(screen.getByText(/12 figures/)).toBeInTheDocument()
  })

  it('offers a download link pointing at the stored bundle', async () => {
    render(<ExtractsPanel />, { wrapper })
    const link = await screen.findByRole('link', { name: /Download extract/i })
    expect(link.getAttribute('href')).toContain(
      '/collections/mydocs/extracts/20260102-030405-deadbeef/download'
    )
  })

  it('shows progress from the shared job stream', async () => {
    const store = useIngestJobsStore.getState()
    store.appendEvent('j1', frame('extract_started', { job_id: 'j1' }))
    store.appendEvent('j1', frame('extract_progress', { job_id: 'j1', rendered: 2, total_units: 5 }))
    render(<ExtractsPanel />, { wrapper })
    expect(await screen.findByText('Rendering 2 of 5')).toBeInTheDocument()
  })

  it('clears the progress card once the run terminates', async () => {
    const store = useIngestJobsStore.getState()
    store.appendEvent('j1', frame('extract_started', { job_id: 'j1' }))
    store.appendEvent('j1', frame('extract_completed', { job_id: 'j1' }))
    render(<ExtractsPanel />, { wrapper })
    expect(await screen.findByText('mydocs-extract-20260102-0304.zip')).toBeInTheDocument()
    expect(screen.queryByText(/Rendering/)).not.toBeInTheDocument()
  })

  it('reports a failed run rather than spinning forever', async () => {
    const store = useIngestJobsStore.getState()
    store.appendEvent('j1', frame('extract_started', { job_id: 'j1' }))
    store.appendEvent('j1', frame('error', { job_id: 'j1', code: 'extract_failed' }))
    render(<ExtractsPanel />, { wrapper })
    expect(await screen.findByText('The extract could not be built.')).toBeInTheDocument()
  })

  it('ignores another kind of job running at the same time', async () => {
    const store = useIngestJobsStore.getState()
    store.appendEvent('j2', frame('ingestion_started', { job_id: 'j2' }))
    store.appendEvent('j2', frame('ingestion_progress', { job_id: 'j2', processed: 1 }))
    render(<ExtractsPanel />, { wrapper })
    expect(await screen.findByText('mydocs-extract-20260102-0304.zip')).toBeInTheDocument()
    expect(screen.queryByText(/Rendering/)).not.toBeInTheDocument()
  })
})
