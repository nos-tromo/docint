import { render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactElement } from 'react'
import { IngestStagedCard } from './IngestStagedCard'
import { useIngestRunStore } from '@/stores/ingestRun'

const getStagedBatch = vi.fn()
vi.mock('@/api/jobs', () => ({
  getStagedBatch: (...args: unknown[]) => getStagedBatch(...args)
}))

function renderIn(ui: ReactElement) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(<QueryClientProvider client={qc}>{ui}</QueryClientProvider>)
}

function staged(
  over: Partial<{
    files: number
    bytes: number
    running: number
    queued: number
    partial: number
  }> = {}
) {
  return {
    collection: 'mydocs',
    files: over.files ?? 1_000,
    bytes: over.bytes ?? 105_000_000,
    partial: over.partial ?? 0,
    entries: [],
    entries_truncated: false,
    preprocess: { running: over.running ?? 0, queued: over.queued ?? 0 }
  }
}

beforeEach(() => {
  useIngestRunStore.getState().reset()
  getStagedBatch.mockReset()
})

describe('IngestStagedCard', () => {
  it('reports files the server holds that no job accounts for', async () => {
    getStagedBatch.mockResolvedValue(staged())

    renderIn(<IngestStagedCard collection="mydocs" />)

    expect(await screen.findByText(/1000 files/)).toBeInTheDocument()
    expect(screen.getByText(/no ingestion is running/i)).toBeInTheDocument()
  })

  it('says nothing at all when the collection has no staged batch', async () => {
    getStagedBatch.mockResolvedValue(staged({ files: 0, bytes: 0 }))

    const { container } = renderIn(<IngestStagedCard collection="mydocs" />)

    await waitFor(() => expect(getStagedBatch).toHaveBeenCalled())
    expect(container).toBeEmptyDOMElement()
  })

  it('reports the preprocessing the pool is still doing for the batch', async () => {
    getStagedBatch.mockResolvedValue(staged({ running: 4, queued: 812 }))

    renderIn(<IngestStagedCard collection="mydocs" />)

    // Upload-time preprocessing belongs to no job, so this card is the only
    // place a user can see it happening at all.
    expect(await screen.findByText(/4 in progress, 812 waiting/)).toBeInTheDocument()
  })

  it('offers no way to ingest the batch, only to finish the upload', async () => {
    // The only route to a staged-but-unfinalized batch is an upload that
    // stopped early, so what is on disk is an arbitrary fraction of what the
    // user picked. A button here would silently index that fraction.
    getStagedBatch.mockResolvedValue(staged())

    renderIn(<IngestStagedCard collection="mydocs" />)

    expect(await screen.findByText(/never finished/i)).toBeInTheDocument()
    expect(screen.queryByRole('button')).toBeNull()
  })

  it('names a transfer cut off mid-file, which is discarded rather than staged', async () => {
    getStagedBatch.mockResolvedValue(staged({ partial: 1 }))

    renderIn(<IngestStagedCard collection="mydocs" />)

    expect(await screen.findByText(/cut off mid-file/i)).toBeInTheDocument()
  })
})
