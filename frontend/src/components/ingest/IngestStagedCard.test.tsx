import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactElement } from 'react'
import { IngestStagedCard } from './IngestStagedCard'
import { useIngestRunStore } from '@/stores/ingestRun'

const getStagedBatch = vi.fn()
const createIngestJob = vi.fn()
vi.mock('@/api/jobs', () => ({
  getStagedBatch: (...args: unknown[]) => getStagedBatch(...args),
  createIngestJob: (...args: unknown[]) => createIngestJob(...args)
}))

function renderIn(ui: ReactElement) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(<QueryClientProvider client={qc}>{ui}</QueryClientProvider>)
}

function staged(over: Partial<{ files: number; bytes: number; running: number; queued: number }> = {}) {
  return {
    collection: 'mydocs',
    files: over.files ?? 1_000,
    bytes: over.bytes ?? 105_000_000,
    preprocess: { running: over.running ?? 0, queued: over.queued ?? 0 }
  }
}

beforeEach(() => {
  useIngestRunStore.getState().reset()
  getStagedBatch.mockReset()
  createIngestJob.mockReset()
  createIngestJob.mockResolvedValue({ job_id: 'job-1', adopted: false })
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

  it('ingests the staged batch and tracks the job it produces', async () => {
    getStagedBatch.mockResolvedValue(staged())
    useIngestRunStore.getState().setNer(true)

    renderIn(<IngestStagedCard collection="mydocs" />)
    fireEvent.click(await screen.findByRole('button', { name: /ingest these files/i }))

    await waitFor(() =>
      expect(createIngestJob).toHaveBeenCalledWith({
        collection: 'mydocs',
        ner: true,
        hate_speech: false
      })
    )
    // Tracked, so the run gets a job card and the staged card steps aside.
    await waitFor(() =>
      expect(useIngestRunStore.getState().trackedJobs).toEqual([
        { job_id: 'job-1', collection: 'mydocs' }
      ])
    )
  })
})
