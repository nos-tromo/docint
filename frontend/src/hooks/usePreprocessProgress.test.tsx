import { renderHook, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactNode } from 'react'
import { usePreprocessProgress } from './usePreprocessProgress'

const getStagedBatch = vi.fn()
vi.mock('@/api/jobs', () => ({
  getStagedBatch: (...args: unknown[]) => getStagedBatch(...args)
}))

function wrapper({ children }: { children: ReactNode }) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return <QueryClientProvider client={qc}>{children}</QueryClientProvider>
}

function staged(stages: unknown[]) {
  return {
    collection: 'mydocs',
    files: 3,
    bytes: 10,
    partial: 0,
    entries: [],
    entries_truncated: true,
    preprocess: { running: 1, queued: 2, stages }
  }
}

beforeEach(() => getStagedBatch.mockReset())

describe('usePreprocessProgress', () => {
  it('reports the stages the server is working through', async () => {
    getStagedBatch.mockResolvedValue(staged([{ stage: 'pdf', done: 1, total: 4, failed: 0 }]))

    const { result } = renderHook(() => usePreprocessProgress('mydocs', 'uploading'), { wrapper })

    await waitFor(() =>
      expect(result.current).toEqual([{ stage: 'pdf', done: 1, total: 4, failed: 0 }])
    )
  })

  it('leaves the file names behind, since only the counts are polled for', async () => {
    getStagedBatch.mockResolvedValue(staged([]))

    renderHook(() => usePreprocessProgress('mydocs', 'processing'), { wrapper })

    // A folder-sized batch lists tens of thousands of names; re-sending them
    // every few seconds for the length of an ingest is the cost this avoids.
    await waitFor(() =>
      expect(getStagedBatch).toHaveBeenCalledWith('mydocs', { includeEntries: false })
    )
  })

  it('asks nothing at all once the run has ended', async () => {
    getStagedBatch.mockResolvedValue(staged([]))

    const { result } = renderHook(() => usePreprocessProgress('mydocs', 'complete'), { wrapper })

    await new Promise((resolve) => setTimeout(resolve, 20))
    expect(getStagedBatch).not.toHaveBeenCalled()
    expect(result.current).toBeUndefined()
  })

  it('asks nothing without a collection to ask about', async () => {
    const { result } = renderHook(() => usePreprocessProgress('', 'uploading'), { wrapper })

    await new Promise((resolve) => setTimeout(resolve, 20))
    expect(getStagedBatch).not.toHaveBeenCalled()
    expect(result.current).toBeUndefined()
  })

  it('survives an answer that carries no tally', async () => {
    // The route answers 404 for a collection nothing is staged for, and a
    // test double may answer with nothing at all; neither may break the card
    // the bars are drawn on.
    getStagedBatch.mockResolvedValue(null)

    const { result } = renderHook(() => usePreprocessProgress('mydocs', 'uploading'), { wrapper })

    await waitFor(() => expect(result.current).toEqual([]))
  })
})
