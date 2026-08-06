import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi, beforeEach } from 'vitest'
import { SummaryPanel } from './SummaryPanel'
import { useUiStore } from '@/stores/ui'
import { summarize } from '@/api/analysis'
import { streamSseGet } from '@/api/sse'
import { ApiError } from '@/api/client'
import type { SummarizeResult } from '@/api/types'
import type { SseEvent } from '@/api/sse'

vi.mock('@/api/analysis', () => ({ summarize: vi.fn() }))
vi.mock('@/api/sse', () => ({ streamSseGet: vi.fn() }))

const mockSummarize = summarize as unknown as ReturnType<typeof vi.fn>
const mockStreamSseGet = streamSseGet as unknown as ReturnType<typeof vi.fn>

/**
 * Build an async generator streamSseGet can be mocked to return.
 *
 * Yields across a real macrotask boundary (not just a microtask) so
 * `waitFor` can observe the intermediate `building`/progress render between
 * frames instead of the whole sequence draining before the test's first
 * poll — mirrors real SSE delivery, where frames arrive on separate ticks.
 */
async function* framesOf(frames: SseEvent[]): AsyncGenerator<SseEvent, void, unknown> {
  for (const f of frames) {
    await new Promise((resolve) => setTimeout(resolve, 0))
    yield f
  }
}

beforeEach(() => {
  useUiStore.setState({ selectedCollection: 'c1' })
  mockSummarize.mockReset()
  mockStreamSseGet.mockReset()
})

describe('SummaryPanel cache hit', () => {
  it('renders a cached summary immediately on 200', async () => {
    const cached: SummarizeResult = {
      summary: 'A summary.',
      sources: [{ id: 's1', filename: 'handbook.pdf', page: 3, text: 'chunk', citation_index: 1 }]
    }
    mockSummarize.mockResolvedValue(cached)

    render(<SummaryPanel />)
    await userEvent.click(screen.getByRole('button', { name: /generate/i }))

    await waitFor(() => {
      expect(screen.getByText('A summary.')).toBeInTheDocument()
    })
    expect(screen.getByTestId('source-pill')).toHaveTextContent('handbook.pdf · page 3')
    expect(mockStreamSseGet).not.toHaveBeenCalled()
  })
})

describe('SummaryPanel job-driven build', () => {
  it('shows progress and refetches when the API answers 202', async () => {
    mockSummarize
      .mockResolvedValueOnce({ job_id: 'j1' })
      .mockResolvedValueOnce({ summary: 'Built summary.', sources: [] })
    mockStreamSseGet.mockReturnValue(
      framesOf([
        { event: 'summary_progress', data: { job_id: 'j1', mapped: 1, total_units: 2 } },
        { event: 'summary_completed', data: { job_id: 'j1' } }
      ])
    )

    render(<SummaryPanel />)
    await userEvent.click(screen.getByRole('button', { name: /generate/i }))

    await waitFor(() => {
      expect(screen.getByText('Summarizing 1/2 units…')).toBeInTheDocument()
    })

    await waitFor(() => {
      expect(screen.getByText('Built summary.')).toBeInTheDocument()
    })
    // The progress copy is gone once the build lands.
    expect(screen.queryByText('Summarizing 1/2 units…')).not.toBeInTheDocument()
    // Second summarize() call is the cache-hit re-fetch after completion.
    expect(mockSummarize).toHaveBeenCalledTimes(2)
    expect(mockSummarize).toHaveBeenNthCalledWith(2, false, 'c1')
  })

  it('ignores events for other job ids', async () => {
    mockSummarize
      .mockResolvedValueOnce({ job_id: 'j1' })
      .mockResolvedValueOnce({ summary: 'Built summary.', sources: [] })
    mockStreamSseGet.mockReturnValue(
      framesOf([
        { event: 'summary_progress', data: { job_id: 'other', mapped: 5, total_units: 5 } },
        { event: 'summary_completed', data: { job_id: 'other' } },
        { event: 'summary_progress', data: { job_id: 'j1', mapped: 1, total_units: 2 } },
        { event: 'summary_completed', data: { job_id: 'j1' } }
      ])
    )

    render(<SummaryPanel />)
    await userEvent.click(screen.getByRole('button', { name: /generate/i }))

    // The foreign job's frames must not resolve the panel: progress for the
    // real job (j1) must still show before the summary lands.
    await waitFor(() => {
      expect(screen.getByText('Summarizing 1/2 units…')).toBeInTheDocument()
    })
    await waitFor(() => {
      expect(screen.getByText('Built summary.')).toBeInTheDocument()
    })
    // Only one cache-hit re-fetch — the foreign job's summary_completed must
    // not have triggered a (premature) re-fetch of its own.
    expect(mockSummarize).toHaveBeenCalledTimes(2)
  })

  it('adopts the in-flight job on 409', async () => {
    mockSummarize
      .mockRejectedValueOnce(new ApiError(409, { detail: { job_id: 'j2' } }))
      .mockResolvedValueOnce({ summary: 'Adopted summary.', sources: [] })
    mockStreamSseGet.mockReturnValue(
      framesOf([{ event: 'summary_completed', data: { job_id: 'j2' } }])
    )

    render(<SummaryPanel />)
    await userEvent.click(screen.getByRole('button', { name: /generate/i }))

    await waitFor(() => {
      expect(screen.getByText('Adopted summary.')).toBeInTheDocument()
    })
    expect(mockStreamSseGet).toHaveBeenCalled()
    expect(mockSummarize).toHaveBeenNthCalledWith(2, false, 'c1')
  })

  it('renders a partial build as a summary with an incomplete notice, not a failure', async () => {
    // A capped build used to be withheld from the cache, so the
    // post-completion refetch missed, silently queued another full build and
    // showed the failure copy. It is now cached and served as a 200 whose
    // diagnostics carry `partial`.
    mockSummarize.mockResolvedValueOnce({ job_id: 'j1' }).mockResolvedValueOnce({
      summary: 'Partial summary.',
      sources: [],
      summary_diagnostics: {
        total_documents: 10,
        covered_documents: 4,
        coverage_ratio: 0.4,
        uncovered_documents: [],
        coverage_target: 0.7,
        candidate_count: 10,
        deduped_count: 4,
        sampled_count: 4,
        partial: true
      }
    })
    mockStreamSseGet.mockReturnValue(
      framesOf([{ event: 'summary_completed', data: { job_id: 'j1' } }])
    )

    render(<SummaryPanel />)
    await userEvent.click(screen.getByRole('button', { name: /generate/i }))

    await waitFor(() => {
      expect(screen.getByText('Partial summary.')).toBeInTheDocument()
    })
    expect(screen.getByTestId('coverage-partial-notice')).toHaveTextContent(/incomplete summary/i)
    expect(screen.queryByText(/summary generation failed/i)).not.toBeInTheDocument()
    // Exactly one refetch — no second build silently queued.
    expect(mockSummarize).toHaveBeenCalledTimes(2)
  })

  it('fails with localized copy on error event', async () => {
    mockSummarize.mockResolvedValueOnce({ job_id: 'j1' })
    mockStreamSseGet.mockReturnValue(
      framesOf([{ event: 'error', data: { job_id: 'j1', code: 'summary_failed' } }])
    )

    render(<SummaryPanel />)
    await userEvent.click(screen.getByRole('button', { name: /generate/i }))

    await waitFor(() => {
      expect(screen.getByText(/\(summary_failed\)/)).toBeInTheDocument()
    })
    expect(screen.getByText(/summary generation failed/i)).toBeInTheDocument()
    // Only the initial 202 call — no cache-hit re-fetch after a failure.
    expect(mockSummarize).toHaveBeenCalledTimes(1)
  })
})
