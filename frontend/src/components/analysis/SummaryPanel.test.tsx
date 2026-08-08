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
 * Yields across a real macrotask boundary (not just a microtask), mirroring
 * real SSE delivery where frames arrive on separate ticks. The whole sequence
 * still drains on its own, so only assert *terminal* state on a stream built
 * this way — an intermediate render may be superseded before React commits
 * it. Use `gatedFramesOf` when the assertion targets an intermediate render.
 */
async function* framesOf(frames: SseEvent[]): AsyncGenerator<SseEvent, void, unknown> {
  for (const f of frames) {
    await new Promise((resolve) => setTimeout(resolve, 0))
    yield f
  }
}

/**
 * Build a stream that parks until the test releases each frame.
 *
 * `release()` hands over the next frame and the generator parks again, so an
 * assertion on the render a frame caused cannot race the frame after it.
 * `waitFor` polls the *current* DOM: it can wait for a state to arrive, never
 * for one that has already been superseded. Spacing frames a macrotask apart
 * (`framesOf`) does not guarantee React commits in that gap — instrumented
 * locally, the progress render was skipped roughly 1 run in 15, and on CI
 * often enough to redden `main`; with the two frames back to back it was
 * skipped every time.
 *
 * @param frames Frames to deliver, in order, one per `release()` call.
 * @returns The generator to mock `streamSseGet` with, plus its `release`.
 */
function gatedFramesOf(frames: SseEvent[]): {
  stream: AsyncGenerator<SseEvent, void, unknown>
  release: () => void
} {
  const opens: Array<() => void> = []
  const gates = frames.map(
    (_, i) =>
      new Promise<void>((resolve) => {
        opens[i] = resolve
      })
  )
  let released = 0
  async function* stream(): AsyncGenerator<SseEvent, void, unknown> {
    for (let i = 0; i < frames.length; i++) {
      await gates[i]
      yield frames[i]
    }
  }
  return { stream: stream(), release: () => opens[released++]() }
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
    const frames = gatedFramesOf([
      { event: 'summary_progress', data: { job_id: 'j1', mapped: 1, total_units: 2 } },
      { event: 'summary_completed', data: { job_id: 'j1' } }
    ])
    mockStreamSseGet.mockReturnValue(frames.stream)

    render(<SummaryPanel />)
    await userEvent.click(screen.getByRole('button', { name: /generate/i }))

    // Progress is asserted while completion is still gated behind `release`,
    // so the render under test cannot be superseded before it is observed.
    frames.release()
    await waitFor(() => {
      expect(screen.getByText('Summarizing 1/2 units…')).toBeInTheDocument()
    })

    frames.release()
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
    const frames = gatedFramesOf([
      { event: 'summary_progress', data: { job_id: 'other', mapped: 5, total_units: 5 } },
      { event: 'summary_completed', data: { job_id: 'other' } },
      { event: 'summary_progress', data: { job_id: 'j1', mapped: 1, total_units: 2 } },
      { event: 'summary_completed', data: { job_id: 'j1' } }
    ])
    mockStreamSseGet.mockReturnValue(frames.stream)

    render(<SummaryPanel />)
    await userEvent.click(screen.getByRole('button', { name: /generate/i }))

    // The foreign job's frames must not resolve the panel: progress for the
    // real job (j1) must still show before the summary lands. j1's completion
    // stays gated so that render cannot be superseded before it is observed.
    frames.release() // foreign progress
    frames.release() // foreign completion
    frames.release() // j1 progress
    await waitFor(() => {
      expect(screen.getByText('Summarizing 1/2 units…')).toBeInTheDocument()
    })
    frames.release() // j1 completion
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

  it('re-attaches to a fresh build when the post-completion refetch 202s once', async () => {
    // A concurrent ingest bumping the summary revision mid-build makes the
    // server-side cache write a no-op (compare-and-set guard), even though
    // the build succeeded. The refetch after `summary_completed` then
    // legitimately 202s with a fresh job_id for the server's own requeued
    // rebuild — the panel must follow it, not report a failure.
    mockSummarize
      .mockResolvedValueOnce({ job_id: 'j1' })
      .mockResolvedValueOnce({ job_id: 'j2' })
      .mockResolvedValueOnce({ summary: 'Rebuilt summary.', sources: [] })
    mockStreamSseGet
      .mockReturnValueOnce(framesOf([{ event: 'summary_completed', data: { job_id: 'j1' } }]))
      .mockReturnValueOnce(framesOf([{ event: 'summary_completed', data: { job_id: 'j2' } }]))

    render(<SummaryPanel />)
    await userEvent.click(screen.getByRole('button', { name: /generate/i }))

    await waitFor(() => {
      expect(screen.getByText('Rebuilt summary.')).toBeInTheDocument()
    })
    expect(screen.queryByText(/summary generation failed/i)).not.toBeInTheDocument()
    // Initial 202 + two refetches (the first re-queues, the second lands).
    expect(mockSummarize).toHaveBeenCalledTimes(3)
    expect(mockStreamSseGet).toHaveBeenCalledTimes(2)
  })

  it('fails after a second consecutive requeue instead of looping forever', async () => {
    // Bounds the re-attach: two job_id refetches in a row is no longer the
    // ordinary revision-bump race, so it must surface as the existing
    // failure rather than re-attaching indefinitely.
    mockSummarize
      .mockResolvedValueOnce({ job_id: 'j1' })
      .mockResolvedValueOnce({ job_id: 'j2' })
      .mockResolvedValueOnce({ job_id: 'j3' })
    mockStreamSseGet
      .mockReturnValueOnce(framesOf([{ event: 'summary_completed', data: { job_id: 'j1' } }]))
      .mockReturnValueOnce(framesOf([{ event: 'summary_completed', data: { job_id: 'j2' } }]))

    render(<SummaryPanel />)
    await userEvent.click(screen.getByRole('button', { name: /generate/i }))

    await waitFor(() => {
      expect(screen.getByText(/summary generation failed/i)).toBeInTheDocument()
    })
    // Initial 202 + exactly one bounded re-attach's refetch.
    expect(mockSummarize).toHaveBeenCalledTimes(3)
    expect(mockStreamSseGet).toHaveBeenCalledTimes(2)
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
