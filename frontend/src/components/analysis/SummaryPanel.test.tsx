import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { SummaryPanel } from './SummaryPanel'
import { useUiStore } from '@/stores/ui'

function bodyFromString(s: string): ReadableStream<Uint8Array> {
  return new ReadableStream({
    start(c) {
      c.enqueue(new TextEncoder().encode(s))
      c.close()
    },
  })
}

describe('SummaryPanel sources', () => {
  it('renders the summary sources as pills', async () => {
    const frames =
      'data: {"token":"A summary."}\n\n' +
      'data: {"summary":"A summary.","sources":[{"id":"s1","filename":"handbook.pdf","page":3,"text":"chunk","citation_index":1}]}\n\n'
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: true, status: 200, body: bodyFromString(frames) })
    )
    useUiStore.setState({ selectedCollection: 'c1' })
    render(<SummaryPanel />)
    await userEvent.click(screen.getByRole('button', { name: /generate/i }))
    await waitFor(() => {
      expect(screen.getByTestId('source-pill')).toHaveTextContent('handbook.pdf · page 3')
    })
    // Compact by default: the detail card only opens from a pill click.
    expect(screen.queryByText('chunk')).not.toBeInTheDocument()
  })
})

describe('SummaryPanel stream errors', () => {
  it('appends the machine-readable code to the failure copy', async () => {
    const frames = 'data: {"error":"An internal error occurred during streaming.","code":"summary_failed"}\n\n'
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: true, status: 200, body: bodyFromString(frames) })
    )
    useUiStore.setState({ selectedCollection: 'c1' })
    render(<SummaryPanel />)
    await userEvent.click(screen.getByRole('button', { name: /generate/i }))
    await waitFor(() => {
      expect(screen.getByText(/\(summary_failed\)/)).toBeInTheDocument()
    })
    expect(screen.getByText(/summary generation failed/i)).toBeInTheDocument()
  })
})
