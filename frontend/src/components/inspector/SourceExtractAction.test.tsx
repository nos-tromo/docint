import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactNode } from 'react'
import userEvent from '@testing-library/user-event'
import { SourceExtractAction } from './SourceExtractAction'
import { useReportStore } from '@/stores/report'
import { useUiStore } from '@/stores/ui'

const createExtract = vi.fn(async () => ({ job_id: 'j1', adopted: false }))

vi.mock('@/api/extracts', async () => {
  const actual = await vi.importActual<typeof import('@/api/extracts')>('@/api/extracts')
  return { ...actual, createExtract: (...args: unknown[]) => createExtract(...(args as [])) }
})

const originalFetch = globalThis.fetch
const originalCreate = URL.createObjectURL
const originalRevoke = URL.revokeObjectURL

// The action reads the appendix fields off the active report, so it needs the
// query client the app always provides.
function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>
}

describe('SourceExtractAction', () => {
  beforeEach(() => {
    useUiStore.setState({ selectedCollection: 'mydocs' })
    useReportStore.setState({ activeReportId: null })
    createExtract.mockClear()
    URL.createObjectURL = vi.fn(() => 'blob:stub')
    URL.revokeObjectURL = vi.fn()
  })
  afterEach(() => {
    globalThis.fetch = originalFetch
    URL.createObjectURL = originalCreate
    URL.revokeObjectURL = originalRevoke
  })

  it('renders nothing without a hash to address the source by', () => {
    const { container } = render(<SourceExtractAction fileHash={null} filename="report.pdf" />, { wrapper })
    expect(container).toBeEmptyDOMElement()
  })

  it('downloads the bundle the server rendered', async () => {
    globalThis.fetch = vi.fn(async () => ({
      ok: true,
      status: 200,
      blob: async () => new Blob(['PK'])
    })) as unknown as typeof fetch
    render(<SourceExtractAction fileHash="a1b2c3d4" filename="report.pdf" />, { wrapper })
    await userEvent.click(screen.getByRole('button'))
    expect(String((globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0][0])).toContain(
      '/collections/mydocs/sources/a1b2c3d4/extract.zip'
    )
    expect(createExtract).not.toHaveBeenCalled()
  })

  it('queues a targeted build when the source is too large to render inline', async () => {
    globalThis.fetch = vi.fn(async () => ({ ok: false, status: 413 })) as unknown as typeof fetch
    render(<SourceExtractAction fileHash="table-hash" filename="postings.csv" />, { wrapper })
    await userEvent.click(screen.getByRole('button'))
    expect(createExtract).toHaveBeenCalledWith('mydocs', 'table-hash', expect.anything())
  })
})
