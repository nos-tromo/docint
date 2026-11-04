import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactNode } from 'react'
import userEvent from '@testing-library/user-event'
import { SourceExtractAction } from './SourceExtractAction'
import { useReportStore } from '@/stores/report'
import { useUiStore } from '@/stores/ui'
import { captureDownloads, fileResponse } from '@/test/downloads'

const createExtract = vi.fn(async () => ({ job_id: 'j1', adopted: false }))

vi.mock('@/api/extracts', async () => {
  const actual = await vi.importActual<typeof import('@/api/extracts')>('@/api/extracts')
  return { ...actual, createExtract: (...args: unknown[]) => createExtract(...(args as [])) }
})

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
  })
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('renders nothing without a hash to address the source by', () => {
    const { container } = render(<SourceExtractAction fileHash={null} filename="report.pdf" />, { wrapper })
    expect(container).toBeEmptyDOMElement()
  })

  it('downloads the bundle the server rendered, through the purpose wrapper', async () => {
    const downloads = captureDownloads()
    const fetchMock = vi.fn(async (..._args: unknown[]) => fileResponse('PK', { type: 'application/zip' }))
    vi.stubGlobal('fetch', fetchMock)
    render(<SourceExtractAction fileHash="a1b2c3d4" filename="report.pdf" />, { wrapper })
    await userEvent.click(screen.getByRole('button'))
    await waitFor(() => expect(downloads.saved()).toHaveLength(1))
    expect(String(fetchMock.mock.calls[0][0])).toContain('/collections/mydocs/sources/a1b2c3d4/extract.zip')
    expect(downloads.saved()[0].download).toBe('report-extract.zip')
    expect(createExtract).not.toHaveBeenCalled()
  })

  it('queues a targeted build when the source is too large to render inline', async () => {
    const downloads = captureDownloads()
    vi.stubGlobal('fetch', vi.fn(async () => new Response('{}', { status: 413 })))
    render(<SourceExtractAction fileHash="table-hash" filename="postings.csv" />, { wrapper })
    await userEvent.click(screen.getByRole('button'))
    await waitFor(() => expect(createExtract).toHaveBeenCalledWith('mydocs', 'table-hash', expect.anything()))
    expect(downloads.saved()).toEqual([])
  })
})
