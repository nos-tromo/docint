import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { act, renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactNode } from 'react'
import { useReport, useReports } from './useReports'
import { useReportStore } from '@/stores/report'
import { useUiStore } from '@/stores/ui'

let client: QueryClient

function wrapper({ children }: { children: ReactNode }) {
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>
}

/** Answer every request with one status, recording the URLs asked for. */
function answer(status: number, body: unknown): string[] {
  const urls: string[] = []
  vi.stubGlobal(
    'fetch',
    vi.fn(async (u: string) => {
      urls.push(String(u))
      return { ok: status < 400, status, json: async () => body, text: async () => JSON.stringify(body) }
    })
  )
  return urls
}

/** Let a query that was going to fetch on mount do so. */
const settle = () => act(() => new Promise((resolve) => setTimeout(resolve, 0)))

beforeEach(() => {
  localStorage.clear()
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  useUiStore.setState({ selectedCollection: 'docs', selectedOwner: null })
  useReportStore.setState({ activeReportId: 7 })
})

afterEach(() => {
  vi.unstubAllGlobals()
  client.clear()
})

describe('useReports', () => {
  it("lists only the active collection's reports", async () => {
    const urls = answer(200, { reports: [] })
    const { result } = renderHook(() => useReports(), { wrapper })
    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(urls).toHaveLength(1)
    expect(urls[0]).toContain('/reports?collection=docs')
  })

  it('asks for no reports while no collection is active', async () => {
    useUiStore.setState({ selectedCollection: null })
    const urls = answer(200, { reports: [] })
    renderHook(() => useReports(), { wrapper })
    await settle()
    expect(urls).toEqual([])
  })
})

describe('useReport', () => {
  it('opens the report through the active collection', async () => {
    const urls = answer(200, { id: 7, items: [] })
    const { result } = renderHook(() => useReport(7), { wrapper })
    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(urls[0]).toContain('/reports/7?collection=docs')
  })

  it('drops the active report once the active collection no longer answers for it', async () => {
    // Another tab's report, or one kept from before reports were scoped to their
    // collection: kept active, every add would fail against it.
    answer(404, { detail: 'Report not found.' })
    renderHook(() => useReport(7), { wrapper })
    await waitFor(() => expect(useReportStore.getState().activeReportId).toBeNull())
  })

  it('never drops a report other than the one that came back 404', async () => {
    answer(404, { detail: 'Report not found.' })
    const { result } = renderHook(() => useReport(5), { wrapper })
    await waitFor(() => expect(result.current.isError).toBe(true))
    expect(useReportStore.getState().activeReportId).toBe(7)
  })

  it('keeps the active report through a failure that is not a 404', async () => {
    answer(500, { detail: 'Request failed.' })
    const { result } = renderHook(() => useReport(7), { wrapper })
    await waitFor(() => expect(result.current.isError).toBe(true))
    expect(useReportStore.getState().activeReportId).toBe(7)
  })
})
