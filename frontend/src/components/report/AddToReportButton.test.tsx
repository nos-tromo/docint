import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { AddToReportButton } from './AddToReportButton'
import { reportKey } from '@/hooks/useReports'
import { useReportStore } from '@/stores/report'
import { useUiStore } from '@/stores/ui'
import type { Report, ReportItemInput } from '@/api/types'

const item: ReportItemInput = {
  artifact_type: 'entity_finding',
  dedupe_key: 'entity:c1',
  snapshot: { chunk_id: 'c1' }
}

function renderButton(inReport: boolean, qc: QueryClient) {
  return render(
    <QueryClientProvider client={qc}>
      <AddToReportButton item={item} inReport={inReport} />
    </QueryClientProvider>
  )
}

beforeEach(() => {
  localStorage.clear()
  useReportStore.setState({ activeReportId: null })
  useUiStore.setState({ selectedCollection: 'docs' })
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('AddToReportButton', () => {
  it('auto-creates a report then adds the item when none is active', async () => {
    const calls: Array<{ url: string; method: string }> = []
    vi.stubGlobal(
      'fetch',
      vi.fn(async (u: string, init?: RequestInit) => {
        const url = String(u)
        calls.push({ url, method: init?.method ?? 'GET' })
        if (url.endsWith('/reports') && init?.method === 'POST') {
          return { ok: true, status: 200, json: async () => ({ id: 1, title: 'Untitled report', items: [] }) }
        }
        return { ok: true, status: 200, json: async () => ({ id: 9, dedupe_key: 'entity:c1' }) }
      })
    )

    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    renderButton(false, qc)
    await userEvent.click(screen.getByRole('button', { name: /\+ report/i }))

    await waitFor(() => {
      expect(calls.some((c) => c.url.endsWith('/reports') && c.method === 'POST')).toBe(true)
      expect(calls.some((c) => c.url.includes('/reports/1/items') && c.method === 'POST')).toBe(true)
    })
    expect(useReportStore.getState().activeReportId).toBe(1)
  })

  it('removes the item when it is already in the report', async () => {
    useReportStore.setState({ activeReportId: 1 })
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const report: Report = {
      id: 1,
      title: 'R',
      collection_name: 'docs',
      operator: null,
      reference_number: null,
      session_id: null,
      created_at: null,
      updated_at: null,
      item_count: 1,
      items: [
        {
          id: 9,
          artifact_type: 'entity_finding',
          dedupe_key: 'entity:c1',
          position: 0,
          note: null,
          snapshot: {},
          created_at: null
        }
      ]
    }
    qc.setQueryData(reportKey(1), report)

    const calls: Array<{ url: string; method: string }> = []
    vi.stubGlobal(
      'fetch',
      vi.fn(async (u: string, init?: RequestInit) => {
        calls.push({ url: String(u), method: init?.method ?? 'GET' })
        return { ok: true, status: 200, json: async () => ({ ok: true }) }
      })
    )

    renderButton(true, qc)
    expect(screen.getByRole('button', { name: /in report/i })).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: /in report/i }))

    await waitFor(() => {
      expect(calls.some((c) => c.url.includes('/reports/1/items/9') && c.method === 'DELETE')).toBe(true)
    })
  })
})
