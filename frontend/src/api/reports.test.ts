import { describe, it, expect, vi, afterEach } from 'vitest'
import {
  addReportItem,
  createReport,
  deleteReport,
  getReport,
  listReports,
  removeReportItem,
  reorderReportItems,
  reportExportHref,
  updateReport,
  updateReportItem
} from './reports'

afterEach(() => {
  vi.restoreAllMocks()
})

function mockFetch(body: unknown) {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => body,
      text: async () => JSON.stringify(body)
    })
  )
}

function lastCall() {
  return (fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0]
}

describe('reports api', () => {
  it('createReport POSTs the body', async () => {
    mockFetch({ id: 1 })
    await createReport({ title: 'A', collection_name: 'docs' })
    const call = lastCall()
    expect(String(call[0])).toContain('/reports')
    expect(call[1].method).toBe('POST')
    expect(JSON.parse(call[1].body)).toMatchObject({ title: 'A', collection_name: 'docs' })
  })

  it('listReports passes the collection filter', async () => {
    mockFetch({ reports: [] })
    await listReports('docs')
    expect(String(lastCall()[0])).toContain('collection=docs')
  })

  it('getReport GETs the report path', async () => {
    mockFetch({ id: 4, items: [] })
    await getReport(4)
    const call = lastCall()
    expect(String(call[0])).toContain('/reports/4')
    expect(call[1]).toBeUndefined()
  })

  it('updateReport uses PATCH', async () => {
    mockFetch({ id: 4 })
    await updateReport(4, { title: 'New' })
    const call = lastCall()
    expect(String(call[0])).toContain('/reports/4')
    expect(call[1].method).toBe('PATCH')
  })

  it('deleteReport uses DELETE', async () => {
    mockFetch({ ok: true })
    await deleteReport(4)
    expect(lastCall()[1].method).toBe('DELETE')
  })

  it('addReportItem POSTs to the items path', async () => {
    mockFetch({ id: 5 })
    await addReportItem(3, { artifact_type: 'entity_finding', dedupe_key: 'entity:c1', snapshot: {} })
    const call = lastCall()
    expect(String(call[0])).toContain('/reports/3/items')
    expect(call[1].method).toBe('POST')
  })

  it('updateReportItem uses PATCH on the item path', async () => {
    mockFetch({ id: 5 })
    await updateReportItem(3, 5, { note: 'n' })
    const call = lastCall()
    expect(String(call[0])).toContain('/reports/3/items/5')
    expect(call[1].method).toBe('PATCH')
  })

  it('removeReportItem uses DELETE on the item path', async () => {
    mockFetch({ ok: true })
    await removeReportItem(3, 5)
    const call = lastCall()
    expect(String(call[0])).toContain('/reports/3/items/5')
    expect(call[1].method).toBe('DELETE')
  })

  it('reorderReportItems POSTs item_ids', async () => {
    mockFetch({ id: 3, items: [] })
    await reorderReportItems(3, [2, 1])
    const call = lastCall()
    expect(String(call[0])).toContain('/reports/3/items/reorder')
    expect(JSON.parse(call[1].body)).toEqual({ item_ids: [2, 1] })
  })

  it('reportExportHref builds the export URL for each format', () => {
    expect(reportExportHref(7, 'pdf')).toContain('/reports/7/export.pdf')
    expect(reportExportHref(7, 'zip')).toContain('/reports/7/export.zip')
    expect(reportExportHref(7, 'md')).toContain('/reports/7/export.md')
  })
})
