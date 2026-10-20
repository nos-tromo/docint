import { describe, it, expect, vi, afterEach } from 'vitest'
import { ApiError, setOwnerParam } from './client'
import {
  addReportItem,
  addReportItems,
  assignReport,
  createReport,
  deleteReport,
  getReport,
  listReports,
  listUnassignedReports,
  queueReportPdf,
  refreshCollectionOverview,
  removeReportItem,
  reorderReportItems,
  reportExportHref,
  reportPdfHref,
  reportPdfStatus,
  updateReport,
  updateReportItem
} from './reports'
import type { ReportItemInput } from './types'

afterEach(() => {
  vi.restoreAllMocks()
  setOwnerParam(null)
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

  it('listUnassignedReports GETs the reports that belong to no collection', async () => {
    mockFetch({ reports: [] })
    await listUnassignedReports()
    const call = lastCall()
    expect(String(call[0])).toMatch(/\/reports\/unassigned$/)
    expect(call[1]).toBeUndefined()
  })

  it('assignReport POSTs the collection to move the report into', async () => {
    mockFetch({ id: 3, items: [] })
    await assignReport(3, 'docs 2')
    const call = lastCall()
    expect(String(call[0])).toMatch(/\/reports\/unassigned\/3\/assign\?collection=docs%202$/)
    expect(call[1].method).toBe('POST')
  })

  it('getReport GETs the report path in the active collection', async () => {
    mockFetch({ id: 4, items: [] })
    await getReport('docs', 4)
    const call = lastCall()
    expect(String(call[0])).toContain('/reports/4?collection=docs')
    expect(call[1]).toBeUndefined()
  })

  it('escapes the collection, so its name cannot spill into the query', async () => {
    mockFetch({ id: 4, items: [] })
    await getReport('Fall A&B', 4)
    expect(String(lastCall()[0])).toContain('/reports/4?collection=Fall%20A%26B')
  })

  it('updateReport uses PATCH', async () => {
    mockFetch({ id: 4 })
    await updateReport('docs', 4, { title: 'New' })
    const call = lastCall()
    expect(String(call[0])).toContain('/reports/4?collection=docs')
    expect(call[1].method).toBe('PATCH')
  })

  it('deleteReport uses DELETE', async () => {
    mockFetch({ ok: true })
    await deleteReport('docs', 4)
    const call = lastCall()
    expect(String(call[0])).toContain('/reports/4?collection=docs')
    expect(call[1].method).toBe('DELETE')
  })

  it('addReportItem POSTs to the items path', async () => {
    mockFetch({ id: 5 })
    await addReportItem('docs', 3, { artifact_type: 'entity_finding', dedupe_key: 'entity:c1', snapshot: {} })
    const call = lastCall()
    expect(String(call[0])).toContain('/reports/3/items?collection=docs')
    expect(call[1].method).toBe('POST')
  })

  it('addReportItems POSTs the batch with nothing but its items', async () => {
    mockFetch({ added: 1, skipped: 0, updated: 0, item_count: 1 })
    const items: ReportItemInput[] = [{ artifact_type: 'entity_finding', dedupe_key: 'entity:c1', snapshot: {} }]
    await addReportItems('docs', 3, items)
    const call = lastCall()
    expect(String(call[0])).toContain('/reports/3/items/batch?collection=docs')
    expect(JSON.parse(call[1].body)).toEqual({ items })
  })

  it('updateReportItem uses PATCH on the item path', async () => {
    mockFetch({ id: 5 })
    await updateReportItem('docs', 3, 5, { note: 'n' })
    const call = lastCall()
    expect(String(call[0])).toContain('/reports/3/items/5?collection=docs')
    expect(call[1].method).toBe('PATCH')
  })

  it('removeReportItem uses DELETE on the item path', async () => {
    mockFetch({ ok: true })
    await removeReportItem('docs', 3, 5)
    const call = lastCall()
    expect(String(call[0])).toContain('/reports/3/items/5?collection=docs')
    expect(call[1].method).toBe('DELETE')
  })

  it('reorderReportItems POSTs item_ids', async () => {
    mockFetch({ id: 3, items: [] })
    await reorderReportItems('docs', 3, [2, 1])
    const call = lastCall()
    expect(String(call[0])).toContain('/reports/3/items/reorder?collection=docs')
    expect(JSON.parse(call[1].body)).toEqual({ item_ids: [2, 1] })
  })

  it('reportExportHref builds the export URL for each format', () => {
    expect(reportExportHref('docs', 7, 'pdf')).toContain('/reports/7/export.pdf?collection=docs')
    expect(reportExportHref('docs', 7, 'zip')).toContain('/reports/7/export.zip?collection=docs')
    expect(reportExportHref('docs', 7, 'md')).toContain('/reports/7/export.md?collection=docs')
  })

  it('refreshCollectionOverview POSTs to the refresh route', async () => {
    mockFetch({ id: 7 })
    await refreshCollectionOverview('docs', 7)
    const call = lastCall()
    expect(String(call[0])).toContain('/reports/7/collection-overview/refresh?collection=docs')
    expect(call[1].method).toBe('POST')
  })
})

describe('report PDF job api', () => {
  /** Stub `fetch` with one non-2xx answer whose body is JSON. */
  function mockError(status: number, body: unknown) {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: false, status, text: async () => JSON.stringify(body) })
    )
  }

  it('queues a render and returns its job id', async () => {
    mockFetch({ job_id: 'a1b2c3' })
    await expect(queueReportPdf('docs', 7)).resolves.toEqual({ job_id: 'a1b2c3', adopted: false })
    const call = lastCall()
    expect(String(call[0])).toContain('/reports/7/pdf?collection=docs')
    expect(call[1].method).toBe('POST')
  })

  it('adopts the render a 409 names under detail', async () => {
    mockError(409, {
      detail: { message: 'A PDF of this report is already being rendered.', job_id: 'd4e5f6' }
    })
    await expect(queueReportPdf('docs', 7)).resolves.toEqual({ job_id: 'd4e5f6', adopted: true })
  })

  it('adopts the render a 409 names at the top level', async () => {
    mockError(409, { message: 'A PDF of this report is already being rendered.', job_id: 'd4e5f6' })
    await expect(queueReportPdf('docs', 7)).resolves.toEqual({ job_id: 'd4e5f6', adopted: true })
  })

  it('rethrows a refusal that names no job', async () => {
    mockError(503, { detail: 'PDF export is not available.' })
    await expect(queueReportPdf('docs', 7)).rejects.toBeInstanceOf(ApiError)
  })

  it('reads the status from its own route', async () => {
    mockFetch({ job: null, pdf: null })
    await expect(reportPdfStatus('docs', 7)).resolves.toEqual({ job: null, pdf: null })
    const call = lastCall()
    expect(String(call[0])).toContain('/reports/7/pdf/status?collection=docs')
    expect(call[1]).toBeUndefined()
  })

  it('links the stored PDF', () => {
    expect(reportPdfHref('docs', 7)).toMatch(/\/reports\/7\/pdf\?collection=docs$/)
  })

  it('carries the admin owner context on every download link', () => {
    // A link is not an apiGet, so nothing else adds it — and without it an
    // admin in another user's namespace is answered 404.
    setOwnerParam('other.operator')
    expect(reportPdfHref('docs', 7)).toContain('/reports/7/pdf?collection=docs&owner=other.operator')
    expect(reportExportHref('docs', 7, 'md')).toContain('/reports/7/export.md?collection=docs&owner=other.operator')
  })
})
