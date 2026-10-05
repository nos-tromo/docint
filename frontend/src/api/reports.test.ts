import { describe, it, expect, vi, afterEach } from 'vitest'
import { ApiError, setOwnerParam } from './client'
import {
  addReportItem,
  createReport,
  deleteReport,
  getReport,
  listReports,
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

  it('refreshCollectionOverview POSTs to the refresh route', async () => {
    mockFetch({ id: 7 })
    await refreshCollectionOverview(7)
    const call = lastCall()
    expect(String(call[0])).toContain('/reports/7/collection-overview/refresh')
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
    await expect(queueReportPdf(7)).resolves.toEqual({ job_id: 'a1b2c3', adopted: false })
    const call = lastCall()
    expect(String(call[0])).toContain('/reports/7/pdf')
    expect(call[1].method).toBe('POST')
  })

  it('adopts the render a 409 names under detail', async () => {
    mockError(409, {
      detail: { message: 'A PDF of this report is already being rendered.', job_id: 'd4e5f6' }
    })
    await expect(queueReportPdf(7)).resolves.toEqual({ job_id: 'd4e5f6', adopted: true })
  })

  it('adopts the render a 409 names at the top level', async () => {
    mockError(409, { message: 'A PDF of this report is already being rendered.', job_id: 'd4e5f6' })
    await expect(queueReportPdf(7)).resolves.toEqual({ job_id: 'd4e5f6', adopted: true })
  })

  it('rethrows a refusal that names no job', async () => {
    mockError(503, { detail: 'PDF export is not available.' })
    await expect(queueReportPdf(7)).rejects.toBeInstanceOf(ApiError)
  })

  it('reads the status from its own route', async () => {
    mockFetch({ job: null, pdf: null })
    await expect(reportPdfStatus(7)).resolves.toEqual({ job: null, pdf: null })
    const call = lastCall()
    expect(String(call[0])).toContain('/reports/7/pdf/status')
    expect(call[1]).toBeUndefined()
  })

  it('links the stored PDF', () => {
    expect(reportPdfHref(7)).toMatch(/\/reports\/7\/pdf$/)
  })

  it('carries the admin owner context on every download link', () => {
    // A link is not an apiGet, so nothing else adds it — and without it an
    // admin in another user's namespace is answered 404.
    setOwnerParam('other.operator')
    expect(reportPdfHref(7)).toContain('/reports/7/pdf?owner=other.operator')
    expect(reportExportHref(7, 'md')).toContain('/reports/7/export.md?owner=other.operator')
  })
})
