import { apiDelete, apiGet, apiPatch, apiPost, url, withOwner } from './client'
import { queueJob } from './jobs'
import type {
  Report,
  ReportBatchResult,
  ReportExportFormat,
  ReportItem,
  ReportItemInput,
  ReportPdfStatus,
  ReportSummary
} from './types'

export const listReports = (collection?: string) =>
  apiGet<{ reports: ReportSummary[] }>('/reports', collection ? { collection } : undefined)

export const getReport = (id: number) => apiGet<Report>(`/reports/${id}`)

export const createReport = (body: {
  title: string
  collection_name?: string | null
  session_id?: string | null
  operator?: string
}) => apiPost<Report>('/reports', body)

export const updateReport = (
  id: number,
  body: { title?: string; operator?: string; reference_number?: string; show_toc?: boolean; show_collection_overview?: boolean }
) => apiPatch<Report>(`/reports/${id}`, body)

export const deleteReport = (id: number) => apiDelete<{ ok: boolean }>(`/reports/${id}`)

export const addReportItem = (id: number, item: ReportItemInput) =>
  apiPost<ReportItem>(`/reports/${id}/items`, item)

/**
 * Add many artifacts in one request (the Analysis screens' "Add all").
 * Idempotent server-side, so items the report already holds come back in
 * `skipped` rather than as an error, and the call is safe to retry.
 */
export const addReportItems = (id: number, items: ReportItemInput[], collection?: string | null) =>
  apiPost<ReportBatchResult>(`/reports/${id}/items/batch`, { items, collection: collection ?? null })

export const updateReportItem = (id: number, itemId: number, body: { note?: string | null }) =>
  apiPatch<ReportItem>(`/reports/${id}/items/${itemId}`, body)

export const removeReportItem = (id: number, itemId: number) =>
  apiDelete<{ ok: boolean }>(`/reports/${id}/items/${itemId}`)

export const reorderReportItems = (id: number, itemIds: number[]) =>
  apiPost<Report>(`/reports/${id}/items/reorder`, { item_ids: itemIds })

export const refreshCollectionOverview = (id: number) =>
  apiPost<Report>(`/reports/${id}/collection-overview/refresh`, {})

/**
 * Build an absolute URL for one of the report export endpoints. Use as the
 * `href` of a download/view anchor so the browser handles the response
 * natively (the `.html` form is served inline; the rest are attachments).
 *
 * A link bypasses `apiGet`, so it carries the admin owner context itself —
 * without it, an admin working in another user's namespace gets a 404.
 */
export function reportExportHref(id: number, format: ReportExportFormat): string {
  return url(withOwner(`/reports/${id}/export.${format}`))
}

/**
 * Queue a background render of the report's PDF.
 *
 * Rendering a large report outlasts the gateway's request timeout, so the PDF
 * is a job rather than a response; progress arrives on the shared job stream
 * and the result is fetched from {@link reportPdfHref}. A render already in
 * flight for this report is adopted (see {@link queueJob}).
 *
 * @param id - The report to render.
 * @returns The job id, and whether it was adopted from an in-flight render.
 */
export const queueReportPdf = (id: number) => queueJob(`/reports/${id}/pdf`, {})

/** The report's newest PDF job and its stored render, if any. */
export const reportPdfStatus = (id: number) => apiGet<ReportPdfStatus>(`/reports/${id}/pdf/status`)

/** Absolute URL of the report's stored PDF, for a download anchor's `href`. */
export function reportPdfHref(id: number): string {
  return url(withOwner(`/reports/${id}/pdf`))
}
