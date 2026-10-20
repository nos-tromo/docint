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

/**
 * Scope a report path to the caller's active collection.
 *
 * A report is reachable only while its collection is the active one: every
 * report route takes it, and answers 404 for a report of another collection.
 */
function inCollection(path: string, collection: string): string {
  return `${path}?collection=${encodeURIComponent(collection)}`
}

export const listReports = (collection: string) =>
  apiGet<{ reports: ReportSummary[] }>('/reports', { collection })

export const getReport = (collection: string, id: number) =>
  apiGet<Report>(inCollection(`/reports/${id}`, collection))

/** The caller's reports kept from before every report named its collection — the one listing outside a collection. */
export const listUnassignedReports = () => apiGet<{ reports: ReportSummary[] }>('/reports/unassigned')

/** Place a report without a collection into `collection`. The server assigns only such a report, once. */
export const assignReport = (id: number, collection: string) =>
  apiPost<Report>(inCollection(`/reports/unassigned/${id}/assign`, collection))

export const createReport = (body: {
  title: string
  collection_name: string
  session_id?: string | null
  operator?: string
}) => apiPost<Report>('/reports', body)

export const updateReport = (
  collection: string,
  id: number,
  body: { title?: string; operator?: string; reference_number?: string; show_toc?: boolean; show_collection_overview?: boolean }
) => apiPatch<Report>(inCollection(`/reports/${id}`, collection), body)

export const deleteReport = (collection: string, id: number) =>
  apiDelete<{ ok: boolean }>(inCollection(`/reports/${id}`, collection))

export const addReportItem = (collection: string, id: number, item: ReportItemInput) =>
  apiPost<ReportItem>(inCollection(`/reports/${id}/items`, collection), item)

/**
 * Add many artifacts in one request (the Analysis screens' "Add all").
 * Idempotent server-side, so items the report already holds come back in
 * `skipped` rather than as an error, and the call is safe to retry.
 */
export const addReportItems = (collection: string, id: number, items: ReportItemInput[]) =>
  apiPost<ReportBatchResult>(inCollection(`/reports/${id}/items/batch`, collection), { items })

export const updateReportItem = (collection: string, id: number, itemId: number, body: { note?: string | null }) =>
  apiPatch<ReportItem>(inCollection(`/reports/${id}/items/${itemId}`, collection), body)

export const removeReportItem = (collection: string, id: number, itemId: number) =>
  apiDelete<{ ok: boolean }>(inCollection(`/reports/${id}/items/${itemId}`, collection))

export const reorderReportItems = (collection: string, id: number, itemIds: number[]) =>
  apiPost<Report>(inCollection(`/reports/${id}/items/reorder`, collection), { item_ids: itemIds })

export const refreshCollectionOverview = (collection: string, id: number) =>
  apiPost<Report>(inCollection(`/reports/${id}/collection-overview/refresh`, collection), {})

/**
 * Build an absolute URL for one of the report export endpoints. Use as the
 * `href` of a download/view anchor so the browser handles the response
 * natively (the `.html` form is served inline; the rest are attachments).
 *
 * A link bypasses `apiGet`, so it carries the admin owner context itself —
 * without it, an admin working in another user's namespace gets a 404.
 */
export function reportExportHref(collection: string, id: number, format: ReportExportFormat): string {
  return url(withOwner(inCollection(`/reports/${id}/export.${format}`, collection)))
}

/**
 * Queue a background render of the report's PDF.
 *
 * Rendering a large report outlasts the gateway's request timeout, so the PDF
 * is a job rather than a response; progress arrives on the shared job stream
 * and the result is fetched from {@link reportPdfHref}. A render already in
 * flight for this report is adopted (see {@link queueJob}).
 *
 * @param collection - The caller's active collection, which the report must belong to.
 * @param id - The report to render.
 * @returns The job id, and whether it was adopted from an in-flight render.
 */
export const queueReportPdf = (collection: string, id: number) =>
  queueJob(inCollection(`/reports/${id}/pdf`, collection), {})

/** The report's newest PDF job and its stored render, if any. */
export const reportPdfStatus = (collection: string, id: number) =>
  apiGet<ReportPdfStatus>(inCollection(`/reports/${id}/pdf/status`, collection))

/** Absolute URL of the report's stored PDF, for a download anchor's `href`. */
export function reportPdfHref(collection: string, id: number): string {
  return url(withOwner(inCollection(`/reports/${id}/pdf`, collection)))
}
