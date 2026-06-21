import { apiDelete, apiGet, apiPatch, apiPost, url } from './client'
import type { Report, ReportExportFormat, ReportItem, ReportItemInput, ReportSummary } from './types'

export const listReports = (collection?: string) =>
  apiGet<{ reports: ReportSummary[] }>('/reports', collection ? { collection } : undefined)

export const getReport = (id: number) => apiGet<Report>(`/reports/${id}`)

export const createReport = (body: { title: string; collection_name?: string | null; session_id?: string | null }) =>
  apiPost<Report>('/reports', body)

export const updateReport = (
  id: number,
  body: { title?: string; operator?: string; reference_number?: string }
) => apiPatch<Report>(`/reports/${id}`, body)

export const deleteReport = (id: number) => apiDelete<{ ok: boolean }>(`/reports/${id}`)

export const addReportItem = (id: number, item: ReportItemInput) =>
  apiPost<ReportItem>(`/reports/${id}/items`, item)

export const updateReportItem = (id: number, itemId: number, body: { note?: string | null }) =>
  apiPatch<ReportItem>(`/reports/${id}/items/${itemId}`, body)

export const removeReportItem = (id: number, itemId: number) =>
  apiDelete<{ ok: boolean }>(`/reports/${id}/items/${itemId}`)

export const reorderReportItems = (id: number, itemIds: number[]) =>
  apiPost<Report>(`/reports/${id}/items/reorder`, { item_ids: itemIds })

/**
 * Build an absolute URL for one of the report export endpoints. Use as the
 * `href` of a download/view anchor so the browser handles the response
 * natively (the `.html` form is served inline; the rest are attachments).
 */
export function reportExportHref(id: number, format: ReportExportFormat): string {
  return url(`/reports/${id}/export.${format}`)
}
