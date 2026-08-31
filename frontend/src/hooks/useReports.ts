import { useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  addReportItem,
  addReportItems,
  createReport,
  deleteReport,
  getReport,
  listReports,
  refreshCollectionOverview,
  removeReportItem,
  reorderReportItems,
  updateReport,
  updateReportItem
} from '@/api/reports'
import type { ReportItemInput } from '@/api/types'
import { ApiError } from '@/api/client'
import { useConfig } from '@/hooks/useConfig'
import { useReportStore } from '@/stores/report'
import { useUiStore } from '@/stores/ui'
import { useT } from '@/i18n/LanguageContext'

export const reportsKey = ['reports'] as const
export const reportKey = (id: number) => ['reports', id] as const

export function useReports(collection?: string) {
  return useQuery({
    queryKey: collection ? (['reports', { collection }] as const) : reportsKey,
    queryFn: () => listReports(collection)
  })
}

export function useReport(id: number | null) {
  return useQuery({
    queryKey: id != null ? reportKey(id) : ['reports', 'none'],
    queryFn: () => getReport(id as number),
    enabled: id != null
  })
}

// Every mutation invalidates the `['reports']` prefix, which React Query
// matches against the list, per-id, and collection-filtered queries alike.
function useReportInvalidator() {
  const qc = useQueryClient()
  return () => qc.invalidateQueries({ queryKey: reportsKey })
}

export function useCreateReport() {
  const invalidate = useReportInvalidator()
  return useMutation({ mutationFn: createReport, onSuccess: invalidate })
}

export function useUpdateReport() {
  const invalidate = useReportInvalidator()
  return useMutation({
    mutationFn: ({
      id,
      ...body
    }: {
      id: number
      title?: string
      operator?: string
      reference_number?: string
      show_toc?: boolean
      show_collection_overview?: boolean
    }) => updateReport(id, body),
    onSuccess: invalidate
  })
}

export function useDeleteReport() {
  const invalidate = useReportInvalidator()
  return useMutation({ mutationFn: (id: number) => deleteReport(id), onSuccess: invalidate })
}

export function useAddReportItem() {
  const invalidate = useReportInvalidator()
  return useMutation({
    mutationFn: ({ reportId, item }: { reportId: number; item: ReportItemInput }) => addReportItem(reportId, item),
    onSuccess: invalidate
  })
}

export function useAddReportItems() {
  const invalidate = useReportInvalidator()
  return useMutation({
    mutationFn: ({
      reportId,
      items,
      collection
    }: {
      reportId: number
      items: ReportItemInput[]
      collection?: string | null
    }) => addReportItems(reportId, items, collection),
    onSuccess: invalidate
  })
}

export function useRemoveReportItem() {
  const invalidate = useReportInvalidator()
  return useMutation({
    mutationFn: ({ reportId, itemId }: { reportId: number; itemId: number }) => removeReportItem(reportId, itemId),
    onSuccess: invalidate
  })
}

export function useUpdateReportItem() {
  const invalidate = useReportInvalidator()
  return useMutation({
    mutationFn: ({ reportId, itemId, note }: { reportId: number; itemId: number; note: string | null }) =>
      updateReportItem(reportId, itemId, { note }),
    onSuccess: invalidate
  })
}

export function useReorderReportItems() {
  const invalidate = useReportInvalidator()
  return useMutation({
    mutationFn: ({ reportId, itemIds }: { reportId: number; itemIds: number[] }) =>
      reorderReportItems(reportId, itemIds),
    onSuccess: invalidate
  })
}

export function useRefreshCollectionOverview() {
  const invalidate = useReportInvalidator()
  return useMutation({ mutationFn: (id: number) => refreshCollectionOverview(id), onSuccess: invalidate })
}

/** Dedupe keys already in the active report, for the "already added" UI state. */
export function useReportDedupeKeys(id: number | null): Set<string> {
  const { data } = useReport(id)
  return useMemo(() => new Set((data?.items ?? []).map((i) => i.dedupe_key)), [data])
}

/**
 * Resolve the report an add should land in, creating one if none is active.
 *
 * The first add from anywhere in the app auto-creates an "Untitled report"
 * scoped to the current collection (one click, no modal). Shared by the
 * per-row toggle and the section-wide "Add all" so the two cannot drift.
 */
export function useEnsureActiveReport() {
  const t = useT()
  const activeReportId = useReportStore((s) => s.activeReportId)
  const setActiveReportId = useReportStore((s) => s.setActiveReportId)
  const collection = useUiStore((s) => s.selectedCollection)
  const createReport = useCreateReport()
  return async (): Promise<number> => {
    if (activeReportId != null) return activeReportId
    const created = await createReport.mutateAsync({
      title: t('report.untitled_title'),
      collection_name: collection ?? undefined
    })
    setActiveReportId(created.id)
    return created.id
  }
}

/** Above this many items, "Add all" asks before it commits. */
export const ADD_ALL_CONFIRM_THRESHOLD = 100

/**
 * Cap used until `GET /config` lands, mirroring the server's own default.
 * `useConfig` is fetched at app mount and never refetched, so this stands in
 * only for a click in the first instants of a session.
 */
export const ADD_ALL_MAX_ITEMS_FALLBACK = 2000

export type AddAllStatus =
  | 'idle'
  | 'fetching'
  | 'adding'
  | 'done'
  | 'failed'
  | 'too_many'
  | 'too_large'

export interface AddAllOutcome {
  status: AddAllStatus
  added: number
  skipped: number
}

/**
 * Add every finding of an Analysis section to the active report in one action.
 *
 * The caller supplies `fetchAll` — a walk of the section's own cursor pages,
 * bounded by the item count it is given — and `toItem`, the same pure snapshot
 * builder its rows use. Both read translations from the shared translations
 * store, so a batched snapshot is byte-identical to a hand-added one,
 * `translation` included.
 *
 * The whole set is fetched outside React Query (the rendered list is not
 * force-expanded), pre-filtered against what the report already holds, and
 * posted as one request — one round-trip and one cache invalidation, not one
 * per finding. Above `ADD_ALL_CONFIRM_THRESHOLD` items it confirms first.
 *
 * The cap comes from `GET /config` (the server's own `REPORT_BATCH_MAX_ITEMS`),
 * and the walk asks for one row *past* it: `fetchAllPages` truncates silently,
 * so fetching exactly the cap could not tell a section that just fits from one
 * that does not, and "Add all" would carry an arbitrary sample of a larger set.
 */
export function useAddAllToReport<Row>(params: {
  fetchAll: (maxItems: number) => Promise<Row[]>
  toItem: (row: Row) => ReportItemInput
}) {
  const t = useT()
  const ensureReport = useEnsureActiveReport()
  const activeReportId = useReportStore((s) => s.activeReportId)
  const collection = useUiStore((s) => s.selectedCollection)
  const existingKeys = useReportDedupeKeys(activeReportId)
  const addItems = useAddReportItems()
  const { data: config } = useConfig()
  const cap = Math.max(1, Math.trunc(config?.report_batch_max_items ?? ADD_ALL_MAX_ITEMS_FALLBACK))
  const [outcome, setOutcome] = useState<AddAllOutcome>({ status: 'idle', added: 0, skipped: 0 })

  const run = async (): Promise<void> => {
    setOutcome({ status: 'fetching', added: 0, skipped: 0 })
    try {
      const rows = await params.fetchAll(cap + 1)
      // Checked before dedupe, on the raw walk: the extra row means the walk
      // stopped early, so how much of the section is missing is unknowable and
      // no subset of it can honestly be called "all".
      if (rows.length > cap) {
        setOutcome({ status: 'too_many', added: 0, skipped: 0 })
        return
      }
      const items: ReportItemInput[] = []
      const seen = new Set<string>()
      let skipped = 0
      for (const row of rows) {
        const item = params.toItem(row)
        // Already in the report, or the same chunk twice in the fetched pages:
        // the server dedupes too, but not sending them keeps the request small
        // and the confirmation count honest.
        if (existingKeys.has(item.dedupe_key) || seen.has(item.dedupe_key)) {
          skipped += 1
          continue
        }
        seen.add(item.dedupe_key)
        items.push({ ...item, collection: collection ?? null })
      }
      if (items.length === 0) {
        setOutcome({ status: 'done', added: 0, skipped })
        return
      }
      if (
        items.length > ADD_ALL_CONFIRM_THRESHOLD &&
        !window.confirm(t('report.add_all_confirm', { count: items.length }))
      ) {
        setOutcome({ status: 'idle', added: 0, skipped: 0 })
        return
      }
      setOutcome({ status: 'adding', added: 0, skipped: 0 })
      const reportId = await ensureReport()
      const result = await addItems.mutateAsync({ reportId, items, collection })
      setOutcome({ status: 'done', added: result.added, skipped: skipped + result.skipped })
    } catch (e) {
      console.error('Add all to report failed', e)
      // nginx refuses an oversize body before FastAPI ever sees it, so the
      // same request cannot succeed on a retry — say it is too big instead of
      // offering the retry the generic failure wears.
      if (e instanceof ApiError && e.status === 413) {
        setOutcome({ status: 'too_large', added: 0, skipped: 0 })
        return
      }
      setOutcome({ status: 'failed', added: 0, skipped: 0 })
    }
  }

  const reset = () => setOutcome({ status: 'idle', added: 0, skipped: 0 })
  return { run, reset, cap, ...outcome }
}
