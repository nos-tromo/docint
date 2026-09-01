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
 * Dedupe keys the report holds whose snapshot carries no translation — the one
 * reason to send an item it already has, since that is the only amendment the
 * server makes to a frozen snapshot.
 */
export function useReportUntranslatedKeys(id: number | null): Set<string> {
  const { data } = useReport(id)
  return useMemo(
    () => new Set((data?.items ?? []).filter((i) => i.snapshot?.translation == null).map((i) => i.dedupe_key)),
    [data]
  )
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

/** Cap until `GET /config` lands (fetched once at mount), mirroring its default. */
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
  /** Items already in the report that gained a translation from this batch. */
  updated: number
}

/**
 * Add every finding of an Analysis section to the active report in one action.
 *
 * `fetchAll` walks the section's own cursor pages and `toItem` is the builder
 * its rows use, both reading the shared translations store — so a batched
 * snapshot is byte-identical to a hand-added one. The set posts as one
 * request; items already in the report are dropped first, bar those carrying a
 * translation its snapshot lacks, which the server backfills.
 *
 * The walk asks for one row past the cap because `fetchAllPages` truncates
 * silently: at exactly the cap, a section that just fits and one that
 * overflows look the same, and "Add all" would carry an arbitrary sample.
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
  const untranslatedKeys = useReportUntranslatedKeys(activeReportId)
  const addItems = useAddReportItems()
  const { data: config } = useConfig()
  const cap = Math.max(1, Math.trunc(config?.report_batch_max_items ?? ADD_ALL_MAX_ITEMS_FALLBACK))
  const [outcome, setOutcome] = useState<AddAllOutcome>({ status: 'idle', added: 0, skipped: 0, updated: 0 })

  const run = async (): Promise<void> => {
    setOutcome({ status: 'fetching', added: 0, skipped: 0, updated: 0 })
    try {
      const rows = await params.fetchAll(cap + 1)
      // Checked on the raw walk: the extra row means it stopped early, so no
      // subset of what came back can honestly be called "all".
      if (rows.length > cap) {
        setOutcome({ status: 'too_many', added: 0, skipped: 0, updated: 0 })
        return
      }
      const items: ReportItemInput[] = []
      const seen = new Set<string>()
      let skipped = 0
      for (const row of rows) {
        const item = params.toItem(row)
        // The same chunk twice in the fetched pages: the server dedupes too,
        // but not sending them keeps the request small and the confirmation
        // count honest.
        if (seen.has(item.dedupe_key)) {
          skipped += 1
          continue
        }
        // Already in the report, so normally not worth sending — unless it
        // carries a translation the stored snapshot lacks, the one amendment
        // the server will make.
        if (existingKeys.has(item.dedupe_key)) {
          const gainsTranslation =
            item.snapshot.translation != null && untranslatedKeys.has(item.dedupe_key)
          if (!gainsTranslation) {
            skipped += 1
            continue
          }
        }
        seen.add(item.dedupe_key)
        items.push({ ...item, collection: collection ?? null })
      }
      if (items.length === 0) {
        setOutcome({ status: 'done', added: 0, skipped, updated: 0 })
        return
      }
      if (
        items.length > ADD_ALL_CONFIRM_THRESHOLD &&
        !window.confirm(t('report.add_all_confirm', { count: items.length }))
      ) {
        setOutcome({ status: 'idle', added: 0, skipped: 0, updated: 0 })
        return
      }
      setOutcome({ status: 'adding', added: 0, skipped: 0, updated: 0 })
      const reportId = await ensureReport()
      const result = await addItems.mutateAsync({ reportId, items, collection })
      setOutcome({ status: 'done', added: result.added, skipped: skipped + result.skipped, updated: result.updated })
    } catch (e) {
      console.error('Add all to report failed', e)
      // nginx refuses an oversize body before FastAPI sees it, so a retry
      // cannot succeed — say it is too big rather than offering one.
      if (e instanceof ApiError && e.status === 413) {
        setOutcome({ status: 'too_large', added: 0, skipped: 0, updated: 0 })
        return
      }
      setOutcome({ status: 'failed', added: 0, skipped: 0, updated: 0 })
    }
  }

  const reset = () => setOutcome({ status: 'idle', added: 0, skipped: 0, updated: 0 })
  return { run, reset, cap, ...outcome }
}
