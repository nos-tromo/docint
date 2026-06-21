import { useMemo } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  addReportItem,
  createReport,
  deleteReport,
  getReport,
  listReports,
  removeReportItem,
  reorderReportItems,
  updateReport,
  updateReportItem
} from '@/api/reports'
import type { ReportItemInput } from '@/api/types'

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
    mutationFn: ({ id, ...body }: { id: number; title?: string; operator?: string; reference_number?: string }) =>
      updateReport(id, body),
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

/** Dedupe keys already in the active report, for the "already added" UI state. */
export function useReportDedupeKeys(id: number | null): Set<string> {
  const { data } = useReport(id)
  return useMemo(() => new Set((data?.items ?? []).map((i) => i.dedupe_key)), [data])
}
