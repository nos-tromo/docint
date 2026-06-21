import { useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { Button } from '@infra/ui'
import { cn } from '@/lib/cn'
import type { Report, ReportItemInput } from '@/api/types'
import { reportKey, useAddReportItem, useCreateReport, useRemoveReportItem } from '@/hooks/useReports'
import { useReportStore } from '@/stores/report'
import { useUiStore } from '@/stores/ui'

interface Props {
  item: ReportItemInput
  inReport: boolean
  className?: string
}

/**
 * Toggle one artifact into/out of the active report. With no active report,
 * the first add auto-creates an "Untitled report" scoped to the current
 * collection (one click, no modal). Removal looks the item up by dedupe key
 * from the cached report so the surrounding row never needs its own query.
 */
export function AddToReportButton({ item, inReport, className }: Props) {
  const qc = useQueryClient()
  const activeReportId = useReportStore((s) => s.activeReportId)
  const setActiveReportId = useReportStore((s) => s.setActiveReportId)
  const collection = useUiStore((s) => s.selectedCollection)
  const createReport = useCreateReport()
  const addItem = useAddReportItem()
  const removeItem = useRemoveReportItem()
  const [busy, setBusy] = useState(false)
  const [failed, setFailed] = useState(false)
  const pending = busy || createReport.isPending || addItem.isPending || removeItem.isPending

  async function handleClick() {
    if (pending) return
    setBusy(true)
    setFailed(false)
    try {
      if (inReport && activeReportId != null) {
        const report = qc.getQueryData<Report>(reportKey(activeReportId))
        const existing = report?.items.find((i) => i.dedupe_key === item.dedupe_key)
        if (existing) {
          await removeItem.mutateAsync({ reportId: activeReportId, itemId: existing.id })
        }
        return
      }
      let reportId = activeReportId
      if (reportId == null) {
        const created = await createReport.mutateAsync({
          title: 'Untitled report',
          collection_name: collection ?? undefined
        })
        reportId = created.id
        setActiveReportId(reportId)
      }
      await addItem.mutateAsync({ reportId, item })
    } catch (e) {
      console.error('Report action failed', e)
      setFailed(true)
    } finally {
      setBusy(false)
    }
  }

  const label = pending ? '…' : failed ? 'Retry' : inReport ? '✓ In report' : '+ Report'
  return (
    <Button
      type="button"
      variant={failed ? 'danger' : inReport ? 'secondary' : 'ghost'}
      size="sm"
      disabled={pending}
      aria-pressed={inReport}
      title={
        failed
          ? 'Could not reach the server — click to retry'
          : inReport
            ? 'Remove from report'
            : 'Add to report'
      }
      onClick={handleClick}
      className={cn('shrink-0 whitespace-nowrap', className)}
    >
      {label}
    </Button>
  )
}
