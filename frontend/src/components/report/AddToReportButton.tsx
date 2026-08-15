import { useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { IconButton, ReportCheckIcon, ReportIcon, WarningIcon } from '@infra/ui'
import type { Report, ReportItemInput } from '@/api/types'
import { reportKey, useAddReportItem, useCreateReport, useRemoveReportItem } from '@/hooks/useReports'
import { useReportStore } from '@/stores/report'
import { useUiStore } from '@/stores/ui'
import { useT } from '@/i18n/LanguageContext'

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
  const t = useT()
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
          title: t('report.untitled_title'),
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

  // Icon-only, like the translate and preview actions it sits beside: the
  // accessible name carries the verb ("Add to report"), the drawing carries the
  // state — a page for "not yet", the same page with a check for "in report",
  // so the toggle reads without hovering for the tooltip. A failed round-trip
  // swaps in the warning marker and tints danger; the label then says "Retry".
  const label = failed ? t('report.retry') : inReport ? t('report.in_report') : t('report.add_title')
  const hint = failed ? t('report.retry_title') : inReport ? t('report.remove_title') : undefined
  const Icon = failed ? WarningIcon : inReport ? ReportCheckIcon : ReportIcon
  return (
    <IconButton
      icon={<Icon />}
      label={label}
      hint={hint}
      variant={failed ? 'danger' : inReport ? 'secondary' : 'ghost'}
      busy={pending}
      aria-pressed={inReport}
      onClick={handleClick}
      className={className}
    />
  )
}
