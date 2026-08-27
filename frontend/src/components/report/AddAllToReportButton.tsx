import { IconButton, WarningIcon } from '@infra/ui'
import type { ReportItemInput } from '@/api/types'
import { CheckAllIcon } from '@/components/common/icons'
import { ADD_ALL_MAX_ITEMS, useAddAllToReport } from '@/hooks/useReports'
import { useT } from '@/i18n/LanguageContext'

interface Props<Row> {
  /** Walk the section's cursor pages and return every matching row. */
  fetchAll: () => Promise<Row[]>
  /** The same pure snapshot builder the section's rows use. */
  toItem: (row: Row) => ReportItemInput
  /** Whether the section currently has anything to add. */
  hasRows: boolean
  className?: string
}

/**
 * Add every finding of an Analysis section to the active report at once.
 *
 * The section-wide counterpart of the per-row `AddToReportButton`, sitting in
 * the section header beside its CSV download. "All" means every finding the
 * section's filter matches — not only the rows paged in — because the tables
 * load 50 at a time and a report built from what happened to be scrolled into
 * view would be a silent sample.
 *
 * The button knows nothing about entities vs. hate speech: the section hands
 * it the page walk and the snapshot builder its own rows use.
 */
export function AddAllToReportButton<Row>({ fetchAll, toItem, hasRows, className }: Props<Row>) {
  const t = useT()
  const { run, status, added, skipped } = useAddAllToReport<Row>({ fetchAll, toItem })
  const busy = status === 'fetching' || status === 'adding'
  const failed = status === 'failed'

  // One line of plain text beside the button — the SPA has no toast layer, and
  // the outcome is worth stating: "0 added, 40 already in report" is a
  // different answer from "40 added", and both look identical in the rows.
  let message: string | null = null
  if (status === 'too_many') message = t('report.add_all_too_many', { max: ADD_ALL_MAX_ITEMS })
  else if (failed) message = t('report.add_all_failed')
  else if (status === 'done') {
    message = added === 0 && skipped > 0 ? t('report.add_all_none') : t('report.add_all_done', { added, skipped })
  }

  return (
    <div className={`flex items-center gap-2 ${className ?? ''}`}>
      {message && (
        <span
          className={`text-xs ${failed || status === 'too_many' ? 'text-destructive' : 'text-muted-foreground'}`}
          role="status"
          data-testid="add-all-message"
        >
          {message}
        </span>
      )}
      <IconButton
        icon={failed ? <WarningIcon /> : <CheckAllIcon />}
        label={failed ? t('report.retry') : t('report.add_all')}
        hint={busy ? t('report.add_all_busy') : undefined}
        variant={failed ? 'danger' : 'ghost'}
        busy={busy}
        disabled={!hasRows || busy}
        onClick={run}
      />
    </div>
  )
}
