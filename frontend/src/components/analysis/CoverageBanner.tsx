import { useState } from 'react'
import type { SummaryDiagnostics } from '@/api/types'
import { cn } from '@/lib/cn'
import { useT } from '@/i18n/LanguageContext'

export function CoverageBanner({ d }: { d: SummaryDiagnostics }) {
  const t = useT()
  const [open, setOpen] = useState(false)
  const ratioPct = Math.round((d.coverage_ratio ?? 0) * 100)
  const targetPct = Math.round((d.coverage_target ?? 0) * 100)
  // A capped build can still report full coverage (the cap can bite *inside*
  // one unit's windows, leaving every unit "covered"), so `partial` must
  // drive the tone on its own — the ratio alone would show green.
  const partial = d.partial === true
  const tone =
    ratioPct >= targetPct && !partial
      ? 'border-[var(--status-emerald-border)] bg-[var(--status-emerald-surface)] text-[var(--status-emerald-strong)]'
      : 'border-[var(--status-amber-border)] bg-[var(--status-amber-surface)] text-[var(--status-amber-strong)]'

  return (
    <div className={cn('rounded-md border px-3 py-2 text-xs', tone)}>
      <div className="flex items-center justify-between">
        <div>
          {t('analysis.coverage_label')}{' '}
          <span className="font-medium">
            {d.covered_documents}/{d.total_documents}
          </span>{' '}
          {t('analysis.coverage_documents_pct', { pct: ratioPct, targetPct })}
          <span className="ml-2 text-muted-foreground">
            {t('analysis.coverage_sampled', {
              sampled: d.sampled_count,
              candidate: d.candidate_count,
              deduped: d.deduped_count
            })}
          </span>
        </div>
        {d.uncovered_documents.length > 0 && (
          <button
            type="button"
            className="underline text-[11px]"
            onClick={() => setOpen((v) => !v)}
          >
            {open
              ? t('analysis.coverage_hide')
              : t('analysis.coverage_show_uncovered', { count: d.uncovered_documents.length })}
          </button>
        )}
      </div>
      {partial && (
        <div className="mt-2 text-[11px]" data-testid="coverage-partial-notice">
          <span className="font-medium">{t('analysis.coverage_partial_label')}</span>{' '}
          {t('analysis.coverage_partial_detail')}
        </div>
      )}
      {open && d.uncovered_documents.length > 0 && (
        <ul className="mt-2 max-h-40 overflow-auto space-y-0.5 text-[11px]">
          {d.uncovered_documents.map((f) => (
            <li key={f}>{f}</li>
          ))}
        </ul>
      )}
    </div>
  )
}
