import { useState } from 'react'
import type { SummaryDiagnostics } from '@/api/types'
import { cn } from '@/lib/cn'
import { useT } from '@/i18n/LanguageContext'

export function CoverageBanner({ d }: { d: SummaryDiagnostics }) {
  const t = useT()
  const [open, setOpen] = useState(false)
  const ratioPct = Math.round((d.coverage_ratio ?? 0) * 100)
  const targetPct = Math.round((d.coverage_target ?? 0) * 100)
  const tone =
    ratioPct >= targetPct
      ? 'border-emerald-700 bg-emerald-950 text-emerald-200'
      : 'border-amber-700 bg-amber-950 text-amber-200'

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
