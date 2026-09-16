import { cn } from '@/lib/cn'
import { visiblePreprocessStages } from '@/lib/ingestStatus'
import type { PreprocessStage } from '@/api/jobs'
import { useT } from '@/i18n/LanguageContext'
import type { Strings } from '@/i18n'

export type Tone = 'sky' | 'amber' | 'emerald' | 'red'

/** One labelled counter with a bar under it. */
export interface BarItem {
  key: string
  label: string
  current: number
  total: number
  /** Files whose task failed and are waiting to be retried, if any. */
  failed?: number
}

/**
 * A filled proportion bar.
 *
 * @param value - Units done.
 * @param max - Units in total; a zero max draws an empty bar rather than dividing by it.
 * @param tone - Which status colour to fill with.
 */
export function Bar({ value, max, tone }: { value: number; max: number; tone: Tone }) {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0
  const fill =
    tone === 'sky'
      ? 'bg-sky-500'
      : tone === 'amber'
        ? 'bg-amber-500'
        : tone === 'emerald'
          ? 'bg-emerald-500'
          : 'bg-red-500'
  return (
    <div className="h-1.5 w-full rounded-full bg-muted overflow-hidden">
      <div
        className={cn('h-full transition-[width] duration-300 ease-out', fill)}
        style={{ width: `${pct}%` }}
      />
    </div>
  )
}

/**
 * A stack of labelled bars — one per counter a run is reporting.
 *
 * @param items - The counters, already in render order.
 */
export function BarList({ items }: { items: BarItem[] }) {
  const t = useT()
  if (items.length === 0) return null
  return (
    <div className="space-y-2">
      {items.map((item) => (
        <div key={item.key} className="space-y-1">
          <div className="flex items-baseline justify-between gap-2">
            <span className="text-sm text-foreground">{item.label}</span>
            <span className="tabular-nums text-xs text-muted-foreground">
              {item.current}/{item.total}
            </span>
          </div>
          <Bar value={item.current} max={item.total || 1} tone="amber" />
          {item.failed !== undefined && item.failed > 0 && (
            <div className="text-xs text-[var(--status-amber-fg)]">
              {t('ingest.preprocess_failed', { count: item.failed })}
            </div>
          )}
        </div>
      ))}
    </div>
  )
}

// Stage ids are protocol, English in every locale (see the backend's
// StagedStageOut); an id with no catalog entry falls back to itself rather
// than rendering blank, mirroring the task-label mapping.
const PREPROCESS_LABEL_KEY: Partial<Record<string, keyof Strings>> = {
  pdf: 'ingest.preprocess_pdf',
  ocr_pages: 'ingest.preprocess_ocr_pages',
  image: 'ingest.preprocess_image',
  media: 'ingest.preprocess_media',
  keyframes: 'ingest.preprocess_keyframes'
}

/**
 * Turn the server's preprocessing tally into bars.
 *
 * @param stages - The tally as polled, in any order.
 * @param t - The catalog lookup.
 * @returns One item per stage with work in it.
 */
export function preprocessBarItems(
  stages: PreprocessStage[] | undefined,
  t: (key: keyof Strings) => string
): BarItem[] {
  return visiblePreprocessStages(stages).map((stage) => {
    const key = PREPROCESS_LABEL_KEY[stage.stage]
    return {
      key: stage.stage,
      label: key ? t(key) : stage.stage,
      current: stage.done,
      total: stage.total,
      failed: stage.failed
    }
  })
}
