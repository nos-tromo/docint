import { useT } from '@/i18n/LanguageContext'

export interface ScopeBannerProps {
  /** How many chunks the session's answers are restricted to. */
  count: number
  /** Scoped chunks the backend could no longer find, as it last reported. */
  missing?: number
  onClear: () => void
}

/**
 * "Scoped to N chunks · clear", above the transcript.
 *
 * A scoped answer must never be a surprise: while a scope is pinned, every
 * reply comes from the hand-picked chunks alone and from no other part of the
 * collection. When the backend reports scoped chunks it can no longer find —
 * re-ingestion mints new point ids, so a scope can outlive its chunks — the
 * shortfall is stated rather than left to quietly narrow the evidence.
 *
 * Renders nothing when there is no scope, so callers need no length guard.
 */
export function ScopeBanner({ count, missing = 0, onClear }: ScopeBannerProps) {
  const t = useT()
  if (count <= 0) return null
  return (
    <div
      className="mb-3 flex flex-wrap items-center gap-x-2 gap-y-1 rounded-md border border-border bg-muted px-3 py-2 text-xs"
      data-testid="scope-banner"
    >
      <span>{t('search.scope_banner', { count })}</span>
      <span aria-hidden="true" className="text-muted-foreground">
        ·
      </span>
      <button
        type="button"
        onClick={onClear}
        className="text-muted-foreground underline-offset-2 hover:text-foreground hover:underline"
      >
        {t('search.clear')}
      </button>
      {missing > 0 && (
        <span className="w-full text-red-500" data-testid="scope-missing">
          {t('search.scope_missing', { missing, count })}
        </span>
      )}
    </div>
  )
}
