import { cn } from '@/lib/cn'
import { useTranslatable, type TranslationPayload } from '@/hooks/useTranslatable'
import { TranslateToggle } from './TranslateToggle'
import { ClampedText } from './ClampedText'

export type { TranslationPayload }

interface Props {
  /** The source text to translate (and the default original view). */
  rawText: string
  onTranslated?: (t: TranslationPayload | null) => void
  className?: string
}

/**
 * A source snippet with an in-place Translate toggle: the hover/focus icon flips
 * the clamped text between the original and its translation (original one tap
 * back). For split layouts where the icon and text live in different cells, use
 * `useTranslatable` + `TranslateToggle` + `ClampedText` directly.
 */
export function TranslateControl({ rawText, onTranslated, className }: Props) {
  const t = useTranslatable(rawText, onTranslated)
  const body = t.translation ?? rawText
  return (
    <div className={cn('group relative rounded bg-zinc-950/70 p-2.5 pr-9 text-xs', className)}>
      <TranslateToggle shown={t.shown} busy={t.busy} onClick={t.toggle} className="absolute right-1 top-1" />
      {t.shown && (
        <div className="mb-1 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">Translation</div>
      )}
      <ClampedText length={body.length}>{body}</ClampedText>
      {t.failed && (
        <div className="mt-1 text-[11px] text-muted-foreground">Translation unavailable — showing original.</div>
      )}
    </div>
  )
}
