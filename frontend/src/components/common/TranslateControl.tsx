import { cn } from '@/lib/cn'
import { useTranslatable, type TranslationPayload } from '@/hooks/useTranslatable'
import { useT } from '@/i18n/LanguageContext'
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
  const translatable = useTranslatable(rawText, onTranslated)
  const t = useT()
  const body = translatable.translation ?? rawText
  return (
    <div className={cn('group relative rounded bg-muted/70 p-2.5 pr-9 text-xs', className)}>
      <TranslateToggle
        shown={translatable.shown}
        busy={translatable.busy}
        onClick={translatable.toggle}
        className="absolute right-1 top-1"
      />
      {translatable.shown && (
        <div className="mb-1 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
          {t('common.translation')}
        </div>
      )}
      <ClampedText length={body.length}>{body}</ClampedText>
      {translatable.failed && (
        <div className="mt-1 text-[11px] text-muted-foreground">{t('common.translation_unavailable')}</div>
      )}
    </div>
  )
}
