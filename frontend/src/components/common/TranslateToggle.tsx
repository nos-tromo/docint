import { HoverIconAction, Spinner } from '@infra/ui'
import { useT } from '@/i18n/LanguageContext'

const TranslateGlyph = () => (
  <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
    <path d="M4 5h9M9 3v2c0 4-2 7-6 8M5 9c0 3 3 5 7 5" />
    <path d="m13 21 4-9 4 9M15.5 17h5" />
  </svg>
)

/** The hover/focus-revealed Translate icon. Mount inside a `.group` container. */
export function TranslateToggle({
  shown,
  busy,
  onClick,
  className
}: {
  shown: boolean
  busy: boolean
  onClick: () => void
  className?: string
}) {
  const t = useT()
  return (
    <HoverIconAction
      icon={busy ? <Spinner label={t('common.loading_ellipsis')} /> : <TranslateGlyph />}
      label={shown ? t('common.show_original') : t('common.translate')}
      aria-pressed={shown}
      disabled={busy}
      onClick={onClick}
      className={className}
    />
  )
}
