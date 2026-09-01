import { HoverIconAction, Spinner } from '@infra/ui'
import { TranslateIcon } from '@/components/common/icons'
import { useT } from '@/i18n/LanguageContext'

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
      icon={busy ? <Spinner label={t('common.loading_ellipsis')} /> : <TranslateIcon />}
      label={shown ? t('common.show_original') : t('common.translate')}
      aria-pressed={shown}
      disabled={busy}
      onClick={onClick}
      className={className}
    />
  )
}
