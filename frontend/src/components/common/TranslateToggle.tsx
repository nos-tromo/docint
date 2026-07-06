import { HoverIconAction, Spinner } from '@infra/ui'

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
  return (
    <HoverIconAction
      icon={busy ? <Spinner /> : <TranslateGlyph />}
      label={shown ? 'Show original' : 'Translate'}
      aria-pressed={shown}
      disabled={busy}
      onClick={onClick}
      className={className}
    />
  )
}
