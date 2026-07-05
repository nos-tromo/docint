import { useState } from 'react'
import { HoverIconAction, Spinner } from '@infra/ui'
import { useTranslate } from '@/hooks/useTranslate'

const TranslateGlyph = () => (
  <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
    <path d="M4 5h9M9 3v2c0 4-2 7-6 8M5 9c0 3 3 5 7 5" />
    <path d="m13 21 4-9 4 9M15.5 17h5" />
  </svg>
)

export interface TranslationPayload {
  text: string
  target_lang: string
  model: string
}

interface Props {
  /** The source text to translate (the caller already holds it). */
  text: string
  /** Called with the translation once available (for report carry), or null on reset. */
  onTranslated?: (t: TranslationPayload | null) => void
  className?: string
}

/**
 * Hover/focus-revealed "Translate" icon. On click it fetches a machine
 * translation and shows it below the original (which is never replaced); a
 * second click toggles back. Mount inside a `.group` container so the icon
 * reveals on hover/focus. In-app copy is "Translate"/"Translation" — the
 * report export keeps the fuller "Machine translation" qualifier.
 */
export function TranslateControl({ text, onTranslated, className }: Props) {
  const [shown, setShown] = useState(false)
  const { mutateAsync, data, status } = useTranslate()
  const busy = status === 'pending'
  const failed = status === 'error' || (data && !data.ok)

  async function handleClick() {
    if (shown) {
      setShown(false)
      return
    }
    if (data?.ok && data.translation != null) {
      setShown(true)
      return
    }
    const res = await mutateAsync(text)
    if (res.ok && res.translation != null) {
      setShown(true)
      onTranslated?.({ text: res.translation, target_lang: res.target_lang, model: res.model })
    }
  }

  return (
    <>
      <HoverIconAction
        icon={busy ? <Spinner /> : <TranslateGlyph />}
        label={shown ? 'Show original' : 'Translate'}
        aria-pressed={shown}
        disabled={busy}
        onClick={handleClick}
        className={className}
      />
      {shown && data?.translation != null && (
        <div className="mt-1 border-l-2 border-border pl-2">
          <div className="text-xs uppercase text-muted-foreground">Translation</div>
          <div className="whitespace-pre-wrap">{data.translation}</div>
        </div>
      )}
      {failed && <div className="mt-1 text-xs text-muted-foreground">Translation unavailable — showing original.</div>}
    </>
  )
}
