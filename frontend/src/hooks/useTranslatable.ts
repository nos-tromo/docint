import { useState } from 'react'
import { useTranslate } from '@/hooks/useTranslate'

export interface TranslationPayload {
  text: string
  target_lang: string
  model: string
}

export interface Translatable {
  /** True when the translation is shown in place of the original. */
  shown: boolean
  /** True while a translation request is in flight. */
  busy: boolean
  /** True when the last attempt failed (transport error or ok:false). */
  failed: boolean
  /** Translated text to render when `shown`, else null. */
  translation: string | null
  /** Fetch+show, or hide if already shown. */
  toggle: () => void
}

/**
 * Owns the translate-fetch + show/hide state for one snippet so the toggle icon
 * and the swapped text can live in different DOM nodes while sharing state.
 * Fail-soft: never throws. `onTranslated` fires once per successful fetch with
 * the nested payload (for report carry).
 */
export function useTranslatable(
  rawText: string,
  onTranslated?: (t: TranslationPayload | null) => void
): Translatable {
  const [shown, setShown] = useState(false)
  const { mutateAsync, data, status } = useTranslate()
  const busy = status === 'pending'
  const failed = status === 'error' || (data != null && !data.ok)
  const translation = shown && data?.ok ? (data.translation ?? null) : null

  async function toggle() {
    if (shown) {
      setShown(false)
      return
    }
    if (data?.ok && data.translation != null) {
      setShown(true)
      return
    }
    try {
      const res = await mutateAsync(rawText)
      if (res.ok && res.translation != null) {
        setShown(true)
        onTranslated?.({ text: res.translation, target_lang: res.target_lang, model: res.model })
      }
    } catch {
      // A true transport failure flips status to 'error', which drives `failed`.
    }
  }

  return { shown, busy, failed, translation, toggle }
}
