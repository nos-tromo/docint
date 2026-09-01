import { useState } from 'react'
import { useTranslate } from '@/hooks/useTranslate'
import { useTranslationsStore, type TranslationPayload } from '@/stores/translations'

export type { TranslationPayload }

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
 * the nested payload.
 *
 * The text lives in the shared translations store keyed by `rawText`; only
 * the show/hide flag is local, so a row keeps its translation across a remount.
 */
export function useTranslatable(
  rawText: string,
  onTranslated?: (t: TranslationPayload | null) => void
): Translatable {
  const [shown, setShown] = useState(false)
  const { mutateAsync, data, status } = useTranslate()
  const cached = useTranslationsStore((s) => s.byText[rawText])
  const put = useTranslationsStore((s) => s.put)
  const busy = status === 'pending'
  const failed = status === 'error' || (data != null && !data.ok)
  const translation = shown ? (cached?.text ?? null) : null

  async function toggle() {
    if (shown) {
      setShown(false)
      return
    }
    if (cached) {
      setShown(true)
      return
    }
    try {
      const res = await mutateAsync(rawText)
      if (res.ok && res.translation != null) {
        const payload = { text: res.translation, target_lang: res.target_lang, model: res.model }
        put(rawText, payload)
        setShown(true)
        onTranslated?.(payload)
      }
    } catch {
      // A true transport failure flips status to 'error', which drives `failed`.
    }
  }

  return { shown, busy, failed, translation, toggle }
}
