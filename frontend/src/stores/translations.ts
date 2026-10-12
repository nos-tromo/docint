import { create } from 'zustand'

export interface TranslationPayload {
  text: string
  target_lang: string
  model: string
}

interface TranslationsState {
  /**
   * Successful translations keyed by raw source text, so a row finds its own
   * back after a remount and "Add all" can look one up for a row it never
   * rendered. No language dimension (the SPA only targets `RESPONSE_LANGUAGE`),
   * and not persisted — this is case text.
   */
  byText: Record<string, TranslationPayload>
  /**
   * Texts the endpoint answered as already in `RESPONSE_LANGUAGE`. Kept apart
   * from `byText`, so no snapshot reads one as a translation, and remembered,
   * so "Translate all" never sends one again.
   */
  sameLanguage: Record<string, true>
  put: (rawText: string, payload: TranslationPayload) => void
  markSameLanguage: (rawText: string) => void
}

export const useTranslationsStore = create<TranslationsState>()((set) => ({
  byText: {},
  sameLanguage: {},
  put: (rawText, payload) =>
    set((s) => (rawText ? { byText: { ...s.byText, [rawText]: payload } } : s)),
  markSameLanguage: (rawText) =>
    set((s) => (rawText ? { sameLanguage: { ...s.sameLanguage, [rawText]: true } } : s))
}))

/** Read a stored translation outside React (`toItem` runs in an event handler). */
export function storedTranslation(rawText: string): TranslationPayload | undefined {
  return rawText ? useTranslationsStore.getState().byText[rawText] : undefined
}
