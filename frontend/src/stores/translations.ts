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
  put: (rawText: string, payload: TranslationPayload) => void
}

export const useTranslationsStore = create<TranslationsState>()((set) => ({
  byText: {},
  put: (rawText, payload) =>
    set((s) => (rawText ? { byText: { ...s.byText, [rawText]: payload } } : s))
}))

/** Read a stored translation outside React (`toItem` runs in an event handler). */
export function storedTranslation(rawText: string): TranslationPayload | undefined {
  return rawText ? useTranslationsStore.getState().byText[rawText] : undefined
}
