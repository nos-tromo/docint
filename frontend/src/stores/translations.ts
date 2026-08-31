import { create } from 'zustand'

export interface TranslationPayload {
  text: string
  target_lang: string
  model: string
}

interface TranslationsState {
  /**
   * Successful translations keyed by the raw source text — the same string the
   * Translate control posts, so a row finds its own translation back after the
   * virtualizer unmounts and remounts it, and the section-wide "Add all" can
   * look one up for a row it never rendered.
   *
   * Session-lifetime and deliberately not persisted: this is case text, and a
   * translation would outlive the model that produced it.
   *
   * The key needs no target-language dimension — SPA translations always target
   * the deployment's `RESPONSE_LANGUAGE`, which is constant per deployment.
   */
  byText: Record<string, TranslationPayload>
  put: (rawText: string, payload: TranslationPayload) => void
}

export const useTranslationsStore = create<TranslationsState>()((set) => ({
  byText: {},
  put: (rawText, payload) =>
    set((s) => (rawText ? { byText: { ...s.byText, [rawText]: payload } } : s))
}))

/**
 * Read a stored translation outside React (the "Add all" `toItem` loop runs in
 * an event handler, not a render).
 */
export function storedTranslation(rawText: string): TranslationPayload | undefined {
  return rawText ? useTranslationsStore.getState().byText[rawText] : undefined
}
