import { apiPost } from './client'

export interface TranslateResponse {
  ok: boolean
  translation: string | null
  model: string
  target_lang: string
  error?: string | null
  /** The text is already in `target_lang`: `translation` is null, and that is no failure. */
  same_language?: boolean
}

export const translate = (text: string) => apiPost<TranslateResponse>('/translate', { text })
