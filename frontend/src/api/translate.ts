import { apiPost } from './client'

export interface TranslateResponse {
  ok: boolean
  translation: string | null
  model: string
  target_lang: string
  error?: string | null
}

export const translate = (text: string) => apiPost<TranslateResponse>('/translate', { text })
