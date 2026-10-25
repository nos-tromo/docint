import type { Lang } from '@/i18n'

/** One formatter per language: building an `Intl` formatter costs far more than using one. */
const FORMATS = new Map<Lang, Intl.DateTimeFormat>()

/**
 * A server timestamp as a calendar date, written the way the active language
 * writes dates.
 *
 * Pinned to UTC: a deletion date is the day the server acts on, and reading it
 * in the browser's zone would move it a day for anyone east or west of it.
 */
export function formatDate(iso: string, lang: Lang): string {
  let format = FORMATS.get(lang)
  if (!format) {
    format = new Intl.DateTimeFormat(lang, { dateStyle: 'medium', timeZone: 'UTC' })
    FORMATS.set(lang, format)
  }
  return format.format(new Date(iso))
}
