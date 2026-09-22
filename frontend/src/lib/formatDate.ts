import type { Lang } from '@/i18n'

/**
 * A server timestamp as a calendar date, written the way the active language
 * writes dates.
 *
 * Pinned to UTC: a deletion date is the day the server acts on, and reading it
 * in the browser's zone would move it a day for anyone east or west of it.
 */
export function formatDate(iso: string, lang: Lang): string {
  return new Intl.DateTimeFormat(lang, { dateStyle: 'medium', timeZone: 'UTC' }).format(new Date(iso))
}
