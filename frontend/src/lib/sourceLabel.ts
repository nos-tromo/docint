import type { Source } from '@/api/types'
import type { Strings } from '@/i18n'
import { defaultT } from '@/i18n/defaultT'

export function sourceLabel(
  s: Source,
  t: (key: keyof Strings, vars?: Record<string, string | number>) => string = defaultT
): string {
  if (s.page !== null && s.page !== undefined) {
    return `${s.filename} · ${t('common.loc_page', { page: s.page })}`
  }
  if (s.row !== null && s.row !== undefined) {
    return `${s.filename} · ${t('common.loc_row', { row: s.row })}`
  }
  return s.filename
}
