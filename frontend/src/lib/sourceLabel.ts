import type { Source } from '@/api/types'

export function sourceLabel(s: Source): string {
  if (s.page_label) return `${s.filename} · p. ${s.page_label}`
  if (s.row_label) return `${s.filename} · row ${s.row_label}`
  return s.filename
}

export function formatScore(n: number): string {
  return n.toFixed(3)
}
