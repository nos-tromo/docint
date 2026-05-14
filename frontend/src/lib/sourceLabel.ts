import type { Source } from '@/api/types'

export function sourceLabel(s: Source): string {
  if (s.page !== null && s.page !== undefined) {
    return `${s.filename} · p. ${s.page}`
  }
  if (s.row !== null && s.row !== undefined) {
    return `${s.filename} · row ${s.row}`
  }
  return s.filename
}

export function formatScore(n: number | null | undefined): string {
  if (n === null || n === undefined) return ''
  return n.toFixed(3)
}
