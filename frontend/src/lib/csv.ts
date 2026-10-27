/** Leading characters a spreadsheet reads as the start of a formula. */
const FORMULA_START = /^[=+\-@\t\r]/

function escape(value: unknown): string {
  if (value === null || value === undefined) return ''
  let s = String(value)
  // A leading quote makes a spreadsheet show the cell as text instead of running it; numbers stay as they are.
  if (typeof value === 'string' && FORMULA_START.test(s)) s = `'${s}`
  if (/[",\n\r]/.test(s)) return `"${s.replace(/"/g, '""')}"`
  return s
}

export function toCsv<T extends Record<string, unknown>>(rows: T[], columns: (keyof T)[]): string {
  const header = columns.map((c) => escape(c)).join(',')
  const body = rows.map((r) => columns.map((c) => escape(r[c])).join(',')).join('\n')
  return `${header}\n${body}`
}

export function downloadText(
  filename: string,
  content: string,
  mime = 'text/plain;charset=utf-8'
): void {
  const blob = new Blob([content], { type: mime })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = filename
  a.click()
  URL.revokeObjectURL(a.href)
}

export function downloadCsv(filename: string, csv: string): void {
  downloadText(filename, csv, 'text/csv;charset=utf-8')
}
