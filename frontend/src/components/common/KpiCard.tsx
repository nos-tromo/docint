import type { ReactNode } from 'react'

export function KpiCard({
  label,
  value,
  hint
}: {
  label: string
  value: ReactNode
  hint?: string
}) {
  return (
    <div className="rounded-lg border border-border bg-zinc-900 p-4">
      <div className="text-xs uppercase text-muted-foreground">{label}</div>
      <div className="mt-2 text-2xl font-semibold tracking-tight">{value ?? '—'}</div>
      {hint && <div className="mt-1 text-xs text-muted-foreground">{hint}</div>}
    </div>
  )
}
