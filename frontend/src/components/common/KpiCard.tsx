import type { ReactNode } from 'react'
import { Card } from '@infra/ui'

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
    <Card>
      <div className="text-xs uppercase text-muted-foreground">{label}</div>
      <div className="mt-2 text-2xl font-semibold tracking-tight">{value ?? '—'}</div>
      {hint && <div className="mt-1 text-xs text-muted-foreground">{hint}</div>}
    </Card>
  )
}
