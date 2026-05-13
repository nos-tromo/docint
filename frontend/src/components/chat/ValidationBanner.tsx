import type { ValidationFields } from '@/api/types'
import { cn } from '@/lib/cn'

export function ValidationBanner({ v }: { v: ValidationFields }) {
  if (!v.validation_checked) return null
  const tone = v.validation_mismatch
    ? 'border-amber-700 bg-amber-950 text-amber-200'
    : 'border-emerald-700 bg-emerald-950 text-emerald-200'
  const label = v.validation_mismatch ? 'mismatch' : 'validated'
  return (
    <div className={cn('rounded-md border px-3 py-2 text-xs', tone)}>
      <div className="font-medium uppercase tracking-wide">{label}</div>
      {v.validation_reason && <div className="mt-1">{v.validation_reason}</div>}
    </div>
  )
}
