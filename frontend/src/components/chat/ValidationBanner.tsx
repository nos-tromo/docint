import type { ValidationFields } from '@/api/types'
import { cn } from '@/lib/cn'

export function ValidationBanner({ v }: { v: ValidationFields }) {
  if (!v.validation_status) return null
  const tone =
    v.validation_status === 'ok'
      ? 'border-emerald-700 bg-emerald-950 text-emerald-200'
      : v.validation_status === 'warning'
        ? 'border-amber-700 bg-amber-950 text-amber-200'
        : 'border-red-700 bg-red-950 text-red-200'
  return (
    <div className={cn('rounded-md border px-3 py-2 text-xs', tone)}>
      <div className="font-medium uppercase tracking-wide">{v.validation_status}</div>
      {v.validation_message && <div className="mt-1">{v.validation_message}</div>}
    </div>
  )
}
