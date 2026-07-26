import type { ValidationFields } from '@/api/types'
import { cn } from '@/lib/cn'
import { useT } from '@/i18n/LanguageContext'

interface BannerSpec {
  tone: string
  icon: string
  title: string
  detail?: string
}

function resolveSpec(v: ValidationFields, t: ReturnType<typeof useT>): BannerSpec {
  // Mirrors Streamlit's response_validation_summary: always show *some*
  // signal under each chat turn so the validation status never silently
  // disappears.
  const reason = v.validation_reason ?? undefined
  if (v.validation_checked === true && v.validation_mismatch === true) {
    return {
      tone: 'border-amber-700 bg-amber-950 text-amber-200',
      icon: '⚠',
      title: t('chat.validation_mismatch_title'),
      detail: reason ?? t('chat.validation_mismatch_default_detail')
    }
  }
  if (v.validation_checked === true) {
    return {
      tone: 'border-emerald-700 bg-emerald-950 text-emerald-200',
      icon: '✓',
      title: t('chat.validation_passed_title'),
      detail: reason ?? undefined
    }
  }
  // validation_checked is false / null / undefined — validation either
  // didn't run or couldn't complete. Always surface this rather than
  // suppressing the banner, so users can see at a glance that the
  // response is unverified.
  return {
    tone: 'border-zinc-700 bg-zinc-900 text-zinc-300',
    icon: 'ⓘ',
    title:
      v.validation_checked === false
        ? t('chat.validation_unavailable_title')
        : t('chat.validation_not_validated_title'),
    detail: reason ?? t('chat.validation_default_detail')
  }
}

export function ValidationBanner({ v }: { v: ValidationFields }) {
  const t = useT()
  const spec = resolveSpec(v, t)
  return (
    <div className={cn('mt-3 rounded-md border px-3 py-2 text-xs', spec.tone)}>
      <div className="font-medium flex items-center gap-2">
        <span aria-hidden="true">{spec.icon}</span>
        <span>{spec.title}</span>
      </div>
      {spec.detail && <div className="mt-1 text-[11px] opacity-90">{spec.detail}</div>}
    </div>
  )
}
