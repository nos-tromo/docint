import type { ComponentType } from 'react'
import { CheckIcon, InfoIcon, WarningIcon, type IconProps } from '@infra/ui'
import type { ValidationFields } from '@/api/types'
import { cn } from '@/lib/cn'
import { useT } from '@/i18n/LanguageContext'

interface BannerSpec {
  tone: string
  /** Drawn, never typed — `⚠`/`✓`/`ⓘ` render from whatever font the OS falls
   *  back to, and the first and last of those carry emoji presentation on some
   *  platforms, so they arrive full-colour beside monochrome chrome. */
  Icon: ComponentType<IconProps>
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
      tone: 'border-[var(--status-amber-border)] bg-[var(--status-amber-surface)] text-[var(--status-amber-strong)]',
      Icon: WarningIcon,
      title: t('chat.validation_mismatch_title'),
      detail: reason ?? t('chat.validation_mismatch_default_detail')
    }
  }
  if (v.validation_checked === true) {
    return {
      tone: 'border-[var(--status-emerald-border)] bg-[var(--status-emerald-surface)] text-[var(--status-emerald-strong)]',
      Icon: CheckIcon,
      title: t('chat.validation_passed_title'),
      detail: reason ?? undefined
    }
  }
  // validation_checked is false / null / undefined — validation either
  // didn't run or couldn't complete. Always surface this rather than
  // suppressing the banner, so users can see at a glance that the
  // response is unverified. Unlike the mismatch case above, this branch's
  // backend `reason` can carry a raw caught-exception message (validation
  // model/transport failure) — never render it; catalog copy only.
  return {
    tone: 'border-border bg-muted text-muted-foreground',
    Icon: InfoIcon,
    title:
      v.validation_checked === false
        ? t('chat.validation_unavailable_title')
        : t('chat.validation_not_validated_title'),
    detail: t('chat.validation_default_detail')
  }
}

export function ValidationBanner({ v }: { v: ValidationFields }) {
  const t = useT()
  const spec = resolveSpec(v, t)
  return (
    <div className={cn('mt-3 rounded-md border px-3 py-2 text-xs', spec.tone)}>
      <div className="font-medium flex items-center gap-2">
        <spec.Icon className="h-3.5 w-3.5 shrink-0" />
        <span>{spec.title}</span>
      </div>
      {spec.detail && <div className="mt-1 text-[11px] opacity-90">{spec.detail}</div>}
    </div>
  )
}
