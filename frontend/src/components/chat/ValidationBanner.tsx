import { useId, useState, type ComponentType } from 'react'
import { CheckIcon, DisclosureButton, InfoIcon, WarningIcon, type IconProps } from '@infra/ui'
import type { ChatFinalEvent, ValidationFields } from '@/api/types'
import { cn } from '@/lib/cn'
import { useT } from '@/i18n/LanguageContext'

type BannerFields = ValidationFields & Pick<ChatFinalEvent, 'answer_mode' | 'agent'>

interface BannerSpec {
  /** A mismatch is a warning and keeps a box; the other outcomes are a quiet line. */
  kind: 'mismatch' | 'passed' | 'unverified'
  /** Drawn, never typed — `⚠`/`✓`/`ⓘ` render from whatever font the OS falls
   *  back to, and the first and last of those carry emoji presentation on some
   *  platforms, so they arrive full-colour beside monochrome chrome. */
  Icon: ComponentType<IconProps>
  title: string
  detail?: string
}

function resolveSpec(v: BannerFields, t: ReturnType<typeof useT>): BannerSpec {
  // Mirrors Streamlit's response_validation_summary: always show *some*
  // signal under each chat turn so the validation status never silently
  // disappears.
  const reason = v.validation_reason ?? undefined
  const assessment = v.answer_mode === 'assessment'
  // The backend skips an assessment's validation exactly when it is not `split` or its evidence part is
  // empty; only an assessment's tokens say so, as a lookup's `plain` answer is source-backed whole.
  const answer = v.agent?.answer
  const noSourcePart =
    answer === 'unsplit' || answer === 'empty' || (answer === 'split' && v.agent?.evidence_chars === 0)
  if (v.validation_checked === true && v.validation_mismatch === true) {
    return {
      kind: 'mismatch',
      Icon: WarningIcon,
      title: assessment
        ? t('chat.validation_mismatch_assessment_title')
        : t('chat.validation_mismatch_title'),
      detail: reason ?? t('chat.validation_mismatch_default_detail')
    }
  }
  if (v.validation_checked === true) {
    return {
      kind: 'passed',
      Icon: CheckIcon,
      title: assessment
        ? t('chat.validation_passed_assessment_title')
        : t('chat.validation_passed_title'),
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
    kind: 'unverified',
    Icon: InfoIcon,
    title:
      assessment && v.validation_checked === false && noSourcePart
        ? t('chat.validation_skipped_assessment_title')
        : v.validation_checked === false
          ? t('chat.validation_unavailable_title')
          : t('chat.validation_not_validated_title'),
    detail: t('chat.validation_default_detail')
  }
}

/**
 * The answer's validation, on one line under it: a mismatch as a compact amber warning, a pass or an
 * unverified answer as a quiet note. The detail opens behind a disclosure, so it never outweighs the answer.
 */
export function ValidationBanner({ v }: { v: BannerFields }) {
  const t = useT()
  const spec = resolveSpec(v, t)
  const [open, setOpen] = useState(false)
  const detailId = useId()
  const warning = spec.kind === 'mismatch'
  return (
    <div
      className={cn(
        'mt-3 text-xs',
        warning
          ? 'rounded-md border border-[var(--status-amber-border)] bg-[var(--status-amber-surface)] px-3 py-1.5 text-[var(--status-amber-strong)]'
          : 'text-muted-foreground'
      )}
    >
      <div className="flex items-center gap-2">
        <spec.Icon
          className={cn('h-3.5 w-3.5 shrink-0', spec.kind === 'passed' && 'text-[var(--status-emerald-fg)]')}
        />
        <span className={cn('min-w-0 flex-1', warning && 'font-medium')}>{spec.title}</span>
        {spec.detail && (
          <DisclosureButton
            expanded={open}
            controls={detailId}
            label={open ? t('chat.validation_details_hide') : t('chat.validation_details_show')}
            onClick={() => setOpen((value) => !value)}
            className="h-6 w-6"
          />
        )}
      </div>
      {open && spec.detail && (
        <div id={detailId} className="mt-1 text-[11px]">
          {spec.detail}
        </div>
      )}
    </div>
  )
}
