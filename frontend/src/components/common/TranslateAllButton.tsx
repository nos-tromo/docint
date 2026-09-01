import { IconButton, WarningIcon, XIcon } from '@infra/ui'
import { TranslateIcon } from '@/components/common/icons'
import { useTranslateAll } from '@/hooks/useTranslateAll'
import { useT } from '@/i18n/LanguageContext'

interface Props<Row> {
  /** Walk the section's cursor pages and return every matching row. */
  fetchAll: (maxItems: number) => Promise<Row[]>
  /** The row's canonical text — the same derivation the store is keyed by. */
  textOf: (row: Row) => string
  /** Whether the section currently has anything to translate. */
  hasRows: boolean
  className?: string
}

/**
 * Translate every finding of an Analysis section at once — the section-wide
 * counterpart of the per-row Translate toggle. It knows nothing about entities
 * vs. hate speech: the section hands it the walk and text derivation.
 */
export function TranslateAllButton<Row>({ fetchAll, textOf, hasRows, className }: Props<Row>) {
  const t = useT()
  const { run, stop, status, cap, total, done, failed, skipped } = useTranslateAll<Row>({ fetchAll, textOf })
  const translating = status === 'translating'
  const isFailed = status === 'failed'

  // The SPA has no toast layer, and a run of several minutes has to say where
  // it has got to.
  let message: string | null = null
  if (status === 'too_many') message = t('common.translate_all_too_many', { max: cap })
  else if (translating) message = t('common.translate_all_progress', { done, total })
  else if (isFailed) message = t('common.translate_all_failed')
  else if (status === 'stopped') message = t('common.translate_all_stopped', { done, total })
  else if (status === 'done') {
    if (total === 0) message = t('common.translate_all_none')
    else if (failed > 0) message = t('common.translate_all_done_failed', { done, failed })
    else message = t('common.translate_all_done', { done, skipped })
  }

  // `IconButton`'s `busy` disables the control: right for the page walk, wrong
  // once translating starts, which is when it has to stay clickable to stop.
  return (
    <div className={`flex items-center gap-2 ${className ?? ''}`}>
      {message && (
        <span
          className={`text-xs ${isFailed || status === 'too_many' ? 'text-destructive' : 'text-muted-foreground'}`}
          role="status"
          data-testid="translate-all-message"
        >
          {message}
        </span>
      )}
      <IconButton
        icon={isFailed ? <WarningIcon /> : translating ? <XIcon /> : <TranslateIcon />}
        label={
          translating
            ? t('common.translate_all_stop')
            : isFailed
              ? t('common.translate_all_retry')
              : t('common.translate_all')
        }
        hint={status === 'fetching' ? t('common.translate_all_busy') : undefined}
        variant={isFailed ? 'danger' : 'ghost'}
        busy={status === 'fetching'}
        disabled={!hasRows || status === 'fetching'}
        onClick={translating ? stop : run}
      />
    </div>
  )
}
