import { useRef, useState } from 'react'
import { useVirtualizer } from '@tanstack/react-virtual'
import { csvExportHref } from '@/api/collections'
import type { HateSpeechRow } from '@/api/types'
import { referenceMetadataItems } from '@/lib/referenceMetadata'
import { AddToReportButton } from '@/components/report/AddToReportButton'
import { useTranslatable, type TranslationPayload } from '@/hooks/useTranslatable'
import { TranslateToggle } from '@/components/common/TranslateToggle'
import { ClampedText } from '@/components/common/ClampedText'
import { hateSpeechSnapshot } from '@/lib/reportSnapshots'
import { useT } from '@/i18n/LanguageContext'
import type { Strings } from '@/i18n'
import { hateCategoryLabel } from '@/lib/hateCategoryLabel'

export type { HateSpeechRow }

// Shared column template for the header row and every body row. Metadata is a
// single column (reason / confidence / chunk id / reference metadata).
const HATE_GRID = '2.5rem 6.5rem minmax(8rem,0.8fr) minmax(9rem,1.1fr) minmax(12rem,1.8fr) 6rem'

interface Props {
  rows: HateSpeechRow[]
  isFetching?: boolean
  hasNextPage?: boolean
  onLoadMore?: () => void
  collection: string
  reportDedupeKeys?: Set<string>
}

function locationParts(r: HateSpeechRow, t: (key: keyof Strings, vars?: Record<string, string | number>) => string): string {
  const parts: string[] = []
  if (r.page_label) parts.push(t('common.loc_page', { page: r.page_label }))
  else if (r.page !== null && r.page !== undefined) parts.push(t('common.loc_page', { page: r.page }))
  if (r.row !== null && r.row !== undefined) parts.push(t('common.loc_row', { row: r.row }))
  return parts.join(', ')
}

/**
 * One flagged chunk rendered as a table row. The former accordion's hidden
 * fields (reason, confidence, chunk id, reference metadata) are flattened into
 * a single Metadata column shown inline, mirroring the entity findings table.
 */
function HateSpeechTableRow({
  row,
  index,
  reportDedupeKeys
}: {
  row: HateSpeechRow
  index: number
  reportDedupeKeys?: Set<string>
}) {
  const i18n = useT()
  const [translation, setTranslation] = useState<TranslationPayload | null>(null)
  const reportItem = hateSpeechSnapshot(row, translation ?? undefined)
  const inReport = reportDedupeKeys?.has(reportItem.dedupe_key) ?? false
  const refMeta = referenceMetadataItems(row.reference_metadata, {}, i18n)
  const chunkText = (row.chunk_text ?? row.text ?? '').trim()
  const translationState = useTranslatable(chunkText, setTranslation)
  const source = row.source_ref ?? row.filename ?? i18n('common.unknown_source')
  const location = locationParts(row, i18n)
  const category = hateCategoryLabel((row.category ?? 'unknown').trim(), i18n)
  const reason = (row.reason ?? '').trim()

  const metadata: Array<{ label: string; value: string }> = []
  if (reason) metadata.push({ label: i18n('common.meta_reason'), value: reason })
  // `confidence` is protocol data (the fixed high|medium|low enum) rendered
  // verbatim, like `Filetype`/`Reader` elsewhere — only its label is swept.
  if (row.confidence) {
    metadata.push({ label: i18n('common.meta_confidence'), value: String(row.confidence) })
  }
  if (row.chunk_id) metadata.push({ label: i18n('common.meta_chunk_id'), value: row.chunk_id })
  for (const item of refMeta) metadata.push(item)

  return (
    <div
      className="group grid items-start gap-3 border-b border-border px-3 py-2.5 text-sm hover:bg-zinc-900/40"
      style={{ gridTemplateColumns: HATE_GRID }}
      data-testid="hate-speech-row"
    >
      <div className="text-xs text-muted-foreground tabular-nums pt-0.5">{index}</div>
      <div className="text-xs font-medium uppercase break-words pt-0.5">{category}</div>
      <div className="min-w-0 space-y-0.5">
        <div className="break-words">{source}</div>
        {location && <div className="text-xs text-muted-foreground">{location}</div>}
      </div>
      <div className="min-w-0">
        {metadata.length > 0 ? (
          <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-2 gap-y-0.5 text-xs">
            {metadata.map(({ label, value }) => (
              <div key={label} className="contents">
                <dt className="text-muted-foreground whitespace-nowrap">{label}</dt>
                <dd className="break-words">{value}</dd>
              </div>
            ))}
          </dl>
        ) : (
          <span className="text-xs text-muted-foreground">—</span>
        )}
      </div>
      <div className="min-w-0">
        {chunkText ? (
          <>
            {translationState.shown && (
              <div className="mb-1 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
                {i18n('common.translation')}
              </div>
            )}
            <ClampedText length={(translationState.translation ?? chunkText).length}>
              {translationState.translation ?? chunkText}
            </ClampedText>
            {translationState.failed && (
              <div className="mt-1 text-[11px] text-muted-foreground">
                {i18n('common.translation_unavailable')}
              </div>
            )}
          </>
        ) : (
          <span className="text-xs text-muted-foreground">{i18n('common.chunk_text_unavailable')}</span>
        )}
      </div>
      <div className="flex items-center justify-end gap-1">
        {chunkText && (
          <TranslateToggle
            shown={translationState.shown}
            busy={translationState.busy}
            onClick={translationState.toggle}
          />
        )}
        {reportDedupeKeys && <AddToReportButton item={reportItem} inReport={inReport} />}
      </div>
    </div>
  )
}

/**
 * Hate-speech findings as a virtualized table — one flagged chunk per row, all
 * secondary fields collapsed into a single Metadata column. Preserves the CSV
 * export and per-row "Add to report" control.
 */
export function HateSpeechTable({
  rows,
  isFetching,
  hasNextPage,
  onLoadMore,
  collection,
  reportDedupeKeys
}: Props) {
  const t = useT()
  const scrollRef = useRef<HTMLDivElement>(null)
  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => 120,
    overscan: 8
  })

  if (!rows.length) {
    return (
      <div className="text-sm text-muted-foreground">
        {isFetching ? t('hate.loading') : t('hate.empty')}
      </div>
    )
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <p className="text-sm text-muted-foreground">
          {t(rows.length === 1 ? 'hate.count_one' : 'hate.count_other', { count: rows.length })}
          {hasNextPage ? '+' : ''}.
        </p>
        {collection && (
          <a
            href={csvExportHref(collection, 'hate-speech')}
            download
            className="px-3 py-1 rounded-md border border-border text-sm"
          >
            {t('common.csv_button')}
          </a>
        )}
      </div>
      <div className="rounded-md border border-border overflow-hidden">
        <div
          className="grid gap-3 px-3 py-2 bg-zinc-900 border-b border-border text-[11px] uppercase tracking-wide text-muted-foreground"
          style={{ gridTemplateColumns: HATE_GRID }}
        >
          <span>#</span>
          <span>{t('hate.col_category')}</span>
          <span>{t('common.col_source')}</span>
          <span>{t('common.col_metadata')}</span>
          <span>{t('common.col_text')}</span>
          <span className="text-right">{t('common.col_report')}</span>
        </div>
        <div
          ref={scrollRef}
          className="max-h-[70vh] overflow-y-auto"
          data-testid="hate-speech-scroll"
        >
          <div className="relative" style={{ height: `${virtualizer.getTotalSize()}px` }}>
            {virtualizer.getVirtualItems().map((vRow) => {
              const r = rows[vRow.index]
              return (
                <div
                  key={r.chunk_id ?? vRow.index}
                  data-index={vRow.index}
                  ref={virtualizer.measureElement}
                  className="absolute left-0 right-0"
                  style={{ transform: `translateY(${vRow.start}px)` }}
                >
                  <HateSpeechTableRow
                    row={r}
                    index={vRow.index + 1}
                    reportDedupeKeys={reportDedupeKeys}
                  />
                </div>
              )
            })}
          </div>
          {hasNextPage && onLoadMore && (
            <div className="flex justify-center py-2">
              <button
                type="button"
                onClick={onLoadMore}
                disabled={isFetching}
                className="px-3 py-1 rounded-md border border-border text-sm disabled:opacity-50"
              >
                {isFetching ? t('common.loading_ellipsis') : t('table.load_more')}
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
