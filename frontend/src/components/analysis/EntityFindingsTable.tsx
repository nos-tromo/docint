import { useMemo, useRef } from 'react'
import { DownloadLink } from '@infra/ui'
import { useVirtualizer } from '@tanstack/react-virtual'
import type { EntityMergeMode, NerEntityRow, NerSourceRow } from '@/api/types'
import { csvExportHref, getNerSourcesPage } from '@/api/collections'
import { EntityFinding } from './EntityFinding'
import { AddAllToReportButton } from '@/components/report/AddAllToReportButton'
import { TranslateAllButton } from '@/components/common/TranslateAllButton'
import { fetchAllPages } from '@/lib/fetchAllPages'
import { chunkTextOf, entityFindingSnapshot } from '@/lib/reportSnapshots'
import { storedTranslation } from '@/stores/translations'
import { useT } from '@/i18n/LanguageContext'

// Single source of truth for the table's column widths; the header row and
// every body row share it so columns line up. Metadata is one flexible column.
const FINDINGS_GRID = '2.5rem minmax(9rem,0.9fr) minmax(10rem,1.1fr) minmax(12rem,2fr) 6rem'

interface Props {
  selected: NerEntityRow | null
  findings: NerSourceRow[]
  isFetchingFindings?: boolean
  hasNextPage?: boolean
  onLoadMore?: () => void
  collection: string
  entityMergeMode?: EntityMergeMode
  reportDedupeKeys?: Set<string>
}

function highlightTermsForEntity(entity: NerEntityRow): string[] {
  const terms = new Set<string>()
  if (entity.text) terms.add(entity.text)
  for (const v of entity.variants ?? []) {
    const t = (v.text ?? '').trim()
    if (t) terms.add(t)
  }
  return Array.from(terms)
}

/**
 * Findings (chunks) for the selected entity, rendered as a virtualized table —
 * one row per chunk, with all locator/reference metadata flattened into a
 * single Metadata column. Replaces the former accordion list so a high-mention
 * entity's findings are scannable without expanding each card. Preserves the
 * CSV export and per-row "Add to report" control.
 */
export function EntityFindingsTable({
  selected,
  findings,
  isFetchingFindings,
  hasNextPage,
  onLoadMore,
  collection,
  entityMergeMode,
  reportDedupeKeys
}: Props) {
  const t = useT()
  const scrollRef = useRef<HTMLDivElement>(null)
  const virtualizer = useVirtualizer({
    count: findings.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => 104,
    overscan: 8
  })

  const highlightTerms = useMemo(
    () => (selected ? highlightTermsForEntity(selected) : []),
    [selected]
  )

  if (!selected) {
    return <p className="text-sm text-muted-foreground">{t('entities.pick_entity_hint')}</p>
  }

  // Mirrors the backend's ner-sources `entity_label` (``text [TYPE]``) so a
  // report's entity column matches the CSV export — the 'Unlabeled' fallback
  // is protocol data (matches the backend's own fallback), not UI chrome, and
  // must stay untranslated or this key would stop matching the backend value.
  const entityLabel = `${selected.text} [${selected.type || 'Unlabeled'}]`
  const selectedTypeLower = (selected.type || '').toLowerCase()
  // The same query the table's own infinite pages use, walked to the end at
  // the server's page maximum so a section-wide add sees every match.
  const fetchAllFindings = (maxItems: number) =>
    fetchAllPages<NerSourceRow>(
      (cursor) =>
        getNerSourcesPage({
          cursor,
          limit: 500,
          entity_text: selected.text,
          entity_type: selected.type,
          entity_merge_mode: entityMergeMode,
          collection
        }),
      { maxItems }
    )
  const exportParams = {
    entity_text: selected.text,
    entity_type: selected.type,
    entity_merge_mode: entityMergeMode
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-3">
        <div className="text-sm">
          <span className="text-muted-foreground">{t('entities.findings_for')} </span>
          <span className="font-medium">{selected.text}</span>
          <span className="text-muted-foreground">
            {' '}—{' '}
            {t(
              findings.length === 1 ? 'entities.findings_chunk_one' : 'entities.findings_chunk_other',
              { count: findings.length }
            )}
            {hasNextPage ? '+' : ''}
            {isFetchingFindings ? t('entities.findings_loading_suffix') : ''}
          </span>
        </div>
        <div className="flex items-center gap-1">
          {/* Every matching finding, not just the rows paged in. */}
          <TranslateAllButton fetchAll={fetchAllFindings} textOf={chunkTextOf} hasRows={findings.length > 0} />
          {/* Adds every finding the entity filter matches, not only the rows
              paged in — see AddAllToReportButton. */}
          <AddAllToReportButton
            fetchAll={fetchAllFindings}
            toItem={(row: NerSourceRow) =>
              entityFindingSnapshot(row, entityLabel, storedTranslation(chunkTextOf(row)))
            }
            hasRows={findings.length > 0}
          />
          {collection && (
            <DownloadLink
              href={csvExportHref(collection, 'ner-sources', exportParams)}
              label={t('table.export_csv')}
            />
          )}
        </div>
      </div>

      {findings.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          {isFetchingFindings ? t('entities.loading_findings') : t('entities.no_chunks_matched')}
        </p>
      ) : (
        <div className="rounded-md border border-border overflow-hidden">
          <div
            className="grid gap-3 px-3 py-2 bg-muted border-b border-border text-[11px] uppercase tracking-wide text-muted-foreground"
            style={{ gridTemplateColumns: FINDINGS_GRID }}
          >
            <span>#</span>
            <span>{t('common.col_source')}</span>
            <span>{t('common.col_metadata')}</span>
            <span>{t('common.col_text')}</span>
            <span className="text-right">{t('common.col_report')}</span>
          </div>
          <div
            ref={scrollRef}
            className="max-h-[60vh] overflow-y-auto"
            data-testid="ner-findings-scroll"
          >
            <div className="relative" style={{ height: `${virtualizer.getTotalSize()}px` }}>
              {virtualizer.getVirtualItems().map((vRow) => {
                const source = findings[vRow.index]
                return (
                  <div
                    key={source.chunk_id ?? vRow.index}
                    data-index={vRow.index}
                    ref={virtualizer.measureElement}
                    className="absolute left-0 right-0"
                    style={{ transform: `translateY(${vRow.start}px)` }}
                  >
                    <EntityFinding
                      index={vRow.index + 1}
                      source={source}
                      highlightTerms={highlightTerms}
                      selectedTypeLower={selectedTypeLower}
                      entityLabel={entityLabel}
                      reportDedupeKeys={reportDedupeKeys}
                      gridTemplate={FINDINGS_GRID}
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
                  disabled={isFetchingFindings}
                  className="px-3 py-1 rounded-md border border-border text-sm disabled:opacity-50"
                >
                  {isFetchingFindings ? t('common.loading_ellipsis') : t('table.load_more')}
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
