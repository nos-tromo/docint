import { useMemo, useRef } from 'react'
import { useVirtualizer } from '@tanstack/react-virtual'
import type { EntityMergeMode, NerEntityRow, NerSourceRow } from '@/api/types'
import { csvExportHref } from '@/api/collections'
import { EntityFinding } from './EntityFinding'

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
  const scrollRef = useRef<HTMLDivElement>(null)
  const virtualizer = useVirtualizer({
    count: findings.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => 132,
    overscan: 8
  })

  const highlightTerms = useMemo(
    () => (selected ? highlightTermsForEntity(selected) : []),
    [selected]
  )

  if (!selected) {
    return (
      <p className="text-sm text-muted-foreground">
        Pick an entity to see the chunks where it appears.
      </p>
    )
  }

  // Mirrors the backend's ner-sources `entity_label` (``text [TYPE]``) so a
  // report's entity column matches the CSV export.
  const entityLabel = `${selected.text} [${selected.type || 'Unlabeled'}]`
  const selectedTypeLower = (selected.type || '').toLowerCase()
  const exportParams = {
    entity_text: selected.text,
    entity_type: selected.type,
    entity_merge_mode: entityMergeMode
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-3">
        <div className="text-sm">
          <span className="text-muted-foreground">Findings for </span>
          <span className="font-medium">{selected.text}</span>
          <span className="text-muted-foreground">
            {' '}— {findings.length} chunk{findings.length === 1 ? '' : 's'}
            {hasNextPage ? '+' : ''}
            {isFetchingFindings ? ' (loading…)' : ''}
          </span>
        </div>
        {collection && (
          <a
            href={csvExportHref(collection, 'ner-sources', exportParams)}
            download
            className="px-3 py-1 rounded-md border border-border text-sm"
          >
            CSV
          </a>
        )}
      </div>

      {findings.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          {isFetchingFindings
            ? 'Loading findings…'
            : 'No chunks were matched for the selected entity.'}
        </p>
      ) : (
        <div className="rounded-md border border-border overflow-hidden">
          <div
            className="grid gap-3 px-3 py-2 bg-zinc-900 border-b border-border text-[11px] uppercase tracking-wide text-muted-foreground"
            style={{ gridTemplateColumns: FINDINGS_GRID }}
          >
            <span>#</span>
            <span>Source</span>
            <span>Metadata</span>
            <span>Text</span>
            <span className="text-right">Report</span>
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
                  {isFetchingFindings ? 'Loading…' : 'Load more'}
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
