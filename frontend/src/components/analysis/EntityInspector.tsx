import { useEffect, useMemo, useRef } from 'react'
import { useVirtualizer } from '@tanstack/react-virtual'
import type { EntityMergeMode, NerEntityRow, NerSourceRow } from '@/api/types'
import { csvExportHref } from '@/api/collections'
import { EntityFinding } from './EntityFinding'

interface Props {
  entities: NerEntityRow[]
  selectedKey: string | null
  onSelectEntity: (key: string | null) => void
  findings: NerSourceRow[]
  isFetchingFindings?: boolean
  hasNextPage?: boolean
  onLoadMore?: () => void
  collection: string
  keyOf: (e: NerEntityRow) => string
  entityMergeMode?: EntityMergeMode
  // Dedupe keys in the active report; when provided each finding shows an
  // "Add to report" control.
  reportDedupeKeys?: Set<string>
}

function entityOptionLabel(entity: NerEntityRow): string {
  const type = entity.type || 'Unlabeled'
  return `${entity.text} [${type}] · ${entity.mentions}`
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

export function EntityInspector({
  entities,
  selectedKey,
  onSelectEntity,
  findings,
  isFetchingFindings,
  hasNextPage,
  onLoadMore,
  collection,
  keyOf,
  entityMergeMode,
  reportDedupeKeys
}: Props) {
  const entityList = useMemo(
    () => entities.filter((e) => (e.text ?? '').trim().length > 0),
    [entities]
  )
  const types = useMemo(
    () => Array.from(new Set(entityList.map((e) => e.type || 'Unlabeled'))).sort(),
    [entityList]
  )
  // The dropdown's type filter is owned locally — it doesn't affect the
  // server fetch, only which entities the user can pick from.
  const typeFilterRef = useRef<HTMLSelectElement>(null)
  const filtered = useMemo(() => {
    const t = typeFilterRef.current?.value || ''
    if (!t) return entityList
    return entityList.filter((e) => (e.type || 'Unlabeled') === t)
  }, [entityList])

  const selected = useMemo(() => {
    if (selectedKey) {
      const hit = entityList.find((e) => keyOf(e) === selectedKey)
      if (hit) return hit
    }
    return filtered[0] ?? null
  }, [entityList, filtered, keyOf, selectedKey])

  // When the entity list arrives, seed selection so the dropdown has a
  // sensible default and the parent fires the source fetch.
  useEffect(() => {
    if (!selectedKey && selected) {
      onSelectEntity(keyOf(selected))
    }
  }, [keyOf, onSelectEntity, selected, selectedKey])

  // --- virtualized findings list ---
  const scrollRef = useRef<HTMLDivElement>(null)
  const virtualizer = useVirtualizer({
    count: findings.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => 96,
    overscan: 8
  })

  if (entityList.length === 0) {
    return (
      <div className="text-sm text-muted-foreground">No entities found in this collection.</div>
    )
  }

  // Mirrors the backend's ner-sources `entity_label` (``text [TYPE]``) so a
  // report's entity column matches the CSV export.
  const entityLabel = selected ? `${selected.text} [${selected.type || 'Unlabeled'}]` : undefined

  const exportParams =
    selected && selectedKey
      ? {
          entity_text: selected.text,
          entity_type: selected.type,
          entity_merge_mode: entityMergeMode
        }
      : undefined

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-[12rem_1fr] gap-3 items-end">
        <label className="flex flex-col gap-1 text-sm">
          <span className="text-xs uppercase text-muted-foreground">Entity category</span>
          <select
            ref={typeFilterRef}
            defaultValue=""
            onChange={() => onSelectEntity(null)}
            className="bg-zinc-900 border border-border rounded-md px-2 py-1"
          >
            <option value="">All</option>
            {types.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </label>
        <label className="flex flex-col gap-1 text-sm">
          <span className="text-xs uppercase text-muted-foreground">Entity</span>
          <select
            value={selected ? keyOf(selected) : ''}
            onChange={(e) => onSelectEntity(e.target.value || null)}
            className="bg-zinc-900 border border-border rounded-md px-2 py-1"
          >
            {filtered.length === 0 && <option value="">No entities</option>}
            {filtered.map((e) => (
              <option key={keyOf(e)} value={keyOf(e)}>
                {entityOptionLabel(e)}
              </option>
            ))}
          </select>
        </label>
      </div>

      {selected ? (
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
            {exportParams && (
              <div className="flex gap-2">
                <a
                  href={csvExportHref(collection, 'ner-sources', exportParams)}
                  download
                  className="px-3 py-1 rounded-md border border-border text-sm"
                >
                  CSV
                </a>
              </div>
            )}
          </div>
          {findings.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              {isFetchingFindings
                ? 'Loading findings…'
                : 'No chunks were matched for the selected entity.'}
            </p>
          ) : (
            <div
              ref={scrollRef}
              className="max-h-[60vh] overflow-y-auto"
              data-testid="ner-findings-scroll"
            >
              <ul
                className="relative space-y-2"
                style={{ height: `${virtualizer.getTotalSize()}px` }}
              >
                {virtualizer.getVirtualItems().map((vRow) => {
                  const source = findings[vRow.index]
                  return (
                    <li
                      key={source.chunk_id ?? vRow.index}
                      data-index={vRow.index}
                      ref={virtualizer.measureElement}
                      className="absolute left-0 right-0"
                      style={{ transform: `translateY(${vRow.start}px)` }}
                    >
                      <EntityFinding
                        index={vRow.index + 1}
                        source={source}
                        highlightTerms={highlightTermsForEntity(selected)}
                        selectedTypeLower={(selected.type || '').toLowerCase()}
                        defaultOpen={findings.length === 1}
                        entityLabel={entityLabel}
                        reportDedupeKeys={reportDedupeKeys}
                      />
                    </li>
                  )
                })}
              </ul>
              {hasNextPage && onLoadMore && (
                <div className="flex justify-center pt-2">
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
          )}
        </div>
      ) : (
        <p className="text-sm text-muted-foreground">
          Pick an entity from the dropdown to see the chunks where it appears.
        </p>
      )}
    </div>
  )
}
