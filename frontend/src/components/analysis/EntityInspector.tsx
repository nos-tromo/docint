import { useMemo, useState } from 'react'
import type { NerEntityRow, NerSourceRow } from '@/api/types'
import { EntityFinding } from './EntityFinding'

interface Props {
  entities: NerEntityRow[]
  sources: NerSourceRow[]
}

function compactLookup(text: string): string {
  // Mirrors the backend's _compact_lookup so we match the same way the
  // aggregator did when it grouped variants under a cluster key.
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, '')
    .trim()
}

function matchTerms(entity: NerEntityRow): { exact: Set<string>; compact: Set<string> } {
  const exact = new Set<string>()
  const compact = new Set<string>()
  const seed = (t: string) => {
    const lower = t.trim().toLowerCase()
    if (!lower) return
    exact.add(lower)
    const c = compactLookup(lower)
    if (c) compact.add(c)
  }
  seed(entity.text)
  for (const v of entity.variants ?? []) seed(v.text ?? '')
  return { exact, compact }
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

function entityOptionLabel(entity: NerEntityRow): string {
  const type = entity.type || 'Unlabeled'
  return `${entity.text} [${type}] · ${entity.mentions}`
}

function sourceContainsEntity(source: NerSourceRow, entity: NerEntityRow): boolean {
  const { exact, compact } = matchTerms(entity)
  const typeLower = (entity.type || '').toLowerCase()
  const candidates = source.entities ?? []
  for (const cand of candidates) {
    const candText = (cand.text ?? '').trim().toLowerCase()
    if (!candText) continue
    const candType = (cand.type ?? '').toLowerCase()
    // Require the entity type to match so e.g. PER "Apple" doesn't match
    // ORG "Apple" — same behavior as the backend's cluster key.
    if (typeLower && candType && candType !== typeLower) continue
    if (exact.has(candText)) return true
    const candCompact = compactLookup(candText)
    if (candCompact && compact.has(candCompact)) return true
  }
  return false
}

export function EntityInspector({ entities, sources }: Props) {
  const [typeFilter, setTypeFilter] = useState('')
  const [selectedKey, setSelectedKey] = useState<string | null>(null)

  const entityList = useMemo(
    () => entities.filter((e) => (e.text ?? '').trim().length > 0),
    [entities]
  )
  const types = useMemo(
    () => Array.from(new Set(entityList.map((e) => e.type || 'Unlabeled'))).sort(),
    [entityList]
  )
  const filtered = useMemo(() => {
    if (!typeFilter) return entityList
    return entityList.filter((e) => (e.type || 'Unlabeled') === typeFilter)
  }, [entityList, typeFilter])

  const keyOf = (e: NerEntityRow) => `${e.text}::${e.type}`
  const selected =
    filtered.find((e) => keyOf(e) === selectedKey) ?? filtered[0] ?? null

  const findings = useMemo(() => {
    if (!selected) return [] as NerSourceRow[]
    return sources.filter((s) => sourceContainsEntity(s, selected))
  }, [sources, selected])

  if (entityList.length === 0) {
    return <div className="text-sm text-muted-foreground">No entities found in this collection.</div>
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-[12rem_1fr] gap-3 items-end">
        <label className="flex flex-col gap-1 text-sm">
          <span className="text-xs uppercase text-muted-foreground">Entity category</span>
          <select
            value={typeFilter}
            onChange={(e) => {
              setTypeFilter(e.target.value)
              setSelectedKey(null)
            }}
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
            onChange={(e) => setSelectedKey(e.target.value)}
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
          <div className="text-sm">
            <span className="text-muted-foreground">Findings for </span>
            <span className="font-medium">{selected.text}</span>
            <span className="text-muted-foreground">
              {' '}— {findings.length} chunk{findings.length === 1 ? '' : 's'}
            </span>
          </div>
          {findings.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              No chunks were matched for the selected entity.
            </p>
          ) : (
            <ul className="space-y-2">
              {findings.map((source, i) => (
                <li key={source.chunk_id ?? i}>
                  <EntityFinding
                    index={i + 1}
                    source={source}
                    highlightTerms={highlightTermsForEntity(selected)}
                    selectedTypeLower={(selected.type || '').toLowerCase()}
                    defaultOpen={findings.length === 1}
                  />
                </li>
              ))}
            </ul>
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
