import { useMemo, useState } from 'react'
import type { NerEntityRow } from '@/api/types'

interface Props {
  entities: NerEntityRow[]
  selectedKey: string | null
  onSelectEntity: (key: string | null) => void
  keyOf: (e: NerEntityRow) => string
}

function entityOptionLabel(entity: NerEntityRow): string {
  const type = entity.type || 'Unlabeled'
  return `${entity.text} [${type}] · ${entity.mentions}`
}

/**
 * Category + entity dropdown picker for the NER table view.
 *
 * The category filter is controlled state (not a ref) so changing it actually
 * re-filters the entity list, and picking a category pre-selects that
 * category's top entity — fixing the previous no-op selector that read a ref
 * inside a `useMemo` keyed only on the entity list.
 */
export function EntitySelect({ entities, selectedKey, onSelectEntity, keyOf }: Props) {
  const entityList = useMemo(
    () => entities.filter((e) => (e.text ?? '').trim().length > 0),
    [entities]
  )
  const types = useMemo(
    () => Array.from(new Set(entityList.map((e) => e.type || 'Unlabeled'))).sort(),
    [entityList]
  )
  const [category, setCategory] = useState('')

  const filtered = useMemo(
    () => (category ? entityList.filter((e) => (e.type || 'Unlabeled') === category) : entityList),
    [entityList, category]
  )

  function handleCategoryChange(next: string) {
    setCategory(next)
    // Pre-select the chosen category's top entity so the findings panel
    // updates immediately instead of stranding a now-filtered-out selection.
    const pool = next ? entityList.filter((e) => (e.type || 'Unlabeled') === next) : entityList
    onSelectEntity(pool.length ? keyOf(pool[0]) : null)
  }

  if (entityList.length === 0) {
    return (
      <div className="text-sm text-muted-foreground">No entities found in this collection.</div>
    )
  }

  // Keep the entity dropdown's value coherent with the active category: if the
  // current selection was filtered out, show this category's first entity.
  const valueInFiltered = filtered.some((e) => keyOf(e) === selectedKey)
    ? (selectedKey ?? '')
    : filtered[0]
      ? keyOf(filtered[0])
      : ''

  return (
    <div className="grid grid-cols-[12rem_1fr] gap-3 items-end">
      <label className="flex flex-col gap-1 text-sm">
        <span className="text-xs uppercase text-muted-foreground">Entity category</span>
        <select
          aria-label="Entity category"
          value={category}
          onChange={(e) => handleCategoryChange(e.target.value)}
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
          aria-label="Entity"
          value={valueInFiltered}
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
  )
}
