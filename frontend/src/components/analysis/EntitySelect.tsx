import { useMemo, useState } from 'react'
import { SelectMenu } from '@infra/ui'
import type { NerEntityRow } from '@/api/types'
import { useT } from '@/i18n/LanguageContext'

interface Props {
  entities: NerEntityRow[]
  selectedKey: string | null
  onSelectEntity: (key: string | null) => void
  keyOf: (e: NerEntityRow) => string
}

// `type || 'Unlabeled'` mirrors the backend's own fallback (docint/core/ner.py
// et al.) — the entity's own "type" is protocol data and must not be
// translated, or the frontend key would diverge from the backend's CSV/report
// exports and the same-collection value it is deduped/matched against.
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
  const t = useT()
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
    return <div className="text-sm text-muted-foreground">{t('entities.empty')}</div>
  }

  // Keep the entity dropdown's value coherent with the active category: if the
  // current selection was filtered out, show this category's first entity.
  const valueInFiltered = filtered.some((e) => keyOf(e) === selectedKey)
    ? (selectedKey ?? '')
    : filtered[0]
      ? keyOf(filtered[0])
      : ''

  return (
    // Captions are spans rather than labels: a picker's trigger is a button,
    // and its text is the chosen entity rather than the name of the field.
    <div className="grid grid-cols-[12rem_1fr] gap-3 items-end">
      <div className="flex flex-col gap-1 text-sm">
        <span className="text-xs uppercase text-muted-foreground">{t('entities.category_label')}</span>
        <SelectMenu
          variant="field"
          label={t('entities.category_label')}
          options={[
            { value: '', label: t('entities.category_all') },
            ...types.map((ty) => ({ value: ty, label: ty }))
          ]}
          value={category}
          onChange={handleCategoryChange}
        />
      </div>
      <div className="flex flex-col gap-1 text-sm">
        <span className="text-xs uppercase text-muted-foreground">{t('entities.entity_label')}</span>
        <SelectMenu
          variant="field"
          label={t('entities.entity_label')}
          options={filtered.map((e) => ({ value: keyOf(e), label: entityOptionLabel(e) }))}
          value={valueInFiltered || null}
          onChange={(key) => onSelectEntity(key || null)}
          emptyLabel={t('entities.no_entities_option')}
        />
      </div>
    </div>
  )
}
