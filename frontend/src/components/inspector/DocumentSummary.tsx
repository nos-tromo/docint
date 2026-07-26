import type { DocumentsSummary } from '@/api/types'
import { KpiCard } from '@/components/common/KpiCard'
import { useT } from '@/i18n/LanguageContext'

interface Props {
  /**
   * Collection-wide aggregates from `GET /collections/documents/summary`.
   * Undefined while loading. Computed over the whole collection, so the file-
   * type / entity-type counts are accurate no matter how many document pages
   * the inspector table has scrolled in.
   */
  summary?: DocumentsSummary
}

/** Join a list with a separator, collapsing the tail past `max` into `+N`. */
function joinCapped(items: string[], max: number, sep: string): string {
  if (items.length <= max) return items.join(sep)
  return `${items.slice(0, max).join(sep)}${sep}+${items.length - max}`
}

/**
 * At-a-glance summary strip above the inspector table: document and node totals,
 * the file-type breakdown, and the distinct entity types across the collection.
 * Every figure is collection-wide (server-aggregated), so it never undercounts
 * as the paginated document table lazily loads rows.
 */
export function DocumentSummary({ summary }: Props) {
  const t = useT()
  const fileTypes = summary?.file_types ?? []
  const entityTypes = summary?.entity_types ?? []

  const fileTypesHint = fileTypes.length
    ? joinCapped(
        fileTypes.map((ft) => `${ft.count} ${ft.label}`),
        4,
        ' · '
      )
    : undefined
  const entityHint = entityTypes.length
    ? joinCapped(entityTypes, 8, ', ')
    : t('inspector.none_extracted')

  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
      <KpiCard label={t('table.aria_documents')} value={summary?.document_count ?? '—'} />
      <KpiCard label={t('table.col_nodes')} value={summary?.node_count ?? '—'} />
      <KpiCard
        label={t('inspector.file_types')}
        value={fileTypes.length || '—'}
        hint={fileTypesHint}
      />
      <KpiCard
        label={t('inspector.entity_types')}
        value={entityTypes.length || '—'}
        hint={entityHint}
      />
    </div>
  )
}
