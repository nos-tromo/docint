import { useMemo } from 'react'
import type { DocumentRecord } from '@/api/types'
import { KpiCard } from '@/components/common/KpiCard'
import { summarizeDocuments } from '@/lib/documentSummary'

interface Props {
  docs: DocumentRecord[]
  /** True unique-document count for the collection (accurate even when paginated). */
  totalCount?: number
  /** Whether more pages exist, i.e. the aggregate cards cover only loaded rows. */
  partial?: boolean
}

/** Join a list with a separator, collapsing the tail past `max` into `+N`. */
function joinCapped(items: string[], max: number, sep: string): string {
  if (items.length <= max) return items.join(sep)
  return `${items.slice(0, max).join(sep)}${sep}+${items.length - max}`
}

/**
 * At-a-glance summary strip above the inspector table: document and node totals,
 * the file-type breakdown, and the distinct entity types across the collection.
 */
export function DocumentSummary({ docs, totalCount, partial }: Props) {
  const summary = useMemo(() => summarizeDocuments(docs), [docs])

  const partialSuffix = partial ? '+' : ''
  const fileTypesHint = summary.fileTypes.length
    ? joinCapped(
        summary.fileTypes.map((t) => `${t.count} ${t.label}`),
        4,
        ' · '
      )
    : undefined
  const entityHint = summary.entityTypes.length
    ? joinCapped(summary.entityTypes, 8, ', ')
    : 'none extracted'

  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
      <KpiCard label="Documents" value={totalCount ?? summary.documentCount} />
      <KpiCard
        label="Nodes"
        value={`${summary.nodeCount}${partialSuffix}`}
        hint={partial ? 'across loaded documents' : undefined}
      />
      <KpiCard label="File types" value={summary.fileTypes.length || '—'} hint={fileTypesHint} />
      <KpiCard
        label="Entity types"
        value={summary.entityTypes.length || '—'}
        hint={entityHint}
      />
    </div>
  )
}
