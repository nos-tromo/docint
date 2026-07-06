import type { DocumentRecord } from '@/api/types'
import { mimeLabel } from './documentFormat'

/** A file-type tally: the human label and how many documents carry it. */
export interface FileTypeTally {
  label: string
  count: number
}

/** Aggregate figures for the inspector's summary strip. */
export interface DocumentSummary {
  documentCount: number
  nodeCount: number
  fileTypes: FileTypeTally[]
  entityTypes: string[]
}

/**
 * Roll a page of document records up into the figures shown above the table.
 *
 * Pure and order-independent: file types are tallied by their {@link mimeLabel}
 * (most common first, ties broken alphabetically) and entity types are the sorted
 * distinct union. Counts reflect exactly the records passed in — callers signal
 * partial (paginated) data separately.
 *
 * @param docs The currently loaded document records.
 * @returns Document/node totals, the file-type breakdown, and distinct entity types.
 */
export function summarizeDocuments(docs: DocumentRecord[]): DocumentSummary {
  let nodeCount = 0
  const typeCounts = new Map<string, number>()
  const entitySet = new Set<string>()

  for (const doc of docs) {
    nodeCount += doc.node_count ?? 0
    const label = mimeLabel(doc.mimetype)
    typeCounts.set(label, (typeCounts.get(label) ?? 0) + 1)
    for (const entityType of doc.entity_types ?? []) {
      if (entityType) entitySet.add(entityType)
    }
  }

  const fileTypes = [...typeCounts.entries()]
    .map(([label, count]) => ({ label, count }))
    .sort((a, b) => b.count - a.count || a.label.localeCompare(b.label))
  const entityTypes = [...entitySet].sort((a, b) => a.localeCompare(b))

  return { documentCount: docs.length, nodeCount, fileTypes, entityTypes }
}
