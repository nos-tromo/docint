import type { CollectionsView } from '@/api/collections'

/** One selectable collection: own (owner === null) or a foreign owner's. */
export interface CollectionEntry {
  owner: string | null
  name: string
}

/** Flatten the admin listing into selectable entries: own first, then per-owner. */
export function buildCollectionEntries(view: CollectionsView): CollectionEntry[] {
  return [
    ...view.mine.map((name) => ({ owner: null, name })),
    ...view.others.flatMap((g) => g.collections.map((name) => ({ owner: g.owner, name })))
  ]
}

/**
 * Stable identity for one entry, for use as a picker's option value.
 *
 * Two owners may name a collection the same thing, so the name alone cannot
 * identify a row. The separator is a control character precisely because it
 * cannot occur in either half.
 */
export function entryKey(entry: CollectionEntry): string {
  return `${entry.owner ?? ''}\u001f${entry.name}`
}

export function entryMatches(entry: CollectionEntry, name: string | null, owner: string | null): boolean {
  return entry.name === name && entry.owner === owner
}
