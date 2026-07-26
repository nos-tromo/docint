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

export function entryMatches(entry: CollectionEntry, name: string | null, owner: string | null): boolean {
  return entry.name === name && entry.owner === owner
}
