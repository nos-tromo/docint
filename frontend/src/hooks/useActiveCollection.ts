import { useMemo } from 'react'
import { useUiStore } from '@/stores/ui'

/** The active collection as the server resolves it: a logical name and, for an admin in another user's namespace, that user. */
export interface CollectionScope {
  collection: string | null
  owner: string | null
}

/**
 * The active collection together with its owner.
 *
 * Two owners can name a collection the same, so the name alone does not
 * identify one: an admin moving between two `docs` collections changes only
 * the owner. Every request already carries that owner (`withOwner`), so every
 * collection-scoped query key and effect must depend on this pair rather than
 * on the name, or it keeps showing the previous owner's data.
 *
 * @returns The active `{ collection, owner }`, stable until either changes.
 */
export function useActiveCollection(): CollectionScope {
  const collection = useUiStore((s) => s.selectedCollection)
  const owner = useUiStore((s) => s.selectedOwner)
  return useMemo(() => ({ collection, owner }), [collection, owner])
}
