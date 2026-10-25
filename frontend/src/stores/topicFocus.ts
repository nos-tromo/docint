import { create } from 'zustand'
import { TOPIC_FIELD, topicValue } from '@/api/topics'
import type { MetadataFilter } from '@/api/types'
import type { CollectionScope } from '@/hooks/useActiveCollection'

/** A topic the chat and the search are narrowed to, with the collection and run it was picked from. */
export interface TopicFocus {
  collection: string
  /** The collection's owner when an admin picked it in another user's namespace; null for one's own. */
  owner: string | null
  runId: string
  topicId: number
  label: string
}

interface TopicFocusState {
  focus: TopicFocus | null
  setFocus: (focus: TopicFocus) => void
  clear: () => void
}

/**
 * The topic "Ask in chat" narrowed the chat to.
 *
 * Not persisted: a topic belongs to one run of one collection, and a rebuild
 * replaces the run.
 */
export const useTopicFocusStore = create<TopicFocusState>()((set) => ({
  focus: null,
  setFocus: (focus) => set({ focus }),
  clear: () => set({ focus: null })
}))

/**
 * Whether a focus was picked in the active collection: the same name under the
 * same owner, since two owners can name a collection the same.
 *
 * @param focus - The focus, or null for none.
 * @param scope - The active collection and its owner.
 * @returns True when the focus belongs to that collection.
 */
export function focusInScope(focus: TopicFocus | null, scope: CollectionScope): focus is TopicFocus {
  return focus !== null && focus.collection === scope.collection && focus.owner === scope.owner
}

/**
 * The metadata rule a topic focus adds to a turn or a search.
 *
 * Only while the focus names a topic of the active collection's active run: an
 * earlier build's value matches no chunk, so sending it would empty the answer
 * rather than narrow it.
 *
 * @param focus - The focus, or null for none.
 * @param scope - The active collection and its owner.
 * @param activeRunId - The collection's active run, or null when it has none or it is not known yet.
 * @returns The rule, or null when the focus does not apply.
 */
export function topicFocusRule(
  focus: TopicFocus | null,
  scope: CollectionScope,
  activeRunId: string | null
): MetadataFilter | null {
  if (!focusInScope(focus, scope) || focus.runId !== activeRunId) return null
  return { field: TOPIC_FIELD, operator: 'eq', value: topicValue(focus.runId, focus.topicId) }
}
