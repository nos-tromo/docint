import { create } from 'zustand'
import { TOPIC_FIELD, topicValue } from '@/api/topics'
import type { MetadataFilter } from '@/api/types'

/** A topic the chat and the search are narrowed to, with the collection and run it was picked from. */
export interface TopicFocus {
  collection: string
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
 * The metadata rule a topic focus adds to a turn or a search.
 *
 * Only while the focus names a topic of the active collection's active run: an
 * earlier build's value matches no chunk, so sending it would empty the answer
 * rather than narrow it.
 *
 * @param focus - The focus, or null for none.
 * @param collection - The active collection, or null for none.
 * @param activeRunId - The collection's active run, or null when it has none or it is not known yet.
 * @returns The rule, or null when the focus does not apply.
 */
export function topicFocusRule(
  focus: TopicFocus | null,
  collection: string | null,
  activeRunId: string | null
): MetadataFilter | null {
  if (!focus || focus.collection !== collection || focus.runId !== activeRunId) return null
  return { field: TOPIC_FIELD, operator: 'eq', value: topicValue(focus.runId, focus.topicId) }
}
