import { create } from 'zustand'
import { TOPIC_FIELD, topicValue, type TopicSummary } from '@/api/topics'
import type { MetadataFilter, SessionTopicFocus } from '@/api/types'
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
 * The topic "Ask in chat" picked for a chat that has not started yet.
 *
 * Only that unstarted chat reads it. Its first turn sends it, the server keeps
 * it on the session that turn creates, and from then on the session's own copy
 * is the chat's topic (`useChatTopicFocus`) — which is what lets every chat keep
 * its own topic and get it back after a reload. Not persisted: before that
 * first turn there is no chat to keep it with.
 */
export const useTopicFocusStore = create<TopicFocusState>()((set) => ({
  focus: null,
  setFocus: (focus) => set({ focus }),
  clear: () => set({ focus: null })
}))

/**
 * A focus in the shape the API keeps on a session.
 *
 * @param focus - The focus.
 * @returns Its run, topic id and label.
 */
export function toSessionTopic(focus: TopicFocus): SessionTopicFocus {
  return { run_id: focus.runId, topic_id: focus.topicId, label: focus.label }
}

/**
 * A session's stored topic as a focus of the active collection.
 *
 * The API serves a session's topic only through the collection the session is
 * pinned to, so the active collection and its owner are the topic's own.
 *
 * @param stored - The session's topic as the API returns it.
 * @param scope - The active collection and its owner.
 * @returns The focus, or null with no collection active.
 */
export function fromSessionTopic(stored: SessionTopicFocus, scope: CollectionScope): TopicFocus | null {
  if (!scope.collection) return null
  return {
    collection: scope.collection,
    owner: scope.owner,
    runId: stored.run_id,
    topicId: stored.topic_id,
    label: stored.label
  }
}

/**
 * Whether a focus was picked in the active collection: the same name under the
 * same owner, since two owners can name a collection the same.
 *
 * @param focus - The focus, or null for none.
 * @param scope - The active collection and its owner.
 * @returns True when the focus belongs to that collection.
 */
export function focusInScope(focus: TopicFocus | null, scope: CollectionScope): boolean {
  return focus !== null && focus.collection === scope.collection && focus.owner === scope.owner
}

/** What a focus is checked against: the active run and the topics it shows. */
export interface FocusRun {
  run_id: string
  topics: Pick<TopicSummary, 'id' | 'members'>[]
}

/**
 * The topic a focus names in the active run, as the run shows it now.
 *
 * @param focus - The focus, or null for none.
 * @param scope - The active collection and its owner.
 * @param run - The collection's active run, or null/undefined when it has none or it is not read yet.
 * @returns The topic, or null when the focus is not one of the run's topics.
 */
export function focusedTopic<T extends Pick<TopicSummary, 'id' | 'members'>>(
  focus: TopicFocus | null,
  scope: CollectionScope,
  run: { run_id: string; topics: T[] } | null | undefined
): T | null {
  if (!focus || !focusInScope(focus, scope) || !run || focus.runId !== run.run_id) return null
  return run.topics.find((topic) => topic.id === focus.topicId) ?? null
}

/**
 * The metadata rule a topic focus adds to a turn or a search.
 *
 * Only while the focus names a topic the active run shows: an earlier build's
 * value matches no chunk, and a topic merged into another is no longer one of
 * the run's own, so sending either would empty the answer rather than narrow
 * it. A topic that merges several of the run's topics matches any of them.
 *
 * @param focus - The focus, or null for none.
 * @param scope - The active collection and its owner.
 * @param run - The collection's active run, or null/undefined when it has none or it is not read yet.
 * @returns The rule, or null when the focus does not apply.
 */
export function topicFocusRule(
  focus: TopicFocus | null,
  scope: CollectionScope,
  run: FocusRun | null | undefined
): MetadataFilter | null {
  const topic = focusedTopic(focus, scope, run)
  if (!focus || !topic) return null
  const members = topic.members ?? [topic.id]
  if (members.length === 1) return { field: TOPIC_FIELD, operator: 'eq', value: topicValue(focus.runId, members[0]) }
  return { field: TOPIC_FIELD, operator: 'in', values: members.map((id) => topicValue(focus.runId, id)) }
}
