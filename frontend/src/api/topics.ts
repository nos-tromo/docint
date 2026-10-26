import { apiGet, apiGetOrNull } from './client'
import type { Page } from './collections'
import { queueJob } from './jobs'
import type { Source } from './types'

/** The payload key a chunk's topic is stored under, valued `<run id>:<topic id>` (see {@link topicValue}). */
export const TOPIC_FIELD = 'docint_topic'

/**
 * What a topics job does: `build` fits the topics from scratch, `discover` looks
 * for new ones among the unassigned chunks, `align` places the chunks not yet placed.
 */
export type TopicsAction = 'build' | 'discover' | 'align'

/** One topic of a run, as `GET /collections/{name}/topics` lists it. */
export interface TopicSummary {
  id: number
  label: string
  description: string
  /** The words that set the topic apart, each with its weight. */
  keywords: [string, number][]
  /** Chunks in the topic. */
  size: number
  /** `size` over the collection's eligible chunks, a fraction from 0 to 1 and not a percentage. */
  share: number
  origin: 'build' | 'discover'
  added_at: string
  first_seen: string | null
  last_seen: string | null
}

/** How much of the collection the run has placed. */
export interface TopicsCoverage {
  total: number
  assigned: number
  unassigned: number
  /** Chunks the active run has not reached yet. */
  pending: number
}

/** A collection's active topic run, its coverage and its topics by size. */
export interface TopicsOverview {
  run_id: string
  created_at: string
  updated_at: string
  embed_model: string
  /** True when the collection is embedded with another model than the run was built with. */
  embed_model_changed: boolean
  coverage: TopicsCoverage
  topics: TopicSummary[]
}

/** One source file's chunk count within a topic. */
export interface TopicDocument {
  file_hash: string
  filename: string
  count: number
}

/**
 * The value {@link TOPIC_FIELD} holds for a topic.
 *
 * @param runId - The run the topic belongs to.
 * @param topicId - The topic's id, or `-1` for the run's unassigned chunks.
 * @returns `<run id>:<topic id>`.
 */
export function topicValue(runId: string, topicId: number): string {
  return `${runId}:${topicId}`
}

const topicsPath = (collection: string) => `/collections/${encodeURIComponent(collection)}/topics`

/**
 * Read a collection's active topic run.
 *
 * @param collection - The caller's logical collection name.
 * @returns The overview, or null when the collection has no run yet.
 */
export function getTopics(collection: string): Promise<TopicsOverview | null> {
  return apiGetOrNull<TopicsOverview>(topicsPath(collection))
}

/**
 * Queue a topics job over a collection.
 *
 * A 409 means a topics run is already in flight for that collection, whose
 * `job_id` is adopted (see {@link queueJob}). A 400 means there is no run to
 * discover on or align yet.
 *
 * @param collection - The caller's logical collection name.
 * @param action - What the job does.
 * @returns The job id, and whether it was adopted from an in-flight run.
 */
export function startTopics(
  collection: string,
  action: TopicsAction
): Promise<{ job_id: string; adopted: boolean }> {
  return queueJob(topicsPath(collection), { action })
}

/** A topic's passages nearest its centre. */
export function getTopicEvidence(
  collection: string,
  topicId: number,
  runId?: string
): Promise<{ topic: number; sources: Source[] }> {
  return apiGet(`${topicsPath(collection)}/${topicId}/evidence`, { run_id: runId })
}

/** The source files a topic's chunks come from, largest first; `more` when the list was cut. */
export function getTopicDocuments(
  collection: string,
  topicId: number,
  runId?: string
): Promise<{ documents: TopicDocument[]; more: boolean }> {
  return apiGet(`${topicsPath(collection)}/${topicId}/documents`, { run_id: runId })
}

/** One page of a topic's chunks. */
export function getTopicChunksPage(args: {
  collection: string
  topicId: number
  runId?: string
  cursor: string | null
  limit?: number
}): Promise<Page<Source>> {
  return apiGet(`${topicsPath(args.collection)}/${args.topicId}/chunks`, {
    cursor: args.cursor ?? undefined,
    limit: args.limit ?? 50,
    run_id: args.runId
  })
}
