import { useEffect, useMemo, useRef } from 'react'
import { useInfiniteQuery, useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  getTopicChunksPage,
  getTopicDocuments,
  getTopicEvidence,
  getTopics,
  startTopics,
  type TopicsAction
} from '@/api/topics'
import { useActiveCollection } from '@/hooks/useActiveCollection'
import { useConfig } from '@/hooks/useConfig'
import { selectJobEvents, useIngestJobsStore } from '@/stores/ingestJobs'
import { topicFocusRule, useTopicFocusStore } from '@/stores/topicFocus'
import type { IngestEvent, MetadataFilter } from '@/api/types'

/**
 * The active collection's topic run, its coverage and its topics by size; null
 * when it has no run yet. Fetched only while the deployment has topics switched on.
 */
export function useTopicsOverview() {
  const scope = useActiveCollection()
  const { collection } = scope
  const { data: config } = useConfig()
  return useQuery({
    queryKey: ['topics', scope],
    queryFn: () => getTopics(collection ?? ''),
    enabled: !!collection && config?.topics_enabled === true
  })
}

/**
 * The rule the topic focus adds to a turn or a search: null unless the focus
 * names a topic of the active collection's active run (see {@link topicFocusRule}).
 */
export function useTopicFocusRule(): MetadataFilter | null {
  const { collection } = useActiveCollection()
  const focus = useTopicFocusStore((s) => s.focus)
  const { data } = useTopicsOverview()
  return topicFocusRule(focus, collection, data?.run_id ?? null)
}

// A topic's reads are keyed by its run as well as its id: ids restart at 0 with every build, so
// without the run a rebuilt T0 would show the cached passages of the topic T0 used to name.

/** A topic's passages nearest its centre. */
export function useTopicEvidence(runId: string, topicId: number, enabled: boolean) {
  const scope = useActiveCollection()
  const { collection } = scope
  return useQuery({
    queryKey: ['topics', scope, runId, 'evidence', topicId],
    queryFn: () => getTopicEvidence(collection ?? '', topicId),
    enabled: !!collection && enabled
  })
}

/** The source files a topic's chunks come from. */
export function useTopicDocuments(runId: string, topicId: number, enabled: boolean) {
  const scope = useActiveCollection()
  const { collection } = scope
  return useQuery({
    queryKey: ['topics', scope, runId, 'documents', topicId],
    queryFn: () => getTopicDocuments(collection ?? '', topicId),
    enabled: !!collection && enabled
  })
}

/** A topic's chunks, a page at a time. */
export function useTopicChunks(runId: string, topicId: number, enabled: boolean) {
  const scope = useActiveCollection()
  const { collection } = scope
  return useInfiniteQuery({
    queryKey: ['topics', scope, runId, 'chunks', topicId],
    queryFn: ({ pageParam }) =>
      getTopicChunksPage({ collection: collection ?? '', topicId, cursor: pageParam as string | null }),
    initialPageParam: null as string | null,
    getNextPageParam: (last) => last.next_cursor,
    enabled: !!collection && enabled
  })
}

/**
 * Queue a topics job on the active collection. Its progress is not this
 * mutation's to report: it arrives on the shared job stream (see {@link useTopicsJob}).
 */
export function useStartTopics() {
  const { collection } = useActiveCollection()
  return useMutation({
    mutationFn: (action: TopicsAction) => {
      if (!collection) return Promise.reject(new Error('No collection is selected'))
      return startTopics(collection, action)
    }
  })
}

/** What the active collection's newest topics job is doing, as its frames say. */
export interface TopicsJobState {
  jobId: string | null
  action: TopicsAction | null
  /** Started and neither ended nor failed. */
  running: boolean
  stage: string | null
  done: number
  /** Units in the stage; 0 when the stage cannot say. */
  total: number
  /** Each warning's code is what to show copy for; its message is the backend's own and is not for display. */
  warnings: { code: string; message: string }[]
  /** The failure's code; null unless an `error` frame ended the job. */
  error: string | null
  /** Ended by completing or by being cancelled. A refusal completes too, with a coded warning. */
  finished: boolean
  /** Watched while it ran on this page, not only replayed on a connect (see `IngestJobsState.liveJobs`). */
  live: boolean
}

const NO_JOB: TopicsJobState = {
  jobId: null,
  action: null,
  running: false,
  stage: null,
  done: 0,
  total: 0,
  warnings: [],
  error: null,
  finished: false,
  live: false
}

function isTopicsAction(value: unknown): value is TopicsAction {
  return value === 'build' || value === 'discover' || value === 'align'
}

/**
 * Find the newest topics job on a collection among the job logs.
 *
 * Only a job's started frame names its collection, so that frame decides both
 * which collection a job belongs to and how new it is. The stream replays jobs
 * oldest first, so on a tie the job the store holds last is the newer one.
 *
 * @param events - Every job's log, by job id.
 * @param collection - The logical collection name, or null for none.
 * @returns The job id, or null when the collection has no topics job.
 */
function newestTopicsJob(events: Record<string, IngestEvent[]>, collection: string | null): string | null {
  if (!collection) return null
  let newest: string | null = null
  let newestAt = -Infinity
  for (const [jobId, frames] of Object.entries(events)) {
    const started = frames.find((frame) => frame.event === 'topics_started')
    if (!started || (started.data ?? {}).collection !== collection) continue
    const at = started.receivedAt ?? 0
    if (at >= newestAt) {
      newest = jobId
      newestAt = at
    }
  }
  return newest
}

/** Fold one job's frames into its state. */
function readJob(jobId: string, frames: IngestEvent[], live: boolean): TopicsJobState {
  let action: TopicsAction | null = null
  let stage: string | null = null
  let done = 0
  let total = 0
  const warnings: TopicsJobState['warnings'] = []
  let error: string | null = null
  let finished = false
  for (const frame of frames) {
    const data = frame.data ?? {}
    switch (frame.event) {
      case 'topics_started':
        action = isTopicsAction(data.action) ? data.action : null
        break
      case 'topics_progress':
        stage = typeof data.stage === 'string' ? data.stage : null
        done = typeof data.done === 'number' ? data.done : 0
        total = typeof data.total === 'number' ? data.total : 0
        break
      case 'warning':
        warnings.push({
          code: typeof data.code === 'string' ? data.code : '',
          message: typeof data.message === 'string' ? data.message : ''
        })
        break
      case 'topics_completed':
      case 'topics_cancelled':
        finished = true
        break
      case 'error':
        error = typeof data.code === 'string' && data.code ? data.code : 'topics_failed'
        break
    }
  }
  return { jobId, action, running: !finished && error === null, stage, done, total, warnings, error, finished, live }
}

/**
 * The newest topics job on the active collection, if any.
 *
 * Frames come from the shared job store, which the single owner-multiplexed
 * stream in `Shell` feeds, so a run survives leaving the screen it was started
 * from and a reload re-attaches to it.
 *
 * However a job ends, the topics and the document list (which carries each
 * file's topics) are refetched once per job. That is once per mounted caller,
 * so call this where the job is shown, not in every row; a job that ended
 * while no caller was mounted is refetched by the next caller to mount.
 */
export function useTopicsJob(): TopicsJobState {
  const scope = useActiveCollection()
  const queryClient = useQueryClient()
  const jobId = useIngestJobsStore((s) => newestTopicsJob(s.events, scope.collection))
  const frames = useIngestJobsStore(selectJobEvents(jobId))
  const live = useIngestJobsStore((s) => (jobId ? s.liveJobs[jobId] === true : false))
  const job = useMemo(() => (jobId ? readJob(jobId, frames, live) : NO_JOB), [jobId, frames, live])
  const ended = job.finished || job.error !== null
  // Created on first use: `useRef(new Set())` would build a set on every render.
  const invalidatedRef = useRef<Set<string> | null>(null)

  useEffect(() => {
    const invalidated = (invalidatedRef.current ??= new Set())
    if (!jobId || !ended || invalidated.has(jobId)) return
    invalidated.add(jobId)
    void queryClient.invalidateQueries({ queryKey: ['topics', scope] })
    void queryClient.invalidateQueries({ queryKey: ['documents-pages', scope] })
  }, [jobId, ended, scope, queryClient])

  return job
}
