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
import { getSessionTopic } from '@/api/sessions'
import { useActiveCollection, type CollectionScope } from '@/hooks/useActiveCollection'
import { useConfig } from '@/hooks/useConfig'
import { selectJobEvents, useIngestJobsStore } from '@/stores/ingestJobs'
import {
  focusInScope,
  fromSessionTopic,
  topicFocusRule,
  useTopicFocusStore,
  type TopicFocus
} from '@/stores/topicFocus'
import { useUiStore } from '@/stores/ui'
import type { IngestEvent, MetadataFilter } from '@/api/types'

/**
 * The active collection's topic run, its coverage and its topics by size; null
 * when it has no run yet. Fetched only while the deployment has topics switched on.
 *
 * @param enabled - False leaves the run unread, for a caller with nothing to ask of it yet.
 */
export function useTopicsOverview(enabled = true) {
  const scope = useActiveCollection()
  const { collection } = scope
  const { data: config } = useConfig()
  return useQuery({
    queryKey: ['topics', scope],
    queryFn: () => getTopics(collection ?? ''),
    enabled: enabled && !!collection && config?.topics_enabled === true
  })
}

/**
 * The key a session's stored topic is cached under.
 *
 * Deliberately outside `['sessions']`: every finished turn invalidates that
 * prefix, and a refetch landing while the topic is being removed would put it
 * back. The collection and its owner are part of it, as for the history.
 *
 * @param sessionId - The session.
 * @param scope - The active collection and its owner.
 * @returns The query key.
 */
export const sessionTopicKey = (sessionId: string, scope: CollectionScope) =>
  ['session-topic', sessionId, scope] as const

/** The open chat's topic, and whether it is still being read. */
export interface ChatTopicFocus {
  focus: TopicFocus | null
  /** True while a started chat's topic is not known yet. */
  pending: boolean
}

/**
 * The open chat's topic.
 *
 * Before its first turn a chat has no session, and its topic is the "Ask in
 * chat" pick. Once started, its topic is the one stored with its session, read
 * through the session's own collection — so each chat keeps its own topic and
 * gets it back after a reload, wherever it is reopened.
 */
export function useChatTopicFocus(): ChatTopicFocus {
  const sessionId = useUiStore((s) => s.currentSessionId)
  const scope = useActiveCollection()
  const picked = useTopicFocusStore((s) => s.focus)
  const config = useConfig()
  const enabled = !!sessionId && !!scope.collection && config.data?.topics_enabled === true
  const stored = useQuery({
    queryKey: sessionId ? sessionTopicKey(sessionId, scope) : ['session-topic', 'none'],
    queryFn: () => getSessionTopic(sessionId!, scope.collection!),
    enabled
  })
  const storedTopic = stored.data?.topic_focus
  const focus = useMemo(
    () => (enabled && storedTopic ? fromSessionTopic(storedTopic, scope) : null),
    [enabled, storedTopic, scope]
  )
  if (!sessionId) return { focus: picked, pending: false }
  return {
    focus,
    // A disabled query reads as pending too, so only an enabled read in flight counts; until the
    // flag is known, the read may still be due.
    pending: !!scope.collection && (config.isPending || (enabled && stored.isPending))
  }
}

/** The rule the topic focus adds to a turn or a search, and whether it is decided yet. */
export interface TopicFocusRuleState {
  /** Null unless the focus names a topic of the active collection's active run (see {@link topicFocusRule}). */
  rule: MetadataFilter | null
  /** True while the open chat's topic, or its run, is still being read. */
  pending: boolean
}

/**
 * The rule the open chat's topic adds to a turn or a search.
 *
 * The run is read only while a focus belongs to the active collection, so a
 * chat without one asks nothing of the topics routes.
 */
export function useTopicFocusRule(): TopicFocusRuleState {
  const scope = useActiveCollection()
  const { focus, pending: focusPending } = useChatTopicFocus()
  const inScope = focusInScope(focus, scope)
  const config = useConfig()
  const overview = useTopicsOverview(inScope)
  return {
    rule: topicFocusRule(focus, scope, overview.data?.run_id ?? null),
    // While the flag is unknown the run may still be read; once it is known, only a read in flight is.
    pending:
      focusPending ||
      (inScope && (config.isPending || (config.data?.topics_enabled === true && overview.isPending)))
  }
}

// A topic's reads are keyed by its run as well as its id: ids restart at 0 with every build, so
// without the run a rebuilt T0 would show the cached passages of the topic T0 used to name.

/** A topic's passages nearest its centre. */
export function useTopicEvidence(runId: string, topicId: number, enabled: boolean) {
  const scope = useActiveCollection()
  const { collection } = scope
  return useQuery({
    queryKey: ['topics', scope, runId, 'evidence', topicId],
    queryFn: () => getTopicEvidence(collection ?? '', topicId, runId),
    enabled: !!collection && enabled
  })
}

/** The source files a topic's chunks come from. */
export function useTopicDocuments(runId: string, topicId: number, enabled: boolean) {
  const scope = useActiveCollection()
  const { collection } = scope
  return useQuery({
    queryKey: ['topics', scope, runId, 'documents', topicId],
    queryFn: () => getTopicDocuments(collection ?? '', topicId, runId),
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
      getTopicChunksPage({ collection: collection ?? '', topicId, runId, cursor: pageParam as string | null }),
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
  /**
   * Each warning's code is what to show copy for; its message is the backend's own and is not for display.
   * A sample too small to fit also says how many chunks it held (`found`) and needs (`needed`).
   */
  warnings: { code: string; message: string; found?: number; needed?: number }[]
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
      case 'warning': {
        const warning: TopicsJobState['warnings'][number] = {
          code: typeof data.code === 'string' ? data.code : '',
          message: typeof data.message === 'string' ? data.message : ''
        }
        // Both or neither: copy that names only one count would read as a fault.
        if (typeof data.found === 'number' && typeof data.needed === 'number') {
          warning.found = data.found
          warning.needed = data.needed
        }
        warnings.push(warning)
        break
      }
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
 * Refetch the topics and the document list (which carries each file's topics)
 * once per ended job, however it ended: once per mounted caller, so a job that
 * ended while no caller was mounted is refetched by the next caller to mount.
 *
 * @param scope - The collection the job ran on.
 * @param jobId - The job, or null for none.
 * @param ended - Whether its frames include its end.
 */
function useRefreshOnJobEnd(scope: CollectionScope, jobId: string | null, ended: boolean): void {
  const queryClient = useQueryClient()
  // Created on first use: `useRef(new Set())` would build a set on every render.
  const invalidatedRef = useRef<Set<string> | null>(null)

  useEffect(() => {
    const invalidated = (invalidatedRef.current ??= new Set())
    if (!jobId || !ended || invalidated.has(jobId)) return
    invalidated.add(jobId)
    void queryClient.invalidateQueries({ queryKey: ['topics', scope] })
    void queryClient.invalidateQueries({ queryKey: ['documents-pages', scope] })
  }, [jobId, ended, scope, queryClient])
}

/**
 * Refetch what a topics job changes once the active collection's newest one
 * ends, for a screen that reads the topics without showing the job: the chat's
 * topic chip, the Inspector's column.
 *
 * Subscribes to whether the job has ended, not to its frames, so a running
 * job's progress re-renders nothing.
 */
export function useTopicsRefreshOnJobEnd(): void {
  const scope = useActiveCollection()
  const jobId = useIngestJobsStore((s) => newestTopicsJob(s.events, scope.collection))
  const ended = useIngestJobsStore((s) => (jobId ? s.terminal[jobId] === true : false))
  useRefreshOnJobEnd(scope, jobId, ended)
}

/**
 * The newest topics job on the active collection, if any.
 *
 * Frames come from the shared job store, which the single owner-multiplexed
 * stream in `Shell` feeds, so a run survives leaving the screen it was started
 * from and a reload re-attaches to it.
 *
 * However a job ends, the topics and the document list are refetched as
 * {@link useTopicsRefreshOnJobEnd} does, once per job per mounted caller, so
 * call this where the job is shown, not in every row.
 */
export function useTopicsJob(): TopicsJobState {
  const scope = useActiveCollection()
  const jobId = useIngestJobsStore((s) => newestTopicsJob(s.events, scope.collection))
  const frames = useIngestJobsStore(selectJobEvents(jobId))
  const live = useIngestJobsStore((s) => (jobId ? s.liveJobs[jobId] === true : false))
  const job = useMemo(() => (jobId ? readJob(jobId, frames, live) : NO_JOB), [jobId, frames, live])
  useRefreshOnJobEnd(scope, jobId, job.finished || job.error !== null)
  return job
}
