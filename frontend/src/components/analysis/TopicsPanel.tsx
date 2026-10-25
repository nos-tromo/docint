import { memo, useCallback, useState } from 'react'
import { Banner, Button, DownloadButton, NewButton, PlayButton, RefreshButton } from '@infra/ui'
import { topicValue, type TopicSummary, type TopicsAction } from '@/api/topics'
import { describeError } from '@/api/errorMessage'
import { useActiveCollection } from '@/hooks/useActiveCollection'
import { useStartTopics, useTopicsJob, useTopicsOverview } from '@/hooks/useTopics'
import { downloadCsv, toCsv } from '@/lib/csv'
import { useIngestJobsStore } from '@/stores/ingestJobs'
import { useT } from '@/i18n/LanguageContext'
import type { Strings } from '@/i18n'
import { TopicRow } from './TopicRow'

type AskInChat = (topic: TopicSummary, runId: string) => void

/** Copy for each stage a topics job reports; a stage the SPA does not know reads as `stage_other`. */
const STAGE_KEYS = new Map<string, keyof Strings>([
  ['starting', 'topics.stage_starting'],
  ['waiting', 'topics.stage_waiting'],
  ['scanning', 'topics.stage_scanning'],
  ['sampling', 'topics.stage_sampling'],
  ['fitting', 'topics.stage_fitting'],
  ['labelling', 'topics.stage_labelling'],
  ['assigning', 'topics.stage_assigning'],
  ['aligning', 'topics.stage_aligning']
])

/** Copy for each warning code a topics job can carry; the frame's own message is never shown. */
const WARNING_KEYS = new Map<string, keyof Strings>([
  ['none_found', 'topics.warning_none_found'],
  ['none_new', 'topics.warning_none_new'],
  ['too_few_chunks', 'topics.warning_too_few_chunks'],
  ['too_few_unassigned', 'topics.warning_too_few_unassigned'],
  ['no_run', 'topics.warning_no_run'],
  ['dimension_changed', 'topics.warning_dimension_changed'],
  ['collection_changed', 'topics.warning_collection_changed'],
  ['topics_unreachable', 'topics.warning_topics_unreachable'],
  ['topics_busy', 'topics.warning_topics_busy'],
  ['topics_too_large', 'topics.warning_topics_too_large'],
  ['topics_rejected', 'topics.warning_topics_rejected'],
  ['topics_bad_response', 'topics.warning_topics_bad_response']
])

const CSV_COLUMNS = ['id', 'label', 'size', 'share', 'keywords', 'first_seen', 'last_seen', 'origin'] as const

/**
 * The run's topics as CSV, one row per topic with every keyword.
 *
 * @param topics - The topics, in the order shown.
 * @returns The CSV text; `share` stays the API's fraction.
 */
function topicsCsv(topics: TopicSummary[]): string {
  const rows = topics.map((topic) => ({
    id: topic.id,
    label: topic.label,
    size: topic.size,
    share: topic.share,
    keywords: topic.keywords.map(([word]) => word).join('; '),
    first_seen: topic.first_seen,
    last_seen: topic.last_seen,
    origin: topic.origin
  }))
  return toCsv(rows, [...CSV_COLUMNS])
}

/**
 * The run's topics, one row each, with at most one open.
 *
 * Memoised on stable props: the panel re-renders on every progress frame of a
 * running job, and a run can list hundreds of topics that frame says nothing about.
 */
const TopicList = memo(function TopicList({
  topics,
  runId,
  openTopic,
  onToggleTopic,
  onAsk
}: {
  topics: TopicSummary[]
  runId: string
  openTopic: string | null
  onToggleTopic: (key: string) => void
  onAsk: AskInChat
}) {
  return (
    <ul className="divide-y divide-border rounded-md border border-border">
      {topics.map((topic) => {
        const key = topicValue(runId, topic.id)
        return (
          <TopicRow
            key={key}
            topic={topic}
            runId={runId}
            expanded={openTopic === key}
            onToggle={() => onToggleTopic(key)}
            onAsk={onAsk}
          />
        )
      })}
    </ul>
  )
})

/**
 * The Analysis → Topics tab: the active run's coverage, its topics, and the
 * actions that build, extend and align it.
 *
 * Keyed by the active collection, so a start, an open row or a run's outcome
 * never carries over to another collection.
 */
export function TopicsPanel({ onAsk }: { onAsk: AskInChat }) {
  const { collection, owner } = useActiveCollection()
  return <TopicsPanelBody key={JSON.stringify([collection, owner])} onAsk={onAsk} />
}

function TopicsPanelBody({ onAsk }: { onAsk: AskInChat }) {
  const t = useT()
  const { collection } = useActiveCollection()
  const overview = useTopicsOverview()
  const job = useTopicsJob()
  const start = useStartTopics()
  const [openTopic, setOpenTopic] = useState<string | null>(null)
  const toggleTopic = useCallback((key: string) => setOpenTopic((open) => (open === key ? null : key)), [])

  // Between the click and the new job's first frame, the job store still names
  // the previous job, finished; until that frame arrives the run is "starting".
  const startedJobId = start.data?.job_id ?? null
  const startedSeen = useIngestJobsStore(
    (s) => startedJobId !== null && (s.events[startedJobId] ?? []).some((frame) => frame.event === 'topics_started')
  )
  const awaiting = start.isPending || (startedJobId !== null && !startedSeen)
  const busy = awaiting || job.running
  const busyAction: TopicsAction | null = job.running ? job.action : awaiting ? (start.variables ?? null) : null

  // A job's outcome shows if it was started here or watched live on this page, never one only
  // replayed on a connect; progress shows for any running job, and a new start hides the last outcome.
  const current = !awaiting && job.jobId !== null && (job.jobId === startedJobId || job.live)
  const failed = current && job.error !== null
  const warningKeys = current
    ? [...new Set(job.warnings.map((w) => WARNING_KEYS.get(w.code) ?? 'topics.warning_generic'))]
    : []

  const progress = job.running
    ? { stage: job.stage ?? 'starting', done: job.done, total: job.total }
    : awaiting
      ? { stage: 'starting', done: 0, total: 0 }
      : null
  const stageText = progress
    ? t(STAGE_KEYS.get(progress.stage) ?? 'topics.stage_other', { done: progress.done, total: progress.total })
    : null

  if (!collection) {
    return <p className="text-sm text-muted-foreground">{t('topics.select_collection')}</p>
  }

  const data = overview.data
  const coverage = data?.coverage
  const errorText = (error: unknown) => {
    const described = describeError(error)
    return t(described.key, described.vars)
  }
  const rebuild = () => {
    // A rebuild replaces every topic, so a chat filter on the current ones stops matching.
    if (!window.confirm(t('topics.rebuild_confirm'))) return
    start.mutate('build')
  }
  const download = () => {
    if (data) downloadCsv(`${collection}-topics.csv`, topicsCsv(data.topics))
  }

  return (
    <div className="space-y-3">
      {data && coverage && (
        // The caption's basis drops the actions below it on a narrow screen rather than
        // squeezing four counts into a column beside them.
        <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-2">
          <p className="min-w-0 grow basis-72 text-sm text-muted-foreground">
            {t(data.topics.length === 1 ? 'topics.caption_topics_one' : 'topics.caption_topics_other', {
              count: data.topics.length
            })}
            {' · '}
            {t(coverage.total === 1 ? 'topics.caption_coverage_one' : 'topics.caption_coverage_other', {
              assigned: coverage.assigned,
              total: coverage.total,
              unassigned: coverage.unassigned,
              pending: coverage.pending
            })}
          </p>
          <div className="ml-auto flex shrink-0 items-center gap-1">
            <RefreshButton
              label={t('topics.rebuild')}
              busy={busyAction === 'build'}
              disabled={busy}
              onClick={rebuild}
            />
            <NewButton
              label={t('topics.discover')}
              busy={busyAction === 'discover'}
              disabled={busy || coverage.unassigned === 0}
              onClick={() => start.mutate('discover')}
            />
            <PlayButton
              label={t('topics.align')}
              busy={busyAction === 'align'}
              disabled={busy || coverage.pending === 0}
              onClick={() => start.mutate('align')}
            />
            <DownloadButton label={t('topics.download_csv')} onClick={download} />
          </div>
        </div>
      )}

      {overview.isPending && <p className="text-sm text-muted-foreground">{t('common.loading_ellipsis')}</p>}
      {overview.isError && <Banner variant="danger">{errorText(overview.error)}</Banner>}

      {data === null && (
        <div className="flex flex-col items-start gap-3 rounded-md border border-border border-dashed p-6">
          <p className="text-sm text-muted-foreground">{t('topics.empty')}</p>
          <Button variant="primary" disabled={busy} onClick={() => start.mutate('build')}>
            {t('topics.build')}
          </Button>
        </div>
      )}

      {progress && stageText && (
        <div className="space-y-1.5">
          <p className="text-sm text-muted-foreground">{stageText}</p>
          {progress.total > 0 && (
            <div
              data-testid="topics-progress"
              role="progressbar"
              aria-label={stageText}
              aria-valuemin={0}
              aria-valuemax={progress.total}
              aria-valuenow={Math.min(progress.done, progress.total)}
              className="h-1.5 w-full overflow-hidden rounded-full bg-muted"
            >
              <div
                className="h-full bg-primary transition-all"
                style={{ width: `${Math.min(100, Math.round((progress.done / progress.total) * 100))}%` }}
              />
            </div>
          )}
        </div>
      )}

      {start.isError && <Banner variant="danger">{errorText(start.error)}</Banner>}
      {(failed || warningKeys.length > 0) && (
        <Banner variant={failed ? 'danger' : 'info'} className="space-y-1">
          {failed && <p className="font-medium">{t('topics.failed')}</p>}
          {warningKeys.map((key) => (
            <p key={key}>{t(key)}</p>
          ))}
        </Banner>
      )}
      {data?.embed_model_changed && <Banner variant="info">{t('topics.embed_model_changed')}</Banner>}

      {data && data.topics.length > 0 && (
        <TopicList
          topics={data.topics}
          runId={data.run_id}
          openTopic={openTopic}
          onToggleTopic={toggleTopic}
          onAsk={onAsk}
        />
      )}
    </div>
  )
}
