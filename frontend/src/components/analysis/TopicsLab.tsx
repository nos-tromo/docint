import { useId, useState, type ReactNode } from 'react'
import { ActionBar, Badge, Banner, Button, Input, SectionHeader, ToggleButton, UndoButton } from '@infra/ui'
import type { TopicsOverview, TopicsRequest, TopicsTuning } from '@/api/topics'
import { describeError } from '@/api/errorMessage'
import { useTopicsTuning } from '@/hooks/useTopics'
import {
  cutSimilarity,
  cutTree,
  draftTopics,
  highestBar,
  mergeGroups,
  relaxCount,
  sameGroups,
  splitGroup,
  topicCount,
  type TopicGroups
} from '@/lib/topicMerge'
import { useLang, useT } from '@/i18n/LanguageContext'
import type { Lang } from '@/i18n'

/** Bounds of a rebuild's minimum topic size, as the API takes it. */
const MIN_TOPIC_SIZE = 2
const MAX_TOPIC_SIZE = 10_000

/** One formatter per language for similarities, which read as two decimals. */
const SIMILARITY_FORMATS = new Map<Lang, Intl.NumberFormat>()

function similarityText(value: number, lang: Lang): string {
  let format = SIMILARITY_FORMATS.get(lang)
  if (!format) {
    format = new Intl.NumberFormat(lang, { minimumFractionDigits: 2, maximumFractionDigits: 2 })
    SIMILARITY_FORMATS.set(lang, format)
  }
  return format.format(value)
}

interface TopicsLabProps {
  overview: TopicsOverview
  /** Queues a topics job through the panel's own mutation, whose progress and outcome the panel shows. */
  onStart: (request: TopicsRequest) => void
  /** A topics job is starting or running: every control waits for it. */
  busy: boolean
}

/**
 * The Topics tab's fine-tuning controls: merge similar topics, place
 * unassigned chunks under a lowered bar, and recompute with another minimum
 * topic size.
 *
 * A merge is previewed here, from the tree the tuning view sends once, so
 * moving the slider asks nothing of the server; only applying it queues a job.
 * The panel keys this by run, so a rebuild starts a fresh draft.
 */
export function TopicsLab({ overview, onStart, busy }: TopicsLabProps) {
  const t = useT()
  const tuning = useTopicsTuning(true)
  if (tuning.isError) {
    const described = describeError(tuning.error)
    return <Banner variant="danger">{t(described.key, described.vars)}</Banner>
  }
  // Until the run read here is the overview's, its tree is not the one shown.
  if (!tuning.data || tuning.data.run_id !== overview.run_id) {
    return <p className="text-sm text-muted-foreground">{t('common.loading_ellipsis')}</p>
  }
  return (
    <fieldset disabled={busy} className="min-w-0 space-y-5 rounded-md border border-border p-4">
      <legend className="px-1 text-sm font-semibold">{t('topics.lab_title')}</legend>
      <MergeSection tuning={tuning.data} onStart={onStart} />
      <OutlierSection tuning={tuning.data} unassigned={overview.coverage.unassigned} onStart={onStart} />
      <RebuildSection minTopicSize={tuning.data.min_topic_size} onStart={onStart} />
    </fieldset>
  )
}

/** One fine-tuning step: its header (title, hint, quiet actions), then its body. */
function Section({
  title,
  hint,
  actions,
  children
}: {
  title: string
  hint: string
  actions?: ReactNode
  children: ReactNode
}) {
  const titleId = useId()
  return (
    <section aria-labelledby={titleId} className="space-y-3">
      <SectionHeader level={3} title={title} caption={hint} titleId={titleId} actions={actions} />
      {children}
    </section>
  )
}

function MergeSection({ tuning, onStart }: { tuning: TopicsTuning; onStart: (request: TopicsRequest) => void }) {
  const t = useT()
  const lang = useLang()
  // The draft's groups; null while it is the stored reduction.
  const [draft, setDraft] = useState<TopicGroups | null>(null)
  const [selected, setSelected] = useState<number[]>([])
  const leafIds = tuning.leaves.map((leaf) => leaf.id)
  const total = leafIds.length
  const groups = draft ?? tuning.groups
  const count = topicCount(total, groups)
  const dirty = draft !== null && !sameGroups(draft, tuning.groups)
  const topics = draftTopics(tuning, groups)
  const shown = new Set(topics.map((topic) => topic.id))
  const picked = selected.filter((id) => shown.has(id))
  // The similarity a cut stops at says something only while the draft is a cut of the tree.
  const similarity = sameGroups(groups, cutTree(leafIds, tuning.merges, count))
    ? cutSimilarity(total, tuning.merges, count)
    : null

  const edit = (next: TopicGroups | null) => {
    setDraft(next)
    setSelected([])
  }
  const toggle = (id: number) =>
    setSelected((current) => (current.includes(id) ? current.filter((other) => other !== id) : [...current, id]))

  return (
    <Section
      title={t('topics.lab_merge_title')}
      hint={t('topics.lab_merge_hint')}
      actions={
        <UndoButton
          label={t('topics.lab_reset')}
          disabled={tuning.groups.length === 0}
          onClick={() => {
            // It queues a job that drops every merge applied so far.
            if (!window.confirm(t('topics.lab_reset_confirm'))) return
            edit([])
            onStart({ action: 'reduce', run_id: tuning.run_id, groups: [] })
          }}
        />
      }
    >
      {total > 2 && (
        <div className="space-y-1">
          <input
            type="range"
            min={2}
            max={total}
            step={1}
            value={count}
            aria-label={t('topics.lab_merge_slider')}
            aria-valuetext={t('topics.lab_merge_count', { count, total })}
            onChange={(e) => edit(cutTree(leafIds, tuning.merges, Number(e.target.value)))}
            className="w-full cursor-pointer accent-primary"
          />
          <p className="text-xs tabular-nums text-muted-foreground">
            {t('topics.lab_merge_count', { count, total })}
            {similarity !== null &&
              ` · ${t('topics.lab_merge_similarity', { similarity: similarityText(similarity, lang) })}`}
          </p>
        </div>
      )}
      {dirty && (
        <p role="status" className="text-xs text-foreground">
          {t('topics.lab_preview', { count })}
        </p>
      )}
      <div className="rounded-md border border-border">
        <SectionHeader
          className="border-b border-border px-2 py-1.5"
          caption={
            <span className="tabular-nums">
              {t('topics.lab_selected', { selected: picked.length, count: topics.length })}
            </span>
          }
          actions={
            <Button
              variant="ghost"
              size="sm"
              disabled={picked.length < 2}
              onClick={() => edit(mergeGroups(groups, picked))}
            >
              {t('topics.lab_merge_selected')}
            </Button>
          }
        />
        <ul className="max-h-96 divide-y divide-border overflow-y-auto">
          {topics.map((topic) => (
            <li key={topic.id} className="flex items-center gap-2 px-2 py-1.5">
              <ToggleButton
                size="sm"
                pressed={picked.includes(topic.id)}
                onClick={() => toggle(topic.id)}
                className="min-w-0 flex-1 justify-start gap-2 text-left"
              >
                <span className="font-mono text-xs tabular-nums">T{topic.id}</span>
                <span className="min-w-0 truncate">{topic.label || `T${topic.id}`}</span>
              </ToggleButton>
              <span className="shrink-0 text-xs tabular-nums text-muted-foreground">{topic.size}</span>
              {topic.members.length > 1 && (
                <>
                  <Badge variant="accent" className="shrink-0">
                    {t('topics.merged_from', { count: topic.members.length })}
                  </Badge>
                  <Button variant="ghost" size="sm" onClick={() => edit(splitGroup(groups, topic.id))}>
                    {t('topics.lab_split')}
                  </Button>
                </>
              )}
            </li>
          ))}
        </ul>
      </div>
      <ActionBar>
        <Button variant="secondary" disabled={!dirty} onClick={() => edit(null)}>
          {t('topics.lab_discard')}
        </Button>
        <Button disabled={!dirty} onClick={() => onStart({ action: 'reduce', run_id: tuning.run_id, groups })}>
          {t('topics.lab_apply')}
        </Button>
      </ActionBar>
    </Section>
  )
}

function OutlierSection({
  tuning,
  unassigned,
  onStart
}: {
  tuning: TopicsTuning
  unassigned: number
  onStart: (request: TopicsRequest) => void
}) {
  const t = useT()
  const lang = useLang()
  const { relax } = tuning
  const histogram = relax.histogram
  const top = histogram ? highestBar(histogram) : null
  const [bar, setBar] = useState<number | null>(null)
  const value = Math.min(bar ?? relax.threshold ?? top ?? 0.01, top ?? 0.01)
  const placed = histogram ? relaxCount(histogram, value) : 0
  const preview = t(placed === 1 ? 'topics.lab_outliers_preview_one' : 'topics.lab_outliers_preview_other', {
    count: placed,
    threshold: similarityText(value, lang)
  })
  const relaxAt = (threshold: number | null) => onStart({ action: 'relax', run_id: tuning.run_id, threshold })

  return (
    <Section
      title={t('topics.lab_outliers_title')}
      hint={t('topics.lab_outliers_hint')}
      actions={
        <UndoButton label={t('topics.lab_outliers_undo')} disabled={relax.applied === 0} onClick={() => relaxAt(null)} />
      }
    >
      <p className="text-xs tabular-nums text-muted-foreground">
        {t('topics.lab_outliers_counts', { unassigned, applied: relax.applied })}
      </p>
      {histogram && relax.stale && <p className="text-xs text-foreground">{t('topics.lab_outliers_stale')}</p>}
      {histogram && !relax.stale && top === null && (
        <p className="text-xs text-muted-foreground">{t('topics.lab_outliers_none')}</p>
      )}
      {histogram && !relax.stale && top !== null && (
        <div className="space-y-1">
          <input
            type="range"
            min={0.01}
            max={top}
            step={0.01}
            value={value}
            aria-label={t('topics.lab_outliers_slider')}
            aria-valuetext={preview}
            onChange={(e) => setBar(Number(e.target.value))}
            className="w-full cursor-pointer accent-primary"
          />
          <p className="text-xs tabular-nums text-muted-foreground">{preview}</p>
        </div>
      )}
      <ActionBar>
        {(!histogram || relax.stale) && (
          <Button
            variant="secondary"
            // Measuring undoes an applied relax, so a stale one is applied again at its own bar instead.
            onClick={() => relaxAt(relax.applied > 0 ? relax.threshold : null)}
          >
            {t(relax.applied > 0 ? 'topics.lab_outliers_reapply' : 'topics.lab_outliers_measure')}
          </Button>
        )}
        {histogram && !relax.stale && top !== null && (
          <Button
            variant="primary"
            disabled={placed === 0 || value === relax.threshold}
            onClick={() => relaxAt(value)}
          >
            {t('topics.lab_outliers_apply')}
          </Button>
        )}
      </ActionBar>
    </Section>
  )
}

function RebuildSection({
  minTopicSize,
  onStart
}: {
  minTopicSize: number | null
  onStart: (request: TopicsRequest) => void
}) {
  const t = useT()
  const hintId = useId()
  const [size, setSize] = useState('')
  const asked = size.trim() === '' ? null : Number(size)
  const valid = asked === null || (Number.isInteger(asked) && asked >= MIN_TOPIC_SIZE && asked <= MAX_TOPIC_SIZE)
  const rebuild = () => {
    // A rebuild replaces every topic, its merges and placements, so a chat filter on the current ones stops matching.
    if (!window.confirm(t('topics.rebuild_confirm'))) return
    onStart(asked === null ? { action: 'build' } : { action: 'build', min_topic_size: asked })
  }
  return (
    <Section title={t('topics.lab_rebuild_title')} hint={t('topics.lab_rebuild_hint')}>
      <div>
        <label className="block text-sm">
          <span>{t('topics.lab_rebuild_size')}</span>
          <Input
            type="number"
            min={MIN_TOPIC_SIZE}
            max={MAX_TOPIC_SIZE}
            step={1}
            value={size}
            placeholder={t('topics.lab_rebuild_auto')}
            aria-invalid={!valid || undefined}
            aria-describedby={minTopicSize !== null ? hintId : undefined}
            onChange={(e) => setSize(e.target.value)}
            className="mt-1"
          />
        </label>
        {minTopicSize !== null && (
          <p id={hintId} className="mt-1 text-xs text-muted-foreground">
            {t('topics.lab_rebuild_current', { size: minTopicSize })}
          </p>
        )}
      </div>
      <ActionBar>
        <Button variant="secondary" disabled={!valid} onClick={rebuild}>
          {t('topics.lab_rebuild')}
        </Button>
      </ActionBar>
    </Section>
  )
}
