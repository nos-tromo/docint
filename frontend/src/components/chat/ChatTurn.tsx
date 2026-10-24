import { useId, useState } from 'react'
import { DisclosureButton, DocumentsIcon, ImageIcon, InfoIcon, SearchIcon, WarningIcon } from '@infra/ui'
import type { AgentAction, AgentReason, AgentStep, AgentTrail, ChatFinalEvent, Source } from '@/api/types'
import type { Strings } from '@/i18n'
import { cn } from '@/lib/cn'
import { SourcePills } from './SourcePills'
import { AnswerEntities } from './AnswerEntities'
import { ValidationBanner } from './ValidationBanner'
import { GraphDebugPanel } from './GraphDebugPanel'
import { AddToReportButton } from '@/components/report/AddToReportButton'
import { MarkdownProse } from '@/components/common/MarkdownProse'
import { chatAnswerSnapshot } from '@/lib/reportSnapshots'
import { retrievalTargetUsed, type NamedTarget } from '@/lib/retrievalTargetUsed'
import { useT } from '@/i18n/LanguageContext'

export interface ChatTurnData {
  user: string
  assistant: string
  done: boolean
  meta: ChatFinalEvent | null
  error?: string | null
  /** How many hand-picked chunks this turn was sent with, if any. */
  scopeRequested?: number
  /**
   * The reformulated query a corrective retry used, when the first answer was
   * rejected and re-answered. Set live from the stream's retry frame; on a
   * reloaded session it arrives on `meta` instead.
   */
  retryQuery?: string
  /**
   * Set on a turn sent in stateless mode. It is saved in no session, so it
   * names itself for the report instead of borrowing a session's id — every
   * stateless answer once shared one placeholder key, and adding one could
   * remove another.
   */
  unsavedId?: string
  /** Steps streamed while an agent turn works; `meta.agent` replaces them once done. */
  agentSteps?: AgentStep[]
}

/**
 * Whether this turn asked to be scoped and the server did not confirm it was.
 *
 * The scope banner describes the *chat*; only the final event says what a given
 * answer actually ran on. Without that comparison the two are indistinguishable
 * in the transcript — which is how an unscoped answer came to be presented as
 * hand-picked evidence.
 */
/**
 * The reformulated query behind this answer, if it came from a corrective retry.
 *
 * Reads the live stream frame first and the persisted turn second, so the
 * notice survives a reload rather than only existing while the tab that
 * watched the retry stays open.
 */
function retryQuery(turn: ChatTurnData): string | undefined {
  return turn.retryQuery ?? turn.meta?.retry_query
}

/**
 * Whether the server reported that this turn's sources were *not* re-ranked.
 *
 * The reranker degrades silently by design (a transport failure returns the
 * raw retrieval order rather than failing the turn). Measured on a live stack,
 * that let a day-long outage ship every answer's top-5 by fusion order with
 * nothing on screen to tell it apart from a healthy turn.
 */
function rerankWasSkipped(turn: ChatTurnData): boolean {
  if (!turn.done || !turn.meta?.rerank) return false
  return turn.meta.rerank.applied === false
}

function scopeWasDropped(turn: ChatTurnData): boolean {
  if (!turn.done || !turn.scopeRequested || !turn.meta) return false
  return turn.meta.retrieval_mode !== 'scoped'
}

/** Whether a visual turn answered without ever seeing the pictures.
 *
 *  The point of the visual target is that the model looks at the imagery, so
 *  a turn where the thumbnails could not be fetched answered from captions
 *  alone — a degraded turn the transcript must not present as a normal one,
 *  exactly like a skipped rerank.
 */
function visualPixelsMissing(turn: ChatTurnData): boolean {
  if (!turn.done || !turn.meta) return false
  return turn.meta.retrieval_target === 'visual' && turn.meta.visual?.images_attached === 0
}

const TARGET_USED_ICON: Record<NamedTarget, typeof DocumentsIcon> = {
  documents: DocumentsIcon,
  visual: ImageIcon
}

/** What a narrowed turn answered from, as a quiet chip beside the "Assistant" label. */
function TargetUsedChip({ target }: { target: NamedTarget }) {
  const t = useT()
  const Icon = TARGET_USED_ICON[target]
  const label = t(`chat.retrieval_target_used.${target}`)
  return (
    <span
      data-testid="retrieval-target-used"
      title={label}
      className="inline-flex min-w-0 max-w-full items-center gap-1 rounded border border-border bg-muted px-1.5 py-0.5 text-[11px]"
    >
      <Icon className="h-3 w-3 shrink-0" />
      <span className="truncate">{label}</span>
    </span>
  )
}

function dedupeSources(sources: Source[]): Source[] {
  // Image-only ingests emit a text-source plus an image-source for the
  // same file; the image-source often lacks file_hash, so its preview
  // link 404s. Drop those broken-preview duplicates only — keep every
  // other distinct chunk so multi-reference answers surface all of
  // their citations (transcript segments, multi-page PDFs, etc. share a
  // filename but point at different chunks).
  const filenameHasResolvableSibling = new Set<string>()
  for (const s of sources) {
    if (s.file_hash && s.filename) filenameHasResolvableSibling.add(s.filename)
  }
  const seen = new Set<string>()
  const out: Source[] = []
  for (const s of sources) {
    if (!s.file_hash && s.filename && filenameHasResolvableSibling.has(s.filename)) {
      continue
    }
    // Discriminate chunks by filename + page/row + a text-prefix
    // fingerprint so distinct chunks from the same page/file survive
    // (plain-text files have no page/row at all).
    const key = [
      s.filename ?? '',
      s.page ?? '',
      s.row ?? '',
      (s.text ?? s.preview_text ?? '').slice(0, 120)
    ].join('|')
    if (seen.has(key)) continue
    seen.add(key)
    out.push(s)
  }
  return out
}

/** Catalog copy per trail token. A token missing here renders no notice, never a crash. */
const REASON_COPY: Record<AgentReason, keyof Strings> = {
  tools_unavailable: 'chat.agent_reason.tools_unavailable',
  round_budget: 'chat.agent_reason.round_budget',
  time_budget: 'chat.agent_reason.time_budget',
  context_budget: 'chat.agent_reason.context_budget',
  idle: 'chat.agent_reason.idle',
  research_failed: 'chat.agent_reason.research_failed',
  no_evidence: 'chat.agent_reason.no_evidence',
  answer_cut: 'chat.agent_reason.answer_cut'
}

/** The limits a turn stops at by design, often at the smallest window; every other reason is a failure. */
const LIMIT_REASONS: ReadonlySet<AgentReason> = new Set<AgentReason>([
  'round_budget',
  'time_budget',
  'context_budget',
  'idle'
])

const ANSWER_COPY: Partial<Record<AgentTrail['answer'], keyof Strings>> = {
  unsplit: 'chat.assessment_answer.unsplit',
  empty: 'chat.assessment_answer.empty'
}

type TrailStep = Pick<AgentStep, 'action' | 'detail' | 'found' | 'field' | 'matches' | 'unchecked'>

/** A keyframes step's detail: the clip, then ` #n` when the model opened one frame. */
const ONE_FRAME = /^(.*) #(\d+)$/s

/** The Search panel's labels for the fields a keyword check can match besides the text. */
const KEYWORD_FIELD_LABEL: Partial<Record<string, keyof Strings>> = {
  author: 'search.field.author',
  network: 'search.field.network',
  uuid: 'search.field.uuid'
}

/** A keyword check: its words, its field unless that is the text, its exact count and what it could not reach. */
function keywordLabel(step: TrailStep, t: ReturnType<typeof useT>): string {
  // Wire tokens reach `t()` only through the table, by own key: "constructor" is not copy.
  const key = step.field ?? ''
  const field = Object.hasOwn(KEYWORD_FIELD_LABEL, key) ? KEYWORD_FIELD_LABEL[key] : undefined
  const unchecked = step.unchecked ?? 0
  const vars = {
    query: step.detail,
    where: field ? t('chat.agent_trail_keyword_field', { field: t(field) }) : '',
    matches: step.matches ?? 0,
    count: step.found,
    // A count over a partly indexed collection says what it left out.
    unchecked:
      unchecked > 0
        ? t(unchecked === 1 ? 'chat.agent_trail_keyword_unchecked_one' : 'chat.agent_trail_keyword_unchecked_other', {
            count: unchecked
          })
        : ''
  }
  // A failed count is unknown, never a zero, and its label has no number for the unchecked note to qualify.
  if (typeof step.matches !== 'number') return t('chat.agent_trail_keyword_unknown', vars)
  return t(step.matches === 1 ? 'chat.agent_trail_keyword_one' : 'chat.agent_trail_keyword_other', vars)
}

function stepLabel(step: TrailStep, t: ReturnType<typeof useT>): string {
  const single = step.found === 1
  switch (step.action) {
    case 'seed':
      return t(single ? 'chat.agent_trail_seed_one' : 'chat.agent_trail_seed_other', { count: step.found })
    case 'carried':
      return t(single ? 'chat.agent_trail_carried_one' : 'chat.agent_trail_carried_other', { count: step.found })
    case 'answer':
      return t('chat.agent_trail_answer')
    case 'note':
      return t('chat.agent_trail_note', { text: step.detail })
    case 'read':
      // The model may write the number as it cites it, `[2]`.
      return t('chat.agent_trail_read', { n: /\d+/.exec(step.detail)?.[0] ?? step.detail })
    case 'keyword':
      return keywordLabel(step, t)
    case 'search_images':
      return t(single ? 'chat.agent_trail_search_images_one' : 'chat.agent_trail_search_images_other', {
        query: step.detail,
        count: step.found
      })
    case 'keyframes': {
      const frame = ONE_FRAME.exec(step.detail)
      if (frame) return t('chat.agent_trail_keyframe', { clip: frame[1], index: frame[2] })
      return t(single ? 'chat.agent_trail_keyframes_one' : 'chat.agent_trail_keyframes_other', {
        clip: step.detail,
        count: step.found
      })
    }
    default:
      return t(single ? 'chat.agent_trail_search_one' : 'chat.agent_trail_search_other', {
        query: step.detail,
        count: step.found
      })
  }
}

/** What a step did, drawn. The model's own writing reads nothing: a same-size spacer keeps its text aligned. */
function StepMarker({ action }: { action: AgentAction }) {
  const className = 'mt-px h-3.5 w-3.5 shrink-0'
  switch (action) {
    case 'note':
    case 'answer':
      return <span aria-hidden="true" className={className} />
    case 'keyframes':
      return <ImageIcon className={className} />
    case 'seed':
    case 'carried':
    case 'read':
      return <DocumentsIcon className={className} />
    case 'keyword':
      return <SearchIcon className={className} />
    default:
      return <SearchIcon className={className} />
  }
}

/** A trail's steps, one row each: live while the turn runs, behind the disclosure once it is done. */
function TrailSteps({
  steps,
  id,
  testId,
  className
}: {
  steps: TrailStep[]
  id?: string
  testId?: string
  className?: string
}) {
  const t = useT()
  return (
    <ul id={id} className={cn('space-y-1', className)} data-testid={testId}>
      {steps.map((step, index) => (
        // Append-only and never reordered; one round's calls share their step number, so the index keys them.
        // eslint-disable-next-line @eslint-react/no-array-index-key
        <li key={index} className="flex items-start gap-2">
          <StepMarker action={step.action} />
          <span className="min-w-0 break-words">{stepLabel(step, t)}</span>
        </li>
      ))}
    </ul>
  )
}

/**
 * The steps behind the disclosure: not the seed a scope pinned, nor an earlier assessment's answer marker.
 * A follow-up's carried passages stay: no banner says the turn started from them.
 */
function researchSteps(actions: TrailStep[] | undefined): TrailStep[] {
  if (!Array.isArray(actions)) return []
  return actions.filter(
    (step) =>
      typeof step === 'object' &&
      step !== null &&
      typeof step.action === 'string' &&
      step.action !== 'seed' &&
      step.action !== 'answer'
  )
}

/** The searches behind a finished agent turn, behind a disclosure where the live trail stood. */
function AgentTrailDisclosure({ steps }: { steps: TrailStep[] }) {
  const t = useT()
  const [open, setOpen] = useState(false)
  const listId = useId()
  return (
    <div className="mt-3 text-xs text-muted-foreground" data-testid="agent-trail">
      <div className="flex items-center gap-2">
        <span className="flex-1">{t('chat.agent_trail_title')}</span>
        <DisclosureButton
          expanded={open}
          controls={listId}
          label={open ? t('chat.agent_trail_hide') : t('chat.agent_trail_show')}
          onClick={() => setOpen((value) => !value)}
          className="h-6 w-6"
        />
      </div>
      {open && <TrailSteps id={listId} steps={steps} className="mt-2" />}
    </div>
  )
}

/**
 * A finished turn's trail: behind the disclosure, unless all the turn did was carry passages over. It searched
 * nothing then, so its one row stands under the answer rather than behind a title that names searches.
 */
function FinishedTrail({ steps }: { steps: TrailStep[] }) {
  if (steps.every((step) => step.action === 'carried')) {
    return <TrailSteps steps={steps} testId="agent-trail" className="mt-3 text-xs text-muted-foreground" />
  }
  return <AgentTrailDisclosure steps={steps} />
}

/** Whether the answer text carries the assessment label the notice points to. */
function carriesLabel(trail: AgentTrail): boolean {
  return trail.answer === 'unsplit' || (trail.answer === 'split' && typeof trail.evidence_chars === 'number')
}

/**
 * Why research stopped: a limit is a muted note on how it ended, a failure an alert that evidence may be
 * missing or, for a cut answer, that the answer may be incomplete.
 */
function ReasonNotice({ reason }: { reason: AgentReason }) {
  const t = useT()
  if (LIMIT_REASONS.has(reason)) {
    return (
      <div
        className="mt-3 flex items-start gap-2 text-xs text-muted-foreground"
        data-testid="agent-reason-notice"
        role="note"
      >
        <InfoIcon className="mt-px h-3.5 w-3.5 shrink-0" />
        <span className="min-w-0">{t(REASON_COPY[reason])}</span>
      </div>
    )
  }
  return (
    <div
      className="mt-3 rounded-md border border-[var(--status-amber-border)] bg-[var(--status-amber-surface)] px-3 py-2 text-xs text-[var(--status-amber-strong)]"
      data-testid="agent-reason-notice"
      role="alert"
    >
      <WarningIcon className="mr-2 inline h-3.5 w-3.5 align-[-0.15em]" />
      {t(REASON_COPY[reason])}
    </div>
  )
}

/** Says the answer carries the model's own judgement. */
function AssessmentNotice({ trail }: { trail: AgentTrail }) {
  const t = useT()
  const shortened = trail.shortened ?? 0
  return (
    <div
      className="mt-3 flex items-center gap-2 rounded-md border border-border bg-muted px-3 py-2 text-xs text-muted-foreground"
      data-testid="assessment-notice"
      role="note"
    >
      <InfoIcon className="h-3.5 w-3.5 shrink-0" />
      <span className="flex-1">
        {t('chat.assessment_notice')}
        {shortened > 0 &&
          ` ${t(shortened === 1 ? 'chat.assessment_shortened_one' : 'chat.assessment_shortened_other', {
            count: shortened
          })}`}
      </span>
    </div>
  )
}

export function ChatTurn({
  turn,
  sessionId,
  turnIdx,
  reportDedupeKeys
}: {
  turn: ChatTurnData
  sessionId?: string
  turnIdx?: number
  reportDedupeKeys?: Set<string>
}) {
  const t = useT()
  const sources = dedupeSources(turn.meta?.sources ?? [])
  const reportSessionId = turn.unsavedId ? `stateless-${turn.unsavedId}` : sessionId
  // A failed turn is never saved: after a reload the next saved answer takes its index, and so its report key.
  const reportItem =
    turn.done && !turn.error && turn.assistant && reportSessionId && turnIdx != null
      ? chatAnswerSnapshot({
          sessionId: reportSessionId,
          turnIdx,
          userText: turn.user,
          modelResponse: turn.assistant,
          sources
        })
      : null
  const inReport = reportItem != null && (reportDedupeKeys?.has(reportItem.dedupe_key) ?? false)
  // Any agent turn's trail, and the one a plain answer carries when an agent turn handed it over.
  const trail = turn.done ? (turn.meta?.agent ?? null) : null
  const assessment = turn.meta?.answer_mode === 'assessment'
  // A failed turn sends no envelope: the steps it streamed are all that says what it did.
  const failedSteps = turn.done && !turn.meta ? turn.agentSteps : undefined
  const research = researchSteps(trail ? trail.actions : failedSteps)
  // Wire tokens reach `t()` only through the tables above, by own key: "constructor" is not copy.
  const reason = trail?.reason && Object.hasOwn(REASON_COPY, trail.reason) ? trail.reason : undefined
  const answerKey =
    assessment && trail && Object.hasOwn(ANSWER_COPY, trail.answer) ? ANSWER_COPY[trail.answer] : undefined
  const targetUsed = retrievalTargetUsed(turn.meta)
  return (
    <article className="space-y-3">
      <div className="rounded-md bg-muted px-4 py-2 self-end max-w-2xl ml-auto">
        <div className="text-xs text-muted-foreground mb-1">{t('chat.you')}</div>
        <div className="whitespace-pre-wrap">{turn.user}</div>
      </div>
      <div className="rounded-md bg-muted border border-border px-4 py-3">
        <div className="flex items-center justify-between gap-2 mb-1">
          <div className="flex min-w-0 items-center gap-2 text-xs text-muted-foreground">
            <span className="shrink-0">{t('chat.assistant')}</span>
            {targetUsed && <TargetUsedChip target={targetUsed} />}
          </div>
          {reportItem && reportDedupeKeys && <AddToReportButton item={reportItem} inReport={inReport} />}
        </div>
        {turn.assistant ? (
          <MarkdownProse text={turn.assistant} />
        ) : (
          <div className="text-muted-foreground">{turn.done ? t('chat.no_answer') : '…'}</div>
        )}
        {!turn.done && (
          // Mounted before the first step arrives, so a screen reader announces it; `atomic` off keeps
          // each new step from re-reading the whole list.
          <div role="status" aria-live="polite" aria-atomic="false">
            {turn.agentSteps && turn.agentSteps.length > 0 && (
              <TrailSteps
                steps={turn.agentSteps}
                testId="agent-trail-live"
                className="mt-3 text-xs text-muted-foreground"
              />
            )}
          </div>
        )}
        {research.length > 0 && <FinishedTrail steps={research} />}
        {turn.error && (
          <div className="mt-3 rounded-md border border-red-700 bg-red-950 px-3 py-2 text-xs text-red-200">
            <div className="font-medium">{t('chat.error_title')}</div>
            <div className="mt-1 whitespace-pre-wrap">{turn.error}</div>
          </div>
        )}
        {assessment && trail && carriesLabel(trail) && <AssessmentNotice trail={trail} />}
        {reason && <ReasonNotice reason={reason} />}
        {answerKey && (
          <div
            className="mt-3 rounded-md border border-[var(--status-amber-border)] bg-[var(--status-amber-surface)] px-3 py-2 text-xs text-[var(--status-amber-strong)]"
            data-testid="assessment-answer-degraded"
            role="alert"
          >
            <WarningIcon className="mr-2 inline h-3.5 w-3.5 align-[-0.15em]" />
            {t(answerKey)}
          </div>
        )}
        {retryQuery(turn) && (
          <div
            className="mt-3 rounded-md border border-[var(--status-amber-border)] bg-[var(--status-amber-surface)] px-3 py-2 text-xs text-[var(--status-amber-strong)]"
            data-testid="retry-notice"
            role="status"
          >
            {t('chat.retry_notice', { query: retryQuery(turn) ?? '' })}
          </div>
        )}
        {scopeWasDropped(turn) && (
          <div
            className="mt-3 rounded-md border border-[var(--status-amber-border)] bg-[var(--status-amber-surface)] px-3 py-2 text-xs text-[var(--status-amber-strong)]"
            data-testid="scope-not-applied"
            role="alert"
          >
            <WarningIcon className="mr-2 inline h-3.5 w-3.5 align-[-0.15em]" />
            {t('chat.scope_not_applied', { count: turn.scopeRequested ?? 0 })}
          </div>
        )}
        {rerankWasSkipped(turn) && (
          <div
            className="mt-3 rounded-md border border-[var(--status-amber-border)] bg-[var(--status-amber-surface)] px-3 py-2 text-xs text-[var(--status-amber-strong)]"
            data-testid="rerank-not-applied"
            role="alert"
          >
            <WarningIcon className="mr-2 inline h-3.5 w-3.5 align-[-0.15em]" />
            {t('chat.rerank_not_applied')}
          </div>
        )}
        {visualPixelsMissing(turn) && (
          <div
            className="mt-3 rounded-md border border-[var(--status-amber-border)] bg-[var(--status-amber-surface)] px-3 py-2 text-xs text-[var(--status-amber-strong)]"
            data-testid="visual-pixels-missing"
            role="alert"
          >
            <WarningIcon className="mr-2 inline h-3.5 w-3.5 align-[-0.15em]" />
            {t('chat.visual_pixels_missing')}
          </div>
        )}
        {turn.meta && <ValidationBanner v={turn.meta} />}
        {/* The entities summarize what the answer is about, so they sit
            between the answer and the evidence it rests on. */}
        <AnswerEntities sources={sources} />
        {sources.length > 0 && (
          <div className="mt-3 space-y-1.5">
            <div className="text-xs uppercase text-muted-foreground">{t('chat.sources')}</div>
            <SourcePills sources={sources} />
          </div>
        )}
        {!!turn.meta?.graph_debug && <GraphDebugPanel data={turn.meta.graph_debug} />}
      </div>
    </article>
  )
}
