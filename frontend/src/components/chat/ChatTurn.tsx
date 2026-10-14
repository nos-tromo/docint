import { useId, useState } from 'react'
import { DisclosureButton, InfoIcon, SearchIcon, WarningIcon } from '@infra/ui'
import type { AgentReason, AgentStep, AgentTrail, ChatFinalEvent, Source } from '@/api/types'
import type { Strings } from '@/i18n'
import { SourcePills } from './SourcePills'
import { AnswerEntities } from './AnswerEntities'
import { ValidationBanner } from './ValidationBanner'
import { GraphDebugPanel } from './GraphDebugPanel'
import { AddToReportButton } from '@/components/report/AddToReportButton'
import { MarkdownProse } from '@/components/common/MarkdownProse'
import { chatAnswerSnapshot } from '@/lib/reportSnapshots'
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
  /** Research steps streamed while an assessment turn works; `meta.agent` replaces them once done. */
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
  tools_unavailable: 'chat.assessment_reason.tools_unavailable',
  research_failed: 'chat.assessment_reason.research_failed',
  context_budget: 'chat.assessment_reason.context_budget',
  time_budget: 'chat.assessment_reason.time_budget',
  no_evidence: 'chat.assessment_reason.no_evidence'
}

const ANSWER_COPY: Partial<Record<AgentTrail['answer'], keyof Strings>> = {
  unsplit: 'chat.assessment_answer.unsplit',
  empty: 'chat.assessment_answer.empty'
}

type TrailStep = Pick<AgentStep, 'action' | 'detail' | 'found'>

function stepLabel(step: TrailStep, t: ReturnType<typeof useT>): string {
  const single = step.found === 1
  if (step.action === 'seed') {
    return t(single ? 'chat.assessment_trail_seed_one' : 'chat.assessment_trail_seed_other', { count: step.found })
  }
  if (step.action === 'answer') return t('chat.assessment_trail_answer')
  return t(single ? 'chat.assessment_trail_search_one' : 'chat.assessment_trail_search_other', {
    query: step.detail,
    count: step.found
  })
}

/** The research steps of an assessment turn while it runs. */
function AgentTrailLive({ steps }: { steps: AgentStep[] }) {
  const t = useT()
  return (
    <ul className="mt-3 space-y-1 text-xs text-muted-foreground" data-testid="agent-trail-live">
      {steps.map((step) => (
        <li key={`${step.step}-${step.action}`} className="flex items-center gap-2">
          <SearchIcon className="h-3.5 w-3.5 shrink-0" />
          <span>{stepLabel(step, t)}</span>
        </li>
      ))}
    </ul>
  )
}

/** Says the answer carries the model's own judgement; the searches sit behind a disclosure. */
function AssessmentNotice({ trail }: { trail: AgentTrail }) {
  const t = useT()
  const [open, setOpen] = useState(false)
  const listId = useId()
  const searches = trail.actions.filter((action) => action.action === 'search')
  const shortened =
    trail.shortened > 0
      ? t(trail.shortened === 1 ? 'chat.assessment_shortened_one' : 'chat.assessment_shortened_other', {
          count: trail.shortened
        })
      : null
  return (
    <div
      className="mt-3 rounded-md border border-border bg-muted px-3 py-2 text-xs text-muted-foreground"
      data-testid="assessment-notice"
      role="note"
    >
      <div className="flex items-center gap-2">
        <InfoIcon className="h-3.5 w-3.5 shrink-0" />
        <span className="flex-1">
          {t('chat.assessment_notice')}
          {shortened && ` ${shortened}`}
        </span>
        {searches.length > 0 && (
          <DisclosureButton
            expanded={open}
            controls={listId}
            label={open ? t('chat.assessment_trail_hide') : t('chat.assessment_trail_show')}
            onClick={() => setOpen((value) => !value)}
            className="h-6 w-6"
          />
        )}
      </div>
      {open && (
        <ul id={listId} className="mt-2 list-disc space-y-1 pl-5">
          {searches.map((action, index) => (
            // A finished trail is read-only and never reordered; a repeated query rules out `detail` alone.
            // eslint-disable-next-line @eslint-react/no-array-index-key
            <li key={`${index}-${action.detail}`}>{stepLabel(action, t)}</li>
          ))}
        </ul>
      )}
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
  const reportItem =
    turn.done && turn.assistant && reportSessionId && turnIdx != null
      ? chatAnswerSnapshot({
          sessionId: reportSessionId,
          turnIdx,
          userText: turn.user,
          modelResponse: turn.assistant,
          sources
        })
      : null
  const inReport = reportItem != null && (reportDedupeKeys?.has(reportItem.dedupe_key) ?? false)
  const trail = turn.done && turn.meta?.answer_mode === 'assessment' ? (turn.meta.agent ?? null) : null
  // Wire tokens reach `t()` only through the tables above, by own key: "constructor" is not copy.
  const reasonKey =
    trail?.reason && Object.hasOwn(REASON_COPY, trail.reason) ? REASON_COPY[trail.reason] : undefined
  const answerKey = trail && Object.hasOwn(ANSWER_COPY, trail.answer) ? ANSWER_COPY[trail.answer] : undefined
  return (
    <article className="space-y-3">
      <div className="rounded-md bg-muted px-4 py-2 self-end max-w-2xl ml-auto">
        <div className="text-xs text-muted-foreground mb-1">{t('chat.you')}</div>
        <div className="whitespace-pre-wrap">{turn.user}</div>
      </div>
      <div className="rounded-md bg-muted border border-border px-4 py-3">
        <div className="flex items-center justify-between gap-2 mb-1">
          <div className="text-xs text-muted-foreground">{t('chat.assistant')}</div>
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
            {turn.agentSteps && turn.agentSteps.length > 0 && <AgentTrailLive steps={turn.agentSteps} />}
          </div>
        )}
        {turn.error && (
          <div className="mt-3 rounded-md border border-red-700 bg-red-950 px-3 py-2 text-xs text-red-200">
            <div className="font-medium">{t('chat.error_title')}</div>
            <div className="mt-1 whitespace-pre-wrap">{turn.error}</div>
          </div>
        )}
        {trail && <AssessmentNotice trail={trail} />}
        {reasonKey && (
          <div
            className="mt-3 rounded-md border border-[var(--status-amber-border)] bg-[var(--status-amber-surface)] px-3 py-2 text-xs text-[var(--status-amber-strong)]"
            data-testid="assessment-research-degraded"
            role="alert"
          >
            <WarningIcon className="mr-2 inline h-3.5 w-3.5 align-[-0.15em]" />
            {t(reasonKey)}
          </div>
        )}
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
