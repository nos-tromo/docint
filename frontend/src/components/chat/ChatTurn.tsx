import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { WarningIcon, useTheme } from '@infra/ui'
import { cn } from '@/lib/cn'
import type { ChatFinalEvent, Source } from '@/api/types'
import { SourcePills } from './SourcePills'
import { AnswerEntities } from './AnswerEntities'
import { ValidationBanner } from './ValidationBanner'
import { GraphDebugPanel } from './GraphDebugPanel'
import { AddToReportButton } from '@/components/report/AddToReportButton'
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
  // The typography plugin's `prose-invert` hardcodes light-on-dark prose
  // colors; only apply it when the resolved theme is actually dark, or
  // markdown answers render illegibly on the light-theme `bg-muted` panel.
  const { resolved } = useTheme()
  const sources = dedupeSources(turn.meta?.sources ?? [])
  const reportItem =
    turn.done && turn.assistant && sessionId && turnIdx != null
      ? chatAnswerSnapshot({
          sessionId,
          turnIdx,
          userText: turn.user,
          modelResponse: turn.assistant,
          sources
        })
      : null
  const inReport = reportItem != null && (reportDedupeKeys?.has(reportItem.dedupe_key) ?? false)
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
          <div
            className={cn(
              'prose prose-sm max-w-none prose-p:my-2 prose-pre:bg-muted prose-code:before:content-none prose-code:after:content-none',
              resolved === 'dark' && 'prose-invert'
            )}
          >
            <Markdown remarkPlugins={[remarkGfm]}>{turn.assistant}</Markdown>
          </div>
        ) : (
          <div className="text-muted-foreground">{turn.done ? t('chat.no_answer') : '…'}</div>
        )}
        {turn.error && (
          <div className="mt-3 rounded-md border border-red-700 bg-red-950 px-3 py-2 text-xs text-red-200">
            <div className="font-medium">{t('chat.error_title')}</div>
            <div className="mt-1 whitespace-pre-wrap">{turn.error}</div>
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
