import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useTheme } from '@infra/ui'
import { cn } from '@/lib/cn'
import type { ChatFinalEvent, Source } from '@/api/types'
import { SourcePills } from './SourcePills'
import { AnswerEntities } from './AnswerEntities'
import { ValidationBanner } from './ValidationBanner'
import { GraphDebugPanel } from './GraphDebugPanel'
import { EntityCandidatesPanel } from './EntityCandidatesPanel'
import { AddToReportButton } from '@/components/report/AddToReportButton'
import { chatAnswerSnapshot } from '@/lib/reportSnapshots'
import { useT } from '@/i18n/LanguageContext'

export interface ChatTurnData {
  user: string
  assistant: string
  done: boolean
  meta: ChatFinalEvent | null
  error?: string | null
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
        {turn.meta && <EntityCandidatesPanel meta={turn.meta} />}
      </div>
    </article>
  )
}
