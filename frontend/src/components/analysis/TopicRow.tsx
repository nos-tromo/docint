import { useId, type ReactNode } from 'react'
import { Badge, Button, DisclosureButton } from '@infra/ui'
import type { TopicSummary } from '@/api/topics'
import type { Source } from '@/api/types'
import { describeError } from '@/api/errorMessage'
import { ClampedText } from '@/components/common/ClampedText'
import { useTopicChunks, useTopicDocuments, useTopicEvidence } from '@/hooks/useTopics'
import { formatDate } from '@/lib/formatDate'
import { sourceLabel } from '@/lib/sourceLabel'
import { useLang, useT } from '@/i18n/LanguageContext'
import type { Lang, Strings } from '@/i18n'

/** Keywords a row shows; the CSV carries them all. */
const ROW_KEYWORDS = 5

type Translate = (key: keyof Strings, vars?: Record<string, string | number>) => string

/**
 * A topic's share of the collection as a percentage with one decimal.
 *
 * @param share - The share as the API sends it: a fraction from 0 to 1.
 * @param lang - The active language, which picks the decimal separator.
 * @returns The percentage, without its sign.
 */
function sharePercent(share: number, lang: Lang): string {
  return new Intl.NumberFormat(lang, { minimumFractionDigits: 1, maximumFractionDigits: 1 }).format(share * 100)
}

/**
 * The dates a topic's chunks span, as one date when they fall on the same day.
 *
 * @param topic - The topic.
 * @param lang - The active language.
 * @param t - The translator.
 * @returns The span, or null when the topic has no readable date.
 */
function seenRange(topic: TopicSummary, lang: Lang, t: Translate): string | null {
  const dates = [topic.first_seen, topic.last_seen]
    // An unreadable timestamp is skipped: `Intl` throws on an invalid date.
    .filter((iso): iso is string => typeof iso === 'string' && !Number.isNaN(Date.parse(iso)))
    .map((iso) => formatDate(iso, lang))
  if (dates.length === 0) return null
  if (dates.length === 1 || dates[0] === dates[1]) return dates[0]
  return t('topics.seen_range', { first: dates[0], last: dates[1] })
}

/** One of the expanded row's sections, named by its caption. */
function Section({ title, children }: { title: string; children: ReactNode }) {
  const captionId = useId()
  return (
    <section aria-labelledby={captionId} className="space-y-1.5">
      <div id={captionId} className="text-xs uppercase text-muted-foreground">
        {title}
      </div>
      {children}
    </section>
  )
}

/** A section's state line: loading, or nothing to list. */
function Muted({ children }: { children: ReactNode }) {
  return <p className="text-sm text-muted-foreground">{children}</p>
}

/** A read that failed, in catalog words: the backend's own message is never shown. */
function ReadError({ error }: { error: unknown }) {
  const t = useT()
  const described = describeError(error)
  return (
    <p role="alert" className="text-sm text-danger">
      {t(described.key, described.vars)}
    </p>
  )
}

/** One passage: where it comes from, then its text. */
function Passage({ source }: { source: Source }) {
  const t = useT()
  const text = (source.text ?? source.preview_text ?? '').trim()
  return (
    <li className="min-w-0 space-y-0.5">
      <div className="break-words text-xs font-medium">
        {source.filename ? sourceLabel(source, t) : t('common.unknown_source')}
      </div>
      {text ? (
        <ClampedText length={text.length} className="text-sm">
          {text}
        </ClampedText>
      ) : (
        <span className="text-xs text-muted-foreground">{t('common.chunk_text_unavailable')}</span>
      )}
    </li>
  )
}

/** Passages in the order the API sends them. */
function Passages({ sources }: { sources: Source[] }) {
  return (
    <ul className="space-y-2.5">
      {sources.map((source, index) => (
        // A source is keyed by its id; one without any keeps its place, since pages only append.
        // eslint-disable-next-line @eslint-react/no-array-index-key
        <Passage key={source.id ?? source.chunk_id ?? index} source={source} />
      ))}
    </ul>
  )
}

export interface TopicRowProps {
  topic: TopicSummary
  /** The run the topic belongs to; "Ask in chat" hands it on with the topic. */
  runId: string
  expanded: boolean
  onToggle: () => void
  onAsk: (topic: TopicSummary, runId: string) => void
}

/**
 * One topic of the active run: what it is about, how much of the collection it
 * holds, and, opened, the passages and documents behind it.
 *
 * The three reads behind the open row run only while it is open, so a run of
 * a few hundred topics costs nothing until one is looked at.
 */
export function TopicRow({ topic, runId, expanded, onToggle, onAsk }: TopicRowProps) {
  const t = useT()
  const lang = useLang()
  const labelId = useId()
  const panelId = useId()
  const evidence = useTopicEvidence(topic.id, expanded)
  const documents = useTopicDocuments(topic.id, expanded)
  const chunks = useTopicChunks(topic.id, expanded)

  const label = topic.label || `T${topic.id}`
  const keywords = topic.keywords.slice(0, ROW_KEYWORDS).map(([word]) => word)
  const seen = seenRange(topic, lang, t)
  const passages = chunks.data?.pages.flatMap((page) => page.items) ?? []

  return (
    <li data-testid="topic-row">
      {/* The middle column's basis is what lets the actions drop below it on a
          narrow screen instead of squeezing the label to a word per line. */}
      <div className="flex flex-wrap items-start gap-x-3 gap-y-2 px-3 py-2.5">
        <span className="min-w-8 pt-0.5 font-mono text-xs tabular-nums text-muted-foreground">T{topic.id}</span>
        <div className="min-w-0 flex-1 basis-56 space-y-1.5">
          <div className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5">
            <span id={labelId} className="min-w-0 break-words text-sm font-semibold">
              {label}
            </span>
            <span className="text-xs tabular-nums text-muted-foreground">
              {t('topics.size_share', { size: topic.size, share: sharePercent(topic.share, lang) })}
            </span>
          </div>
          {(topic.origin === 'discover' || keywords.length > 0 || seen) && (
            <div className="flex flex-wrap items-center gap-1">
              {topic.origin === 'discover' && <Badge variant="accent">{t('topics.origin_new')}</Badge>}
              {keywords.map((word) => (
                <Badge key={word} variant="neutral" className="max-w-full wrap-anywhere">
                  {word}
                </Badge>
              ))}
              {seen && <span className="ml-1 text-xs text-muted-foreground">{seen}</span>}
            </div>
          )}
        </div>
        <div className="ml-auto flex shrink-0 items-center gap-1">
          {/* Described by the label, so "Ask in chat" says which topic to a screen reader. */}
          <Button variant="ghost" size="sm" aria-describedby={labelId} onClick={() => onAsk(topic, runId)}>
            {t('topics.ask_in_chat')}
          </Button>
          <DisclosureButton
            expanded={expanded}
            controls={panelId}
            label={t(expanded ? 'topics.hide' : 'topics.show', { label })}
            onClick={onToggle}
          />
        </div>
      </div>
      <div id={panelId} hidden={!expanded}>
        {expanded && (
          <div className="space-y-4 border-t border-border bg-muted/30 px-3 py-3 sm:pl-14">
            {topic.description && <p className="text-sm leading-6">{topic.description}</p>}
            <Section title={t('topics.evidence')}>
              {evidence.isPending ? (
                <Muted>{t('common.loading_ellipsis')}</Muted>
              ) : evidence.isError ? (
                <ReadError error={evidence.error} />
              ) : evidence.data.sources.length === 0 ? (
                <Muted>{t('topics.section_empty')}</Muted>
              ) : (
                <Passages sources={evidence.data.sources} />
              )}
            </Section>
            <Section title={t('topics.documents')}>
              {documents.isPending ? (
                <Muted>{t('common.loading_ellipsis')}</Muted>
              ) : documents.isError ? (
                <ReadError error={documents.error} />
              ) : documents.data.documents.length === 0 ? (
                <Muted>{t('topics.section_empty')}</Muted>
              ) : (
                <>
                  {/* Narrower than the row, so a count stays within reach of its file name. */}
                  <ul className="max-w-xl space-y-1">
                    {documents.data.documents.map((doc) => (
                      <li key={doc.file_hash} className="flex items-baseline justify-between gap-3 text-sm">
                        <span className="min-w-0 break-words">{doc.filename || doc.file_hash}</span>
                        <span className="shrink-0 text-xs tabular-nums text-muted-foreground">{doc.count}</span>
                      </li>
                    ))}
                  </ul>
                  {documents.data.more && <Muted>{t('topics.documents_more')}</Muted>}
                </>
              )}
            </Section>
            <Section title={t('topics.passages')}>
              {chunks.isPending ? (
                <Muted>{t('common.loading_ellipsis')}</Muted>
              ) : passages.length > 0 ? (
                <Passages sources={passages} />
              ) : chunks.isError ? null : (
                <Muted>{t('topics.section_empty')}</Muted>
              )}
              {/* A later page failing keeps the passages already read. */}
              {chunks.isError && <ReadError error={chunks.error} />}
              {chunks.hasNextPage && (
                <Button
                  variant="secondary"
                  size="sm"
                  disabled={chunks.isFetchingNextPage}
                  onClick={() => void chunks.fetchNextPage()}
                >
                  {chunks.isFetchingNextPage ? t('common.loading_ellipsis') : t('table.load_more')}
                </Button>
              )}
            </Section>
          </div>
        )}
      </div>
    </li>
  )
}
