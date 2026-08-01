import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import type { Entity, Source } from '@/api/types'
import { entityKey } from '@/lib/entityKey'
import { useUiStore } from '@/stores/ui'
import { useAnalysisUiStore } from '@/stores/analysisUi'
import { useT } from '@/i18n/LanguageContext'

const VISIBLE_LIMIT = 12

interface Mention {
  key: string
  text: string
  type: string
  count: number
}

/**
 * Merge the entities of every source backing one answer.
 *
 * Sources routinely repeat entities — the same name appears in several
 * retrieved chunks — so surfaces are folded case-insensitively per type and
 * their mention counts summed. The first surface seen for a key wins as the
 * display form, matching the order the backend ranked the sources in.
 *
 * @param sources - The answer's deduped sources.
 * @returns Merged mentions, most-mentioned first, ties broken alphabetically.
 */
export function mergeSourceEntities(sources: Source[]): Mention[] {
  const byKey = new Map<string, Mention>()
  for (const source of sources) {
    const entities: Entity[] = source.entities ?? source.ner?.entities ?? []
    for (const entity of entities) {
      const text = (entity.text ?? '').trim()
      if (!text) continue
      const type = entity.type ?? ''
      const foldKey = `${text.toLowerCase()}::${type}`
      const existing = byKey.get(foldKey)
      const count = entity.count ?? 1
      if (existing) existing.count += count
      else byKey.set(foldKey, { key: entityKey(text, type), text, type, count })
    }
  }
  return [...byKey.values()].sort((a, b) => b.count - a.count || a.text.localeCompare(b.text))
}

/**
 * The entities behind an answer, as pills between the answer and its sources.
 *
 * Clicking one opens it in the Analysis tab. The pill's key is built from the
 * raw chunk surface; Analysis resolves it against its own merged aggregate
 * (see `lib/entityKey.ts`), which is where surfaces like `africa` and `Africa`
 * are reconciled.
 */
export function AnswerEntities({ sources }: { sources: Source[] }) {
  const t = useT()
  const navigate = useNavigate()
  const collection = useUiStore((s) => s.selectedCollection)
  const setTab = useAnalysisUiStore((s) => s.setTab)
  const setEntity = useAnalysisUiStore((s) => s.setEntity)
  const [expanded, setExpanded] = useState(false)

  const mentions = useMemo(() => mergeSourceEntities(sources), [sources])
  if (mentions.length === 0) return null

  const shown = expanded ? mentions : mentions.slice(0, VISIBLE_LIMIT)
  const hidden = mentions.length - shown.length

  const open = (mention: Mention) => {
    if (!collection) return
    setTab('ner')
    setEntity(mention.key, collection)
    navigate('/analysis')
  }

  return (
    <div className="mt-3">
      <div className="text-xs uppercase text-muted-foreground">{t('chat.entities')}</div>
      <ul className="mt-1.5 flex flex-wrap gap-1" data-testid="answer-entities">
        {shown.map((mention) => {
          const body = (
            <>
              {mention.type && <span className="text-muted-foreground">{mention.type}</span>}
              <span className="break-all">{mention.text}</span>
              <span className="text-muted-foreground">{mention.count}</span>
            </>
          )
          return (
            <li
              key={mention.key}
              data-testid="answer-entity"
              className="inline-flex items-center gap-1 rounded border border-border bg-muted px-1.5 py-0.5 text-[11px]"
            >
              {/* Without an active collection there is nothing to resolve the
                  entity against, so the pill stays informational. */}
              {collection ? (
                <button
                  type="button"
                  className="inline-flex items-center gap-1 hover:text-blue-300"
                  title={t('chat.entity_open_analysis')}
                  onClick={() => open(mention)}
                >
                  {body}
                </button>
              ) : (
                body
              )}
            </li>
          )
        })}
        {hidden > 0 && (
          <li>
            <button
              type="button"
              className="rounded border border-border px-1.5 py-0.5 text-[11px] text-muted-foreground hover:text-foreground"
              onClick={() => setExpanded(true)}
            >
              {t('chat.entities_more', { count: hidden })}
            </button>
          </li>
        )}
      </ul>
    </div>
  )
}
