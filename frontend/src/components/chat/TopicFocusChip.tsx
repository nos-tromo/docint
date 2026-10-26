import { RemoveButton } from '@infra/ui'
import { useActiveCollection, type CollectionScope } from '@/hooks/useActiveCollection'
import { useTopicsOverview, useTopicsRefreshOnJobEnd } from '@/hooks/useTopics'
import { useChatFiltersStore } from '@/stores/chatFilters'
import { scopeChunkIds, scopeFor, searchKeyFor, useSearchUiStore } from '@/stores/searchUi'
import { focusInScope, topicFocusRule, useTopicFocusStore, type TopicFocus } from '@/stores/topicFocus'
import { useUiStore } from '@/stores/ui'
import { cn } from '@/lib/cn'
import { useT } from '@/i18n/LanguageContext'

/**
 * The topic "Ask in chat" narrowed the chat to, beside the chat's other
 * retrieval settings: which topic, whether the next turn applies it, and a
 * control to drop it.
 *
 * It says what the next turn sends. A topic from an earlier build is not
 * applied, because its value matches no chunk of the run that replaced it; a
 * visual turn gets no topic rule either, because imagery carries no topic.
 * Either way the rule is left out rather than allowed to empty the answer, and
 * this is where that is said. With chunks selected, an answer comes from them
 * alone whatever its filters say, so only the searches keep to the topic.
 *
 * A focus picked in another collection, or under another owner, is kept out of
 * sight until that collection is active again, as an entity selection is in Analysis.
 */
export function TopicFocusChip() {
  const scope = useActiveCollection()
  const focus = useTopicFocusStore((s) => s.focus)
  if (!focus || !focusInScope(focus, scope)) return null
  return <FocusChip focus={focus} scope={scope} />
}

function FocusChip({ focus, scope }: { focus: TopicFocus; scope: CollectionScope }) {
  const t = useT()
  const clear = useTopicFocusStore((s) => s.clear)
  const target = useChatFiltersStore((s) => s.retrievalTarget)
  const sessionId = useUiStore((s) => s.currentSessionId)
  const scoped = useSearchUiStore((s) => scopeChunkIds(scopeFor(s, searchKeyFor(sessionId))).length > 0)
  const { data } = useTopicsOverview()
  // A rebuild that ends while the chat is open replaces the run; its end refetches the run read here.
  useTopicsRefreshOnJobEnd()
  // Until the active run is read, nothing can be said of the topic.
  if (data === undefined) return null

  const stale = topicFocusRule(focus, scope, data?.run_id ?? null) === null
  const applied = !stale && !scoped && target !== 'visual'
  const hint = stale
    ? t('chat.topic_focus_stale')
    : scoped
      ? t('chat.topic_focus_scoped')
      : target === 'visual'
        ? t('chat.topic_focus_not_visual')
        : t('chat.topic_focus_applied')
  const name = focus.label ? `T${focus.topicId} · ${focus.label}` : `T${focus.topicId}`

  // The topic over its hint, so the chip is as wide as the longer of the two, not their sum:
  // side by side, a German hint leaves the topic no room beside the controls at 1024px.
  return (
    <div
      role="group"
      aria-label={t('chat.topic_focus')}
      data-applied={applied || undefined}
      className={cn(
        'flex h-8 min-w-0 max-w-[20rem] items-center gap-1 rounded-md border pl-2.5 pr-1',
        applied ? 'border-transparent bg-primary/15' : 'border-dashed border-border'
      )}
    >
      <div className="min-w-0 flex-1">
        <div
          className={cn(
            'truncate text-xs leading-[14px] font-medium',
            applied ? 'text-primary' : 'text-muted-foreground'
          )}
          title={name}
        >
          {name}
        </div>
        {/* Quiet while applied; while not, the reason is what matters. */}
        <div
          className={cn('truncate text-[11px] leading-[14px]', applied ? 'text-muted-foreground' : 'text-foreground')}
          title={hint}
        >
          {hint}
        </div>
      </div>
      <RemoveButton label={t('chat.topic_focus_remove')} onClick={clear} className="h-6" />
    </div>
  )
}
