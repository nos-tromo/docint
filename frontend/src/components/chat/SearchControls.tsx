import { Button } from '@infra/ui'
import { useChatFiltersStore } from '@/stores/chatFilters'
import { FilterBuilder } from '@/components/chat/FilterBuilder'
import { ChatContextIcon, SingleMessageIcon } from '@/components/common/icons'
import { useT } from '@/i18n/LanguageContext'

/**
 * How much of the conversation the next answer retrieves against, as a single
 * icon button: stateful (the whole chat) or stateless (only the message just
 * sent).
 *
 * It was a full-width `<select>` carrying two long sentences at the foot of the
 * column — a lot of furniture for a setting that is flipped rarely and has
 * exactly two values. The two states use *different icons* rather than one icon
 * pressed and unpressed, because a label-less control whose state is only
 * legible on hover is a control people leave in the wrong state; the accessible
 * name and tooltip then spell the active mode out in full.
 */
export function RetrievalToggle() {
  const t = useT()
  const mode = useChatFiltersStore((s) => s.retrievalMode)
  const setRetrievalMode = useChatFiltersStore((s) => s.setRetrievalMode)
  const stateful = mode === 'session'
  const name = t('chat.retrieval_state', {
    mode: stateful ? t('chat.retrieval_session') : t('chat.retrieval_stateless')
  })

  return (
    <Button
      type="button"
      variant="ghost"
      size="sm"
      aria-pressed={stateful}
      aria-label={name}
      title={name}
      onClick={() => setRetrievalMode(stateful ? 'stateless' : 'session')}
      className="h-7 w-7 shrink-0 px-0"
    >
      {stateful ? (
        <ChatContextIcon className="h-4 w-4" />
      ) : (
        <SingleMessageIcon className="h-4 w-4" />
      )}
    </Button>
  )
}

/**
 * The search panel's controls band: the two settings that change what a search
 * and the next answer see, on one quiet row under the query field.
 *
 * Both used to live at the *foot* of the column, stacked, where opening the
 * filters covered the retrieval control — the reason the pair read as one
 * confusing toggle. Up here they sit side by side and neither hides the other.
 *
 * This element owns the `relative` box that `FilterBuilder`'s overlay anchors
 * to, so the panel spans the full column width while its trigger stays
 * trigger-sized. Moving that positioning back into `FilterBuilder` would pin
 * the panel to the width of its own button.
 */
export function SearchControls() {
  return (
    <div className="relative flex items-center justify-between gap-2">
      <FilterBuilder />
      <RetrievalToggle />
    </div>
  )
}
