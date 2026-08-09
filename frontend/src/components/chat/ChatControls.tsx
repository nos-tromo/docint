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
 * It was a full-width `<select>` carrying two long sentences — a lot of
 * furniture for a setting that is flipped rarely and has exactly two values.
 * The two states use *different icons* rather than one icon pressed and
 * unpressed, because a label-less control whose state is only legible on hover
 * is a control people leave in the wrong state; the accessible name and tooltip
 * then spell the active mode out in full.
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
      className="h-8 w-8 shrink-0 px-0"
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
 * The two settings that govern what the next answer retrieves against: the
 * metadata filters and the retrieval mode.
 *
 * They belong to the **chat**, not to the search panel, and sit beside the
 * Chat heading accordingly. Living in the panel they read as index controls —
 * something that shapes the keyword search — which they are not: they narrow
 * what any retrieval sees, whether or not the panel is even open. Stacked at
 * the panel's foot they were worse than mislabelled, because opening the
 * filters covered the retrieval control directly above them, and the pair
 * behaved like one control with two faces.
 *
 * They hold the header's right edge, outboard of Download — which appears only
 * once a transcript exists, so anything placed to *its* right would slide
 * sideways the first time an answer lands. Every control in that row is 32px
 * tall, the same as the search panel's own Search button across from it.
 */
export function ChatControls() {
  return (
    <div className="flex items-center gap-2">
      <RetrievalToggle />
      {/* Last, so its right edge *is* the header row's right edge: the panel it
          drops is right-aligned under it and therefore lands on the same line
          as the transcript and the composer below. Ordered the other way the
          overlay hung short of that edge and read as misplaced. */}
      <FilterBuilder />
    </div>
  )
}
