import type { ReactNode } from 'react'
import {
  BrainActiveIcon,
  BrainIcon,
  Button,
  CycleButton,
  DocumentsIcon,
  ImageIcon,
  LayersIcon
} from '@infra/ui'
import { RETRIEVAL_TARGETS, type RetrievalTarget } from '@/api/types'
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
 * Whether the model thinks before it answers, as a single icon button.
 *
 * Reasoning buys answer quality with latency and tokens, so it is off until
 * asked for and flipped per chat rather than per deployment — the server's
 * env default only governs API clients that omit the field. Like the
 * retrieval toggle it carries no label, so the state lives in the accessible
 * name and the drawing itself changes: the brain lights up rather than merely
 * taking a pressed tint (the `BrainIcon`/`BrainActiveIcon` state pair from
 * `@infra/ui`).
 */
export function ReasoningToggle() {
  const t = useT()
  const reasoning = useChatFiltersStore((s) => s.reasoning)
  const setReasoning = useChatFiltersStore((s) => s.setReasoning)
  const name = t('chat.reasoning_state', {
    mode: reasoning ? t('chat.reasoning_on') : t('chat.reasoning_off')
  })

  return (
    <Button
      type="button"
      variant="ghost"
      size="sm"
      aria-pressed={reasoning}
      aria-label={name}
      title={name}
      onClick={() => setReasoning(!reasoning)}
      className="h-8 w-8 shrink-0 px-0"
    >
      {reasoning ? <BrainActiveIcon className="h-4 w-4" /> : <BrainIcon className="h-4 w-4" />}
    </Button>
  )
}

/**
 * The settings that govern the next answer: the reasoning toggle, the
 * retrieval mode and the metadata filters. The last two decide what the answer
 * retrieves against; the first, how hard the model works on what it found.
 *
 * The two retrieval settings: the
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
/**
 * The drawing for each target: the whole corpus, the written documents, the
 * stored imagery. One per value, because the icon is the control's only
 * visible state — a single icon tinted would leave the setting legible on
 * hover alone, which is how a setting gets left wrong.
 */
const TARGET_ICON: Record<RetrievalTarget, ReactNode> = {
  all: <LayersIcon className="h-4 w-4" />,
  documents: <DocumentsIcon className="h-4 w-4" />,
  visual: <ImageIcon className="h-4 w-4" />
}

/**
 * Which evidence the next answer may come from: everything, documents only,
 * or stored imagery only.
 *
 * Three values rather than two, so it is not a pressed/unpressed toggle — but
 * it is set rarely and sits in a row of 32px icon controls, where a labelled
 * dropdown reads as a different *kind* of control and spends far more width
 * than the setting is worth. So it steps through the three
 * (`CycleButton` from `@infra/ui`), each with its own drawing, and the
 * accessible name and tooltip carry the current value in full — "Answer from:
 * Images" — exactly as its two neighbours name their states.
 *
 * Under the visual target the filter panel grows its own presets (clip, time
 * range, kind of imagery).
 */
export function RetrievalTargetPicker() {
  const t = useT()
  const target = useChatFiltersStore((s) => s.retrievalTarget)
  const setRetrievalTarget = useChatFiltersStore((s) => s.setRetrievalTarget)

  return (
    <CycleButton
      name={t('chat.retrieval_target')}
      options={RETRIEVAL_TARGETS.map((value) => ({
        value,
        icon: TARGET_ICON[value],
        label: t(`chat.retrieval_target.${value}`)
      }))}
      value={target}
      onChange={setRetrievalTarget}
    />
  )
}

export function ChatControls() {
  return (
    <div className="flex items-center gap-2">
      <RetrievalTargetPicker />
      <ReasoningToggle />
      <RetrievalToggle />
      {/* Last, so its right edge *is* the header row's right edge: the panel it
          drops is right-aligned under it and therefore lands on the same line
          as the transcript and the composer below. Ordered the other way the
          overlay hung short of that edge and read as misplaced. */}
      <FilterBuilder />
    </div>
  )
}
