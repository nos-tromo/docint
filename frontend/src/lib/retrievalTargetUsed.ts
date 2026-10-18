import type { ChatFinalEvent, RetrievalTarget } from '@/api/types'

/** A target worth naming on a turn: the default (`all`) says nothing. */
export type NamedTarget = Exclude<RetrievalTarget, 'all'>

/**
 * The narrowed evidence a turn answered from, when the transcript should say so.
 *
 * Only `documents` and `visual` are named. A turn with no recorded target (one
 * from before it was stored, or a scoped one) is never read as `all` or as a
 * narrowing: it simply says nothing. A scoped turn also says nothing when the
 * final frame still reports the requested target, because the pinned selection
 * outranks it and "documents only" would describe a restriction that never
 * applied. The chip and the transcript export share this one rule.
 *
 * @param meta - The turn's final envelope, or its reloaded equivalent.
 * @returns The target to name, or `null` when nothing needs saying.
 */
export function retrievalTargetUsed(meta: ChatFinalEvent | null | undefined): NamedTarget | null {
  if (!meta || meta.retrieval_mode === 'scoped') return null
  const target = meta.retrieval_target
  return target === 'documents' || target === 'visual' ? target : null
}
