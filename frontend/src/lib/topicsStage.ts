import type { Strings } from '@/i18n'

type Translate = (key: keyof Strings, vars?: Record<string, string | number>) => string

/**
 * Copy for each stage a topics run reports, whether as a topics job (the Topics
 * tab) or as the stage of an ingest (the ingest card). `finished` closes an
 * ingest's stage only. A stage the SPA does not know reads as `stage_other`.
 */
export const TOPICS_STAGE_KEYS = new Map<string, keyof Strings>([
  ['starting', 'topics.stage_starting'],
  ['waiting', 'topics.stage_waiting'],
  ['scanning', 'topics.stage_scanning'],
  ['sampling', 'topics.stage_sampling'],
  ['fitting', 'topics.stage_fitting'],
  ['labelling', 'topics.stage_labelling'],
  ['assigning', 'topics.stage_assigning'],
  ['aligning', 'topics.stage_aligning'],
  ['finished', 'topics.stage_finished']
])

/**
 * A topics stage in catalog words, never the frame's own message.
 *
 * @param t - The translator.
 * @param stage - The stage the frame names.
 * @param done - Units done.
 * @param total - Units in all, 0 when the stage cannot say.
 * @returns The stage's copy with its counts filled in.
 */
export function topicsStageText(t: Translate, stage: string, done: number, total: number): string {
  return t(TOPICS_STAGE_KEYS.get(stage) ?? 'topics.stage_other', { done, total })
}
