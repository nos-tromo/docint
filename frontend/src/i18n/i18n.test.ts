import { describe, expect, it } from 'vitest'
import { en } from './en'
import { de } from './de'
import { format } from './index'

/** The `agent.reason` tokens an agent turn reports. Protocol, so they key the copy in every locale. */
const AGENT_REASONS = [
  'tools_unavailable',
  'round_budget',
  'time_budget',
  'context_budget',
  'idle',
  'research_failed',
  'no_evidence',
  'answer_cut'
] as const

/** One label per trail step kind, singular and plural where a step counts what it found. */
const AGENT_TRAIL_KEYS = [
  'chat.agent_trail_title',
  'chat.agent_trail_show',
  'chat.agent_trail_hide',
  'chat.agent_trail_seed_one',
  'chat.agent_trail_seed_other',
  'chat.agent_trail_search_one',
  'chat.agent_trail_search_other',
  'chat.agent_trail_search_images_one',
  'chat.agent_trail_search_images_other',
  'chat.agent_trail_keyword_one',
  'chat.agent_trail_keyword_other',
  'chat.agent_trail_keyword_unchecked_one',
  'chat.agent_trail_keyword_unchecked_other',
  'chat.agent_trail_keyword_unknown',
  'chat.agent_trail_keyword_field',
  'chat.agent_trail_keyframes_one',
  'chat.agent_trail_keyframes_other',
  'chat.agent_trail_keyframe',
  'chat.agent_trail_read',
  'chat.agent_trail_note',
  'chat.agent_trail_answer'
] as const

function placeholders(template: string): string[] {
  return [...template.matchAll(/\{(\w+)\}/g)].map((match) => match[1]).sort()
}

describe('i18n', () => {
  it('en and de have identical key sets', () => {
    expect(Object.keys(de).sort()).toEqual(Object.keys(en).sort())
  })
  it('uses the same placeholders in en and de for every key', () => {
    const catalog: Record<string, string> = de
    const differing = Object.entries(en).filter(
      ([key, copy]) => placeholders(copy).join() !== placeholders(catalog[key] ?? '').join()
    )
    expect(differing.map(([key]) => key)).toEqual([])
  })
  it.each([
    ['en', en],
    ['de', de]
  ] as const)('names every agent reason and trail step in %s', (_lang, catalog) => {
    const copy: Record<string, string> = catalog
    const keys = [...AGENT_REASONS.map((reason) => `chat.agent_reason.${reason}`), ...AGENT_TRAIL_KEYS]
    expect(keys.filter((key) => !Object.hasOwn(copy, key) || copy[key].trim() === '')).toEqual([])
  })
  it.each([
    ['en', en],
    ['de', de]
  ] as const)('words every limit in %s as how research ended, never as evidence lost', (_lang, catalog) => {
    const copy: Record<string, string> = catalog
    const limits = ['round_budget', 'time_budget', 'context_budget', 'idle'].map(
      (reason) => copy[`chat.agent_reason.${reason}`]
    )
    const claims = limits.filter((text) =>
      /left out|missing|lost|incomplete|ausgelassen|fehlen|verloren|unvollständig/i.test(text)
    )
    expect(claims).toEqual([])
  })
  it('keeps no assessment trail or reason key once they are the agent’s', () => {
    const stale = Object.keys({ ...en, ...de }).filter(
      (key) => key.startsWith('chat.assessment_trail_') || key.startsWith('chat.assessment_reason.')
    )
    expect(stale).toEqual([])
  })
  it('interpolates named placeholders', () => {
    expect(format('{n} Treffer', { n: 3 })).toBe('3 Treffer')
  })
  it('leaves unknown placeholders visible', () => {
    expect(format('{missing}', {})).toBe('{missing}')
  })
})
