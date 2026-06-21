import { describe, it, expect } from 'vitest'
import { chatAnswerSnapshot, entityFindingSnapshot, hateSpeechSnapshot, summarySnapshot } from './reportSnapshots'
import type { HateSpeechRow, NerSourceRow } from '@/api/types'

describe('reportSnapshots', () => {
  it('chatAnswerSnapshot builds a type-prefixed dedupe key and freezes sources', () => {
    const input = chatAnswerSnapshot({
      sessionId: 's1',
      turnIdx: 2,
      userText: 'q',
      modelResponse: 'a',
      sources: [{ filename: 'f.pdf', page: 3, score: 0.5, text: 'body' }]
    })
    expect(input.artifact_type).toBe('chat_answer')
    expect(input.dedupe_key).toBe('chat:s1:2')
    expect(input.snapshot.turn_idx).toBe(2)
    const sources = input.snapshot.sources as Record<string, unknown>[]
    expect(sources[0]).toMatchObject({ filename: 'f.pdf', page: 3, text: 'body' })
  })

  it('entityFindingSnapshot dedupes by chunk_id and keeps every entity', () => {
    const row: NerSourceRow = {
      chunk_id: 'c1',
      chunk_text: 'Acme & Bob',
      filename: 'a.pdf',
      page: 1,
      entities: [
        { text: 'Acme', type: 'ORG' },
        { text: 'Bob', type: 'PERSON' }
      ]
    }
    const input = entityFindingSnapshot(row, 'Acme [ORG]')
    expect(input.artifact_type).toBe('entity_finding')
    expect(input.dedupe_key).toBe('entity:c1')
    expect(input.snapshot.entity_label).toBe('Acme [ORG]')
    expect((input.snapshot.entities as Record<string, unknown>[]).length).toBe(2)
  })

  it('hateSpeechSnapshot dedupes by chunk_id with a hate prefix', () => {
    const row: HateSpeechRow = { chunk_id: 'c9', category: 'slur', confidence: 'high', reason: 'r', chunk_text: 't' }
    const input = hateSpeechSnapshot(row)
    expect(input.artifact_type).toBe('hate_speech_finding')
    expect(input.dedupe_key).toBe('hate:c9')
    expect(input.snapshot.category).toBe('slur')
  })

  it('summarySnapshot dedupes by collection', () => {
    const input = summarySnapshot({ collection: 'docs', text: 'overview' })
    expect(input.artifact_type).toBe('summary')
    expect(input.dedupe_key).toBe('summary:docs')
    expect(input.snapshot.text).toBe('overview')
  })

  it('the same chunk gets distinct dedupe keys across artifact types', () => {
    const entity = entityFindingSnapshot({ chunk_id: 'x' }, 'X [ORG]')
    const hate = hateSpeechSnapshot({ chunk_id: 'x' })
    expect(entity.dedupe_key).toBe('entity:x')
    expect(hate.dedupe_key).toBe('hate:x')
    expect(entity.dedupe_key).not.toBe(hate.dedupe_key)
  })
})
