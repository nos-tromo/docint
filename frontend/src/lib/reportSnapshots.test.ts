import { describe, it, expect } from 'vitest'
import {
  chatAnswerSnapshot,
  chunkTextOf,
  entityFindingSnapshot,
  hateSpeechSnapshot,
  summarySnapshot
} from './reportSnapshots'
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

  it('chatAnswerSnapshot carries image identity so the server can freeze a thumbnail', () => {
    const input = chatAnswerSnapshot({
      sessionId: 's1',
      turnIdx: 0,
      userText: 'q',
      modelResponse: 'a',
      sources: [
        { filename: 'fig.png', text: 'caption', image_id: 'img-1', image_collection: 'docs_images', file_hash: 'h1' },
        { filename: 'a.pdf', page: 2, text: 'prose' }
      ]
    })
    const sources = input.snapshot.sources as Record<string, unknown>[]
    expect(sources[0]).toMatchObject({ image_id: 'img-1', image_collection: 'docs_images', file_hash: 'h1' })
    expect(sources[1]).not.toHaveProperty('image_id')
    expect(sources[1]).not.toHaveProperty('image_collection')
  })

  it('chatAnswerSnapshot carries the citation number, and never invents one', () => {
    const input = chatAnswerSnapshot({
      sessionId: 's1',
      turnIdx: 0,
      userText: 'q',
      modelResponse: 'a',
      sources: [
        { filename: 'fig.png', text: 'caption', citation_index: 2 },
        { filename: 'late.png', text: 'never reached the prompt' }
      ]
    })
    const sources = input.snapshot.sources as Record<string, unknown>[]
    expect(sources[0].citation_index).toBe(2)
    expect(sources[1]).not.toHaveProperty('citation_index')
  })

  it('chunkTextOf prefers chunk_text, falls back to text, and trims', () => {
    // One derivation for the row's display text, the string the Translate
    // control posts, and the translations-store key — a batch add looks a
    // translation up by exactly what the row stored it under.
    expect(chunkTextOf({ chunk_text: '  flagged line  ', text: 'other' })).toBe('flagged line')
    expect(chunkTextOf({ text: '  fallback line ' })).toBe('fallback line')
    expect(chunkTextOf({ chunk_text: null, text: null })).toBe('')
    expect(chunkTextOf({})).toBe('')
  })

  it('finding snapshots carry image identity only when the row has one', () => {
    const withImage = entityFindingSnapshot({ chunk_id: 'c1', image_id: 'img-9' }, 'X [ORG]')
    const withoutImage = hateSpeechSnapshot({ chunk_id: 'c2' })
    expect(withImage.snapshot.image_id).toBe('img-9')
    expect(withoutImage.snapshot).not.toHaveProperty('image_id')
    expect(hateSpeechSnapshot({ chunk_id: 'c3', image_id: 'kf-1' }).snapshot.image_id).toBe('kf-1')
  })
})
