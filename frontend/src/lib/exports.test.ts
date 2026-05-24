import { describe, it, expect } from 'vitest'
import {
  TXT_EXPORT_SEPARATOR,
  chatTranscriptToText,
  entityFindingsToText,
  hateSpeechToText,
  summaryToMarkdown
} from './exports'
import type {
  ChatFinalEvent,
  HateSpeechRow,
  NerEntityRow,
  NerSourceRow,
  SummaryResponse
} from '@/api/types'

describe('chatTranscriptToText', () => {
  it('renders role headers, validation, and source blocks', () => {
    const meta: ChatFinalEvent = {
      sources: [
        {
          filename: 'doc.pdf',
          file_hash: 'h1',
          page: 3,
          score: 0.91,
          preview_text: 'first snippet'
        },
        {
          filename: 'doc.pdf',
          file_hash: 'h1',
          page: 7,
          score: 0.84,
          preview_text: 'second snippet'
        }
      ],
      session_id: 's',
      validation_checked: true,
      validation_mismatch: false,
      validation_reason: 'all clear'
    }
    const text = chatTranscriptToText([
      { user: 'Hello?', assistant: 'Hi.', done: true, meta: null },
      { user: 'Cite please', assistant: 'See refs.', done: true, meta }
    ])
    expect(text).toContain('USER: Hello?')
    expect(text).toContain('ASSISTANT: Hi.')
    expect(text).toContain(
      'VALIDATION: checked=true, mismatch=false, reason=all clear'
    )
    expect(text).toContain('SOURCES:')
    expect(text).toContain(TXT_EXPORT_SEPARATOR)
    // Both references survive — regression we fixed earlier.
    expect(text).toContain('[1] doc.pdf')
    expect(text).toContain('- Page: 3')
    expect(text).toContain('[2] doc.pdf')
    expect(text).toContain('- Page: 7')
    expect(text).toContain('first snippet')
    expect(text).toContain('second snippet')
  })

  it('does not duplicate body text in the source metadata block', () => {
    const meta: ChatFinalEvent = {
      sources: [
        {
          filename: 'doc.pdf',
          page: 1,
          score: 0.5,
          preview_text: 'the body preview',
          reference_metadata: {
            author: 'alice',
            text: 'the body preview',
            parent_text: 'parent context',
            anchor_text: 'anchor span'
          }
        }
      ],
      session_id: 's'
    }
    const text = chatTranscriptToText([
      { user: 'q', assistant: 'a', done: true, meta }
    ])
    const occurrences = text.match(/the body preview/g) ?? []
    // Preview prints once; the metadata block excludes text/parent_text/
    // anchor_text so they don't render a second time inside the citation.
    expect(occurrences).toHaveLength(1)
    expect(text).not.toContain('Parent Text:')
    expect(text).not.toContain('Anchor Text:')
    expect(text).toContain('Author: alice')
  })
})

describe('summaryToMarkdown', () => {
  it('produces a markdown doc with sources', () => {
    const meta: SummaryResponse = {
      summary: '## Overview\n\nstuff happened',
      sources: [{ filename: 'a.pdf', page: 1, score: 0.7, preview_text: 'opening text' }]
    }
    const md = summaryToMarkdown(meta, 'fallback streamed text')
    expect(md.startsWith('# Summary')).toBe(true)
    expect(md).toContain('## Overview')
    expect(md).toContain('## Sources')
    expect(md).toContain(TXT_EXPORT_SEPARATOR)
    expect(md).toContain('[1] a.pdf')
    expect(md).toContain('- Page: 1')
  })

  it('falls back to streamed text when meta is null', () => {
    const md = summaryToMarkdown(null, 'streamed body')
    expect(md).toContain('streamed body')
  })
})

describe('entity findings TXT export', () => {
  const entity: NerEntityRow = {
    text: 'Alice',
    type: 'PER',
    mentions: 2,
    key: 'alice::PER'
  } as unknown as NerEntityRow

  it('uses the unified analysis block with separator, metadata, and sectioned text', () => {
    const chunks: NerSourceRow[] = [
      {
        chunk_id: 'c1',
        filename: 'doc.pdf',
        page: 1,
        score: 0.9,
        chunk_text: 'Alice spoke loudly.',
        reference_metadata: {
          network: 'twitter',
          author: 'bob',
          timestamp: '2026-01-01',
          parent_text: 'Earlier in the thread.',
          anchor_text: 'Alice'
        }
      }
    ]
    const txt = entityFindingsToText(entity, chunks)
    expect(txt).toContain('Entity Findings: Alice [PER]')
    expect(txt).toContain(TXT_EXPORT_SEPARATOR)
    expect(txt).toContain('[1] doc.pdf')
    expect(txt).toContain('- Page: 1')
    expect(txt).toContain('- Chunk ID: c1')
    // Metadata lines exclude the body-text trio.
    expect(txt).toContain('- Network: twitter')
    expect(txt).toContain('- Author: bob')
    expect(txt).not.toMatch(/- Anchor Text:/)
    expect(txt).not.toMatch(/- Parent Text:/)
    expect(txt).not.toMatch(/- Text:/)
    // Sectioned text blocks appear in order: Anchor → Parent → Text.
    // Match the standalone "Text" heading using a leading newline so it
    // doesn't collide with the "Anchor Text"/"Parent Text" headings,
    // which both contain "Text" as a substring.
    const anchorIdx = txt.indexOf('Anchor Text\n-----------')
    const parentIdx = txt.indexOf('Parent Text\n-----------')
    const textIdx = txt.indexOf('\nText\n----\n')
    expect(anchorIdx).toBeGreaterThan(-1)
    expect(parentIdx).toBeGreaterThan(anchorIdx)
    expect(textIdx).toBeGreaterThan(parentIdx)
    expect(txt).toContain('Earlier in the thread.')
    expect(txt).toContain('Alice spoke loudly.')
  })

  it('falls back to chunk_text for the Text section when reference_metadata.text is absent', () => {
    const chunks: NerSourceRow[] = [
      { chunk_id: 'c1', filename: 'plain.txt', chunk_text: 'plain body text' }
    ]
    const txt = entityFindingsToText(entity, chunks)
    expect(txt).toContain('\nText\n----\n')
    expect(txt).toContain('plain body text')
    // Body text appears exactly once (the duplicate-text regression).
    const occurrences = txt.match(/plain body text/g) ?? []
    expect(occurrences).toHaveLength(1)
  })

  it('falls back to row when page is absent', () => {
    const chunks: NerSourceRow[] = [
      { chunk_id: 'c1', filename: 'table.csv', row: 12, chunk_text: 'cell' }
    ]
    const txt = entityFindingsToText(entity, chunks)
    expect(txt).toContain('- Row: 12')
    expect(txt).not.toContain('- Page:')
  })
})

describe('hate-speech TXT export', () => {
  it('renders findings with extras, metadata, and sectioned text', () => {
    const rows: HateSpeechRow[] = [
      {
        chunk_id: 'c1',
        filename: 'bad.txt',
        page: 2,
        chunk_text: 'offensive line',
        category: 'slur',
        confidence: 'high',
        reason: 'matches lexicon',
        reference_metadata: { author: 'mallory', parent_text: 'context above' }
      }
    ]
    const txt = hateSpeechToText(rows)
    expect(txt).toContain('Hate-Speech Findings')
    expect(txt).toContain(TXT_EXPORT_SEPARATOR)
    expect(txt).toContain('[1] bad.txt')
    expect(txt).toContain('- Category: slur')
    expect(txt).toContain('- Confidence: high')
    expect(txt).toContain('- Reason: matches lexicon')
    expect(txt).toContain('- Author: mallory')
    expect(txt).toContain('Parent Text\n-----------')
    expect(txt).toContain('context above')
    expect(txt).toContain('\nText\n----\n')
    expect(txt).toContain('offensive line')
  })
})

