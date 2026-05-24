import { describe, it, expect } from 'vitest'
import {
  TXT_EXPORT_SEPARATOR,
  chatTranscriptToText,
  summaryToMarkdown
} from './exports'
import type { ChatFinalEvent, SummaryResponse } from '@/api/types'

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


