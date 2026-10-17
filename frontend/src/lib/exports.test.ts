import { describe, it, expect } from 'vitest'
import {
  TXT_EXPORT_SEPARATOR,
  chatTranscriptToText,
  summaryToMarkdown
} from './exports'
import type { ChatFinalEvent, SummaryResponse } from '@/api/types'
import { catalogs, format, type Strings } from '@/i18n'

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

  it('renders the chunk id, preferring the payload chunk id over the node id', () => {
    const meta: ChatFinalEvent = {
      sources: [
        { filename: 'doc.pdf', page: 3, id: 'node-1', chunk_id: 'doc-p3-b2-0' },
        { filename: 'doc.pdf', page: 7, id: 'node-2' },
        { filename: 'doc.pdf', page: 9 }
      ],
      session_id: 's'
    }
    const text = chatTranscriptToText([
      { user: 'Cite please', assistant: 'See refs.', done: true, meta }
    ])
    expect(text).toContain('- Chunk ID: doc-p3-b2-0')
    expect(text).toContain('- Chunk ID: node-2')
    expect(text).toContain('- Chunk ID: n/a')
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

  it('the transcript names a non-default target', () => {
    // The download is what leaves the app, so a text-only answer has to say so
    // there as well: a reader of the file cannot see the picker that set it.
    const turn = (n: number, extra: Record<string, unknown>) => ({
      user: `q${n}`,
      assistant: `a${n}`,
      done: true,
      meta: { session_id: 's', sources: [], ...extra } as unknown as ChatFinalEvent
    })

    const text = chatTranscriptToText([
      turn(1, { retrieval_target: 'documents' }),
      turn(2, { retrieval_target: 'visual' }),
      turn(3, { retrieval_target: 'all' }),
      turn(4, {}),
      turn(5, { retrieval_mode: 'scoped', retrieval_target: 'documents' }),
      { user: 'q6', assistant: 'a6', done: true, meta: null }
    ])

    // One line, directly after the answer it qualifies.
    expect(text).toContain('ASSISTANT: a1\nAnswered from documents only\n\n')
    expect(text).toContain('ASSISTANT: a2\nAnswered from images only\n\n')
    // The default, an unrecorded target and a scoped turn say nothing.
    expect(text.match(/Answered from/g)).toHaveLength(2)
    expect(text).toContain('ASSISTANT: a3\n\n')
    expect(text).toContain('ASSISTANT: a4\n\n')
    expect(text).toContain('ASSISTANT: a5\n\n')
  })

  it('writes the target in the language of the export', () => {
    const de = (key: keyof Strings, vars?: Record<string, string | number>) =>
      format(catalogs.de[key], vars)
    const turn = (target: string) => ({
      user: 'q',
      assistant: 'a',
      done: true,
      meta: { session_id: 's', sources: [], retrieval_target: target } as unknown as ChatFinalEvent
    })

    const text = chatTranscriptToText([turn('documents'), turn('visual')], de)

    expect(text).toContain('ASSISTENT: a\nNur aus Dokumenten beantwortet\n')
    expect(text).toContain('ASSISTENT: a\nNur aus Bildern beantwortet\n')
  })

  it('renders posting reference fields in the source metadata block', () => {
    const meta: ChatFinalEvent = {
      sources: [
        {
          filename: 'clip.mp4',
          score: 0.5,
          preview_text: 'spoken words',
          reference_metadata: {
            network: 'nextext',
            type: 'transcript_segment',
            posting_network: 'Facebook',
            posting_author: 'Jane Poster',
            posting_url: 'https://fb.example/p1',
            posting_text: 'Original post body'
          }
        }
      ],
      session_id: 's'
    }
    const text = chatTranscriptToText([{ user: 'q', assistant: 'a', done: true, meta }])
    // The artifact identity stays, the posting reference rides along.
    expect(text).toContain('- Network: nextext')
    expect(text).toContain('- Posting Network: Facebook')
    expect(text).toContain('- Posting Author: Jane Poster')
    expect(text).toContain('- Posting URL: https://fb.example/p1')
    expect(text).toContain('- Posting Text: Original post body')
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


