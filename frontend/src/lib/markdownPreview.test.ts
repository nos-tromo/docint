import { describe, it, expect } from 'vitest'
import { markdownPreview } from './markdownPreview'

describe('markdownPreview', () => {
  it('drops heading, list and emphasis syntax but keeps the lines', () => {
    const md = ['### Board members', '', '- **Alice Weber** chairs the board [1]', '2. Bob *Stein* keeps `minutes`'].join(
      '\n'
    )

    expect(markdownPreview(md)).toBe('Board members\nAlice Weber chairs the board [1]\nBob Stein keeps minutes')
  })

  it('reads a table as its rows, without the rule', () => {
    const md = ['| Name | Role |', '| :--- | ---: |', '| Alice Weber | Chair |'].join('\n')

    expect(markdownPreview(md)).toBe('Name · Role\nAlice Weber · Chair')
  })

  it('keeps citation markers and link text, not link targets', () => {
    expect(markdownPreview('See [the minutes](https://example.org/m) [2][3]')).toBe('See the minutes [2][3]')
  })

  it('leaves intraword underscores and arithmetic alone', () => {
    expect(markdownPreview('file_name_v2 has 2 * 3 rows')).toBe('file_name_v2 has 2 * 3 rows')
  })

  it('drops code fences and thematic breaks', () => {
    expect(markdownPreview(['```', 'code', '```', '---', 'after'].join('\n'))).toBe('code\nafter')
  })
})
