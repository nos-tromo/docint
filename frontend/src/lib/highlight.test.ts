import { describe, it, expect } from 'vitest'
import { highlightSegments, keywordSegments } from './highlight'

describe('highlightSegments', () => {
  it('marks every case-insensitive occurrence of a term', () => {
    expect(highlightSegments('Alpha and alpha', ['alpha'])).toEqual([
      { text: 'Alpha', highlight: true },
      { text: ' and ', highlight: false },
      { text: 'alpha', highlight: true }
    ])
  })

  it('prefers the longer term when several overlap', () => {
    const marked = highlightSegments('European Union', ['Union', 'European Union'])
      .filter((s) => s.highlight)
      .map((s) => s.text)

    expect(marked).toEqual(['European Union'])
  })
})

describe('keywordSegments', () => {
  it('marks a word the keyword prefixes', () => {
    // The index matches word prefixes, so the head of a compound finds the
    // compound; the whole compound is what matched, so all of it is marked.
    const marked = keywordSegments('Der Parteitag begann.', ['Partei'])
      .filter((s) => s.highlight)
      .map((s) => s.text)

    expect(marked).toEqual(['Parteitag'])
  })

  it('never marks mid-word, matching what the backend can actually find', () => {
    // `tag` does not find `Parteitag` server-side. Painting it here would
    // advertise a search capability that does not exist.
    const marked = keywordSegments('Der Parteitag begann.', ['tag']).filter((s) => s.highlight)

    expect(marked).toEqual([])
  })

  it('folds case like the lowercase index does', () => {
    const marked = keywordSegments('BUNDESTAG und bundesrat', ['bundes'])
      .filter((s) => s.highlight)
      .map((s) => s.text)

    expect(marked).toEqual(['BUNDESTAG', 'bundesrat'])
  })

  it('keeps the unmatched text intact around the marks', () => {
    expect(keywordSegments('a beta c', ['beta'])).toEqual([
      { text: 'a ', highlight: false },
      { text: 'beta', highlight: true },
      { text: ' c', highlight: false }
    ])
  })

  it('returns the text unmarked when there are no keywords', () => {
    expect(keywordSegments('plain text', [])).toEqual([{ text: 'plain text', highlight: false }])
    expect(keywordSegments('', ['x'])).toEqual([])
  })
})
