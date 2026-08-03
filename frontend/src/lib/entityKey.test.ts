import { describe, it, expect } from 'vitest'
import { entityKey, resolveEntityKey } from './entityKey'

const AGGREGATE = [
  { text: 'Acme Corp', type: 'ORG' },
  { text: 'Africa', type: 'LOC' },
  { text: 'Alice Weber', type: 'PER' }
]

describe('entityKey', () => {
  it('builds the text::type shorthand the analysis screen indexes by', () => {
    expect(entityKey('Acme Corp', 'ORG')).toBe('Acme Corp::ORG')
  })

  it('tolerates missing parts rather than producing undefined', () => {
    expect(entityKey(null, 'ORG')).toBe('::ORG')
    expect(entityKey('Acme Corp', undefined)).toBe('Acme Corp::')
  })
})

describe('resolveEntityKey', () => {
  it('returns the key unchanged when the aggregate holds it verbatim', () => {
    expect(resolveEntityKey('Africa::LOC', AGGREGATE)).toBe('Africa::LOC')
  })

  it('matches a differently-cased surface', () => {
    // A chunk can carry "africa" while the orthographically merged aggregate
    // lists "Africa" — the same entity under a different surface.
    expect(resolveEntityKey('africa::LOC', AGGREGATE)).toBe('Africa::LOC')
  })

  it('matches a surface that differs only in punctuation or spacing', () => {
    expect(resolveEntityKey('acme-corp::ORG', AGGREGATE)).toBe('Acme Corp::ORG')
  })

  it('does not match across entity types', () => {
    expect(resolveEntityKey('Africa::ORG', AGGREGATE)).toBeNull()
  })

  it('returns null for an entity the aggregate does not hold', () => {
    expect(resolveEntityKey('Bob Fischer::PER', AGGREGATE)).toBeNull()
  })

  it('returns null when there is no key or no aggregate', () => {
    expect(resolveEntityKey(null, AGGREGATE)).toBeNull()
    expect(resolveEntityKey('Africa::LOC', [])).toBeNull()
  })
})
