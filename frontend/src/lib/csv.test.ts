import { describe, it, expect } from 'vitest'
import { toCsv } from './csv'

describe('toCsv', () => {
  it('emits header and rows', () => {
    const csv = toCsv([{ a: 1, b: 'x' }, { a: 2, b: 'y, z' }], ['a', 'b'])
    expect(csv).toBe('a,b\n1,x\n2,"y, z"')
  })

  it('escapes quotes', () => {
    const csv = toCsv([{ a: 'he said "hi"' }], ['a'])
    expect(csv).toBe('a\n"he said ""hi"""')
  })
})
