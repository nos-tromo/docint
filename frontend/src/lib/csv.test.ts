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

  it.each([
    ['=', '=SUM(A1:A9)', "'=SUM(A1:A9)"],
    ['+', '+49 30 1234', "'+49 30 1234"],
    ['-', '-2+3', "'-2+3"],
    ['@', '@cmd', "'@cmd"]
  ])('neutralises a text cell starting with %s so a spreadsheet reads it as text', (_trigger, value, cell) => {
    expect(toCsv([{ a: value }], ['a'])).toBe(`a\n${cell}`)
  })

  it('neutralises a text cell starting with a tab', () => {
    expect(toCsv([{ a: '\t=1+1' }], ['a'])).toBe("a\n'\t=1+1")
  })

  it('neutralises a text cell starting with a carriage return, then quotes it', () => {
    expect(toCsv([{ a: '\r=1+1' }], ['a'])).toBe(`a\n"'\r=1+1"`)
  })

  it('leaves numbers alone, negative ones included', () => {
    expect(toCsv([{ a: -5, b: 0.25 }], ['a', 'b'])).toBe('a,b\n-5,0.25')
  })

  it('neutralises before it quotes, so a formula with a comma stays one guarded cell', () => {
    expect(toCsv([{ a: '=a,b' }], ['a'])).toBe(`a\n"'=a,b"`)
  })

  it('quotes a carriage return anywhere in a cell', () => {
    expect(toCsv([{ a: 'line\rbreak' }], ['a'])).toBe('a\n"line\rbreak"')
  })

  it('leaves a cell with a trigger character inside it alone', () => {
    expect(toCsv([{ a: 'a=b' }], ['a'])).toBe('a\na=b')
  })
})
