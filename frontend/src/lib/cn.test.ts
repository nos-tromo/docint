import { describe, it, expect } from 'vitest'
import { cn } from './cn'

describe('cn', () => {
  it('merges class names', () => {
    expect(cn('a', 'b')).toBe('a b')
  })

  it('deduplicates conflicting tailwind classes', () => {
    expect(cn('p-2', 'p-4')).toBe('p-4')
  })

  it('drops falsy values', () => {
    // eslint-disable-next-line no-constant-binary-expression -- intentional: cn must drop literal falsy class args
    expect(cn('a', false && 'b', null, undefined, 'c')).toBe('a c')
  })
})
