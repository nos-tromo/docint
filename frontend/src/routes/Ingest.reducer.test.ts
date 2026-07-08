import { describe, expect, it } from 'vitest'
import { reducer, type State } from './Ingest'

function baseState(files: File[] = []): State {
  return { collection: '', files, events: [], fileSizes: {}, busy: false }
}

function file(name: string, size = 1024): File {
  return new File(['x'.repeat(size)], name)
}

describe('Ingest reducer', () => {
  it('add_files dedups by name+size', () => {
    let s = baseState()
    s = reducer(s, { type: 'add_files', v: [file('a.txt', 1024), file('b.txt', 2048)] })
    s = reducer(s, { type: 'add_files', v: [file('a.txt', 1024), file('c.txt', 512)] })
    expect(s.files.map((f) => f.name)).toEqual(['a.txt', 'b.txt', 'c.txt'])
  })

  it('remove_file removes the file at the given index', () => {
    let s = baseState([file('a.txt'), file('b.txt'), file('c.txt')])
    s = reducer(s, { type: 'remove_file', i: 1 })
    expect(s.files.map((f) => f.name)).toEqual(['a.txt', 'c.txt'])
  })

  it('reset_files clears all files', () => {
    let s = baseState([file('a.txt')])
    s = reducer(s, { type: 'reset_files' })
    expect(s.files).toEqual([])
  })
})
