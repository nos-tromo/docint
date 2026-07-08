import { describe, it, expect } from 'vitest'
import { planUploadBatches } from './uploadBatches'

/** Build a File whose reported `size` is `size` bytes (no real allocation). */
function fileOfSize(name: string, size: number): File {
  const f = new File([new Uint8Array(0)], name)
  Object.defineProperty(f, 'size', { value: size })
  return f
}

const names = (batches: File[][]): string[][] => batches.map((b) => b.map((f) => f.name))

describe('planUploadBatches', () => {
  it('packs small files under budget into a single batch', () => {
    const files = [fileOfSize('a', 100), fileOfSize('b', 100), fileOfSize('c', 100)]
    expect(names(planUploadBatches(files, 1000))).toEqual([['a', 'b', 'c']])
  })

  it('splits into new batches when the budget would overflow, preserving order', () => {
    const files = [
      fileOfSize('a', 600),
      fileOfSize('b', 600), // a+b = 1200 > 1000 -> b starts batch 2
      fileOfSize('c', 300) // b+c = 900 <= 1000 -> stays with b
    ]
    expect(names(planUploadBatches(files, 1000))).toEqual([['a'], ['b', 'c']])
  })

  it('gives a single oversize file its own solo batch without swallowing neighbours', () => {
    const files = [
      fileOfSize('small1', 200),
      fileOfSize('huge', 5000), // larger than the whole budget
      fileOfSize('small2', 200)
    ]
    // small1 fills batch 1; huge overflows -> solo batch 2; small2 overflows
    // (huge alone already exceeds budget) -> batch 3.
    expect(names(planUploadBatches(files, 1000))).toEqual([['small1'], ['huge'], ['small2']])
  })

  it('concatenating the batches reproduces the input in order', () => {
    const files = Array.from({ length: 20 }, (_, i) => fileOfSize(`f${i}`, 150))
    const flat = planUploadBatches(files, 1000).flat()
    expect(flat.map((f) => f.name)).toEqual(files.map((f) => f.name))
  })

  it('treats a file exactly at the budget as fitting, then splits the next', () => {
    const files = [fileOfSize('a', 1000), fileOfSize('b', 1)]
    expect(names(planUploadBatches(files, 1000))).toEqual([['a'], ['b']])
  })

  it('returns an empty array for no files', () => {
    expect(planUploadBatches([], 1000)).toEqual([])
  })

  it('degrades to one file per batch for a non-positive budget', () => {
    const files = [fileOfSize('a', 100), fileOfSize('b', 100)]
    expect(names(planUploadBatches(files, 0))).toEqual([['a'], ['b']])
  })
})
