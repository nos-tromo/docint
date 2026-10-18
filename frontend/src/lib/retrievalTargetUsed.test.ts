import { describe, it, expect } from 'vitest'
import { retrievalTargetUsed } from './retrievalTargetUsed'
import type { ChatFinalEvent } from '@/api/types'

const meta = (extra: Record<string, unknown>) =>
  ({ session_id: 's', sources: [], ...extra }) as unknown as ChatFinalEvent

describe('retrievalTargetUsed', () => {
  it('names a turn that was narrowed to documents or to images', () => {
    expect(retrievalTargetUsed(meta({ retrieval_target: 'documents' }))).toBe('documents')
    expect(retrievalTargetUsed(meta({ retrieval_target: 'visual' }))).toBe('visual')
  })

  it('says nothing for the default target', () => {
    expect(retrievalTargetUsed(meta({ retrieval_target: 'all' }))).toBeNull()
  })

  it('says nothing when no target was recorded, rather than assuming one', () => {
    // A turn from before the target was stored, or a scoped one, arrives with
    // no key at all.
    expect(retrievalTargetUsed(meta({}))).toBeNull()
    expect(retrievalTargetUsed(null)).toBeNull()
    expect(retrievalTargetUsed(undefined)).toBeNull()
  })

  it('says nothing for a value it does not know', () => {
    expect(retrievalTargetUsed(meta({ retrieval_target: 'everything' }))).toBeNull()
  })

  it('says nothing for a scoped turn, which answered from hand-picked chunks', () => {
    // The final frame still reports the requested target; the pinned selection
    // outranks it, so naming it would describe a restriction that did not apply.
    expect(
      retrievalTargetUsed(meta({ retrieval_mode: 'scoped', retrieval_target: 'documents' }))
    ).toBeNull()
  })
})
