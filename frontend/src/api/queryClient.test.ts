import { describe, it, expect } from 'vitest'
import { retryPolicy } from './queryClient'
import { ApiError } from './client'

describe('retryPolicy', () => {
  it('does not retry deterministic 4xx ApiError', () => {
    expect(retryPolicy(0, new ApiError(401, 'Missing authenticated principal.'))).toBe(false)
    expect(retryPolicy(0, new ApiError(404, 'not found'))).toBe(false)
  })

  it('retries transient 5xx ApiError up to once', () => {
    expect(retryPolicy(0, new ApiError(500, 'boom'))).toBe(true)
    expect(retryPolicy(1, new ApiError(500, 'boom'))).toBe(false)
  })

  it('retries non-ApiError (network) failures up to once', () => {
    expect(retryPolicy(0, new Error('network down'))).toBe(true)
    expect(retryPolicy(1, new Error('network down'))).toBe(false)
  })
})
