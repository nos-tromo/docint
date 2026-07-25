import { afterEach, describe, expect, it } from 'vitest'
import { sourcePreviewUrl } from './ingest'
import { setOwnerParam } from './client'

describe('sourcePreviewUrl owner query param', () => {
  afterEach(() => setOwnerParam(null))

  it('includes owner when an admin has a foreign collection selected', () => {
    setOwnerParam('jane.doe')
    expect(sourcePreviewUrl('alpha', 'hash1')).toContain('owner=jane.doe')
  })

  it('adds nothing when no owner is set — otherwise chat citations and entity findings 404 on foreign collections', () => {
    expect(sourcePreviewUrl('alpha', 'hash1')).not.toContain('owner=')
  })
})
