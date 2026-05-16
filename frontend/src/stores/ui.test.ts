import { describe, it, expect, beforeEach } from 'vitest'
import { useUiStore } from './ui'

beforeEach(() => {
  localStorage.clear()
  useUiStore.setState({ selectedCollection: null, currentSessionId: null, previewModal: null })
})

describe('useUiStore', () => {
  it('updates selected collection', () => {
    useUiStore.getState().setSelectedCollection('c1')
    expect(useUiStore.getState().selectedCollection).toBe('c1')
  })

  it('does not persist the active collection across reloads', () => {
    useUiStore.getState().setSelectedCollection('c1')
    const persisted = JSON.parse(localStorage.getItem('docint-ui')!).state
    expect(persisted.selectedCollection).toBeUndefined()
  })

  it('clears current session', () => {
    useUiStore.getState().setCurrentSessionId('s1')
    useUiStore.getState().setCurrentSessionId(null)
    expect(useUiStore.getState().currentSessionId).toBeNull()
  })
})
