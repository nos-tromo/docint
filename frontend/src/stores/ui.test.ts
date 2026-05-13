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

  it('persists collection to localStorage', () => {
    useUiStore.getState().setSelectedCollection('c1')
    expect(JSON.parse(localStorage.getItem('docint-ui')!).state.selectedCollection).toBe('c1')
  })

  it('clears current session', () => {
    useUiStore.getState().setCurrentSessionId('s1')
    useUiStore.getState().setCurrentSessionId(null)
    expect(useUiStore.getState().currentSessionId).toBeNull()
  })
})
