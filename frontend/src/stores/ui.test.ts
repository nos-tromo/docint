import { describe, it, expect, beforeEach } from 'vitest'
import { useUiStore } from './ui'

beforeEach(() => {
  localStorage.clear()
  useUiStore.setState({
    selectedCollection: null,
    currentSessionId: null,
    previewModal: null,
    entityMergeMode: 'resolved',
    graphTopK: null
  })
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

  it('defaults the entity merge mode to resolved', () => {
    expect(useUiStore.getState().entityMergeMode).toBe('resolved')
  })

  it('updates the entity merge mode', () => {
    useUiStore.getState().setEntityMergeMode('orthographic')
    expect(useUiStore.getState().entityMergeMode).toBe('orthographic')
  })

  it('persists the entity merge mode across reloads', () => {
    useUiStore.getState().setEntityMergeMode('exact')
    const persisted = JSON.parse(localStorage.getItem('docint-ui')!).state
    expect(persisted.entityMergeMode).toBe('exact')
  })

  it('defaults graphTopK to null', () => {
    expect(useUiStore.getState().graphTopK).toBeNull()
  })

  it('updates graphTopK', () => {
    useUiStore.getState().setGraphTopK(200)
    expect(useUiStore.getState().graphTopK).toBe(200)
  })

  it('persists graphTopK across reloads', () => {
    useUiStore.getState().setGraphTopK(150)
    const persisted = JSON.parse(localStorage.getItem('docint-ui')!).state
    expect(persisted.graphTopK).toBe(150)
  })
})
