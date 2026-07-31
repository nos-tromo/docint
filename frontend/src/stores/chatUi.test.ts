import { beforeEach, describe, expect, it } from 'vitest'
import { draftKey, useChatUiStore } from './chatUi'

beforeEach(() => useChatUiStore.setState({ drafts: {} }))

describe('useChatUiStore', () => {
  it('keys an unstarted chat under "new"', () => {
    expect(draftKey(null)).toBe('new')
    expect(draftKey('s-1')).toBe('s-1')
  })

  it('keeps drafts isolated per session', () => {
    const { setDraft } = useChatUiStore.getState()
    setDraft('s-1', 'question one')
    setDraft('s-2', 'question two')

    expect(useChatUiStore.getState().drafts['s-1']).toBe('question one')
    expect(useChatUiStore.getState().drafts['s-2']).toBe('question two')
  })

  it('drops a draft on clear', () => {
    const { setDraft, clearDraft } = useChatUiStore.getState()
    setDraft('s-1', 'question')
    clearDraft('s-1')

    expect(useChatUiStore.getState().drafts['s-1']).toBeUndefined()
  })

  it('persists drafts across a store rehydration', () => {
    useChatUiStore.getState().setDraft('s-1', 'half typed')
    const persisted = JSON.parse(localStorage.getItem('docint-chat-ui') ?? '{}')
    expect(persisted.state.drafts['s-1']).toBe('half typed')
  })
})
