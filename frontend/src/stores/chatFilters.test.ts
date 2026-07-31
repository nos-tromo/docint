import { describe, it, expect, beforeEach } from 'vitest'
import { useChatFiltersStore } from './chatFilters'

beforeEach(() => {
  localStorage.clear()
  useChatFiltersStore.setState({
    queryMode: 'answer',
    retrievalMode: 'session',
    filterEnabled: false,
    mimePattern: '',
    dateFrom: '',
    dateTo: '',
    hateSpeechOnly: false,
    customRules: []
  })
})

describe('useChatFiltersStore', () => {
  it('persists the query and retrieval modes', () => {
    useChatFiltersStore.getState().setQueryMode('entity_occurrence')
    useChatFiltersStore.getState().setRetrievalMode('stateless')

    const persisted = JSON.parse(localStorage.getItem('docint-chat-filters') ?? '{}')
    expect(persisted.state.queryMode).toBe('entity_occurrence')
    expect(persisted.state.retrievalMode).toBe('stateless')
  })

  it('persists custom filter rules', () => {
    useChatFiltersStore.getState().setFilterEnabled(true)
    useChatFiltersStore.getState().addRule()

    const persisted = JSON.parse(localStorage.getItem('docint-chat-filters') ?? '{}')
    expect(persisted.state.filterEnabled).toBe(true)
    expect(persisted.state.customRules).toHaveLength(1)
  })

  it('does not persist action functions', () => {
    useChatFiltersStore.getState().setQueryMode('entity_occurrence')

    const persisted = JSON.parse(localStorage.getItem('docint-chat-filters') ?? '{}')
    expect(persisted.state.setQueryMode).toBeUndefined()
    expect(persisted.state.buildPayload).toBeUndefined()
  })
})
