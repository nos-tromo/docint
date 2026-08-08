import { describe, it, expect, beforeEach } from 'vitest'
import { useChatFiltersStore } from './chatFilters'

beforeEach(() => {
  localStorage.clear()
  useChatFiltersStore.setState({
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
  it('persists the retrieval mode', () => {
    useChatFiltersStore.getState().setRetrievalMode('stateless')

    const persisted = JSON.parse(localStorage.getItem('docint-chat-filters') ?? '{}')
    expect(persisted.state.retrievalMode).toBe('stateless')
  })

  it('persists custom filter rules', () => {
    useChatFiltersStore.getState().setFilterEnabled(true)
    useChatFiltersStore.getState().addRule()

    const persisted = JSON.parse(localStorage.getItem('docint-chat-filters') ?? '{}')
    expect(persisted.state.filterEnabled).toBe(true)
    expect(persisted.state.customRules).toHaveLength(1)
  })

  it('emits the API date operators, not gte/lte aliases', () => {
    const s = useChatFiltersStore.getState()
    s.setFilterEnabled(true)
    s.setDateFrom('2026-01-01')
    s.setDateTo('2026-02-01')

    const operators = useChatFiltersStore.getState().buildPayload().map((r) => r.operator)

    expect(operators).toContain('date_on_or_after')
    expect(operators).toContain('date_on_or_before')
    expect(operators).not.toContain('date_gte')
    expect(operators).not.toContain('date_lte')
  })

  it('applies date bounds to both timestamp keys', () => {
    const s = useChatFiltersStore.getState()
    s.setFilterEnabled(true)
    s.setDateFrom('2026-01-01')

    const [rule] = useChatFiltersStore.getState().buildPayload()

    expect(rule.fields).toEqual([
      'reference_metadata.timestamp',
      'reference_metadata.posting_timestamp'
    ])
    expect(rule.field).toBeUndefined()
  })

  it('filters hate speech on the real nested payload key', () => {
    const s = useChatFiltersStore.getState()
    s.setFilterEnabled(true)
    s.setHateSpeechOnly(true)

    const [rule] = useChatFiltersStore.getState().buildPayload()

    expect(rule.field).toBe('hate_speech.hate_speech')
    expect(rule.operator).toBe('eq')
    expect(rule.value).toBe(true)
  })

  it('emits nothing while filters are disabled', () => {
    const s = useChatFiltersStore.getState()
    s.setFilterEnabled(false)
    s.setMimePattern('application/pdf')
    s.setDateFrom('2026-01-01')

    expect(useChatFiltersStore.getState().buildPayload()).toEqual([])
  })

  it('does not persist action functions', () => {
    useChatFiltersStore.getState().setRetrievalMode('stateless')

    const persisted = JSON.parse(localStorage.getItem('docint-chat-filters') ?? '{}')
    expect(persisted.state.setRetrievalMode).toBeUndefined()
    expect(persisted.state.buildPayload).toBeUndefined()
  })
})
