import { beforeEach, describe, expect, it } from 'vitest'
import { topicFocusRule, useTopicFocusStore, type TopicFocus } from './topicFocus'

const RUN = '20260102-030405-a1b2c3'
const EARLIER_RUN = '20251201-101010-0f0f0f'

const FOCUS: TopicFocus = {
  collection: 'projekt-alpha',
  owner: null,
  runId: RUN,
  topicId: 3,
  label: 'Harbour cranes'
}

beforeEach(() => useTopicFocusStore.setState({ focus: null }))

describe('topicFocusRule', () => {
  it('builds the rule only for the matching collection and run', () => {
    expect(topicFocusRule(FOCUS, { collection: 'projekt-alpha', owner: null }, RUN)).toEqual({
      field: 'docint_topic',
      operator: 'eq',
      value: '20260102-030405-a1b2c3:3'
    })

    expect(topicFocusRule(null, { collection: 'projekt-alpha', owner: null }, RUN)).toBeNull()
    expect(topicFocusRule(FOCUS, { collection: 'projekt-beta', owner: null }, RUN)).toBeNull()
    expect(topicFocusRule(FOCUS, { collection: null, owner: null }, RUN)).toBeNull()
    // A rebuild gives the collection a new run, whose values the old topic's never match.
    expect(topicFocusRule(FOCUS, { collection: 'projekt-alpha', owner: null }, EARLIER_RUN)).toBeNull()
    // No run known yet, or none at all.
    expect(topicFocusRule(FOCUS, { collection: 'projekt-alpha', owner: null }, null)).toBeNull()
  })

  it('builds the rule only under the owner the topic was picked under', () => {
    // Two owners can name a collection the same; an admin moves between them by owner alone.
    expect(topicFocusRule(FOCUS, { collection: 'projekt-alpha', owner: 'analyst-b' }, RUN)).toBeNull()

    const theirs: TopicFocus = { ...FOCUS, owner: 'analyst-b' }
    expect(topicFocusRule(theirs, { collection: 'projekt-alpha', owner: null }, RUN)).toBeNull()
    expect(topicFocusRule(theirs, { collection: 'projekt-alpha', owner: 'analyst-b' }, RUN)).toEqual({
      field: 'docint_topic',
      operator: 'eq',
      value: '20260102-030405-a1b2c3:3'
    })
  })
})

describe('useTopicFocusStore', () => {
  it('clears the focus', () => {
    useTopicFocusStore.getState().setFocus(FOCUS)
    expect(useTopicFocusStore.getState().focus).toEqual(FOCUS)

    useTopicFocusStore.getState().clear()

    expect(useTopicFocusStore.getState().focus).toBeNull()
  })
})
