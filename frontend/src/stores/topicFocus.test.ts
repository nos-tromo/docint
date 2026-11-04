import { beforeEach, describe, expect, it } from 'vitest'
import {
  fromSessionTopic,
  toSessionTopic,
  topicFocusRule,
  useTopicFocusStore,
  type TopicFocus
} from './topicFocus'

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

/** The active run, showing topic 3 and topic 5. */
const ACTIVE = { run_id: RUN, topics: [{ id: 3 }, { id: 5, members: [5] }] }

describe('topicFocusRule', () => {
  it('builds the rule only for the matching collection and run', () => {
    expect(topicFocusRule(FOCUS, { collection: 'projekt-alpha', owner: null }, ACTIVE)).toEqual({
      field: 'docint_topic',
      operator: 'eq',
      value: '20260102-030405-a1b2c3:3'
    })

    expect(topicFocusRule(null, { collection: 'projekt-alpha', owner: null }, ACTIVE)).toBeNull()
    expect(topicFocusRule(FOCUS, { collection: 'projekt-beta', owner: null }, ACTIVE)).toBeNull()
    expect(topicFocusRule(FOCUS, { collection: null, owner: null }, ACTIVE)).toBeNull()
    // A rebuild gives the collection a new run, whose values the old topic's never match.
    const rebuilt = { ...ACTIVE, run_id: EARLIER_RUN }
    expect(topicFocusRule(FOCUS, { collection: 'projekt-alpha', owner: null }, rebuilt)).toBeNull()
    // No run known yet, or none at all.
    expect(topicFocusRule(FOCUS, { collection: 'projekt-alpha', owner: null }, null)).toBeNull()
    expect(topicFocusRule(FOCUS, { collection: 'projekt-alpha', owner: null }, undefined)).toBeNull()
  })

  it('matches every topic a merged topic stands for', () => {
    const merged = { run_id: RUN, topics: [{ id: 3, members: [3, 7, 12] }] }

    expect(topicFocusRule(FOCUS, { collection: 'projekt-alpha', owner: null }, merged)).toEqual({
      field: 'docint_topic',
      operator: 'in',
      values: ['20260102-030405-a1b2c3:3', '20260102-030405-a1b2c3:7', '20260102-030405-a1b2c3:12']
    })
  })

  it('sends nothing for a topic since merged into another', () => {
    // Topic 3 went into the group under topic 1, so the run no longer shows it on its own.
    const absorbed = { run_id: RUN, topics: [{ id: 1, members: [1, 3] }] }

    expect(topicFocusRule(FOCUS, { collection: 'projekt-alpha', owner: null }, absorbed)).toBeNull()
  })

  it('builds the rule only under the owner the topic was picked under', () => {
    // Two owners can name a collection the same; an admin moves between them by owner alone.
    expect(topicFocusRule(FOCUS, { collection: 'projekt-alpha', owner: 'analyst-b' }, ACTIVE)).toBeNull()

    const theirs: TopicFocus = { ...FOCUS, owner: 'analyst-b' }
    expect(topicFocusRule(theirs, { collection: 'projekt-alpha', owner: null }, ACTIVE)).toBeNull()
    expect(topicFocusRule(theirs, { collection: 'projekt-alpha', owner: 'analyst-b' }, ACTIVE)).toEqual({
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

describe('session topics', () => {
  it('round-trips a focus through the shape the API keeps', () => {
    const stored = toSessionTopic(FOCUS)

    expect(stored).toEqual({ run_id: RUN, topic_id: 3, label: 'Harbour cranes' })
    expect(fromSessionTopic(stored, { collection: 'projekt-alpha', owner: null })).toEqual(FOCUS)
  })

  it('reads a stored topic as one of the active collection, under its owner', () => {
    // Served only through the session's own collection, so that collection is the topic's.
    const stored = toSessionTopic(FOCUS)

    expect(fromSessionTopic(stored, { collection: 'projekt-alpha', owner: 'analyst-b' })).toEqual({
      ...FOCUS,
      owner: 'analyst-b'
    })
    expect(fromSessionTopic(stored, { collection: null, owner: null })).toBeNull()
  })
})
