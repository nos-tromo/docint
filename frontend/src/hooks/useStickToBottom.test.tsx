import { describe, it, expect } from 'vitest'
import { act, renderHook } from '@testing-library/react'
import { useStickToBottom } from './useStickToBottom'

/** Build a fake scroll container. jsdom does no layout, so the metrics are
 * plain defined properties; `scrollTop` stays writable like the real one. */
function makeScrollEl(opts: { scrollHeight: number; clientHeight: number; scrollTop: number }) {
  const el = document.createElement('div')
  Object.defineProperty(el, 'scrollHeight', { value: opts.scrollHeight, configurable: true })
  Object.defineProperty(el, 'clientHeight', { value: opts.clientHeight, configurable: true })
  el.scrollTop = opts.scrollTop
  return el
}

describe('useStickToBottom', () => {
  it('scrolls to the bottom on a dep change while pinned (initial state)', () => {
    const el = makeScrollEl({ scrollHeight: 1000, clientHeight: 300, scrollTop: 0 })
    const { result, rerender } = renderHook(({ dep }) => useStickToBottom<HTMLDivElement>(dep), {
      initialProps: { dep: 0 }
    })
    result.current.ref.current = el

    rerender({ dep: 1 })

    expect(el.scrollTop).toBe(1000)
  })

  it('stays put when the user has scrolled up past the threshold', () => {
    const el = makeScrollEl({ scrollHeight: 1000, clientHeight: 300, scrollTop: 200 })
    const { result, rerender } = renderHook(({ dep }) => useStickToBottom<HTMLDivElement>(dep), {
      initialProps: { dep: 0 }
    })
    result.current.ref.current = el

    act(() => result.current.onScroll())
    rerender({ dep: 1 })

    expect(el.scrollTop).toBe(200)
  })

  it('re-pins once the user scrolls back within the threshold of the bottom', () => {
    const el = makeScrollEl({ scrollHeight: 1000, clientHeight: 300, scrollTop: 200 })
    const { result, rerender } = renderHook(({ dep }) => useStickToBottom<HTMLDivElement>(dep), {
      initialProps: { dep: 0 }
    })
    result.current.ref.current = el

    act(() => result.current.onScroll()) // unpin (far from bottom)
    el.scrollTop = 680 // 1000 - 680 - 300 = 20px from bottom, inside threshold
    act(() => result.current.onScroll())
    rerender({ dep: 1 })

    expect(el.scrollTop).toBe(1000)
  })

  it('does nothing when no element is attached', () => {
    const { rerender } = renderHook(({ dep }) => useStickToBottom<HTMLDivElement>(dep), {
      initialProps: { dep: 0 }
    })
    expect(() => rerender({ dep: 1 })).not.toThrow()
  })
})
