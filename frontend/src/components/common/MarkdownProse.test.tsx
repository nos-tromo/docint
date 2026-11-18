import { describe, it, expect } from 'vitest'
import { render } from '@testing-library/react'
import { MarkdownProse } from './MarkdownProse'

function sups(container: HTMLElement): (string | null)[] {
  return Array.from(container.querySelectorAll('sup'), (sup) => sup.textContent)
}

describe('MarkdownProse citations', () => {
  it('sets a run of citation markers as one superscript on the word before it', () => {
    const { container } = render(<MarkdownProse text="The meeting set three dates [1] [2]: a kickoff [3]." />)

    expect(sups(container)).toEqual(['1,2', '3'])
    const paragraph = container.querySelector('p')
    expect(paragraph?.textContent).toBe('The meeting set three dates1,2: a kickoff3.')
    expect(paragraph?.textContent).not.toContain('[')
  })

  it('finds markers inside emphasis, lists and tables', () => {
    const { container } = render(
      <MarkdownProse
        text={'**Alice chairs it [1]**\n\n- Bob joined [2]\n\n| Who | Source |\n| --- | --- |\n| Carol | seen [3] |'}
      />
    )

    expect(sups(container)).toEqual(['1', '2', '3'])
  })

  it('leaves code as written', () => {
    const { container } = render(<MarkdownProse text={'Index with `items[0]` and\n\n```\nrows[1] = 2\n```'} />)

    expect(sups(container)).toEqual([])
    expect(container.querySelector('code')?.textContent).toBe('items[0]')
    expect(container.querySelector('pre')?.textContent).toContain('rows[1] = 2')
  })

  it('leaves bracketed words and links alone', () => {
    const { container } = render(<MarkdownProse text="A [draft] note, see [the plan](https://example.invalid/plan)." />)

    expect(sups(container)).toEqual([])
    expect(container.textContent).toContain('[draft]')
  })
})
