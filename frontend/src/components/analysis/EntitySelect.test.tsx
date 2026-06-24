import { describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { EntitySelect } from './EntitySelect'
import type { NerEntityRow } from '@/api/types'

const entities: NerEntityRow[] = [
  { text: 'Berlin', type: 'LOC', mentions: 4 },
  { text: 'Paris', type: 'LOC', mentions: 3 },
  { text: 'Alice', type: 'PER', mentions: 2 }
]

const keyOf = (e: NerEntityRow) => `${e.text}::${e.type}`

describe('EntitySelect', () => {
  it('lists every entity option labelled with type and mentions by default', () => {
    render(
      <EntitySelect entities={entities} selectedKey="Berlin::LOC" onSelectEntity={() => {}} keyOf={keyOf} />
    )
    const dropdown = screen.getByLabelText(/^entity$/i) as HTMLSelectElement
    expect(Array.from(dropdown.options).map((o) => o.value)).toEqual([
      'Berlin::LOC',
      'Paris::LOC',
      'Alice::PER'
    ])
    expect(dropdown.options[0].textContent).toMatch(/Berlin \[LOC\] · 4/)
  })

  it('filters the entity list by the chosen category and pre-selects its top entity', async () => {
    const onSelectEntity = vi.fn()
    render(
      <EntitySelect entities={entities} selectedKey="Berlin::LOC" onSelectEntity={onSelectEntity} keyOf={keyOf} />
    )
    // Selecting the PER category must (a) re-filter the entity dropdown and
    // (b) pre-select that category's top entity — the previous ref-based
    // selector did neither.
    await userEvent.selectOptions(screen.getByLabelText(/entity category/i), 'PER')
    expect(onSelectEntity).toHaveBeenCalledWith('Alice::PER')

    const dropdown = screen.getByLabelText(/^entity$/i) as HTMLSelectElement
    expect(Array.from(dropdown.options).map((o) => o.value)).toEqual(['Alice::PER'])
  })

  it('emits the new key when the user picks a different entity', async () => {
    const onSelectEntity = vi.fn()
    render(
      <EntitySelect entities={entities} selectedKey="Berlin::LOC" onSelectEntity={onSelectEntity} keyOf={keyOf} />
    )
    await userEvent.selectOptions(screen.getByLabelText(/^entity$/i), 'Alice::PER')
    expect(onSelectEntity).toHaveBeenCalledWith('Alice::PER')
  })

  it('falls back to a helpful message when there are no entities', () => {
    render(<EntitySelect entities={[]} selectedKey={null} onSelectEntity={() => {}} keyOf={keyOf} />)
    expect(screen.getByText(/no entities found/i)).toBeInTheDocument()
  })
})
