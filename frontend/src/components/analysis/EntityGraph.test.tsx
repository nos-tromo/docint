import { describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { EntityGraph } from './EntityGraph'
import type { NerGraphEdge, NerGraphNode } from '@/api/types'

const nodes: NerGraphNode[] = [
  { id: 'acme::org', text: 'Acme', type: 'ORG', mentions: 9 },
  { id: 'rivertown::loc', text: 'Rivertown', type: 'LOC', mentions: 4 },
  { id: 'widget::product', text: 'Widget', type: 'PRODUCT', mentions: 2 }
]

const edges: NerGraphEdge[] = [
  { source: 'acme::org', target: 'rivertown::loc', label: 'located_in', kind: 'relation', weight: 3 },
  { source: 'acme::org', target: 'widget::product', label: 'makes', kind: 'relation', weight: 1 }
]

const keyForNode = (n: NerGraphNode) => `${n.text}::${n.type}`

describe('EntityGraph', () => {
  it('renders a node per entity with its label', () => {
    render(
      <EntityGraph
        nodes={nodes}
        edges={edges}
        selectedKey={null}
        onSelectEntity={() => {}}
        keyForNode={keyForNode}
      />
    )
    expect(screen.getByText('Acme')).toBeInTheDocument()
    expect(screen.getByText('Rivertown')).toBeInTheDocument()
    expect(screen.getByText('Widget')).toBeInTheDocument()
  })

  it('selects an entity (by text::type key) when its node is clicked', async () => {
    const onSelectEntity = vi.fn()
    render(
      <EntityGraph
        nodes={nodes}
        edges={edges}
        selectedKey={null}
        onSelectEntity={onSelectEntity}
        keyForNode={keyForNode}
      />
    )
    await userEvent.click(screen.getByRole('button', { name: /Acme \(ORG, 9 mentions\)/ }))
    expect(onSelectEntity).toHaveBeenCalledWith('Acme::ORG')
  })

  it('marks the selected node pressed for assistive tech', () => {
    render(
      <EntityGraph
        nodes={nodes}
        edges={edges}
        selectedKey="Acme::ORG"
        onSelectEntity={() => {}}
        keyForNode={keyForNode}
      />
    )
    const selected = screen.getByRole('button', { name: /Acme \(ORG/ })
    expect(selected).toHaveAttribute('aria-pressed', 'true')
  })

  it('renders a type legend', () => {
    render(
      <EntityGraph
        nodes={nodes}
        edges={edges}
        selectedKey={null}
        onSelectEntity={() => {}}
        keyForNode={keyForNode}
      />
    )
    expect(screen.getByText('Types')).toBeInTheDocument()
    expect(screen.getByText('ORG')).toBeInTheDocument()
    expect(screen.getByText('PRODUCT')).toBeInTheDocument()
  })

  it('filters out low-degree nodes via the min-edges stepper (default 0 shows all)', async () => {
    render(
      <EntityGraph
        nodes={nodes}
        edges={edges}
        selectedKey={null}
        onSelectEntity={() => {}}
        keyForNode={keyForNode}
      />
    )
    // Default threshold is 0: every node visible, and decrement is disabled.
    expect(screen.getByText('Rivertown')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /decrease minimum edges/i })).toBeDisabled()

    const inc = screen.getByRole('button', { name: /increase minimum edges/i })
    await userEvent.click(inc) // min edges = 1 — every node still has ≥1 edge.
    expect(screen.getByText('Widget')).toBeInTheDocument()

    await userEvent.click(inc) // min edges = 2 — only Acme (degree 2) survives.
    expect(screen.getByText('Acme')).toBeInTheDocument()
    expect(screen.queryByText('Rivertown')).not.toBeInTheDocument()
    expect(screen.queryByText('Widget')).not.toBeInTheDocument()
    // Acme is the most-connected node, so the threshold cannot climb further.
    expect(inc).toBeDisabled()
  })

  it('renders an edge-length slider defaulting to 1x density', () => {
    render(
      <EntityGraph
        nodes={nodes}
        edges={edges}
        selectedKey={null}
        onSelectEntity={() => {}}
        keyForNode={keyForNode}
      />
    )
    const slider = screen.getByRole('slider', { name: /edge length/i }) as HTMLInputElement
    expect(slider).toBeInTheDocument()
    // Default keeps today's density (1x); users widen toward 3x or compact to 0.5x.
    expect(slider.value).toBe('1')
  })

  it('updates the edge-length value when the slider is dragged', () => {
    render(
      <EntityGraph
        nodes={nodes}
        edges={edges}
        selectedKey={null}
        onSelectEntity={() => {}}
        keyForNode={keyForNode}
      />
    )
    const slider = screen.getByRole('slider', { name: /edge length/i }) as HTMLInputElement
    fireEvent.change(slider, { target: { value: '2.5' } })
    expect(slider.value).toBe('2.5')
  })

  it('labels the zoom controls as their own group (distinct from min-edges)', () => {
    render(
      <EntityGraph
        nodes={nodes}
        edges={edges}
        selectedKey={null}
        onSelectEntity={() => {}}
        keyForNode={keyForNode}
      />
    )
    expect(screen.getByRole('group', { name: /zoom/i })).toBeInTheDocument()
  })

  it('reset restores min-edges, edge-length, and zoom to their defaults', async () => {
    render(
      <EntityGraph
        nodes={nodes}
        edges={edges}
        selectedKey={null}
        onSelectEntity={() => {}}
        keyForNode={keyForNode}
      />
    )
    // Move both filter controls off their defaults.
    await userEvent.click(screen.getByRole('button', { name: /increase minimum edges/i }))
    const slider = screen.getByRole('slider', { name: /edge length/i }) as HTMLInputElement
    fireEvent.change(slider, { target: { value: '2.5' } })
    // Sanity: they are now off-default (decrement enabled, slider at 2.5x).
    expect(screen.getByRole('button', { name: /decrease minimum edges/i })).not.toBeDisabled()
    expect(slider.value).toBe('2.5')

    await userEvent.click(screen.getByRole('button', { name: /^reset$/i }))

    // Reset returns min-edges to 0 (decrement disabled) and edge length to 1x.
    expect(screen.getByRole('button', { name: /decrease minimum edges/i })).toBeDisabled()
    expect((screen.getByRole('slider', { name: /edge length/i }) as HTMLInputElement).value).toBe(
      '1'
    )
  })

  it('attaches a non-passive wheel listener so wheel-zoom cannot scroll the page', () => {
    // React registers wheel listeners as passive (and, failing jsdom's
    // passive-support probe, with a boolean capture flag), where the handler's
    // preventDefault() is a no-op and the page scrolls behind the graph. The
    // component must attach its own { passive: false } listener instead — assert
    // that exact registration happened. (jsdom gives DOM nodes their own
    // EventTarget prototype, distinct from the global `EventTarget`, so we probe
    // a real SVG element to find the prototype that actually owns
    // addEventListener before spying on it.)
    const probe = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
    let proto: object | null = Object.getPrototypeOf(probe)
    while (proto && !Object.prototype.hasOwnProperty.call(proto, 'addEventListener')) {
      proto = Object.getPrototypeOf(proto)
    }
    const addSpy = vi.spyOn(proto as EventTarget, 'addEventListener')
    try {
      render(
        <EntityGraph
          nodes={nodes}
          edges={edges}
          selectedKey={null}
          onSelectEntity={() => {}}
          keyForNode={keyForNode}
        />
      )
      const nonPassiveWheel = addSpy.mock.calls.some(
        ([type, , opts]) =>
          type === 'wheel' &&
          typeof opts === 'object' &&
          opts !== null &&
          (opts as AddEventListenerOptions).passive === false
      )
      expect(nonPassiveWheel).toBe(true)
    } finally {
      addSpy.mockRestore()
    }
  })

  it('shows an empty state when there are no nodes', () => {
    render(
      <EntityGraph
        nodes={[]}
        edges={[]}
        selectedKey={null}
        onSelectEntity={() => {}}
        keyForNode={keyForNode}
      />
    )
    expect(screen.getByText(/no entity relationships to graph/i)).toBeInTheDocument()
  })
})
