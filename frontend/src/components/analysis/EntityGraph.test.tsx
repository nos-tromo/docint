import { useState } from 'react'
import { describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { EntityGraph } from './EntityGraph'
import { colorForType } from '@/lib/entityGraphElements'
import type { NerGraphEdge, NerGraphNode } from '@/api/types'

async function selectAcme() {
  await userEvent.click(screen.getByRole('button', { name: /Acme \(ORG\)/ }))
}

/** A minimal controlled wrapper, mirroring how a real parent owns `selectedKey`. */
function ControlledEntityGraph(props: {
  nodes: NerGraphNode[]
  edges: NerGraphEdge[]
  initialSelectedKey?: string | null
  onSelectEntity?: (key: string | null) => void
}) {
  const [selectedKey, setSelectedKey] = useState<string | null>(props.initialSelectedKey ?? null)
  return (
    <EntityGraph
      nodes={props.nodes}
      edges={props.edges}
      selectedKey={selectedKey}
      onSelectEntity={(key) => {
        setSelectedKey(key)
        props.onSelectEntity?.(key)
      }}
      keyForNode={keyForNode}
    />
  )
}

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

  it('renders an edge (svg line) per graph edge', () => {
    const { container } = render(
      <EntityGraph
        nodes={nodes}
        edges={edges}
        selectedKey={null}
        onSelectEntity={() => {}}
        keyForNode={keyForNode}
      />
    )
    expect(container.querySelectorAll('line')).toHaveLength(edges.length)
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
    await userEvent.click(screen.getByRole('button', { name: /Acme \(ORG\)/ }))
    expect(onSelectEntity).toHaveBeenCalledWith('Acme::ORG')
  })

  it('clears the selection when the background is clicked', () => {
    const onSelectEntity = vi.fn()
    const { container } = render(
      <EntityGraph
        nodes={nodes}
        edges={edges}
        selectedKey="Acme::ORG"
        onSelectEntity={onSelectEntity}
        keyForNode={keyForNode}
      />
    )
    const backgroundRect = container.querySelector('svg > rect')
    expect(backgroundRect).toBeTruthy()
    fireEvent.pointerDown(backgroundRect!, { clientX: 10, clientY: 10 })
    fireEvent.pointerUp(backgroundRect!, { clientX: 10, clientY: 10 })
    expect(onSelectEntity).toHaveBeenCalledWith(null)
  })

  it('still selects an entity by key when a node is clicked (regression)', async () => {
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
    await userEvent.click(screen.getByRole('button', { name: /Acme \(ORG\)/ }))
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
    const selected = screen.getByRole('button', { name: /Acme \(ORG\)/ })
    expect(selected).toHaveAttribute('aria-pressed', 'true')
  })

  it('renders a type legend for the top entity types', () => {
    render(
      <EntityGraph
        nodes={nodes}
        edges={edges}
        selectedKey={null}
        onSelectEntity={() => {}}
        keyForNode={keyForNode}
      />
    )
    expect(screen.getByText('ORG')).toBeInTheDocument()
    expect(screen.getByText('LOC')).toBeInTheDocument()
    expect(screen.getByText('PRODUCT')).toBeInTheDocument()
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

  it('does not show the empty state while loading, even with zero nodes', () => {
    render(
      <EntityGraph
        nodes={[]}
        edges={[]}
        selectedKey={null}
        onSelectEntity={() => {}}
        keyForNode={keyForNode}
        isLoading
      />
    )
    expect(screen.queryByText(/no entity relationships to graph/i)).not.toBeInTheDocument()
  })

  it('renders the node-count control inline with the other graph controls', () => {
    render(
      <EntityGraph
        nodes={nodes}
        edges={edges}
        selectedKey={null}
        onSelectEntity={() => {}}
        keyForNode={keyForNode}
        nodeCount={80}
        nodeCountMax={500}
        onNodeCountChange={() => {}}
        onResetNodeCount={() => {}}
      />
    )
    expect(screen.getByLabelText('Graph node count')).toHaveValue(80)
  })

  it('hides the node-count control when no handler is wired', () => {
    render(
      <EntityGraph
        nodes={nodes}
        edges={edges}
        selectedKey={null}
        onSelectEntity={() => {}}
        keyForNode={keyForNode}
      />
    )
    expect(screen.queryByLabelText('Graph node count')).not.toBeInTheDocument()
  })

  it('fires onNodeCountChange when the node-count control commits a new value', async () => {
    const onNodeCountChange = vi.fn()
    render(
      <EntityGraph
        nodes={nodes}
        edges={edges}
        selectedKey={null}
        onSelectEntity={() => {}}
        keyForNode={keyForNode}
        nodeCount={80}
        nodeCountMax={500}
        onNodeCountChange={onNodeCountChange}
        onResetNodeCount={() => {}}
      />
    )
    const input = screen.getByLabelText('Graph node count')
    await userEvent.clear(input)
    await userEvent.type(input, '120')
    await userEvent.tab()
    expect(onNodeCountChange).toHaveBeenCalledWith(120)
  })

  it('resets the node count to the deploy default via its own reset control', async () => {
    const onResetNodeCount = vi.fn()
    render(
      <EntityGraph
        nodes={nodes}
        edges={edges}
        selectedKey={null}
        onSelectEntity={() => {}}
        keyForNode={keyForNode}
        nodeCount={250}
        nodeCountMax={500}
        onNodeCountChange={() => {}}
        onResetNodeCount={onResetNodeCount}
      />
    )
    await userEvent.click(screen.getByRole('button', { name: /reset node count/i }))
    expect(onResetNodeCount).toHaveBeenCalledTimes(1)
  })
})

describe('EntityGraph node removal (view-only)', () => {
  it('removes a selected node on "Remove node", keeping other nodes', async () => {
    render(<ControlledEntityGraph nodes={nodes} edges={edges} />)
    await selectAcme()
    await userEvent.click(screen.getByRole('button', { name: /remove node/i }))

    expect(screen.queryByText('Acme')).not.toBeInTheDocument()
    expect(screen.getByText('Rivertown')).toBeInTheDocument()
    expect(screen.getByText('Widget')).toBeInTheDocument()
  })

  it('removing a node also removes its incident edges', async () => {
    const { container } = render(<ControlledEntityGraph nodes={nodes} edges={edges} />)
    await selectAcme()
    await userEvent.click(screen.getByRole('button', { name: /remove node/i }))

    // Both edges touched Acme, so both should be gone.
    expect(container.querySelectorAll('line')).toHaveLength(0)
  })

  it('deselects via onSelectEntity(null) when the removed node was the selected entity', async () => {
    const onSelectEntity = vi.fn()
    render(
      <EntityGraph
        nodes={nodes}
        edges={edges}
        selectedKey="Acme::ORG"
        onSelectEntity={onSelectEntity}
        keyForNode={keyForNode}
      />
    )
    await userEvent.click(screen.getByRole('button', { name: /remove node/i }))
    expect(onSelectEntity).toHaveBeenCalledWith(null)
  })

  it('removes the selected node on Backspace', async () => {
    render(<ControlledEntityGraph nodes={nodes} edges={edges} />)
    await selectAcme()
    fireEvent.keyDown(document, { key: 'Backspace' })

    expect(screen.queryByText('Acme')).not.toBeInTheDocument()
    expect(screen.getByText('Rivertown')).toBeInTheDocument()
  })

  it('restores removed nodes when the nodes prop identity changes (fresh fetch)', async () => {
    const { rerender, container } = render(
      <EntityGraph
        nodes={nodes}
        edges={edges}
        selectedKey="Acme::ORG"
        onSelectEntity={() => {}}
        keyForNode={keyForNode}
      />
    )
    await userEvent.click(screen.getByRole('button', { name: /remove node/i }))
    expect(screen.queryByText('Acme')).not.toBeInTheDocument()

    const freshNodes = nodes.map((n) => ({ ...n }))
    const freshEdges = edges.map((e) => ({ ...e }))
    rerender(
      <EntityGraph
        nodes={freshNodes}
        edges={freshEdges}
        selectedKey={null}
        onSelectEntity={() => {}}
        keyForNode={keyForNode}
      />
    )

    expect(screen.getByText('Acme')).toBeInTheDocument()
    expect(container.querySelectorAll('line')).toHaveLength(edges.length)
  })

  it('reflects the filtered node/edge set in the type legend after removal', async () => {
    render(<ControlledEntityGraph nodes={nodes} edges={edges} />)
    await selectAcme()
    await userEvent.click(screen.getByRole('button', { name: /remove node/i }))
    expect(screen.queryByText('ORG')).not.toBeInTheDocument()
    expect(screen.getByText('LOC')).toBeInTheDocument()
    expect(screen.getByText('PRODUCT')).toBeInTheDocument()
  })
})

describe('colorForType', () => {
  it('is stable: the same type always resolves to the same color', () => {
    expect(colorForType('ORG')).toBe(colorForType('ORG'))
    expect(colorForType('PERSON')).toBe(colorForType('PERSON'))
  })

  it('differentiates at least some distinct types', () => {
    const colors = new Set(['ORG', 'PERSON', 'LOC', 'PRODUCT', 'EVENT'].map(colorForType))
    expect(colors.size).toBeGreaterThan(1)
  })
})
