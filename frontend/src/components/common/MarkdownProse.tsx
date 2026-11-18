import Markdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useTheme } from '@infra/ui'
import { cn } from '@/lib/cn'

/** The part of an mdast node the citation plugin reads and writes. */
interface MdNode {
  type: string
  value?: string
  children?: MdNode[]
  data?: { hName?: string; hProperties?: Record<string, unknown> }
}

/**
 * A run of citation markers, `[1]` or `[1] [2]`, with the space before it: the
 * superscript hangs on the word it follows, as in print.
 */
const CITATION_RUN = /\s*((?:\[\d{1,3}\]\s*)*\[\d{1,3}\])/g

/** A text's citation runs as superscripts (`<sup>1,2</sup>`), or `null` when it has none. */
function splitCitations(value: string): MdNode[] | null {
  const parts: MdNode[] = []
  let last = 0
  for (const match of value.matchAll(CITATION_RUN)) {
    if (match.index > last) parts.push({ type: 'text', value: value.slice(last, match.index) })
    const numbers = Array.from(match[1].matchAll(/\d+/g), (n) => n[0])
    parts.push({
      type: 'citation',
      data: { hName: 'sup', hProperties: { className: 'citation text-[0.7em] text-muted-foreground' } },
      children: [{ type: 'text', value: numbers.join(',') }]
    })
    last = match.index + match[0].length
  }
  if (parts.length === 0) return null
  if (last < value.length) parts.push({ type: 'text', value: value.slice(last) })
  return parts
}

function superscriptCitations(node: MdNode): void {
  if (!node.children) return
  node.children = node.children.flatMap((child) => {
    // Only prose text: code (`inlineCode`, `code`) carries a value, not text children, so it is never touched.
    const split = child.type === 'text' && child.value ? splitCitations(child.value) : null
    if (split) return split
    superscriptCitations(child)
    return [child]
  })
}

/**
 * Citation markers as superscripts. The source text keeps its `[n]`, so a copy
 * or a download still reads as the model wrote it; only the page shows `<sup>1,2</sup>`.
 */
function remarkCitations() {
  return (tree: unknown) => {
    superscriptCitations(tree as MdNode)
  }
}

const components: Components = {
  // A model-written table is often wider than its panel, so it scrolls in its own box.
  table: ({ node: _node, ...props }) => (
    <div className="overflow-x-auto" data-testid="markdown-table">
      <table {...props} />
    </div>
  )
}

/**
 * Model-written Markdown (chat answers, collection summaries) rendered as prose.
 *
 * Raw HTML in the source is shown as text, never parsed — there is no
 * `rehype-raw` — so an answer cannot inject markup into the page.
 */
export function MarkdownProse({ text }: { text: string }) {
  // The typography plugin's `prose-invert` hardcodes light-on-dark prose
  // colors; only apply it when the resolved theme is actually dark, or
  // markdown renders illegibly on the light-theme `bg-muted` panel.
  const { resolved } = useTheme()
  return (
    <div
      className={cn(
        'prose prose-sm max-w-none break-words prose-p:my-2 prose-pre:bg-muted prose-code:before:content-none prose-code:after:content-none',
        resolved === 'dark' && 'prose-invert'
      )}
    >
      <Markdown remarkPlugins={[remarkGfm, remarkCitations]} components={components}>
        {text}
      </Markdown>
    </div>
  )
}
