import Markdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useTheme } from '@infra/ui'
import { cn } from '@/lib/cn'

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
      <Markdown remarkPlugins={[remarkGfm]} components={components}>
        {text}
      </Markdown>
    </div>
  )
}
