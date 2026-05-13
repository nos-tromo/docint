import type { ChatFinalEvent, Source } from '@/api/types'
import { Citation } from './Citation'
import { ValidationBanner } from './ValidationBanner'
import { GraphDebugPanel } from './GraphDebugPanel'

export interface ChatTurnData {
  user: string
  assistant: string
  done: boolean
  meta: ChatFinalEvent | null
}

export function ChatTurn({ turn }: { turn: ChatTurnData }) {
  const sources: Source[] = turn.meta?.sources ?? []
  return (
    <article className="space-y-3">
      <div className="rounded-md bg-zinc-900 px-4 py-2 self-end max-w-2xl ml-auto">
        <div className="text-xs text-muted-foreground mb-1">You</div>
        <div className="whitespace-pre-wrap">{turn.user}</div>
      </div>
      <div className="rounded-md bg-zinc-950 border border-border px-4 py-3">
        <div className="text-xs text-muted-foreground mb-1">Assistant</div>
        <div className="whitespace-pre-wrap">
          {turn.assistant || (turn.done ? '(no answer)' : '…')}
        </div>
        {turn.meta && <ValidationBanner v={turn.meta} />}
        {sources.length > 0 && (
          <div className="mt-3 space-y-2">
            <div className="text-xs uppercase text-muted-foreground">Sources</div>
            {sources.map((s) => (
              <Citation key={s.id} source={s} />
            ))}
          </div>
        )}
        {!!turn.meta?.graph_debug && <GraphDebugPanel data={turn.meta.graph_debug} />}
      </div>
    </article>
  )
}
