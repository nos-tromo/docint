import { apiPost } from './client'
import { streamSse } from './sse'
import type { SummaryResponse } from './types'

export const summarize = (refresh?: boolean) =>
  apiPost<SummaryResponse>('/summarize' + (refresh ? '?refresh=true' : ''))

export const streamSummary = (refresh?: boolean) =>
  streamSse('/summarize/stream' + (refresh ? '?refresh=true' : ''))
