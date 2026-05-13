import { streamSse } from './sse'
import type { ChatRequest } from './types'

export const streamAgentChat = (req: ChatRequest, signal?: AbortSignal) =>
  streamSse('/agent/chat/stream', req, signal)

export const streamQuery = (req: ChatRequest, signal?: AbortSignal) =>
  streamSse('/stream_query', req, signal)
