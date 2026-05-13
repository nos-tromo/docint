import { url } from './client'

export interface SseEvent {
  event: string
  data: unknown
}

export async function* streamSse(
  path: string,
  body?: unknown,
  signal?: AbortSignal
): AsyncGenerator<SseEvent, void, unknown> {
  const res = await fetch(url(path), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal
  })
  if (!res.ok || !res.body) {
    throw new Error(`SSE request failed: ${res.status}`)
  }
  const reader = res.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    let sep: number
    while ((sep = buffer.indexOf('\n\n')) !== -1) {
      const frame = buffer.slice(0, sep)
      buffer = buffer.slice(sep + 2)
      const event = parseFrame(frame)
      if (event) yield event
    }
  }
}

function parseFrame(frame: string): SseEvent | null {
  let event = 'message'
  const dataLines: string[] = []
  for (const line of frame.split('\n')) {
    if (line.startsWith('event:')) event = line.slice(6).trim()
    else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim())
  }
  if (dataLines.length === 0) return null
  const raw = dataLines.join('\n')
  try {
    return { event, data: JSON.parse(raw) }
  } catch {
    return { event, data: raw }
  }
}
