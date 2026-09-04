import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { MetadataFilter, RetrievalMode, RetrievalTarget } from '@/api/types'
import { parseClockSeconds } from '@/lib/clock'

/** Both metadata keys a date bound has to cover. A chunk or transcript segment
 *  carries `timestamp`; a media artifact linked to a posting carries
 *  `posting_timestamp` instead. */
const TIMESTAMP_FIELDS = [
  'reference_metadata.timestamp',
  'reference_metadata.posting_timestamp'
]

/** Which stored imagery a visual turn may answer from. `video` and `social`
 *  are the two keyframe kinds; `image` is everything that is not a keyframe —
 *  a loose picture, a social image, a figure lifted out of a document. */
export type VisualSourceType = 'any' | 'video' | 'social' | 'image'

/** The `source_type` values each preset selects on the `_images` companion. */
const VISUAL_SOURCE_TYPES: Record<Exclude<VisualSourceType, 'any'>, string[]> = {
  video: ['video_keyframe'],
  social: ['social_media_keyframe'],
  image: ['social_media', 'standalone', 'document']
}

export interface CustomRule {
  id: string
  field: string
  operator: string
  value: string
}

interface ChatFiltersState {
  retrievalMode: RetrievalMode
  retrievalTarget: RetrievalTarget
  visualSourceType: VisualSourceType
  visualClipFile: string
  visualTimeFrom: string
  visualTimeTo: string
  reasoning: boolean
  filterEnabled: boolean
  mimePattern: string
  dateFrom: string
  dateTo: string
  hateSpeechOnly: boolean
  customRules: CustomRule[]
  setRetrievalMode: (m: RetrievalMode) => void
  setRetrievalTarget: (t: RetrievalTarget) => void
  setVisualSourceType: (t: VisualSourceType) => void
  setVisualClipFile: (s: string) => void
  setVisualTimeFrom: (s: string) => void
  setVisualTimeTo: (s: string) => void
  setReasoning: (b: boolean) => void
  setFilterEnabled: (b: boolean) => void
  setMimePattern: (s: string) => void
  setDateFrom: (s: string) => void
  setDateTo: (s: string) => void
  setHateSpeechOnly: (b: boolean) => void
  addRule: () => void
  updateRule: (id: string, patch: Partial<CustomRule>) => void
  removeRule: (id: string) => void
  reset: () => void
  buildPayload: () => MetadataFilter[]
}

const initial = {
  retrievalMode: 'session' as RetrievalMode,
  // Documents and imagery together, which is what chat always did.
  retrievalTarget: 'all' as RetrievalTarget,
  visualSourceType: 'any' as VisualSourceType,
  visualClipFile: '',
  visualTimeFrom: '',
  visualTimeTo: '',
  // Off until asked for: thinking buys answer quality with latency and
  // tokens, so the user opts in per chat rather than paying it on every turn.
  reasoning: false,
  filterEnabled: false,
  mimePattern: '',
  dateFrom: '',
  dateTo: '',
  hateSpeechOnly: false,
  customRules: [] as CustomRule[]
}

export const useChatFiltersStore = create<ChatFiltersState>()(
  persist(
    (set, get) => ({
      ...initial,
      setRetrievalMode: (retrievalMode) => set({ retrievalMode }),
      setRetrievalTarget: (retrievalTarget) => set({ retrievalTarget }),
      setVisualSourceType: (visualSourceType) => set({ visualSourceType }),
      setVisualClipFile: (visualClipFile) => set({ visualClipFile }),
      setVisualTimeFrom: (visualTimeFrom) => set({ visualTimeFrom }),
      setVisualTimeTo: (visualTimeTo) => set({ visualTimeTo }),
      setReasoning: (reasoning) => set({ reasoning }),
      setFilterEnabled: (filterEnabled) => set({ filterEnabled }),
      setMimePattern: (mimePattern) => set({ mimePattern }),
      setDateFrom: (dateFrom) => set({ dateFrom }),
      setDateTo: (dateTo) => set({ dateTo }),
      setHateSpeechOnly: (hateSpeechOnly) => set({ hateSpeechOnly }),
      addRule: () =>
        set((s) => ({
          customRules: [
            ...s.customRules,
            { id: crypto.randomUUID(), field: '', operator: 'eq', value: '' }
          ]
        })),
      updateRule: (id, patch) =>
        set((s) => ({
          customRules: s.customRules.map((r) => (r.id === id ? { ...r, ...patch } : r))
        })),
      removeRule: (id) =>
        set((s) => ({ customRules: s.customRules.filter((r) => r.id !== id) })),
      reset: () => set(initial),
      buildPayload: () => {
        const s = get()
        if (!s.filterEnabled) return []
        const out: MetadataFilter[] = []
        if (s.mimePattern) out.push({ field: 'mimetype', operator: 'mime_match', value: s.mimePattern })
        // One bound covers both timestamp keys: chunks and transcript segments
        // carry `timestamp`, while media artifacts linked to a posting carry
        // `posting_timestamp`. The API ORs a rule's `fields`.
        if (s.dateFrom)
          out.push({ fields: TIMESTAMP_FIELDS, operator: 'date_on_or_after', value: s.dateFrom })
        if (s.dateTo)
          out.push({ fields: TIMESTAMP_FIELDS, operator: 'date_on_or_before', value: s.dateTo })
        if (s.hateSpeechOnly)
          out.push({ field: 'hate_speech.hate_speech', operator: 'eq', value: true })
        for (const r of s.customRules) {
          if (r.field && r.operator) out.push({ field: r.field, operator: r.operator, value: r.value })
        }
        // Only under the visual target: these narrow the image companion, and
        // a `keyframe_time_sec` rule against a text chunk matches nothing at
        // all, so leaving them on would silently empty an ordinary turn.
        if (s.retrievalTarget === 'visual') {
          if (s.visualSourceType !== 'any') {
            const values = VISUAL_SOURCE_TYPES[s.visualSourceType]
            out.push(
              values.length === 1
                ? { field: 'source_type', operator: 'eq', value: values[0] }
                : { field: 'source_type', operator: 'in', values }
            )
          }
          if (s.visualClipFile)
            out.push({ field: 'source_file', operator: 'eq', value: s.visualClipFile })
          // An unreadable bound is dropped rather than sent as zero: the field
          // says so with `aria-invalid`, and a bound of zero would look like a
          // filter that worked.
          const from = parseClockSeconds(s.visualTimeFrom)
          if (from !== null) out.push({ field: 'keyframe_time_sec', operator: 'gte', value: from })
          const to = parseClockSeconds(s.visualTimeTo)
          if (to !== null) out.push({ field: 'keyframe_time_sec', operator: 'lte', value: to })
        }
        return out
      }
    }),
    {
      name: 'docint-chat-filters',
      // The retrieval mode and a built-up filter set are part of "where I
      // was" — losing them on reload is the same complaint as losing the
      // open chat. Actions are excluded automatically by partialize.
      partialize: (s) => ({
        retrievalMode: s.retrievalMode,
        retrievalTarget: s.retrievalTarget,
        visualSourceType: s.visualSourceType,
        visualClipFile: s.visualClipFile,
        visualTimeFrom: s.visualTimeFrom,
        visualTimeTo: s.visualTimeTo,
        reasoning: s.reasoning,
        filterEnabled: s.filterEnabled,
        mimePattern: s.mimePattern,
        dateFrom: s.dateFrom,
        dateTo: s.dateTo,
        hateSpeechOnly: s.hateSpeechOnly,
        customRules: s.customRules
      }),
      version: 2,
      // A v1 state predates the retrieval target entirely; its absence must
      // read as the default rather than as `undefined`, which the API would
      // reject.
      migrate: (persisted, version) =>
        version < 2
          ? {
              ...(persisted as object),
              retrievalTarget: initial.retrievalTarget,
              visualSourceType: initial.visualSourceType,
              visualClipFile: initial.visualClipFile,
              visualTimeFrom: initial.visualTimeFrom,
              visualTimeTo: initial.visualTimeTo
            }
          : persisted
    }
  )
)
