import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { MetadataFilter, RetrievalMode } from '@/api/types'

/** Both metadata keys a date bound has to cover. A chunk or transcript segment
 *  carries `timestamp`; a media artifact linked to a posting carries
 *  `posting_timestamp` instead. */
const TIMESTAMP_FIELDS = [
  'reference_metadata.timestamp',
  'reference_metadata.posting_timestamp'
]

export interface CustomRule {
  id: string
  field: string
  operator: string
  value: string
}

interface ChatFiltersState {
  retrievalMode: RetrievalMode
  filterEnabled: boolean
  mimePattern: string
  dateFrom: string
  dateTo: string
  hateSpeechOnly: boolean
  customRules: CustomRule[]
  setRetrievalMode: (m: RetrievalMode) => void
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
        filterEnabled: s.filterEnabled,
        mimePattern: s.mimePattern,
        dateFrom: s.dateFrom,
        dateTo: s.dateTo,
        hateSpeechOnly: s.hateSpeechOnly,
        customRules: s.customRules
      }),
      version: 1
    }
  )
)
