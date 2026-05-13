import { create } from 'zustand'
import type { MetadataFilter, QueryMode, RetrievalMode } from '@/api/types'

export interface CustomRule {
  id: string
  field: string
  operator: string
  value: string
}

interface ChatFiltersState {
  queryMode: QueryMode
  retrievalMode: RetrievalMode
  filterEnabled: boolean
  mimePattern: string
  dateFrom: string
  dateTo: string
  hateSpeechOnly: boolean
  customRules: CustomRule[]
  setQueryMode: (m: QueryMode) => void
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
  queryMode: 'answer' as QueryMode,
  retrievalMode: 'session' as RetrievalMode,
  filterEnabled: false,
  mimePattern: '',
  dateFrom: '',
  dateTo: '',
  hateSpeechOnly: false,
  customRules: [] as CustomRule[]
}

export const useChatFiltersStore = create<ChatFiltersState>((set, get) => ({
  ...initial,
  setQueryMode: (queryMode) => set({ queryMode }),
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
        { id: crypto.randomUUID(), field: '', operator: 'equals', value: '' }
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
    if (s.mimePattern) out.push({ field: 'mimetype', operator: 'matches', value: s.mimePattern })
    if (s.dateFrom) out.push({ field: 'date', operator: 'gte', value: s.dateFrom })
    if (s.dateTo) out.push({ field: 'date', operator: 'lte', value: s.dateTo })
    if (s.hateSpeechOnly) out.push({ field: 'hate_speech_flagged', operator: 'eq', value: true })
    for (const r of s.customRules) {
      if (r.field && r.operator) out.push({ field: r.field, operator: r.operator, value: r.value })
    }
    return out
  }
}))
