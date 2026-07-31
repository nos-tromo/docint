import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export type AnalysisTab = 'ner' | 'hate' | 'summary'
export type NerView = 'table' | 'graph'

/**
 * Analysis screen position.
 *
 * `tab` and `nerView` are global preferences: "I work in the graph view" should
 * follow the user across collections. The entity selection is not — an entity
 * key from another collection does not resolve against the new aggregate — so
 * it is stored with the collection it belongs to and restored only on a match.
 */
export interface AnalysisUiState {
  tab: AnalysisTab
  nerView: NerView
  entity: { key: string; collection: string } | null
  setTab: (tab: AnalysisTab) => void
  setNerView: (nerView: NerView) => void
  setEntity: (key: string | null, collection: string | null) => void
}

export const useAnalysisUiStore = create<AnalysisUiState>()(
  persist(
    (set) => ({
      tab: 'ner',
      nerView: 'table',
      entity: null,
      setTab: (tab) => set({ tab }),
      setNerView: (nerView) => set({ nerView }),
      setEntity: (key, collection) =>
        set({ entity: key && collection ? { key, collection } : null })
    }),
    { name: 'docint-analysis-ui', version: 1 }
  )
)

/**
 * Selector returning the stored entity key only when it belongs to the active
 * collection; null otherwise. This is what preserves the reset-on-switch
 * invariant the route enforced with an effect.
 *
 * @param collection - The active collection, or null when none is selected.
 * @returns A selector yielding the matching key, or null.
 */
export const selectEntityKeyFor =
  (collection: string | null) =>
  (s: AnalysisUiState): string | null =>
    collection && s.entity?.collection === collection ? s.entity.key : null
