import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { EntityMergeMode } from '@/api/types'

export interface PreviewModal {
  collection: string
  file_hash: string
  filename: string
}

interface UiState {
  selectedCollection: string | null
  currentSessionId: string | null
  previewModal: PreviewModal | null
  entityMergeMode: EntityMergeMode
  graphTopK: number | null
  setSelectedCollection: (name: string | null) => void
  setCurrentSessionId: (id: string | null) => void
  setEntityMergeMode: (mode: EntityMergeMode) => void
  setGraphTopK: (n: number | null) => void
  openPreview: (modal: PreviewModal) => void
  closePreview: () => void
}

export const useUiStore = create<UiState>()(
  persist(
    (set) => ({
      selectedCollection: null,
      currentSessionId: null,
      previewModal: null,
      entityMergeMode: 'resolved',
      graphTopK: null,
      setSelectedCollection: (name) =>
        set((s) =>
          // Invariant: the open chat always belongs to the active collection,
          // or is null. Enforced here at the single source of truth, so every
          // caller (Sidebar switch/delete/reconcile, Ingest's post-ingest
          // collection flip, any future one) drops the current session whenever
          // the active collection actually changes. Re-selecting the same
          // collection is a no-op and keeps the open chat.
          name === s.selectedCollection
            ? { selectedCollection: name }
            : { selectedCollection: name, currentSessionId: null }
        ),
      setCurrentSessionId: (id) => set({ currentSessionId: id }),
      setEntityMergeMode: (mode) => set({ entityMergeMode: mode }),
      setGraphTopK: (n) => set({ graphTopK: n }),
      openPreview: (modal) => set({ previewModal: modal }),
      closePreview: () => set({ previewModal: null })
    }),
    {
      name: 'docint-ui',
      // The active collection is client-authoritative post-WS2 (sent per
      // request; there is no server-side "active collection" singleton), so it
      // is safe — and desirable — to persist it across reloads: a resumed chat
      // keeps its collection and no longer errors after a refresh. On load the
      // Sidebar reconciles the persisted selection against the owned-collections
      // list and clears it if the collection no longer exists.
      partialize: (s) => ({
        selectedCollection: s.selectedCollection,
        currentSessionId: s.currentSessionId,
        entityMergeMode: s.entityMergeMode,
        graphTopK: s.graphTopK
      }),
      version: 2,
      migrate: (persisted) => {
        const prior = (persisted ?? {}) as {
          selectedCollection?: string | null
          currentSessionId?: string | null
          entityMergeMode?: EntityMergeMode
          graphTopK?: number | null
        }
        return {
          selectedCollection: prior.selectedCollection ?? null,
          currentSessionId: prior.currentSessionId ?? null,
          entityMergeMode: prior.entityMergeMode ?? 'resolved',
          graphTopK: prior.graphTopK ?? null
        }
      }
    }
  )
)
