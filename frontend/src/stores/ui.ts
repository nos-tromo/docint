import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { setOwnerParam } from '@/api/client'
import { useReportStore } from '@/stores/report'

export interface PreviewModal {
  collection: string
  file_hash: string
  filename: string
}

interface UiState {
  selectedCollection: string | null
  selectedOwner: string | null
  currentSessionId: string | null
  previewModal: PreviewModal | null
  graphTopK: number | null
  setSelectedCollection: (name: string | null, owner?: string | null) => void
  setCurrentSessionId: (id: string | null) => void
  setGraphTopK: (n: number | null) => void
  openPreview: (modal: PreviewModal) => void
  closePreview: () => void
}

export const useUiStore = create<UiState>()(
  persist(
    (set) => ({
      selectedCollection: null,
      selectedOwner: null,
      currentSessionId: null,
      previewModal: null,
      graphTopK: null,
      setSelectedCollection: (name, owner = null) =>
        set((s) => {
          // Invariant: the open chat always belongs to the active collection,
          // or is null. Enforced here at the single source of truth, so every
          // caller (Sidebar switch/delete/reconcile, Ingest's post-ingest
          // collection flip, any future one) drops the current session whenever
          // the active collection actually changes. Re-selecting the same
          // collection is a no-op and keeps the open chat. A foreign collection
          // with the same name is a different collection — the (name, owner)
          // pair is compared as a whole.
          if (name === s.selectedCollection && owner === s.selectedOwner) {
            return { selectedCollection: name, selectedOwner: owner }
          }
          // The same invariant holds for the open report: it is scoped to one
          // collection, so evidence taken from another one has no place in it
          // — its document overview would describe a collection its findings
          // never came from, and the server would look for their images in a
          // companion that never held them. Dropping the active id makes the
          // next add mint a report for the collection actually in front of the
          // operator.
          useReportStore.getState().setActiveReportId(null)
          return { selectedCollection: name, selectedOwner: owner, currentSessionId: null }
        }),
      setCurrentSessionId: (id) => set({ currentSessionId: id }),
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
        selectedOwner: s.selectedOwner,
        currentSessionId: s.currentSessionId,
        graphTopK: s.graphTopK
      }),
      version: 4,
      migrate: (persisted) => {
        const prior = (persisted ?? {}) as {
          selectedCollection?: string | null
          selectedOwner?: string | null
          currentSessionId?: string | null
          graphTopK?: number | null
        }
        return {
          selectedCollection: prior.selectedCollection ?? null,
          selectedOwner: prior.selectedOwner ?? null,
          currentSessionId: prior.currentSessionId ?? null,
          graphTopK: prior.graphTopK ?? null
        }
      }
    }
  )
)

setOwnerParam(useUiStore.getState().selectedOwner)
useUiStore.subscribe((s) => setOwnerParam(s.selectedOwner))
