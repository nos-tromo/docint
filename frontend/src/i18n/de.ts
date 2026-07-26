import type { Strings } from './index'

export const de: Strings = {
  // nav
  'nav.dashboard': 'Dashboard',
  'nav.chat': 'Chat',
  'nav.ingest': 'Datenaufnahme',
  'nav.analysis': 'Analyse',
  'nav.inspector': 'Inspektion',
  'nav.report': 'Bericht',

  // common
  'common.copied': 'Kopiert',
  'common.copy': 'Kopieren',
  'common.active': 'Aktiv',
  'common.collection': 'Sammlung',
  'common.choose_collection': 'Sammlung auswählen…',
  'common.no_collections': 'Noch keine Sammlungen',
  'common.select_collection_aria': 'Sammlung auswählen',
  'common.delete_collection_aria': 'Sammlung {name} löschen',
  'common.delete_collection_title': 'Diese Sammlung löschen',
  'common.delete_collection_confirm':
    'Sammlung {label} löschen? Dies kann nicht rückgängig gemacht werden.',
  'common.no_active_collection': 'Keine aktive Sammlung — wählen Sie eine aus, um Anfragen zu stellen.',
  'common.sessions': 'Sitzungen',
  'common.new_session': '+ Neu',
  'common.loading_chats': 'Lädt Chats...',
  'common.sessions_error_default': 'Chats konnten nicht geladen werden.',
  'common.sessions_error_auth':
    'Der Sitzungsverlauf erfordert einen authentifizierten Benutzer oder DOCINT_DEFAULT_IDENTITY.',
  'common.no_chats_in_collection': 'Noch keine Chats in dieser Sammlung.',
  'common.select_collection_to_see_chats': 'Wählen Sie eine Sammlung aus, um deren Chats zu sehen.',
  'common.session_title_fallback': 'Sitzung {id}',
  'common.delete_session_aria': 'Sitzung löschen',
  'common.delete_session_confirm': 'Diesen Chat löschen?',
  'common.show_more': 'Mehr anzeigen',
  'common.show_less': 'Weniger anzeigen',
  'common.translate': 'Übersetzen',
  'common.translation': 'Übersetzung',
  'common.show_original': 'Original anzeigen',
  'common.translation_unavailable': 'Übersetzung nicht verfügbar — Original wird angezeigt.',
  'common.entity_merge_mode_aria': 'Entitäten-Zusammenführungsmodus',
  'common.merge_resolved': 'Aufgelöst',
  'common.merge_orthographic': 'Orthografisch',
  'common.merge_exact': 'Exakt',
  'common.loading_ellipsis': 'Lädt…',

  // table (shared DataTable chrome)
  'table.col_filename': 'Dateiname',
  'table.col_type': 'Typ',
  'table.col_units': 'Einheiten',
  'table.col_nodes': 'Knoten',
  'table.col_entities': 'Entitäten',
  'table.col_hash': 'Hash',
  'table.documents_one': '{count} Dokument',
  'table.documents_other': '{count} Dokumente',
  'table.loading_suffix': '· lädt…',
  'table.export_csv': 'CSV exportieren',
  'table.empty_title': 'Noch keine Dokumente in dieser Sammlung.',
  'table.empty_hint': 'Nehmen Sie Dateien für die Datenaufnahme auf, um sie hier zu sehen.',
  'table.load_more': 'Weitere laden',
  'table.aria_documents': 'Dokumente',
  'table.copy_hash': 'Hash für {filename} kopieren',

  // upload
  'upload.drop_hint': 'Dateien hier ablegen oder klicken, um sie auszuwählen.',
  'upload.choose_folder': 'Oder einen Ordner auswählen',
}
