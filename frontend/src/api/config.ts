import { apiGet } from './client'
import type { AppConfig } from './types'

/** Deploy-time frontend configuration served by the backend `/config` route. */
export const getConfig = () => apiGet<AppConfig>('/config')

/** Running app version served by the backend `/version` route. */
export const getVersion = (): Promise<{ version: string }> =>
  apiGet<{ version: string }>('/version')
