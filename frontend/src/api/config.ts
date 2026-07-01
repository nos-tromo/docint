import { apiGet } from './client'
import type { AppConfig } from './types'

/** Deploy-time frontend configuration served by the backend `/config` route. */
export const getConfig = () => apiGet<AppConfig>('/config')
