import { apiGet } from './client'
import type { AppConfig } from './types'

/** Deploy-time frontend configuration served by the backend `/config` route. */
export const getConfig = () => apiGet<AppConfig>('/config')

/** Running app version served by the backend `/version` route. */
export const getVersion = (): Promise<{ version: string }> =>
  apiGet<{ version: string }>('/version')

/** Signed-in principal served by the backend `/whoami` route (authenticated).
 * `display_name` is the edge gateway's decorative Authelia displayname
 * (`X-Auth-Name`) — undefined/null when the gateway isn't in front (dev). */
export const getWhoami = (): Promise<{ username: string; display_name: string | null }> =>
  apiGet<{ username: string; display_name: string | null }>('/whoami')
