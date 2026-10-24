/**
 * GuessMyText API Configuration
 *
 * Resolves the backend API base URL using Vite environment variables:
 * - VITE_API_BASE_URL: e.g. "https://guessmytext-backend.onrender.com" or "http://localhost:5000"
 * - Defaults to "" (empty string), which uses relative paths (e.g. "/api/...").
 *   This works seamlessly with Vite's dev proxy locally or when deployed behind a reverse proxy.
 */

const rawEnvUrl = import.meta.env.VITE_API_BASE_URL;

export const API_BASE_URL: string = (
  typeof rawEnvUrl === 'string' ? rawEnvUrl.trim() : ''
).replace(/\/+$/, '');

/**
 * Builds a full API URL given a relative endpoint path.
 *
 * @param endpoint - Relative path (e.g. "/api/predict" or "api/predict")
 * @returns Fully qualified URL or relative URL with leading slash
 *
 * Examples:
 *   apiUrl('/api/predict') -> "http://localhost:5000/api/predict" (if VITE_API_BASE_URL is set)
 *   apiUrl('/api/predict') -> "/api/predict"                      (if VITE_API_BASE_URL is empty)
 */
export function apiUrl(endpoint: string): string {
  if (!endpoint) return API_BASE_URL;
  if (endpoint.startsWith('http://') || endpoint.startsWith('https://')) {
    return endpoint;
  }
  const cleanEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
  return `${API_BASE_URL}${cleanEndpoint}`;
}
