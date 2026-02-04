/**
 * Debug logging utility for WebGPU engine
 * Gate all verbose output behind a single flag so production runs are clean.
 */

let _enabled = false;

export function setDebugEnabled(enabled: boolean): void {
  _enabled = enabled;
}

export function isDebugEnabled(): boolean {
  return _enabled;
}

/** Log only when debug mode is on. Same signature as console.log. */
export function debugLog(...args: unknown[]): void {
  if (_enabled) console.log(...args);
}
