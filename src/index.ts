/**
 * Spitfire - Node.js module for running LLMs locally
 * Compatible with Ollama models
 *
 * Standalone inference engine - no external dependencies required
 */

// Export all types
export * from './types/index.js';

// Export utilities
export * from './utils/index.js';

// Export inference engine (WASM-based, standalone)
export * from './engine/index.js';

// Export runner (legacy, for compatibility)
export * from './runner/index.js';

// Export model management
export * from './model/index.js';

// Export scheduler
export * from './scheduler/index.js';

// Export server
export * from './server/index.js';

// Export main Spitfire class
export { Spitfire } from './spitfire.js';
export type { SpitfireOptions } from './spitfire.js';

// Version
export const VERSION = '0.1.0';
