/**
 * Engine module exports
 * Provides the inference engine abstraction
 */

export { WasmEngine, createWasmEngine } from './wasm-engine.js';
export type {
  WasmEngineOptions,
  LoadModelOptions,
  GenerateOptions,
  GenerateResult,
} from './wasm-engine.js';

export { WebGPUEngine, createWebGPUEngine } from './webgpu-engine.js';
export type { WebGPUEngineOptions } from './webgpu-engine.js';

import { WasmEngine, type WasmEngineOptions } from './wasm-engine.js';
import { WebGPUEngine, type WebGPUEngineOptions } from './webgpu-engine.js';
import { WebGPUDevice } from './webgpu/device.js';

export type EngineType = 'wasm' | 'webgpu' | 'native';

export interface EngineOptions extends WasmEngineOptions, WebGPUEngineOptions {
  type?: EngineType;
}

/**
 * Engine interface that all backends must implement
 */
export interface Engine {
  init(): Promise<void>;
  loadModel(modelPath: string, options?: Record<string, unknown>): Promise<void>;
  generate(prompt: string, options?: Record<string, unknown>): Promise<{ text: string; tokensGenerated: number; done: boolean }>;
  generateStream(prompt: string, options?: Record<string, unknown>): AsyncGenerator<string, void, unknown>;
  embed(text: string): Promise<number[]>;
  isLoaded(): boolean;
  unload(): Promise<void>;
  shutdown(): Promise<void>;
}

/**
 * Create an inference engine
 */
export function createEngine(options: EngineOptions = {}): Engine {
  const engineType = options.type || 'wasm';

  switch (engineType) {
    case 'wasm':
      return new WasmEngine(options);

    case 'webgpu':
      return new WebGPUEngine(options);

    case 'native':
      throw new Error(
        'Native engine not yet implemented. Use WASM or WebGPU engine.'
      );

    default:
      throw new Error(`Unknown engine type: ${engineType}`);
  }
}

/**
 * Detect the best available engine
 */
export async function detectBestEngine(): Promise<EngineType> {
  // Check for WebGPU availability first (fastest)
  try {
    const webgpuAvailable = await WebGPUDevice.isAvailable();
    if (webgpuAvailable) {
      return 'webgpu';
    }
  } catch {
    // WebGPU not available
  }

  // Fallback to WASM
  return 'wasm';
}

/**
 * Create the best available engine
 */
export async function createBestEngine(options: Omit<EngineOptions, 'type'> = {}): Promise<Engine> {
  const bestType = await detectBestEngine();
  return createEngine({ ...options, type: bestType });
}
