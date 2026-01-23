/**
 * WASM Engine
 * TypeScript wrapper for the llama.cpp WASM module
 */

import { EventEmitter } from 'events';
import { readFile } from 'fs/promises';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

export interface WasmEngineOptions {
  wasmPath?: string;
  numThreads?: number;
}

export interface LoadModelOptions {
  contextLength?: number;
  batchSize?: number;
  numThreads?: number;
}

export interface GenerateOptions {
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  repeatPenalty?: number;
  stop?: string[];
}

export interface GenerateResult {
  text: string;
  tokensGenerated: number;
  done: boolean;
}

interface WasmModule {
  ccall: (
    name: string,
    returnType: string,
    argTypes: string[],
    args: unknown[]
  ) => unknown;
  cwrap: (
    name: string,
    returnType: string,
    argTypes: string[]
  ) => (...args: unknown[]) => unknown;
  UTF8ToString: (ptr: number) => string;
  stringToUTF8: (str: string, ptr: number, maxLen: number) => void;
  lengthBytesUTF8: (str: string) => number;
  _malloc: (size: number) => number;
  _free: (ptr: number) => void;
  FS: {
    writeFile: (path: string, data: Uint8Array) => void;
    readFile: (path: string) => Uint8Array;
    mkdir: (path: string) => void;
    unlink: (path: string) => void;
  };
}

// StreamCallback type for future streaming implementation
// type StreamCallback = (text: string, done: boolean) => void;

export class WasmEngine extends EventEmitter {
  private module: WasmModule | null = null;
  private initialized = false;
  private modelLoaded = false;
  private options: WasmEngineOptions;

  // Wrapped functions
  private _init!: () => number;
  private _loadModel!: (
    path: number,
    nCtx: number,
    nBatch: number,
    nThreads: number
  ) => number;
  private _generate!: (
    prompt: number,
    maxTokens: number,
    temperature: number,
    topP: number,
    topK: number,
    repeatPenalty: number,
    callback: number
  ) => number;
  private _embedding!: (
    text: number,
    embeddingOut: number,
    maxDim: number
  ) => number;
  private _free!: () => void;
  private _getNCtx!: () => number;
  private _getNEmbd!: () => number;
  private _getNVocab!: () => number;

  constructor(options: WasmEngineOptions = {}) {
    super();
    this.options = options;
  }

  /**
   * Initialize the WASM module
   */
  async init(): Promise<void> {
    if (this.initialized) {
      return;
    }

    try {
      // Dynamically import the WASM module
      const wasmPath = this.options.wasmPath || this.getDefaultWasmPath();

      // In Node.js, we need to handle the module loading differently
      const createModule = await this.loadWasmModule(wasmPath);
      this.module = await createModule();

      // Wrap the exported functions
      this._init = this.module.cwrap('llama_init', 'number', []) as () => number;
      this._loadModel = this.module.cwrap('llama_load_model', 'number', [
        'number',
        'number',
        'number',
        'number',
      ]) as (path: number, nCtx: number, nBatch: number, nThreads: number) => number;
      this._generate = this.module.cwrap('llama_generate', 'number', [
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
      ]) as (
        prompt: number,
        maxTokens: number,
        temperature: number,
        topP: number,
        topK: number,
        repeatPenalty: number,
        callback: number
      ) => number;
      this._embedding = this.module.cwrap('llama_embedding', 'number', [
        'number',
        'number',
        'number',
      ]) as (text: number, embeddingOut: number, maxDim: number) => number;
      this._free = this.module.cwrap('llama_free_resources', 'void', []) as () => void;
      this._getNCtx = this.module.cwrap('llama_get_n_ctx', 'number', []) as () => number;
      this._getNEmbd = this.module.cwrap('llama_get_n_embd', 'number', []) as () => number;
      this._getNVocab = this.module.cwrap('llama_get_n_vocab', 'number', []) as () => number;

      // Initialize llama backend
      this._init();

      this.initialized = true;
      this.emit('initialized');
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(`Failed to initialize WASM engine: ${message}`);
    }
  }

  /**
   * Get the default WASM module path
   */
  private getDefaultWasmPath(): string {
    // Get the directory of the current module
    const currentDir = dirname(fileURLToPath(import.meta.url));
    return join(currentDir, '..', '..', 'dist', 'wasm', 'llama-wasm.js');
  }

  /**
   * Load the WASM module
   */
  private async loadWasmModule(
    wasmPath: string
  ): Promise<() => Promise<WasmModule>> {
    // For Node.js, we need to dynamically import the module
    try {
      const module = await import(wasmPath);
      return module.default || module.createLlamaModule;
    } catch {
      // Fallback: try to load as a file and eval (not recommended for production)
      const jsCode = await readFile(wasmPath, 'utf-8');
      const moduleFactory = new Function('module', 'exports', jsCode);
      const exports: Record<string, unknown> = {};
      const mod = { exports };
      moduleFactory(mod, exports);
      return mod.exports as unknown as () => Promise<WasmModule>;
    }
  }

  /**
   * Load a model from a file path
   */
  async loadModel(modelPath: string, options: LoadModelOptions = {}): Promise<void> {
    if (!this.initialized) {
      await this.init();
    }

    if (!this.module) {
      throw new Error('WASM module not initialized');
    }

    this.emit('loading', { path: modelPath });

    try {
      // Read the model file
      const modelData = await readFile(modelPath);

      // Write to WASM filesystem
      const wasmPath = '/model.gguf';
      this.module.FS.mkdir('/models');
      this.module.FS.writeFile(wasmPath, new Uint8Array(modelData));

      // Allocate string for path
      const pathLen = this.module.lengthBytesUTF8(wasmPath) + 1;
      const pathPtr = this.module._malloc(pathLen);
      this.module.stringToUTF8(wasmPath, pathPtr, pathLen);

      // Load the model
      const result = this._loadModel(
        pathPtr,
        options.contextLength || 2048,
        options.batchSize || 512,
        options.numThreads || this.options.numThreads || 4
      );

      this.module._free(pathPtr);

      if (result !== 0) {
        throw new Error('Failed to load model');
      }

      this.modelLoaded = true;
      this.emit('loaded', {
        contextLength: this._getNCtx(),
        embeddingSize: this._getNEmbd(),
        vocabSize: this._getNVocab(),
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(`Failed to load model: ${message}`);
    }
  }

  /**
   * Generate text completion
   */
  async generate(
    prompt: string,
    options: GenerateOptions = {}
  ): Promise<GenerateResult> {
    if (!this.modelLoaded || !this.module) {
      throw new Error('Model not loaded');
    }

    const {
      maxTokens = 256,
      temperature = 0.8,
      topP = 0.95,
      topK = 40,
      repeatPenalty = 1.1,
    } = options;

    // Allocate prompt string
    const promptLen = this.module.lengthBytesUTF8(prompt) + 1;
    const promptPtr = this.module._malloc(promptLen);
    this.module.stringToUTF8(prompt, promptPtr, promptLen);

    let fullText = '';

    // Create callback (simplified - in real implementation would use function pointer)
    // For now, we collect all output after generation
    const tokensGenerated = this._generate(
      promptPtr,
      maxTokens,
      temperature,
      topP,
      topK,
      repeatPenalty,
      0 // No streaming callback for now
    );

    this.module._free(promptPtr);

    return {
      text: fullText,
      tokensGenerated,
      done: true,
    };
  }

  /**
   * Generate text with streaming
   */
  async *generateStream(
    prompt: string,
    options: GenerateOptions = {}
  ): AsyncGenerator<string, void, unknown> {
    if (!this.modelLoaded || !this.module) {
      throw new Error('Model not loaded');
    }

    // For WASM, streaming is more complex due to the async nature
    // We'll implement a polling-based approach
    const result = await this.generate(prompt, options);

    // Yield the result character by character to simulate streaming
    for (const char of result.text) {
      yield char;
    }
  }

  /**
   * Get embeddings for text
   */
  async embed(text: string): Promise<number[]> {
    if (!this.modelLoaded || !this.module) {
      throw new Error('Model not loaded');
    }

    const embeddingSize = this._getNEmbd();
    if (embeddingSize <= 0) {
      throw new Error('Model does not support embeddings');
    }

    // Allocate text string
    const textLen = this.module.lengthBytesUTF8(text) + 1;
    const textPtr = this.module._malloc(textLen);
    this.module.stringToUTF8(text, textPtr, textLen);

    // Allocate output buffer
    const embeddingPtr = this.module._malloc(embeddingSize * 4); // 4 bytes per float

    const resultDim = this._embedding(textPtr, embeddingPtr, embeddingSize);

    this.module._free(textPtr);

    if (resultDim < 0) {
      this.module._free(embeddingPtr);
      throw new Error('Failed to compute embeddings');
    }

    // Read embeddings from WASM memory
    const embeddings: number[] = [];
    const heap = new Float32Array(
      (this.module as unknown as { HEAPF32: { buffer: ArrayBuffer } }).HEAPF32.buffer,
      embeddingPtr,
      resultDim
    );
    for (let i = 0; i < resultDim; i++) {
      embeddings.push(heap[i]);
    }

    this.module._free(embeddingPtr);

    return embeddings;
  }

  /**
   * Get model info
   */
  getModelInfo(): { contextLength: number; embeddingSize: number; vocabSize: number } | null {
    if (!this.modelLoaded) {
      return null;
    }

    return {
      contextLength: this._getNCtx(),
      embeddingSize: this._getNEmbd(),
      vocabSize: this._getNVocab(),
    };
  }

  /**
   * Check if model is loaded
   */
  isLoaded(): boolean {
    return this.modelLoaded;
  }

  /**
   * Unload the model and free resources
   */
  async unload(): Promise<void> {
    if (this.modelLoaded && this._free) {
      this._free();
      this.modelLoaded = false;
      this.emit('unloaded');
    }
  }

  /**
   * Shutdown the engine
   */
  async shutdown(): Promise<void> {
    await this.unload();
    this.initialized = false;
    this.module = null;
    this.emit('shutdown');
  }
}

/**
 * Create a WASM engine instance
 */
export function createWasmEngine(options?: WasmEngineOptions): WasmEngine {
  return new WasmEngine(options);
}
