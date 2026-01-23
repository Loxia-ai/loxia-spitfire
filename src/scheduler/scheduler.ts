/**
 * Model Scheduler
 * Manages model loading, unloading, and request routing
 */

import { EventEmitter } from 'events';
import { RunnerManager, type RunnerInstance } from '../runner/index.js';
import { ModelManager, type LoadedModelInfo } from '../model/index.js';
import { parseDuration } from '../utils/index.js';
import type { Duration } from '../types/index.js';

export interface SchedulerOptions {
  modelsPath?: string;
  maxLoadedModels?: number;
  defaultKeepAlive?: Duration;
  loadTimeout?: number;
  runnerPath?: string;
}

export interface LoadedModel {
  name: string;
  modelInfo: LoadedModelInfo;
  runner: RunnerInstance;
  expiresAt: number;
  requestCount: number;
  lastUsedAt: Date;
}

interface PendingRequest {
  modelName: string;
  resolve: (model: LoadedModel) => void;
  reject: (error: Error) => void;
}

export class Scheduler extends EventEmitter {
  private runnerManager: RunnerManager;
  private modelManager: ModelManager;
  private loadedModels: Map<string, LoadedModel> = new Map();
  private pendingRequests: Map<string, PendingRequest[]> = new Map();
  private loadingModels: Set<string> = new Set();
  private expirationTimer: ReturnType<typeof setInterval> | null = null;

  private maxLoadedModels: number;
  private defaultKeepAlive: number;
  private loadTimeout: number;

  constructor(options: SchedulerOptions = {}) {
    super();

    this.maxLoadedModels = options.maxLoadedModels || 1;
    this.defaultKeepAlive = parseDuration(options.defaultKeepAlive || '5m');
    this.loadTimeout = options.loadTimeout || 300000;

    this.modelManager = new ModelManager({
      modelsPath: options.modelsPath,
    });

    this.runnerManager = new RunnerManager({
      modelsPath: options.modelsPath,
      loadTimeout: this.loadTimeout,
      runnerPath: options.runnerPath,
    });

    // Forward runner events
    this.runnerManager.on('spawning', (data) => this.emit('spawning', data));
    this.runnerManager.on('launched', (data) => this.emit('launched', data));
    this.runnerManager.on('loading', (data) => this.emit('loading', data));
    this.runnerManager.on('loadProgress', (data) =>
      this.emit('loadProgress', data)
    );
    this.runnerManager.on('loaded', (data) => this.emit('loaded', data));
    this.runnerManager.on('error', (data) => this.emit('error', data));
    this.runnerManager.on('exit', (data) => this.handleRunnerExit(data));

    // Start expiration checker
    this.startExpirationChecker();
  }

  /**
   * Get a runner for a model, loading it if necessary
   */
  async getRunner(
    modelName: string,
    keepAlive?: Duration
  ): Promise<LoadedModel> {
    const keepAliveMs = parseDuration(keepAlive) || this.defaultKeepAlive;

    // Check if already loaded
    const loaded = this.loadedModels.get(modelName);
    if (loaded && loaded.runner.status === 'ready') {
      loaded.expiresAt = Date.now() + keepAliveMs;
      loaded.lastUsedAt = new Date();
      loaded.requestCount++;
      return loaded;
    }

    // Check if currently loading
    if (this.loadingModels.has(modelName)) {
      return this.waitForLoad(modelName);
    }

    // Need to load the model
    return this.loadModel(modelName, keepAliveMs);
  }

  /**
   * Wait for a model that's currently loading
   */
  private waitForLoad(modelName: string): Promise<LoadedModel> {
    return new Promise((resolve, reject) => {
      const pending = this.pendingRequests.get(modelName) || [];
      pending.push({ modelName, resolve, reject });
      this.pendingRequests.set(modelName, pending);
    });
  }

  /**
   * Load a model
   */
  private async loadModel(
    modelName: string,
    keepAliveMs: number
  ): Promise<LoadedModel> {
    this.loadingModels.add(modelName);
    this.emit('modelLoading', { name: modelName });

    try {
      // First, ensure we have room
      await this.ensureCapacity();

      // Load model info
      const modelInfo = await this.modelManager.load(modelName);
      if (!modelInfo) {
        throw new Error(`Model not found: ${modelName}`);
      }

      // Spawn runner
      const runner = await this.runnerManager.spawn({
        modelPath: modelInfo.modelPath,
        ollamaEngine: true, // Prefer new engine
      });

      // Load the model on the runner
      const options: {
        parallel?: number;
        batchSize?: number;
        numCtx?: number;
        flashAttention?: boolean;
      } = {};

      // Apply model parameters
      if (modelInfo.parameters) {
        if (modelInfo.parameters['num_ctx']) {
          options.numCtx = modelInfo.parameters['num_ctx'] as number;
        }
        if (modelInfo.parameters['num_batch']) {
          options.batchSize = modelInfo.parameters['num_batch'] as number;
        }
      }

      // Use architecture context length if available
      if (modelInfo.architecture?.contextLength) {
        options.numCtx = Math.min(
          options.numCtx || modelInfo.architecture.contextLength,
          modelInfo.architecture.contextLength
        );
      }

      await this.runnerManager.load(runner, options);

      const loaded: LoadedModel = {
        name: modelName,
        modelInfo,
        runner,
        expiresAt: Date.now() + keepAliveMs,
        requestCount: 1,
        lastUsedAt: new Date(),
      };

      this.loadedModels.set(modelName, loaded);
      this.loadingModels.delete(modelName);

      // Resolve pending requests
      this.resolvePendingRequests(modelName, loaded);

      this.emit('modelLoaded', { name: modelName, runner: runner.id });

      return loaded;
    } catch (error) {
      this.loadingModels.delete(modelName);

      // Reject pending requests
      this.rejectPendingRequests(
        modelName,
        error instanceof Error ? error : new Error(String(error))
      );

      throw error;
    }
  }

  /**
   * Ensure we have capacity to load another model
   */
  private async ensureCapacity(): Promise<void> {
    while (this.loadedModels.size >= this.maxLoadedModels) {
      // Find the oldest model that's not currently being used
      let oldest: LoadedModel | null = null;
      let oldestTime = Infinity;

      for (const [, model] of this.loadedModels) {
        if (model.lastUsedAt.getTime() < oldestTime) {
          oldest = model;
          oldestTime = model.lastUsedAt.getTime();
        }
      }

      if (oldest) {
        await this.unloadModel(oldest.name);
      } else {
        throw new Error('No models available to unload');
      }
    }
  }

  /**
   * Unload a model
   */
  async unloadModel(modelName: string): Promise<void> {
    const loaded = this.loadedModels.get(modelName);
    if (!loaded) return;

    this.emit('modelUnloading', { name: modelName });

    await this.runnerManager.kill(loaded.runner.id);
    this.loadedModels.delete(modelName);

    this.emit('modelUnloaded', { name: modelName });
  }

  /**
   * Resolve pending requests for a model
   */
  private resolvePendingRequests(
    modelName: string,
    loaded: LoadedModel
  ): void {
    const pending = this.pendingRequests.get(modelName) || [];
    this.pendingRequests.delete(modelName);

    for (const request of pending) {
      loaded.requestCount++;
      loaded.lastUsedAt = new Date();
      request.resolve(loaded);
    }
  }

  /**
   * Reject pending requests for a model
   */
  private rejectPendingRequests(modelName: string, error: Error): void {
    const pending = this.pendingRequests.get(modelName) || [];
    this.pendingRequests.delete(modelName);

    for (const request of pending) {
      request.reject(error);
    }
  }

  /**
   * Handle runner exit
   */
  private handleRunnerExit(data: {
    id: string;
    code: number | null;
    signal: string | null;
    lastError: string;
  }): void {
    // Find and remove the model associated with this runner
    for (const [name, model] of this.loadedModels) {
      if (model.runner.id === data.id) {
        this.loadedModels.delete(name);
        this.emit('modelUnloaded', {
          name,
          reason: 'runner_exit',
          code: data.code,
          error: data.lastError,
        });
        break;
      }
    }
  }

  /**
   * Start the expiration checker
   */
  private startExpirationChecker(): void {
    this.expirationTimer = setInterval(() => {
      this.checkExpirations();
    }, 5000); // Check every 5 seconds
  }

  /**
   * Check and expire old models
   */
  private checkExpirations(): void {
    const now = Date.now();

    for (const [name, model] of this.loadedModels) {
      if (model.expiresAt <= now) {
        this.unloadModel(name).catch((err) => {
          this.emit('error', {
            type: 'expiration',
            name,
            error: err,
          });
        });
      }
    }
  }

  /**
   * Get list of currently loaded models
   */
  getLoadedModels(): Array<{
    name: string;
    expiresAt: Date;
    requestCount: number;
    sizeVram: number;
  }> {
    return Array.from(this.loadedModels.values()).map((model) => ({
      name: model.name,
      expiresAt: new Date(model.expiresAt),
      requestCount: model.requestCount,
      sizeVram: 0, // TODO: Track VRAM usage
    }));
  }

  /**
   * Check if a model is loaded
   */
  isLoaded(modelName: string): boolean {
    const loaded = this.loadedModels.get(modelName);
    return loaded !== undefined && loaded.runner.status === 'ready';
  }

  /**
   * Extend the keep-alive for a model
   */
  extendKeepAlive(modelName: string, keepAlive?: Duration): void {
    const loaded = this.loadedModels.get(modelName);
    if (loaded) {
      const keepAliveMs = parseDuration(keepAlive) || this.defaultKeepAlive;
      loaded.expiresAt = Date.now() + keepAliveMs;
    }
  }

  /**
   * Get the model manager
   */
  getModelManager(): ModelManager {
    return this.modelManager;
  }

  /**
   * Get the runner manager
   */
  getRunnerManager(): RunnerManager {
    return this.runnerManager;
  }

  /**
   * Shutdown the scheduler
   */
  async shutdown(): Promise<void> {
    // Stop expiration checker
    if (this.expirationTimer) {
      clearInterval(this.expirationTimer);
      this.expirationTimer = null;
    }

    // Unload all models
    const names = Array.from(this.loadedModels.keys());
    await Promise.all(names.map((name) => this.unloadModel(name)));

    // Kill any remaining runners
    await this.runnerManager.killAll();

    this.emit('shutdown');
  }
}
