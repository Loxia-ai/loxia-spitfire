/**
 * Runner Manager
 * Manages spawning and lifecycle of llama.cpp runner processes
 */

import { spawn, type ChildProcess } from 'child_process';
import { EventEmitter } from 'events';
import { platform } from 'os';
import { RunnerClient } from './client.js';
import type { RunnerConfig, SystemInfo } from '../types/index.js';
import { getRandomPort, getSystemInfo } from '../utils/index.js';

// String-based status for simplicity
export type RunnerInstanceStatus =
  | 'starting'
  | 'loading'
  | 'ready'
  | 'busy'
  | 'error'
  | 'stopped';

export interface RunnerInstance {
  id: string;
  pid: number;
  port: number;
  status: RunnerInstanceStatus;
  modelPath: string;
  client: RunnerClient;
  process: ChildProcess;
  startedAt: Date;
  lastUsedAt: Date;
}

export interface SpawnOptions {
  modelPath: string;
  ollamaEngine?: boolean;
  gpuLibPaths?: string[];
  env?: Record<string, string>;
}

export class RunnerManager extends EventEmitter {
  private config: RunnerConfig;
  private runners: Map<string, RunnerInstance> = new Map();
  private runnerPath: string | null = null;

  constructor(config: Partial<RunnerConfig> = {}) {
    super();
    this.config = {
      modelsPath: config.modelsPath || '',
      host: config.host || '127.0.0.1',
      port: config.port || 0,
      maxLoadedModels: config.maxLoadedModels || 1,
      gpuOverhead: config.gpuOverhead || 0,
      loadTimeout: config.loadTimeout || 300000,
      requestTimeout: config.requestTimeout || 120000,
      keepAliveTimeout: config.keepAliveTimeout || 300000,
      flashAttention: config.flashAttention ?? true,
      kvCacheType: config.kvCacheType || 'f16',
      runnerPath: config.runnerPath,
      cudaVisibleDevices: config.cudaVisibleDevices,
      rocmVisibleDevices: config.rocmVisibleDevices,
    };
  }

  /**
   * Find the runner executable
   */
  async findRunner(): Promise<string> {
    if (this.runnerPath) {
      return this.runnerPath;
    }

    // Check if custom path is configured
    if (this.config.runnerPath) {
      this.runnerPath = this.config.runnerPath;
      return this.runnerPath;
    }

    // Try to find ollama executable (which includes the runner)
    const ollamaPath = await this.findOllama();
    if (ollamaPath) {
      this.runnerPath = ollamaPath;
      return this.runnerPath;
    }

    throw new Error(
      'Could not find runner executable. Please ensure Ollama is installed or provide a custom runner path.'
    );
  }

  /**
   * Find the ollama executable in system PATH
   */
  private async findOllama(): Promise<string | null> {
    const { execSync } = await import('child_process');
    const isWindows = platform() === 'win32';

    try {
      const cmd = isWindows ? 'where ollama' : 'which ollama';
      const result = execSync(cmd, { encoding: 'utf-8' }).trim();
      // Take first result if multiple
      const firstPath = result.split('\n')[0].trim();
      return firstPath || null;
    } catch {
      return null;
    }
  }

  /**
   * Spawn a new runner process
   */
  async spawn(options: SpawnOptions): Promise<RunnerInstance> {
    const runnerPath = await this.findRunner();
    const port = this.config.port || getRandomPort();
    const id = `runner-${Date.now()}-${port}`;

    const args = ['runner'];

    if (options.ollamaEngine) {
      args.push('--ollama-engine');
    }

    if (options.modelPath) {
      args.push('--model', options.modelPath);
    }

    args.push('--port', port.toString());

    // Set up environment
    const env: Record<string, string> = { ...process.env } as Record<
      string,
      string
    >;

    // Set library paths
    const pathEnv = this.getLibraryPathEnv();
    if (options.gpuLibPaths && options.gpuLibPaths.length > 0) {
      const separator = platform() === 'win32' ? ';' : ':';
      const currentPath = env[pathEnv] || '';
      env[pathEnv] = options.gpuLibPaths.join(separator) + separator + currentPath;
    }

    // GPU visibility
    if (this.config.cudaVisibleDevices) {
      env['CUDA_VISIBLE_DEVICES'] = this.config.cudaVisibleDevices;
    }
    if (this.config.rocmVisibleDevices) {
      env['ROCR_VISIBLE_DEVICES'] = this.config.rocmVisibleDevices;
    }

    // Additional env vars
    if (options.env) {
      Object.assign(env, options.env);
    }

    this.emit('spawning', { id, modelPath: options.modelPath, port });

    const child = spawn(runnerPath, args, {
      env,
      stdio: ['ignore', 'pipe', 'pipe'],
      detached: false,
    });

    const client = new RunnerClient({
      host: this.config.host,
      port,
      timeout: this.config.requestTimeout,
    });

    const instance: RunnerInstance = {
      id,
      pid: child.pid || -1,
      port,
      status: 'starting',
      modelPath: options.modelPath,
      client,
      process: child,
      startedAt: new Date(),
      lastUsedAt: new Date(),
    };

    // Handle process output
    let lastError = '';
    child.stdout?.on('data', (data: Buffer) => {
      const msg = data.toString();
      this.emit('stdout', { id, message: msg });
    });

    child.stderr?.on('data', (data: Buffer) => {
      const msg = data.toString();
      lastError = msg;
      this.emit('stderr', { id, message: msg });
    });

    // Handle process exit
    child.on('exit', (code, signal) => {
      instance.status = 'stopped';
      this.emit('exit', { id, code, signal, lastError });
      this.runners.delete(id);
    });

    child.on('error', (error) => {
      instance.status = 'error';
      this.emit('error', { id, error });
    });

    this.runners.set(id, instance);

    // Wait for runner to be responsive
    try {
      await client.waitUntilLaunched(30000);
      instance.status = 'loading';
      this.emit('launched', { id, port, pid: instance.pid });
    } catch (error) {
      await this.kill(id);
      throw new Error(`Runner failed to launch: ${lastError || error}`);
    }

    return instance;
  }

  /**
   * Get the library path environment variable name for the current platform
   */
  private getLibraryPathEnv(): string {
    switch (platform()) {
      case 'win32':
        return 'PATH';
      case 'darwin':
        return 'DYLD_LIBRARY_PATH';
      default:
        return 'LD_LIBRARY_PATH';
    }
  }

  /**
   * Load a model on a runner
   */
  async load(
    instance: RunnerInstance,
    options: {
      parallel?: number;
      batchSize?: number;
      numCtx?: number;
      numThreads?: number;
      flashAttention?: boolean;
      kvCacheType?: string;
    } = {}
  ): Promise<void> {
    instance.status = 'loading';
    instance.lastUsedAt = new Date();

    const loadRequest = {
      operation: 'commit' as const,
      parallel: options.parallel || 1,
      batchSize: options.batchSize || 512,
      kvSize: (options.numCtx || 2048) * (options.parallel || 1),
      numThreads: options.numThreads,
      flashAttention: options.flashAttention ?? this.config.flashAttention,
      kvCacheType: options.kvCacheType || this.config.kvCacheType,
      useMmap: true,
    };

    this.emit('loading', { id: instance.id, modelPath: instance.modelPath });

    const response = await instance.client.load(loadRequest);

    if (!response.success) {
      instance.status = 'error';
      throw new Error('Failed to load model');
    }

    // Wait until model is fully loaded
    await instance.client.waitUntilReady(this.config.loadTimeout, (progress) => {
      this.emit('loadProgress', { id: instance.id, progress });
    });

    instance.status = 'ready';
    this.emit('loaded', { id: instance.id, modelPath: instance.modelPath });
  }

  /**
   * Get a runner instance by ID
   */
  get(id: string): RunnerInstance | undefined {
    return this.runners.get(id);
  }

  /**
   * Get all runner instances
   */
  getAll(): RunnerInstance[] {
    return Array.from(this.runners.values());
  }

  /**
   * Find a runner by model path
   */
  findByModel(modelPath: string): RunnerInstance | undefined {
    return Array.from(this.runners.values()).find(
      (r) => r.modelPath === modelPath && r.status === 'ready'
    );
  }

  /**
   * Update last used timestamp
   */
  touch(id: string): void {
    const instance = this.runners.get(id);
    if (instance) {
      instance.lastUsedAt = new Date();
    }
  }

  /**
   * Kill a runner process
   */
  async kill(id: string): Promise<void> {
    const instance = this.runners.get(id);
    if (!instance) return;

    this.emit('killing', { id });

    try {
      // Try graceful close first
      await instance.client.close();
    } catch {
      // Ignore errors
    }

    // Force kill if still running
    if (instance.process && !instance.process.killed) {
      instance.process.kill('SIGTERM');

      // Give it a moment, then force kill
      await new Promise((resolve) => setTimeout(resolve, 1000));
      if (!instance.process.killed) {
        instance.process.kill('SIGKILL');
      }
    }

    this.runners.delete(id);
    this.emit('killed', { id });
  }

  /**
   * Kill all runners
   */
  async killAll(): Promise<void> {
    const ids = Array.from(this.runners.keys());
    await Promise.all(ids.map((id) => this.kill(id)));
  }

  /**
   * Get system info
   */
  getSystemInfo(): SystemInfo {
    return getSystemInfo();
  }

  /**
   * Check if any runners are running
   */
  hasRunningRunners(): boolean {
    return this.runners.size > 0;
  }

  /**
   * Get count of running runners
   */
  get runnerCount(): number {
    return this.runners.size;
  }
}
