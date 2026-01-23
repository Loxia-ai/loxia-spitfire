/**
 * Runner HTTP Client
 * Communicates with the llama.cpp runner process via HTTP
 */

import type { HealthResponse } from '../types/index.js';

// Runner-specific completion request (different from API types)
export interface RunnerCompletionRequest {
  prompt: string;
  images?: string[]; // Base64 encoded
  grammar?: string;
  cachePrompt?: boolean;
  shift?: boolean;
  truncate?: boolean;

  // Sampling parameters
  temperature?: number;
  topK?: number;
  topP?: number;
  minP?: number;
  typicalP?: number;
  repeatPenalty?: number;
  repeatLastN?: number;
  presencePenalty?: number;
  frequencyPenalty?: number;
  seed?: number;
  stop?: string[];
  nPredict?: number;
  nKeep?: number;

  // Logprobs
  logprobs?: boolean;
  topLogprobs?: number;

  // Stream
  stream?: boolean;
}

// Runner-specific completion response
export interface RunnerCompletionResponse {
  content: string;
  stop: boolean;
  stopReason?: 'stop' | 'length' | 'eos';

  // Metrics
  promptEvalCount?: number;
  promptEvalDuration?: number; // nanoseconds
  evalCount?: number;
  evalDuration?: number; // nanoseconds

  // Logprobs
  completionProbabilities?: TokenProbability[];
}

export interface TokenProbability {
  content: string;
  probs: Array<{
    tok_str: string;
    prob: number;
  }>;
}

export interface LoadRequest {
  operation: 'fit' | 'alloc' | 'commit' | 'close';
  loraPath?: string[];
  parallel?: number;
  batchSize?: number;
  flashAttention?: boolean;
  kvSize?: number;
  kvCacheType?: string;
  numThreads?: number;
  gpuLayers?: GPULayers[];
  multiUserCache?: boolean;
  projectorPath?: string;
  mainGpu?: number;
  useMmap?: boolean;
}

export interface GPULayers {
  deviceId: string;
  layers: number[];
}

export interface LoadResponse {
  success: boolean;
  memory?: BackendMemory;
}

export interface BackendMemory {
  inputWeights: number;
  cpu: DeviceMemory;
  gpus: DeviceMemory[];
}

export interface DeviceMemory {
  name: string;
  deviceId?: string;
  weights: number[];
  cache: number[];
  graph: number;
}

export interface EmbeddingRequest {
  content: string;
}

export interface EmbeddingResponse {
  embedding: number[];
  promptEvalCount: number;
}

export interface RunnerClientOptions {
  host: string;
  port: number;
  timeout?: number;
}

export class RunnerClient {
  private baseUrl: string;
  private timeout: number;

  constructor(options: RunnerClientOptions) {
    this.baseUrl = `http://${options.host}:${options.port}`;
    this.timeout = options.timeout || 120000;
  }

  get url(): string {
    return this.baseUrl;
  }

  /**
   * Check runner health status
   */
  async health(): Promise<HealthResponse> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);

    try {
      const response = await fetch(`${this.baseUrl}/health`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        return { status: 'error' };
      }

      const data = (await response.json()) as {
        status: number;
        progress?: number;
        slots_idle?: number;
        slots_processing?: number;
      };

      // Map numeric status to string
      const statusMap: Record<number, 'ok' | 'loading' | 'error'> = {
        0: 'ok', // ServerStatusReady
        1: 'ok', // ServerStatusNoSlotsAvailable (still technically OK)
        2: 'ok', // ServerStatusLaunched
        3: 'loading', // ServerStatusLoadingModel
        4: 'error', // ServerStatusNotResponding
        5: 'error', // ServerStatusError
      };

      return {
        status: statusMap[data.status] || 'error',
        progress: data.progress,
        slotsIdle: data.slots_idle,
        slotsProcessing: data.slots_processing,
      };
    } catch (error) {
      clearTimeout(timeoutId);
      return { status: 'error' };
    }
  }

  /**
   * Wait until the runner is responding
   */
  async waitUntilLaunched(timeoutMs = 30000): Promise<void> {
    const start = Date.now();
    while (Date.now() - start < timeoutMs) {
      const health = await this.health();
      if (health.status !== 'error') {
        return;
      }
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
    throw new Error('Runner failed to launch within timeout');
  }

  /**
   * Wait until the runner is ready (model loaded)
   */
  async waitUntilReady(
    timeoutMs = 300000,
    onProgress?: (progress: number) => void
  ): Promise<void> {
    const start = Date.now();
    let lastProgress = 0;

    while (Date.now() - start < timeoutMs) {
      const health = await this.health();

      if (health.status === 'ok') {
        return;
      }

      if (health.status === 'loading' && health.progress !== undefined) {
        if (health.progress > lastProgress) {
          lastProgress = health.progress;
          onProgress?.(health.progress);
        }
      }

      if (health.status === 'error') {
        throw new Error('Runner entered error state');
      }

      await new Promise((resolve) => setTimeout(resolve, 250));
    }

    throw new Error('Runner failed to become ready within timeout');
  }

  /**
   * Load a model
   */
  async load(request: LoadRequest): Promise<LoadResponse> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(`${this.baseUrl}/load`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          operation: this.operationToNumber(request.operation),
          lora_path: request.loraPath,
          parallel: request.parallel,
          batch_size: request.batchSize,
          flash_attention: request.flashAttention,
          kv_size: request.kvSize,
          kv_cache_type: request.kvCacheType,
          num_threads: request.numThreads,
          gpu_layers: request.gpuLayers,
          multi_user_cache: request.multiUserCache,
          projector_path: request.projectorPath,
          main_gpu: request.mainGpu,
          use_mmap: request.useMmap,
        }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const body = await response.text();
        throw new Error(`Load failed: ${body}`);
      }

      const data = (await response.json()) as { Success: boolean };
      return {
        success: data.Success,
      };
    } catch (error) {
      clearTimeout(timeoutId);
      throw error;
    }
  }

  private operationToNumber(op: string): number {
    const map: Record<string, number> = {
      fit: 0,
      alloc: 1,
      commit: 2,
      close: 3,
    };
    return map[op] || 0;
  }

  /**
   * Run completion (streaming)
   */
  async *completion(
    request: RunnerCompletionRequest
  ): AsyncGenerator<RunnerCompletionResponse, void, unknown> {
    const controller = new AbortController();

    try {
      const response = await fetch(`${this.baseUrl}/completion`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: request.prompt,
          images: request.images?.map((img, idx) => ({
            data: Buffer.from(img, 'base64'),
            id: idx,
          })),
          grammar: request.grammar,
          shift: request.shift,
          truncate: request.truncate,
          logprobs: request.logprobs,
          top_logprobs: request.topLogprobs,
          options: {
            num_keep: request.nKeep,
            seed: request.seed,
            num_predict: request.nPredict,
            top_k: request.topK,
            top_p: request.topP,
            min_p: request.minP,
            typical_p: request.typicalP,
            repeat_last_n: request.repeatLastN,
            temperature: request.temperature,
            repeat_penalty: request.repeatPenalty,
            presence_penalty: request.presencePenalty,
            frequency_penalty: request.frequencyPenalty,
            stop: request.stop,
          },
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        const body = await response.text();
        throw new Error(`Completion failed: ${body}`);
      }

      if (!response.body) {
        throw new Error('No response body');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.trim()) continue;

          let data = line;
          if (data.startsWith('data: ')) {
            data = data.slice(6);
          }

          try {
            const parsed = JSON.parse(data) as {
              content?: string;
              done?: boolean;
              done_reason?: number;
              prompt_eval_count?: number;
              prompt_eval_duration?: number;
              eval_count?: number;
              eval_duration?: number;
            };

            const completionResponse: RunnerCompletionResponse = {
              content: parsed.content || '',
              stop: parsed.done || false,
              stopReason: this.mapDoneReason(parsed.done_reason),
              promptEvalCount: parsed.prompt_eval_count,
              promptEvalDuration: parsed.prompt_eval_duration,
              evalCount: parsed.eval_count,
              evalDuration: parsed.eval_duration,
            };

            yield completionResponse;

            if (parsed.done) {
              return;
            }
          } catch {
            // Skip malformed JSON lines
          }
        }
      }
    } finally {
      controller.abort();
    }
  }

  private mapDoneReason(
    reason: number | undefined
  ): 'stop' | 'length' | 'eos' | undefined {
    if (reason === undefined) return undefined;
    const map: Record<number, 'stop' | 'length' | 'eos'> = {
      0: 'stop',
      1: 'length',
    };
    return map[reason];
  }

  /**
   * Get embeddings
   */
  async embedding(content: string): Promise<EmbeddingResponse> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(`${this.baseUrl}/embedding`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const body = await response.text();
        throw new Error(`Embedding failed: ${body}`);
      }

      const data = (await response.json()) as {
        embedding: number[];
        prompt_eval_count: number;
      };

      return {
        embedding: data.embedding,
        promptEvalCount: data.prompt_eval_count,
      };
    } catch (error) {
      clearTimeout(timeoutId);
      throw error;
    }
  }

  /**
   * Close the connection (for cleanup)
   */
  async close(): Promise<void> {
    try {
      await this.load({ operation: 'close' });
    } catch {
      // Ignore errors on close
    }
  }
}
