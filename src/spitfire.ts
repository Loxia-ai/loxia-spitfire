/**
 * Spitfire - Main API class
 * Provides a clean, promise-based API for running LLMs
 */

import { Scheduler, type SchedulerOptions } from './scheduler/index.js';
import { ModelManager } from './model/index.js';
import type {
  GenerateRequest,
  GenerateResponse,
  ChatRequest,
  ChatResponse,
  EmbedRequest,
  EmbedResponse,
  ListResponse,
  ShowResponse,
  ProcessResponse,
  Message,
  Duration,
} from './types/index.js';

export interface SpitfireOptions extends SchedulerOptions {
  // Additional options can be added here
}

export class Spitfire {
  private scheduler: Scheduler;
  private modelManager: ModelManager;

  constructor(options: SpitfireOptions = {}) {
    this.scheduler = new Scheduler(options);
    this.modelManager = this.scheduler.getModelManager();
  }

  /**
   * Generate text completion
   */
  async generate(request: GenerateRequest): Promise<GenerateResponse> {
    const loaded = await this.scheduler.getRunner(request.model, request.keepAlive);
    const runner = loaded.runner;

    const startTime = Date.now();
    let fullResponse = '';
    let finalMetrics: {
      promptEvalCount?: number;
      evalCount?: number;
      evalDuration?: number;
      promptEvalDuration?: number;
      doneReason?: string;
    } = {};

    for await (const chunk of runner.client.completion({
      prompt: request.prompt,
      images: request.images,
      temperature: request.options?.temperature,
      topK: request.options?.topK,
      topP: request.options?.topP,
      seed: request.options?.seed,
      stop: request.options?.stop,
      nPredict: request.options?.numPredict,
      repeatPenalty: request.options?.repeatPenalty,
    })) {
      fullResponse += chunk.content;
      if (chunk.stop) {
        finalMetrics = {
          promptEvalCount: chunk.promptEvalCount,
          evalCount: chunk.evalCount,
          evalDuration: chunk.evalDuration,
          promptEvalDuration: chunk.promptEvalDuration,
          doneReason: chunk.stopReason,
        };
      }
    }

    return {
      model: request.model,
      createdAt: new Date().toISOString(),
      response: fullResponse,
      done: true,
      doneReason: finalMetrics.doneReason || 'stop',
      totalDuration: (Date.now() - startTime) * 1_000_000,
      promptEvalCount: finalMetrics.promptEvalCount,
      evalCount: finalMetrics.evalCount,
      evalDuration: finalMetrics.evalDuration,
      promptEvalDuration: finalMetrics.promptEvalDuration,
    };
  }

  /**
   * Generate text completion with streaming
   */
  async *generateStream(
    request: GenerateRequest
  ): AsyncGenerator<GenerateResponse, void, unknown> {
    const loaded = await this.scheduler.getRunner(request.model, request.keepAlive);
    const runner = loaded.runner;

    const startTime = Date.now();

    for await (const chunk of runner.client.completion({
      prompt: request.prompt,
      images: request.images,
      temperature: request.options?.temperature,
      topK: request.options?.topK,
      topP: request.options?.topP,
      seed: request.options?.seed,
      stop: request.options?.stop,
      nPredict: request.options?.numPredict,
      repeatPenalty: request.options?.repeatPenalty,
    })) {
      const response: GenerateResponse = {
        model: request.model,
        createdAt: new Date().toISOString(),
        response: chunk.content,
        done: chunk.stop,
      };

      if (chunk.stop) {
        response.doneReason = chunk.stopReason || 'stop';
        response.totalDuration = (Date.now() - startTime) * 1_000_000;
        response.promptEvalCount = chunk.promptEvalCount;
        response.evalCount = chunk.evalCount;
        response.evalDuration = chunk.evalDuration;
        response.promptEvalDuration = chunk.promptEvalDuration;
      }

      yield response;
    }
  }

  /**
   * Chat completion
   */
  async chat(request: ChatRequest): Promise<ChatResponse> {
    const loaded = await this.scheduler.getRunner(request.model, request.keepAlive);
    const runner = loaded.runner;

    const prompt = this.buildChatPrompt(request.messages);
    const startTime = Date.now();
    let fullContent = '';
    let finalMetrics: {
      promptEvalCount?: number;
      evalCount?: number;
      evalDuration?: number;
      promptEvalDuration?: number;
      doneReason?: string;
    } = {};

    for await (const chunk of runner.client.completion({
      prompt,
      temperature: request.options?.temperature,
      topK: request.options?.topK,
      topP: request.options?.topP,
      seed: request.options?.seed,
      stop: request.options?.stop,
      nPredict: request.options?.numPredict,
      repeatPenalty: request.options?.repeatPenalty,
    })) {
      fullContent += chunk.content;
      if (chunk.stop) {
        finalMetrics = {
          promptEvalCount: chunk.promptEvalCount,
          evalCount: chunk.evalCount,
          evalDuration: chunk.evalDuration,
          promptEvalDuration: chunk.promptEvalDuration,
          doneReason: chunk.stopReason,
        };
      }
    }

    return {
      model: request.model,
      createdAt: new Date().toISOString(),
      message: {
        role: 'assistant',
        content: fullContent,
      },
      done: true,
      doneReason: finalMetrics.doneReason || 'stop',
      totalDuration: (Date.now() - startTime) * 1_000_000,
      promptEvalCount: finalMetrics.promptEvalCount,
      evalCount: finalMetrics.evalCount,
      evalDuration: finalMetrics.evalDuration,
      promptEvalDuration: finalMetrics.promptEvalDuration,
    };
  }

  /**
   * Chat completion with streaming
   */
  async *chatStream(
    request: ChatRequest
  ): AsyncGenerator<ChatResponse, void, unknown> {
    const loaded = await this.scheduler.getRunner(request.model, request.keepAlive);
    const runner = loaded.runner;

    const prompt = this.buildChatPrompt(request.messages);
    const startTime = Date.now();

    for await (const chunk of runner.client.completion({
      prompt,
      temperature: request.options?.temperature,
      topK: request.options?.topK,
      topP: request.options?.topP,
      seed: request.options?.seed,
      stop: request.options?.stop,
      nPredict: request.options?.numPredict,
      repeatPenalty: request.options?.repeatPenalty,
    })) {
      const response: ChatResponse = {
        model: request.model,
        createdAt: new Date().toISOString(),
        message: {
          role: 'assistant',
          content: chunk.content,
        },
        done: chunk.stop,
      };

      if (chunk.stop) {
        response.doneReason = chunk.stopReason || 'stop';
        response.totalDuration = (Date.now() - startTime) * 1_000_000;
        response.promptEvalCount = chunk.promptEvalCount;
        response.evalCount = chunk.evalCount;
        response.evalDuration = chunk.evalDuration;
        response.promptEvalDuration = chunk.promptEvalDuration;
      }

      yield response;
    }
  }

  /**
   * Build chat prompt from messages
   */
  private buildChatPrompt(messages: Message[]): string {
    let prompt = '';

    for (const msg of messages) {
      switch (msg.role) {
        case 'system':
          prompt += `System: ${msg.content}\n\n`;
          break;
        case 'user':
          prompt += `User: ${msg.content}\n\n`;
          break;
        case 'assistant':
          prompt += `Assistant: ${msg.content}\n\n`;
          break;
      }
    }

    prompt += 'Assistant: ';
    return prompt;
  }

  /**
   * Get embeddings
   */
  async embed(request: EmbedRequest): Promise<EmbedResponse> {
    const loaded = await this.scheduler.getRunner(request.model, request.keepAlive);
    const runner = loaded.runner;

    const inputs = Array.isArray(request.input) ? request.input : [request.input];
    const embeddings: number[][] = [];

    const startTime = Date.now();
    let totalPromptEvalCount = 0;

    for (const input of inputs) {
      const result = await runner.client.embedding(input);
      embeddings.push(result.embedding);
      totalPromptEvalCount += result.promptEvalCount;
    }

    return {
      model: request.model,
      embeddings,
      totalDuration: (Date.now() - startTime) * 1_000_000,
      promptEvalCount: totalPromptEvalCount,
    };
  }

  /**
   * List installed models
   */
  async list(): Promise<ListResponse> {
    const models = await this.modelManager.list();

    return {
      models: models.map((m) => ({
        name: m.name,
        model: m.name,
        modifiedAt: m.modifiedAt.toISOString(),
        size: m.size,
        digest: m.digest,
      })),
    };
  }

  /**
   * Show model information
   */
  async show(model: string): Promise<ShowResponse | null> {
    const info = await this.modelManager.show(model);
    return info || null;
  }

  /**
   * Get running models
   */
  ps(): ProcessResponse {
    const loaded = this.scheduler.getLoadedModels();

    return {
      models: loaded.map((m) => ({
        name: m.name,
        model: m.name,
        size: 0,
        digest: '',
        expiresAt: m.expiresAt.toISOString(),
        sizeVram: m.sizeVram,
        contextLength: 0,
      })),
    };
  }

  /**
   * Check if a model exists
   */
  async exists(model: string): Promise<boolean> {
    return this.modelManager.exists(model);
  }

  /**
   * Preload a model (load it into memory)
   */
  async preload(model: string, keepAlive?: Duration): Promise<void> {
    await this.scheduler.getRunner(model, keepAlive);
  }

  /**
   * Unload a model from memory
   */
  async unload(model: string): Promise<void> {
    await this.scheduler.unloadModel(model);
  }

  /**
   * Shutdown and cleanup
   */
  async shutdown(): Promise<void> {
    await this.scheduler.shutdown();
  }
}
