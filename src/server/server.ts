/**
 * Spitfire HTTP Server
 * Ollama-compatible REST API
 */

import Fastify, { type FastifyInstance, type FastifyReply, type FastifyRequest } from 'fastify';
import { Scheduler, type SchedulerOptions } from '../scheduler/index.js';
import type {
  GenerateRequest,
  GenerateResponse,
  ChatRequest,
  ChatResponse,
  EmbedRequest,
  EmbedResponse,
  ListResponse,
  ShowRequest,
  ShowResponse,
  ProcessResponse,
  Message,
} from '../types/index.js';

export interface ServerOptions extends SchedulerOptions {
  host?: string;
  port?: number;
  cors?: boolean;
}

export class SpitfireServer {
  private app: FastifyInstance;
  private scheduler: Scheduler;
  private options: ServerOptions;

  constructor(options: ServerOptions = {}) {
    this.options = {
      host: options.host || '127.0.0.1',
      port: options.port || 11434,
      cors: options.cors ?? true,
      ...options,
    };

    this.scheduler = new Scheduler(options);

    this.app = Fastify({
      logger: false,
    });

    this.setupRoutes();
    this.setupErrorHandling();
  }

  private setupRoutes(): void {
    // Health check
    this.app.get('/', async () => {
      return { status: 'Spitfire is running' };
    });

    // Generate endpoint
    this.app.post('/api/generate', async (request, reply) => {
      return this.handleGenerate(request as FastifyRequest<{ Body: GenerateRequest }>, reply);
    });

    // Chat endpoint
    this.app.post('/api/chat', async (request, reply) => {
      return this.handleChat(request as FastifyRequest<{ Body: ChatRequest }>, reply);
    });

    // Embed endpoint
    this.app.post('/api/embed', async (request, reply) => {
      return this.handleEmbed(request as FastifyRequest<{ Body: EmbedRequest }>, reply);
    });

    // List models
    this.app.get('/api/tags', async () => {
      return this.handleList();
    });

    this.app.get('/api/list', async () => {
      return this.handleList();
    });

    // Show model
    this.app.post('/api/show', async (request) => {
      return this.handleShow(request as FastifyRequest<{ Body: ShowRequest }>);
    });

    // Running models
    this.app.get('/api/ps', async () => {
      return this.handlePs();
    });

    // Version
    this.app.get('/api/version', async () => {
      return { version: '0.1.0' };
    });
  }

  private setupErrorHandling(): void {
    this.app.setErrorHandler((error, _request, reply) => {
      const statusCode = error.statusCode || 500;
      reply.status(statusCode).send({
        error: error.message || 'Internal Server Error',
      });
    });
  }

  /**
   * Handle /api/generate endpoint
   */
  private async handleGenerate(
    request: FastifyRequest<{ Body: GenerateRequest }>,
    reply: FastifyReply
  ): Promise<void> {
    const body = request.body;
    const stream = body.stream !== false; // Default to streaming

    try {
      const loaded = await this.scheduler.getRunner(body.model, body.keepAlive);
      const runner = loaded.runner;

      if (stream) {
        reply.raw.writeHead(200, {
          'Content-Type': 'application/x-ndjson',
          'Transfer-Encoding': 'chunked',
          'Cache-Control': 'no-cache',
        });

        const startTime = Date.now();
        let fullResponse = '';
        let promptEvalCount = 0;
        let evalCount = 0;

        for await (const chunk of runner.client.completion({
          prompt: body.prompt,
          images: body.images,
          temperature: body.options?.temperature,
          topK: body.options?.topK,
          topP: body.options?.topP,
          seed: body.options?.seed,
          stop: body.options?.stop,
          nPredict: body.options?.numPredict,
          repeatPenalty: body.options?.repeatPenalty,
        })) {
          fullResponse += chunk.content;

          const response: GenerateResponse = {
            model: body.model,
            createdAt: new Date().toISOString(),
            response: chunk.content,
            done: chunk.stop,
          };

          if (chunk.stop) {
            response.doneReason = chunk.stopReason || 'stop';
            response.totalDuration = (Date.now() - startTime) * 1_000_000; // ns
            response.promptEvalCount = chunk.promptEvalCount || promptEvalCount;
            response.evalCount = chunk.evalCount || evalCount;
            response.evalDuration = chunk.evalDuration;
            response.promptEvalDuration = chunk.promptEvalDuration;
          } else {
            if (chunk.promptEvalCount) promptEvalCount = chunk.promptEvalCount;
            if (chunk.evalCount) evalCount = chunk.evalCount;
          }

          reply.raw.write(JSON.stringify(response) + '\n');
        }

        reply.raw.end();
      } else {
        // Non-streaming
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
          prompt: body.prompt,
          images: body.images,
          temperature: body.options?.temperature,
          topK: body.options?.topK,
          topP: body.options?.topP,
          seed: body.options?.seed,
          stop: body.options?.stop,
          nPredict: body.options?.numPredict,
          repeatPenalty: body.options?.repeatPenalty,
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

        const response: GenerateResponse = {
          model: body.model,
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

        return reply.send(response);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      reply.status(500).send({ error: message });
    }
  }

  /**
   * Handle /api/chat endpoint
   */
  private async handleChat(
    request: FastifyRequest<{ Body: ChatRequest }>,
    reply: FastifyReply
  ): Promise<void> {
    const body = request.body;
    const stream = body.stream !== false;

    try {
      const loaded = await this.scheduler.getRunner(body.model, body.keepAlive);
      const runner = loaded.runner;

      // Build prompt from messages
      const prompt = this.buildChatPrompt(body.messages, loaded.modelInfo.template);

      if (stream) {
        reply.raw.writeHead(200, {
          'Content-Type': 'application/x-ndjson',
          'Transfer-Encoding': 'chunked',
          'Cache-Control': 'no-cache',
        });

        const startTime = Date.now();
        let fullContent = '';

        for await (const chunk of runner.client.completion({
          prompt,
          temperature: body.options?.temperature,
          topK: body.options?.topK,
          topP: body.options?.topP,
          seed: body.options?.seed,
          stop: body.options?.stop,
          nPredict: body.options?.numPredict,
          repeatPenalty: body.options?.repeatPenalty,
        })) {
          fullContent += chunk.content;

          const response: ChatResponse = {
            model: body.model,
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

          reply.raw.write(JSON.stringify(response) + '\n');
        }

        reply.raw.end();
      } else {
        // Non-streaming
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
          temperature: body.options?.temperature,
          topK: body.options?.topK,
          topP: body.options?.topP,
          seed: body.options?.seed,
          stop: body.options?.stop,
          nPredict: body.options?.numPredict,
          repeatPenalty: body.options?.repeatPenalty,
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

        const response: ChatResponse = {
          model: body.model,
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

        return reply.send(response);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      reply.status(500).send({ error: message });
    }
  }

  /**
   * Build chat prompt from messages
   */
  private buildChatPrompt(messages: Message[], _template?: string): string {
    // Simple default template if none provided
    // TODO: Implement proper template rendering using _template
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
   * Handle /api/embed endpoint
   */
  private async handleEmbed(
    request: FastifyRequest<{ Body: EmbedRequest }>,
    reply: FastifyReply
  ): Promise<FastifyReply> {
    const body = request.body;

    try {
      const loaded = await this.scheduler.getRunner(body.model, body.keepAlive);
      const runner = loaded.runner;

      const inputs = Array.isArray(body.input) ? body.input : [body.input];
      const embeddings: number[][] = [];

      const startTime = Date.now();
      let totalPromptEvalCount = 0;

      for (const input of inputs) {
        const result = await runner.client.embedding(input);
        embeddings.push(result.embedding);
        totalPromptEvalCount += result.promptEvalCount;
      }

      const response: EmbedResponse = {
        model: body.model,
        embeddings,
        totalDuration: (Date.now() - startTime) * 1_000_000,
        promptEvalCount: totalPromptEvalCount,
      };

      return reply.send(response);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      return reply.status(500).send({ error: message });
    }
  }

  /**
   * Handle /api/tags and /api/list endpoints
   */
  private async handleList(): Promise<ListResponse> {
    const modelManager = this.scheduler.getModelManager();
    const models = await modelManager.list();

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
   * Handle /api/show endpoint
   */
  private async handleShow(
    request: FastifyRequest<{ Body: ShowRequest }>
  ): Promise<ShowResponse | { error: string }> {
    const body = request.body;
    const modelManager = this.scheduler.getModelManager();

    const info = await modelManager.show(body.model);
    if (!info) {
      return { error: 'Model not found' };
    }

    return info;
  }

  /**
   * Handle /api/ps endpoint
   */
  private handlePs(): ProcessResponse {
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
   * Get the scheduler
   */
  getScheduler(): Scheduler {
    return this.scheduler;
  }

  /**
   * Start the server
   */
  async start(): Promise<string> {
    const address = await this.app.listen({
      host: this.options.host,
      port: this.options.port,
    });
    return address;
  }

  /**
   * Stop the server
   */
  async stop(): Promise<void> {
    await this.app.close();
    await this.scheduler.shutdown();
  }
}

/**
 * Create and start a Spitfire server
 */
export async function createServer(options: ServerOptions = {}): Promise<SpitfireServer> {
  const server = new SpitfireServer(options);
  await server.start();
  return server;
}
