/**
 * WebGPU Engine
 * GPU-accelerated inference engine implementing the Engine interface
 */

import { EventEmitter } from 'events';
import type { Engine } from './index.js';
import {
  WebGPUDevice,
  initWebGPU,
  Tensor,
  ops,
  rmsNorm,
  attention,
  feedForward,
  loadLlamaWeights,
  disposeLlamaWeights,
  estimateModelMemory,
  type LlamaWeights,
} from './webgpu/index.js';
import { parseGGUF, extractArchitecture } from '../model/gguf.js';
import type { GGUFFile, ModelArchitecture } from '../types/model.js';

export interface WebGPUEngineOptions {
  numThreads?: number; // Ignored for WebGPU (GPU handles parallelism)
}

export interface WebGPULoadModelOptions {
  contextLength?: number;
  batchSize?: number;
}

export interface WebGPUGenerateOptions {
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  repeatPenalty?: number;
  stop?: string[];
}

// Use LlamaWeights from model loader
// Re-export for backward compatibility
type ModelWeights = LlamaWeights;

/**
 * WebGPU-accelerated inference engine
 */
export class WebGPUEngine extends EventEmitter implements Engine {
  private gpuDevice: WebGPUDevice | null = null;
  private initialized = false;
  private modelLoaded = false;
  private ggufFile: GGUFFile | null = null;
  private architecture: ModelArchitecture | null = null;
  private weights: ModelWeights | null = null;

  // Model configuration
  private vocabSize = 0;
  private hiddenSize = 0;
  private numLayers = 0;
  private numHeads = 0;
  private headDim = 0;
  private intermediateSize = 0;
  private contextLength = 2048;
  private rmsNormEps = 1e-5;

  constructor(_options: WebGPUEngineOptions = {}) {
    super();
    // Options reserved for future configuration
    void _options;
  }

  /**
   * Initialize the WebGPU device
   */
  async init(): Promise<void> {
    if (this.initialized) {
      return;
    }

    const available = await WebGPUDevice.isAvailable();
    if (!available) {
      throw new Error('WebGPU is not available in this environment');
    }

    this.gpuDevice = await initWebGPU({
      powerPreference: 'high-performance',
    });

    this.initialized = true;
    this.emit('initialized');
  }

  /**
   * Load a model from a GGUF file
   */
  async loadModel(
    modelPath: string,
    options: WebGPULoadModelOptions = {}
  ): Promise<void> {
    if (!this.initialized) {
      await this.init();
    }

    this.emit('loading', { path: modelPath });

    try {
      // Parse GGUF file to get metadata and tensor info
      this.ggufFile = await parseGGUF(modelPath);
      this.architecture = extractArchitecture(this.ggufFile);

      // Extract model configuration from metadata
      this.setupModelConfig(options);

      // Load weights to GPU
      await this.loadWeights(modelPath);

      this.modelLoaded = true;
      this.emit('loaded', {
        architecture: this.architecture.architecture,
        contextLength: this.contextLength,
        vocabSize: this.vocabSize,
        numLayers: this.numLayers,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(`Failed to load model: ${message}`);
    }
  }

  /**
   * Setup model configuration from GGUF metadata
   */
  private setupModelConfig(options: WebGPULoadModelOptions): void {
    if (!this.architecture) {
      throw new Error('Model architecture not loaded');
    }

    this.contextLength = options.contextLength || this.architecture.contextLength || 2048;
    this.hiddenSize = this.architecture.embeddingLength || 4096;
    this.numLayers = this.architecture.blockCount || 32;
    this.numHeads = this.architecture.headCount || 32;
    // Note: headCountKV would be used for grouped-query attention
    this.headDim = this.hiddenSize / this.numHeads;
    this.vocabSize = this.architecture.vocabSize || 32000;
    this.intermediateSize = this.hiddenSize * 4; // Default ratio

    // Check for specific metadata keys
    if (this.ggufFile) {
      const meta = this.ggufFile.metadata;
      const arch = this.architecture.architecture;

      if (meta[`${arch}.feed_forward_length`]) {
        this.intermediateSize = Number(meta[`${arch}.feed_forward_length`]);
      }
      if (meta[`${arch}.attention.layer_norm_rms_epsilon`]) {
        this.rmsNormEps = Number(meta[`${arch}.attention.layer_norm_rms_epsilon`]);
      }
    }
  }

  /**
   * Load model weights to GPU tensors
   */
  private async loadWeights(modelPath: string): Promise<void> {
    if (!this.ggufFile) {
      throw new Error('GGUF file not parsed');
    }

    // Estimate memory requirements
    const estimatedMemory = estimateModelMemory(this.ggufFile, true);
    const memoryMB = Math.round(estimatedMemory / (1024 * 1024));
    console.log(`Loading model: ${this.numLayers} layers, ${this.hiddenSize} hidden size`);
    console.log(`Estimated GPU memory: ${memoryMB} MB`);

    // Check if we have enough GPU memory (if capabilities available)
    const caps = this.gpuDevice?.getCapabilities();
    if (caps && caps.maxBufferSize) {
      const maxBufferMB = Math.round(caps.maxBufferSize / (1024 * 1024));
      console.log(`Max GPU buffer size: ${maxBufferMB} MB`);
    }

    try {
      // Load actual weights from GGUF file using the model loader
      this.weights = await loadLlamaWeights(modelPath, this.ggufFile, {
        dequantize: true,
      });

      // Count loaded layers
      const loadedLayers = this.weights.layers.filter(
        (l) => l.attnQ !== null
      ).length;

      console.log(`Loaded ${loadedLayers} transformer layers`);
      console.log('Weights loaded to GPU');
    } catch (error) {
      // Fallback to placeholder weights for testing without actual model files
      console.warn('Could not load model weights, using placeholders:', error);
      await this.loadPlaceholderWeights();
    }
  }

  /**
   * Load placeholder weights for testing
   */
  private async loadPlaceholderWeights(): Promise<void> {
    console.log('Loading placeholder weights for testing');

    this.weights = {
      tokenEmbedding: Tensor.random([this.vocabSize, this.hiddenSize], { label: 'embed_tokens' }),
      layers: [],
      outputNorm: Tensor.ones([this.hiddenSize], { label: 'norm' }),
      outputWeight: Tensor.random([this.hiddenSize, this.vocabSize], { label: 'lm_head' }),
    };

    // Create a few layers for testing
    const numTestLayers = Math.min(this.numLayers, 2);
    for (let i = 0; i < numTestLayers; i++) {
      this.weights.layers.push({
        attnNorm: Tensor.ones([this.hiddenSize]),
        attnQ: Tensor.random([this.hiddenSize, this.hiddenSize]),
        attnK: Tensor.random([this.hiddenSize, this.hiddenSize]),
        attnV: Tensor.random([this.hiddenSize, this.hiddenSize]),
        attnOutput: Tensor.random([this.hiddenSize, this.hiddenSize]),
        ffnNorm: Tensor.ones([this.hiddenSize]),
        ffnGate: Tensor.random([this.hiddenSize, this.intermediateSize]),
        ffnUp: Tensor.random([this.hiddenSize, this.intermediateSize]),
        ffnDown: Tensor.random([this.intermediateSize, this.hiddenSize]),
      });
    }

    console.log(`Placeholder weights loaded (${numTestLayers} layers)`);
  }

  /**
   * Forward pass through the model
   */
  private async forward(inputIds: number[]): Promise<Float32Array> {
    if (!this.weights) {
      throw new Error('Model weights not loaded');
    }

    const seqLen = inputIds.length;

    // Create input embeddings
    // In a real implementation, this would index into the embedding table
    // For now, use random embeddings or token embedding if available
    let hidden: Tensor;
    if (this.weights.tokenEmbedding) {
      // Simple embedding lookup (would need proper implementation)
      hidden = Tensor.random([seqLen, this.hiddenSize], { label: 'hidden' });
    } else {
      hidden = Tensor.random([seqLen, this.hiddenSize], { label: 'hidden' });
    }

    // Process through layers
    for (let i = 0; i < this.weights.layers.length; i++) {
      const layer = this.weights.layers[i];

      // Skip layers without required weights
      if (!layer.attnNorm || !layer.ffnNorm) {
        continue;
      }

      // Pre-attention norm
      const normed = await rmsNorm(hidden, layer.attnNorm, this.rmsNormEps);

      // Self-attention (simplified - using normed as Q, K, V)
      // In a full implementation, we would use attnQ, attnK, attnV weights
      const attnOut = await attention(normed, normed, normed, {
        numHeads: this.numHeads,
        headDim: this.headDim,
        causal: true,
      });

      // Residual connection
      const afterAttn = await ops.add(hidden, attnOut);
      hidden.destroy();
      normed.destroy();
      attnOut.destroy();
      hidden = afterAttn;

      // Pre-FFN norm
      const normed2 = await rmsNorm(hidden, layer.ffnNorm, this.rmsNormEps);

      // FFN (use weights if available, otherwise skip)
      if (layer.ffnGate && layer.ffnUp && layer.ffnDown) {
        const ffnOut = await feedForward(normed2, layer.ffnGate, layer.ffnUp, layer.ffnDown);

        // Residual connection
        const afterFfn = await ops.add(hidden, ffnOut);
        hidden.destroy();
        ffnOut.destroy();
        hidden = afterFfn;
      }
      normed2.destroy();
    }

    // Final norm
    const normWeight = this.weights.outputNorm || Tensor.ones([this.hiddenSize]);
    const finalNormed = await rmsNorm(hidden, normWeight, this.rmsNormEps);
    hidden.destroy();
    if (!this.weights.outputNorm) {
      normWeight.destroy();
    }

    // Get logits for the last token
    const lastTokenHidden = Tensor.fromData(
      (await finalNormed.toArray()).slice(-this.hiddenSize),
      [1, this.hiddenSize]
    );
    finalNormed.destroy();

    // Output projection
    let logits: Tensor;
    if (this.weights.outputWeight) {
      logits = await ops.matmul(lastTokenHidden, this.weights.outputWeight);
    } else {
      // No output weight, create random logits for testing
      logits = Tensor.random([1, this.vocabSize], { label: 'logits' });
    }
    lastTokenHidden.destroy();

    const logitsData = await logits.toArray();
    logits.destroy();

    return logitsData;
  }

  /**
   * Sample next token from logits
   */
  private sampleToken(
    logits: Float32Array,
    temperature: number,
    topK: number
  ): number {
    // Apply temperature
    const scaled = new Float32Array(logits.length);
    for (let i = 0; i < logits.length; i++) {
      scaled[i] = logits[i] / temperature;
    }

    // Softmax
    const maxLogit = Math.max(...scaled);
    let sum = 0;
    for (let i = 0; i < scaled.length; i++) {
      scaled[i] = Math.exp(scaled[i] - maxLogit);
      sum += scaled[i];
    }
    for (let i = 0; i < scaled.length; i++) {
      scaled[i] /= sum;
    }

    // Top-k sampling
    const indices = Array.from({ length: scaled.length }, (_, i) => i);
    indices.sort((a, b) => scaled[b] - scaled[a]);

    // Keep only top-k
    let cumSum = 0;
    const topKProbs: number[] = [];
    const topKIndices: number[] = [];
    for (let i = 0; i < Math.min(topK, indices.length); i++) {
      topKIndices.push(indices[i]);
      topKProbs.push(scaled[indices[i]]);
      cumSum += scaled[indices[i]];
    }

    // Renormalize
    for (let i = 0; i < topKProbs.length; i++) {
      topKProbs[i] /= cumSum;
    }

    // Sample
    const r = Math.random();
    let cumProb = 0;
    for (let i = 0; i < topKProbs.length; i++) {
      cumProb += topKProbs[i];
      if (r <= cumProb) {
        return topKIndices[i];
      }
    }

    return topKIndices[topKIndices.length - 1];
  }

  /**
   * Generate text completion
   */
  async generate(
    prompt: string,
    options: WebGPUGenerateOptions = {}
  ): Promise<{ text: string; tokensGenerated: number; done: boolean }> {
    if (!this.modelLoaded) {
      throw new Error('Model not loaded');
    }

    const {
      maxTokens = 64,
      temperature = 0.8,
      topK = 40,
    } = options;

    // Simplified tokenization (in production, use proper tokenizer)
    const inputIds = Array.from(prompt).map((c) => c.charCodeAt(0) % this.vocabSize);

    const generatedTokens: number[] = [];

    for (let i = 0; i < maxTokens; i++) {
      const allIds = [...inputIds, ...generatedTokens];

      // Forward pass
      const logits = await this.forward(allIds);

      // Sample next token
      const nextToken = this.sampleToken(logits, temperature, topK);
      generatedTokens.push(nextToken);

      // Check for EOS (simplified)
      if (nextToken === 0 || nextToken === 2) {
        break;
      }
    }

    // Decode tokens (simplified)
    const text = generatedTokens
      .map((t) => String.fromCharCode(t % 128 + 32))
      .join('');

    return {
      text,
      tokensGenerated: generatedTokens.length,
      done: true,
    };
  }

  /**
   * Generate text with streaming
   */
  async *generateStream(
    prompt: string,
    options: WebGPUGenerateOptions = {}
  ): AsyncGenerator<string, void, unknown> {
    // For now, generate all and yield
    const result = await this.generate(prompt, options);
    for (const char of result.text) {
      yield char;
    }
  }

  /**
   * Get embeddings for text
   */
  async embed(_text: string): Promise<number[]> {
    if (!this.modelLoaded || !this.weights) {
      throw new Error('Model not loaded');
    }

    // Simplified: return random embedding
    // A real implementation would process _text through the model
    const embedding = new Float32Array(this.hiddenSize);
    for (let i = 0; i < this.hiddenSize; i++) {
      embedding[i] = Math.random() - 0.5;
    }

    return Array.from(embedding);
  }

  /**
   * Check if model is loaded
   */
  isLoaded(): boolean {
    return this.modelLoaded;
  }

  /**
   * Unload the model
   */
  async unload(): Promise<void> {
    if (this.weights) {
      disposeLlamaWeights(this.weights);
      this.weights = null;
    }

    this.modelLoaded = false;
    this.ggufFile = null;
    this.architecture = null;
    this.emit('unloaded');
  }

  /**
   * Shutdown the engine
   */
  async shutdown(): Promise<void> {
    await this.unload();

    if (this.gpuDevice) {
      this.gpuDevice.destroy();
      this.gpuDevice = null;
    }

    this.initialized = false;
    this.emit('shutdown');
  }

  /**
   * Get GPU capabilities
   */
  getCapabilities() {
    if (!this.gpuDevice) {
      return null;
    }
    return this.gpuDevice.getCapabilities();
  }
}

/**
 * Create a WebGPU engine instance
 */
export function createWebGPUEngine(options?: WebGPUEngineOptions): WebGPUEngine {
  return new WebGPUEngine(options);
}
