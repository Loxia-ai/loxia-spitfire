/**
 * WebGPU Engine
 * GPU-accelerated inference engine implementing the Engine interface
 */

import { EventEmitter } from 'events';
import type { Engine } from './index.js';
import nunjucks from 'nunjucks';
import {
  WebGPUDevice,
  initWebGPU,
  Tensor,
  ops,
  rmsNorm,
  addRmsNorm,
  applyRope,
  fusedRopeAndKVCache,
  feedForwardQ,
  matmulQ,
  loadLlamaWeights,
  disposeLlamaWeights,
  estimateModelMemory,
  getCommandBatcherStats,
  resetCommandBatcherStats,
  resetMatmulQDebugCount,
  resetAllPipelines,
  QuantizedTensor,
  type LlamaWeights,
  type LlamaLayerWeights,
} from './webgpu/index.js';
import { topKSoftmaxGPU, sampleFromTopK } from './webgpu/ops/index.js';
import { fusedQKVProjectionQ4K, fusedQKVProjectionQ6K } from './webgpu/quant/index.js';
import { extractArchitecture, getChatTemplate } from '../model/gguf.js';
import { GGMLType, type GGUFFile, type ModelArchitecture } from '../types/model.js';
import { Tokenizer, parseGGUFWithTokenizer } from '../tokenizer/index.js';
import {
  NgramDraftCache,
  type SpeculativeOptions,
  DEFAULT_SPECULATIVE_OPTIONS,
} from './speculative.js';
import { debugLog, setDebugEnabled } from './webgpu/debug.js';

// Configure nunjucks for Jinja2 compatibility
const nunjucksEnv = new nunjucks.Environment(null, { autoescape: false });
// Add 'raise_exception' filter used by some HuggingFace templates
nunjucksEnv.addFilter('raise_exception', (msg: string) => { throw new Error(msg); });
// Add 'tojson' filter for JSON serialization
nunjucksEnv.addFilter('tojson', (obj: unknown) => JSON.stringify(obj));

export interface WebGPUEngineOptions {
  numThreads?: number; // Ignored for WebGPU (GPU handles parallelism)
  debug?: boolean; // Enable verbose debug output (slower due to GPU-CPU transfers)
  dawnToggles?: string[]; // Dawn backend toggles (Node.js only)
}

export interface WebGPULoadModelOptions {
  contextLength?: number;
  batchSize?: number;
  keepQuantized?: boolean; // Keep weights in quantized format (Q4_K, Q8_0) for reduced VRAM
}

export interface WebGPUGenerateOptions {
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  repeatPenalty?: number;
  stop?: string[];
  rawPrompt?: boolean; // If true, skip chat template formatting
  speculative?: Partial<SpeculativeOptions>; // Speculative decoding options
}

// Use LlamaWeights from model loader
// Re-export for backward compatibility
type ModelWeights = LlamaWeights;

/**
 * GPU-Resident KV Cache for storing key/value projections across generation steps
 * All data stays on GPU - no CPU transfers during generation
 */
interface GPUKVCache {
  // Each layer has its own K and V cache tensors (pre-allocated to maxSeqLen)
  // Shape: [maxSeqLen, numKVHeads * headDim]
  keys: Tensor[];      // One per layer, pre-allocated
  values: Tensor[];    // One per layer, pre-allocated
  seqLen: number;      // Current cached sequence length (how much is filled)
  maxSeqLen: number;   // Maximum sequence length this cache can hold
}

/**
 * Chat message for conversation history
 */
export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

/**
 * Chat session for persistent KV cache across multiple messages
 * Enables fast follow-up messages by only prefilling new tokens
 */
export interface ChatSession {
  id: string;
  messages: ChatMessage[];           // Full conversation history
  cachedTokenIds: number[];          // Token IDs already in KV cache
  cachedSeqLen: number;              // How many tokens are cached
  createdAt: number;
}

/**
 * Options for creating a chat session
 */
export interface ChatSessionOptions {
  systemPrompt?: string;             // Optional system prompt to prepend
}

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
  private tokenizer: Tokenizer | null = null;

  // Model configuration
  private vocabSize = 0;
  private hiddenSize = 0;
  private numLayers = 0;
  private numHeads = 0;
  private numKVHeads = 0;
  private headDim = 0;
  private intermediateSize = 0;
  private contextLength = 2048;
  private rmsNormEps = 1e-5;
  private ropeBase = 10000.0;
  private chatTemplate: string | null = null;

  // Debug mode - when false, skips expensive GPU-CPU transfers for logging
  private debugMode = false;

  // GPU-resident KV cache for efficient incremental generation
  private kvCache: GPUKVCache | null = null;

  // Active chat session for KV cache persistence across messages
  private activeSession: ChatSession | null = null;

  private dawnToggles?: string[];

  constructor(options: WebGPUEngineOptions = {}) {
    super();
    this.debugMode = options.debug ?? false;
    this.dawnToggles = options.dawnToggles;
    setDebugEnabled(this.debugMode);
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
      dawnToggles: this.dawnToggles,
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
      // Parse GGUF file with full tokenizer data (no array size limit)
      debugLog('Parsing GGUF file with tokenizer data...');
      this.ggufFile = await parseGGUFWithTokenizer(modelPath);
      this.architecture = extractArchitecture(this.ggufFile);

      // Extract model configuration from metadata
      this.setupModelConfig(options);

      // Initialize tokenizer from GGUF metadata
      try {
        this.tokenizer = Tokenizer.fromGGUF(this.ggufFile);
        debugLog(`Tokenizer loaded: ${this.tokenizer.vocabSize} tokens`);
        debugLog(`Special tokens: BOS=${this.tokenizer.bosTokenId}, EOS=${this.tokenizer.eosTokenId}`);
      } catch (tokenizerError) {
        console.warn('Could not load tokenizer from GGUF:', tokenizerError);
        this.tokenizer = null;
      }

      // Load chat template from GGUF metadata (Jinja2 format, rendered with nunjucks)
      const ggufTemplate = getChatTemplate(this.ggufFile);
      const arch = this.architecture?.architecture?.toLowerCase();

      if (ggufTemplate) {
        this.chatTemplate = ggufTemplate;
        debugLog('Chat template loaded from GGUF metadata (Jinja2 format)');
      } else if (arch === 'qwen2' || arch === 'qwen') {
        // Fallback Qwen2/Qwen3 template if not in GGUF
        // Format: <|im_start|>role\ncontent<|im_end|>\n
        this.chatTemplate = `{% for message in messages %}<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}`;
        debugLog('Using fallback Qwen chat template');
      } else if (arch === 'llama' || arch === 'llama2') {
        // Fallback Llama 2 template
        this.chatTemplate = '{% for message in messages %}{% if message.role == "system" %}<<SYS>>\n{{ message.content }}\n<</SYS>>\n\n{% elif message.role == "user" %}[INST] {{ message.content }} [/INST]{% elif message.role == "assistant" %}{{ message.content }}{% endif %}{% endfor %}';
        debugLog('Using fallback Llama2 chat template');
      } else {
        this.chatTemplate = null;
        debugLog('No chat template configured - prompts will be used as-is');
      }

      // Load weights to GPU
      await this.loadWeights(modelPath, options);

      // Warm up shader pipelines to avoid JIT compilation spike on first token
      await this.warmupShaders();

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
    this.numKVHeads = this.architecture.headCountKV || this.numHeads; // For GQA
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
      if (meta[`${arch}.rope.freq_base`]) {
        this.ropeBase = Number(meta[`${arch}.rope.freq_base`]);
      }
    }

    debugLog(`Model config: hidden=${this.hiddenSize}, heads=${this.numHeads}, kv_heads=${this.numKVHeads}, layers=${this.numLayers}`);
    debugLog(`RoPE base=${this.ropeBase}, RMS norm eps=${this.rmsNormEps}`);
  }

  /**
   * Load model weights to GPU tensors
   */
  private async loadWeights(modelPath: string, options: WebGPULoadModelOptions = {}): Promise<void> {
    if (!this.ggufFile) {
      throw new Error('GGUF file not parsed');
    }

    // Estimate memory requirements
    const estimatedMemory = estimateModelMemory(this.ggufFile, true);
    const memoryMB = Math.round(estimatedMemory / (1024 * 1024));
    debugLog(`Loading model: ${this.numLayers} layers, ${this.hiddenSize} hidden size`);
    debugLog(`Estimated GPU memory: ${memoryMB} MB`);

    // Check if we have enough GPU memory (if capabilities available)
    const caps = this.gpuDevice?.getCapabilities();
    if (caps && caps.maxBufferSize) {
      const maxBufferMB = Math.round(caps.maxBufferSize / (1024 * 1024));
      debugLog(`Max GPU buffer size: ${maxBufferMB} MB`);
    }

    try {
      // Load actual weights from GGUF file using the model loader
      const keepQuantized = options.keepQuantized ?? false;
      this.weights = await loadLlamaWeights(modelPath, this.ggufFile, {
        dequantize: !keepQuantized,  // Dequantize to f32 if not keeping quantized
        keepQuantized,               // Keep Q4_K/Q8_0 weights on GPU for reduced VRAM
      });

      if (keepQuantized) {
        debugLog('Keeping weights in quantized format (reduced VRAM mode)');
      }

      // Count loaded layers
      const loadedLayers = this.weights.layers.filter(
        (l) => l.attnQ !== null
      ).length;

      debugLog(`Loaded ${loadedLayers} transformer layers`);
      debugLog('Weights loaded to GPU');
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
    debugLog('Loading placeholder weights for testing');

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
        attnQBias: null, // Placeholder doesn't use biases
        attnK: Tensor.random([this.hiddenSize, this.hiddenSize]),
        attnKBias: null,
        attnV: Tensor.random([this.hiddenSize, this.hiddenSize]),
        attnVBias: null,
        attnOutput: Tensor.random([this.hiddenSize, this.hiddenSize]),
        ffnNorm: Tensor.ones([this.hiddenSize]),
        ffnGate: Tensor.random([this.hiddenSize, this.intermediateSize]),
        ffnUp: Tensor.random([this.hiddenSize, this.intermediateSize]),
        ffnDown: Tensor.random([this.intermediateSize, this.hiddenSize]),
      });
    }

    debugLog(`Placeholder weights loaded (${numTestLayers} layers)`);
  }

  /**
   * Warm up shader pipelines by running a minimal forward pass.
   * Triggers JIT compilation of all unique compute shaders during model load
   * instead of on the first user token, removing the first-token latency spike.
   */
  private async warmupShaders(): Promise<void> {
    if (!this.weights) return;

    console.log('Warming up shader pipelines...');
    const warmupStart = performance.now();

    try {
      // Phase 1: Prefill with 2 tokens to trigger GEMM pipelines (batch path)
      const prefillLogits = await this.forwardTensor([0, 1], true);
      prefillLogits.destroy();
      console.log('  GEMM pipelines compiled');

      // Phase 2: Single incremental token to trigger GEMV pipelines (decode path)
      const decodeLogits = await this.forwardTensor([0, 1, 2], true);
      decodeLogits.destroy();
      console.log('  GEMV pipelines compiled');
    } catch (e) {
      // Warmup errors are non-fatal — shaders will compile on first real use
      console.warn(`[Warmup] Forward pass failed: ${e}`);
    }

    // Trigger top-K sampling pipeline compilation
    try {
      const dummyLogits = Tensor.random([1, this.vocabSize], { label: 'warmup_logits' });
      await this.sampleTokenGPU(dummyLogits, 0.8, 40);
      dummyLogits.destroy();
      console.log('  Sampling pipeline compiled');
    } catch (e) {
      console.warn(`[Warmup] Sampling failed: ${e}`);
    }

    // Clean up warmup state
    this.clearKVCache();
    resetMatmulQDebugCount();

    const warmupTime = performance.now() - warmupStart;
    console.log(`Shader warmup complete in ${(warmupTime / 1000).toFixed(2)}s`);
  }

  /**
   * Create a new GPU-resident KV cache with pre-allocated buffers
   * @param maxSeqLen - Maximum sequence length to support
   */
  private createGPUKVCache(maxSeqLen: number): GPUKVCache {
    const numLayers = this.weights?.layers.length || this.numLayers;
    const kvDim = this.numKVHeads * this.headDim;

    debugLog(`[KVCache] Allocating GPU cache: ${numLayers} layers × ${maxSeqLen} tokens × ${kvDim} dim`);
    const memoryMB = (numLayers * 2 * maxSeqLen * kvDim * 4) / (1024 * 1024);
    debugLog(`[KVCache] Estimated memory: ${memoryMB.toFixed(1)} MB`);

    return {
      keys: Array(numLayers).fill(null).map((_, i) =>
        Tensor.zeros([maxSeqLen, kvDim], { label: `kv_cache_k_layer${i}` })
      ),
      values: Array(numLayers).fill(null).map((_, i) =>
        Tensor.zeros([maxSeqLen, kvDim], { label: `kv_cache_v_layer${i}` })
      ),
      seqLen: 0,
      maxSeqLen,
    };
  }

  /**
   * Clear the KV cache and free GPU memory
   */
  private clearKVCache(): void {
    if (this.kvCache) {
      // Destroy all cache tensors to free GPU memory
      for (const k of this.kvCache.keys) {
        if (k && !k.isDestroyed()) k.destroy();
      }
      for (const v of this.kvCache.values) {
        if (v && !v.isDestroyed()) v.destroy();
      }
      this.kvCache = null;
    }
  }

  /**
   * Rollback KV cache to a previous sequence length
   * Used when speculative tokens are rejected
   * The data beyond targetSeqLen is not zeroed - it will be overwritten on next forward pass
   */
  private rollbackKVCache(targetSeqLen: number): void {
    if (!this.kvCache) return;

    if (targetSeqLen < 0 || targetSeqLen > this.kvCache.seqLen) {
      throw new Error(`Invalid rollback target: ${targetSeqLen}, current: ${this.kvCache.seqLen}`);
    }

    this.kvCache.seqLen = targetSeqLen;
  }

  /**
   * Compute debug statistics for an array
   */
  private debugStats(arr: Float32Array | number[], label: string): void {
    const data = arr instanceof Float32Array ? arr : new Float32Array(arr);
    let sum = 0, sumSq = 0, min = Infinity, max = -Infinity;
    let nanCount = 0, infCount = 0;
    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      if (isNaN(v)) { nanCount++; continue; }
      if (!isFinite(v)) { infCount++; continue; }
      sum += v;
      sumSq += v * v;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const n = data.length - nanCount - infCount;
    const mean = n > 0 ? sum / n : 0;
    const variance = n > 0 ? (sumSq / n - mean * mean) : 0;
    const std = Math.sqrt(Math.max(0, variance));
    debugLog(`[Stats ${label}] n=${data.length}, mean=${mean.toFixed(6)}, std=${std.toFixed(6)}, min=${min.toFixed(4)}, max=${max.toFixed(4)}, nan=${nanCount}, inf=${infCount}, sum=${sum.toFixed(2)}`);
  }

  /**
   * Forward pass through the model
   * @param inputIds - Token IDs to process
   * @param useCache - If true, use KV cache for incremental generation (only process new tokens)
   */
  private async forward(inputIds: number[], useCache = false): Promise<Float32Array> {
    if (!this.weights) {
      throw new Error('Model weights not loaded');
    }

    const forwardStart = performance.now();
    const fullSeqLen = inputIds.length;
    const DEBUG = this.debugMode;

    // For cached inference, we only process new tokens
    // startPos is where new tokens start (= cached sequence length)
    let startPos = 0;
    let tokensToProcess: number[];
    let isIncremental = false;

    if (useCache && this.kvCache && this.kvCache.seqLen > 0) {
      // Incremental mode: only process the new token(s)
      isIncremental = true;
      startPos = this.kvCache.seqLen;
      tokensToProcess = inputIds.slice(startPos);
      // Debug: always log for diagnosis
      console.log(`[Forward] INCREMENTAL: seqLen=${this.kvCache.seqLen}, inputIds.length=${inputIds.length}, tokensToProcess.length=${tokensToProcess.length}`);
    } else {
      // Full mode: process all tokens (prefill)
      tokensToProcess = inputIds;
      console.log(`[Forward] PREFILL: kvCache=${!!this.kvCache}, seqLen=${this.kvCache?.seqLen ?? 'N/A'}, inputIds.length=${inputIds.length}`);

      // Initialize or reset GPU cache if using cache
      if (useCache) {
        // Clear any existing cache
        this.clearKVCache();
        // Allocate new cache with reasonable max sequence length
        const maxSeqLen = Math.max(this.contextLength, fullSeqLen + 256);
        this.kvCache = this.createGPUKVCache(maxSeqLen);
      }

      debugLog(`[Forward] PREFILL: processing ${tokensToProcess.length} tokens`);
    }

    const processLen = tokensToProcess.length;
    const timings: Record<string, number> = {};
    let t0: number;

    // Debug: log input token IDs for first call
    if (DEBUG) {
      debugLog(`[Forward] seqLen=${fullSeqLen}, first 5 tokens: [${inputIds.slice(0, 5).join(', ')}], last 5 tokens: [${inputIds.slice(-5).join(', ')}]`);
    }

    // Create input embeddings only for tokens we need to process
    // GGUF stores embeddings as [hiddenSize, vocabSize], pass vocabSize hint
    t0 = performance.now();
    let hidden: Tensor;
    if (this.weights.tokenEmbedding) {
      hidden = await ops.embeddingLookup(this.weights.tokenEmbedding, tokensToProcess, this.vocabSize);
      timings['embedding'] = performance.now() - t0;

      // Debug: check if embeddings are different per position
      if (DEBUG) {
        const embData = await hidden.toArray();
        this.debugStats(embData, 'Embedding');
        const pos0 = Array.from(embData.slice(0, 5));
        const posLast = Array.from(embData.slice(-this.hiddenSize, -this.hiddenSize + 5));
        debugLog(`[Forward] Embedding pos 0: [${pos0.map((v: number) => v.toFixed(4)).join(', ')}]`);
        debugLog(`[Forward] Embedding pos ${processLen - 1}: [${posLast.map((v: number) => v.toFixed(4)).join(', ')}]`);
      }
    } else {
      // Fallback to random embeddings (shouldn't happen with real model)
      console.warn('No token embedding found, using random');
      hidden = Tensor.random([processLen, this.hiddenSize], { label: 'hidden' });
      timings['embedding'] = performance.now() - t0;
    }

    // Process through all layers
    t0 = performance.now();
    let totalAttnTime = 0;
    let totalFfnTime = 0;
    let layerT0: number;

    const numLayersF = this.weights.layers.length;
    // Pre-compute first layer's pre-attention norm
    let normedF: Tensor | null = null;
    {
      const firstLayer = this.weights.layers[0];
      if (firstLayer?.attnNorm) {
        normedF = await rmsNorm(hidden, firstLayer.attnNorm, this.rmsNormEps);
      }
    }

    for (let layerIdx = 0; layerIdx < numLayersF; layerIdx++) {
      const layer = this.weights.layers[layerIdx];

      // Skip layers without required weights
      if (!layer.attnNorm || !layer.ffnNorm) {
        continue;
      }

      // normedF is already computed (from previous layer's fusion or initial computation)
      const normed = normedF!;

      // Debug: check normalized input for layer 0
      if (layerIdx === 0 && DEBUG) {
        const normedData = await normed.toArray();
        this.debugStats(normedData, 'L0 Normed');
        const sample = Array.from(normedData.slice(0, 8));
        debugLog(`[Layer0] Normed input (pos0): [${sample.map(v => v.toFixed(4)).join(', ')}]`);
      }

      // Self-attention with Q, K, V projections
      layerT0 = performance.now();
      let attnOut: Tensor;
      if (layer.attnQ && layer.attnK && layer.attnV && layer.attnOutput) {
        attnOut = await this.selfAttention(normed, layer, processLen, startPos, layerIdx, useCache);
      } else {
        // Fallback: identity (skip attention)
        console.warn(`Layer ${layerIdx}: missing attention weights, skipping`);
        attnOut = normed;
      }
      totalAttnTime += performance.now() - layerT0;

      // Debug: check attention output for last position (layer 0 only)
      if (layerIdx === 0 && DEBUG) {
        const attnData = await attnOut.toArray();
        this.debugStats(attnData, 'L0 AttnOut');
        const lastPosAttn = Array.from(attnData.slice(-this.hiddenSize, -this.hiddenSize + 8));
        debugLog(`[Layer0] Attn out last pos: [${lastPosAttn.map(v => v.toFixed(4)).join(', ')}]`);
      }

      // Fused residual add + pre-FFN norm (saves one dispatch)
      const { sum: afterAttn, normed: normed2 } = await addRmsNorm(
        hidden, attnOut, layer.ffnNorm!, this.rmsNormEps
      );
      hidden.destroy();
      if (attnOut !== normed) {
        normed.destroy();
        attnOut.destroy();
      }
      hidden = afterAttn;

      // FFN (use weights if available, otherwise skip)
      // GGUF weights are [in_features, out_features], correct for x @ W
      layerT0 = performance.now();
      if (layer.ffnGate && layer.ffnUp && layer.ffnDown) {
        const ffnOut = await feedForwardQ(
          normed2,
          layer.ffnGate,
          layer.ffnUp,
          layer.ffnDown
        );

        // Debug: check hidden after FFN for last position (layer 0 only)
        if (layerIdx === 0 && DEBUG) {
          const ffnData = await ffnOut.toArray();
          this.debugStats(ffnData, 'L0 FFNOut');
          const lastPosFfn = Array.from(ffnData.slice(-this.hiddenSize, -this.hiddenSize + 8));
          debugLog(`[Layer0] FFN out last pos: [${lastPosFfn.map(v => v.toFixed(4)).join(', ')}]`);
        }

        // Check if we can fuse post-FFN add + next layer's pre-attention norm
        const nextLayer = layerIdx + 1 < numLayersF ? this.weights.layers[layerIdx + 1] : null;
        if (nextLayer?.attnNorm) {
          const fused = await addRmsNorm(hidden, ffnOut, nextLayer.attnNorm, this.rmsNormEps);
          hidden.destroy();
          ffnOut.destroy();
          hidden = fused.sum;
          normedF = fused.normed;
        } else {
          const afterFfn = await ops.add(hidden, ffnOut);
          hidden.destroy();
          ffnOut.destroy();
          hidden = afterFfn;
          normedF = null;
        }
      } else {
        normedF = null;
      }
      totalFfnTime += performance.now() - layerT0;
      normed2.destroy();

      // Debug: hidden state after layer 0
      if (layerIdx === 0 && DEBUG) {
        const hiddenData = await hidden.toArray();
        const lastPosHidden = Array.from(hiddenData.slice(-this.hiddenSize, -this.hiddenSize + 8));
        debugLog(`[Layer0] Hidden after layer: [${lastPosHidden.map(v => v.toFixed(4)).join(', ')}]`);
      }
    }

    if (isIncremental) {
      debugLog(`[Layers] attn=${totalAttnTime.toFixed(0)}ms, ffn=${totalFfnTime.toFixed(0)}ms`);
    }

    // Update KV cache sequence length after processing all layers
    if (useCache && this.kvCache) {
      this.kvCache.seqLen = startPos + processLen;
      if (DEBUG) {
        debugLog(`[KVCache] Updated seqLen to ${this.kvCache.seqLen}`);
      }
    }
    timings['layers'] = performance.now() - t0;

    // Final norm
    t0 = performance.now();
    const normWeight = this.weights.outputNorm || Tensor.ones([this.hiddenSize]);
    const finalNormed = await rmsNorm(hidden, normWeight, this.rmsNormEps);
    hidden.destroy();
    if (!this.weights.outputNorm) {
      normWeight.destroy();
    }
    timings['finalNorm'] = performance.now() - t0;

    // Extract last token's hidden state on GPU (avoids transferring entire sequence to CPU)
    t0 = performance.now();
    const lastTokenHidden = await ops.sliceLastRow(finalNormed);
    timings['sliceLastRow'] = performance.now() - t0;

    // Debug: check final hidden state (only when debug enabled)
    if (DEBUG) {
      const lastTokenData = await lastTokenHidden.toArray();
      this.debugStats(lastTokenData, 'Final Hidden');
      const lastPosVals = Array.from(lastTokenData.slice(0, 8));
      debugLog(`[Final] Last token hidden: [${lastPosVals.map(v => v.toFixed(4)).join(', ')}]`);
    }

    finalNormed.destroy();

    // Output projection
    // outputWeight is pre-transposed to [hiddenSize, vocabSize] = [2048, 151936]
    t0 = performance.now();
    let logits: Tensor;
    if (this.weights.outputWeight) {
      logits = await matmulQ(lastTokenHidden, this.weights.outputWeight);

      // Debug: check logits
      if (DEBUG) {
        const logitsArr = await logits.toArray();
        this.debugStats(logitsArr, 'Logits');
        debugLog(`[DEBUG] Logit[0] = ${logitsArr[0].toFixed(4)}, logit[1] = ${logitsArr[1].toFixed(4)}`);
      }
    } else {
      // No output weight, create random logits for testing
      logits = Tensor.random([1, this.vocabSize], { label: 'logits' });
    }
    lastTokenHidden.destroy();
    timings['outputProj'] = performance.now() - t0;

    t0 = performance.now();
    const logitsData = await logits.toArray();
    logits.destroy();
    timings['toArray'] = performance.now() - t0;

    const forwardEnd = performance.now();
    const totalTime = forwardEnd - forwardStart;

    // Print timing breakdown for incremental mode (to diagnose slowness)
    if (isIncremental) {
      debugLog(`[Timing] embed=${timings['embedding']?.toFixed(0) || '?'}ms, layers=${timings['layers']?.toFixed(0)}ms, norm=${timings['finalNorm']?.toFixed(0)}ms, slice=${timings['sliceLastRow']?.toFixed(0)}ms, outProj=${timings['outputProj']?.toFixed(0)}ms, toArray=${timings['toArray']?.toFixed(0)}ms`);
    }
    debugLog(`[Forward] ${isIncremental ? 'INCREMENTAL' : 'PREFILL'} took ${totalTime.toFixed(1)}ms for ${processLen} token(s)`);

    return logitsData;
  }

  /**
   * Forward pass returning logits as GPU Tensor (no CPU transfer)
   * Used for GPU-accelerated sampling to avoid expensive toArray()
   * Caller is responsible for destroying the returned tensor!
   */
  // Counter for periodic profiling of incremental forward passes
  private forwardTensorCount = 0;

  private async forwardTensor(inputIds: number[], useCache = false): Promise<Tensor> {

    if (!this.weights) {
      throw new Error('Model weights not loaded');
    }

    const forwardStart = performance.now();
    const fullSeqLen = inputIds.length;

    // For cached inference, we only process new tokens
    let startPos = 0;
    let tokensToProcess: number[];
    let isIncremental = false;

    if (useCache && this.kvCache && this.kvCache.seqLen > 0) {
      isIncremental = true;
      startPos = this.kvCache.seqLen;
      tokensToProcess = inputIds.slice(startPos);
    } else {
      tokensToProcess = inputIds;
      if (useCache) {
        this.clearKVCache();
        const maxSeqLen = Math.max(this.contextLength, fullSeqLen + 256);
        this.kvCache = this.createGPUKVCache(maxSeqLen);
      }
    }

    const processLen = tokensToProcess.length;

    // Profile every 5th incremental token for visibility
    const shouldProfile = isIncremental && processLen === 1 && (this.forwardTensorCount % 5 === 0);
    let tProf: number = 0;
    const prof: Record<string, number> = {};

    // Create input embeddings only for tokens we need to process
    if (shouldProfile) tProf = performance.now();
    let hidden: Tensor;
    if (this.weights.tokenEmbedding) {
      hidden = await ops.embeddingLookup(this.weights.tokenEmbedding, tokensToProcess, this.vocabSize);
    } else {
      hidden = Tensor.random([processLen, this.hiddenSize], { label: 'hidden' });
    }
    if (shouldProfile) prof['embed'] = performance.now() - tProf;

    // Process through all layers with fused add+rmsNorm between layers
    if (shouldProfile) tProf = performance.now();
    const numLayers = this.weights.layers.length;
    // Pre-compute first layer's pre-attention norm
    let normed: Tensor | null = null;
    {
      const firstLayer = this.weights.layers[0];
      if (firstLayer?.attnNorm) {
        normed = await rmsNorm(hidden, firstLayer.attnNorm, this.rmsNormEps);
      }
    }

    for (let layerIdx = 0; layerIdx < numLayers; layerIdx++) {
      const layer = this.weights.layers[layerIdx];
      if (!layer.attnNorm || !layer.ffnNorm) continue;

      // normed is already computed (from previous layer's fusion or initial computation)

      // Self-attention
      let attnOut: Tensor;
      if (layer.attnQ && layer.attnK && layer.attnV && layer.attnOutput) {
        attnOut = await this.selfAttention(normed!, layer, processLen, startPos, layerIdx, useCache);
      } else {
        attnOut = normed!;
      }

      // Fused residual add + pre-FFN norm (saves one dispatch)
      const { sum: afterAttn, normed: normed2 } = await addRmsNorm(
        hidden, attnOut, layer.ffnNorm!, this.rmsNormEps
      );
      hidden.destroy();
      if (attnOut !== normed) {
        normed!.destroy();
        attnOut.destroy();
      }
      hidden = afterAttn;

      // FFN
      if (layer.ffnGate && layer.ffnUp && layer.ffnDown) {
        const ffnOut = await feedForwardQ(normed2, layer.ffnGate, layer.ffnUp, layer.ffnDown);

        // Check if we can fuse post-FFN add + next layer's pre-attention norm
        const nextLayer = layerIdx + 1 < numLayers ? this.weights.layers[layerIdx + 1] : null;
        if (nextLayer?.attnNorm) {
          // Fused: add(hidden, ffnOut) + rmsNorm(sum, nextAttnNorm) → saves another dispatch
          const fused = await addRmsNorm(hidden, ffnOut, nextLayer.attnNorm, this.rmsNormEps);
          hidden.destroy();
          ffnOut.destroy();
          hidden = fused.sum;
          normed = fused.normed;
        } else {
          // Last layer: just add, no next norm to fuse with
          const afterFfn = await ops.add(hidden, ffnOut);
          hidden.destroy();
          ffnOut.destroy();
          hidden = afterFfn;
          normed = null;
        }
      } else {
        normed = null;
      }
      normed2.destroy();
    }
    if (shouldProfile) prof['layers'] = performance.now() - tProf;

    // Update KV cache sequence length
    if (useCache && this.kvCache) {
      this.kvCache.seqLen = startPos + processLen;
    }

    // Final norm
    if (shouldProfile) tProf = performance.now();
    const normWeight = this.weights.outputNorm || Tensor.ones([this.hiddenSize]);
    const finalNormed = await rmsNorm(hidden, normWeight, this.rmsNormEps);
    hidden.destroy();
    if (!this.weights.outputNorm) normWeight.destroy();
    if (shouldProfile) prof['finalNorm'] = performance.now() - tProf;

    // Extract last token's hidden state
    if (shouldProfile) tProf = performance.now();
    const lastTokenHidden = await ops.sliceLastRow(finalNormed);
    finalNormed.destroy();
    if (shouldProfile) prof['slice'] = performance.now() - tProf;

    // Output projection - returns logits Tensor (caller must destroy!)
    if (shouldProfile) tProf = performance.now();
    let logits: Tensor;
    if (this.weights.outputWeight) {
      logits = await matmulQ(lastTokenHidden, this.weights.outputWeight);
    } else {
      logits = Tensor.random([1, this.vocabSize], { label: 'logits' });
    }
    lastTokenHidden.destroy();
    if (shouldProfile) prof['outProj'] = performance.now() - tProf;

    const totalTime = performance.now() - forwardStart;
    if (isIncremental && processLen === 1) {
      this.forwardTensorCount++;
      if (shouldProfile) {
        debugLog(`[ForwardTensor] INCR #${this.forwardTensorCount} ${totalTime.toFixed(0)}ms | embed=${prof['embed']?.toFixed(0)}ms layers=${prof['layers']?.toFixed(0)}ms norm=${prof['finalNorm']?.toFixed(0)}ms slice=${prof['slice']?.toFixed(0)}ms outProj=${prof['outProj']?.toFixed(0)}ms`);
      }
    } else {
      debugLog(`[ForwardTensor] ${isIncremental ? 'INCR' : 'PREFILL'} took ${totalTime.toFixed(1)}ms for ${processLen} token(s)`);
    }

    return logits;
  }

  /**
   * Sample next token using GPU-accelerated top-k selection
   * Avoids expensive toArray() by doing top-k and softmax on GPU
   * Only transfers k values (~160 bytes) instead of full vocab (~512KB)
   */
  private async sampleTokenGPU(
    logits: Tensor,
    temperature: number,
    topK: number
  ): Promise<number> {
    const result = await topKSoftmaxGPU(logits, topK, temperature);
    return sampleFromTopK(result.probs, result.indices);
  }

  /**
   * Forward pass returning logits for multiple positions
   * Used for speculative decoding verification
   *
   * @param inputIds - Full sequence including draft tokens
   * @param numLogits - Number of logit vectors to return (from the end)
   * @param useCache - Whether to use KV cache
   * @returns Array of logit vectors, one per position
   */
  private async forwardMultiLogits(
    inputIds: number[],
    numLogits: number,
    useCache: boolean
  ): Promise<Float32Array[]> {
    if (!this.weights) {
      throw new Error('Model weights not loaded');
    }

    const fullSeqLen = inputIds.length;
    let tokensToProcess: number[];
    let startPos: number;

    // Determine if this is incremental (using cache) or prefill
    if (useCache && this.kvCache && this.kvCache.seqLen > 0) {
      startPos = this.kvCache.seqLen;
      tokensToProcess = inputIds.slice(startPos);
    } else {
      startPos = 0;
      tokensToProcess = inputIds;

      if (useCache) {
        this.clearKVCache();
        const maxSeqLen = Math.max(this.contextLength, fullSeqLen + 256);
        this.kvCache = this.createGPUKVCache(maxSeqLen);
      }
    }

    const processLen = tokensToProcess.length;

    // Embedding lookup (pass token IDs as number[], not Tensor)
    let hidden = await ops.embeddingLookup(this.weights.tokenEmbedding!, tokensToProcess, this.vocabSize);

    // Process through all layers with fused add+rmsNorm between layers
    let normedM: Tensor | null = null;
    {
      const firstLayer = this.weights.layers[0];
      if (firstLayer?.attnNorm) {
        normedM = await rmsNorm(hidden, firstLayer.attnNorm, this.rmsNormEps);
      }
    }

    for (let layerIdx = 0; layerIdx < this.numLayers; layerIdx++) {
      const layer = this.weights.layers[layerIdx];

      // normedM is already computed (from previous layer's fusion or initial computation)
      const normed1 = normedM!;

      // Self-attention
      const attnOut = await this.selfAttention(
        normed1,
        layer,
        processLen,
        startPos,
        layerIdx,
        useCache
      );
      normed1.destroy();

      // Fused residual add + pre-FFN norm (saves one dispatch)
      const { sum: afterAttn, normed: normed2 } = await addRmsNorm(
        hidden, attnOut, layer.ffnNorm!, this.rmsNormEps
      );
      hidden.destroy();
      attnOut.destroy();
      hidden = afterAttn;

      // Feed-forward
      const ffnOut = await feedForwardQ(
        normed2,
        layer.ffnGate!,
        layer.ffnUp!,
        layer.ffnDown!
      );
      normed2.destroy();

      // Fuse post-FFN add + next layer's pre-attention norm
      const nextLayer = layerIdx + 1 < this.numLayers ? this.weights.layers[layerIdx + 1] : null;
      if (nextLayer?.attnNorm) {
        const fused = await addRmsNorm(hidden, ffnOut, nextLayer.attnNorm, this.rmsNormEps);
        hidden.destroy();
        ffnOut.destroy();
        hidden = fused.sum;
        normedM = fused.normed;
      } else {
        const afterFFN = await ops.add(hidden, ffnOut);
        hidden.destroy();
        ffnOut.destroy();
        hidden = afterFFN;
        normedM = null;
      }
    }

    // Update KV cache sequence length
    if (useCache && this.kvCache) {
      this.kvCache.seqLen = startPos + processLen;
    }

    // Final norm
    const normWeight = this.weights.outputNorm || Tensor.ones([this.hiddenSize]);
    const finalNormed = await rmsNorm(hidden, normWeight, this.rmsNormEps);
    hidden.destroy();
    if (!this.weights.outputNorm) {
      normWeight.destroy();
    }

    // Extract last numLogits rows (for multi-position verification)
    const multiHidden = await ops.sliceLastRows(finalNormed, numLogits);
    finalNormed.destroy();

    // Output projection for all positions
    let logitsTensor: Tensor;
    if (this.weights.outputWeight) {
      logitsTensor = await matmulQ(multiHidden, this.weights.outputWeight);
    } else {
      logitsTensor = Tensor.random([numLogits, this.vocabSize], { label: 'logits' });
    }
    multiHidden.destroy();

    // Transfer to CPU and split into individual logit vectors
    const allLogitsData = await logitsTensor.toArray();
    logitsTensor.destroy();

    // Split into array of Float32Arrays
    const result: Float32Array[] = [];
    for (let i = 0; i < numLogits; i++) {
      const start = i * this.vocabSize;
      const end = start + this.vocabSize;
      result.push(allLogitsData.slice(start, end));
    }

    return result;
  }

  /**
   * Early-exit forward pass for draft token generation
   * Runs only the first N layers (no KV cache) for fast speculation
   *
   * @param inputIds - Token IDs to process
   * @param numLayers - Number of layers to run (default: 6)
   * @returns Logits for sampling draft token
   */
  private async forwardDraft(
    inputIds: number[],
    numLayers: number = 6
  ): Promise<Float32Array> {
    if (!this.weights) {
      throw new Error('Model weights not loaded');
    }

    const processLen = inputIds.length;

    // Embedding lookup
    let hidden = await ops.embeddingLookup(
      this.weights.tokenEmbedding!,
      inputIds,
      this.vocabSize
    );

    // Initial norm for first layer
    let normed: Tensor | null = null;
    {
      const firstLayer = this.weights.layers[0];
      if (firstLayer?.attnNorm) {
        normed = await rmsNorm(hidden, firstLayer.attnNorm, this.rmsNormEps);
      }
    }

    // Process only first N layers (NO KV cache - standalone draft)
    const layersToRun = Math.min(numLayers, this.weights.layers.length);

    for (let layerIdx = 0; layerIdx < layersToRun; layerIdx++) {
      const layer = this.weights.layers[layerIdx];

      if (!layer.attnNorm || !layer.ffnNorm) {
        continue;
      }

      const normed1 = normed!;

      // Self-attention WITHOUT KV cache (draft mode)
      // Use simple attention computation for speed
      const attnOut = await this.selfAttentionNoCacheForDraft(
        normed1,
        layer,
        processLen
      );

      // Fused residual add + pre-FFN norm
      const { sum: afterAttn, normed: normed2 } = await addRmsNorm(
        hidden, attnOut, layer.ffnNorm!, this.rmsNormEps
      );
      hidden.destroy();
      if (attnOut !== normed1) {
        normed1.destroy();
        attnOut.destroy();
      }
      hidden = afterAttn;

      // Feed-forward
      if (layer.ffnGate && layer.ffnUp && layer.ffnDown) {
        const ffnOut = await feedForwardQ(
          normed2,
          layer.ffnGate,
          layer.ffnUp,
          layer.ffnDown
        );

        // Fuse post-FFN add + next layer's pre-attention norm
        const nextLayer = layerIdx + 1 < layersToRun ? this.weights.layers[layerIdx + 1] : null;
        if (nextLayer?.attnNorm) {
          const fused = await addRmsNorm(hidden, ffnOut, nextLayer.attnNorm, this.rmsNormEps);
          hidden.destroy();
          ffnOut.destroy();
          hidden = fused.sum;
          normed = fused.normed;
        } else {
          const afterFFN = await ops.add(hidden, ffnOut);
          hidden.destroy();
          ffnOut.destroy();
          hidden = afterFFN;
          normed = null;
        }
        normed2.destroy();
      } else {
        normed2.destroy();
        normed = null;
      }
    }

    // Final norm (using the main output norm)
    const normWeight = this.weights.outputNorm || Tensor.ones([this.hiddenSize]);
    const finalNormed = await rmsNorm(hidden, normWeight, this.rmsNormEps);
    hidden.destroy();
    if (!this.weights.outputNorm) {
      normWeight.destroy();
    }

    // Extract last token's hidden state
    const lastTokenHidden = await ops.sliceLastRow(finalNormed);
    finalNormed.destroy();

    // Output projection (LM head)
    let logits: Tensor;
    if (this.weights.outputWeight) {
      logits = await matmulQ(lastTokenHidden, this.weights.outputWeight);
    } else {
      logits = Tensor.random([1, this.vocabSize], { label: 'draft_logits' });
    }
    lastTokenHidden.destroy();

    // Transfer to CPU
    const logitsData = await logits.toArray();
    logits.destroy();

    return logitsData;
  }

  /**
   * Simplified self-attention for draft mode (no KV cache)
   * This is a streamlined version that computes attention from scratch
   */
  private async selfAttentionNoCacheForDraft(
    x: Tensor,
    layer: LlamaLayerWeights,
    seqLen: number
  ): Promise<Tensor> {
    // Project to Q, K, V
    const q = await matmulQ(x, layer.attnQ!);
    const k = await matmulQ(x, layer.attnK!);
    const v = await matmulQ(x, layer.attnV!);

    // Apply RoPE (startPos = 0 for draft, full sequence)
    const qRoped = await applyRope(q, seqLen, this.numHeads, this.headDim, this.ropeBase, 0);
    const kRoped = await applyRope(k, seqLen, this.numKVHeads, this.headDim, this.ropeBase, 0);
    q.destroy();
    k.destroy();

    // Causal attention (full, no cache)
    const attnResult = await ops.causalAttention(
      qRoped, kRoped, v,
      this.numHeads,
      this.numKVHeads,
      this.headDim,
      seqLen  // Full sequence length
    );
    qRoped.destroy();
    kRoped.destroy();
    v.destroy();

    // Output projection
    const output = await matmulQ(attnResult, layer.attnOutput!);
    attnResult.destroy();

    return output;
  }

  /**
   * Self-attention with Q, K, V projections and GPU-resident KV cache
   * Supports Grouped Query Attention (GQA) where numKVHeads < numHeads
   * Uses GPU KV cache for O(n) per-token generation instead of O(n²)
   *
   * @param x - Input tensor [processLen, hiddenSize]
   * @param layer - Layer weights
   * @param processLen - Number of tokens being processed (1 for incremental, full for prefill)
   * @param startPos - Starting position in sequence (0 for prefill, cached length for incremental)
   * @param layerIdx - Layer index for KV cache access
   * @param useCache - Whether to use/update KV cache
   */
  private async selfAttention(
    x: Tensor,
    layer: LlamaLayerWeights,
    processLen: number,
    startPos: number,
    layerIdx: number,
    useCache: boolean
  ): Promise<Tensor> {
    const shouldTime = layerIdx === 0 && processLen === 1; // Time first layer in incremental mode
    let t0 = shouldTime ? performance.now() : 0;
    const attnTimings: Record<string, number> = {};

    // Project to Q, K, V for new tokens only
    // Use fused kernel for f32, separate matmulQ for quantized
    let q: Tensor, k: Tensor, vProj: Tensor;
    const hasQuantizedQKV =
      layer.attnQ instanceof QuantizedTensor ||
      layer.attnK instanceof QuantizedTensor ||
      layer.attnV instanceof QuantizedTensor;

    if (hasQuantizedQKV) {
      // Quantized path
      // Check if ALL QKV weights are specifically Q4_K or Q6_K (required for fused kernels)
      const allQ4K =
        layer.attnQ instanceof QuantizedTensor && layer.attnQ.quantType === GGMLType.Q4_K &&
        layer.attnK instanceof QuantizedTensor && layer.attnK.quantType === GGMLType.Q4_K &&
        layer.attnV instanceof QuantizedTensor && layer.attnV.quantType === GGMLType.Q4_K;

      const allQ6K =
        layer.attnQ instanceof QuantizedTensor && layer.attnQ.quantType === GGMLType.Q6_K &&
        layer.attnK instanceof QuantizedTensor && layer.attnK.quantType === GGMLType.Q6_K &&
        layer.attnV instanceof QuantizedTensor && layer.attnV.quantType === GGMLType.Q6_K;

      // Use fused GEMV only for single-token (M=1) incremental decoding
      // For prefill (M>1), use separate GEMM operations
      const M = x.shape[0];

      if (allQ4K && M === 1) {
        // Fused Q4_K GEMV - single kernel, reads x once
        const result = await fusedQKVProjectionQ4K(
          x,
          layer.attnQ as QuantizedTensor,
          layer.attnK as QuantizedTensor,
          layer.attnV as QuantizedTensor
        );
        q = result.Q;
        k = result.K;
        vProj = result.V;
      } else if (allQ6K && M === 1) {
        // Fused Q6_K GEMV - single kernel, reads x once
        const result = await fusedQKVProjectionQ6K(
          x,
          layer.attnQ as QuantizedTensor,
          layer.attnK as QuantizedTensor,
          layer.attnV as QuantizedTensor
        );
        q = result.Q;
        k = result.K;
        vProj = result.V;
      } else {
        // Prefill (M>1) or mixed quantization - use separate matmuls
        q = await matmulQ(x, layer.attnQ!);
        k = await matmulQ(x, layer.attnK!);
        vProj = await matmulQ(x, layer.attnV!);
      }
    } else {
      // f32 path - use fused kernel (reads x once)
      const { Q: qRaw, K: kRaw, V: vProjRaw } = await ops.fusedQKVProjection(
        x,
        layer.attnQ! as Tensor,
        layer.attnK! as Tensor,
        layer.attnV! as Tensor
      );
      q = qRaw;
      k = kRaw;
      vProj = vProjRaw;
    }
    if (shouldTime) { attnTimings['qkvProj'] = performance.now() - t0; t0 = performance.now(); }

    // Apply biases if present (Qwen uses Q/K/V biases)
    if (layer.attnQBias) {
      q = await this.addBias(q, layer.attnQBias);
    }
    if (layer.attnKBias) {
      k = await this.addBias(k, layer.attnKBias);
    }
    if (layer.attnVBias) {
      vProj = await this.addBias(vProj, layer.attnVBias);
    }
    if (shouldTime) { attnTimings['biases'] = performance.now() - t0; t0 = performance.now(); }

    // Debug: check Q, K, V before RoPE
    const debugRope = this.debugMode;
    let qBeforeRope: number[] | null = null;
    if (debugRope) {
      const qArr = await q.toArray();
      const kArr = await k.toArray();
      const vArr = await vProj.toArray();
      this.debugStats(qArr, 'Q pre-RoPE');
      this.debugStats(kArr, 'K pre-RoPE');
      this.debugStats(vArr, 'V');
      qBeforeRope = Array.from(qArr.slice(0, 8)); // First 8 values of pos 0
    }

    // Determine K/V to use for attention
    let kForAttention: Tensor;
    let vForAttention: Tensor;
    let totalSeqLen: number;
    let qRope: Tensor;

    if (useCache && this.kvCache) {
      // FUSED PATH: RoPE + KV Cache Write in ONE kernel
      // Saves 4 kernel calls (2 rope + 2 copy_rows)
      qRope = await fusedRopeAndKVCache(
        q, k, vProj,
        this.kvCache.keys[layerIdx],
        this.kvCache.values[layerIdx],
        this.numHeads,
        this.numKVHeads,
        this.headDim,
        startPos,
        this.ropeBase
      );
      q.destroy();
      k.destroy();
      vProj.destroy();
      if (shouldTime) { attnTimings['fusedRopeKV'] = performance.now() - t0; t0 = performance.now(); }

      totalSeqLen = startPos + processLen;
      kForAttention = this.kvCache.keys[layerIdx];
      vForAttention = this.kvCache.values[layerIdx];
    } else {
      // No cache mode: apply RoPE separately
      qRope = await applyRope(q, processLen, this.numHeads, this.headDim, this.ropeBase, startPos);
      const kRope = await applyRope(k, processLen, this.numKVHeads, this.headDim, this.ropeBase, startPos);
      q.destroy();
      k.destroy();
      if (shouldTime) { attnTimings['rope'] = performance.now() - t0; t0 = performance.now(); }

      kForAttention = kRope;
      vForAttention = vProj;
      totalSeqLen = processLen;
    }

    // Debug: check Q after RoPE
    if (debugRope && qBeforeRope) {
      const qData = await qRope.toArray();
      this.debugStats(qData, 'Q post-RoPE');
      const qAfterRope = Array.from(qData.slice(0, 8));
      const qLastPosAfter = Array.from(qData.slice(-this.numHeads * this.headDim, -this.numHeads * this.headDim + 8));
      debugLog(`[RoPE] Q pos0 before: [${qBeforeRope.map(v => v.toFixed(4)).join(', ')}]`);
      debugLog(`[RoPE] Q pos0 after:  [${qAfterRope.map(v => v.toFixed(4)).join(', ')}]`);
      debugLog(`[RoPE] Q pos${processLen - 1} after: [${qLastPosAfter.map(v => v.toFixed(4)).join(', ')}]`);
    }

    // GPU-accelerated causal attention
    // Q: [processLen, numHeads * headDim] - queries for new positions only
    // K: [bufferSize, numKVHeads * headDim] - all keys (may be pre-allocated cache buffer)
    // V: [bufferSize, numKVHeads * headDim] - all values (may be pre-allocated cache buffer)
    //
    // When using cache, pass totalSeqLen to tell attention how many keys are actually valid
    const attnOut = await ops.causalAttention(
      qRope,
      kForAttention,
      vForAttention,
      this.numHeads,
      this.numKVHeads,
      this.headDim,
      startPos,
      useCache && this.kvCache ? totalSeqLen : undefined
    );
    if (shouldTime) { attnTimings['causalAttn'] = performance.now() - t0; t0 = performance.now(); }

    // Cleanup Q (K/V cleanup depends on cache mode)
    qRope.destroy();

    // Only destroy K/V if not using cache (cache tensors are reused)
    if (!useCache || !this.kvCache) {
      kForAttention.destroy();
      vForAttention.destroy();
    }

    // Output projection: [processLen, numHeads * headDim] @ [numHeads * headDim, hiddenSize]
    const projected = await matmulQ(attnOut, layer.attnOutput!);
    attnOut.destroy();
    if (shouldTime) {
      attnTimings['outProj'] = performance.now() - t0;
      debugLog(`[AttnL0] qkvProj=${attnTimings['qkvProj']?.toFixed(1)}ms biases=${attnTimings['biases']?.toFixed(1)}ms fusedRopeKV=${attnTimings['fusedRopeKV']?.toFixed(1) || attnTimings['rope']?.toFixed(1)}ms causalAttn=${attnTimings['causalAttn']?.toFixed(1)}ms outProj=${attnTimings['outProj']?.toFixed(1)}ms`);
    }

    return projected;
  }

  /**
   * Sample next token from logits using efficient top-k selection
   * Uses a min-heap for O(vocab) top-k selection instead of O(vocab × k)
   */
  private sampleToken(
    logits: Float32Array,
    temperature: number,
    topK: number
  ): number {
    const vocabSize = logits.length;
    const k = Math.min(topK, vocabSize);

    // Find max logit for numerical stability
    let maxLogit = -Infinity;
    for (let i = 0; i < vocabSize; i++) {
      if (logits[i] > maxLogit) {
        maxLogit = logits[i];
      }
    }

    // Use a min-heap to efficiently find top-k elements in O(vocab × log(k))
    // Heap stores {prob, idx} pairs, sorted by prob ascending (so we can remove smallest)
    const heap: { prob: number; idx: number }[] = [];

    // Helper functions for min-heap
    const heapPush = (item: { prob: number; idx: number }) => {
      heap.push(item);
      let i = heap.length - 1;
      while (i > 0) {
        const parent = Math.floor((i - 1) / 2);
        if (heap[parent].prob <= heap[i].prob) break;
        [heap[parent], heap[i]] = [heap[i], heap[parent]];
        i = parent;
      }
    };

    const heapPop = () => {
      const result = heap[0];
      const last = heap.pop()!;
      if (heap.length > 0) {
        heap[0] = last;
        let i = 0;
        while (true) {
          const left = 2 * i + 1;
          const right = 2 * i + 2;
          let smallest = i;
          if (left < heap.length && heap[left].prob < heap[smallest].prob) {
            smallest = left;
          }
          if (right < heap.length && heap[right].prob < heap[smallest].prob) {
            smallest = right;
          }
          if (smallest === i) break;
          [heap[i], heap[smallest]] = [heap[smallest], heap[i]];
          i = smallest;
        }
      }
      return result;
    };

    // Single pass: compute scaled logits and maintain top-k heap
    for (let i = 0; i < vocabSize; i++) {
      const scaled = (logits[i] - maxLogit) / temperature;
      const prob = Math.exp(scaled);

      if (heap.length < k) {
        heapPush({ prob, idx: i });
      } else if (prob > heap[0].prob) {
        heapPop();
        heapPush({ prob, idx: i });
      }
    }

    // Extract top-k from heap (will be in ascending order)
    const topKItems: { prob: number; idx: number }[] = [];
    while (heap.length > 0) {
      topKItems.push(heapPop());
    }
    topKItems.reverse(); // Now descending by probability

    // Compute softmax normalization for just the top-k
    let sum = 0;
    for (const item of topKItems) {
      sum += item.prob;
    }

    // Sample from top-k
    const r = Math.random() * sum;
    let cumProb = 0;
    for (const item of topKItems) {
      cumProb += item.prob;
      if (r <= cumProb) {
        return item.idx;
      }
    }

    return topKItems[topKItems.length - 1]?.idx || 0;
  }

  /**
   * Generate text completion with GPU-resident KV cache for O(n) generation
   * Supports speculative decoding for higher GPU utilization
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
      stop = [],
      rawPrompt = false,
      speculative: speculativeOpts,
    } = options;

    // Merge speculative options with defaults
    const specConfig: SpeculativeOptions = speculativeOpts?.enabled !== false
      ? { ...DEFAULT_SPECULATIVE_OPTIONS, ...speculativeOpts }
      : { ...DEFAULT_SPECULATIVE_OPTIONS, enabled: false };

    // Create n-gram cache if speculative decoding is enabled
    const ngramCache = specConfig.enabled
      ? new NgramDraftCache(specConfig.ngramSize)
      : null;

    // Clear any existing KV cache for fresh generation
    this.clearKVCache();

    // Apply chat template unless rawPrompt is set
    const formattedPrompt = rawPrompt ? prompt : this.applyChatTemplate(prompt);

    // Tokenize input using proper tokenizer
    let inputIds: number[];
    if (this.tokenizer) {
      inputIds = this.tokenizer.encode(formattedPrompt);
    } else {
      // Fallback to simple byte encoding
      inputIds = Array.from(formattedPrompt).map((c) => c.charCodeAt(0) % this.vocabSize);
    }

    // Seed n-gram cache with prompt patterns for better speculation
    if (ngramCache && inputIds.length > 10) {
      ngramCache.seedFromPrompt(inputIds);
    }

    const generatedTokens: number[] = [];
    let generatedText = '';
    const eosId = this.tokenizer?.eosTokenId ?? 2;

    // Helper to decode and append tokens
    const appendToken = (token: number): boolean => {
      generatedTokens.push(token);

      let tokenText: string;
      if (this.tokenizer) {
        tokenText = this.tokenizer.decodeToken(token);
      } else {
        tokenText = String.fromCharCode(token % 128);
      }
      generatedText += tokenText;

      // Check for EOS
      if (token === eosId) {
        return true; // should stop
      }

      // Check for stop sequences
      for (const stopSeq of stop) {
        if (generatedText.endsWith(stopSeq)) {
          generatedText = generatedText.slice(0, -stopSeq.length);
          return true; // should stop
        }
      }
      return false;
    };

    let tokenCount = 0;
    let forwardPasses = 0;  // Track forward passes for efficiency metrics
    let draftPasses = 0;    // Track early-exit draft passes
    let earlyExitAccepted = 0;  // Track accepted tokens from early-exit
    let earlyExitProposed = 0;  // Track proposed tokens from early-exit
    const genStartTime = performance.now();

    // Early-exit speculation config
    const DRAFT_LAYERS = 6;       // Use first 6 layers for drafting
    const DRAFT_TOKENS = 4;       // Generate 4 draft tokens
    const USE_EARLY_EXIT = false;  // Disabled: early-exit needs KV cache support to be fast

    while (tokenCount < maxTokens) {
      // Try N-gram speculation first if enabled
      if (ngramCache && generatedTokens.length >= 2) {
        const allTokens = [...inputIds, ...generatedTokens];
        const draftTokens = ngramCache.getDrafts(allTokens, specConfig.maxDraftTokens);

        if (draftTokens.length > 0) {
          // N-gram speculative path (existing code)
          const cacheSeqLenBefore = this.kvCache?.seqLen ?? 0;
          const fullSequence = [...allTokens, ...draftTokens];
          const allLogits = await this.forwardMultiLogits(fullSequence, draftTokens.length, true);
          forwardPasses++;

          let accepted = 0;
          for (let i = 0; i < draftTokens.length; i++) {
            if (i < draftTokens.length - 1) {
              const predictedNext = this.sampleToken(allLogits[i], temperature, topK);
              if (draftTokens[i + 1] === predictedNext) {
                accepted++;
              } else {
                break;
              }
            } else {
              accepted++;
            }
          }

          const newCacheLen = cacheSeqLenBefore + accepted;
          this.rollbackKVCache(newCacheLen);

          let shouldStop = false;
          for (let i = 0; i < accepted && !shouldStop; i++) {
            shouldStop = appendToken(draftTokens[i]);
            tokenCount++;
          }

          ngramCache.update([...inputIds, ...generatedTokens], inputIds.length);
          ngramCache.recordAccepted(accepted);

          if (shouldStop || tokenCount >= maxTokens) break;

          if (accepted > 0 && !shouldStop && tokenCount < maxTokens) {
            const bonusLogits = allLogits[accepted - 1];
            const bonusToken = this.sampleToken(bonusLogits, temperature, topK);
            const currentTokens = [...inputIds, ...generatedTokens];
            await this.forward([...currentTokens, bonusToken], true);
            forwardPasses++;
            shouldStop = appendToken(bonusToken);
            tokenCount++;
            ngramCache.update([...inputIds, ...generatedTokens], inputIds.length);
            if (shouldStop) break;
          }
          continue;
        }
      }

      // Early-exit speculation: use first N layers to generate draft tokens
      if (USE_EARLY_EXIT && generatedTokens.length >= 1) {
        const allTokens = [...inputIds, ...generatedTokens];
        const cacheSeqLenBefore = this.kvCache?.seqLen ?? 0;

        // Generate draft tokens using early-exit (first 6 layers only)
        const draftTokens: number[] = [];
        let currentDraftSequence = [...allTokens];

        for (let d = 0; d < DRAFT_TOKENS; d++) {
          const draftLogits = await this.forwardDraft(currentDraftSequence, DRAFT_LAYERS);
          draftPasses++;

          const draftToken = this.sampleToken(draftLogits, temperature, topK);
          draftTokens.push(draftToken);
          currentDraftSequence.push(draftToken);

          // Stop drafting on EOS
          if (draftToken === eosId) break;
        }

        if (draftTokens.length > 0) {
          earlyExitProposed += draftTokens.length;

          // Verify all draft tokens with full model in ONE pass
          const fullSequence = [...allTokens, ...draftTokens];
          const allLogits = await this.forwardMultiLogits(fullSequence, draftTokens.length, true);
          forwardPasses++;

          // Verify: check if full model agrees with each draft
          // We need to check: given logits[i-1], would we sample draftTokens[i]?
          // But logits[0] is output AFTER draftTokens[0], predicting what comes next
          // So logits[i] predicts draftTokens[i+1]
          let accepted = 0;

          for (let i = 0; i < draftTokens.length; i++) {
            if (i < draftTokens.length - 1) {
              // Check if logits[i] would produce draftTokens[i+1]
              const predictedNext = this.sampleToken(allLogits[i], temperature, topK);
              if (draftTokens[i + 1] === predictedNext) {
                accepted++;
              } else {
                // Mismatch - accept up to here, plus the correct next token
                accepted++;  // Accept current draft token
                break;
              }
            } else {
              // Last draft token - always accept
              accepted++;
            }
          }

          earlyExitAccepted += accepted;

          // Rollback KV cache to keep only accepted tokens
          const newCacheLen = cacheSeqLenBefore + accepted;
          this.rollbackKVCache(newCacheLen);

          // Append accepted draft tokens
          let shouldStop = false;
          for (let i = 0; i < accepted && !shouldStop; i++) {
            shouldStop = appendToken(draftTokens[i]);
            tokenCount++;
          }

          if (shouldStop || tokenCount >= maxTokens) break;

          // Sample bonus token from last accepted position
          if (accepted > 0 && !shouldStop && tokenCount < maxTokens) {
            const bonusLogits = allLogits[accepted - 1];
            const bonusToken = this.sampleToken(bonusLogits, temperature, topK);

            // Update KV cache with the bonus token
            const currentTokens = [...inputIds, ...generatedTokens];
            await this.forward([...currentTokens, bonusToken], true);
            forwardPasses++;

            shouldStop = appendToken(bonusToken);
            tokenCount++;
            if (shouldStop) break;
          }

          // Update n-gram cache for future use
          if (ngramCache) {
            ngramCache.update([...inputIds, ...generatedTokens], inputIds.length);
          }

          continue;
        }
      }

      // Fallback: single token generation (no speculation)
      const allIds = [...inputIds, ...generatedTokens];
      const logitsTensor = await this.forwardTensor(allIds, true);
      forwardPasses++;

      const nextToken = await this.sampleTokenGPU(logitsTensor, temperature, topK);
      logitsTensor.destroy();
      const shouldStop = appendToken(nextToken);
      tokenCount++;

      if (ngramCache) {
        ngramCache.update([...inputIds, ...generatedTokens], inputIds.length);
      }

      if (shouldStop) break;
    }

    // Clear cache after generation to free GPU memory
    this.clearKVCache();

    // Calculate and log performance metrics
    const genEndTime = performance.now();
    const genTimeSeconds = (genEndTime - genStartTime) / 1000;
    const tokensPerSecond = generatedTokens.length / genTimeSeconds;
    const tokensPerForward = generatedTokens.length / Math.max(1, forwardPasses);

    debugLog(`[Generate] ${generatedTokens.length} tokens in ${genTimeSeconds.toFixed(2)}s = ${tokensPerSecond.toFixed(1)} tok/s`);
    debugLog(`[Generate] ${forwardPasses} full passes, ${draftPasses} draft passes, ${tokensPerForward.toFixed(2)} tokens/forward`);

    // Log early-exit speculation statistics
    if (earlyExitProposed > 0) {
      const acceptRate = ((earlyExitAccepted / earlyExitProposed) * 100).toFixed(1);
      debugLog(`[EarlyExit] ${earlyExitAccepted}/${earlyExitProposed} accepted (${acceptRate}%), ${DRAFT_LAYERS} layers, ${DRAFT_TOKENS} drafts/iter`);
    }

    // Log N-gram speculative decoding statistics
    if (ngramCache) {
      const stats = ngramCache.getStats();
      debugLog(`[Ngram] hitRate=${(stats.hitRate * 100).toFixed(1)}%, acceptRate=${(stats.acceptanceRate * 100).toFixed(1)}%, accepted=${stats.acceptedTokens}/${stats.proposedTokens}`);
    }

    // Log command batcher statistics
    const batcherStats = getCommandBatcherStats();
    if (batcherStats && batcherStats.totalBatches > 0) {
      debugLog(`[Batcher] ${batcherStats.totalBatches} batches, ${batcherStats.totalPasses} passes, avg ${batcherStats.avgPassesPerBatch.toFixed(1)} passes/batch`);
    }
    resetCommandBatcherStats();

    return {
      text: generatedText,
      tokensGenerated: generatedTokens.length,
      done: true,
    };
  }

  /**
   * Apply chat template to format the user prompt using Jinja2/nunjucks
   * Templates expect a 'messages' array with {role, content} objects
   */
  private applyChatTemplate(prompt: string): string {
    if (!this.chatTemplate) {
      return prompt;
    }

    // Build messages array in the format expected by chat templates
    const messages = [
      { role: 'user', content: prompt }
    ];

    // Template context variables (matching HuggingFace chat template conventions)
    const context = {
      messages,
      add_generation_prompt: true,
      bos_token: this.tokenizer?.getToken(this.tokenizer.bosTokenId) || '<s>',
      eos_token: this.tokenizer?.getToken(this.tokenizer.eosTokenId) || '</s>',
    };

    try {
      const formatted = nunjucksEnv.renderString(this.chatTemplate, context);
      return formatted;
    } catch (error) {
      console.warn('Failed to render chat template:', error);
      // Fallback: return prompt as-is
      return prompt;
    }
  }

  /**
   * Apply chat template to format multiple messages (for session-based chat)
   */
  private applyChatTemplateForMessages(messages: ChatMessage[], addGenerationPrompt = true): string {
    if (!this.chatTemplate) {
      // Fallback: concatenate messages
      return messages.map(m => `${m.role}: ${m.content}`).join('\n');
    }

    const context = {
      messages,
      add_generation_prompt: addGenerationPrompt,
      bos_token: this.tokenizer?.getToken(this.tokenizer.bosTokenId) || '<s>',
      eos_token: this.tokenizer?.getToken(this.tokenizer.eosTokenId) || '</s>',
    };

    try {
      return nunjucksEnv.renderString(this.chatTemplate, context);
    } catch (error) {
      console.warn('Failed to render chat template:', error);
      return messages.map(m => `${m.role}: ${m.content}`).join('\n');
    }
  }

  /**
   * Create a new chat session for persistent KV cache across messages
   * This enables fast follow-up messages by only prefilling new tokens
   */
  createSession(options: ChatSessionOptions = {}): ChatSession {
    // End any existing session
    if (this.activeSession) {
      this.endSession();
    }

    const session: ChatSession = {
      id: `session_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
      messages: [],
      cachedTokenIds: [],
      cachedSeqLen: 0,
      createdAt: Date.now(),
    };

    // Add system prompt if provided
    if (options.systemPrompt) {
      session.messages.push({ role: 'system', content: options.systemPrompt });
    }

    this.activeSession = session;
    debugLog(`[Session] Created session ${session.id}`);
    return session;
  }

  /**
   * End the current chat session and free KV cache
   */
  endSession(): void {
    if (this.activeSession) {
      debugLog(`[Session] Ending session ${this.activeSession.id}`);
      this.activeSession = null;
      this.clearKVCache();
    }
  }

  /**
   * Get the current active session (if any)
   */
  getActiveSession(): ChatSession | null {
    return this.activeSession;
  }

  /**
   * Chat within a session - preserves KV cache for fast follow-up messages
   * Only prefills NEW tokens, reusing cached tokens from previous messages
   */
  async sessionChat(
    message: string,
    options: WebGPUGenerateOptions = {}
  ): Promise<{ text: string; tokensGenerated: number; prefillTokens: number; cachedTokens: number }> {
    if (!this.modelLoaded) {
      throw new Error('Model not loaded');
    }

    // Create session if none exists
    if (!this.activeSession) {
      this.createSession();
    }

    const session = this.activeSession!;
    const {
      maxTokens = 256,
      temperature = 0.8,
      topK = 40,
      stop = [],
    } = options;

    // Add user message to history
    session.messages.push({ role: 'user', content: message });

    // Format full conversation with chat template
    const fullPrompt = this.applyChatTemplateForMessages(session.messages, true);

    // Tokenize full conversation
    let fullTokenIds: number[];
    if (this.tokenizer) {
      fullTokenIds = this.tokenizer.encode(fullPrompt);
    } else {
      fullTokenIds = Array.from(fullPrompt).map(c => c.charCodeAt(0) % this.vocabSize);
    }

    // Calculate which tokens are new (not yet in cache)
    const cachedLen = session.cachedSeqLen;
    const newTokenIds = fullTokenIds.slice(cachedLen);
    const cachedTokens = cachedLen;
    const prefillTokens = newTokenIds.length;

    debugLog(`[Session] Full context: ${fullTokenIds.length} tokens, cached: ${cachedTokens}, new prefill: ${prefillTokens}`);

    // Check if we need to reallocate cache (context overflow)
    const totalNeeded = fullTokenIds.length + maxTokens;
    if (this.kvCache && totalNeeded > this.kvCache.maxSeqLen) {
      debugLog(`[Session] Context overflow: need ${totalNeeded}, max ${this.kvCache.maxSeqLen}. Clearing cache.`);
      this.clearKVCache();
      session.cachedTokenIds = [];
      session.cachedSeqLen = 0;
    }

    // Allocate cache if needed (first message or after overflow)
    if (!this.kvCache) {
      const maxSeqLen = Math.max(this.contextLength, totalNeeded + 256);
      this.kvCache = this.createGPUKVCache(maxSeqLen);
      // Need to prefill from start
      session.cachedTokenIds = [];
      session.cachedSeqLen = 0;
    }

    // Prefill: forwardTensor will automatically use incremental mode if kvCache.seqLen > 0
    // Pass full token sequence - it will only process tokens beyond kvCache.seqLen
    if (fullTokenIds.length > (this.kvCache?.seqLen ?? 0)) {
      const prefillStart = performance.now();
      await this.forwardTensor(fullTokenIds, true);
      const prefillTime = performance.now() - prefillStart;
      debugLog(`[Session] Prefill took ${prefillTime.toFixed(1)}ms for ${prefillTokens} new tokens (cache had ${cachedTokens})`);
    }

    // Update cached state
    session.cachedTokenIds = fullTokenIds.slice();
    session.cachedSeqLen = fullTokenIds.length;

    // Now generate response tokens
    const generatedTokens: number[] = [];
    let generatedText = '';
    const eosId = this.tokenizer?.eosTokenId ?? 2;
    const genStart = performance.now();

    while (generatedTokens.length < maxTokens) {
      // Get all token IDs including generated ones
      const allIds = [...session.cachedTokenIds, ...generatedTokens];

      // Forward pass for next token
      const logitsTensor = await this.forwardTensor(allIds, true);
      const nextToken = await this.sampleTokenGPU(logitsTensor, temperature, topK);
      logitsTensor.destroy();

      generatedTokens.push(nextToken);

      // Decode token
      let tokenText: string;
      if (this.tokenizer) {
        tokenText = this.tokenizer.decodeToken(nextToken);
      } else {
        tokenText = String.fromCharCode(nextToken % 128);
      }
      generatedText += tokenText;

      // Check for EOS
      if (nextToken === eosId) break;

      // Check for stop sequences
      let shouldStop = false;
      for (const stopSeq of stop) {
        if (generatedText.endsWith(stopSeq)) {
          generatedText = generatedText.slice(0, -stopSeq.length);
          shouldStop = true;
          break;
        }
      }
      if (shouldStop) break;
    }

    const genTime = performance.now() - genStart;
    const tokPerSec = generatedTokens.length / (genTime / 1000);
    debugLog(`[Session] Generated ${generatedTokens.length} tokens in ${genTime.toFixed(0)}ms (${tokPerSec.toFixed(1)} tok/s)`);

    // Add assistant response to history
    session.messages.push({ role: 'assistant', content: generatedText });

    // Update cache to include generated tokens
    session.cachedTokenIds.push(...generatedTokens);
    session.cachedSeqLen += generatedTokens.length;

    return {
      text: generatedText,
      tokensGenerated: generatedTokens.length,
      prefillTokens,
      cachedTokens,
    };
  }

  /**
   * OpenAI/Claude-compatible chat completion API
   * Takes full messages array and intelligently caches conversation prefix
   *
   * @example
   * ```typescript
   * const result = await engine.chatCompletion({
   *   messages: [
   *     { role: 'system', content: 'You are helpful' },
   *     { role: 'user', content: 'Hello!' },
   *     { role: 'assistant', content: 'Hi there!' },
   *     { role: 'user', content: 'How are you?' }  // Only this gets prefilled on follow-up
   *   ],
   *   maxTokens: 256,
   *   temperature: 0.7
   * });
   * ```
   */
  async chatCompletion(options: {
    messages: ChatMessage[];
    maxTokens?: number;
    temperature?: number;
    topK?: number;
    topP?: number;
    stop?: string[];
  }): Promise<{
    message: ChatMessage;
    tokensGenerated: number;
    prefillTokens: number;
    cachedTokens: number;
    totalTokens: number;
  }> {
    if (!this.modelLoaded) {
      throw new Error('Model not loaded');
    }

    const {
      messages,
      maxTokens = 256,
      temperature = 0.8,
      topK = 40,
      stop = [],
    } = options;

    if (!messages || messages.length === 0) {
      throw new Error('Messages array cannot be empty');
    }

    // Format conversation with chat template
    const fullPrompt = this.applyChatTemplateForMessages(messages, true);

    // Tokenize full conversation
    let fullTokenIds: number[];
    if (this.tokenizer) {
      fullTokenIds = this.tokenizer.encode(fullPrompt);
    } else {
      fullTokenIds = Array.from(fullPrompt).map(c => c.charCodeAt(0) % this.vocabSize);
    }

    // Check if we can reuse cached tokens
    // Compare token prefix to detect continuation vs new conversation
    let cachedTokens = 0;

    if (this.kvCache && this.activeSession && this.kvCache.seqLen > 0) {
      // Check how many tokens match the cached prefix
      const cachedIds = this.activeSession.cachedTokenIds;
      let matchLen = 0;
      for (let i = 0; i < Math.min(cachedIds.length, fullTokenIds.length); i++) {
        if (cachedIds[i] === fullTokenIds[i]) {
          matchLen++;
        } else {
          break;
        }
      }

      // If significant prefix matches, reuse cache
      if (matchLen > 0 && matchLen >= cachedIds.length * 0.8) {
        // Good match - reuse cache up to matchLen
        cachedTokens = Math.min(matchLen, this.kvCache.seqLen);
        debugLog(`[ChatCompletion] Cache hit: ${cachedTokens}/${fullTokenIds.length} tokens matched`);
      } else if (matchLen < cachedIds.length * 0.5) {
        // Poor match - new conversation, clear cache
        debugLog(`[ChatCompletion] Cache miss: only ${matchLen}/${cachedIds.length} tokens matched, clearing cache`);
        this.clearKVCache();
        if (this.activeSession) {
          this.activeSession.cachedTokenIds = [];
          this.activeSession.cachedSeqLen = 0;
        }
      }
    }

    const prefillTokens = fullTokenIds.length - cachedTokens;
    debugLog(`[ChatCompletion] Total: ${fullTokenIds.length} tokens, cached: ${cachedTokens}, prefill: ${prefillTokens}`);

    // Ensure we have a session for tracking
    if (!this.activeSession) {
      this.createSession();
    }

    // Allocate or check cache capacity
    const totalNeeded = fullTokenIds.length + maxTokens;
    if (!this.kvCache || totalNeeded > this.kvCache.maxSeqLen) {
      if (this.kvCache) {
        debugLog(`[ChatCompletion] Cache overflow, reallocating`);
        this.clearKVCache();
      }
      const maxSeqLen = Math.max(this.contextLength, totalNeeded + 256);
      this.kvCache = this.createGPUKVCache(maxSeqLen);
      cachedTokens = 0; // Must prefill everything
    }

    // Prefill if needed - capture logits for first token generation
    let prefillLogits: Tensor | null = null;
    if (fullTokenIds.length > (this.kvCache?.seqLen ?? 0)) {
      const prefillStart = performance.now();
      prefillLogits = await this.forwardTensor(fullTokenIds, true);
      const prefillTime = performance.now() - prefillStart;
      debugLog(`[ChatCompletion] Prefill took ${prefillTime.toFixed(1)}ms`);
    }

    // Update session state
    this.activeSession!.messages = [...messages];
    this.activeSession!.cachedTokenIds = fullTokenIds.slice();
    this.activeSession!.cachedSeqLen = fullTokenIds.length;

    // Generate response
    const generatedTokens: number[] = [];
    let generatedText = '';
    const eosId = this.tokenizer?.eosTokenId ?? 2;
    const genStart = performance.now();

    while (generatedTokens.length < maxTokens) {
      const tTok = performance.now();
      let logitsTensor: Tensor;
      let fwdTime: number;

      if (prefillLogits) {
        // Use logits from prefill for first token
        logitsTensor = prefillLogits;
        prefillLogits = null;
        fwdTime = 0;
      } else {
        // Incremental decode for subsequent tokens
        const allIds = [...fullTokenIds, ...generatedTokens];
        const tFwd = performance.now();
        logitsTensor = await this.forwardTensor(allIds, true);
        fwdTime = performance.now() - tFwd;
      }

      const tSamp = performance.now();
      const nextToken = await this.sampleTokenGPU(logitsTensor, temperature, topK);
      const sampTime = performance.now() - tSamp;
      logitsTensor.destroy();

      const tokTime = performance.now() - tTok;

      // Log timing for every 5th token
      if (generatedTokens.length % 5 === 0) {
        debugLog(`[Token#${generatedTokens.length}] total=${tokTime.toFixed(0)}ms fwd=${fwdTime.toFixed(0)}ms sample=${sampTime.toFixed(0)}ms`);
      }

      generatedTokens.push(nextToken);

      let tokenText: string;
      if (this.tokenizer) {
        tokenText = this.tokenizer.decodeToken(nextToken);
      } else {
        tokenText = String.fromCharCode(nextToken % 128);
      }
      generatedText += tokenText;

      if (nextToken === eosId) break;

      let shouldStop = false;
      for (const stopSeq of stop) {
        if (generatedText.endsWith(stopSeq)) {
          generatedText = generatedText.slice(0, -stopSeq.length);
          shouldStop = true;
          break;
        }
      }
      if (shouldStop) break;
    }

    const genTime = performance.now() - genStart;
    const tokPerSec = generatedTokens.length / (genTime / 1000);
    debugLog(`[ChatCompletion] Generated ${generatedTokens.length} tokens in ${genTime.toFixed(0)}ms (${tokPerSec.toFixed(1)} tok/s)`);

    // Update cache with generated tokens
    this.activeSession!.cachedTokenIds.push(...generatedTokens);
    this.activeSession!.cachedSeqLen += generatedTokens.length;
    this.activeSession!.messages.push({ role: 'assistant', content: generatedText });

    return {
      message: { role: 'assistant', content: generatedText },
      tokensGenerated: generatedTokens.length,
      prefillTokens,
      cachedTokens,
      totalTokens: fullTokenIds.length + generatedTokens.length,
    };
  }

  /**
   * Streaming chat completion with KV cache persistence
   * Yields tokens as they're generated, caches conversation for fast follow-ups
   *
   * @example
   * ```typescript
   * for await (const token of engine.chatCompletionStream({
   *   messages: [
   *     { role: 'system', content: 'You are helpful' },
   *     { role: 'user', content: 'Hello!' }
   *   ],
   *   maxTokens: 256
   * })) {
   *   process.stdout.write(token);
   * }
   * ```
   */
  async *chatCompletionStream(options: {
    messages: ChatMessage[];
    maxTokens?: number;
    temperature?: number;
    topK?: number;
    stop?: string[];
  }): AsyncGenerator<string, void, unknown> {
    if (!this.modelLoaded) {
      throw new Error('Model not loaded');
    }

    const {
      messages,
      maxTokens = 256,
      temperature = 0.8,
      topK = 40,
      stop = [],
    } = options;

    if (!messages || messages.length === 0) {
      throw new Error('Messages array cannot be empty');
    }

    // Format conversation with chat template
    const fullPrompt = this.applyChatTemplateForMessages(messages, true);

    // Tokenize full conversation
    let fullTokenIds: number[];
    if (this.tokenizer) {
      fullTokenIds = this.tokenizer.encode(fullPrompt);
    } else {
      fullTokenIds = Array.from(fullPrompt).map(c => c.charCodeAt(0) % this.vocabSize);
    }

    // Check if we can reuse cached tokens
    let cachedTokens = 0;

    if (this.kvCache && this.activeSession && this.kvCache.seqLen > 0) {
      const cachedIds = this.activeSession.cachedTokenIds;
      let matchLen = 0;
      for (let i = 0; i < Math.min(cachedIds.length, fullTokenIds.length); i++) {
        if (cachedIds[i] === fullTokenIds[i]) {
          matchLen++;
        } else {
          break;
        }
      }

      if (matchLen > 0 && matchLen >= cachedIds.length * 0.8) {
        cachedTokens = Math.min(matchLen, this.kvCache.seqLen);
        debugLog(`[ChatStream] Cache hit: ${cachedTokens}/${fullTokenIds.length} tokens matched`);
      } else if (matchLen < cachedIds.length * 0.5) {
        debugLog(`[ChatStream] Cache miss: clearing cache`);
        this.clearKVCache();
        if (this.activeSession) {
          this.activeSession.cachedTokenIds = [];
          this.activeSession.cachedSeqLen = 0;
        }
      }
    }

    debugLog(`[ChatStream] Total: ${fullTokenIds.length} tokens, cached: ${cachedTokens}, prefill: ${fullTokenIds.length - cachedTokens}`);

    // Ensure we have a session for tracking
    if (!this.activeSession) {
      this.createSession();
    }

    // Allocate or check cache capacity
    const totalNeeded = fullTokenIds.length + maxTokens;
    if (!this.kvCache || totalNeeded > this.kvCache.maxSeqLen) {
      if (this.kvCache) {
        this.clearKVCache();
      }
      const maxSeqLen = Math.max(this.contextLength, totalNeeded + 256);
      this.kvCache = this.createGPUKVCache(maxSeqLen);
      cachedTokens = 0;
    }

    // Prefill if needed - capture logits for first token generation
    let prefillLogits: Tensor | null = null;
    if (fullTokenIds.length > (this.kvCache?.seqLen ?? 0)) {
      prefillLogits = await this.forwardTensor(fullTokenIds, true);
    }

    // Update session state
    this.activeSession!.messages = [...messages];
    this.activeSession!.cachedTokenIds = fullTokenIds.slice();
    this.activeSession!.cachedSeqLen = fullTokenIds.length;

    // Generate and yield tokens
    const generatedTokens: number[] = [];
    let generatedText = '';
    const eosId = this.tokenizer?.eosTokenId ?? 2;

    // Add default stop sequences for Qwen
    const allStop = [...stop];
    const arch = this.architecture?.architecture?.toLowerCase();
    if ((arch === 'qwen2' || arch === 'qwen') && !allStop.includes('<|im_end|>')) {
      allStop.push('<|im_end|>');
    }

    while (generatedTokens.length < maxTokens) {
      let logitsTensor: Tensor;

      if (prefillLogits) {
        // Use logits from prefill for first token
        logitsTensor = prefillLogits;
        prefillLogits = null;
      } else {
        // Incremental decode for subsequent tokens
        const allIds = [...fullTokenIds, ...generatedTokens];
        logitsTensor = await this.forwardTensor(allIds, true);
      }
      const nextToken = await this.sampleTokenGPU(logitsTensor, temperature, topK);
      logitsTensor.destroy();

      generatedTokens.push(nextToken);

      let tokenText: string;
      if (this.tokenizer) {
        tokenText = this.tokenizer.decodeToken(nextToken);
      } else {
        tokenText = String.fromCharCode(nextToken % 128);
      }
      generatedText += tokenText;

      // Check for EOS
      if (nextToken === eosId) break;

      // Check for stop sequences
      let shouldStop = false;
      for (const stopSeq of allStop) {
        if (generatedText.endsWith(stopSeq)) {
          // Don't yield the stop sequence
          const stopLen = stopSeq.length;
          if (generatedText.length > stopLen) {
            // Remove stop sequence from end
            generatedText = generatedText.slice(0, -stopLen);
          }
          shouldStop = true;
          break;
        }
      }
      if (shouldStop) break;

      // Yield the token
      yield tokenText;
    }

    // Update cache with generated tokens
    this.activeSession!.cachedTokenIds.push(...generatedTokens);
    this.activeSession!.cachedSeqLen += generatedTokens.length;
    this.activeSession!.messages.push({ role: 'assistant', content: generatedText });
  }

  /**
   * Generate text with streaming and GPU-resident KV cache for O(n) generation
   */
  async *generateStream(
    prompt: string,
    options: WebGPUGenerateOptions = {}
  ): AsyncGenerator<string, void, unknown> {
    if (!this.modelLoaded) {
      throw new Error('Model not loaded');
    }

    const {
      maxTokens = 64,
      temperature = 0.8,
      topK = 40,
      stop: userStop = [],
      rawPrompt = false,
    } = options;

    // Clear any existing KV cache for fresh generation
    this.clearKVCache();

    // Add default stop sequences for known architectures
    const stop = [...userStop];
    const arch = this.architecture?.architecture?.toLowerCase();
    if ((arch === 'qwen2' || arch === 'qwen') && !stop.includes('<|im_end|>')) {
      stop.push('<|im_end|>');
    }

    // Apply chat template unless rawPrompt is set
    const formattedPrompt = rawPrompt ? prompt : this.applyChatTemplate(prompt);

    if (!rawPrompt && this.chatTemplate) {
      debugLog(`[Chat] Using chat template, formatted prompt length: ${formattedPrompt.length}`);
    }

    // Tokenize input
    let inputIds: number[];
    if (this.tokenizer) {
      inputIds = this.tokenizer.encode(formattedPrompt);
      debugLog(`[Chat] Tokenized prompt: ${inputIds.length} tokens`);
    } else {
      inputIds = Array.from(formattedPrompt).map((c) => c.charCodeAt(0) % this.vocabSize);
    }

    const generatedTokens: number[] = [];
    let generatedText = '';

    try {
      for (let i = 0; i < maxTokens; i++) {
        // Use KV cache: first iteration does prefill, subsequent do incremental decode
        const allIds = [...inputIds, ...generatedTokens];
        let nextToken: number;

        // Debug mode uses CPU sampling to inspect logits
        if (this.debugMode) {
          const logits = await this.forward(allIds, true);

          // Debug: show logits distribution for first 3 tokens
          if (i < 3) {
            let minLogit = Infinity, maxLogit = -Infinity, sum = 0;
            for (let j = 0; j < logits.length; j++) {
              if (logits[j] < minLogit) minLogit = logits[j];
              if (logits[j] > maxLogit) maxLogit = logits[j];
              sum += logits[j];
            }
            const mean = sum / logits.length;

            const indexed = Array.from(logits).map((v, idx) => ({ v, idx }));
            indexed.sort((a, b) => b.v - a.v);
            const top5 = indexed.slice(0, 5);

            debugLog(`[Gen ${i}] Logits: min=${minLogit.toFixed(2)}, max=${maxLogit.toFixed(2)}, mean=${mean.toFixed(4)}`);
            debugLog(`[Gen ${i}] Top 5: ${top5.map(t => `${t.idx}(${t.v.toFixed(2)})`).join(', ')}`);
            if (this.tokenizer) {
              debugLog(`[Gen ${i}] Top 5 tokens: ${top5.map(t => this.tokenizer!.getToken(t.idx)).join(', ')}`);
            }
          }
          nextToken = this.sampleToken(logits, temperature, topK);
        } else {
          // Normal mode: GPU-accelerated sampling (avoids expensive toArray)
          const logitsTensor = await this.forwardTensor(allIds, true);
          nextToken = await this.sampleTokenGPU(logitsTensor, temperature, topK);
          logitsTensor.destroy();
        }

        generatedTokens.push(nextToken);

        // Decode and yield the token
        let tokenText: string;
        if (this.tokenizer) {
          tokenText = this.tokenizer.decodeToken(nextToken);
        } else {
          tokenText = String.fromCharCode(nextToken % 128);
        }

        generatedText += tokenText;
        yield tokenText;

        // Check for EOS
        const eosId = this.tokenizer?.eosTokenId ?? 2;
        if (nextToken === eosId) {
          break;
        }

        // Check for stop sequences
        let shouldStop = false;
        for (const stopSeq of stop) {
          if (generatedText.endsWith(stopSeq)) {
            shouldStop = true;
            break;
          }
        }
        if (shouldStop) {
          break;
        }
      }
    } finally {
      // Clear cache after generation to free GPU memory
      this.clearKVCache();

      // Log command batcher statistics
      const batcherStats = getCommandBatcherStats();
      if (batcherStats && batcherStats.totalBatches > 0) {
        debugLog(`[Batcher] ${batcherStats.totalBatches} batches, ${batcherStats.totalPasses} passes, avg ${batcherStats.avgPassesPerBatch.toFixed(1)} passes/batch`);
      }
      resetCommandBatcherStats();
    }
  }

  /**
   * Add bias to a 2D tensor (broadcast add) - GPU accelerated
   * input: [seqLen, dim], bias: [dim] -> output: [seqLen, dim]
   */
  private async addBias(input: Tensor, bias: Tensor): Promise<Tensor> {
    // Use GPU broadcast add - bias is broadcast across seqLen dimension
    const result = await ops.broadcastAdd(input, bias);
    input.destroy();
    return result;
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
   * Benchmark GPU scaling with different token counts
   * Measures forward pass time to determine if parallel speculation will help
   */
  async benchmark(): Promise<void> {
    if (!this.modelLoaded || !this.weights) {
      throw new Error('Model not loaded for benchmark');
    }

    debugLog('\n========== GPU SCALING BENCHMARK ==========');
    debugLog('Measuring forward pass time with different token counts...\n');

    // Create a dummy prompt
    const dummyTokens = Array.from({ length: 512 }, (_, i) => i % this.vocabSize);

    const tokenCounts = [1, 4, 16, 64, 128, 256];
    const results: { tokens: number; timeMs: number; tokPerSec: number }[] = [];

    for (const numTokens of tokenCounts) {
      // Clear cache for fresh measurement
      this.clearKVCache();

      const tokens = dummyTokens.slice(0, numTokens);

      // Warm up run
      await this.forward(tokens, true);
      this.clearKVCache();

      // Timed runs (average of 3)
      const times: number[] = [];
      for (let run = 0; run < 3; run++) {
        this.clearKVCache();
        const start = performance.now();
        await this.forward(tokens, true);
        const end = performance.now();
        times.push(end - start);
      }

      const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
      const tokPerSec = (numTokens / avgTime) * 1000;

      results.push({ tokens: numTokens, timeMs: avgTime, tokPerSec });
      debugLog(`${numTokens.toString().padStart(3)} tokens: ${avgTime.toFixed(1)}ms (${tokPerSec.toFixed(0)} tok/s)`);
    }

    // Analyze scaling
    debugLog('\n--- SCALING ANALYSIS ---');
    const baseline = results[0];
    for (const r of results.slice(1)) {
      const actualRatio = r.timeMs / baseline.timeMs;
      const efficiency = r.tokens / actualRatio;
      debugLog(`${r.tokens} tokens: ${actualRatio.toFixed(1)}x time for ${r.tokens}x tokens → ${efficiency.toFixed(1)}x effective parallelism`);
    }

    // Recommendation
    const best = results.reduce((a, b) => a.tokPerSec > b.tokPerSec ? a : b);
    debugLog(`\n✓ OPTIMAL: ${best.tokens} tokens (${best.tokPerSec.toFixed(0)} tok/s)`);

    if (best.tokens > 16) {
      debugLog('→ GPU parallelism IS helping. Tree speculation will improve speed.');
    } else {
      debugLog('→ GPU is memory-bound. More tokens won\'t help much.');
    }

    debugLog('==========================================\n');

    // Clean up
    this.clearKVCache();
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
    // Clear KV cache
    this.clearKVCache();

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

    // Reset all cached pipelines before destroying device
    // This is critical when switching models/devices
    resetAllPipelines();

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
