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
  applyRope,
  feedForward,
  loadLlamaWeights,
  disposeLlamaWeights,
  estimateModelMemory,
  type LlamaWeights,
  type LlamaLayerWeights,
} from './webgpu/index.js';
import { extractArchitecture, getChatTemplate } from '../model/gguf.js';
import type { GGUFFile, ModelArchitecture } from '../types/model.js';
import { Tokenizer, parseGGUFWithTokenizer } from '../tokenizer/index.js';

// Configure nunjucks for Jinja2 compatibility
const nunjucksEnv = new nunjucks.Environment(null, { autoescape: false });
// Add 'raise_exception' filter used by some HuggingFace templates
nunjucksEnv.addFilter('raise_exception', (msg: string) => { throw new Error(msg); });
// Add 'tojson' filter for JSON serialization
nunjucksEnv.addFilter('tojson', (obj: unknown) => JSON.stringify(obj));

export interface WebGPUEngineOptions {
  numThreads?: number; // Ignored for WebGPU (GPU handles parallelism)
  debug?: boolean; // Enable verbose debug output (slower due to GPU-CPU transfers)
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
  rawPrompt?: boolean; // If true, skip chat template formatting
}

// Use LlamaWeights from model loader
// Re-export for backward compatibility
type ModelWeights = LlamaWeights;

/**
 * KV Cache for storing key/value projections across generation steps
 * This avoids recomputing attention for previously processed tokens
 */
interface KVCache {
  // Each layer has its own K and V cache
  // Shape: [seqLen, numKVHeads * headDim]
  keys: Float32Array[];    // One per layer
  values: Float32Array[];  // One per layer
  seqLen: number;          // Current cached sequence length
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

  // KV cache for efficient incremental generation
  private kvCache: KVCache | null = null;

  constructor(options: WebGPUEngineOptions = {}) {
    super();
    this.debugMode = options.debug ?? false;
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
      // Parse GGUF file with full tokenizer data (no array size limit)
      console.log('Parsing GGUF file with tokenizer data...');
      this.ggufFile = await parseGGUFWithTokenizer(modelPath);
      this.architecture = extractArchitecture(this.ggufFile);

      // Extract model configuration from metadata
      this.setupModelConfig(options);

      // Initialize tokenizer from GGUF metadata
      try {
        this.tokenizer = Tokenizer.fromGGUF(this.ggufFile);
        console.log(`Tokenizer loaded: ${this.tokenizer.vocabSize} tokens`);
        console.log(`Special tokens: BOS=${this.tokenizer.bosTokenId}, EOS=${this.tokenizer.eosTokenId}`);
      } catch (tokenizerError) {
        console.warn('Could not load tokenizer from GGUF:', tokenizerError);
        this.tokenizer = null;
      }

      // Load chat template from GGUF metadata (Jinja2 format, rendered with nunjucks)
      const ggufTemplate = getChatTemplate(this.ggufFile);
      const arch = this.architecture?.architecture?.toLowerCase();

      if (ggufTemplate) {
        this.chatTemplate = ggufTemplate;
        console.log('Chat template loaded from GGUF metadata (Jinja2 format)');
      } else if (arch === 'qwen2' || arch === 'qwen') {
        // Fallback Qwen2 template if not in GGUF
        this.chatTemplate = '{% for message in messages %}{% if message.role == "user" %}<|im_start|>user\n{{ message.content }}<|im_end|>\n<|im_start|>assistant\n{% endif %}{% endfor %}';
        console.log('Using fallback Qwen2 chat template');
      } else if (arch === 'llama' || arch === 'llama2') {
        // Fallback Llama 2 template
        this.chatTemplate = '{% for message in messages %}{% if message.role == "user" %}[INST] {{ message.content }} [/INST]{% endif %}{% endfor %}';
        console.log('Using fallback Llama2 chat template');
      } else {
        this.chatTemplate = null;
        console.log('No chat template configured - prompts will be used as-is');
      }

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

    console.log(`Model config: hidden=${this.hiddenSize}, heads=${this.numHeads}, kv_heads=${this.numKVHeads}, layers=${this.numLayers}`);
    console.log(`RoPE base=${this.ropeBase}, RMS norm eps=${this.rmsNormEps}`);
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

    console.log(`Placeholder weights loaded (${numTestLayers} layers)`);
  }

  /**
   * Create a new empty KV cache
   */
  private createKVCache(): KVCache {
    const numLayers = this.weights?.layers.length || this.numLayers;
    return {
      keys: Array(numLayers).fill(null).map(() => new Float32Array(0)),
      values: Array(numLayers).fill(null).map(() => new Float32Array(0)),
      seqLen: 0,
    };
  }

  /**
   * Clear the KV cache (call when starting a new generation)
   */
  private clearKVCache(): void {
    this.kvCache = null;
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
    console.log(`[Stats ${label}] n=${data.length}, mean=${mean.toFixed(6)}, std=${std.toFixed(6)}, min=${min.toFixed(4)}, max=${max.toFixed(4)}, nan=${nanCount}, inf=${infCount}, sum=${sum.toFixed(2)}`);
  }

  /**
   * Forward pass through the model
   * @param inputIds - Token IDs to process
   * @param useCache - If true, use KV cache for incremental generation (only process last token)
   */
  private async forward(inputIds: number[], useCache = false): Promise<Float32Array> {
    if (!this.weights) {
      throw new Error('Model weights not loaded');
    }

    const fullSeqLen = inputIds.length;
    const DEBUG = this.debugMode;

    // For cached inference, we only process new tokens
    // startPos is where new tokens start (= cached sequence length)
    let startPos = 0;
    let tokensToProcess: number[];

    if (useCache && this.kvCache && this.kvCache.seqLen > 0) {
      // Incremental mode: only process the last token
      startPos = this.kvCache.seqLen;
      tokensToProcess = inputIds.slice(startPos);
      if (DEBUG) {
        console.log(`\n[Forward] Incremental: processing ${tokensToProcess.length} new token(s) at pos ${startPos}`);
      }
    } else {
      // Full mode: process all tokens (prefill)
      tokensToProcess = inputIds;
      // Initialize or reset cache
      if (useCache) {
        this.kvCache = this.createKVCache();
      }
      if (DEBUG) {
        console.log(`\n[Forward] Prefill: processing ${tokensToProcess.length} tokens`);
      }
    }

    const processLen = tokensToProcess.length;

    // Debug: log input token IDs for first call
    if (DEBUG) {
      console.log(`[Forward] seqLen=${fullSeqLen}, first 5 tokens: [${inputIds.slice(0, 5).join(', ')}], last 5 tokens: [${inputIds.slice(-5).join(', ')}]`);
    }

    // Create input embeddings only for tokens we need to process
    // GGUF stores embeddings as [hiddenSize, vocabSize], pass vocabSize hint
    let hidden: Tensor;
    if (this.weights.tokenEmbedding) {
      hidden = await ops.embeddingLookup(this.weights.tokenEmbedding, tokensToProcess, this.vocabSize);

      // Debug: check if embeddings are different per position
      if (DEBUG) {
        const embData = await hidden.toArray();
        this.debugStats(embData, 'Embedding');
        const pos0 = Array.from(embData.slice(0, 5));
        const posLast = Array.from(embData.slice(-this.hiddenSize, -this.hiddenSize + 5));
        console.log(`[Forward] Embedding pos 0: [${pos0.map((v: number) => v.toFixed(4)).join(', ')}]`);
        console.log(`[Forward] Embedding pos ${processLen - 1}: [${posLast.map((v: number) => v.toFixed(4)).join(', ')}]`);
      }
    } else {
      // Fallback to random embeddings (shouldn't happen with real model)
      console.warn('No token embedding found, using random');
      hidden = Tensor.random([processLen, this.hiddenSize], { label: 'hidden' });
    }

    // Process through all layers
    for (let layerIdx = 0; layerIdx < this.weights.layers.length; layerIdx++) {
      const layer = this.weights.layers[layerIdx];

      // Skip layers without required weights
      if (!layer.attnNorm || !layer.ffnNorm) {
        continue;
      }

      // Pre-attention norm
      const normed = await rmsNorm(hidden, layer.attnNorm, this.rmsNormEps);

      // Debug: check normalized input for layer 0
      if (layerIdx === 0 && DEBUG) {
        const normedData = await normed.toArray();
        this.debugStats(normedData, 'L0 Normed');
        const sample = Array.from(normedData.slice(0, 8));
        console.log(`[Layer0] Normed input (pos0): [${sample.map(v => v.toFixed(4)).join(', ')}]`);
      }

      // Self-attention with Q, K, V projections
      let attnOut: Tensor;
      if (layer.attnQ && layer.attnK && layer.attnV && layer.attnOutput) {
        attnOut = await this.selfAttention(normed, layer, processLen, startPos, layerIdx, useCache);
      } else {
        // Fallback: identity (skip attention)
        console.warn(`Layer ${layerIdx}: missing attention weights, skipping`);
        attnOut = normed;
      }

      // Debug: check attention output for last position (layer 0 only)
      if (layerIdx === 0 && DEBUG) {
        const attnData = await attnOut.toArray();
        this.debugStats(attnData, 'L0 AttnOut');
        const lastPosAttn = Array.from(attnData.slice(-this.hiddenSize, -this.hiddenSize + 8));
        console.log(`[Layer0] Attn out last pos: [${lastPosAttn.map(v => v.toFixed(4)).join(', ')}]`);
      }

      // Residual connection
      const afterAttn = await ops.add(hidden, attnOut);
      hidden.destroy();
      if (attnOut !== normed) {
        normed.destroy();
        attnOut.destroy();
      }
      hidden = afterAttn;

      // Pre-FFN norm
      const normed2 = await rmsNorm(hidden, layer.ffnNorm, this.rmsNormEps);

      // FFN (use weights if available, otherwise skip)
      // GGUF weights are [in_features, out_features], correct for x @ W
      if (layer.ffnGate && layer.ffnUp && layer.ffnDown) {
        const ffnOut = await feedForward(normed2, layer.ffnGate, layer.ffnUp, layer.ffnDown);

        // Debug: check hidden after FFN for last position (layer 0 only)
        if (layerIdx === 0 && DEBUG) {
          const ffnData = await ffnOut.toArray();
          this.debugStats(ffnData, 'L0 FFNOut');
          const lastPosFfn = Array.from(ffnData.slice(-this.hiddenSize, -this.hiddenSize + 8));
          console.log(`[Layer0] FFN out last pos: [${lastPosFfn.map(v => v.toFixed(4)).join(', ')}]`);
        }

        // Residual connection
        const afterFfn = await ops.add(hidden, ffnOut);
        hidden.destroy();
        ffnOut.destroy();
        hidden = afterFfn;
      }
      normed2.destroy();

      // Debug: hidden state after layer 0
      if (layerIdx === 0 && DEBUG) {
        const hiddenData = await hidden.toArray();
        const lastPosHidden = Array.from(hiddenData.slice(-this.hiddenSize, -this.hiddenSize + 8));
        console.log(`[Layer0] Hidden after layer: [${lastPosHidden.map(v => v.toFixed(4)).join(', ')}]`);
      }
    }

    // Final norm
    const normWeight = this.weights.outputNorm || Tensor.ones([this.hiddenSize]);
    const finalNormed = await rmsNorm(hidden, normWeight, this.rmsNormEps);
    hidden.destroy();
    if (!this.weights.outputNorm) {
      normWeight.destroy();
    }

    // Extract last token's hidden state on GPU (avoids transferring entire sequence to CPU)
    const lastTokenHidden = await ops.sliceLastRow(finalNormed);

    // Debug: check final hidden state (only when debug enabled)
    if (DEBUG) {
      const lastTokenData = await lastTokenHidden.toArray();
      this.debugStats(lastTokenData, 'Final Hidden');
      const lastPosVals = Array.from(lastTokenData.slice(0, 8));
      console.log(`[Final] Last token hidden: [${lastPosVals.map(v => v.toFixed(4)).join(', ')}]`);
    }

    finalNormed.destroy();

    // Output projection
    // outputWeight is pre-transposed to [hiddenSize, vocabSize] = [2048, 151936]
    let logits: Tensor;
    if (this.weights.outputWeight) {
      logits = await ops.matmul(lastTokenHidden, this.weights.outputWeight);

      // Debug: check logits
      if (DEBUG) {
        const logitsArr = await logits.toArray();
        this.debugStats(logitsArr, 'Logits');
        console.log(`[DEBUG] Logit[0] = ${logitsArr[0].toFixed(4)}, logit[1] = ${logitsArr[1].toFixed(4)}`);
      }
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
   * Self-attention with Q, K, V projections
   * Supports Grouped Query Attention (GQA) where numKVHeads < numHeads
   * Supports KV caching for incremental generation
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
    _layerIdx: number,
    _useCache: boolean
  ): Promise<Tensor> {
    // Note: layerIdx and useCache are reserved for future GPU-based KV cache implementation

    // Project to Q, K, V for new tokens only
    // Q: [processLen, hiddenSize] @ [hiddenSize, numHeads * headDim] = [processLen, numHeads * headDim]
    let q = await ops.matmul(x, layer.attnQ!);
    // K: [processLen, hiddenSize] @ [hiddenSize, numKVHeads * headDim] = [processLen, numKVHeads * headDim]
    let k = await ops.matmul(x, layer.attnK!);
    // V: [processLen, hiddenSize] @ [hiddenSize, numKVHeads * headDim] = [processLen, numKVHeads * headDim]
    let vProj = await ops.matmul(x, layer.attnV!);

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

    // Apply RoPE to Q and K
    // IMPORTANT: RoPE needs absolute positions, not relative to batch
    // For incremental: Q positions are [startPos, startPos+1, ...], K same
    const qRope = await applyRope(q, processLen, this.numHeads, this.headDim, this.ropeBase, startPos);
    const kRope = await applyRope(k, processLen, this.numKVHeads, this.headDim, this.ropeBase, startPos);
    q.destroy();
    k.destroy();

    // Debug: check Q after RoPE
    if (debugRope && qBeforeRope) {
      const qData = await qRope.toArray();
      const kData = await kRope.toArray();
      this.debugStats(qData, 'Q post-RoPE');
      this.debugStats(kData, 'K post-RoPE');
      const qAfterRope = Array.from(qData.slice(0, 8));
      const qLastPosAfter = Array.from(qData.slice(-this.numHeads * this.headDim, -this.numHeads * this.headDim + 8));
      console.log(`[RoPE] Q pos0 before: [${qBeforeRope.map(v => v.toFixed(4)).join(', ')}]`);
      console.log(`[RoPE] Q pos0 after:  [${qAfterRope.map(v => v.toFixed(4)).join(', ')}]`);
      console.log(`[RoPE] Q pos${processLen - 1} after: [${qLastPosAfter.map(v => v.toFixed(4)).join(', ')}]`);
    }

    // For now, skip KV cache and compute attention directly on GPU
    // The CPU-based KV cache was causing too many GPU-CPU transfers
    // TODO: Implement GPU-based KV cache with pre-allocated buffers

    // GPU-accelerated causal attention
    // Q: [processLen, numHeads * headDim]
    // K: [processLen, numKVHeads * headDim] (just new keys for now)
    // V: [processLen, numKVHeads * headDim] (just new values for now)
    //
    // Note: Without proper GPU KV cache, we need to recompute K/V for full sequence
    // This is O(n²) but avoids the GPU-CPU transfer bottleneck
    const attnOut = await ops.causalAttention(
      qRope,
      kRope,
      vProj,
      this.numHeads,
      this.numKVHeads,
      this.headDim,
      startPos
    );

    // Cleanup
    qRope.destroy();
    kRope.destroy();
    vProj.destroy();

    // Output projection: [processLen, numHeads * headDim] @ [numHeads * headDim, hiddenSize]
    const projected = await ops.matmul(attnOut, layer.attnOutput!);
    attnOut.destroy();

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
      stop = [],
      rawPrompt = false,
    } = options;

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

    const generatedTokens: number[] = [];
    let generatedText = '';

    for (let i = 0; i < maxTokens; i++) {
      // Process full sequence each time (no KV cache for now - stays on GPU)
      const allIds = [...inputIds, ...generatedTokens];
      const logits = await this.forward(allIds, false);

      // Sample next token
      const nextToken = this.sampleToken(logits, temperature, topK);
      generatedTokens.push(nextToken);

      // Decode the token
      let tokenText: string;
      if (this.tokenizer) {
        tokenText = this.tokenizer.decodeToken(nextToken);
      } else {
        tokenText = String.fromCharCode(nextToken % 128);
      }

      generatedText += tokenText;

      // Check for EOS
      const eosId = this.tokenizer?.eosTokenId ?? 2;
      if (nextToken === eosId) {
        break;
      }

      // Check for stop sequences
      let shouldStop = false;
      for (const stopSeq of stop) {
        if (generatedText.endsWith(stopSeq)) {
          // Remove stop sequence from output
          generatedText = generatedText.slice(0, -stopSeq.length);
          shouldStop = true;
          break;
        }
      }
      if (shouldStop) break;
    }

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
   * Generate text with streaming
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

    // Add default stop sequences for known architectures
    const stop = [...userStop];
    const arch = this.architecture?.architecture?.toLowerCase();
    if ((arch === 'qwen2' || arch === 'qwen') && !stop.includes('<|im_end|>')) {
      stop.push('<|im_end|>');
    }

    // Apply chat template unless rawPrompt is set
    const formattedPrompt = rawPrompt ? prompt : this.applyChatTemplate(prompt);

    if (!rawPrompt && this.chatTemplate) {
      console.log(`[Chat] Using chat template, formatted prompt length: ${formattedPrompt.length}`);
    }

    // Tokenize input
    let inputIds: number[];
    if (this.tokenizer) {
      inputIds = this.tokenizer.encode(formattedPrompt);
      console.log(`[Chat] Tokenized prompt: ${inputIds.length} tokens`);
    } else {
      inputIds = Array.from(formattedPrompt).map((c) => c.charCodeAt(0) % this.vocabSize);
    }

    const generatedTokens: number[] = [];
    let generatedText = '';

    for (let i = 0; i < maxTokens; i++) {
      // Process full sequence each time (no KV cache for now - stays on GPU)
      const allIds = [...inputIds, ...generatedTokens];
      const logits = await this.forward(allIds, false);

      // Debug: show logits distribution for first 3 tokens
      if (this.debugMode && i < 3) {
        let minLogit = Infinity, maxLogit = -Infinity, sum = 0;
        for (let j = 0; j < logits.length; j++) {
          if (logits[j] < minLogit) minLogit = logits[j];
          if (logits[j] > maxLogit) maxLogit = logits[j];
          sum += logits[j];
        }
        const mean = sum / logits.length;

        // Find top 5 tokens
        const indexed = Array.from(logits).map((v, idx) => ({ v, idx }));
        indexed.sort((a, b) => b.v - a.v);
        const top5 = indexed.slice(0, 5);

        console.log(`[Gen ${i}] Logits: min=${minLogit.toFixed(2)}, max=${maxLogit.toFixed(2)}, mean=${mean.toFixed(4)}`);
        console.log(`[Gen ${i}] Top 5: ${top5.map(t => `${t.idx}(${t.v.toFixed(2)})`).join(', ')}`);
        if (this.tokenizer) {
          console.log(`[Gen ${i}] Top 5 tokens: ${top5.map(t => this.tokenizer!.getToken(t.idx)).join(', ')}`);
        }
      }

      // Sample next token
      const nextToken = this.sampleToken(logits, temperature, topK);
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
