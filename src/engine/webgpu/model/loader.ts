/**
 * GGUF Model Loader for WebGPU
 * Loads GGUF model weights to GPU tensors with dequantization
 */

import { open, type FileHandle } from 'fs/promises';
import { Tensor } from '../tensor.js';
import {
  dequantize,
  getBytesPerBlock,
  getBlockSize,
  requiresDequantization,
  QuantizedTensor,
} from '../quant/index.js';
import { transpose } from '../ops/index.js';
import type { GGUFFile, GGUFTensorInfo, GGMLType } from '../../../types/model.js';
import { GGMLType as GGMLTypeEnum } from '../../../types/model.js';
import { debugLog, isDebugEnabled } from '../debug.js';

/**
 * Options for loading tensors
 */
export interface TensorLoadOptions {
  /** Whether to dequantize quantized tensors to f32 (default: true) */
  dequantize?: boolean;
  /**
   * Keep weights in quantized format on GPU (for supported types: Q8_0, Q4_0, Q4_K).
   * When true, returns QuantizedTensor instead of dequantizing to f32.
   * This saves VRAM but requires using quantized GEMV for computation.
   */
  keepQuantized?: boolean;
  /** Maximum number of elements to load (for memory limits) */
  maxElements?: number;
  /** Labels to use for GPU tensors */
  label?: string;
}

/**
 * A weight that can be either f32 Tensor or quantized
 */
export type WeightTensor = Tensor | QuantizedTensor;

/**
 * Information about a loaded tensor
 */
export interface LoadedTensor {
  name: string;
  tensor: Tensor;
  originalType: GGMLType;
  shape: number[];
}

/**
 * Information about a loaded weight (may be quantized)
 */
export interface LoadedWeight {
  name: string;
  weight: WeightTensor;
  originalType: GGMLType;
  shape: number[];
  isQuantized: boolean;
}

/**
 * Calculate the total bytes for a tensor
 */
export function calculateTensorBytes(info: GGUFTensorInfo): number {
  const numElements = info.dimensions.reduce(
    (a, b) => Number(a) * Number(b),
    1
  );

  if (info.type === GGMLTypeEnum.F32) {
    return numElements * 4;
  } else if (info.type === GGMLTypeEnum.F16) {
    return numElements * 2;
  } else {
    // Quantized types
    const blockSize = getBlockSize(info.type);
    const bytesPerBlock = getBytesPerBlock(info.type);
    const numBlocks = Math.ceil(numElements / blockSize);
    return numBlocks * bytesPerBlock;
  }
}

/**
 * Calculate total number of elements in a tensor
 */
export function calculateTensorElements(info: GGUFTensorInfo): number {
  return info.dimensions.reduce((a, b) => Number(a) * Number(b), 1);
}

/**
 * Load a single tensor from a GGUF file
 *
 * GGUF stores tensors in column-major order (like Fortran).
 * For a 2D tensor with GGUF dimensions [ne0, ne1], the element at logical
 * position (i, j) is at memory index: i + j*ne0
 *
 * This means in row-major terms (like C/JavaScript), the data is laid out
 * as a [ne1, ne0] matrix. We handle this by:
 * 1. Reshaping the flat data to [ne1, ne0] (which correctly interprets the column-major data)
 * 2. Returning both the tensor and a flag indicating if transpose is needed for matmul
 */
export async function loadTensor(
  handle: FileHandle,
  tensorInfo: GGUFTensorInfo,
  tensorDataOffset: bigint,
  options: TensorLoadOptions = {}
): Promise<LoadedTensor> {
  const { dequantize: shouldDequantize = true, label } = options;

  // Calculate dimensions and size
  // GGUF dimensions are [ne0, ne1, ...] in column-major order
  const ggufShape = tensorInfo.dimensions.map(Number);
  const numElements = calculateTensorElements(tensorInfo);
  const numBytes = calculateTensorBytes(tensorInfo);

  // For 2D tensors: GGUF [ne0, ne1] column-major = [ne1, ne0] row-major
  // The data is stored such that elements of the first dimension (ne0) are contiguous
  const shape = ggufShape.length === 2 ? [ggufShape[1], ggufShape[0]] : ggufShape;

  // Read raw tensor data from file
  const buffer = Buffer.alloc(numBytes);
  const fileOffset = Number(tensorDataOffset) + Number(tensorInfo.offset);

  await handle.read(buffer, 0, numBytes, fileOffset);

  const data = new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.length);

  // Create tensor (with optional dequantization)
  let tensor: Tensor;

  if (shouldDequantize && requiresDequantization(tensorInfo.type)) {
    // Dequantize to f32 on GPU
    tensor = await dequantize(data, numElements, tensorInfo.type);
    // Reshape to row-major dimensions
    if (shape.length > 1) {
      tensor = tensor.reshape(shape);
    }
  } else if (tensorInfo.type === GGMLTypeEnum.F32) {
    // Already f32, load directly
    // Create a copy to avoid alignment issues with Buffer's internal pool
    const alignedBuffer = new ArrayBuffer(numElements * 4);
    const alignedView = new Uint8Array(alignedBuffer);
    alignedView.set(data.subarray(0, numElements * 4));
    const f32Data = new Float32Array(alignedBuffer);

    // Debug: print first few F32 values for bias tensors
    if (tensorInfo.name.includes('bias') && tensorInfo.name.includes('blk.0')) {
      debugLog(`[DEBUG F32] ${tensorInfo.name}:`);
      debugLog(`  raw bytes[0-15]: [${Array.from(data.slice(0, 16)).join(', ')}]`);
      debugLog(`  f32 values[0-3]: [${Array.from(f32Data.slice(0, 4)).map(v => v.toFixed(6)).join(', ')}]`);
    }

    tensor = Tensor.fromData(f32Data, shape, { label: label || tensorInfo.name });
  } else if (tensorInfo.type === GGMLTypeEnum.F16) {
    // Convert f16 to f32
    tensor = await dequantize(data, numElements, tensorInfo.type);
    if (shape.length > 1) {
      tensor = tensor.reshape(shape);
    }
  } else {
    // Keep quantized (store raw bytes for later GPU dequantization)
    // For now, just dequantize
    tensor = await dequantize(data, numElements, tensorInfo.type);
    if (shape.length > 1) {
      tensor = tensor.reshape(shape);
    }
  }

  return {
    name: tensorInfo.name,
    tensor,
    originalType: tensorInfo.type,
    shape,
  };
}

/**
 * Load a weight tensor, optionally keeping it in quantized format.
 *
 * For supported quantization types (Q8_0, Q4_0, Q4_K), this stores
 * the raw quantized bytes on GPU instead of dequantizing to f32.
 * This reduces VRAM usage by 4-8x but requires quantized GEMV for computation.
 *
 * Note: The shape returned is the GGUF shape interpreted as row-major,
 * which is [cols, rows] for 2D tensors. The quantized GEMV handles this
 * transposed layout directly.
 */
export async function loadWeight(
  handle: FileHandle,
  tensorInfo: GGUFTensorInfo,
  tensorDataOffset: bigint,
  options: TensorLoadOptions = {}
): Promise<LoadedWeight> {
  const { keepQuantized = false, label } = options;

  // Calculate dimensions
  const ggufShape = tensorInfo.dimensions.map(Number);
  const numBytes = calculateTensorBytes(tensorInfo);

  // Check if we should keep this tensor quantized
  const canKeepQuantized = keepQuantized && QuantizedTensor.isSupported(tensorInfo.type);

  // GGUF stores 2D weight tensors with shape [ne0, ne1] = [K, N] where:
  //   ne0 = input dimension (K), which is the contiguous/stride-1 dimension
  //   ne1 = output dimension (N)
  // This matches what we need for matmul x @ W where x is [M, K] and W is [K, N].
  //
  // For QUANTIZED tensors: keep original GGUF shape [K, N].
  //   The shader uses linearIdx = n * K + k to match GGUF's storage order.
  //
  // For F32 tensors: swap to [N, K] then transpose later to get proper [K, N] layout.
  //   (The dequantization + transpose path needs the swap.)
  const shape = (ggufShape.length === 2 && !canKeepQuantized)
    ? [ggufShape[1], ggufShape[0]]
    : ggufShape;

  // Read raw tensor data from file
  const buffer = Buffer.alloc(numBytes);
  const fileOffset = Number(tensorDataOffset) + Number(tensorInfo.offset);
  await handle.read(buffer, 0, numBytes, fileOffset);
  const data = new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.length);

  if (canKeepQuantized) {
    // Store as QuantizedTensor (no dequantization!)
    // Shape is [K, N] where K = input dim, N = output dim (same as GGUF [ne0, ne1])
    const quantTensor = QuantizedTensor.fromQuantizedData(
      data,
      shape,
      tensorInfo.type as Parameters<typeof QuantizedTensor.fromQuantizedData>[2],
      { label: label || tensorInfo.name }
    );

    return {
      name: tensorInfo.name,
      weight: quantTensor,
      originalType: tensorInfo.type,
      shape,
      isQuantized: true,
    };
  }

  // Fall back to dequantizing (original behavior)
  const loaded = await loadTensor(handle, tensorInfo, tensorDataOffset, options);

  return {
    name: loaded.name,
    weight: loaded.tensor,
    originalType: loaded.originalType,
    shape: loaded.shape,
    isQuantized: false,
  };
}

/**
 * Load multiple tensors from a GGUF file
 */
export async function loadTensors(
  filePath: string,
  ggufFile: GGUFFile,
  tensorNames: string[],
  options: TensorLoadOptions = {}
): Promise<Map<string, LoadedTensor>> {
  const handle = await open(filePath, 'r');
  const results = new Map<string, LoadedTensor>();

  try {
    // Create a set for quick lookup
    const nameSet = new Set(tensorNames);

    // Find matching tensors
    for (const tensorInfo of ggufFile.tensors) {
      if (nameSet.has(tensorInfo.name)) {
        const loaded = await loadTensor(
          handle,
          tensorInfo,
          ggufFile.tensorDataOffset,
          { ...options, label: tensorInfo.name }
        );
        results.set(tensorInfo.name, loaded);
      }
    }

    return results;
  } finally {
    await handle.close();
  }
}

/**
 * Load all tensors matching a pattern
 */
export async function loadTensorsByPattern(
  filePath: string,
  ggufFile: GGUFFile,
  pattern: RegExp,
  options: TensorLoadOptions = {}
): Promise<Map<string, LoadedTensor>> {
  const handle = await open(filePath, 'r');
  const results = new Map<string, LoadedTensor>();

  try {
    for (const tensorInfo of ggufFile.tensors) {
      if (pattern.test(tensorInfo.name)) {
        const loaded = await loadTensor(
          handle,
          tensorInfo,
          ggufFile.tensorDataOffset,
          { ...options, label: tensorInfo.name }
        );
        results.set(tensorInfo.name, loaded);
      }
    }

    return results;
  } finally {
    await handle.close();
  }
}

/**
 * Estimate total GPU memory needed for model tensors
 */
export function estimateModelMemory(
  ggufFile: GGUFFile,
  dequantize = true
): number {
  let totalBytes = 0;

  for (const tensor of ggufFile.tensors) {
    const numElements = calculateTensorElements(tensor);

    if (dequantize) {
      // All tensors will be f32 after dequantization
      totalBytes += numElements * 4;
    } else {
      totalBytes += calculateTensorBytes(tensor);
    }
  }

  return totalBytes;
}

/**
 * Get tensor info by name
 */
export function getTensorInfo(
  ggufFile: GGUFFile,
  name: string
): GGUFTensorInfo | undefined {
  return ggufFile.tensors.find((t) => t.name === name);
}

/**
 * List all tensor names in a GGUF file
 */
export function listTensorNames(ggufFile: GGUFFile): string[] {
  return ggufFile.tensors.map((t) => t.name);
}

/**
 * Group tensors by layer
 * Returns a map of layer index to tensor names
 */
export function groupTensorsByLayer(
  ggufFile: GGUFFile
): Map<number | 'embed' | 'output', string[]> {
  const groups = new Map<number | 'embed' | 'output', string[]>();

  for (const tensor of ggufFile.tensors) {
    const name = tensor.name;

    // Match layer patterns like "blk.0", "layers.0", etc.
    const layerMatch = name.match(/(?:blk|layers?)\.(\d+)/);

    if (layerMatch) {
      const layerIdx = parseInt(layerMatch[1], 10);
      if (!groups.has(layerIdx)) {
        groups.set(layerIdx, []);
      }
      groups.get(layerIdx)!.push(name);
    } else if (name.includes('embed') || name.includes('token_embd')) {
      if (!groups.has('embed')) {
        groups.set('embed', []);
      }
      groups.get('embed')!.push(name);
    } else if (name.includes('output') || name.includes('lm_head')) {
      if (!groups.has('output')) {
        groups.set('output', []);
      }
      groups.get('output')!.push(name);
    }
  }

  return groups;
}

/**
 * Model weights organizer for Llama-style models
 *
 * Large projection weights (attnQ/K/V/Output, ffnGate/Up/Down, outputWeight)
 * can be either f32 Tensor or QuantizedTensor depending on load options.
 *
 * Small weights (norms, biases, embeddings) are always kept as f32 Tensor.
 */
export interface LlamaWeights {
  tokenEmbedding: Tensor | null;
  layers: LlamaLayerWeights[];
  outputNorm: Tensor | null;
  outputWeight: WeightTensor | null;  // Can be quantized
}

export interface LlamaLayerWeights {
  // Norms are always f32 (small, 1D)
  attnNorm: Tensor | null;
  ffnNorm: Tensor | null;

  // Attention projections (can be quantized)
  attnQ: WeightTensor | null;
  attnK: WeightTensor | null;
  attnV: WeightTensor | null;
  attnOutput: WeightTensor | null;

  // Biases are always f32 (small, often not present)
  attnQBias: Tensor | null;
  attnKBias: Tensor | null;
  attnVBias: Tensor | null;

  // FFN projections (can be quantized - these are the largest!)
  ffnGate: WeightTensor | null;
  ffnUp: WeightTensor | null;
  ffnDown: WeightTensor | null;
}

/**
 * Load Llama-style model weights
 */
export async function loadLlamaWeights(
  filePath: string,
  ggufFile: GGUFFile,
  options: TensorLoadOptions = {}
): Promise<LlamaWeights> {
  const handle = await open(filePath, 'r');

  try {
    // Determine the number of layers from metadata or tensor names
    const layerGroups = groupTensorsByLayer(ggufFile);
    const layerIndices = Array.from(layerGroups.keys())
      .filter((k): k is number => typeof k === 'number')
      .sort((a, b) => a - b);
    const numLayers = layerIndices.length;

    const weights: LlamaWeights = {
      tokenEmbedding: null,
      layers: [],
      outputNorm: null,
      outputWeight: null,
    };

    // Helper to find tensor info by name pattern
    const findByPattern = (patterns: string[]): GGUFTensorInfo | null => {
      for (const pattern of patterns) {
        const info = ggufFile.tensors.find((t) => t.name.includes(pattern));
        if (info) return info;
      }
      return null;
    };

    // Helper to load a tensor by name pattern (always f32)
    const loadByPattern = async (
      patterns: string[]
    ): Promise<Tensor | null> => {
      const info = findByPattern(patterns);
      if (info) {
        const loaded = await loadTensor(
          handle,
          info,
          ggufFile.tensorDataOffset,
          { ...options, keepQuantized: false }
        );
        return loaded.tensor;
      }
      return null;
    };

    // Helper to load weight tensor - may be quantized or f32+transposed
    const loadWeightByPattern = async (
      patterns: string[]
    ): Promise<WeightTensor | null> => {
      const info = findByPattern(patterns);
      if (!info) return null;

      if (options.keepQuantized && QuantizedTensor.isSupported(info.type)) {
        // Load as QuantizedTensor (no transpose)
        const loaded = await loadWeight(
          handle,
          info,
          ggufFile.tensorDataOffset,
          { ...options, keepQuantized: true }
        );
        return loaded.weight;
      } else {
        // Load as f32 and transpose
        const loaded = await loadTensor(
          handle,
          info,
          ggufFile.tensorDataOffset,
          options
        );
        if (loaded.tensor.ndim === 2) {
          const transposed = await transpose(loaded.tensor);
          loaded.tensor.destroy();
          return transposed;
        }
        return loaded.tensor;
      }
    };

    // Load embedding (keep original format - embedding lookup handles it)
    weights.tokenEmbedding = await loadByPattern([
      'token_embd.weight',
      'tok_embeddings.weight',
      'embed_tokens.weight',
    ]);

    // Debug: check embedding tensor
    if (isDebugEnabled() && weights.tokenEmbedding) {
      const embData = await weights.tokenEmbedding.toArray();
      debugLog(`[Embedding] shape=[${weights.tokenEmbedding.shape.join(', ')}]`);
      // Check first token (id 0) and a random token
      const hiddenSize = weights.tokenEmbedding.shape[1];
      const token0 = embData.slice(0, 8);
      const token1 = embData.slice(hiddenSize, hiddenSize + 8);
      const token1000 = embData.slice(1000 * hiddenSize, 1000 * hiddenSize + 8);
      // Also check token 151644 which is the first token in test input
      const token151644 = embData.slice(151644 * hiddenSize, 151644 * hiddenSize + 8);
      debugLog(`  Token 0 first 8: [${Array.from(token0).map(v => v.toFixed(4)).join(', ')}]`);
      debugLog(`  Token 1 first 8: [${Array.from(token1).map(v => v.toFixed(4)).join(', ')}]`);
      debugLog(`  Token 1000 first 8: [${Array.from(token1000).map(v => v.toFixed(4)).join(', ')}]`);
      debugLog(`  Token 151644 first 8: [${Array.from(token151644).map(v => v.toFixed(4)).join(', ')}]`);
      debugLog(`  Token 151644 indices: ${151644 * hiddenSize} to ${151644 * hiddenSize + 7}`);
    }

    // Load output norm (1D, no transpose needed)
    weights.outputNorm = await loadByPattern([
      'output_norm.weight',
      'norm.weight',
      'model.norm.weight',
    ]);

    // Load output weight (may be quantized or f32+transposed)
    weights.outputWeight = await loadWeightByPattern([
      'output.weight',
      'lm_head.weight',
    ]);

    // Debug: check output weight statistics
    if (isDebugEnabled() && weights.outputWeight) {
      if (weights.outputWeight instanceof Tensor) {
        const outData = await weights.outputWeight.toArray();
        let sum = 0, sumSq = 0, min = Infinity, max = -Infinity;
        const sampleSize = Math.min(100000, outData.length);
        for (let j = 0; j < sampleSize; j++) {
          sum += outData[j];
          sumSq += outData[j] * outData[j];
          if (outData[j] < min) min = outData[j];
          if (outData[j] > max) max = outData[j];
        }
        const mean = sum / sampleSize;
        const variance = sumSq / sampleSize - mean * mean;
        debugLog(`[Output Weight] shape=[${weights.outputWeight.shape.join(', ')}] (f32)`);
        debugLog(`  mean=${mean.toFixed(6)}, std=${Math.sqrt(variance).toFixed(6)}, min=${min.toFixed(4)}, max=${max.toFixed(4)}`);
      } else {
        const qt = weights.outputWeight as QuantizedTensor;
        const stats = qt.getMemoryStats();
        debugLog(`[Output Weight] shape=[${qt.shape.join(', ')}] (${QuantizedTensor.getTypeName(qt.quantType)})`);
        debugLog(`  ${(stats.quantizedBytes / 1024 / 1024).toFixed(1)} MB quantized, ${stats.compressionRatio.toFixed(1)}x compression`);
      }
    }

    // Load layer weights
    for (let i = 0; i < numLayers; i++) {
      const layer: LlamaLayerWeights = {
        attnNorm: null,
        attnQ: null,
        attnQBias: null,
        attnK: null,
        attnKBias: null,
        attnV: null,
        attnVBias: null,
        attnOutput: null,
        ffnNorm: null,
        ffnGate: null,
        ffnUp: null,
        ffnDown: null,
      };

      // Find tensors for this layer
      const layerPrefix = `blk.${i}`;
      const altPrefix = `layers.${i}`;

      // Debug: print GGUF tensor info for layer 0
      if (i === 0 && isDebugEnabled()) {
        const typeNames: { [key: number]: string } = {
          0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 6: 'Q5_0', 7: 'Q5_1',
          8: 'Q8_0', 9: 'Q8_1', 10: 'Q2_K', 11: 'Q3_K', 12: 'Q4_K',
          13: 'Q5_K', 14: 'Q6_K', 15: 'Q8_K'
        };
        debugLog('[Layer 0] GGUF tensor info:');
        for (const t of ggufFile.tensors) {
          if (t.name.includes('blk.0') &&
              (t.name.includes('attn_q') || t.name.includes('attn_k') || t.name.includes('attn_v'))) {
            const typeName = typeNames[t.type] || `type${t.type}`;
            debugLog(`  ${t.name}: ${typeName}, dims=[${t.dimensions.map(Number).join(', ')}]`);
          }
        }
      }

      // Helper to find tensor info by suffix patterns
      const findTensorInfo = (suffixes: string[]): GGUFTensorInfo | null => {
        for (const suffix of suffixes) {
          const name = `${layerPrefix}.${suffix}`;
          const altName = `${altPrefix}.${suffix}`;

          let info = ggufFile.tensors.find(
            (t) => t.name === name || t.name === altName
          );
          if (!info) {
            // Try partial match
            info = ggufFile.tensors.find(
              (t) =>
                (t.name.includes(layerPrefix) || t.name.includes(altPrefix)) &&
                suffixes.some((s) => t.name.includes(s))
            );
          }
          if (info) return info;
        }
        return null;
      };

      // Load a tensor (always dequantized to f32) - for norms and biases
      const loadLayerTensor = async (
        suffixes: string[]
      ): Promise<Tensor | null> => {
        const info = findTensorInfo(suffixes);
        if (info) {
          const loaded = await loadTensor(
            handle,
            info,
            ggufFile.tensorDataOffset,
            { ...options, keepQuantized: false }  // Force f32 for small tensors
          );
          return loaded.tensor;
        }
        return null;
      };

      // Helper to load weight matrices - may be quantized or f32+transposed
      const loadLayerWeight = async (
        suffixes: string[]
      ): Promise<WeightTensor | null> => {
        const info = findTensorInfo(suffixes);
        if (!info) return null;

        if (options.keepQuantized && QuantizedTensor.isSupported(info.type)) {
          // Load as QuantizedTensor (no transpose - GEMV handles the layout)
          const loaded = await loadWeight(
            handle,
            info,
            ggufFile.tensorDataOffset,
            { ...options, keepQuantized: true }
          );
          return loaded.weight;
        } else {
          // Load as f32 and transpose for standard matmul
          const loaded = await loadTensor(
            handle,
            info,
            ggufFile.tensorDataOffset,
            options
          );
          if (loaded.tensor.ndim === 2) {
            const transposed = await transpose(loaded.tensor);
            loaded.tensor.destroy();
            return transposed;
          }
          return loaded.tensor;
        }
      };

      // Norm weights are 1D, no transpose needed
      layer.attnNorm = await loadLayerTensor(['attn_norm.weight', 'input_layernorm.weight']);
      // Attention weights need transpose for matmul
      layer.attnQ = await loadLayerWeight(['attn_q.weight', 'self_attn.q_proj.weight']);
      layer.attnQBias = await loadLayerTensor(['attn_q.bias', 'self_attn.q_proj.bias']);
      layer.attnK = await loadLayerWeight(['attn_k.weight', 'self_attn.k_proj.weight']);
      layer.attnKBias = await loadLayerTensor(['attn_k.bias', 'self_attn.k_proj.bias']);
      layer.attnV = await loadLayerWeight(['attn_v.weight', 'self_attn.v_proj.weight']);
      layer.attnVBias = await loadLayerTensor(['attn_v.bias', 'self_attn.v_proj.bias']);
      layer.attnOutput = await loadLayerWeight(['attn_output.weight', 'self_attn.o_proj.weight']);
      // FFN norm is 1D
      layer.ffnNorm = await loadLayerTensor(['ffn_norm.weight', 'post_attention_layernorm.weight']);
      // FFN weights need transpose
      layer.ffnGate = await loadLayerWeight(['ffn_gate.weight', 'mlp.gate_proj.weight']);
      layer.ffnUp = await loadLayerWeight(['ffn_up.weight', 'mlp.up_proj.weight']);
      layer.ffnDown = await loadLayerWeight(['ffn_down.weight', 'mlp.down_proj.weight']);

      // Debug: print GGUF dimensions vs final shape for layer 0
      if (i === 0 && isDebugEnabled()) {
        const printShape = async (name: string, weight: WeightTensor | null, patterns: string[]) => {
          if (!weight) return;
          const isQuant = weight instanceof QuantizedTensor;
          const shape = weight.shape;
          // Find original GGUF tensor info
          for (const pattern of patterns) {
            const info = ggufFile.tensors.find(t =>
              t.name.includes(`blk.0`) && t.name.includes(pattern.split('.')[0])
            );
            if (info) {
              const typeSuffix = isQuant ? ` (${QuantizedTensor.getTypeName((weight as QuantizedTensor).quantType)})` : ' (f32)';
              debugLog(`  ${name}: GGUF dims=[${info.dimensions.join(', ')}] -> final shape=[${shape.join(', ')}]${typeSuffix}`);
              break;
            }
          }
        };
        debugLog('[Layer 0] GGUF dimensions vs final shapes:');
        await printShape('attnQ', layer.attnQ, ['attn_q.weight']);
        await printShape('attnK', layer.attnK, ['attn_k.weight']);
        await printShape('attnV', layer.attnV, ['attn_v.weight']);
        await printShape('attnOutput', layer.attnOutput, ['attn_output.weight']);
        await printShape('ffnGate', layer.ffnGate, ['ffn_gate.weight']);
        await printShape('ffnUp', layer.ffnUp, ['ffn_up.weight']);
        await printShape('ffnDown', layer.ffnDown, ['ffn_down.weight']);
      }

      // Debug: print weight shapes and statistics for layer 0
      if (i === 0 && isDebugEnabled()) {
        debugLog(`[Layer 0] Weight shapes and statistics:`);
        // Only print detailed stats for f32 tensors (not quantized)
        if (layer.attnQ && layer.attnQ instanceof Tensor) {
          const qData = await layer.attnQ.toArray();
          let sum = 0, sumSq = 0, min = Infinity, max = -Infinity;
          for (let j = 0; j < qData.length; j++) {
            sum += qData[j];
            sumSq += qData[j] * qData[j];
            if (qData[j] < min) min = qData[j];
            if (qData[j] > max) max = qData[j];
          }
          const mean = sum / qData.length;
          const variance = sumSq / qData.length - mean * mean;
          debugLog(`  attnQ: [${layer.attnQ.shape.join(', ')}] (f32)`);
          debugLog(`    mean=${mean.toFixed(6)}, std=${Math.sqrt(variance).toFixed(6)}, min=${min.toFixed(4)}, max=${max.toFixed(4)}`);
          debugLog(`    first row (8 vals): [${Array.from(qData.slice(0, 8)).map(v => v.toFixed(4)).join(', ')}]`);
          // Also print column sum for first column
          let col0Sum = 0;
          for (let r = 0; r < layer.attnQ.shape[0]; r++) {
            col0Sum += qData[r * layer.attnQ.shape[1]];
          }
          debugLog(`    column 0 sum: ${col0Sum.toFixed(4)}`);
        } else if (layer.attnQ && layer.attnQ instanceof QuantizedTensor) {
          const qt = layer.attnQ;
          const stats = qt.getMemoryStats();
          debugLog(`  attnQ: [${qt.shape.join(', ')}] (${QuantizedTensor.getTypeName(qt.quantType)})`);
          debugLog(`    ${(stats.quantizedBytes / 1024).toFixed(1)} KB, ${stats.compressionRatio.toFixed(1)}x compression`);
        }
        if (layer.attnK && layer.attnK instanceof Tensor) {
          const kData = await layer.attnK.toArray();
          let sum = 0, sumSq = 0, min = Infinity, max = -Infinity;
          for (let j = 0; j < kData.length; j++) {
            sum += kData[j];
            sumSq += kData[j] * kData[j];
            if (kData[j] < min) min = kData[j];
            if (kData[j] > max) max = kData[j];
          }
          const mean = sum / kData.length;
          const variance = sumSq / kData.length - mean * mean;
          debugLog(`  attnK: [${layer.attnK.shape.join(', ')}] (f32)`);
          debugLog(`    mean=${mean.toFixed(6)}, std=${Math.sqrt(variance).toFixed(6)}, min=${min.toFixed(4)}, max=${max.toFixed(4)}`);
          debugLog(`    first 8: [${Array.from(kData.slice(0, 8)).map(v => v.toFixed(4)).join(', ')}]`);
        } else if (layer.attnK && layer.attnK instanceof QuantizedTensor) {
          const qt = layer.attnK;
          const stats = qt.getMemoryStats();
          debugLog(`  attnK: [${qt.shape.join(', ')}] (${QuantizedTensor.getTypeName(qt.quantType)})`);
          debugLog(`    ${(stats.quantizedBytes / 1024).toFixed(1)} KB, ${stats.compressionRatio.toFixed(1)}x compression`);
        }
        if (layer.attnV && layer.attnV instanceof Tensor) {
          const vData = await layer.attnV.toArray();
          let sum = 0, sumSq = 0, min = Infinity, max = -Infinity;
          for (let j = 0; j < vData.length; j++) {
            sum += vData[j];
            sumSq += vData[j] * vData[j];
            if (vData[j] < min) min = vData[j];
            if (vData[j] > max) max = vData[j];
          }
          const mean = sum / vData.length;
          const variance = sumSq / vData.length - mean * mean;
          debugLog(`  attnV: [${layer.attnV.shape.join(', ')}] (f32)`);
          debugLog(`    mean=${mean.toFixed(6)}, std=${Math.sqrt(variance).toFixed(6)}, min=${min.toFixed(4)}, max=${max.toFixed(4)}`);
        } else if (layer.attnV && layer.attnV instanceof QuantizedTensor) {
          const qt = layer.attnV;
          const stats = qt.getMemoryStats();
          debugLog(`  attnV: [${qt.shape.join(', ')}] (${QuantizedTensor.getTypeName(qt.quantType)})`);
          debugLog(`    ${(stats.quantizedBytes / 1024).toFixed(1)} KB, ${stats.compressionRatio.toFixed(1)}x compression`);
        }
        // Check biases (always f32)
        if (layer.attnKBias) {
          const biasData = await layer.attnKBias.toArray();
          let sum = 0, sumSq = 0, min = Infinity, max = -Infinity;
          for (let j = 0; j < biasData.length; j++) {
            sum += biasData[j];
            sumSq += biasData[j] * biasData[j];
            if (biasData[j] < min) min = biasData[j];
            if (biasData[j] > max) max = biasData[j];
          }
          const mean = sum / biasData.length;
          const variance = sumSq / biasData.length - mean * mean;
          debugLog(`  attnKBias: [${layer.attnKBias.shape.join(', ')}], mean=${mean.toFixed(4)}, std=${Math.sqrt(variance).toFixed(4)}, min=${min.toFixed(4)}, max=${max.toFixed(4)}`);
        }
      }

      weights.layers.push(layer);
    }

    return weights;
  } finally {
    await handle.close();
  }
}

/**
 * Dispose of all tensors in LlamaWeights
 */
export function disposeLlamaWeights(weights: LlamaWeights): void {
  if (weights.tokenEmbedding) weights.tokenEmbedding.destroy();
  if (weights.outputNorm) weights.outputNorm.destroy();
  if (weights.outputWeight) weights.outputWeight.destroy();

  for (const layer of weights.layers) {
    if (layer.attnNorm) layer.attnNorm.destroy();
    if (layer.attnQ) layer.attnQ.destroy();
    if (layer.attnQBias) layer.attnQBias.destroy();
    if (layer.attnK) layer.attnK.destroy();
    if (layer.attnKBias) layer.attnKBias.destroy();
    if (layer.attnV) layer.attnV.destroy();
    if (layer.attnVBias) layer.attnVBias.destroy();
    if (layer.attnOutput) layer.attnOutput.destroy();
    if (layer.ffnNorm) layer.ffnNorm.destroy();
    if (layer.ffnGate) layer.ffnGate.destroy();
    if (layer.ffnUp) layer.ffnUp.destroy();
    if (layer.ffnDown) layer.ffnDown.destroy();
  }
}
