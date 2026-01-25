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
} from '../quant/index.js';
import { transpose } from '../ops/index.js';
import type { GGUFFile, GGUFTensorInfo, GGMLType } from '../../../types/model.js';
import { GGMLType as GGMLTypeEnum } from '../../../types/model.js';

/**
 * Options for loading tensors
 */
export interface TensorLoadOptions {
  /** Whether to dequantize quantized tensors to f32 */
  dequantize?: boolean;
  /** Maximum number of elements to load (for memory limits) */
  maxElements?: number;
  /** Labels to use for GPU tensors */
  label?: string;
}

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
      console.log(`[DEBUG F32] ${tensorInfo.name}:`);
      console.log(`  raw bytes[0-15]: [${Array.from(data.slice(0, 16)).join(', ')}]`);
      console.log(`  f32 values[0-3]: [${Array.from(f32Data.slice(0, 4)).map(v => v.toFixed(6)).join(', ')}]`);
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
 */
export interface LlamaWeights {
  tokenEmbedding: Tensor | null;
  layers: LlamaLayerWeights[];
  outputNorm: Tensor | null;
  outputWeight: Tensor | null;
}

export interface LlamaLayerWeights {
  attnNorm: Tensor | null;
  attnQ: Tensor | null;
  attnQBias: Tensor | null;
  attnK: Tensor | null;
  attnKBias: Tensor | null;
  attnV: Tensor | null;
  attnVBias: Tensor | null;
  attnOutput: Tensor | null;
  ffnNorm: Tensor | null;
  ffnGate: Tensor | null;
  ffnUp: Tensor | null;
  ffnDown: Tensor | null;
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

    // Helper to load a tensor by name pattern
    const loadByPattern = async (
      patterns: string[]
    ): Promise<Tensor | null> => {
      for (const pattern of patterns) {
        const info = ggufFile.tensors.find((t) =>
          t.name.includes(pattern)
        );
        if (info) {
          const loaded = await loadTensor(
            handle,
            info,
            ggufFile.tensorDataOffset,
            options
          );
          return loaded.tensor;
        }
      }
      return null;
    };

    // Helper to load and transpose a 2D weight tensor
    // GGUF column-major [ne0, ne1] becomes row-major [ne1, ne0]
    // For matmul x @ W, we need W in [in_features, out_features] format
    const loadAndTransposeWeight = async (
      patterns: string[]
    ): Promise<Tensor | null> => {
      const tensor = await loadByPattern(patterns);
      if (tensor && tensor.ndim === 2) {
        const transposed = await transpose(tensor);
        tensor.destroy();
        return transposed;
      }
      return tensor;
    };

    // Load embedding (keep original format - embedding lookup handles it)
    weights.tokenEmbedding = await loadByPattern([
      'token_embd.weight',
      'tok_embeddings.weight',
      'embed_tokens.weight',
    ]);

    // Debug: check embedding tensor
    if (weights.tokenEmbedding) {
      const embData = await weights.tokenEmbedding.toArray();
      console.log(`[Embedding] shape=[${weights.tokenEmbedding.shape.join(', ')}]`);
      // Check first token (id 0) and a random token
      const hiddenSize = weights.tokenEmbedding.shape[1];
      const token0 = embData.slice(0, 8);
      const token1 = embData.slice(hiddenSize, hiddenSize + 8);
      const token1000 = embData.slice(1000 * hiddenSize, 1000 * hiddenSize + 8);
      // Also check token 151644 which is the first token in test input
      const token151644 = embData.slice(151644 * hiddenSize, 151644 * hiddenSize + 8);
      console.log(`  Token 0 first 8: [${Array.from(token0).map(v => v.toFixed(4)).join(', ')}]`);
      console.log(`  Token 1 first 8: [${Array.from(token1).map(v => v.toFixed(4)).join(', ')}]`);
      console.log(`  Token 1000 first 8: [${Array.from(token1000).map(v => v.toFixed(4)).join(', ')}]`);
      console.log(`  Token 151644 first 8: [${Array.from(token151644).map(v => v.toFixed(4)).join(', ')}]`);
      console.log(`  Token 151644 indices: ${151644 * hiddenSize} to ${151644 * hiddenSize + 7}`);
    }

    // Load output norm (1D, no transpose needed)
    weights.outputNorm = await loadByPattern([
      'output_norm.weight',
      'norm.weight',
      'model.norm.weight',
    ]);

    // Load output weight and transpose for matmul
    weights.outputWeight = await loadAndTransposeWeight([
      'output.weight',
      'lm_head.weight',
    ]);

    // Debug: check output weight statistics (Q6_K tensor)
    if (weights.outputWeight) {
      const outData = await weights.outputWeight.toArray();
      let sum = 0, sumSq = 0, min = Infinity, max = -Infinity;
      // Sample first 100k elements to avoid slowdown
      const sampleSize = Math.min(100000, outData.length);
      for (let j = 0; j < sampleSize; j++) {
        sum += outData[j];
        sumSq += outData[j] * outData[j];
        if (outData[j] < min) min = outData[j];
        if (outData[j] > max) max = outData[j];
      }
      const mean = sum / sampleSize;
      const variance = sumSq / sampleSize - mean * mean;
      console.log(`[Output Weight] shape=[${weights.outputWeight.shape.join(', ')}]`);
      console.log(`  mean=${mean.toFixed(6)}, std=${Math.sqrt(variance).toFixed(6)}, min=${min.toFixed(4)}, max=${max.toFixed(4)}`);
      console.log(`  first 8 vals: [${Array.from(outData.slice(0, 8)).map(v => v.toFixed(4)).join(', ')}]`);
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
      if (i === 0) {
        const typeNames: { [key: number]: string } = {
          0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 6: 'Q5_0', 7: 'Q5_1',
          8: 'Q8_0', 9: 'Q8_1', 10: 'Q2_K', 11: 'Q3_K', 12: 'Q4_K',
          13: 'Q5_K', 14: 'Q6_K', 15: 'Q8_K'
        };
        console.log('[Layer 0] GGUF tensor info:');
        for (const t of ggufFile.tensors) {
          if (t.name.includes('blk.0') &&
              (t.name.includes('attn_q') || t.name.includes('attn_k') || t.name.includes('attn_v'))) {
            const typeName = typeNames[t.type] || `type${t.type}`;
            console.log(`  ${t.name}: ${typeName}, dims=[${t.dimensions.map(Number).join(', ')}]`);
          }
        }
      }

      const loadLayerTensor = async (
        suffixes: string[]
      ): Promise<Tensor | null> => {
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

          if (info) {
            const loaded = await loadTensor(
              handle,
              info,
              ggufFile.tensorDataOffset,
              options
            );
            return loaded.tensor;
          }
        }
        return null;
      };

      // Helper to load and transpose layer weights
      const loadLayerWeight = async (
        suffixes: string[]
      ): Promise<Tensor | null> => {
        const tensor = await loadLayerTensor(suffixes);
        if (tensor && tensor.ndim === 2) {
          const transposed = await transpose(tensor);
          tensor.destroy();
          return transposed;
        }
        return tensor;
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
      if (i === 0) {
        const printShape = async (name: string, tensor: Tensor | null, patterns: string[]) => {
          if (!tensor) return;
          // Find original GGUF tensor info
          for (const pattern of patterns) {
            const info = ggufFile.tensors.find(t =>
              t.name.includes(`blk.0`) && t.name.includes(pattern.split('.')[0])
            );
            if (info) {
              console.log(`  ${name}: GGUF dims=[${info.dimensions.join(', ')}] -> final shape=[${tensor.shape.join(', ')}]`);
              break;
            }
          }
        };
        console.log('[Layer 0] GGUF dimensions vs final shapes:');
        await printShape('attnQ', layer.attnQ, ['attn_q.weight']);
        await printShape('attnK', layer.attnK, ['attn_k.weight']);
        await printShape('attnV', layer.attnV, ['attn_v.weight']);
        await printShape('attnOutput', layer.attnOutput, ['attn_output.weight']);
        await printShape('ffnGate', layer.ffnGate, ['ffn_gate.weight']);
        await printShape('ffnUp', layer.ffnUp, ['ffn_up.weight']);
        await printShape('ffnDown', layer.ffnDown, ['ffn_down.weight']);
      }

      // Debug: print weight shapes and statistics for layer 0
      if (i === 0) {
        console.log(`[Layer 0] Weight shapes and statistics:`);
        if (layer.attnQ) {
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
          console.log(`  attnQ: [${layer.attnQ.shape.join(', ')}]`);
          console.log(`    mean=${mean.toFixed(6)}, std=${Math.sqrt(variance).toFixed(6)}, min=${min.toFixed(4)}, max=${max.toFixed(4)}`);
          console.log(`    first row (8 vals): [${Array.from(qData.slice(0, 8)).map(v => v.toFixed(4)).join(', ')}]`);
          // Also print column sum for first column
          let col0Sum = 0;
          for (let r = 0; r < layer.attnQ.shape[0]; r++) {
            col0Sum += qData[r * layer.attnQ.shape[1]];
          }
          console.log(`    column 0 sum: ${col0Sum.toFixed(4)}`);
        }
        if (layer.attnK) {
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
          console.log(`  attnK: [${layer.attnK.shape.join(', ')}]`);
          console.log(`    mean=${mean.toFixed(6)}, std=${Math.sqrt(variance).toFixed(6)}, min=${min.toFixed(4)}, max=${max.toFixed(4)}`);
          console.log(`    first 8: [${Array.from(kData.slice(0, 8)).map(v => v.toFixed(4)).join(', ')}]`);
        }
        if (layer.attnV) {
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
          console.log(`  attnV: [${layer.attnV.shape.join(', ')}]`);
          console.log(`    mean=${mean.toFixed(6)}, std=${Math.sqrt(variance).toFixed(6)}, min=${min.toFixed(4)}, max=${max.toFixed(4)}`);
        }
        // Check biases
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
          console.log(`  attnKBias: [${layer.attnKBias.shape.join(', ')}], mean=${mean.toFixed(4)}, std=${Math.sqrt(variance).toFixed(4)}, min=${min.toFixed(4)}, max=${max.toFixed(4)}`);
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
