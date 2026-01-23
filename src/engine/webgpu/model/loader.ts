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
 */
export async function loadTensor(
  handle: FileHandle,
  tensorInfo: GGUFTensorInfo,
  tensorDataOffset: bigint,
  options: TensorLoadOptions = {}
): Promise<LoadedTensor> {
  const { dequantize: shouldDequantize = true, label } = options;

  // Calculate dimensions and size
  const shape = tensorInfo.dimensions.map(Number);
  const numElements = calculateTensorElements(tensorInfo);
  const numBytes = calculateTensorBytes(tensorInfo);

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
    // Reshape to original dimensions
    if (shape.length > 1) {
      const reshapedTensor = tensor.reshape(shape);
      tensor.destroy();
      tensor = reshapedTensor;
    }
  } else if (tensorInfo.type === GGMLTypeEnum.F32) {
    // Already f32, load directly
    const f32Data = new Float32Array(data.buffer, data.byteOffset, numElements);
    tensor = Tensor.fromData(f32Data, shape, { label: label || tensorInfo.name });
  } else if (tensorInfo.type === GGMLTypeEnum.F16) {
    // Convert f16 to f32
    tensor = await dequantize(data, numElements, tensorInfo.type);
    if (shape.length > 1) {
      const reshapedTensor = tensor.reshape(shape);
      tensor.destroy();
      tensor = reshapedTensor;
    }
  } else {
    // Keep quantized (store raw bytes for later GPU dequantization)
    // For now, just dequantize
    tensor = await dequantize(data, numElements, tensorInfo.type);
    if (shape.length > 1) {
      const reshapedTensor = tensor.reshape(shape);
      tensor.destroy();
      tensor = reshapedTensor;
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
  attnK: Tensor | null;
  attnV: Tensor | null;
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

    // Load embedding
    weights.tokenEmbedding = await loadByPattern([
      'token_embd.weight',
      'tok_embeddings.weight',
      'embed_tokens.weight',
    ]);

    // Load output norm and weights
    weights.outputNorm = await loadByPattern([
      'output_norm.weight',
      'norm.weight',
      'model.norm.weight',
    ]);

    weights.outputWeight = await loadByPattern([
      'output.weight',
      'lm_head.weight',
    ]);

    // Load layer weights
    for (let i = 0; i < numLayers; i++) {
      const layer: LlamaLayerWeights = {
        attnNorm: null,
        attnQ: null,
        attnK: null,
        attnV: null,
        attnOutput: null,
        ffnNorm: null,
        ffnGate: null,
        ffnUp: null,
        ffnDown: null,
      };

      // Find tensors for this layer
      const layerPrefix = `blk.${i}`;
      const altPrefix = `layers.${i}`;

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

      layer.attnNorm = await loadLayerTensor(['attn_norm.weight', 'input_layernorm.weight']);
      layer.attnQ = await loadLayerTensor(['attn_q.weight', 'self_attn.q_proj.weight']);
      layer.attnK = await loadLayerTensor(['attn_k.weight', 'self_attn.k_proj.weight']);
      layer.attnV = await loadLayerTensor(['attn_v.weight', 'self_attn.v_proj.weight']);
      layer.attnOutput = await loadLayerTensor(['attn_output.weight', 'self_attn.o_proj.weight']);
      layer.ffnNorm = await loadLayerTensor(['ffn_norm.weight', 'post_attention_layernorm.weight']);
      layer.ffnGate = await loadLayerTensor(['ffn_gate.weight', 'mlp.gate_proj.weight']);
      layer.ffnUp = await loadLayerTensor(['ffn_up.weight', 'mlp.up_proj.weight']);
      layer.ffnDown = await loadLayerTensor(['ffn_down.weight', 'mlp.down_proj.weight']);

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
    if (layer.attnK) layer.attnK.destroy();
    if (layer.attnV) layer.attnV.destroy();
    if (layer.attnOutput) layer.attnOutput.destroy();
    if (layer.ffnNorm) layer.ffnNorm.destroy();
    if (layer.ffnGate) layer.ffnGate.destroy();
    if (layer.ffnUp) layer.ffnUp.destroy();
    if (layer.ffnDown) layer.ffnDown.destroy();
  }
}
