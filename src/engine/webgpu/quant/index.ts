/**
 * WebGPU Quantization Support
 * Dequantization shaders for GGML quantization formats
 */

/// <reference types="@webgpu/types" />

import { Tensor } from '../tensor.js';
import {
  createComputePipelineFromSource,
  createBindGroup,
  executeCompute,
  calculateWorkgroups,
} from '../shader.js';
import {
  createStorageBuffer,
  createStorageBufferWithData,
  createUniformBufferWithData,
} from '../buffer.js';
import { GGMLType } from '../../../types/model.js';

// ============================================================================
// Q4_0 Quantization
// Block size: 32 values per block
// Storage: 2 bytes (f16 scale) + 16 bytes (32 x 4-bit values) = 18 bytes per block
// ============================================================================

const Q4_0_DEQUANT_SHADER = `
struct Params {
  numBlocks: u32,
  _pad1: u32,
  _pad2: u32,
  _pad3: u32,
}

@group(0) @binding(0) var<storage, read> quantized: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const BLOCK_SIZE: u32 = 32u;
const QK4_0: u32 = 32u;

// Unpack f16 from u32 (lower 16 bits)
fn unpack_f16(packed: u32) -> f32 {
  let sign = (packed >> 15u) & 1u;
  let exp = (packed >> 10u) & 0x1Fu;
  let mant = packed & 0x3FFu;

  if (exp == 0u) {
    if (mant == 0u) {
      return select(0.0, -0.0, sign == 1u);
    }
    // Subnormal
    return select(1.0, -1.0, sign == 1u) * f32(mant) * 5.960464478e-8;
  } else if (exp == 31u) {
    return select(1.0, -1.0, sign == 1u) * 65504.0; // Clamp inf to max
  }

  let e = i32(exp) - 15;
  let m = 1.0 + f32(mant) / 1024.0;
  return select(1.0, -1.0, sign == 1u) * m * pow(2.0, f32(e));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let blockIdx = gid.x;
  if (blockIdx >= params.numBlocks) {
    return;
  }

  // Q4_0 block layout:
  // - 2 bytes: f16 scale (stored as u16 in first u32, lower bits)
  // - 16 bytes: 32 x 4-bit quantized values (stored in 4 u32s)
  // Total: 18 bytes, but we read as 5 u32s (20 bytes, with 2 bytes padding per block)

  // For simplicity, assume packed as: scale (f16 in u32[0]), then 4 u32s of nibbles
  let baseIdx = blockIdx * 5u; // 5 u32s per block

  let scalePacked = quantized[baseIdx] & 0xFFFFu;
  let scale = unpack_f16(scalePacked);

  let outputBase = blockIdx * BLOCK_SIZE;

  // Unpack 32 4-bit values from 4 u32s
  for (var i = 0u; i < 4u; i++) {
    let packed = quantized[baseIdx + 1u + i];
    for (var j = 0u; j < 8u; j++) {
      let nibble = (packed >> (j * 4u)) & 0xFu;
      // Q4_0: values are stored as unsigned 0-15, subtract 8 to get signed
      let value = f32(i32(nibble) - 8) * scale;
      output[outputBase + i * 8u + j] = value;
    }
  }
}
`;

// ============================================================================
// Q8_0 Quantization
// Block size: 32 values per block
// Storage: 2 bytes (f16 scale) + 32 bytes (32 x 8-bit values) = 34 bytes per block
// ============================================================================

const Q8_0_DEQUANT_SHADER = `
struct Params {
  numBlocks: u32,
  _pad1: u32,
  _pad2: u32,
  _pad3: u32,
}

@group(0) @binding(0) var<storage, read> quantized: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const BLOCK_SIZE: u32 = 32u;

fn unpack_f16(packed: u32) -> f32 {
  let sign = (packed >> 15u) & 1u;
  let exp = (packed >> 10u) & 0x1Fu;
  let mant = packed & 0x3FFu;

  if (exp == 0u) {
    if (mant == 0u) {
      return select(0.0, -0.0, sign == 1u);
    }
    return select(1.0, -1.0, sign == 1u) * f32(mant) * 5.960464478e-8;
  } else if (exp == 31u) {
    return select(1.0, -1.0, sign == 1u) * 65504.0;
  }

  let e = i32(exp) - 15;
  let m = 1.0 + f32(mant) / 1024.0;
  return select(1.0, -1.0, sign == 1u) * m * pow(2.0, f32(e));
}

fn unpack_i8(packed: u32, idx: u32) -> i32 {
  let byte = (packed >> (idx * 8u)) & 0xFFu;
  // Sign extend from 8 bits
  return select(i32(byte), i32(byte) - 256, byte >= 128u);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let blockIdx = gid.x;
  if (blockIdx >= params.numBlocks) {
    return;
  }

  // Q8_0 block layout:
  // - 2 bytes: f16 scale
  // - 32 bytes: 32 x int8 values
  // We pack as: 1 u32 for scale (lower 16 bits) + 8 u32s for values
  let baseIdx = blockIdx * 9u;

  let scalePacked = quantized[baseIdx] & 0xFFFFu;
  let scale = unpack_f16(scalePacked);

  let outputBase = blockIdx * BLOCK_SIZE;

  // Unpack 32 int8 values from 8 u32s
  for (var i = 0u; i < 8u; i++) {
    let packed = quantized[baseIdx + 1u + i];
    for (var j = 0u; j < 4u; j++) {
      let value = f32(unpack_i8(packed, j)) * scale;
      output[outputBase + i * 4u + j] = value;
    }
  }
}
`;

// ============================================================================
// Q4_K Quantization (K-quant)
// More complex block structure with per-group scales
// Block size: 256 values per super-block
// ============================================================================

const Q4_K_DEQUANT_SHADER = `
struct Params {
  numBlocks: u32,
  _pad1: u32,
  _pad2: u32,
  _pad3: u32,
}

@group(0) @binding(0) var<storage, read> quantized: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const QK_K: u32 = 256u;
const K_SCALE_SIZE: u32 = 12u;

fn unpack_f16(packed: u32) -> f32 {
  let sign = (packed >> 15u) & 1u;
  let exp = (packed >> 10u) & 0x1Fu;
  let mant = packed & 0x3FFu;

  if (exp == 0u) {
    return select(1.0, -1.0, sign == 1u) * f32(mant) * 5.960464478e-8;
  } else if (exp == 31u) {
    return select(1.0, -1.0, sign == 1u) * 65504.0;
  }

  let e = i32(exp) - 15;
  let m = 1.0 + f32(mant) / 1024.0;
  return select(1.0, -1.0, sign == 1u) * m * pow(2.0, f32(e));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let blockIdx = gid.x;
  if (blockIdx >= params.numBlocks) {
    return;
  }

  // Q4_K super-block layout (simplified):
  // - 2 bytes: d (f16) - overall scale
  // - 2 bytes: dmin (f16) - min scale
  // - 12 bytes: scales for 8 sub-blocks (6-bit each, packed)
  // - 128 bytes: 256 x 4-bit quantized values
  // Total: 144 bytes per super-block = 36 u32s

  let baseIdx = blockIdx * 36u;
  let outputBase = blockIdx * QK_K;

  // Read d and dmin
  let dPacked = quantized[baseIdx];
  let d = unpack_f16(dPacked & 0xFFFFu);
  let dmin = unpack_f16(dPacked >> 16u);

  // For simplicity, use uniform scale (full K-quant decoding is complex)
  // This is an approximation - real implementation needs full scale unpacking

  // Unpack 256 4-bit values from 32 u32s (starting after header)
  for (var i = 0u; i < 32u; i++) {
    let packed = quantized[baseIdx + 4u + i]; // Skip 4 u32s of header
    for (var j = 0u; j < 8u; j++) {
      let nibble = (packed >> (j * 4u)) & 0xFu;
      // Simplified dequantization
      let value = d * f32(i32(nibble) - 8);
      output[outputBase + i * 8u + j] = value;
    }
  }
}
`;

// Pipeline cache
let q4_0Pipeline: GPUComputePipeline | null = null;
let q8_0Pipeline: GPUComputePipeline | null = null;
let q4_kPipeline: GPUComputePipeline | null = null;

/**
 * Get the block size for a quantization type
 */
export function getBlockSize(type: GGMLType): number {
  switch (type) {
    case GGMLType.Q4_0:
    case GGMLType.Q4_1:
    case GGMLType.Q5_0:
    case GGMLType.Q5_1:
    case GGMLType.Q8_0:
    case GGMLType.Q8_1:
      return 32;
    case GGMLType.Q2_K:
    case GGMLType.Q3_K:
    case GGMLType.Q4_K:
    case GGMLType.Q5_K:
    case GGMLType.Q6_K:
    case GGMLType.Q8_K:
      return 256;
    default:
      return 1;
  }
}

/**
 * Get bytes per block for a quantization type
 */
export function getBytesPerBlock(type: GGMLType): number {
  switch (type) {
    case GGMLType.F32:
      return 4;
    case GGMLType.F16:
      return 2;
    case GGMLType.Q4_0:
      return 18; // 2 (scale) + 16 (32 * 4bit)
    case GGMLType.Q4_1:
      return 20; // 2 (scale) + 2 (min) + 16 (32 * 4bit)
    case GGMLType.Q5_0:
      return 22; // 2 (scale) + 4 (high bits) + 16 (32 * 4bit)
    case GGMLType.Q5_1:
      return 24;
    case GGMLType.Q8_0:
      return 34; // 2 (scale) + 32 (32 * 8bit)
    case GGMLType.Q8_1:
      return 36;
    case GGMLType.Q2_K:
      return 84;
    case GGMLType.Q3_K:
      return 110;
    case GGMLType.Q4_K:
      return 144;
    case GGMLType.Q5_K:
      return 176;
    case GGMLType.Q6_K:
      return 210;
    case GGMLType.Q8_K:
      return 292;
    default:
      return 4;
  }
}

/**
 * Check if a type requires dequantization
 */
export function requiresDequantization(type: GGMLType): boolean {
  return type !== GGMLType.F32 && type !== GGMLType.F16;
}

/**
 * Dequantize Q4_0 data to f32
 */
export async function dequantizeQ4_0(
  quantizedData: Uint8Array,
  numElements: number
): Promise<Tensor> {
  if (!q4_0Pipeline) {
    q4_0Pipeline = createComputePipelineFromSource(Q4_0_DEQUANT_SHADER, {
      label: 'dequant_q4_0',
      entryPoint: 'main',
    });
  }

  const blockSize = 32;
  const numBlocks = Math.ceil(numElements / blockSize);

  // Pack data for GPU (align to u32)
  const alignedSize = Math.ceil(quantizedData.length / 4) * 4;
  const alignedData = new Uint8Array(alignedSize);
  alignedData.set(quantizedData);

  const inputBuffer = createStorageBufferWithData(alignedData, 'q4_0_input');
  const outputBuffer = createStorageBuffer(numElements * 4, 'q4_0_output');
  const params = createUniformBufferWithData(
    new Uint32Array([numBlocks, 0, 0, 0]),
    'q4_0_params'
  );

  const bindGroup = createBindGroup(q4_0Pipeline, 0, [
    { binding: 0, resource: inputBuffer },
    { binding: 1, resource: outputBuffer },
    { binding: 2, resource: params },
  ]);

  const workgroups = calculateWorkgroups(numBlocks, 256);
  await executeCompute(q4_0Pipeline, [bindGroup], [workgroups, 1, 1]);

  inputBuffer.destroy();
  params.destroy();

  // Determine output shape (1D for now)
  return new Tensor([numElements], outputBuffer, { label: 'dequantized_q4_0' });
}

/**
 * Dequantize Q8_0 data to f32
 */
export async function dequantizeQ8_0(
  quantizedData: Uint8Array,
  numElements: number
): Promise<Tensor> {
  if (!q8_0Pipeline) {
    q8_0Pipeline = createComputePipelineFromSource(Q8_0_DEQUANT_SHADER, {
      label: 'dequant_q8_0',
      entryPoint: 'main',
    });
  }

  const blockSize = 32;
  const numBlocks = Math.ceil(numElements / blockSize);

  const alignedSize = Math.ceil(quantizedData.length / 4) * 4;
  const alignedData = new Uint8Array(alignedSize);
  alignedData.set(quantizedData);

  const inputBuffer = createStorageBufferWithData(alignedData, 'q8_0_input');
  const outputBuffer = createStorageBuffer(numElements * 4, 'q8_0_output');
  const params = createUniformBufferWithData(
    new Uint32Array([numBlocks, 0, 0, 0]),
    'q8_0_params'
  );

  const bindGroup = createBindGroup(q8_0Pipeline, 0, [
    { binding: 0, resource: inputBuffer },
    { binding: 1, resource: outputBuffer },
    { binding: 2, resource: params },
  ]);

  const workgroups = calculateWorkgroups(numBlocks, 256);
  await executeCompute(q8_0Pipeline, [bindGroup], [workgroups, 1, 1]);

  inputBuffer.destroy();
  params.destroy();

  return new Tensor([numElements], outputBuffer, { label: 'dequantized_q8_0' });
}

/**
 * Dequantize Q4_K data to f32
 */
export async function dequantizeQ4_K(
  quantizedData: Uint8Array,
  numElements: number
): Promise<Tensor> {
  if (!q4_kPipeline) {
    q4_kPipeline = createComputePipelineFromSource(Q4_K_DEQUANT_SHADER, {
      label: 'dequant_q4_k',
      entryPoint: 'main',
    });
  }

  const blockSize = 256;
  const numBlocks = Math.ceil(numElements / blockSize);

  const alignedSize = Math.ceil(quantizedData.length / 4) * 4;
  const alignedData = new Uint8Array(alignedSize);
  alignedData.set(quantizedData);

  const inputBuffer = createStorageBufferWithData(alignedData, 'q4_k_input');
  const outputBuffer = createStorageBuffer(numElements * 4, 'q4_k_output');
  const params = createUniformBufferWithData(
    new Uint32Array([numBlocks, 0, 0, 0]),
    'q4_k_params'
  );

  const bindGroup = createBindGroup(q4_kPipeline, 0, [
    { binding: 0, resource: inputBuffer },
    { binding: 1, resource: outputBuffer },
    { binding: 2, resource: params },
  ]);

  const workgroups = calculateWorkgroups(numBlocks, 256);
  await executeCompute(q4_kPipeline, [bindGroup], [workgroups, 1, 1]);

  inputBuffer.destroy();
  params.destroy();

  return new Tensor([numElements], outputBuffer, { label: 'dequantized_q4_k' });
}

/**
 * Dequantize data based on type
 */
export async function dequantize(
  quantizedData: Uint8Array,
  numElements: number,
  type: GGMLType
): Promise<Tensor> {
  switch (type) {
    case GGMLType.F32:
      // Already f32, just wrap in tensor
      return Tensor.fromData(
        new Float32Array(quantizedData.buffer, quantizedData.byteOffset, numElements),
        [numElements]
      );

    case GGMLType.F16:
      // Convert f16 to f32 on CPU for now
      return dequantizeF16(quantizedData, numElements);

    case GGMLType.Q4_0:
      return dequantizeQ4_0(quantizedData, numElements);

    case GGMLType.Q8_0:
      return dequantizeQ8_0(quantizedData, numElements);

    case GGMLType.Q4_K:
      return dequantizeQ4_K(quantizedData, numElements);

    default:
      throw new Error(`Unsupported quantization type: ${GGMLType[type] || type}`);
  }
}

/**
 * Dequantize F16 to F32 (CPU fallback)
 */
function dequantizeF16(data: Uint8Array, numElements: number): Tensor {
  const f16View = new Uint16Array(data.buffer, data.byteOffset, numElements);
  const f32Data = new Float32Array(numElements);

  for (let i = 0; i < numElements; i++) {
    const h = f16View[i];
    const sign = (h >> 15) & 1;
    const exp = (h >> 10) & 0x1f;
    const mant = h & 0x3ff;

    let value: number;
    if (exp === 0) {
      if (mant === 0) {
        value = sign ? -0 : 0;
      } else {
        // Subnormal
        value = (sign ? -1 : 1) * mant * Math.pow(2, -24);
      }
    } else if (exp === 31) {
      value = mant === 0 ? (sign ? -Infinity : Infinity) : NaN;
    } else {
      value = (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + mant / 1024);
    }

    f32Data[i] = value;
  }

  return Tensor.fromData(f32Data, [numElements], { label: 'dequantized_f16' });
}

/**
 * Quantize f32 to Q8_0 (for testing)
 */
export function quantizeToQ8_0(data: Float32Array): Uint8Array {
  const blockSize = 32;
  const numBlocks = Math.ceil(data.length / blockSize);
  const bytesPerBlock = 34; // 2 (scale) + 32 (values)
  const result = new Uint8Array(numBlocks * bytesPerBlock);

  for (let b = 0; b < numBlocks; b++) {
    const blockStart = b * blockSize;
    const blockEnd = Math.min(blockStart + blockSize, data.length);

    // Find max absolute value for scale
    let maxAbs = 0;
    for (let i = blockStart; i < blockEnd; i++) {
      maxAbs = Math.max(maxAbs, Math.abs(data[i]));
    }

    const scale = maxAbs / 127;
    const invScale = scale > 0 ? 127 / maxAbs : 0;

    // Convert scale to f16
    const scaleF16 = floatToHalf(scale);

    // Write scale (little-endian)
    const outBase = b * bytesPerBlock;
    result[outBase] = scaleF16 & 0xff;
    result[outBase + 1] = (scaleF16 >> 8) & 0xff;

    // Quantize values
    for (let i = 0; i < blockSize; i++) {
      const srcIdx = blockStart + i;
      const value = srcIdx < data.length ? data[srcIdx] : 0;
      const quantized = Math.round(value * invScale);
      // Clamp to [-128, 127] and store as uint8
      result[outBase + 2 + i] = Math.max(-128, Math.min(127, quantized)) & 0xff;
    }
  }

  return result;
}

/**
 * Convert f32 to f16
 */
function floatToHalf(value: number): number {
  const floatView = new Float32Array(1);
  const int32View = new Int32Array(floatView.buffer);

  floatView[0] = value;
  const f = int32View[0];

  const sign = (f >> 16) & 0x8000;
  let exp = ((f >> 23) & 0xff) - 127 + 15;
  let mant = (f >> 13) & 0x3ff;

  if (exp <= 0) {
    if (exp < -10) {
      return sign;
    }
    mant = (mant | 0x400) >> (1 - exp);
    return sign | mant;
  } else if (exp === 0xff - 127 + 15) {
    if (mant) {
      return sign | 0x7c00 | mant; // NaN
    }
    return sign | 0x7c00; // Inf
  } else if (exp > 30) {
    return sign | 0x7c00; // Overflow to Inf
  }

  return sign | (exp << 10) | mant;
}
