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
  requestBufferDestroy,
} from '../shader.js';
import {
  createStorageBuffer,
  createStorageBufferWithData,
  createUniformBufferWithData,
} from '../buffer.js';
import { GGMLType } from '../../../types/model.js';

// Re-export QuantizedTensor for GPU-resident quantized storage
export { QuantizedTensor, type SupportedQuantType } from './qtensor.js';

// Re-export quantized GEMV/GEMM operations
export { gemvQ8_0, gemvQ4_K, gemmQ8_0, gemmQ4_K, resetQGemvPipelines } from './qgemv.js';

// Re-export optimized quantized GEMV operations (Phase 1: Memory Coalescing)
export { gemvQ4_K_optimized, resetOptimizedPipelines } from './qgemv-optimized.js';

// Re-export fused QKV projection for quantized weights
export { fusedQKVProjectionQ4K, resetFusedQKVPipeline } from './fused-qkv.js';

// Re-export fused FFN gate+up for quantized weights
export { fusedFFNGateUpQ4K, resetFusedFFNPipeline } from './fused-ffn.js';

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

// Q8_0 data stored as raw bytes (accessed as u32 array)
@group(0) @binding(0) var<storage, read> quantized: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const BLOCK_SIZE: u32 = 32u;
const BYTES_PER_BLOCK: u32 = 34u;  // 2 bytes scale + 32 bytes values

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

// Read a byte from the quantized array (which is stored as u32s)
fn read_byte(byteOffset: u32) -> u32 {
  let u32Idx = byteOffset / 4u;
  let byteInU32 = byteOffset % 4u;
  return (quantized[u32Idx] >> (byteInU32 * 8u)) & 0xFFu;
}

// Read u16 from byte offset (little-endian)
fn read_u16(byteOffset: u32) -> u32 {
  return read_byte(byteOffset) | (read_byte(byteOffset + 1u) << 8u);
}

// Read signed i8 from byte offset
fn read_i8(byteOffset: u32) -> i32 {
  let byte = read_byte(byteOffset);
  return select(i32(byte), i32(byte) - 256, byte >= 128u);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let blockIdx = gid.x;
  if (blockIdx >= params.numBlocks) {
    return;
  }

  // Q8_0 block layout (34 bytes):
  // - bytes 0-1: f16 scale
  // - bytes 2-33: 32 x int8 values
  let blockByteOffset = blockIdx * BYTES_PER_BLOCK;

  // Read f16 scale from bytes 0-1
  let scaleBits = read_u16(blockByteOffset);
  let scale = unpack_f16(scaleBits);

  let outputBase = blockIdx * BLOCK_SIZE;

  // Read 32 int8 values from bytes 2-33
  for (var i = 0u; i < BLOCK_SIZE; i++) {
    let val = read_i8(blockByteOffset + 2u + i);
    output[outputBase + i] = f32(val) * scale;
  }
}
`;

// ============================================================================
// Pipeline cache
let q4_0Pipeline: GPUComputePipeline | null = null;
let q8_0Pipeline: GPUComputePipeline | null = null;

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
  const usedBuffers = [inputBuffer, outputBuffer, params];
  await executeCompute(q4_0Pipeline, [bindGroup], [workgroups, 1, 1], undefined, false, true, usedBuffers);

  requestBufferDestroy(inputBuffer);
  requestBufferDestroy(params);

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
  const usedBuffers = [inputBuffer, outputBuffer, params];
  await executeCompute(q8_0Pipeline, [bindGroup], [workgroups, 1, 1], undefined, false, true, usedBuffers);

  requestBufferDestroy(inputBuffer);
  requestBufferDestroy(params);

  return new Tensor([numElements], outputBuffer, { label: 'dequantized_q8_0' });
}

// Helper to convert f16 to f32 on CPU
function f16ToF32(bits: number): number {
  const sign = (bits >> 15) & 1;
  const exp = (bits >> 10) & 0x1f;
  const mant = bits & 0x3ff;

  if (exp === 0) {
    // Subnormal or zero
    return (sign ? -1 : 1) * mant * 5.960464478e-8;
  } else if (exp === 31) {
    // Inf or NaN
    return (sign ? -1 : 1) * (mant ? NaN : Infinity);
  }

  const e = exp - 15;
  const m = 1.0 + mant / 1024.0;
  return (sign ? -1 : 1) * m * Math.pow(2, e);
}

/**
 * Dequantize Q4_K data to f32 using CPU (correct implementation)
 * Based on ggml-quants.c dequantize_row_q4_K
 */
export async function dequantizeQ4_K(
  quantizedData: Uint8Array,
  numElements: number
): Promise<Tensor> {
  const QK_K = 256;
  const BLOCK_SIZE = 144; // bytes per Q4_K block
  const numBlocks = Math.floor(numElements / QK_K);
  const output = new Float32Array(numElements);

  const view = new DataView(quantizedData.buffer, quantizedData.byteOffset, quantizedData.byteLength);

  for (let i = 0; i < numBlocks; i++) {
    const blockOffset = i * BLOCK_SIZE;

    // Read d and dmin (f16 values)
    const dBits = view.getUint16(blockOffset, true);
    const dminBits = view.getUint16(blockOffset + 2, true);
    const d = f16ToF32(dBits);
    const dmin = f16ToF32(dminBits);

    // Read scales (12 bytes at offset 4)
    const scales = new Uint8Array(12);
    for (let j = 0; j < 12; j++) {
      scales[j] = quantizedData[blockOffset + 4 + j];
    }

    // Decode scales and mins for 8 sub-blocks using ggml's get_scale_min_k4 formula
    // The 12 scale bytes encode 8 6-bit scales and 8 6-bit mins in a packed format
    const sc = new Float32Array(8);
    const m = new Float32Array(8);

    // For j < 4: get_scale_min_k4 uses:
    //   scale = q[j] & 63 (lower 6 bits of scales[0-3])
    //   min   = q[j+4] & 63 (lower 6 bits of scales[4-7])
    sc[0] = scales[0] & 63;
    sc[1] = scales[1] & 63;
    sc[2] = scales[2] & 63;
    sc[3] = scales[3] & 63;

    m[0] = scales[4] & 63;
    m[1] = scales[5] & 63;
    m[2] = scales[6] & 63;
    m[3] = scales[7] & 63;

    // For j >= 4: get_scale_min_k4 uses:
    //   scale = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)
    //   min   = (q[j+4] >> 4) | ((q[j] >> 6) << 4)
    // For j=4: scale = (scales[8] & 0xF) | ((scales[0] >> 6) << 4)
    //          min   = (scales[8] >> 4) | ((scales[4] >> 6) << 4)
    sc[4] = (scales[8] & 0xF) | ((scales[0] >> 6) << 4);
    sc[5] = (scales[9] & 0xF) | ((scales[1] >> 6) << 4);
    sc[6] = (scales[10] & 0xF) | ((scales[2] >> 6) << 4);
    sc[7] = (scales[11] & 0xF) | ((scales[3] >> 6) << 4);

    m[4] = (scales[8] >> 4) | ((scales[4] >> 6) << 4);
    m[5] = (scales[9] >> 4) | ((scales[5] >> 6) << 4);
    m[6] = (scales[10] >> 4) | ((scales[6] >> 6) << 4);
    m[7] = (scales[11] >> 4) | ((scales[7] >> 6) << 4);

    // Read quantized values (128 bytes at offset 16)
    const qs = quantizedData.subarray(blockOffset + 16, blockOffset + 16 + 128);
    const outputBase = i * QK_K;

    // Q4_K data layout (from quantize_row_q4_K_ref):
    // Values are packed in groups of 64, with low nibble = first 32, high nibble = next 32
    // Bytes 0-31: low = values 0-31, high = values 32-63
    // Bytes 32-63: low = values 64-95, high = values 96-127
    // Bytes 64-95: low = values 128-159, high = values 160-191
    // Bytes 96-127: low = values 192-223, high = values 224-255

    // Process in chunks of 64 values (using 32 bytes each)
    // Each chunk uses 2 consecutive sub-blocks for scales/mins:
    //   chunk 0: outputs 0-31 (sc[0]), outputs 32-63 (sc[1])
    //   chunk 1: outputs 64-95 (sc[2]), outputs 96-127 (sc[3])
    //   chunk 2: outputs 128-159 (sc[4]), outputs 160-191 (sc[5])
    //   chunk 3: outputs 192-223 (sc[6]), outputs 224-255 (sc[7])
    for (let chunk = 0; chunk < 4; chunk++) {
      const qOffset = chunk * 32;
      const outOffset = chunk * 64;

      // Sub-block indices for scales/mins (consecutive pairs)
      const sb1 = chunk * 2;      // For low nibble values (first 32 of chunk)
      const sb2 = chunk * 2 + 1;  // For high nibble values (next 32 of chunk)

      const scale1 = d * sc[sb1];
      const min1 = dmin * m[sb1];
      const scale2 = d * sc[sb2];
      const min2 = dmin * m[sb2];

      for (let l = 0; l < 32; l++) {
        const qByte = qs[qOffset + l];

        // Low nibble -> first 32 values of chunk
        const q1 = qByte & 0xF;
        output[outputBase + outOffset + l] = scale1 * q1 - min1;

        // High nibble -> next 32 values of chunk
        const q2 = qByte >> 4;
        output[outputBase + outOffset + 32 + l] = scale2 * q2 - min2;
      }
    }
  }

  return Tensor.fromData(output, [numElements], { label: 'dequantized_q4_k' });
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

    case GGMLType.Q5_K:
      return dequantizeQ5_K(quantizedData, numElements);

    case GGMLType.Q6_K:
      return dequantizeQ6_K(quantizedData, numElements);

    default:
      throw new Error(`Unsupported quantization type: ${GGMLType[type] || type}`);
  }
}

/**
 * Dequantize Q5_K data to f32 (CPU fallback for now)
 * Q5_K: 256 values per super-block, 176 bytes per block
 */
function dequantizeQ5_K(data: Uint8Array, numElements: number): Tensor {
  const blockSize = 256;
  const bytesPerBlock = 176;
  const numBlocks = Math.ceil(numElements / blockSize);
  const f32Data = new Float32Array(numElements);

  for (let b = 0; b < numBlocks; b++) {
    const blockOffset = b * bytesPerBlock;
    const outputOffset = b * blockSize;

    // Read d and dmin (f16 values)
    const d = halfToFloat(data[blockOffset] | (data[blockOffset + 1] << 8));
    // dmin used in full implementation for min value offset
    const _dmin = halfToFloat(data[blockOffset + 2] | (data[blockOffset + 3] << 8));
    void _dmin; // Reserved for full implementation

    // Simplified dequantization - uses uniform scale
    // Full implementation would decode per-group scales
    const _scalesOffset = blockOffset + 4;
    void _scalesOffset; // Reserved for full implementation
    const qhOffset = blockOffset + 4 + 12; // 12 bytes of scales
    const qsOffset = qhOffset + 32; // 32 bytes of high bits

    for (let i = 0; i < Math.min(blockSize, numElements - outputOffset); i++) {
      const qsIdx = qsOffset + Math.floor(i / 2);
      const qhIdx = qhOffset + Math.floor(i / 8);

      // Get 4-bit base value
      const qs = (i % 2 === 0) ? (data[qsIdx] & 0x0F) : (data[qsIdx] >> 4);
      // Get 5th bit
      const qh = (data[qhIdx] >> (i % 8)) & 1;

      // Combine to 5-bit value
      const q = qs | (qh << 4);

      // Simplified dequant (proper impl needs per-group scales)
      f32Data[outputOffset + i] = d * (q - 16);
    }
  }

  return Tensor.fromData(f32Data, [numElements], { label: 'dequantized_q5_k' });
}

/**
 * Dequantize Q6_K data to f32 (CPU implementation matching ggml)
 * Q6_K: 256 values per super-block, 210 bytes per block
 * Layout: ql[128] + qh[64] + scales[16] + d[2]
 */
function dequantizeQ6_K(data: Uint8Array, numElements: number): Tensor {
  const QK_K = 256;
  const bytesPerBlock = 210;
  const numBlocks = Math.floor(numElements / QK_K);
  const f32Data = new Float32Array(numElements);

  for (let b = 0; b < numBlocks; b++) {
    const blockOffset = b * bytesPerBlock;
    const outputOffset = b * QK_K;

    // Read overall scale (d is at the end)
    const dOffset = blockOffset + 128 + 64 + 16;
    const d = halfToFloat(data[dOffset] | (data[dOffset + 1] << 8));

    // Process 2 chunks of 128 values each
    for (let chunk = 0; chunk < 2; chunk++) {
      // Offsets within the block for this chunk
      const qlBase = blockOffset + chunk * 64;
      const qhBase = blockOffset + 128 + chunk * 32;
      const scBase = blockOffset + 128 + 64 + chunk * 8;
      const outBase = outputOffset + chunk * 128;

      // Process 32 iterations, each producing 4 output values
      // Scale index varies: is = l/16 (0 for l<16, 1 for l>=16)
      // Per ggml: sc[is+0] for q1, sc[is+2] for q2, sc[is+4] for q3, sc[is+8] for q4
      for (let l = 0; l < 32; l++) {
        // Get the 4 quantized values
        const ql_l0 = data[qlBase + l];
        const ql_l32 = data[qlBase + l + 32];
        const qh_l = data[qhBase + l];

        // Extract 6-bit values
        const q1 = ((ql_l0 & 0xF) | (((qh_l >> 0) & 3) << 4)) - 32;
        const q2 = ((ql_l32 & 0xF) | (((qh_l >> 2) & 3) << 4)) - 32;
        const q3 = ((ql_l0 >> 4) | (((qh_l >> 4) & 3) << 4)) - 32;
        const q4 = ((ql_l32 >> 4) | (((qh_l >> 6) & 3) << 4)) - 32;

        // Get scales (signed int8) - index varies based on l/16
        // Each 128-value chunk uses 8 scales (indices 0-7 within that chunk)
        // q1: positions 0-31 use scales 0,1; q2: 32-63 use scales 2,3
        // q3: positions 64-95 use scales 4,5; q4: 96-127 use scales 6,7
        const is = Math.floor(l / 16);
        const sc1 = (data[scBase + is + 0] << 24) >> 24;
        const sc2 = (data[scBase + is + 2] << 24) >> 24;
        const sc3 = (data[scBase + is + 4] << 24) >> 24;
        const sc4 = (data[scBase + is + 6] << 24) >> 24;

        // Dequantize and store
        f32Data[outBase + l + 0] = d * sc1 * q1;
        f32Data[outBase + l + 32] = d * sc2 * q2;
        f32Data[outBase + l + 64] = d * sc3 * q3;
        f32Data[outBase + l + 96] = d * sc4 * q4;
      }
    }
  }

  return Tensor.fromData(f32Data, [numElements], { label: 'dequantized_q6_k' });
}

/**
 * Convert f16 to f32
 */
function halfToFloat(h: number): number {
  const sign = (h >> 15) & 1;
  const exp = (h >> 10) & 0x1f;
  const mant = h & 0x3ff;

  if (exp === 0) {
    if (mant === 0) {
      return sign ? -0 : 0;
    }
    // Subnormal
    return (sign ? -1 : 1) * mant * Math.pow(2, -24);
  } else if (exp === 31) {
    return mant === 0 ? (sign ? -Infinity : Infinity) : NaN;
  }

  return (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + mant / 1024);
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

/**
 * Reset all cached pipelines (useful after shader updates)
 */
export function resetQuantPipelines(): void {
  q4_0Pipeline = null;
  q8_0Pipeline = null;
}
