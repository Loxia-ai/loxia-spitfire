/**
 * Quantized GEMV (Matrix-Vector Multiply)
 *
 * Performs y = x @ W where:
 * - x is f32 input vector [1, K]
 * - W is quantized weight matrix [K, N] stored as QuantizedTensor
 * - y is f32 output vector [1, N]
 *
 * Dequantizes weights on-the-fly during computation, avoiding the need
 * to store full f32 weights in VRAM.
 */

import { Tensor } from '../tensor.js';
import { QuantizedTensor } from './qtensor.js';
import {
  createComputePipelineFromSource,
  createBindGroup,
  executeCompute,
  requestBufferDestroy,
} from '../shader.js';
import { createUniformBufferWithData } from '../buffer.js';
import { GGMLType } from '../../../types/model.js';

// ============================================================================
// Q8_0 Quantized GEMV Shader
// ============================================================================

const Q8_0_GEMV_SHADER = `
// Quantized GEMV: y = x @ W_q8
// x: f32 [1, K]
// W: Q8_0 quantized [K, N], stored as raw bytes
// y: f32 [1, N]

struct Params {
  N: u32,           // Output dimension (columns)
  K: u32,           // Input dimension (rows of W)
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> W_quant: array<u32>;  // Raw Q8_0 bytes as u32
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const BLOCK_SIZE: u32 = 32u;
const BYTES_PER_BLOCK: u32 = 34u;
const SHARED_K: u32 = 512u;  // Tile size for K dimension

var<workgroup> sharedX: array<f32, 512>;

// Read a byte from the quantized array (stored as u32s)
fn read_byte(byteOffset: u32) -> u32 {
  let u32Idx = byteOffset / 4u;
  let byteInU32 = byteOffset % 4u;
  return (W_quant[u32Idx] >> (byteInU32 * 8u)) & 0xFFu;
}

// Read u16 from byte offset (little-endian)
fn read_u16(byteOffset: u32) -> u32 {
  return read_byte(byteOffset) | (read_byte(byteOffset + 1u) << 8u);
}

// Unpack f16 to f32
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

// Read and dequantize a single weight value at W[k, n]
// GGUF stores data with ne0 (K) as contiguous dimension: linearIdx = n * K + k
fn dequant_weight(k: u32, n: u32, K: u32) -> f32 {
  let linearIdx = n * K + k;
  let blockIdx = linearIdx / BLOCK_SIZE;
  let posInBlock = linearIdx % BLOCK_SIZE;
  let blockByteOffset = blockIdx * BYTES_PER_BLOCK;

  // Read f16 scale (2 bytes at start of block)
  let scaleBits = read_u16(blockByteOffset);
  let scale = unpack_f16(scaleBits);

  // Read int8 quantized value
  let qByte = read_byte(blockByteOffset + 2u + posInBlock);
  let qVal = select(i32(qByte), i32(qByte) - 256, qByte >= 128u);

  return scale * f32(qVal);
}

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let tid = lid.x;
  let N = params.N;
  let K = params.K;

  // Each workgroup handles 1024 output columns (256 threads × 4 cols/thread)
  // Each thread handles 4 adjacent columns
  let baseCol = wid.x * 1024u + tid * 4u;

  let c0Valid = baseCol < N;
  let c1Valid = (baseCol + 1u) < N;
  let c2Valid = (baseCol + 2u) < N;
  let c3Valid = (baseCol + 3u) < N;

  var acc0: f32 = 0.0;
  var acc1: f32 = 0.0;
  var acc2: f32 = 0.0;
  var acc3: f32 = 0.0;

  let numTilesK = (K + SHARED_K - 1u) / SHARED_K;

  for (var tileK = 0u; tileK < numTilesK; tileK++) {
    let kStart = tileK * SHARED_K;
    let kEnd = min(kStart + SHARED_K, K);
    let tileLen = kEnd - kStart;

    // Cooperative load of x into shared memory
    for (var i = tid; i < tileLen; i += 256u) {
      sharedX[i] = x[kStart + i];
    }
    workgroupBarrier();

    // Accumulate dot products
    for (var kk = 0u; kk < tileLen; kk++) {
      let k = kStart + kk;
      let xVal = sharedX[kk];

      // Read and dequantize weights, accumulate
      if (c0Valid) { acc0 += xVal * dequant_weight(k, baseCol, K); }
      if (c1Valid) { acc1 += xVal * dequant_weight(k, baseCol + 1u, K); }
      if (c2Valid) { acc2 += xVal * dequant_weight(k, baseCol + 2u, K); }
      if (c3Valid) { acc3 += xVal * dequant_weight(k, baseCol + 3u, K); }
    }
    workgroupBarrier();
  }

  // Write outputs
  if (c0Valid) { output[baseCol] = acc0; }
  if (c1Valid) { output[baseCol + 1u] = acc1; }
  if (c2Valid) { output[baseCol + 2u] = acc2; }
  if (c3Valid) { output[baseCol + 3u] = acc3; }
}
`;

// ============================================================================
// Q4_0 Quantized GEMV Shader
// ============================================================================

const Q4_0_GEMV_SHADER = `
// Q4_0 GEMV: y = x @ W_q4_0
// x: f32 [1, K]
// W: Q4_0 quantized [K, N], stored as raw bytes
// y: f32 [1, N]
//
// Q4_0 Block Format (18 bytes per 32 elements):
// - d (f16, 2 bytes): scale at offset 0
// - qs[16]: packed 4-bit quantized values at offset 2 (2 values per byte)
//
// Dequantization: value = (nibble - 8) * scale
// where nibble is 0-15, subtract 8 gives signed range [-8, +7]

struct Params {
  N: u32,           // Output dimension (columns)
  K: u32,           // Input dimension (rows of W)
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> W_quant: array<u32>;  // Raw Q4_0 bytes as u32
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const BLOCK_SIZE: u32 = 32u;
const BYTES_PER_BLOCK: u32 = 18u;
const WG_SIZE: u32 = 256u;

var<workgroup> sharedX: array<f32, 256>;

// Read a byte from the quantized array (stored as u32s)
fn read_byte(byteOffset: u32) -> u32 {
  let u32Idx = byteOffset >> 2u;
  let byteInU32 = byteOffset & 3u;
  return (W_quant[u32Idx] >> (byteInU32 * 8u)) & 0xFFu;
}

// Read u16 from byte offset (little-endian)
fn read_u16(byteOffset: u32) -> u32 {
  return read_byte(byteOffset) | (read_byte(byteOffset + 1u) << 8u);
}

// Unpack f16 to f32
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

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let tid = lid.x;
  let N = params.N;
  let K = params.K;

  // Each thread handles ONE output column
  let col = wid.x * WG_SIZE + tid;

  var acc: f32 = 0.0;
  let blocksPerCol = K / BLOCK_SIZE;

  for (var blk = 0u; blk < blocksPerCol; blk++) {
    let kBase = blk * BLOCK_SIZE;

    // Cooperative load: each thread loads one x value into shared memory
    // For Q4_0 with 32 elements/block, we load multiple blocks worth (256/32 = 8 blocks)
    // but we process one block at a time per column
    if (tid < BLOCK_SIZE) {
      sharedX[tid] = x[kBase + tid];
    }
    workgroupBarrier();

    if (col < N) {
      // Block byte offset for this column and block
      // Layout: blocks are stored column-major (all blocks for col 0, then col 1, ...)
      let blockByteOffset = (col * blocksPerCol + blk) * BYTES_PER_BLOCK;

      // Read f16 scale (2 bytes at start of block)
      let scaleBits = read_u16(blockByteOffset);
      let scale = unpack_f16(scaleBits);

      // Process 32 values from 16 bytes (2 nibbles per byte)
      // Q4_0 packing: low nibble = index l (0-15), high nibble = index l+16 (16-31)
      for (var l = 0u; l < 16u; l++) {
        let qByte = read_byte(blockByteOffset + 2u + l);

        // Low nibble -> value at position l (0-15)
        let q0 = qByte & 0xFu;
        let v0 = (f32(q0) - 8.0) * scale;
        acc += sharedX[l] * v0;

        // High nibble -> value at position l+16 (16-31)
        let q1 = qByte >> 4u;
        let v1 = (f32(q1) - 8.0) * scale;
        acc += sharedX[l + 16u] * v1;
      }
    }
    workgroupBarrier();
  }

  if (col < N) {
    output[col] = acc;
  }
}
`;

// ============================================================================
// Q4_0 Quantized GEMM Shader (for batch/prefill with M > 1)
// ============================================================================

const Q4_0_GEMM_SHADER = `
// Q4_0 GEMM: Y = X @ W_q4_0
// X: f32 [M, K]
// W: Q4_0 quantized [K, N]
// Y: f32 [M, N]

struct Params {
  M: u32,           // Batch size (rows of X)
  N: u32,           // Output dimension (columns)
  K: u32,           // Input dimension
  _pad: u32,
}

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W_quant: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const BLOCK_SIZE: u32 = 32u;
const BYTES_PER_BLOCK: u32 = 18u;
const WG_SIZE: u32 = 64u;

var<workgroup> sharedX: array<f32, 64>;

fn read_byte(byteOffset: u32) -> u32 {
  let u32Idx = byteOffset >> 2u;
  let byteInU32 = byteOffset & 3u;
  return (W_quant[u32Idx] >> (byteInU32 * 8u)) & 0xFFu;
}

fn read_u16(byteOffset: u32) -> u32 {
  return read_byte(byteOffset) | (read_byte(byteOffset + 1u) << 8u);
}

fn unpack_f16(packed: u32) -> f32 {
  let sign = (packed >> 15u) & 1u;
  let exp = (packed >> 10u) & 0x1Fu;
  let mant = packed & 0x3FFu;
  if (exp == 0u) {
    if (mant == 0u) { return select(0.0, -0.0, sign == 1u); }
    return select(1.0, -1.0, sign == 1u) * f32(mant) * 5.960464478e-8;
  } else if (exp == 31u) {
    return select(1.0, -1.0, sign == 1u) * 65504.0;
  }
  let e = i32(exp) - 15;
  let m = 1.0 + f32(mant) / 1024.0;
  return select(1.0, -1.0, sign == 1u) * m * pow(2.0, f32(e));
}

@compute @workgroup_size(64)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let tid = lid.x;
  let M = params.M;
  let N = params.N;
  let K = params.K;

  // workgroup_id.x = output column tile
  // workgroup_id.y = row (M dimension)
  let col = wid.x * WG_SIZE + tid;
  let row = wid.y;

  if (row >= M) { return; }

  var acc: f32 = 0.0;
  let blocksPerCol = K / BLOCK_SIZE;

  for (var blk = 0u; blk < blocksPerCol; blk++) {
    let kBase = blk * BLOCK_SIZE;

    // Cooperative load of X[row, kBase:kBase+32]
    if (tid < BLOCK_SIZE) {
      sharedX[tid] = X[row * K + kBase + tid];
    }
    workgroupBarrier();

    if (col < N) {
      let blockByteOffset = (col * blocksPerCol + blk) * BYTES_PER_BLOCK;
      let scaleBits = read_u16(blockByteOffset);
      let scale = unpack_f16(scaleBits);

      // Q4_0 packing: low nibble = index l (0-15), high nibble = index l+16 (16-31)
      for (var l = 0u; l < 16u; l++) {
        let qByte = read_byte(blockByteOffset + 2u + l);
        let q0 = qByte & 0xFu;
        let q1 = qByte >> 4u;
        acc += sharedX[l] * (f32(q0) - 8.0) * scale;
        acc += sharedX[l + 16u] * (f32(q1) - 8.0) * scale;
      }
    }
    workgroupBarrier();
  }

  if (col < N) {
    output[row * N + col] = acc;
  }
}
`;

// ============================================================================
// Q4_K Quantized GEMV Shader - Heavily Optimized
// ============================================================================

const Q4_K_GEMV_SHADER = `
// Heavily optimized Q4_K GEMV: y = x @ W_q4k
// x: f32 [1, K]
// W: Q4_K quantized [K, N], stored as raw bytes
// y: f32 [1, N]
//
// Q4_K Block Format (144 bytes per 256 elements):
// - d (f16, 2 bytes): scale multiplier at offset 0
// - dmin (f16, 2 bytes): min multiplier at offset 2
// - scales[12]: packed 6-bit scales and mins at offset 4
// - qs[128]: packed 4-bit quantized values at offset 16
//
// Key optimizations:
// 1. Read block header (d, dmin) ONCE per block
// 2. Pre-compute d*scale and dmin*min for all 8 sub-blocks
// 3. Batch u32 reads for qs array (4 bytes = 8 q values at once)
// 4. Process sub-blocks with pre-computed scales

struct Params {
  N: u32,
  K: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> W_quant: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const BLOCK_SIZE: u32 = 256u;
const BYTES_PER_BLOCK: u32 = 144u;
const WG_SIZE: u32 = 256u;

var<workgroup> sharedX: array<f32, 256>;

fn unpack_f16(packed: u32) -> f32 {
  let sign = (packed >> 15u) & 1u;
  let exp = (packed >> 10u) & 0x1Fu;
  let mant = packed & 0x3FFu;
  if (exp == 0u) {
    if (mant == 0u) { return select(0.0, -0.0, sign == 1u); }
    return select(1.0, -1.0, sign == 1u) * f32(mant) * 5.960464478e-8;
  } else if (exp == 31u) {
    return select(1.0, -1.0, sign == 1u) * 65504.0;
  }
  let e = i32(exp) - 15;
  let m = 1.0 + f32(mant) / 1024.0;
  return select(1.0, -1.0, sign == 1u) * m * pow(2.0, f32(e));
}

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let tid = lid.x;
  let N = params.N;
  let K = params.K;

  // Each thread handles ONE output column
  let col = wid.x * WG_SIZE + tid;

  var acc: f32 = 0.0;
  let blocksPerCol = K >> 8u;  // K / 256

  for (var blk = 0u; blk < blocksPerCol; blk++) {
    let kBase = blk << 8u;

    // Cooperative load of 256 x values into shared memory
    sharedX[tid] = x[kBase + tid];
    workgroupBarrier();

    if (col < N) {
      // Calculate block byte offset (144 bytes per block, 4-byte aligned!)
      let blockIdx = col * blocksPerCol + blk;
      let blockU32Idx = blockIdx * 36u;  // 144 / 4 = 36 u32s per block

      // Read d and dmin in ONE u32 read (both f16 values packed)
      let dPacked = W_quant[blockU32Idx];
      let d = unpack_f16(dPacked & 0xFFFFu);
      let dmin = unpack_f16(dPacked >> 16u);

      // Read all 12 scale bytes as 3 u32s
      let sc0 = W_quant[blockU32Idx + 1u];  // scales[0..3]
      let sc1 = W_quant[blockU32Idx + 2u];  // scales[4..7]
      let sc2 = W_quant[blockU32Idx + 3u];  // scales[8..11]

      // Extract 6-bit scales and mins for all 8 sub-blocks
      // Sub-blocks 0-3: simple extraction
      let sc0_0 = sc0 & 0x3Fu;
      let sc0_1 = (sc0 >> 8u) & 0x3Fu;
      let sc0_2 = (sc0 >> 16u) & 0x3Fu;
      let sc0_3 = (sc0 >> 24u) & 0x3Fu;
      let mn0_0 = sc1 & 0x3Fu;
      let mn0_1 = (sc1 >> 8u) & 0x3Fu;
      let mn0_2 = (sc1 >> 16u) & 0x3Fu;
      let mn0_3 = (sc1 >> 24u) & 0x3Fu;

      // Sub-blocks 4-7: combined extraction (high bits from sc0, low bits from sc2)
      let sc0_4 = (sc2 & 0x0Fu) | ((sc0 >> 6u) & 0x30u);
      let sc0_5 = ((sc2 >> 8u) & 0x0Fu) | ((sc0 >> 12u) & 0x30u);
      let sc0_6 = ((sc2 >> 16u) & 0x0Fu) | ((sc0 >> 18u) & 0x30u);
      let sc0_7 = ((sc2 >> 24u) & 0x0Fu) | ((sc0 >> 24u) & 0x30u);
      let mn0_4 = ((sc2 >> 4u) & 0x0Fu) | ((sc1 >> 6u) & 0x30u);
      let mn0_5 = ((sc2 >> 12u) & 0x0Fu) | ((sc1 >> 12u) & 0x30u);
      let mn0_6 = ((sc2 >> 20u) & 0x0Fu) | ((sc1 >> 18u) & 0x30u);
      let mn0_7 = ((sc2 >> 28u) & 0x0Fu) | ((sc1 >> 24u) & 0x30u);

      // Pre-compute d*scale and dmin*min for all 8 sub-blocks
      let ds0 = d * f32(sc0_0); let dm0 = dmin * f32(mn0_0);
      let ds1 = d * f32(sc0_1); let dm1 = dmin * f32(mn0_1);
      let ds2 = d * f32(sc0_2); let dm2 = dmin * f32(mn0_2);
      let ds3 = d * f32(sc0_3); let dm3 = dmin * f32(mn0_3);
      let ds4 = d * f32(sc0_4); let dm4 = dmin * f32(mn0_4);
      let ds5 = d * f32(sc0_5); let dm5 = dmin * f32(mn0_5);
      let ds6 = d * f32(sc0_6); let dm6 = dmin * f32(mn0_6);
      let ds7 = d * f32(sc0_7); let dm7 = dmin * f32(mn0_7);

      // qs array starts at offset 16 bytes = 4 u32s from block start
      let qsU32Base = blockU32Idx + 4u;

      // Process 4 pairs of sub-blocks (0&1, 2&3, 4&5, 6&7)
      // Each pair shares the same 32 qs bytes (8 u32s)
      for (var pair = 0u; pair < 4u; pair++) {
        let sbEven = pair * 2u;
        let sbOdd = sbEven + 1u;
        let kOff = pair * 64u;

        // Select pre-computed scales for this pair
        var dsE: f32; var dmE: f32; var dsO: f32; var dmO: f32;
        if (pair == 0u) { dsE = ds0; dmE = dm0; dsO = ds1; dmO = dm1; }
        else if (pair == 1u) { dsE = ds2; dmE = dm2; dsO = ds3; dmO = dm3; }
        else if (pair == 2u) { dsE = ds4; dmE = dm4; dsO = ds5; dmO = dm5; }
        else { dsE = ds6; dmE = dm6; dsO = ds7; dmO = dm7; }

        // Read 8 u32s (32 bytes) for this sub-block pair
        let qsBase = qsU32Base + pair * 8u;

        for (var w = 0u; w < 8u; w++) {
          let packed = W_quant[qsBase + w];

          // Extract 8 4-bit q values from this u32
          let q0 = packed & 0xFu;
          let q1 = (packed >> 4u) & 0xFu;
          let q2 = (packed >> 8u) & 0xFu;
          let q3 = (packed >> 12u) & 0xFu;
          let q4 = (packed >> 16u) & 0xFu;
          let q5 = (packed >> 20u) & 0xFu;
          let q6 = (packed >> 24u) & 0xFu;
          let q7 = (packed >> 28u) & 0xFu;

          // Low nibbles go to even sub-block, high nibbles to odd
          let idx = kOff + w * 4u;
          acc += sharedX[idx] * (dsE * f32(q0) - dmE);
          acc += sharedX[idx + 1u] * (dsE * f32(q2) - dmE);
          acc += sharedX[idx + 2u] * (dsE * f32(q4) - dmE);
          acc += sharedX[idx + 3u] * (dsE * f32(q6) - dmE);
          acc += sharedX[idx + 32u] * (dsO * f32(q1) - dmO);
          acc += sharedX[idx + 33u] * (dsO * f32(q3) - dmO);
          acc += sharedX[idx + 34u] * (dsO * f32(q5) - dmO);
          acc += sharedX[idx + 35u] * (dsO * f32(q7) - dmO);
        }
      }
    }
    workgroupBarrier();
  }

  if (col < N) {
    output[col] = acc;
  }
}
`;

// ============================================================================
// Q8_0 Quantized GEMM Shader (M > 1)
// ============================================================================

const Q8_0_GEMM_SHADER = `
// Quantized GEMM: Y = X @ W_q8
// X: f32 [M, K]
// W: Q8_0 quantized [K, N], stored as raw bytes
// Y: f32 [M, N]
//
// Conservative approach: 64 threads, 1 column per thread to avoid GPU timeout.
// Uses shared memory tiling for X values.

struct Params {
  M: u32,           // Batch dimension (rows of X)
  N: u32,           // Output dimension (columns of W)
  K: u32,           // Inner dimension
  _pad: u32,
}

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W_quant: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const BLOCK_SIZE: u32 = 32u;
const BYTES_PER_BLOCK: u32 = 34u;
const TILE_K: u32 = 64u;  // Small tile size to reduce work per dispatch

var<workgroup> sharedX: array<f32, 64>;

fn read_byte(byteOffset: u32) -> u32 {
  let u32Idx = byteOffset / 4u;
  let byteInU32 = byteOffset % 4u;
  return (W_quant[u32Idx] >> (byteInU32 * 8u)) & 0xFFu;
}

fn read_u16(byteOffset: u32) -> u32 {
  return read_byte(byteOffset) | (read_byte(byteOffset + 1u) << 8u);
}

fn unpack_f16(packed: u32) -> f32 {
  let sign = (packed >> 15u) & 1u;
  let exp = (packed >> 10u) & 0x1Fu;
  let mant = packed & 0x3FFu;
  if (exp == 0u) {
    if (mant == 0u) { return select(0.0, -0.0, sign == 1u); }
    return select(1.0, -1.0, sign == 1u) * f32(mant) * 5.960464478e-8;
  } else if (exp == 31u) {
    return select(1.0, -1.0, sign == 1u) * 65504.0;
  }
  let e = i32(exp) - 15;
  let m = 1.0 + f32(mant) / 1024.0;
  return select(1.0, -1.0, sign == 1u) * m * pow(2.0, f32(e));
}

// GGUF stores data with ne0 (K) as contiguous dimension: linearIdx = n * K + k
fn dequant_weight(k: u32, n: u32, K: u32) -> f32 {
  let linearIdx = n * K + k;
  let blockIdx = linearIdx / BLOCK_SIZE;
  let posInBlock = linearIdx % BLOCK_SIZE;
  let blockByteOffset = blockIdx * BYTES_PER_BLOCK;
  let scaleBits = read_u16(blockByteOffset);
  let scale = unpack_f16(scaleBits);
  let qByte = read_byte(blockByteOffset + 2u + posInBlock);
  let qVal = select(i32(qByte), i32(qByte) - 256, qByte >= 128u);
  return scale * f32(qVal);
}

@compute @workgroup_size(64)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let tid = lid.x;
  let M = params.M;
  let N = params.N;
  let K = params.K;

  // wid.y = row (m), wid.x = column group
  let m = wid.y;
  // Each workgroup handles 64 output columns (64 threads × 1 col/thread)
  let col = wid.x * 64u + tid;

  // Use validity flag instead of early return (all threads must reach barriers)
  let validThread = (m < M) && (col < N);

  var acc: f32 = 0.0;

  // X row start offset (use 0 for invalid threads to avoid OOB)
  let xRowOffset = select(0u, m * K, validThread);
  let numTiles = (K + TILE_K - 1u) / TILE_K;

  for (var tile = 0u; tile < numTiles; tile++) {
    let kStart = tile * TILE_K;
    let kEnd = min(kStart + TILE_K, K);
    let tileLen = kEnd - kStart;

    // Cooperative load of X tile into shared memory
    if (tid < tileLen) {
      sharedX[tid] = X[xRowOffset + kStart + tid];
    }
    workgroupBarrier();

    // Accumulate using shared memory (only for valid threads)
    if (validThread) {
      for (var kk = 0u; kk < tileLen; kk++) {
        let k = kStart + kk;
        let xVal = sharedX[kk];
        acc += xVal * dequant_weight(k, col, K);
      }
    }
    workgroupBarrier();
  }

  // Write output (only for valid threads)
  if (validThread) {
    output[m * N + col] = acc;
  }
}
`;

// ============================================================================
// Q4_K Quantized GEMM Shader (M > 1)
// ============================================================================

const Q4_K_GEMM_SHADER = `
// Quantized GEMM: Y = X @ W_q4k
// X: f32 [M, K]
// W: Q4_K quantized [K, N], stored as raw bytes
// Y: f32 [M, N]
//
// Conservative approach: 64 threads, 1 column per thread to avoid GPU timeout.
// Uses shared memory tiling for X values.

struct Params {
  M: u32,
  N: u32,
  K: u32,
  _pad: u32,
}

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W_quant: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const BLOCK_SIZE: u32 = 256u;
const BYTES_PER_BLOCK: u32 = 144u;
const TILE_K: u32 = 64u;  // Small tile size to reduce work per dispatch

var<workgroup> sharedX: array<f32, 64>;

fn read_byte(byteOffset: u32) -> u32 {
  let u32Idx = byteOffset / 4u;
  let byteInU32 = byteOffset % 4u;
  return (W_quant[u32Idx] >> (byteInU32 * 8u)) & 0xFFu;
}

fn read_u16(byteOffset: u32) -> u32 {
  return read_byte(byteOffset) | (read_byte(byteOffset + 1u) << 8u);
}

fn unpack_f16(packed: u32) -> f32 {
  let sign = (packed >> 15u) & 1u;
  let exp = (packed >> 10u) & 0x1Fu;
  let mant = packed & 0x3FFu;
  if (exp == 0u) {
    if (mant == 0u) { return select(0.0, -0.0, sign == 1u); }
    return select(1.0, -1.0, sign == 1u) * f32(mant) * 5.960464478e-8;
  } else if (exp == 31u) {
    return select(1.0, -1.0, sign == 1u) * 65504.0;
  }
  let e = i32(exp) - 15;
  let m = 1.0 + f32(mant) / 1024.0;
  return select(1.0, -1.0, sign == 1u) * m * pow(2.0, f32(e));
}

fn get_scale_min(blockByteOffset: u32, j: u32) -> vec2<u32> {
  let scalesOffset = blockByteOffset + 4u;
  var sc: u32;
  var mn: u32;
  if (j < 4u) {
    sc = read_byte(scalesOffset + j) & 0x3Fu;
    mn = read_byte(scalesOffset + j + 4u) & 0x3Fu;
  } else {
    let jm4 = j - 4u;
    sc = (read_byte(scalesOffset + j + 4u) & 0x0Fu) | ((read_byte(scalesOffset + jm4) >> 6u) << 4u);
    mn = (read_byte(scalesOffset + j + 4u) >> 4u) | ((read_byte(scalesOffset + j) >> 6u) << 4u);
  }
  return vec2<u32>(sc, mn);
}

// GGUF stores data with ne0 (K) as contiguous dimension: linearIdx = n * K + k
fn dequant_weight_q4k(k: u32, n: u32, K: u32) -> f32 {
  let linearIdx = n * K + k;
  let blockIdx = linearIdx / BLOCK_SIZE;
  let posInBlock = linearIdx % BLOCK_SIZE;
  let blockByteOffset = blockIdx * BYTES_PER_BLOCK;

  let dBits = read_u16(blockByteOffset);
  let dminBits = read_u16(blockByteOffset + 2u);
  let d = unpack_f16(dBits);
  let dmin = unpack_f16(dminBits);

  let subBlock = posInBlock / 32u;
  let posInSubBlock = posInBlock % 32u;
  let scaleMins = get_scale_min(blockByteOffset, subBlock);
  let scale = f32(scaleMins.x);
  let mn = f32(scaleMins.y);

  let qsOffset = blockByteOffset + 16u;
  let byteIndex = (subBlock / 2u) * 32u + posInSubBlock;
  let qByte = read_byte(qsOffset + byteIndex);
  var q: u32;
  if ((subBlock & 1u) == 0u) {
    q = qByte & 0x0Fu;
  } else {
    q = qByte >> 4u;
  }

  return d * scale * f32(q) - dmin * mn;
}

@compute @workgroup_size(64)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let tid = lid.x;
  let M = params.M;
  let N = params.N;
  let K = params.K;

  let m = wid.y;
  // Each workgroup handles 64 output columns (64 threads × 1 col/thread)
  let col = wid.x * 64u + tid;

  // Use validity flag instead of early return (all threads must reach barriers)
  let validThread = (m < M) && (col < N);

  var acc: f32 = 0.0;

  // X row start offset (use 0 for invalid threads to avoid OOB)
  let xRowOffset = select(0u, m * K, validThread);
  let numTiles = (K + TILE_K - 1u) / TILE_K;

  for (var tile = 0u; tile < numTiles; tile++) {
    let kStart = tile * TILE_K;
    let kEnd = min(kStart + TILE_K, K);
    let tileLen = kEnd - kStart;

    // Cooperative load of X tile into shared memory
    if (tid < tileLen) {
      sharedX[tid] = X[xRowOffset + kStart + tid];
    }
    workgroupBarrier();

    // Accumulate using shared memory (only for valid threads)
    if (validThread) {
      for (var kk = 0u; kk < tileLen; kk++) {
        let k = kStart + kk;
        let xVal = sharedX[kk];
        acc += xVal * dequant_weight_q4k(k, col, K);
      }
    }
    workgroupBarrier();
  }

  // Write output (only for valid threads)
  if (validThread) {
    output[m * N + col] = acc;
  }
}
`;

// ============================================================================
// Q4_K Tiled GEMM Shader (M >= 4) — Shared Memory Weight Dequantization
// ============================================================================

const Q4_K_GEMM_TILED_SHADER = `
// Heavily Optimized Tiled Q4_K GEMM: Y = X @ W_q4k
// X: f32 [M, K]
// W: Q4_K quantized [K, N], stored as raw bytes
// Y: f32 [M, N]
//
// Key optimizations:
// 1. Read block header (d, dmin, scales) ONCE per Q4_K block, cache in registers
// 2. Batch u32 reads for qs array (Q4_K is 4-byte aligned: 144 bytes = 36 u32s)
// 3. Process one sub-block (32 elements) per K-tile, 8 sub-blocks per Q4_K block
// 4. Tile weights into shared memory and reuse across TILE_M=8 input rows
//
// Q4_K Block Format (144 bytes per 256 elements):
// - offset 0: d (f16) + dmin (f16) = 4 bytes = 1 u32
// - offset 4: scales[12] = 12 bytes = 3 u32s
// - offset 16: qs[128] = 128 bytes = 32 u32s

struct Params {
  M: u32,
  N: u32,
  K: u32,
  _pad: u32,
}

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W_quant: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const TILE_M: u32 = 8u;
const TILE_N: u32 = 64u;
const TILE_K: u32 = 32u;

// Shared memory for dequantized weights and input tile
var<workgroup> sharedW: array<f32, 2048>;  // TILE_N * TILE_K = 64 * 32
var<workgroup> sharedX: array<f32, 256>;   // TILE_M * TILE_K = 8 * 32

fn unpack_f16(packed: u32) -> f32 {
  let sign = (packed >> 15u) & 1u;
  let exp = (packed >> 10u) & 0x1Fu;
  let mant = packed & 0x3FFu;
  if (exp == 0u) {
    if (mant == 0u) { return select(0.0, -0.0, sign == 1u); }
    return select(1.0, -1.0, sign == 1u) * f32(mant) * 5.960464478e-8;
  } else if (exp == 31u) {
    return select(1.0, -1.0, sign == 1u) * 65504.0;
  }
  let e = i32(exp) - 15;
  let m = 1.0 + f32(mant) / 1024.0;
  return select(1.0, -1.0, sign == 1u) * m * pow(2.0, f32(e));
}

@compute @workgroup_size(64)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let tid = lid.x;
  let M = params.M;
  let N = params.N;
  let K = params.K;

  let colBase = wid.x * TILE_N;
  let rowBase = wid.y * TILE_M;
  let col = colBase + tid;
  let validCol = col < N;

  var acc0: f32 = 0.0;
  var acc1: f32 = 0.0;
  var acc2: f32 = 0.0;
  var acc3: f32 = 0.0;
  var acc4: f32 = 0.0;
  var acc5: f32 = 0.0;
  var acc6: f32 = 0.0;
  var acc7: f32 = 0.0;

  let blocksPerCol = K >> 8u;  // K / 256 (Q4_K blocks)
  let numKTiles = K >> 5u;     // K / 32 (sub-blocks)

  // Cache for block header - refreshed every 8 K-tiles (one Q4_K block)
  var ds: array<f32, 8>;
  var dm: array<f32, 8>;
  var blockU32Idx: u32 = 0u;
  var lastBlkIdx: u32 = 0xFFFFFFFFu;

  for (var kt = 0u; kt < numKTiles; kt++) {
    let kStart = kt << 5u;  // kt * 32
    let blkIdx = kt >> 3u;  // kt / 8 = which Q4_K block
    let subBlk = kt & 7u;   // kt % 8 = which sub-block within Q4_K block

    // Step 1: Dequantize weights for this K-tile into shared memory
    if (validCol) {
      // Refresh block header cache when entering new Q4_K block
      if (blkIdx != lastBlkIdx) {
        lastBlkIdx = blkIdx;
        blockU32Idx = (col * blocksPerCol + blkIdx) * 36u;

        let dPacked = W_quant[blockU32Idx];
        let d = unpack_f16(dPacked & 0xFFFFu);
        let dmin = unpack_f16(dPacked >> 16u);

        let sc0 = W_quant[blockU32Idx + 1u];
        let sc1 = W_quant[blockU32Idx + 2u];
        let sc2 = W_quant[blockU32Idx + 3u];

        // Extract and pre-compute d*scale, dmin*min for all 8 sub-blocks
        ds[0] = d * f32(sc0 & 0x3Fu);
        ds[1] = d * f32((sc0 >> 8u) & 0x3Fu);
        ds[2] = d * f32((sc0 >> 16u) & 0x3Fu);
        ds[3] = d * f32((sc0 >> 24u) & 0x3Fu);
        dm[0] = dmin * f32(sc1 & 0x3Fu);
        dm[1] = dmin * f32((sc1 >> 8u) & 0x3Fu);
        dm[2] = dmin * f32((sc1 >> 16u) & 0x3Fu);
        dm[3] = dmin * f32((sc1 >> 24u) & 0x3Fu);

        ds[4] = d * f32((sc2 & 0x0Fu) | ((sc0 >> 2u) & 0x30u));
        ds[5] = d * f32(((sc2 >> 8u) & 0x0Fu) | ((sc0 >> 10u) & 0x30u));
        ds[6] = d * f32(((sc2 >> 16u) & 0x0Fu) | ((sc0 >> 18u) & 0x30u));
        ds[7] = d * f32(((sc2 >> 24u) & 0x0Fu) | ((sc0 >> 26u) & 0x30u));
        dm[4] = dmin * f32(((sc2 >> 4u) & 0x0Fu) | ((sc1 >> 2u) & 0x30u));
        dm[5] = dmin * f32(((sc2 >> 12u) & 0x0Fu) | ((sc1 >> 10u) & 0x30u));
        dm[6] = dmin * f32(((sc2 >> 20u) & 0x0Fu) | ((sc1 >> 18u) & 0x30u));
        dm[7] = dmin * f32(((sc2 >> 28u) & 0x0Fu) | ((sc1 >> 26u) & 0x30u));
      }

      // Dequantize 32 elements for this sub-block
      let dsVal = ds[subBlk];
      let dmVal = dm[subBlk];
      let pair = subBlk >> 1u;      // 0,0,1,1,2,2,3,3
      let isOdd = subBlk & 1u;      // 0,1,0,1,0,1,0,1
      let qsU32Base = blockU32Idx + 4u + pair * 8u;
      let wBase = tid * TILE_K;

      // Read 8 u32s (32 bytes) and extract 32 4-bit values
      for (var w = 0u; w < 8u; w++) {
        let packed = W_quant[qsU32Base + w];
        let idx = w * 4u;

        if (isOdd == 0u) {
          // Even sub-block: use low nibbles
          sharedW[wBase + idx] = dsVal * f32(packed & 0xFu) - dmVal;
          sharedW[wBase + idx + 1u] = dsVal * f32((packed >> 8u) & 0xFu) - dmVal;
          sharedW[wBase + idx + 2u] = dsVal * f32((packed >> 16u) & 0xFu) - dmVal;
          sharedW[wBase + idx + 3u] = dsVal * f32((packed >> 24u) & 0xFu) - dmVal;
        } else {
          // Odd sub-block: use high nibbles
          sharedW[wBase + idx] = dsVal * f32((packed >> 4u) & 0xFu) - dmVal;
          sharedW[wBase + idx + 1u] = dsVal * f32((packed >> 12u) & 0xFu) - dmVal;
          sharedW[wBase + idx + 2u] = dsVal * f32((packed >> 20u) & 0xFu) - dmVal;
          sharedW[wBase + idx + 3u] = dsVal * f32((packed >> 28u) & 0xFu) - dmVal;
        }
      }
    } else {
      let wBase = tid * TILE_K;
      for (var i = 0u; i < TILE_K; i++) {
        sharedW[wBase + i] = 0.0;
      }
    }

    // Step 2: Cooperatively load X tile (8 rows × 32 cols = 256 values)
    // 64 threads, 4 values each
    for (var i = 0u; i < 4u; i++) {
      let flatIdx = tid * 4u + i;
      let mLocal = flatIdx >> 5u;  // flatIdx / 32
      let kk = flatIdx & 31u;      // flatIdx % 32
      let globalRow = rowBase + mLocal;
      if (globalRow < M) {
        sharedX[flatIdx] = X[globalRow * K + kStart + kk];
      } else {
        sharedX[flatIdx] = 0.0;
      }
    }

    workgroupBarrier();

    // Step 3: Accumulate
    if (validCol) {
      let wBase = tid * TILE_K;
      for (var kk = 0u; kk < TILE_K; kk++) {
        let wVal = sharedW[wBase + kk];
        acc0 += sharedX[kk] * wVal;
        acc1 += sharedX[TILE_K + kk] * wVal;
        acc2 += sharedX[2u * TILE_K + kk] * wVal;
        acc3 += sharedX[3u * TILE_K + kk] * wVal;
        acc4 += sharedX[4u * TILE_K + kk] * wVal;
        acc5 += sharedX[5u * TILE_K + kk] * wVal;
        acc6 += sharedX[6u * TILE_K + kk] * wVal;
        acc7 += sharedX[7u * TILE_K + kk] * wVal;
      }
    }

    workgroupBarrier();
  }

  // Write results
  if (validCol) {
    if (rowBase + 0u < M) { output[(rowBase + 0u) * N + col] = acc0; }
    if (rowBase + 1u < M) { output[(rowBase + 1u) * N + col] = acc1; }
    if (rowBase + 2u < M) { output[(rowBase + 2u) * N + col] = acc2; }
    if (rowBase + 3u < M) { output[(rowBase + 3u) * N + col] = acc3; }
    if (rowBase + 4u < M) { output[(rowBase + 4u) * N + col] = acc4; }
    if (rowBase + 5u < M) { output[(rowBase + 5u) * N + col] = acc5; }
    if (rowBase + 6u < M) { output[(rowBase + 6u) * N + col] = acc6; }
    if (rowBase + 7u < M) { output[(rowBase + 7u) * N + col] = acc7; }
  }
}
`;

let q8_0GemvPipeline: GPUComputePipeline | null = null;
let q4_0GemvPipeline: GPUComputePipeline | null = null;
let q4_0GemmPipeline: GPUComputePipeline | null = null;
let q4_kGemvPipeline: GPUComputePipeline | null = null;
let q8_0GemmPipeline: GPUComputePipeline | null = null;
let q4_kGemmPipeline: GPUComputePipeline | null = null;
let q4_kGemmTiledPipeline: GPUComputePipeline | null = null;

/**
 * Quantized GEMV for Q8_0 weights
 *
 * Computes y = x @ W where W is stored in Q8_0 format.
 * Dequantizes on-the-fly, using ~4x less memory bandwidth than f32.
 *
 * @param x - Input tensor [1, K] (f32)
 * @param W - Weight tensor [K, N] (Q8_0 quantized)
 * @returns Output tensor [1, N] (f32)
 */
export async function gemvQ8_0(
  x: Tensor,
  W: QuantizedTensor
): Promise<Tensor> {
  if (x.shape[0] !== 1) {
    throw new Error(`gemvQ8_0 requires x.shape[0] === 1, got ${x.shape[0]}`);
  }

  if (W.quantType !== GGMLType.Q8_0) {
    throw new Error(`gemvQ8_0 requires Q8_0 weights, got ${W.quantType}`);
  }

  if (W.ndim !== 2) {
    throw new Error(`gemvQ8_0 requires 2D weight matrix, got ${W.ndim}D`);
  }

  const K = x.shape[1];
  const [wK, N] = W.shape;

  if (K !== wK) {
    throw new Error(`Dimension mismatch: x.shape[1]=${K} vs W.shape[0]=${wK}`);
  }

  if (!q8_0GemvPipeline) {
    q8_0GemvPipeline = createComputePipelineFromSource(Q8_0_GEMV_SHADER, {
      label: 'gemv_q8_0',
      entryPoint: 'main',
    });
  }

  const output = Tensor.empty([1, N], { label: 'gemv_q8_0_output' });

  const params = createUniformBufferWithData(
    new Uint32Array([N, K, 0, 0]),
    'gemv_q8_0_params'
  );

  const bindGroup = createBindGroup(q8_0GemvPipeline, 0, [
    { binding: 0, resource: x.getBuffer() },
    { binding: 1, resource: W.getBuffer() },
    { binding: 2, resource: output.getBuffer() },
    { binding: 3, resource: params },
  ]);

  // 1024 columns per workgroup
  const numWorkgroups = Math.ceil(N / 1024);
  const usedBuffers = [x.getBuffer(), W.getBuffer(), output.getBuffer(), params];

  await executeCompute(
    q8_0GemvPipeline,
    [bindGroup],
    [numWorkgroups, 1, 1],
    undefined,
    false,
    true,
    usedBuffers
  );

  requestBufferDestroy(params);
  return output;
}

/**
 * Quantized GEMV for Q4_0 weights
 *
 * Computes y = x @ W where W is stored in Q4_0 format.
 * Dequantizes on-the-fly, using ~8x less memory bandwidth than f32.
 *
 * Q4_0 Format: 32 values per block, 18 bytes (2 f16 scale + 16 packed nibbles)
 * Dequantization: value = (nibble - 8) * scale
 *
 * @param x - Input tensor [1, K] (f32)
 * @param W - Weight tensor [K, N] (Q4_0 quantized)
 * @returns Output tensor [1, N] (f32)
 */
export async function gemvQ4_0(
  x: Tensor,
  W: QuantizedTensor
): Promise<Tensor> {
  if (x.shape[0] !== 1) {
    throw new Error(`gemvQ4_0 requires x.shape[0] === 1, got ${x.shape[0]}`);
  }

  if (W.quantType !== GGMLType.Q4_0) {
    throw new Error(`gemvQ4_0 requires Q4_0 weights, got ${W.quantType}`);
  }

  if (W.ndim !== 2) {
    throw new Error(`gemvQ4_0 requires 2D weight matrix, got ${W.ndim}D`);
  }

  const K = x.shape[1];
  const [wK, N] = W.shape;

  // Validate dimensions are non-zero
  if (K === 0 || N === 0) {
    throw new Error(`gemvQ4_0: Invalid dimensions K=${K}, N=${N}. All must be > 0.`);
  }

  if (K !== wK) {
    throw new Error(`Dimension mismatch: x.shape[1]=${K} vs W.shape[0]=${wK}`);
  }

  if (!q4_0GemvPipeline) {
    q4_0GemvPipeline = createComputePipelineFromSource(Q4_0_GEMV_SHADER, {
      label: 'gemv_q4_0',
      entryPoint: 'main',
    });
  }

  const output = Tensor.empty([1, N], { label: 'gemv_q4_0_output' });

  const params = createUniformBufferWithData(
    new Uint32Array([N, K, 0, 0]),
    'gemv_q4_0_params'
  );

  const bindGroup = createBindGroup(q4_0GemvPipeline, 0, [
    { binding: 0, resource: x.getBuffer() },
    { binding: 1, resource: W.getBuffer() },
    { binding: 2, resource: output.getBuffer() },
    { binding: 3, resource: params },
  ]);

  // 256 columns per workgroup (1 thread per column)
  const numWorkgroups = Math.ceil(N / 256);
  const usedBuffers = [x.getBuffer(), W.getBuffer(), output.getBuffer(), params];

  await executeCompute(
    q4_0GemvPipeline,
    [bindGroup],
    [numWorkgroups, 1, 1],
    undefined,
    false,
    true,
    usedBuffers
  );

  requestBufferDestroy(params);
  return output;
}

/**
 * Quantized GEMM for Q4_0 weights (batch/prefill with M > 1)
 *
 * Computes Y = X @ W where W is stored in Q4_0 format.
 *
 * @param X - Input tensor [M, K] (f32)
 * @param W - Weight tensor [K, N] (Q4_0 quantized)
 * @returns Output tensor [M, N] (f32)
 */
export async function gemmQ4_0(
  X: Tensor,
  W: QuantizedTensor
): Promise<Tensor> {
  if (W.quantType !== GGMLType.Q4_0) {
    throw new Error(`gemmQ4_0 requires Q4_0 weights, got ${W.quantType}`);
  }

  if (W.ndim !== 2) {
    throw new Error(`gemmQ4_0 requires 2D weight matrix, got ${W.ndim}D`);
  }

  const M = X.shape[0];
  const K = X.shape[1];
  const [wK, N] = W.shape;

  // Validate dimensions are non-zero
  if (M === 0 || K === 0 || N === 0) {
    throw new Error(`gemmQ4_0: Invalid dimensions M=${M}, K=${K}, N=${N}. All must be > 0.`);
  }

  if (K !== wK) {
    throw new Error(`Dimension mismatch: X.shape[1]=${K} vs W.shape[0]=${wK}`);
  }

  if (!q4_0GemmPipeline) {
    q4_0GemmPipeline = createComputePipelineFromSource(Q4_0_GEMM_SHADER, {
      label: 'gemm_q4_0',
      entryPoint: 'main',
    });
  }

  const output = Tensor.empty([M, N], { label: 'gemm_q4_0_output' });

  const params = createUniformBufferWithData(
    new Uint32Array([M, N, K, 0]),
    'gemm_q4_0_params'
  );

  const bindGroup = createBindGroup(q4_0GemmPipeline, 0, [
    { binding: 0, resource: X.getBuffer() },
    { binding: 1, resource: W.getBuffer() },
    { binding: 2, resource: output.getBuffer() },
    { binding: 3, resource: params },
  ]);

  // 64 columns per workgroup, M rows
  const numWorkgroupsX = Math.ceil(N / 64);
  const numWorkgroupsY = M;
  const usedBuffers = [X.getBuffer(), W.getBuffer(), output.getBuffer(), params];

  await executeCompute(
    q4_0GemmPipeline,
    [bindGroup],
    [numWorkgroupsX, numWorkgroupsY, 1],
    undefined,
    false,
    true,
    usedBuffers
  );

  requestBufferDestroy(params);
  return output;
}

/**
 * Quantized GEMV for Q4_K weights
 *
 * Computes y = x @ W where W is stored in Q4_K format.
 * Dequantizes on-the-fly, using ~7x less memory bandwidth than f32.
 *
 * @param x - Input tensor [1, K] (f32)
 * @param W - Weight tensor [K, N] (Q4_K quantized)
 * @returns Output tensor [1, N] (f32)
 */
export async function gemvQ4_K(
  x: Tensor,
  W: QuantizedTensor
): Promise<Tensor> {
  if (x.shape[0] !== 1) {
    throw new Error(`gemvQ4_K requires x.shape[0] === 1, got ${x.shape[0]}`);
  }

  if (W.quantType !== GGMLType.Q4_K) {
    throw new Error(`gemvQ4_K requires Q4_K weights, got ${W.quantType}`);
  }

  if (W.ndim !== 2) {
    throw new Error(`gemvQ4_K requires 2D weight matrix, got ${W.ndim}D`);
  }

  const K = x.shape[1];
  const [wK, N] = W.shape;

  if (K !== wK) {
    throw new Error(`Dimension mismatch: x.shape[1]=${K} vs W.shape[0]=${wK}`);
  }

  if (!q4_kGemvPipeline) {
    q4_kGemvPipeline = createComputePipelineFromSource(Q4_K_GEMV_SHADER, {
      label: 'gemv_q4_k',
      entryPoint: 'main',
    });
  }

  const output = Tensor.empty([1, N], { label: 'gemv_q4_k_output' });

  const params = createUniformBufferWithData(
    new Uint32Array([N, K, 0, 0]),
    'gemv_q4_k_params'
  );

  const bindGroup = createBindGroup(q4_kGemvPipeline, 0, [
    { binding: 0, resource: x.getBuffer() },
    { binding: 1, resource: W.getBuffer() },
    { binding: 2, resource: output.getBuffer() },
    { binding: 3, resource: params },
  ]);

  // 256 columns per workgroup (1 col/thread, 256 threads)
  const numWorkgroups = Math.ceil(N / 256);
  const usedBuffers = [x.getBuffer(), W.getBuffer(), output.getBuffer(), params];

  await executeCompute(
    q4_kGemvPipeline,
    [bindGroup],
    [numWorkgroups, 1, 1],
    undefined,
    false,
    true,
    usedBuffers
  );

  requestBufferDestroy(params);
  return output;
}

/**
 * Quantized GEMM for Q8_0 weights (batch size > 1)
 *
 * Computes Y = X @ W where W is stored in Q8_0 format.
 * Supports arbitrary batch size M.
 *
 * @param X - Input tensor [M, K] (f32)
 * @param W - Weight tensor [K, N] (Q8_0 quantized)
 * @returns Output tensor [M, N] (f32)
 */
export async function gemmQ8_0(
  X: Tensor,
  W: QuantizedTensor
): Promise<Tensor> {
  if (W.quantType !== GGMLType.Q8_0) {
    throw new Error(`gemmQ8_0 requires Q8_0 weights, got ${W.quantType}`);
  }

  if (W.ndim !== 2) {
    throw new Error(`gemmQ8_0 requires 2D weight matrix, got ${W.ndim}D`);
  }

  const M = X.shape[0];
  const K = X.shape[1];
  const [wK, N] = W.shape;

  if (K !== wK) {
    throw new Error(`Dimension mismatch: X.shape[1]=${K} vs W.shape[0]=${wK}`);
  }

  if (!q8_0GemmPipeline) {
    q8_0GemmPipeline = createComputePipelineFromSource(Q8_0_GEMM_SHADER, {
      label: 'gemm_q8_0',
      entryPoint: 'main',
    });
  }

  const output = Tensor.empty([M, N], { label: 'gemm_q8_0_output' });

  const params = createUniformBufferWithData(
    new Uint32Array([M, N, K, 0]),
    'gemm_q8_0_params'
  );

  const bindGroup = createBindGroup(q8_0GemmPipeline, 0, [
    { binding: 0, resource: X.getBuffer() },
    { binding: 1, resource: W.getBuffer() },
    { binding: 2, resource: output.getBuffer() },
    { binding: 3, resource: params },
  ]);

  // workgroups: (ceil(N/64), M, 1) - 64 threads × 1 col = 64 cols/workgroup
  const numWorkgroupsX = Math.ceil(N / 64);
  const usedBuffers = [X.getBuffer(), W.getBuffer(), output.getBuffer(), params];

  await executeCompute(
    q8_0GemmPipeline,
    [bindGroup],
    [numWorkgroupsX, M, 1],
    undefined,
    false,
    true,
    usedBuffers
  );

  requestBufferDestroy(params);
  return output;
}

/**
 * Quantized GEMM for Q4_K weights (batch size > 1)
 *
 * Computes Y = X @ W where W is stored in Q4_K format.
 * Supports arbitrary batch size M.
 * Routes to tiled kernel for M >= 4 (shared memory weight dequantization).
 *
 * @param X - Input tensor [M, K] (f32)
 * @param W - Weight tensor [K, N] (Q4_K quantized)
 * @returns Output tensor [M, N] (f32)
 */
export async function gemmQ4_K(
  X: Tensor,
  W: QuantizedTensor
): Promise<Tensor> {
  if (W.quantType !== GGMLType.Q4_K) {
    throw new Error(`gemmQ4_K requires Q4_K weights, got ${W.quantType}`);
  }

  if (W.ndim !== 2) {
    throw new Error(`gemmQ4_K requires 2D weight matrix, got ${W.ndim}D`);
  }

  const M = X.shape[0];
  const K = X.shape[1];
  const [wK, N] = W.shape;

  if (K !== wK) {
    throw new Error(`Dimension mismatch: X.shape[1]=${K} vs W.shape[0]=${wK}`);
  }

  // Use tiled kernel for M >= 4 (shared memory weight reuse across rows)
  if (M >= 4) {
    return gemmQ4_K_tiled(X, W, M, K, N);
  }

  // Fallback to original scalar kernel for small M
  if (!q4_kGemmPipeline) {
    q4_kGemmPipeline = createComputePipelineFromSource(Q4_K_GEMM_SHADER, {
      label: 'gemm_q4_k',
      entryPoint: 'main',
    });
  }

  const output = Tensor.empty([M, N], { label: 'gemm_q4_k_output' });

  const params = createUniformBufferWithData(
    new Uint32Array([M, N, K, 0]),
    'gemm_q4_k_params'
  );

  const bindGroup = createBindGroup(q4_kGemmPipeline, 0, [
    { binding: 0, resource: X.getBuffer() },
    { binding: 1, resource: W.getBuffer() },
    { binding: 2, resource: output.getBuffer() },
    { binding: 3, resource: params },
  ]);

  // 64 threads × 1 col = 64 cols/workgroup
  const numWorkgroupsX = Math.ceil(N / 64);
  const usedBuffers = [X.getBuffer(), W.getBuffer(), output.getBuffer(), params];

  await executeCompute(
    q4_kGemmPipeline,
    [bindGroup],
    [numWorkgroupsX, M, 1],
    undefined,
    false,
    true,
    usedBuffers
  );

  requestBufferDestroy(params);
  return output;
}

/**
 * Tiled Q4_K GEMM with shared memory weight dequantization.
 * Dequantizes weight tiles once into shared memory and reuses across TILE_M=8 rows.
 * ~4-7x faster than scalar GEMM for typical prefill sizes (M > 4).
 */
async function gemmQ4_K_tiled(
  X: Tensor,
  W: QuantizedTensor,
  M: number,
  K: number,
  N: number
): Promise<Tensor> {
  if (!q4_kGemmTiledPipeline) {
    q4_kGemmTiledPipeline = createComputePipelineFromSource(Q4_K_GEMM_TILED_SHADER, {
      label: 'gemm_q4_k_tiled',
      entryPoint: 'main',
    });
  }

  const output = Tensor.empty([M, N], { label: 'gemm_q4_k_tiled_output' });

  const params = createUniformBufferWithData(
    new Uint32Array([M, N, K, 0]),
    'gemm_q4_k_tiled_params'
  );

  const bindGroup = createBindGroup(q4_kGemmTiledPipeline, 0, [
    { binding: 0, resource: X.getBuffer() },
    { binding: 1, resource: W.getBuffer() },
    { binding: 2, resource: output.getBuffer() },
    { binding: 3, resource: params },
  ]);

  // TILE_N=64 columns per workgroup, TILE_M=8 rows per workgroup
  const numWorkgroupsX = Math.ceil(N / 64);
  const numWorkgroupsY = Math.ceil(M / 8);
  const usedBuffers = [X.getBuffer(), W.getBuffer(), output.getBuffer(), params];

  await executeCompute(
    q4_kGemmTiledPipeline,
    [bindGroup],
    [numWorkgroupsX, numWorkgroupsY, 1],
    undefined,
    false,
    true,
    usedBuffers
  );

  requestBufferDestroy(params);
  return output;
}

// ============================================================================
// Q6_K GEMV Shader - Optimized with pre-computed scales and 4-way processing
// ============================================================================

const Q6_K_GEMV_SHADER = `
// Ultra-Optimized Q6_K GEMV: y = x @ W_q6k
// Key optimizations:
// 1. Process entire Q6_K block (256 elements) without inner loops
// 2. Pre-compute all 16 d*scale products upfront
// 3. Split l<16 and l>=16 cases to eliminate select() branches
// 4. Batch u32 reads with read_4bytes()
// 5. Inline all scale lookups

struct Params {
  N: u32,
  K: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> W_quant: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const BYTES_PER_BLOCK: u32 = 210u;
const WG_SIZE: u32 = 256u;

var<workgroup> sharedX: array<f32, 256>;

fn read_4bytes(byteOffset: u32) -> u32 {
  let u32Idx = byteOffset >> 2u;
  let byteInU32 = byteOffset & 3u;
  if (byteInU32 == 0u) {
    return W_quant[u32Idx];
  }
  let shift = byteInU32 << 3u;
  return (W_quant[u32Idx] >> shift) | (W_quant[u32Idx + 1u] << (32u - shift));
}

fn read_u16(byteOffset: u32) -> u32 {
  let u32Idx = byteOffset >> 2u;
  let byteInU32 = byteOffset & 3u;
  let data = W_quant[u32Idx];
  if (byteInU32 < 3u) {
    return (data >> (byteInU32 << 3u)) & 0xFFFFu;
  }
  return ((data >> 24u) & 0xFFu) | ((W_quant[u32Idx + 1u] & 0xFFu) << 8u);
}

fn f16_to_f32(bits: u32) -> f32 {
  let sign = (bits >> 15u) & 1u;
  let exp = (bits >> 10u) & 0x1Fu;
  let mant = bits & 0x3FFu;
  if (exp == 0u) {
    if (mant == 0u) { return select(0.0, -0.0, sign == 1u); }
    return select(1.0, -1.0, sign == 1u) * f32(mant) * 5.960464478e-8;
  }
  if (exp == 31u) { return select(1.0, -1.0, sign == 1u) * 65504.0; }
  return select(1.0, -1.0, sign == 1u) * (1.0 + f32(mant) / 1024.0) * pow(2.0, f32(i32(exp) - 15));
}

fn int8_to_f32(v: u32) -> f32 {
  return f32(i32(v << 24u) >> 24);
}

// Process 4 l values and accumulate, with pre-computed scale
fn process_quad(ql_lo: u32, ql_hi: u32, qh: u32, s1: f32, s2: f32, s3: f32, s4: f32, base: u32) -> f32 {
  var acc: f32 = 0.0;

  // l+0
  let q1_0 = f32(i32((ql_lo & 0xFu) | (((qh >> 0u) & 3u) << 4u)) - 32);
  let q2_0 = f32(i32((ql_hi & 0xFu) | (((qh >> 2u) & 3u) << 4u)) - 32);
  let q3_0 = f32(i32(((ql_lo >> 4u) & 0xFu) | (((qh >> 4u) & 3u) << 4u)) - 32);
  let q4_0 = f32(i32(((ql_hi >> 4u) & 0xFu) | (((qh >> 6u) & 3u) << 4u)) - 32);
  acc += sharedX[base] * s1 * q1_0 + sharedX[base + 32u] * s2 * q2_0 + sharedX[base + 64u] * s3 * q3_0 + sharedX[base + 96u] * s4 * q4_0;

  // l+1
  let ql1 = (ql_lo >> 8u) & 0xFFu;
  let ql32_1 = (ql_hi >> 8u) & 0xFFu;
  let qh1 = (qh >> 8u) & 0xFFu;
  let q1_1 = f32(i32((ql1 & 0xFu) | (((qh1 >> 0u) & 3u) << 4u)) - 32);
  let q2_1 = f32(i32((ql32_1 & 0xFu) | (((qh1 >> 2u) & 3u) << 4u)) - 32);
  let q3_1 = f32(i32(((ql1 >> 4u) & 0xFu) | (((qh1 >> 4u) & 3u) << 4u)) - 32);
  let q4_1 = f32(i32(((ql32_1 >> 4u) & 0xFu) | (((qh1 >> 6u) & 3u) << 4u)) - 32);
  acc += sharedX[base + 1u] * s1 * q1_1 + sharedX[base + 33u] * s2 * q2_1 + sharedX[base + 65u] * s3 * q3_1 + sharedX[base + 97u] * s4 * q4_1;

  // l+2
  let ql2 = (ql_lo >> 16u) & 0xFFu;
  let ql32_2 = (ql_hi >> 16u) & 0xFFu;
  let qh2 = (qh >> 16u) & 0xFFu;
  let q1_2 = f32(i32((ql2 & 0xFu) | (((qh2 >> 0u) & 3u) << 4u)) - 32);
  let q2_2 = f32(i32((ql32_2 & 0xFu) | (((qh2 >> 2u) & 3u) << 4u)) - 32);
  let q3_2 = f32(i32(((ql2 >> 4u) & 0xFu) | (((qh2 >> 4u) & 3u) << 4u)) - 32);
  let q4_2 = f32(i32(((ql32_2 >> 4u) & 0xFu) | (((qh2 >> 6u) & 3u) << 4u)) - 32);
  acc += sharedX[base + 2u] * s1 * q1_2 + sharedX[base + 34u] * s2 * q2_2 + sharedX[base + 66u] * s3 * q3_2 + sharedX[base + 98u] * s4 * q4_2;

  // l+3
  let ql3 = (ql_lo >> 24u) & 0xFFu;
  let ql32_3 = (ql_hi >> 24u) & 0xFFu;
  let qh3 = (qh >> 24u) & 0xFFu;
  let q1_3 = f32(i32((ql3 & 0xFu) | (((qh3 >> 0u) & 3u) << 4u)) - 32);
  let q2_3 = f32(i32((ql32_3 & 0xFu) | (((qh3 >> 2u) & 3u) << 4u)) - 32);
  let q3_3 = f32(i32(((ql3 >> 4u) & 0xFu) | (((qh3 >> 4u) & 3u) << 4u)) - 32);
  let q4_3 = f32(i32(((ql32_3 >> 4u) & 0xFu) | (((qh3 >> 6u) & 3u) << 4u)) - 32);
  acc += sharedX[base + 3u] * s1 * q1_3 + sharedX[base + 35u] * s2 * q2_3 + sharedX[base + 67u] * s3 * q3_3 + sharedX[base + 99u] * s4 * q4_3;

  return acc;
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let tid = lid.x;
  let N = params.N;
  let K = params.K;
  let col = wid.x * WG_SIZE + tid;

  var acc: f32 = 0.0;
  let blocksPerCol = K >> 8u;

  for (var blk = 0u; blk < blocksPerCol; blk++) {
    sharedX[tid] = x[(blk << 8u) + tid];
    workgroupBarrier();

    if (col < N) {
      let blockByteOffset = (col * blocksPerCol + blk) * BYTES_PER_BLOCK;
      let d = f16_to_f32(read_u16(blockByteOffset + 208u));

      // Read all 16 scales and pre-multiply with d
      let sc_0_3 = read_4bytes(blockByteOffset + 192u);
      let sc_4_7 = read_4bytes(blockByteOffset + 196u);
      let sc_8_11 = read_4bytes(blockByteOffset + 200u);
      let sc_12_15 = read_4bytes(blockByteOffset + 204u);

      // Chunk 0 scales (indices 0-7)
      let ds0_lo = d * int8_to_f32(sc_0_3 & 0xFFu);        // l<16, quadrant 0
      let ds0_hi = d * int8_to_f32((sc_0_3 >> 8u) & 0xFFu); // l>=16, quadrant 0
      let ds1_lo = d * int8_to_f32((sc_0_3 >> 16u) & 0xFFu);
      let ds1_hi = d * int8_to_f32((sc_0_3 >> 24u) & 0xFFu);
      let ds2_lo = d * int8_to_f32(sc_4_7 & 0xFFu);
      let ds2_hi = d * int8_to_f32((sc_4_7 >> 8u) & 0xFFu);
      let ds3_lo = d * int8_to_f32((sc_4_7 >> 16u) & 0xFFu);
      let ds3_hi = d * int8_to_f32((sc_4_7 >> 24u) & 0xFFu);

      // Chunk 1 scales (indices 8-15)
      let ds4_lo = d * int8_to_f32(sc_8_11 & 0xFFu);
      let ds4_hi = d * int8_to_f32((sc_8_11 >> 8u) & 0xFFu);
      let ds5_lo = d * int8_to_f32((sc_8_11 >> 16u) & 0xFFu);
      let ds5_hi = d * int8_to_f32((sc_8_11 >> 24u) & 0xFFu);
      let ds6_lo = d * int8_to_f32(sc_12_15 & 0xFFu);
      let ds6_hi = d * int8_to_f32((sc_12_15 >> 8u) & 0xFFu);
      let ds7_lo = d * int8_to_f32((sc_12_15 >> 16u) & 0xFFu);
      let ds7_hi = d * int8_to_f32((sc_12_15 >> 24u) & 0xFFu);

      // Chunk 0: ql[0..63], qh[0..31], output elements 0-127
      let qlBase0 = blockByteOffset;
      let qhBase0 = blockByteOffset + 128u;

      // l=0..15 (use ds*_lo scales)
      acc += process_quad(read_4bytes(qlBase0), read_4bytes(qlBase0 + 32u), read_4bytes(qhBase0), ds0_lo, ds1_lo, ds2_lo, ds3_lo, 0u);
      acc += process_quad(read_4bytes(qlBase0 + 4u), read_4bytes(qlBase0 + 36u), read_4bytes(qhBase0 + 4u), ds0_lo, ds1_lo, ds2_lo, ds3_lo, 4u);
      acc += process_quad(read_4bytes(qlBase0 + 8u), read_4bytes(qlBase0 + 40u), read_4bytes(qhBase0 + 8u), ds0_lo, ds1_lo, ds2_lo, ds3_lo, 8u);
      acc += process_quad(read_4bytes(qlBase0 + 12u), read_4bytes(qlBase0 + 44u), read_4bytes(qhBase0 + 12u), ds0_lo, ds1_lo, ds2_lo, ds3_lo, 12u);

      // l=16..31 (use ds*_hi scales)
      acc += process_quad(read_4bytes(qlBase0 + 16u), read_4bytes(qlBase0 + 48u), read_4bytes(qhBase0 + 16u), ds0_hi, ds1_hi, ds2_hi, ds3_hi, 16u);
      acc += process_quad(read_4bytes(qlBase0 + 20u), read_4bytes(qlBase0 + 52u), read_4bytes(qhBase0 + 20u), ds0_hi, ds1_hi, ds2_hi, ds3_hi, 20u);
      acc += process_quad(read_4bytes(qlBase0 + 24u), read_4bytes(qlBase0 + 56u), read_4bytes(qhBase0 + 24u), ds0_hi, ds1_hi, ds2_hi, ds3_hi, 24u);
      acc += process_quad(read_4bytes(qlBase0 + 28u), read_4bytes(qlBase0 + 60u), read_4bytes(qhBase0 + 28u), ds0_hi, ds1_hi, ds2_hi, ds3_hi, 28u);

      // Chunk 1: ql[64..127], qh[32..63], output elements 128-255
      let qlBase1 = blockByteOffset + 64u;
      let qhBase1 = blockByteOffset + 160u;

      // l=0..15 (use ds4-7_lo scales)
      acc += process_quad(read_4bytes(qlBase1), read_4bytes(qlBase1 + 32u), read_4bytes(qhBase1), ds4_lo, ds5_lo, ds6_lo, ds7_lo, 128u);
      acc += process_quad(read_4bytes(qlBase1 + 4u), read_4bytes(qlBase1 + 36u), read_4bytes(qhBase1 + 4u), ds4_lo, ds5_lo, ds6_lo, ds7_lo, 132u);
      acc += process_quad(read_4bytes(qlBase1 + 8u), read_4bytes(qlBase1 + 40u), read_4bytes(qhBase1 + 8u), ds4_lo, ds5_lo, ds6_lo, ds7_lo, 136u);
      acc += process_quad(read_4bytes(qlBase1 + 12u), read_4bytes(qlBase1 + 44u), read_4bytes(qhBase1 + 12u), ds4_lo, ds5_lo, ds6_lo, ds7_lo, 140u);

      // l=16..31 (use ds4-7_hi scales)
      acc += process_quad(read_4bytes(qlBase1 + 16u), read_4bytes(qlBase1 + 48u), read_4bytes(qhBase1 + 16u), ds4_hi, ds5_hi, ds6_hi, ds7_hi, 144u);
      acc += process_quad(read_4bytes(qlBase1 + 20u), read_4bytes(qlBase1 + 52u), read_4bytes(qhBase1 + 20u), ds4_hi, ds5_hi, ds6_hi, ds7_hi, 148u);
      acc += process_quad(read_4bytes(qlBase1 + 24u), read_4bytes(qlBase1 + 56u), read_4bytes(qhBase1 + 24u), ds4_hi, ds5_hi, ds6_hi, ds7_hi, 152u);
      acc += process_quad(read_4bytes(qlBase1 + 28u), read_4bytes(qlBase1 + 60u), read_4bytes(qhBase1 + 28u), ds4_hi, ds5_hi, ds6_hi, ds7_hi, 156u);
    }
    workgroupBarrier();
  }

  if (col < N) { output[col] = acc; }
}
`;

let q6_kGemvPipeline: GPUComputePipeline | null = null;

/**
 * Q6_K GEMV for single-token inference
 *
 * @param x - Input tensor [1, K] (f32)
 * @param W - Weight tensor [K, N] (Q6_K quantized)
 * @returns Output tensor [1, N] (f32)
 */
export async function gemvQ6_K(
  x: Tensor,
  W: QuantizedTensor
): Promise<Tensor> {
  if (x.shape[0] !== 1) {
    throw new Error(`gemvQ6_K requires x.shape[0] === 1, got ${x.shape[0]}`);
  }

  if (W.quantType !== GGMLType.Q6_K) {
    throw new Error(`gemvQ6_K requires Q6_K weights, got ${W.quantType}`);
  }

  if (W.ndim !== 2) {
    throw new Error(`gemvQ6_K requires 2D weight matrix, got ${W.ndim}D`);
  }

  const K = x.shape[1];
  const [wK, N] = W.shape;

  if (K !== wK) {
    throw new Error(`Dimension mismatch: x.shape[1]=${K} vs W.shape[0]=${wK}`);
  }

  if (!q6_kGemvPipeline) {
    q6_kGemvPipeline = createComputePipelineFromSource(Q6_K_GEMV_SHADER, {
      label: 'gemv_q6_k',
      entryPoint: 'main',
    });
  }

  const output = Tensor.empty([1, N], { label: 'gemv_q6_k_output' });

  const params = createUniformBufferWithData(
    new Uint32Array([N, K, 0, 0]),
    'gemv_q6_k_params'
  );

  const bindGroup = createBindGroup(q6_kGemvPipeline, 0, [
    { binding: 0, resource: x.getBuffer() },
    { binding: 1, resource: W.getBuffer() },
    { binding: 2, resource: output.getBuffer() },
    { binding: 3, resource: params },
  ]);

  const numWorkgroups = Math.ceil(N / 256);
  const usedBuffers = [x.getBuffer(), W.getBuffer(), output.getBuffer(), params];

  await executeCompute(
    q6_kGemvPipeline,
    [bindGroup],
    [numWorkgroups, 1, 1],
    undefined,
    false,
    true,
    usedBuffers
  );

  requestBufferDestroy(params);
  return output;
}

// ============================================================================
// Q6_K Tiled GEMM Shader (for prefill / batch inference)
// Key optimization: dequantize weight tiles into shared memory ONCE,
// then reuse across TILE_M=8 input rows.
// ============================================================================

const Q6_K_GEMM_TILED_SHADER = `
// Heavily Optimized Tiled Q6_K GEMM: Y = X @ W_q6k
// X: f32 [M, K]
// W: Q6_K quantized [K, N], stored as raw bytes
// Y: f32 [M, N]
//
// Key optimizations:
// 1. Cache block header (d, scales) in registers, refresh every 8 K-tiles
// 2. Batch read_4bytes() for ql, qh values
// 3. Pre-compute d*scale products for all 16 scales
//
// Q6_K Block Format (210 bytes per 256 elements):
// - ql[128] at offset 0: low 4 bits
// - qh[64] at offset 128: high 2 bits packed
// - scales[16] at offset 192: signed int8
// - d at offset 208: f16

struct Params {
  M: u32,
  N: u32,
  K: u32,
  _pad: u32,
}

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> W_quant: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const BYTES_PER_BLOCK: u32 = 210u;
const TILE_M: u32 = 8u;
const TILE_N: u32 = 64u;
const TILE_K: u32 = 32u;

var<workgroup> sharedW: array<f32, 2048>;  // TILE_N * TILE_K = 64 * 32
var<workgroup> sharedX: array<f32, 256>;   // TILE_M * TILE_K = 8 * 32

fn read_4bytes(byteOffset: u32) -> u32 {
  let u32Idx = byteOffset >> 2u;
  let byteInU32 = byteOffset & 3u;
  if (byteInU32 == 0u) {
    return W_quant[u32Idx];
  } else {
    let shift = byteInU32 << 3u;
    let lo = W_quant[u32Idx] >> shift;
    let hi = W_quant[u32Idx + 1u] << (32u - shift);
    return lo | hi;
  }
}

fn read_u16(byteOffset: u32) -> u32 {
  let u32Idx = byteOffset >> 2u;
  let byteInU32 = byteOffset & 3u;
  let data = W_quant[u32Idx];
  if (byteInU32 < 3u) {
    return (data >> (byteInU32 << 3u)) & 0xFFFFu;
  } else {
    let lo = (data >> 24u) & 0xFFu;
    let hi = W_quant[u32Idx + 1u] & 0xFFu;
    return lo | (hi << 8u);
  }
}

fn f16_to_f32(bits: u32) -> f32 {
  let sign = (bits >> 15u) & 1u;
  let exp = (bits >> 10u) & 0x1Fu;
  let mant = bits & 0x3FFu;
  if (exp == 0u) {
    if (mant == 0u) { return select(0.0, -0.0, sign == 1u); }
    return select(1.0, -1.0, sign == 1u) * f32(mant) * 5.960464478e-8;
  } else if (exp == 31u) {
    return select(1.0, -1.0, sign == 1u) * 65504.0;
  }
  let e = i32(exp) - 15;
  let m = 1.0 + f32(mant) / 1024.0;
  return select(1.0, -1.0, sign == 1u) * m * pow(2.0, f32(e));
}

fn int8_to_i32(v: u32) -> i32 {
  return i32(v << 24u) >> 24;
}

@compute @workgroup_size(64)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let tid = lid.x;
  let M = params.M;
  let N = params.N;
  let K = params.K;

  let colBase = wid.x * TILE_N;
  let rowBase = wid.y * TILE_M;
  let col = colBase + tid;
  let validCol = col < N;

  var acc0: f32 = 0.0;
  var acc1: f32 = 0.0;
  var acc2: f32 = 0.0;
  var acc3: f32 = 0.0;
  var acc4: f32 = 0.0;
  var acc5: f32 = 0.0;
  var acc6: f32 = 0.0;
  var acc7: f32 = 0.0;

  let blocksPerCol = K >> 8u;
  let numKTiles = K >> 5u;

  // Cache for block header - 16 pre-computed d*scale values
  var ds: array<f32, 16>;
  var blockByteOffset: u32 = 0u;
  var lastBlkIdx: u32 = 0xFFFFFFFFu;

  for (var kt = 0u; kt < numKTiles; kt++) {
    let kStart = kt << 5u;
    let blkIdx = kt >> 3u;       // which Q6_K block (8 K-tiles per block)
    let quadrant = kt & 7u;      // which quadrant within block (0-7)

    // Step 1: Dequantize weights for this K-tile
    if (validCol) {
      // Refresh block cache when entering new block
      if (blkIdx != lastBlkIdx) {
        lastBlkIdx = blkIdx;
        blockByteOffset = (col * blocksPerCol + blkIdx) * BYTES_PER_BLOCK;

        let d = f16_to_f32(read_u16(blockByteOffset + 208u));

        // Read all 16 scales and pre-multiply with d
        let sc_0_3 = read_4bytes(blockByteOffset + 192u);
        let sc_4_7 = read_4bytes(blockByteOffset + 196u);
        let sc_8_11 = read_4bytes(blockByteOffset + 200u);
        let sc_12_15 = read_4bytes(blockByteOffset + 204u);

        ds[0] = d * f32(int8_to_i32(sc_0_3 & 0xFFu));
        ds[1] = d * f32(int8_to_i32((sc_0_3 >> 8u) & 0xFFu));
        ds[2] = d * f32(int8_to_i32((sc_0_3 >> 16u) & 0xFFu));
        ds[3] = d * f32(int8_to_i32((sc_0_3 >> 24u) & 0xFFu));
        ds[4] = d * f32(int8_to_i32(sc_4_7 & 0xFFu));
        ds[5] = d * f32(int8_to_i32((sc_4_7 >> 8u) & 0xFFu));
        ds[6] = d * f32(int8_to_i32((sc_4_7 >> 16u) & 0xFFu));
        ds[7] = d * f32(int8_to_i32((sc_4_7 >> 24u) & 0xFFu));
        ds[8] = d * f32(int8_to_i32(sc_8_11 & 0xFFu));
        ds[9] = d * f32(int8_to_i32((sc_8_11 >> 8u) & 0xFFu));
        ds[10] = d * f32(int8_to_i32((sc_8_11 >> 16u) & 0xFFu));
        ds[11] = d * f32(int8_to_i32((sc_8_11 >> 24u) & 0xFFu));
        ds[12] = d * f32(int8_to_i32(sc_12_15 & 0xFFu));
        ds[13] = d * f32(int8_to_i32((sc_12_15 >> 8u) & 0xFFu));
        ds[14] = d * f32(int8_to_i32((sc_12_15 >> 16u) & 0xFFu));
        ds[15] = d * f32(int8_to_i32((sc_12_15 >> 24u) & 0xFFu));
      }

      // Determine which chunk and quadrant within chunk
      let chunk = quadrant >> 2u;        // 0 or 1
      let quadInChunk = quadrant & 3u;   // 0-3

      let qlBase = blockByteOffset + chunk * 64u;
      let qhBase = blockByteOffset + 128u + chunk * 32u;
      let wBase = tid * TILE_K;

      // Scale indices for this quadrant
      // quadInChunk 0 -> scales 0,1; quadInChunk 1 -> scales 2,3; etc.
      let scBase = chunk * 8u + quadInChunk * 2u;
      let dsLo = ds[scBase];
      let dsHi = ds[scBase + 1u];

      // Process 32 elements (8 iterations of 4 elements each)
      for (var i = 0u; i < 8u; i++) {
        let l = i * 4u;

        // Read 4 ql bytes, 4 ql+32 bytes, 4 qh bytes
        let ql_lo = read_4bytes(qlBase + l);
        let ql_hi = read_4bytes(qlBase + l + 32u);
        let qh_packed = read_4bytes(qhBase + l);

        // Process 4 l values
        for (var j = 0u; j < 4u; j++) {
          let lj = l + j;
          let shift = j << 3u;

          let ql0 = (ql_lo >> shift) & 0xFFu;
          let ql32 = (ql_hi >> shift) & 0xFFu;
          let qh = (qh_packed >> shift) & 0xFFu;

          // Select scale based on l position (l<16 vs l>=16)
          let dsVal = select(dsLo, dsHi, lj >= 16u);

          var q: i32;
          if (quadInChunk == 0u) {
            q = i32((ql0 & 0xFu) | (((qh >> 0u) & 3u) << 4u)) - 32;
          } else if (quadInChunk == 1u) {
            q = i32((ql32 & 0xFu) | (((qh >> 2u) & 3u) << 4u)) - 32;
          } else if (quadInChunk == 2u) {
            q = i32((ql0 >> 4u) | (((qh >> 4u) & 3u) << 4u)) - 32;
          } else {
            q = i32((ql32 >> 4u) | (((qh >> 6u) & 3u) << 4u)) - 32;
          }

          sharedW[wBase + lj] = dsVal * f32(q);
        }
      }
    } else {
      let wBase = tid * TILE_K;
      for (var i = 0u; i < TILE_K; i++) {
        sharedW[wBase + i] = 0.0;
      }
    }

    // Step 2: Cooperatively load X tile (8 rows × 32 cols)
    for (var i = 0u; i < 4u; i++) {
      let flatIdx = tid * 4u + i;
      let mLocal = flatIdx >> 5u;
      let kk = flatIdx & 31u;
      let globalRow = rowBase + mLocal;
      if (globalRow < M) {
        sharedX[flatIdx] = X[globalRow * K + kStart + kk];
      } else {
        sharedX[flatIdx] = 0.0;
      }
    }

    workgroupBarrier();

    // Step 3: Accumulate
    if (validCol) {
      let wBase = tid * TILE_K;
      for (var kk = 0u; kk < TILE_K; kk++) {
        let wVal = sharedW[wBase + kk];
        acc0 += sharedX[kk] * wVal;
        acc1 += sharedX[TILE_K + kk] * wVal;
        acc2 += sharedX[2u * TILE_K + kk] * wVal;
        acc3 += sharedX[3u * TILE_K + kk] * wVal;
        acc4 += sharedX[4u * TILE_K + kk] * wVal;
        acc5 += sharedX[5u * TILE_K + kk] * wVal;
        acc6 += sharedX[6u * TILE_K + kk] * wVal;
        acc7 += sharedX[7u * TILE_K + kk] * wVal;
      }
    }

    workgroupBarrier();
  }

  // Write results
  if (validCol) {
    if (rowBase + 0u < M) { output[(rowBase + 0u) * N + col] = acc0; }
    if (rowBase + 1u < M) { output[(rowBase + 1u) * N + col] = acc1; }
    if (rowBase + 2u < M) { output[(rowBase + 2u) * N + col] = acc2; }
    if (rowBase + 3u < M) { output[(rowBase + 3u) * N + col] = acc3; }
    if (rowBase + 4u < M) { output[(rowBase + 4u) * N + col] = acc4; }
    if (rowBase + 5u < M) { output[(rowBase + 5u) * N + col] = acc5; }
    if (rowBase + 6u < M) { output[(rowBase + 6u) * N + col] = acc6; }
    if (rowBase + 7u < M) { output[(rowBase + 7u) * N + col] = acc7; }
  }
}
`;

let q6_kGemmTiledPipeline: GPUComputePipeline | null = null;

/**
 * Q6_K GEMM for batch inference (prefill)
 *
 * Uses tiled shader with shared memory weight dequantization.
 * Weights are dequantized once per tile, reused across TILE_M=8 rows.
 *
 * @param X - Input tensor [M, K] (f32)
 * @param W - Weight tensor [K, N] (Q6_K quantized)
 * @returns Output tensor [M, N] (f32)
 */
export async function gemmQ6_K(
  X: Tensor,
  W: QuantizedTensor
): Promise<Tensor> {
  if (W.quantType !== GGMLType.Q6_K) {
    throw new Error(`gemmQ6_K requires Q6_K weights, got ${W.quantType}`);
  }

  if (W.ndim !== 2) {
    throw new Error(`gemmQ6_K requires 2D weight matrix, got ${W.ndim}D`);
  }

  const M = X.shape[0];
  const K = X.shape[1];
  const [wK, N] = W.shape;

  if (K !== wK) {
    throw new Error(`Dimension mismatch: X.shape[1]=${K} vs W.shape[0]=${wK}`);
  }

  if (!q6_kGemmTiledPipeline) {
    q6_kGemmTiledPipeline = createComputePipelineFromSource(Q6_K_GEMM_TILED_SHADER, {
      label: 'gemm_q6_k_tiled',
      entryPoint: 'main',
    });
  }

  const output = Tensor.empty([M, N], { label: 'gemm_q6_k_output' });

  const params = createUniformBufferWithData(
    new Uint32Array([M, N, K, 0]),
    'gemm_q6_k_params'
  );

  const bindGroup = createBindGroup(q6_kGemmTiledPipeline, 0, [
    { binding: 0, resource: X.getBuffer() },
    { binding: 1, resource: W.getBuffer() },
    { binding: 2, resource: output.getBuffer() },
    { binding: 3, resource: params },
  ]);

  // TILE_N=64 columns per workgroup, TILE_M=8 rows per workgroup
  const numWorkgroupsX = Math.ceil(N / 64);
  const numWorkgroupsY = Math.ceil(M / 8);
  const usedBuffers = [X.getBuffer(), W.getBuffer(), output.getBuffer(), params];

  await executeCompute(
    q6_kGemmTiledPipeline,
    [bindGroup],
    [numWorkgroupsX, numWorkgroupsY, 1],
    undefined,
    false,
    true,
    usedBuffers
  );

  requestBufferDestroy(params);
  return output;
}

/**
 * Reset cached pipelines (for testing/cleanup)
 */
export function resetQGemvPipelines(): void {
  q8_0GemvPipeline = null;
  q4_0GemvPipeline = null;
  q4_0GemmPipeline = null;
  q4_kGemvPipeline = null;
  q6_kGemvPipeline = null;
  q8_0GemmPipeline = null;
  q4_kGemmPipeline = null;
  q4_kGemmTiledPipeline = null;
  q6_kGemmTiledPipeline = null;
}
