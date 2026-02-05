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
// Heavily Optimized Q6_K GEMV: y = x @ W_q6k
// Key optimizations:
// 1. Batch u32 reads: read_4bytes() reads 4 consecutive bytes in 1-2 fetches
// 2. Process 4 l values per iteration (16 q values at once)
// 3. Pre-compute d * scale products for all 8 scales per chunk
// 4. Use shared memory for X values
//
// Q6_K Block Format (210 bytes per 256 elements):
// - ql[128] at offset 0: low 4 bits of each q value
// - qh[64] at offset 128: high 2 bits packed (4 q values per byte)
// - scales[16] at offset 192: signed int8 scales for 16 sub-blocks
// - d at offset 208: f16 block scale

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

// Read 4 consecutive bytes as u32, handles unaligned access
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

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let tid = lid.x;
  let N = params.N;
  let K = params.K;
  let col = wid.x * WG_SIZE + tid;

  var acc: f32 = 0.0;
  let blocksPerCol = K >> 8u;

  for (var blk = 0u; blk < blocksPerCol; blk++) {
    let kBase = blk << 8u;

    // Cooperative load of 256 x values
    sharedX[tid] = x[kBase + tid];
    workgroupBarrier();

    if (col < N) {
      let blockIdx = col * blocksPerCol + blk;
      let blockByteOffset = blockIdx * BYTES_PER_BLOCK;

      // Read block scale d ONCE per block
      let d = f16_to_f32(read_u16(blockByteOffset + 208u));

      // Process 2 chunks of 128 elements each
      for (var chunk = 0u; chunk < 2u; chunk++) {
        let qlBase = blockByteOffset + chunk * 64u;
        let qhBase = blockByteOffset + 128u + chunk * 32u;
        let scBase = blockByteOffset + 192u + chunk * 8u;
        let outBase = chunk * 128u;

        // Read all 8 scales in 2 u32 reads and pre-multiply with d
        let sc_0_3 = read_4bytes(scBase);
        let sc_4_7 = read_4bytes(scBase + 4u);
        let ds0 = d * f32(int8_to_i32(sc_0_3 & 0xFFu));
        let ds1 = d * f32(int8_to_i32((sc_0_3 >> 8u) & 0xFFu));
        let ds2 = d * f32(int8_to_i32((sc_0_3 >> 16u) & 0xFFu));
        let ds3 = d * f32(int8_to_i32((sc_0_3 >> 24u) & 0xFFu));
        let ds4 = d * f32(int8_to_i32(sc_4_7 & 0xFFu));
        let ds5 = d * f32(int8_to_i32((sc_4_7 >> 8u) & 0xFFu));
        let ds6 = d * f32(int8_to_i32((sc_4_7 >> 16u) & 0xFFu));
        let ds7 = d * f32(int8_to_i32((sc_4_7 >> 24u) & 0xFFu));

        // Process 8 iterations of 4 l values each (32 total)
        // Each iteration: read 4 ql bytes, 4 ql+32 bytes, 4 qh bytes
        for (var i = 0u; i < 8u; i++) {
          let l = i * 4u;

          // Read 4 consecutive bytes for ql[l..l+3], ql[l+32..l+35], qh[l..l+3]
          let ql_lo = read_4bytes(qlBase + l);
          let ql_hi = read_4bytes(qlBase + l + 32u);
          let qh_packed = read_4bytes(qhBase + l);

          // Extract individual bytes
          let ql0 = ql_lo & 0xFFu;
          let ql1 = (ql_lo >> 8u) & 0xFFu;
          let ql2 = (ql_lo >> 16u) & 0xFFu;
          let ql3 = (ql_lo >> 24u) & 0xFFu;
          let ql32_0 = ql_hi & 0xFFu;
          let ql32_1 = (ql_hi >> 8u) & 0xFFu;
          let ql32_2 = (ql_hi >> 16u) & 0xFFu;
          let ql32_3 = (ql_hi >> 24u) & 0xFFu;
          let qh0 = qh_packed & 0xFFu;
          let qh1 = (qh_packed >> 8u) & 0xFFu;
          let qh2 = (qh_packed >> 16u) & 0xFFu;
          let qh3 = (qh_packed >> 24u) & 0xFFu;

          // Select scales: l<16 uses ds0/ds2/ds4/ds6, l>=16 uses ds1/ds3/ds5/ds7
          let useHigh = l >= 16u;
          let s1 = select(ds0, ds1, useHigh);
          let s2 = select(ds2, ds3, useHigh);
          let s3 = select(ds4, ds5, useHigh);
          let s4 = select(ds6, ds7, useHigh);

          // Process l+0
          let q1_0 = i32((ql0 & 0xFu) | (((qh0 >> 0u) & 3u) << 4u)) - 32;
          let q2_0 = i32((ql32_0 & 0xFu) | (((qh0 >> 2u) & 3u) << 4u)) - 32;
          let q3_0 = i32((ql0 >> 4u) | (((qh0 >> 4u) & 3u) << 4u)) - 32;
          let q4_0 = i32((ql32_0 >> 4u) | (((qh0 >> 6u) & 3u) << 4u)) - 32;
          acc += sharedX[outBase + l] * s1 * f32(q1_0);
          acc += sharedX[outBase + l + 32u] * s2 * f32(q2_0);
          acc += sharedX[outBase + l + 64u] * s3 * f32(q3_0);
          acc += sharedX[outBase + l + 96u] * s4 * f32(q4_0);

          // Process l+1
          let q1_1 = i32((ql1 & 0xFu) | (((qh1 >> 0u) & 3u) << 4u)) - 32;
          let q2_1 = i32((ql32_1 & 0xFu) | (((qh1 >> 2u) & 3u) << 4u)) - 32;
          let q3_1 = i32((ql1 >> 4u) | (((qh1 >> 4u) & 3u) << 4u)) - 32;
          let q4_1 = i32((ql32_1 >> 4u) | (((qh1 >> 6u) & 3u) << 4u)) - 32;
          acc += sharedX[outBase + l + 1u] * s1 * f32(q1_1);
          acc += sharedX[outBase + l + 33u] * s2 * f32(q2_1);
          acc += sharedX[outBase + l + 65u] * s3 * f32(q3_1);
          acc += sharedX[outBase + l + 97u] * s4 * f32(q4_1);

          // Process l+2
          let q1_2 = i32((ql2 & 0xFu) | (((qh2 >> 0u) & 3u) << 4u)) - 32;
          let q2_2 = i32((ql32_2 & 0xFu) | (((qh2 >> 2u) & 3u) << 4u)) - 32;
          let q3_2 = i32((ql2 >> 4u) | (((qh2 >> 4u) & 3u) << 4u)) - 32;
          let q4_2 = i32((ql32_2 >> 4u) | (((qh2 >> 6u) & 3u) << 4u)) - 32;
          acc += sharedX[outBase + l + 2u] * s1 * f32(q1_2);
          acc += sharedX[outBase + l + 34u] * s2 * f32(q2_2);
          acc += sharedX[outBase + l + 66u] * s3 * f32(q3_2);
          acc += sharedX[outBase + l + 98u] * s4 * f32(q4_2);

          // Process l+3
          let q1_3 = i32((ql3 & 0xFu) | (((qh3 >> 0u) & 3u) << 4u)) - 32;
          let q2_3 = i32((ql32_3 & 0xFu) | (((qh3 >> 2u) & 3u) << 4u)) - 32;
          let q3_3 = i32((ql3 >> 4u) | (((qh3 >> 4u) & 3u) << 4u)) - 32;
          let q4_3 = i32((ql32_3 >> 4u) | (((qh3 >> 6u) & 3u) << 4u)) - 32;
          acc += sharedX[outBase + l + 3u] * s1 * f32(q1_3);
          acc += sharedX[outBase + l + 35u] * s2 * f32(q2_3);
          acc += sharedX[outBase + l + 67u] * s3 * f32(q3_3);
          acc += sharedX[outBase + l + 99u] * s4 * f32(q4_3);
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
  q4_kGemvPipeline = null;
  q6_kGemvPipeline = null;
  q8_0GemmPipeline = null;
  q4_kGemmPipeline = null;
  q4_kGemmTiledPipeline = null;
  q6_kGemmTiledPipeline = null;
}
