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
// Q4_K Quantized GEMV Shader
// ============================================================================

const Q4_K_GEMV_SHADER = `
// Quantized GEMV: y = x @ W_q4k
// x: f32 [1, K]
// W: Q4_K quantized [K, N], stored as raw bytes
// y: f32 [1, N]
//
// Q4_K Block Format (144 bytes per 256 elements):
// - d (f16, 2 bytes): scale multiplier
// - dmin (f16, 2 bytes): min multiplier
// - scales[12]: packed 6-bit scales and mins for 8 sub-blocks
// - qs[128]: packed 4-bit quantized values

struct Params {
  N: u32,           // Output dimension (columns)
  K: u32,           // Input dimension (rows of W)
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> W_quant: array<u32>;  // Raw Q4_K bytes as u32
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const BLOCK_SIZE: u32 = 256u;
const BYTES_PER_BLOCK: u32 = 144u;
const SHARED_K: u32 = 256u;  // Tile size for K dimension (one Q4_K block)

var<workgroup> sharedX: array<f32, 256>;

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

// Get scale and min for sub-block j (0..7) from scales array
// scales array is at blockByteOffset + 4
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

// Read and dequantize a single weight value at W[k, n]
// GGUF stores data with ne0 (K) as contiguous dimension: linearIdx = n * K + k
fn dequant_weight_q4k(k: u32, n: u32, K: u32) -> f32 {
  let linearIdx = n * K + k;
  let blockIdx = linearIdx / BLOCK_SIZE;
  let posInBlock = linearIdx % BLOCK_SIZE;
  let blockByteOffset = blockIdx * BYTES_PER_BLOCK;

  // Read d and dmin from block header
  let dBits = read_u16(blockByteOffset);
  let dminBits = read_u16(blockByteOffset + 2u);
  let d = unpack_f16(dBits);
  let dmin = unpack_f16(dminBits);

  // Determine sub-block (0..7, each has 32 elements)
  let subBlock = posInBlock / 32u;
  let posInSubBlock = posInBlock % 32u;

  // Get scale and min for this sub-block
  let scaleMins = get_scale_min(blockByteOffset, subBlock);
  let scale = f32(scaleMins.x);
  let mn = f32(scaleMins.y);

  // Read 4-bit quantized value from qs array (starts at offset 16)
  // qs layout: for sub-block j and position l:
  //   byte_index = (j / 2) * 32 + l
  //   if j even: q = qs[byte_index] & 0xF
  //   if j odd: q = qs[byte_index] >> 4
  let qsOffset = blockByteOffset + 16u;
  let byteIndex = (subBlock / 2u) * 32u + posInSubBlock;
  let qByte = read_byte(qsOffset + byteIndex);
  var q: u32;
  if ((subBlock & 1u) == 0u) {
    q = qByte & 0x0Fu;
  } else {
    q = qByte >> 4u;
  }

  // Dequantize: value = d * scale * q - dmin * min
  return d * scale * f32(q) - dmin * mn;
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
    if (tid < tileLen) {
      sharedX[tid] = x[kStart + tid];
    }
    workgroupBarrier();

    // Accumulate dot products
    for (var kk = 0u; kk < tileLen; kk++) {
      let k = kStart + kk;
      let xVal = sharedX[kk];

      if (c0Valid) { acc0 += xVal * dequant_weight_q4k(k, baseCol, K); }
      if (c1Valid) { acc1 += xVal * dequant_weight_q4k(k, baseCol + 1u, K); }
      if (c2Valid) { acc2 += xVal * dequant_weight_q4k(k, baseCol + 2u, K); }
      if (c3Valid) { acc3 += xVal * dequant_weight_q4k(k, baseCol + 3u, K); }
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
// Tiled Quantized GEMM: Y = X @ W_q4k
// X: f32 [M, K]
// W: Q4_K quantized [K, N], stored as raw bytes
// Y: f32 [M, N]
//
// Key optimization: dequantize weight tiles into shared memory ONCE,
// then reuse across TILE_M=8 input rows. This eliminates redundant
// dequantization that the non-tiled version does M times per weight.

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
const TILE_M: u32 = 8u;
const TILE_N: u32 = 64u;
const TILE_K: u32 = 32u;

// Shared memory for dequantized weights and input tile
var<workgroup> sharedW: array<f32, 2048>;  // TILE_N * TILE_K = 64 * 32
var<workgroup> sharedX: array<f32, 256>;   // TILE_M * TILE_K = 8 * 32

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

// Dequantize a single Q4_K element given global k and n indices
fn dequant_q4k(k: u32, n: u32, K: u32) -> f32 {
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

  // Get scale and min for this sub-block
  let scalesOffset = blockByteOffset + 4u;
  var sc: u32;
  var mn: u32;
  if (subBlock < 4u) {
    sc = read_byte(scalesOffset + subBlock) & 0x3Fu;
    mn = read_byte(scalesOffset + subBlock + 4u) & 0x3Fu;
  } else {
    let jm4 = subBlock - 4u;
    sc = (read_byte(scalesOffset + subBlock + 4u) & 0x0Fu) | ((read_byte(scalesOffset + jm4) >> 6u) << 4u);
    mn = (read_byte(scalesOffset + subBlock + 4u) >> 4u) | ((read_byte(scalesOffset + subBlock) >> 6u) << 4u);
  }

  let qsOffset = blockByteOffset + 16u;
  let byteIndex = (subBlock / 2u) * 32u + posInSubBlock;
  let qByte = read_byte(qsOffset + byteIndex);
  var q: u32;
  if ((subBlock & 1u) == 0u) {
    q = qByte & 0x0Fu;
  } else {
    q = qByte >> 4u;
  }

  return d * f32(sc) * f32(q) - dmin * f32(mn);
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

  // This workgroup handles:
  //   columns: [wid.x * TILE_N .. wid.x * TILE_N + 63]
  //   rows:    [wid.y * TILE_M .. wid.y * TILE_M + 7]
  let colBase = wid.x * TILE_N;
  let rowBase = wid.y * TILE_M;
  let col = colBase + tid;  // Each thread owns one output column
  let validCol = col < N;

  // Accumulators for TILE_M rows
  var acc0: f32 = 0.0;
  var acc1: f32 = 0.0;
  var acc2: f32 = 0.0;
  var acc3: f32 = 0.0;
  var acc4: f32 = 0.0;
  var acc5: f32 = 0.0;
  var acc6: f32 = 0.0;
  var acc7: f32 = 0.0;

  let numKTiles = (K + TILE_K - 1u) / TILE_K;

  for (var kt = 0u; kt < numKTiles; kt++) {
    let kStart = kt * TILE_K;
    let kEnd = min(kStart + TILE_K, K);
    let tileLen = kEnd - kStart;

    // Step 1: Cooperatively load dequantized weights into shared memory
    // Each thread dequantizes TILE_K values for its column
    // sharedW layout: sharedW[tid * TILE_K + kk] = W[kStart+kk, col]
    if (validCol) {
      for (var kk = 0u; kk < tileLen; kk++) {
        sharedW[tid * TILE_K + kk] = dequant_q4k(kStart + kk, col, K);
      }
    } else {
      // Zero out to avoid garbage reads
      for (var kk = 0u; kk < TILE_K; kk++) {
        sharedW[tid * TILE_K + kk] = 0.0;
      }
    }

    // Step 2: Cooperatively load X tile into shared memory
    // sharedX layout: sharedX[m * TILE_K + kk]
    // 64 threads loading up to 8*32=256 values, 4 values per thread
    let xLoadCount = TILE_M * tileLen;
    let loadsPerThread = (xLoadCount + 63u) / 64u;
    for (var i = 0u; i < loadsPerThread; i++) {
      let flatIdx = tid + i * 64u;
      if (flatIdx < xLoadCount) {
        let mLocal = flatIdx / tileLen;
        let kk = flatIdx % tileLen;
        let globalRow = rowBase + mLocal;
        if (globalRow < M) {
          sharedX[mLocal * TILE_K + kk] = X[globalRow * K + kStart + kk];
        } else {
          sharedX[mLocal * TILE_K + kk] = 0.0;
        }
      }
    }

    workgroupBarrier();

    // Step 3: Accumulate — each thread processes its column across all M rows
    let wBase = tid * TILE_K;
    for (var kk = 0u; kk < tileLen; kk++) {
      let wVal = sharedW[wBase + kk];
      acc0 += sharedX[0u * TILE_K + kk] * wVal;
      acc1 += sharedX[1u * TILE_K + kk] * wVal;
      acc2 += sharedX[2u * TILE_K + kk] * wVal;
      acc3 += sharedX[3u * TILE_K + kk] * wVal;
      acc4 += sharedX[4u * TILE_K + kk] * wVal;
      acc5 += sharedX[5u * TILE_K + kk] * wVal;
      acc6 += sharedX[6u * TILE_K + kk] * wVal;
      acc7 += sharedX[7u * TILE_K + kk] * wVal;
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

  // 1024 columns per workgroup
  const numWorkgroups = Math.ceil(N / 1024);
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

/**
 * Reset cached pipelines (for testing/cleanup)
 */
export function resetQGemvPipelines(): void {
  q8_0GemvPipeline = null;
  q4_kGemvPipeline = null;
  q8_0GemmPipeline = null;
  q4_kGemmPipeline = null;
  q4_kGemmTiledPipeline = null;
}
