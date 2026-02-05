/**
 * Fused Quantized QKV Projection
 *
 * Single kernel that reads input x once and computes Q, K, V projections.
 * Writes directly to 3 separate GPU buffers - NO CPU transfer.
 */

import { Tensor } from '../tensor.js';
import { QuantizedTensor } from './qtensor.js';
import {
  createComputePipelineFromSource,
  createBindGroup,
  executeCompute,
} from '../shader.js';
import { createStorageBuffer, createUniformBufferWithData } from '../buffer.js';
import { GGMLType } from '../../../types/model.js';

// ============================================================================
// Fused Q4_K QKV Shader - Writes to 3 separate output buffers (no CPU copy)
// ============================================================================

const FUSED_QKV_Q4K_SHADER = `
// Optimized Fused QKV Projection for Q4_K quantized weights
// Reads input x ONCE into shared memory, writes Q, K, V to SEPARATE buffers
//
// Key optimizations:
// 1. Process K in Q4_K blocks (256 elements), read block header ONCE per block
// 2. Pre-compute d*scale products for all 8 sub-blocks
// 3. Batch u32 reads for qs values (Q4_K is 4-byte aligned: 144 bytes = 36 u32s)

struct Params {
  K: u32,
  qDim: u32,
  kDim: u32,
  vDim: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> Wq: array<u32>;
@group(0) @binding(2) var<storage, read> Wk: array<u32>;
@group(0) @binding(3) var<storage, read> Wv: array<u32>;
@group(0) @binding(4) var<storage, read_write> outQ: array<f32>;
@group(0) @binding(5) var<storage, read_write> outK: array<f32>;
@group(0) @binding(6) var<storage, read_write> outV: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;

const WG_SIZE: u32 = 256u;

var<workgroup> sharedX: array<f32, 4096>;

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

// Block-optimized GEMV: process one Q4_K block (256 elements) at a time
fn compute_gemv_q4k_opt(W: ptr<storage, array<u32>, read>, col: u32, K: u32) -> f32 {
  var acc: f32 = 0.0;
  let blocksPerCol = K >> 8u;

  for (var blk = 0u; blk < blocksPerCol; blk++) {
    let blockIdx = col * blocksPerCol + blk;
    let blockU32Idx = blockIdx * 36u;  // 144 bytes = 36 u32s per Q4_K block
    let kBase = blk << 8u;

    // Read d and dmin in ONE u32 read
    let dPacked = (*W)[blockU32Idx];
    let d = unpack_f16(dPacked & 0xFFFFu);
    let dmin = unpack_f16(dPacked >> 16u);

    // Read all 12 scale bytes as 3 u32s
    let sc0 = (*W)[blockU32Idx + 1u];
    let sc1 = (*W)[blockU32Idx + 2u];
    let sc2 = (*W)[blockU32Idx + 3u];

    // Extract and pre-compute d*scale, dmin*min for all 8 sub-blocks
    let ds0 = d * f32(sc0 & 0x3Fu);
    let ds1 = d * f32((sc0 >> 8u) & 0x3Fu);
    let ds2 = d * f32((sc0 >> 16u) & 0x3Fu);
    let ds3 = d * f32((sc0 >> 24u) & 0x3Fu);
    let dm0 = dmin * f32(sc1 & 0x3Fu);
    let dm1 = dmin * f32((sc1 >> 8u) & 0x3Fu);
    let dm2 = dmin * f32((sc1 >> 16u) & 0x3Fu);
    let dm3 = dmin * f32((sc1 >> 24u) & 0x3Fu);

    let ds4 = d * f32((sc2 & 0x0Fu) | ((sc0 >> 2u) & 0x30u));
    let ds5 = d * f32(((sc2 >> 8u) & 0x0Fu) | ((sc0 >> 10u) & 0x30u));
    let ds6 = d * f32(((sc2 >> 16u) & 0x0Fu) | ((sc0 >> 18u) & 0x30u));
    let ds7 = d * f32(((sc2 >> 24u) & 0x0Fu) | ((sc0 >> 26u) & 0x30u));
    let dm4 = dmin * f32(((sc2 >> 4u) & 0x0Fu) | ((sc1 >> 2u) & 0x30u));
    let dm5 = dmin * f32(((sc2 >> 12u) & 0x0Fu) | ((sc1 >> 10u) & 0x30u));
    let dm6 = dmin * f32(((sc2 >> 20u) & 0x0Fu) | ((sc1 >> 18u) & 0x30u));
    let dm7 = dmin * f32(((sc2 >> 28u) & 0x0Fu) | ((sc1 >> 26u) & 0x30u));

    // qs array starts at u32 offset 4
    let qsU32Base = blockU32Idx + 4u;

    // Process 4 pairs of sub-blocks (0&1, 2&3, 4&5, 6&7)
    for (var pair = 0u; pair < 4u; pair++) {
      let kOff = kBase + pair * 64u;

      // Select pre-computed scales for this pair
      var dsE: f32; var dmE: f32; var dsO: f32; var dmO: f32;
      if (pair == 0u) { dsE = ds0; dmE = dm0; dsO = ds1; dmO = dm1; }
      else if (pair == 1u) { dsE = ds2; dmE = dm2; dsO = ds3; dmO = dm3; }
      else if (pair == 2u) { dsE = ds4; dmE = dm4; dsO = ds5; dmO = dm5; }
      else { dsE = ds6; dmE = dm6; dsO = ds7; dmO = dm7; }

      // Read 8 u32s (32 bytes) for this sub-block pair
      let qsBase = qsU32Base + pair * 8u;

      for (var w = 0u; w < 8u; w++) {
        let packed = (*W)[qsBase + w];

        // Extract 8 4-bit values from this u32
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
  return acc;
}

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let tid = lid.x;
  let K = params.K;
  let qDim = params.qDim;
  let kDim = params.kDim;
  let vDim = params.vDim;
  let totalOut = qDim + kDim + vDim;

  // Load x into shared memory ONCE
  let loadIters = (K + WG_SIZE - 1u) / WG_SIZE;
  for (var i = 0u; i < loadIters; i++) {
    let idx = i * WG_SIZE + tid;
    if (idx < K) { sharedX[idx] = x[idx]; }
  }
  workgroupBarrier();

  // Each thread computes one output element
  let outIdx = wid.x * WG_SIZE + tid;
  if (outIdx >= totalOut) { return; }

  if (outIdx < qDim) {
    outQ[outIdx] = compute_gemv_q4k_opt(&Wq, outIdx, K);
  } else if (outIdx < qDim + kDim) {
    outK[outIdx - qDim] = compute_gemv_q4k_opt(&Wk, outIdx - qDim, K);
  } else {
    outV[outIdx - qDim - kDim] = compute_gemv_q4k_opt(&Wv, outIdx - qDim - kDim, K);
  }
}
`;

// ============================================================================
// Fused Q4_K QKV + Bias Shader - Adds biases inline (saves 3 dispatches/layer)
// ============================================================================

const FUSED_QKV_Q4K_BIAS_SHADER = `
// Fused QKV Projection with bias for Q4_K quantized weights
// Uses two bind groups to stay within WebGPU binding limits (max 8 per group)

struct Params {
  K: u32,
  qDim: u32,
  kDim: u32,
  vDim: u32,
}

// Group 0: Main computation (8 bindings)
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> Wq: array<u32>;
@group(0) @binding(2) var<storage, read> Wk: array<u32>;
@group(0) @binding(3) var<storage, read> Wv: array<u32>;
@group(0) @binding(4) var<storage, read_write> outQ: array<f32>;
@group(0) @binding(5) var<storage, read_write> outK: array<f32>;
@group(0) @binding(6) var<storage, read_write> outV: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;

// Group 1: Biases (3 bindings)
@group(1) @binding(0) var<storage, read> biasQ: array<f32>;
@group(1) @binding(1) var<storage, read> biasK: array<f32>;
@group(1) @binding(2) var<storage, read> biasV: array<f32>;

const BLOCK_SIZE: u32 = 256u;
const BYTES_PER_BLOCK: u32 = 144u;
const WG_SIZE: u32 = 256u;

var<workgroup> sharedX: array<f32, 4096>;

fn f16_to_f32(bits: u32) -> f32 {
  let sign = (bits >> 15u) & 1u;
  let exp = (bits >> 10u) & 0x1Fu;
  let mant = bits & 0x3FFu;
  if (exp == 0u) {
    if (mant == 0u) { return select(0.0, -0.0, sign == 1u); }
    let f = f32(mant) / 1024.0 * pow(2.0, -14.0);
    return select(f, -f, sign == 1u);
  } else if (exp == 31u) {
    return select(1.0e38, -1.0e38, sign == 1u);
  }
  let f = (1.0 + f32(mant) / 1024.0) * pow(2.0, f32(exp) - 15.0);
  return select(f, -f, sign == 1u);
}

fn read_byte_q(W: ptr<storage, array<u32>, read>, byteOffset: u32) -> u32 {
  let u32Idx = byteOffset >> 2u;
  let byteInU32 = byteOffset & 3u;
  return ((*W)[u32Idx] >> (byteInU32 << 3u)) & 0xFFu;
}

fn read_u16_q(W: ptr<storage, array<u32>, read>, byteOffset: u32) -> u32 {
  let u32Idx = byteOffset >> 2u;
  let byteInU32 = byteOffset & 3u;
  let data = (*W)[u32Idx];
  if (byteInU32 < 3u) {
    return (data >> (byteInU32 << 3u)) & 0xFFFFu;
  } else {
    let lo = (data >> 24u) & 0xFFu;
    let hi = (*W)[u32Idx + 1u] & 0xFFu;
    return lo | (hi << 8u);
  }
}

fn dequant_weight_q4k(W: ptr<storage, array<u32>, read>, k: u32, col: u32, K: u32) -> f32 {
  let linearIdx = col * K + k;
  let blockIdx = linearIdx >> 8u;
  let posInBlock = linearIdx & 255u;
  let blockByteOffset = blockIdx * BYTES_PER_BLOCK;

  let dBits = read_u16_q(W, blockByteOffset);
  let dminBits = read_u16_q(W, blockByteOffset + 2u);
  let d = f16_to_f32(dBits);
  let dmin = f16_to_f32(dminBits);

  let subBlock = posInBlock >> 5u;
  let posInSubBlock = posInBlock & 31u;
  let scalesOffset = blockByteOffset + 4u;
  var sc: u32;
  var mn: u32;

  if (subBlock < 4u) {
    sc = read_byte_q(W, scalesOffset + subBlock) & 0x3Fu;
    mn = read_byte_q(W, scalesOffset + subBlock + 4u) & 0x3Fu;
  } else {
    let jm4 = subBlock - 4u;
    sc = (read_byte_q(W, scalesOffset + subBlock + 4u) & 0x0Fu) | ((read_byte_q(W, scalesOffset + jm4) >> 6u) << 4u);
    mn = (read_byte_q(W, scalesOffset + subBlock + 4u) >> 4u) | ((read_byte_q(W, scalesOffset + subBlock) >> 6u) << 4u);
  }

  let qsOffset = blockByteOffset + 16u;
  let byteIndex = ((subBlock >> 1u) << 5u) + posInSubBlock;
  let qByte = read_byte_q(W, qsOffset + byteIndex);
  let q = select(qByte & 0x0Fu, qByte >> 4u, (subBlock & 1u) != 0u);

  return d * f32(sc) * f32(q) - dmin * f32(mn);
}

fn compute_gemv_q4k(W: ptr<storage, array<u32>, read>, col: u32, outDim: u32, K: u32) -> f32 {
  var sum: f32 = 0.0;
  for (var k = 0u; k < K; k++) {
    sum += sharedX[k] * dequant_weight_q4k(W, k, col, K);
  }
  return sum;
}

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let tid = lid.x;
  let K = params.K;
  let qDim = params.qDim;
  let kDim = params.kDim;
  let vDim = params.vDim;
  let totalOut = qDim + kDim + vDim;

  // Load x into shared memory ONCE
  let loadIters = (K + WG_SIZE - 1u) / WG_SIZE;
  for (var i = 0u; i < loadIters; i++) {
    let idx = i * WG_SIZE + tid;
    if (idx < K) { sharedX[idx] = x[idx]; }
  }
  workgroupBarrier();

  // Each thread computes one output element with bias addition
  let outIdx = wid.x * WG_SIZE + tid;
  if (outIdx >= totalOut) { return; }

  if (outIdx < qDim) {
    outQ[outIdx] = compute_gemv_q4k(&Wq, outIdx, qDim, K) + biasQ[outIdx];
  } else if (outIdx < qDim + kDim) {
    let col = outIdx - qDim;
    outK[col] = compute_gemv_q4k(&Wk, col, kDim, K) + biasK[col];
  } else {
    let col = outIdx - qDim - kDim;
    outV[col] = compute_gemv_q4k(&Wv, col, vDim, K) + biasV[col];
  }
}
`;

let fusedQKVPipeline: GPUComputePipeline | null = null;
let fusedQKVBiasPipeline: GPUComputePipeline | null = null;

/**
 * Fused QKV projection for Q4_K quantized weights
 * Single kernel, writes directly to 3 GPU buffers - NO toArray()!
 */
export async function fusedQKVProjectionQ4K(
  x: Tensor,
  wQ: QuantizedTensor,
  wK: QuantizedTensor,
  wV: QuantizedTensor
): Promise<{ Q: Tensor; K: Tensor; V: Tensor }> {
  if (wQ.quantType !== GGMLType.Q4_K || wK.quantType !== GGMLType.Q4_K || wV.quantType !== GGMLType.Q4_K) {
    throw new Error('fusedQKVProjectionQ4K only supports Q4_K weights');
  }

  const K = x.shape[x.ndim - 1];
  const qDim = wQ.shape[1];
  const kDim = wK.shape[1];
  const vDim = wV.shape[1];
  const totalOut = qDim + kDim + vDim;

  if (!fusedQKVPipeline) {
    fusedQKVPipeline = createComputePipelineFromSource(FUSED_QKV_Q4K_SHADER, {
      label: 'fused_qkv_q4k',
      entryPoint: 'main',
    });
  }

  // Create 3 separate output buffers - stay on GPU!
  const outQBuffer = createStorageBuffer(qDim * 4, 'fused_Q');
  const outKBuffer = createStorageBuffer(kDim * 4, 'fused_K');
  const outVBuffer = createStorageBuffer(vDim * 4, 'fused_V');

  const paramsData = new ArrayBuffer(16);
  const view = new DataView(paramsData);
  view.setUint32(0, K, true);
  view.setUint32(4, qDim, true);
  view.setUint32(8, kDim, true);
  view.setUint32(12, vDim, true);
  const params = createUniformBufferWithData(new Uint8Array(paramsData), 'fused_qkv_params');

  const bindGroup = createBindGroup(fusedQKVPipeline, 0, [
    { binding: 0, resource: x.getBuffer() },
    { binding: 1, resource: wQ.getBuffer() },
    { binding: 2, resource: wK.getBuffer() },
    { binding: 3, resource: wV.getBuffer() },
    { binding: 4, resource: outQBuffer },
    { binding: 5, resource: outKBuffer },
    { binding: 6, resource: outVBuffer },
    { binding: 7, resource: params },
  ]);

  const numWorkgroups = Math.ceil(totalOut / 256);
  const usedBuffers = [
    x.getBuffer(), wQ.getBuffer(), wK.getBuffer(), wV.getBuffer(),
    outQBuffer, outKBuffer, outVBuffer, params
  ];

  await executeCompute(fusedQKVPipeline, [bindGroup], [numWorkgroups, 1, 1], undefined, false, true, usedBuffers);

  // Create tensors directly from GPU buffers - NO toArray()!
  const Q = new Tensor([1, qDim], outQBuffer, { label: 'Q' });
  const K_out = new Tensor([1, kDim], outKBuffer, { label: 'K' });
  const V = new Tensor([1, vDim], outVBuffer, { label: 'V' });

  // Don't destroy params here - GPU commands are batched and may not have executed yet
  // The small uniform buffer (16 bytes) will be garbage collected

  return { Q, K: K_out, V };
}

/**
 * Fused QKV projection with bias addition for Q4_K quantized weights
 * Single kernel: computes GEMV + adds bias for Q, K, V in one dispatch.
 * Saves 3 broadcast_add dispatches per layer (108 per token for 36-layer model).
 */
export async function fusedQKVProjectionQ4KBias(
  x: Tensor,
  wQ: QuantizedTensor,
  wK: QuantizedTensor,
  wV: QuantizedTensor,
  biasQ: Tensor,
  biasK: Tensor,
  biasV: Tensor
): Promise<{ Q: Tensor; K: Tensor; V: Tensor }> {
  if (wQ.quantType !== GGMLType.Q4_K || wK.quantType !== GGMLType.Q4_K || wV.quantType !== GGMLType.Q4_K) {
    throw new Error('fusedQKVProjectionQ4KBias only supports Q4_K weights');
  }

  const K = x.shape[x.ndim - 1];
  const qDim = wQ.shape[1];
  const kDim = wK.shape[1];
  const vDim = wV.shape[1];
  const totalOut = qDim + kDim + vDim;

  if (!fusedQKVBiasPipeline) {
    fusedQKVBiasPipeline = createComputePipelineFromSource(FUSED_QKV_Q4K_BIAS_SHADER, {
      label: 'fused_qkv_q4k_bias',
      entryPoint: 'main',
    });
  }

  const outQBuffer = createStorageBuffer(qDim * 4, 'fused_Q');
  const outKBuffer = createStorageBuffer(kDim * 4, 'fused_K');
  const outVBuffer = createStorageBuffer(vDim * 4, 'fused_V');

  const paramsData = new ArrayBuffer(16);
  const view = new DataView(paramsData);
  view.setUint32(0, K, true);
  view.setUint32(4, qDim, true);
  view.setUint32(8, kDim, true);
  view.setUint32(12, vDim, true);
  const params = createUniformBufferWithData(new Uint8Array(paramsData), 'fused_qkv_bias_params');

  // Group 0: Main computation
  const bindGroup0 = createBindGroup(fusedQKVBiasPipeline, 0, [
    { binding: 0, resource: x.getBuffer() },
    { binding: 1, resource: wQ.getBuffer() },
    { binding: 2, resource: wK.getBuffer() },
    { binding: 3, resource: wV.getBuffer() },
    { binding: 4, resource: outQBuffer },
    { binding: 5, resource: outKBuffer },
    { binding: 6, resource: outVBuffer },
    { binding: 7, resource: params },
  ]);

  // Group 1: Biases
  const bindGroup1 = createBindGroup(fusedQKVBiasPipeline, 1, [
    { binding: 0, resource: biasQ.getBuffer() },
    { binding: 1, resource: biasK.getBuffer() },
    { binding: 2, resource: biasV.getBuffer() },
  ]);

  const numWorkgroups = Math.ceil(totalOut / 256);
  const usedBuffers = [
    x.getBuffer(), wQ.getBuffer(), wK.getBuffer(), wV.getBuffer(),
    outQBuffer, outKBuffer, outVBuffer, params,
    biasQ.getBuffer(), biasK.getBuffer(), biasV.getBuffer()
  ];

  await executeCompute(fusedQKVBiasPipeline, [bindGroup0, bindGroup1], [numWorkgroups, 1, 1], undefined, false, true, usedBuffers);

  const Q = new Tensor([1, qDim], outQBuffer, { label: 'Q' });
  const K_out = new Tensor([1, kDim], outKBuffer, { label: 'K' });
  const V = new Tensor([1, vDim], outVBuffer, { label: 'V' });

  return { Q, K: K_out, V };
}

// ============================================================================
// Fused Q6_K QKV Shader - Optimized with batch reads
// ============================================================================

const FUSED_QKV_Q6K_SHADER = `
// Fused QKV Projection for Q6_K quantized weights
// Uses read_byte for proper unaligned access (Q6_K blocks are 210 bytes)

struct Params {
  K: u32,
  qDim: u32,
  kDim: u32,
  vDim: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> Wq: array<u32>;
@group(0) @binding(2) var<storage, read> Wk: array<u32>;
@group(0) @binding(3) var<storage, read> Wv: array<u32>;
@group(0) @binding(4) var<storage, read_write> outQ: array<f32>;
@group(0) @binding(5) var<storage, read_write> outK: array<f32>;
@group(0) @binding(6) var<storage, read_write> outV: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;

const BYTES_PER_BLOCK: u32 = 210u;
const WG_SIZE: u32 = 256u;

var<workgroup> sharedX: array<f32, 4096>;

fn read_byte_q(W: ptr<storage, array<u32>, read>, byteOffset: u32) -> u32 {
  let u32Idx = byteOffset >> 2u;
  let byteInU32 = byteOffset & 3u;
  return ((*W)[u32Idx] >> (byteInU32 << 3u)) & 0xFFu;
}

fn read_u16_q(W: ptr<storage, array<u32>, read>, byteOffset: u32) -> u32 {
  let u32Idx = byteOffset >> 2u;
  let byteInU32 = byteOffset & 3u;
  let data = (*W)[u32Idx];
  if (byteInU32 < 3u) {
    return (data >> (byteInU32 << 3u)) & 0xFFFFu;
  } else {
    let lo = (data >> 24u) & 0xFFu;
    let hi = (*W)[u32Idx + 1u] & 0xFFu;
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

fn compute_gemv_q6k(W: ptr<storage, array<u32>, read>, col: u32, K: u32) -> f32 {
  var acc: f32 = 0.0;
  let blocksPerCol = K >> 8u;

  for (var blk = 0u; blk < blocksPerCol; blk++) {
    let kBase = blk << 8u;
    let blockIdx = col * blocksPerCol + blk;
    let blockByteOffset = blockIdx * BYTES_PER_BLOCK;

    let d = f16_to_f32(read_u16_q(W, blockByteOffset + 208u));

    // Process 2 chunks of 128 elements each
    for (var chunk = 0u; chunk < 2u; chunk++) {
      let qlBase = blockByteOffset + chunk * 64u;
      let qhBase = blockByteOffset + 128u + chunk * 32u;
      let scBase = blockByteOffset + 192u + chunk * 8u;
      let outBase = kBase + chunk * 128u;

      // Read and pre-multiply all 8 scales for this chunk
      let ds0 = d * f32(int8_to_i32(read_byte_q(W, scBase + 0u)));
      let ds1 = d * f32(int8_to_i32(read_byte_q(W, scBase + 1u)));
      let ds2 = d * f32(int8_to_i32(read_byte_q(W, scBase + 2u)));
      let ds3 = d * f32(int8_to_i32(read_byte_q(W, scBase + 3u)));
      let ds4 = d * f32(int8_to_i32(read_byte_q(W, scBase + 4u)));
      let ds5 = d * f32(int8_to_i32(read_byte_q(W, scBase + 5u)));
      let ds6 = d * f32(int8_to_i32(read_byte_q(W, scBase + 6u)));
      let ds7 = d * f32(int8_to_i32(read_byte_q(W, scBase + 7u)));

      // Process l = 0..31, handling all 4 quadrants at once
      for (var l = 0u; l < 32u; l++) {
        let ql_l = read_byte_q(W, qlBase + l);
        let ql_l32 = read_byte_q(W, qlBase + l + 32u);
        let qh_l = read_byte_q(W, qhBase + l);

        // Extract 6-bit quantized values for all 4 quadrants
        let q1 = i32((ql_l & 0xFu) | (((qh_l >> 0u) & 3u) << 4u)) - 32;
        let q2 = i32((ql_l32 & 0xFu) | (((qh_l >> 2u) & 3u) << 4u)) - 32;
        let q3 = i32((ql_l >> 4u) | (((qh_l >> 4u) & 3u) << 4u)) - 32;
        let q4 = i32((ql_l32 >> 4u) | (((qh_l >> 6u) & 3u) << 4u)) - 32;

        // Select scales: is=0 for l<16, is=1 for l>=16
        let useHigh = l >= 16u;
        let s1 = select(ds0, ds1, useHigh);
        let s2 = select(ds2, ds3, useHigh);
        let s3 = select(ds4, ds5, useHigh);
        let s4 = select(ds6, ds7, useHigh);

        // Accumulate all 4 quadrants
        acc += sharedX[outBase + l] * s1 * f32(q1);
        acc += sharedX[outBase + l + 32u] * s2 * f32(q2);
        acc += sharedX[outBase + l + 64u] * s3 * f32(q3);
        acc += sharedX[outBase + l + 96u] * s4 * f32(q4);
      }
    }
  }
  return acc;
}

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let tid = lid.x;
  let K = params.K;
  let qDim = params.qDim;
  let kDim = params.kDim;
  let vDim = params.vDim;
  let totalOut = qDim + kDim + vDim;

  // Load x into shared memory ONCE
  let loadIters = (K + WG_SIZE - 1u) / WG_SIZE;
  for (var i = 0u; i < loadIters; i++) {
    let idx = i * WG_SIZE + tid;
    if (idx < K) {
      sharedX[idx] = x[idx];
    }
  }
  workgroupBarrier();

  let outIdx = wid.x * WG_SIZE + tid;
  if (outIdx >= totalOut) { return; }

  if (outIdx < qDim) {
    outQ[outIdx] = compute_gemv_q6k(&Wq, outIdx, K);
  } else if (outIdx < qDim + kDim) {
    let col = outIdx - qDim;
    outK[col] = compute_gemv_q6k(&Wk, col, K);
  } else {
    let col = outIdx - qDim - kDim;
    outV[col] = compute_gemv_q6k(&Wv, col, K);
  }
}
`;

let fusedQKVQ6KPipeline: GPUComputePipeline | null = null;

/**
 * Fused QKV projection for Q6_K quantized weights
 * Single kernel, writes directly to 3 GPU buffers
 */
export async function fusedQKVProjectionQ6K(
  x: Tensor,
  wQ: QuantizedTensor,
  wK: QuantizedTensor,
  wV: QuantizedTensor
): Promise<{ Q: Tensor; K: Tensor; V: Tensor }> {
  if (wQ.quantType !== GGMLType.Q6_K || wK.quantType !== GGMLType.Q6_K || wV.quantType !== GGMLType.Q6_K) {
    throw new Error('fusedQKVProjectionQ6K only supports Q6_K weights');
  }

  const K = x.shape[x.ndim - 1];
  const qDim = wQ.shape[1];
  const kDim = wK.shape[1];
  const vDim = wV.shape[1];
  const totalOut = qDim + kDim + vDim;

  if (!fusedQKVQ6KPipeline) {
    fusedQKVQ6KPipeline = createComputePipelineFromSource(FUSED_QKV_Q6K_SHADER, {
      label: 'fused_qkv_q6k',
      entryPoint: 'main',
    });
  }

  const outQBuffer = createStorageBuffer(qDim * 4, 'fused_Q_q6k');
  const outKBuffer = createStorageBuffer(kDim * 4, 'fused_K_q6k');
  const outVBuffer = createStorageBuffer(vDim * 4, 'fused_V_q6k');

  const paramsData = new ArrayBuffer(16);
  const view = new DataView(paramsData);
  view.setUint32(0, K, true);
  view.setUint32(4, qDim, true);
  view.setUint32(8, kDim, true);
  view.setUint32(12, vDim, true);
  const params = createUniformBufferWithData(new Uint8Array(paramsData), 'fused_qkv_q6k_params');

  const bindGroup = createBindGroup(fusedQKVQ6KPipeline, 0, [
    { binding: 0, resource: x.getBuffer() },
    { binding: 1, resource: wQ.getBuffer() },
    { binding: 2, resource: wK.getBuffer() },
    { binding: 3, resource: wV.getBuffer() },
    { binding: 4, resource: outQBuffer },
    { binding: 5, resource: outKBuffer },
    { binding: 6, resource: outVBuffer },
    { binding: 7, resource: params },
  ]);

  const numWorkgroups = Math.ceil(totalOut / 256);
  const usedBuffers = [
    x.getBuffer(), wQ.getBuffer(), wK.getBuffer(), wV.getBuffer(),
    outQBuffer, outKBuffer, outVBuffer, params
  ];

  await executeCompute(fusedQKVQ6KPipeline, [bindGroup], [numWorkgroups, 1, 1], undefined, false, true, usedBuffers);

  const Q = new Tensor([1, qDim], outQBuffer, { label: 'Q' });
  const K_out = new Tensor([1, kDim], outKBuffer, { label: 'K' });
  const V = new Tensor([1, vDim], outVBuffer, { label: 'V' });

  return { Q, K: K_out, V };
}

export function resetFusedQKVPipeline(): void {
  fusedQKVPipeline = null;
  fusedQKVBiasPipeline = null;
  fusedQKVQ6KPipeline = null;
}
