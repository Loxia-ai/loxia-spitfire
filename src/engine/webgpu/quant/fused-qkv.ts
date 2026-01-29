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
// Fused QKV Projection for Q4_K quantized weights
// Reads input x ONCE into shared memory, writes Q, K, V to SEPARATE buffers
//
// x: f32 [1, K]
// Wq, Wk, Wv: Q4_K quantized [K, dim] (separate buffers)
// outQ, outK, outV: f32 outputs (separate buffers)

struct Params {
  K: u32,           // Input dimension (hidden size)
  qDim: u32,        // Q output dimension
  kDim: u32,        // K output dimension
  vDim: u32,        // V output dimension
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> Wq: array<u32>;
@group(0) @binding(2) var<storage, read> Wk: array<u32>;
@group(0) @binding(3) var<storage, read> Wv: array<u32>;
@group(0) @binding(4) var<storage, read_write> outQ: array<f32>;
@group(0) @binding(5) var<storage, read_write> outK: array<f32>;
@group(0) @binding(6) var<storage, read_write> outV: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;

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

// Dequantize a single weight element using correct memory layout
// Memory layout: linearIdx = col * K + k, blocks are column-by-column
fn dequant_weight_q4k(W: ptr<storage, array<u32>, read>, k: u32, col: u32, K: u32) -> f32 {
  let linearIdx = col * K + k;
  let blockIdx = linearIdx >> 8u;           // / 256
  let posInBlock = linearIdx & 255u;        // % 256
  let blockByteOffset = blockIdx * BYTES_PER_BLOCK;

  // Read d and dmin from block header
  let dBits = read_u16_q(W, blockByteOffset);
  let dminBits = read_u16_q(W, blockByteOffset + 2u);
  let d = f16_to_f32(dBits);
  let dmin = f16_to_f32(dminBits);

  // Determine sub-block (0..7) and position
  let subBlock = posInBlock >> 5u;          // / 32
  let posInSubBlock = posInBlock & 31u;     // % 32

  // Get scale and min for this sub-block
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

  // Read 4-bit quantized value
  let qsOffset = blockByteOffset + 16u;
  let byteIndex = ((subBlock >> 1u) << 5u) + posInSubBlock;  // (subBlock/2)*32 + pos
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

  // Each thread computes one output element
  let outIdx = wid.x * WG_SIZE + tid;
  if (outIdx >= totalOut) { return; }

  if (outIdx < qDim) {
    outQ[outIdx] = compute_gemv_q4k(&Wq, outIdx, qDim, K);
  } else if (outIdx < qDim + kDim) {
    outK[outIdx - qDim] = compute_gemv_q4k(&Wk, outIdx - qDim, kDim, K);
  } else {
    outV[outIdx - qDim - kDim] = compute_gemv_q4k(&Wv, outIdx - qDim - kDim, vDim, K);
  }
}
`;

let fusedQKVPipeline: GPUComputePipeline | null = null;

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

export function resetFusedQKVPipeline(): void {
  fusedQKVPipeline = null;
}
