/**
 * Fused FFN Gate+Up with SiLU
 *
 * Single kernel that computes: hidden = SiLU(x @ Wgate) * (x @ Wup)
 * Reads x once, applies activation, outputs directly to GPU buffer.
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
// Fused FFN Gate+Up Shader - Computes SiLU(gate) * up in one kernel
// ============================================================================

const FUSED_FFN_GATEUP_SHADER = `
// Fused FFN: hidden = SiLU(x @ Wgate) * (x @ Wup)
// Reads x ONCE, computes both projections, applies SiLU and multiply
//
// x: f32 [1, K]
// Wgate, Wup: Q4_K quantized [K, intermediateSize]
// output: f32 [1, intermediateSize]

struct Params {
  K: u32,              // Input dimension (hidden size)
  intermediateSize: u32,  // Output dimension
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> Wgate: array<u32>;
@group(0) @binding(2) var<storage, read> Wup: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

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

// SiLU activation: x * sigmoid(x)
fn silu(x: f32) -> f32 {
  return x / (1.0 + exp(-x));
}

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let tid = lid.x;
  let K = params.K;
  let intermediateSize = params.intermediateSize;

  // Load x into shared memory ONCE
  let loadIters = (K + WG_SIZE - 1u) / WG_SIZE;
  for (var i = 0u; i < loadIters; i++) {
    let idx = i * WG_SIZE + tid;
    if (idx < K) { sharedX[idx] = x[idx]; }
  }
  workgroupBarrier();

  // Each thread computes one output element: SiLU(gate) * up
  let outIdx = wid.x * WG_SIZE + tid;
  if (outIdx >= intermediateSize) { return; }

  let gate = compute_gemv_q4k(&Wgate, outIdx, intermediateSize, K);
  let up = compute_gemv_q4k(&Wup, outIdx, intermediateSize, K);

  output[outIdx] = silu(gate) * up;
}
`;

let fusedFFNPipeline: GPUComputePipeline | null = null;

/**
 * Fused FFN gate+up with SiLU activation
 * Single kernel: hidden = SiLU(x @ Wgate) * (x @ Wup)
 */
export async function fusedFFNGateUpQ4K(
  x: Tensor,
  wGate: QuantizedTensor,
  wUp: QuantizedTensor
): Promise<Tensor> {
  if (wGate.quantType !== GGMLType.Q4_K || wUp.quantType !== GGMLType.Q4_K) {
    throw new Error('fusedFFNGateUpQ4K only supports Q4_K weights');
  }

  const K = x.shape[x.ndim - 1];
  const intermediateSize = wGate.shape[1];

  if (!fusedFFNPipeline) {
    fusedFFNPipeline = createComputePipelineFromSource(FUSED_FFN_GATEUP_SHADER, {
      label: 'fused_ffn_gateup',
      entryPoint: 'main',
    });
  }

  const outputBuffer = createStorageBuffer(intermediateSize * 4, 'fused_ffn_hidden');

  const paramsData = new ArrayBuffer(16);
  const view = new DataView(paramsData);
  view.setUint32(0, K, true);
  view.setUint32(4, intermediateSize, true);
  view.setUint32(8, 0, true);
  view.setUint32(12, 0, true);
  const params = createUniformBufferWithData(new Uint8Array(paramsData), 'fused_ffn_params');

  const bindGroup = createBindGroup(fusedFFNPipeline, 0, [
    { binding: 0, resource: x.getBuffer() },
    { binding: 1, resource: wGate.getBuffer() },
    { binding: 2, resource: wUp.getBuffer() },
    { binding: 3, resource: outputBuffer },
    { binding: 4, resource: params },
  ]);

  const numWorkgroups = Math.ceil(intermediateSize / 256);
  const usedBuffers = [x.getBuffer(), wGate.getBuffer(), wUp.getBuffer(), outputBuffer, params];

  await executeCompute(fusedFFNPipeline, [bindGroup], [numWorkgroups, 1, 1], undefined, false, true, usedBuffers);

  const output = new Tensor([1, intermediateSize], outputBuffer, { label: 'ffn_hidden' });

  // Don't destroy params here - GPU commands are batched and may not have executed yet
  // The small uniform buffer (16 bytes) will be garbage collected

  return output;
}

export function resetFusedFFNPipeline(): void {
  fusedFFNPipeline = null;
}
