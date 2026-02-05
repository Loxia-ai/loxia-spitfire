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
// Block-aware dequantization: processes both GEMVs with aligned Q4_K block reads.
// Reads block header once per 256 k-values, sub-block scale/min once per 32 k-values.
//
// x: f32 [1, K]
// Wgate, Wup: Q4_K quantized [K, intermediateSize]
// output: f32 [1, intermediateSize]

struct Params {
  K: u32,
  intermediateSize: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> Wgate: array<u32>;
@group(0) @binding(2) var<storage, read> Wup: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const BYTES_PER_BLOCK: u32 = 144u;
const WG_SIZE: u32 = 256u;

// Shared memory for cooperative x loading
var<workgroup> sharedX: array<f32, 256>;

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

fn read_byte_q(W: ptr<storage, array<u32>, read>, byteOffset: u32) -> u32 {
  let u32Idx = byteOffset >> 2u;
  let byteInU32 = byteOffset & 3u;
  return ((*W)[u32Idx] >> (byteInU32 << 3u)) & 0xFFu;
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

fn get_scale_min_q(W: ptr<storage, array<u32>, read>, blockByteOffset: u32, j: u32) -> vec2<u32> {
  let scalesOffset = blockByteOffset + 4u;
  var sc: u32;
  var mn: u32;
  if (j < 4u) {
    sc = read_byte_q(W, scalesOffset + j) & 0x3Fu;
    mn = read_byte_q(W, scalesOffset + j + 4u) & 0x3Fu;
  } else {
    let jm4 = j - 4u;
    sc = (read_byte_q(W, scalesOffset + j + 4u) & 0x0Fu) | ((read_byte_q(W, scalesOffset + jm4) >> 6u) << 4u);
    mn = (read_byte_q(W, scalesOffset + j + 4u) >> 4u) | ((read_byte_q(W, scalesOffset + j) >> 6u) << 4u);
  }
  return vec2<u32>(sc, mn);
}

fn silu(v: f32) -> f32 {
  return v / (1.0 + exp(-v));
}

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let tid = lid.x;
  let K = params.K;
  let intermediateSize = params.intermediateSize;
  let col = wid.x * WG_SIZE + tid;

  var gateAcc: f32 = 0.0;
  var upAcc: f32 = 0.0;

  let blocksPerCol = K >> 8u;

  for (var blk = 0u; blk < blocksPerCol; blk++) {
    let kBase = blk << 8u;

    // Cooperative load of 256 x values
    sharedX[tid] = x[kBase + tid];
    workgroupBarrier();

    if (col < intermediateSize) {
      let blockIdx = col * blocksPerCol + blk;
      let blockByteOffset = blockIdx * BYTES_PER_BLOCK;

      // Read block headers ONCE per 256 k-values (for both weights)
      let dGate = f16_to_f32(read_u16_q(&Wgate, blockByteOffset));
      let dminGate = f16_to_f32(read_u16_q(&Wgate, blockByteOffset + 2u));
      let dUp = f16_to_f32(read_u16_q(&Wup, blockByteOffset));
      let dminUp = f16_to_f32(read_u16_q(&Wup, blockByteOffset + 2u));

      let qsOffsetBase = blockByteOffset + 16u;

      // Process 8 sub-blocks of 32 values each
      for (var sb = 0u; sb < 8u; sb++) {
        // Read sub-block scales ONCE per 32 k-values
        let smGate = get_scale_min_q(&Wgate, blockByteOffset, sb);
        let scGate = dGate * f32(smGate.x);
        let mnGate = dminGate * f32(smGate.y);

        let smUp = get_scale_min_q(&Wup, blockByteOffset, sb);
        let scUp = dUp * f32(smUp.x);
        let mnUp = dminUp * f32(smUp.y);

        let kOff = sb << 5u;
        let byteBase = (sb >> 1u) << 5u;
        let isHigh = (sb & 1u) != 0u;

        let qsU32BaseGate = (qsOffsetBase + byteBase) >> 2u;
        let qsU32BaseUp = qsU32BaseGate;  // Same offset within block

        // Read 32 packed 4-bit values as 8 u32 reads for BOTH weights
        for (var w = 0u; w < 8u; w++) {
          let packedGate = Wgate[qsU32BaseGate + w];
          let packedUp = Wup[qsU32BaseUp + w];

          let b0g = packedGate & 0xFFu;
          let b1g = (packedGate >> 8u) & 0xFFu;
          let b2g = (packedGate >> 16u) & 0xFFu;
          let b3g = (packedGate >> 24u) & 0xFFu;

          let b0u = packedUp & 0xFFu;
          let b1u = (packedUp >> 8u) & 0xFFu;
          let b2u = (packedUp >> 16u) & 0xFFu;
          let b3u = (packedUp >> 24u) & 0xFFu;

          let q0g = f32(select(b0g & 0xFu, b0g >> 4u, isHigh));
          let q1g = f32(select(b1g & 0xFu, b1g >> 4u, isHigh));
          let q2g = f32(select(b2g & 0xFu, b2g >> 4u, isHigh));
          let q3g = f32(select(b3g & 0xFu, b3g >> 4u, isHigh));

          let q0u = f32(select(b0u & 0xFu, b0u >> 4u, isHigh));
          let q1u = f32(select(b1u & 0xFu, b1u >> 4u, isHigh));
          let q2u = f32(select(b2u & 0xFu, b2u >> 4u, isHigh));
          let q3u = f32(select(b3u & 0xFu, b3u >> 4u, isHigh));

          let idx = kOff + (w << 2u);
          let x0 = sharedX[idx];
          let x1 = sharedX[idx + 1u];
          let x2 = sharedX[idx + 2u];
          let x3 = sharedX[idx + 3u];

          gateAcc += x0 * (scGate * q0g - mnGate);
          gateAcc += x1 * (scGate * q1g - mnGate);
          gateAcc += x2 * (scGate * q2g - mnGate);
          gateAcc += x3 * (scGate * q3g - mnGate);

          upAcc += x0 * (scUp * q0u - mnUp);
          upAcc += x1 * (scUp * q1u - mnUp);
          upAcc += x2 * (scUp * q2u - mnUp);
          upAcc += x3 * (scUp * q3u - mnUp);
        }
      }
    }
    workgroupBarrier();
  }

  if (col < intermediateSize) {
    output[col] = silu(gateAcc) * upAcc;
  }
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

// ============================================================================
// Fused Q6_K FFN Gate+Up Shader - Optimized with batch reads
// ============================================================================

const FUSED_FFN_Q6K_SHADER = `
// Fused FFN for Q6_K: hidden = SiLU(x @ Wgate) * (x @ Wup)
// Uses read_byte for proper unaligned access (Q6_K blocks are 210 bytes)

struct Params {
  K: u32,
  intermediateSize: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> Wgate: array<u32>;
@group(0) @binding(2) var<storage, read> Wup: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const BYTES_PER_BLOCK: u32 = 210u;
const WG_SIZE: u32 = 256u;

var<workgroup> sharedX: array<f32, 256>;

fn read_byte_g(byteOffset: u32) -> u32 {
  let u32Idx = byteOffset >> 2u;
  let byteInU32 = byteOffset & 3u;
  return (Wgate[u32Idx] >> (byteInU32 << 3u)) & 0xFFu;
}

fn read_byte_u(byteOffset: u32) -> u32 {
  let u32Idx = byteOffset >> 2u;
  let byteInU32 = byteOffset & 3u;
  return (Wup[u32Idx] >> (byteInU32 << 3u)) & 0xFFu;
}

fn read_u16_g(byteOffset: u32) -> u32 {
  let u32Idx = byteOffset >> 2u;
  let byteInU32 = byteOffset & 3u;
  let data = Wgate[u32Idx];
  if (byteInU32 < 3u) {
    return (data >> (byteInU32 << 3u)) & 0xFFFFu;
  } else {
    let lo = (data >> 24u) & 0xFFu;
    let hi = Wgate[u32Idx + 1u] & 0xFFu;
    return lo | (hi << 8u);
  }
}

fn read_u16_u(byteOffset: u32) -> u32 {
  let u32Idx = byteOffset >> 2u;
  let byteInU32 = byteOffset & 3u;
  let data = Wup[u32Idx];
  if (byteInU32 < 3u) {
    return (data >> (byteInU32 << 3u)) & 0xFFFFu;
  } else {
    let lo = (data >> 24u) & 0xFFu;
    let hi = Wup[u32Idx + 1u] & 0xFFu;
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

fn silu(v: f32) -> f32 {
  return v / (1.0 + exp(-v));
}

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let tid = lid.x;
  let K = params.K;
  let intermediateSize = params.intermediateSize;
  let col = wid.x * WG_SIZE + tid;

  var gateAcc: f32 = 0.0;
  var upAcc: f32 = 0.0;
  let blocksPerCol = K >> 8u;

  for (var blk = 0u; blk < blocksPerCol; blk++) {
    let kBase = blk << 8u;

    sharedX[tid] = x[kBase + tid];
    workgroupBarrier();

    if (col < intermediateSize) {
      let blockIdx = col * blocksPerCol + blk;
      let blockByteOffset = blockIdx * BYTES_PER_BLOCK;

      let dGate = f16_to_f32(read_u16_g(blockByteOffset + 208u));
      let dUp = f16_to_f32(read_u16_u(blockByteOffset + 208u));

      // Process 2 chunks of 128 elements each
      for (var chunk = 0u; chunk < 2u; chunk++) {
        let qlBase = blockByteOffset + chunk * 64u;
        let qhBase = blockByteOffset + 128u + chunk * 32u;
        let scBase = blockByteOffset + 192u + chunk * 8u;
        let outBase = chunk * 128u;

        // Read and pre-multiply all 8 scales for this chunk (for both gate and up)
        let dsG0 = dGate * f32(int8_to_i32(read_byte_g(scBase + 0u)));
        let dsG1 = dGate * f32(int8_to_i32(read_byte_g(scBase + 1u)));
        let dsG2 = dGate * f32(int8_to_i32(read_byte_g(scBase + 2u)));
        let dsG3 = dGate * f32(int8_to_i32(read_byte_g(scBase + 3u)));
        let dsG4 = dGate * f32(int8_to_i32(read_byte_g(scBase + 4u)));
        let dsG5 = dGate * f32(int8_to_i32(read_byte_g(scBase + 5u)));
        let dsG6 = dGate * f32(int8_to_i32(read_byte_g(scBase + 6u)));
        let dsG7 = dGate * f32(int8_to_i32(read_byte_g(scBase + 7u)));

        let dsU0 = dUp * f32(int8_to_i32(read_byte_u(scBase + 0u)));
        let dsU1 = dUp * f32(int8_to_i32(read_byte_u(scBase + 1u)));
        let dsU2 = dUp * f32(int8_to_i32(read_byte_u(scBase + 2u)));
        let dsU3 = dUp * f32(int8_to_i32(read_byte_u(scBase + 3u)));
        let dsU4 = dUp * f32(int8_to_i32(read_byte_u(scBase + 4u)));
        let dsU5 = dUp * f32(int8_to_i32(read_byte_u(scBase + 5u)));
        let dsU6 = dUp * f32(int8_to_i32(read_byte_u(scBase + 6u)));
        let dsU7 = dUp * f32(int8_to_i32(read_byte_u(scBase + 7u)));

        // Process l = 0..31, handling all 4 quadrants at once
        for (var l = 0u; l < 32u; l++) {
          let qlG_l = read_byte_g(qlBase + l);
          let qlG_l32 = read_byte_g(qlBase + l + 32u);
          let qhG_l = read_byte_g(qhBase + l);
          let qlU_l = read_byte_u(qlBase + l);
          let qlU_l32 = read_byte_u(qlBase + l + 32u);
          let qhU_l = read_byte_u(qhBase + l);

          // Extract 6-bit quantized values for all 4 quadrants (gate)
          let qG1 = i32((qlG_l & 0xFu) | (((qhG_l >> 0u) & 3u) << 4u)) - 32;
          let qG2 = i32((qlG_l32 & 0xFu) | (((qhG_l >> 2u) & 3u) << 4u)) - 32;
          let qG3 = i32((qlG_l >> 4u) | (((qhG_l >> 4u) & 3u) << 4u)) - 32;
          let qG4 = i32((qlG_l32 >> 4u) | (((qhG_l >> 6u) & 3u) << 4u)) - 32;

          // Extract 6-bit quantized values for all 4 quadrants (up)
          let qU1 = i32((qlU_l & 0xFu) | (((qhU_l >> 0u) & 3u) << 4u)) - 32;
          let qU2 = i32((qlU_l32 & 0xFu) | (((qhU_l >> 2u) & 3u) << 4u)) - 32;
          let qU3 = i32((qlU_l >> 4u) | (((qhU_l >> 4u) & 3u) << 4u)) - 32;
          let qU4 = i32((qlU_l32 >> 4u) | (((qhU_l >> 6u) & 3u) << 4u)) - 32;

          // Select scales: is=0 for l<16, is=1 for l>=16
          let useHigh = l >= 16u;
          let sG1 = select(dsG0, dsG1, useHigh);
          let sG2 = select(dsG2, dsG3, useHigh);
          let sG3 = select(dsG4, dsG5, useHigh);
          let sG4 = select(dsG6, dsG7, useHigh);
          let sU1 = select(dsU0, dsU1, useHigh);
          let sU2 = select(dsU2, dsU3, useHigh);
          let sU3 = select(dsU4, dsU5, useHigh);
          let sU4 = select(dsU6, dsU7, useHigh);

          let x0 = sharedX[outBase + l];
          let x1 = sharedX[outBase + l + 32u];
          let x2 = sharedX[outBase + l + 64u];
          let x3 = sharedX[outBase + l + 96u];

          gateAcc += x0 * sG1 * f32(qG1);
          gateAcc += x1 * sG2 * f32(qG2);
          gateAcc += x2 * sG3 * f32(qG3);
          gateAcc += x3 * sG4 * f32(qG4);

          upAcc += x0 * sU1 * f32(qU1);
          upAcc += x1 * sU2 * f32(qU2);
          upAcc += x2 * sU3 * f32(qU3);
          upAcc += x3 * sU4 * f32(qU4);
        }
      }
    }
    workgroupBarrier();
  }

  if (col < intermediateSize) {
    output[col] = silu(gateAcc) * upAcc;
  }
}
`;

let fusedFFNQ6KPipeline: GPUComputePipeline | null = null;

/**
 * Fused FFN gate+up with SiLU activation for Q6_K
 * Single kernel: hidden = SiLU(x @ Wgate) * (x @ Wup)
 */
export async function fusedFFNGateUpQ6K(
  x: Tensor,
  wGate: QuantizedTensor,
  wUp: QuantizedTensor
): Promise<Tensor> {
  if (wGate.quantType !== GGMLType.Q6_K || wUp.quantType !== GGMLType.Q6_K) {
    throw new Error('fusedFFNGateUpQ6K only supports Q6_K weights');
  }

  const K = x.shape[x.ndim - 1];
  const intermediateSize = wGate.shape[1];

  if (!fusedFFNQ6KPipeline) {
    fusedFFNQ6KPipeline = createComputePipelineFromSource(FUSED_FFN_Q6K_SHADER, {
      label: 'fused_ffn_q6k',
      entryPoint: 'main',
    });
  }

  const outputBuffer = createStorageBuffer(intermediateSize * 4, 'fused_ffn_q6k_hidden');

  const paramsData = new ArrayBuffer(16);
  const view = new DataView(paramsData);
  view.setUint32(0, K, true);
  view.setUint32(4, intermediateSize, true);
  view.setUint32(8, 0, true);
  view.setUint32(12, 0, true);
  const params = createUniformBufferWithData(new Uint8Array(paramsData), 'fused_ffn_q6k_params');

  const bindGroup = createBindGroup(fusedFFNQ6KPipeline, 0, [
    { binding: 0, resource: x.getBuffer() },
    { binding: 1, resource: wGate.getBuffer() },
    { binding: 2, resource: wUp.getBuffer() },
    { binding: 3, resource: outputBuffer },
    { binding: 4, resource: params },
  ]);

  const numWorkgroups = Math.ceil(intermediateSize / 256);
  const usedBuffers = [x.getBuffer(), wGate.getBuffer(), wUp.getBuffer(), outputBuffer, params];

  await executeCompute(fusedFFNQ6KPipeline, [bindGroup], [numWorkgroups, 1, 1], undefined, false, true, usedBuffers);

  const output = new Tensor([1, intermediateSize], outputBuffer, { label: 'ffn_q6k_hidden' });

  return output;
}

export function resetFusedFFNPipeline(): void {
  fusedFFNPipeline = null;
  fusedFFNQ6KPipeline = null;
}
