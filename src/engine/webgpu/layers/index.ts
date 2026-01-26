/**
 * WebGPU Transformer Layers
 * GPU-accelerated implementations of transformer building blocks
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
import { createUniformBufferWithData } from '../buffer.js';
import { matmul, softmax, mulScalar } from '../ops/index.js';

// ============================================================================
// Layer Normalization
// ============================================================================

const LAYERNORM_SHADER = `
struct Params {
  batchSize: u32,
  hiddenSize: u32,
  eps: f32,
  _pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 256u;
var<workgroup> wg_data: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let batchIdx = wid.x;
  let localIdx = lid.x;
  let hiddenSize = params.hiddenSize;
  let baseIdx = batchIdx * hiddenSize;

  // Step 1: Compute mean
  var localSum: f32 = 0.0;
  for (var i = localIdx; i < hiddenSize; i += WORKGROUP_SIZE) {
    localSum += input[baseIdx + i];
  }
  wg_data[localIdx] = localSum;
  workgroupBarrier();

  for (var s = WORKGROUP_SIZE / 2u; s > 0u; s >>= 1u) {
    if (localIdx < s) {
      wg_data[localIdx] += wg_data[localIdx + s];
    }
    workgroupBarrier();
  }
  let mean = wg_data[0] / f32(hiddenSize);
  workgroupBarrier();

  // Step 2: Compute variance
  var localVar: f32 = 0.0;
  for (var i = localIdx; i < hiddenSize; i += WORKGROUP_SIZE) {
    let diff = input[baseIdx + i] - mean;
    localVar += diff * diff;
  }
  wg_data[localIdx] = localVar;
  workgroupBarrier();

  for (var s = WORKGROUP_SIZE / 2u; s > 0u; s >>= 1u) {
    if (localIdx < s) {
      wg_data[localIdx] += wg_data[localIdx + s];
    }
    workgroupBarrier();
  }
  let variance = wg_data[0] / f32(hiddenSize);
  let rstd = inverseSqrt(variance + params.eps);
  workgroupBarrier();

  // Step 3: Normalize and apply affine transform
  for (var i = localIdx; i < hiddenSize; i += WORKGROUP_SIZE) {
    let normalized = (input[baseIdx + i] - mean) * rstd;
    output[baseIdx + i] = normalized * weight[i] + bias[i];
  }
}
`;

const RMSNORM_SHADER = `
struct Params {
  batchSize: u32,
  hiddenSize: u32,
  eps: f32,
  _pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 256u;
var<workgroup> wg_data: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let batchIdx = wid.x;
  let localIdx = lid.x;
  let hiddenSize = params.hiddenSize;
  let baseIdx = batchIdx * hiddenSize;

  // Compute sum of squares
  var localSumSq: f32 = 0.0;
  for (var i = localIdx; i < hiddenSize; i += WORKGROUP_SIZE) {
    let val = input[baseIdx + i];
    localSumSq += val * val;
  }
  wg_data[localIdx] = localSumSq;
  workgroupBarrier();

  for (var s = WORKGROUP_SIZE / 2u; s > 0u; s >>= 1u) {
    if (localIdx < s) {
      wg_data[localIdx] += wg_data[localIdx + s];
    }
    workgroupBarrier();
  }
  let rms = sqrt(wg_data[0] / f32(hiddenSize) + params.eps);
  let scale = 1.0 / rms;
  workgroupBarrier();

  // Normalize and apply weight
  for (var i = localIdx; i < hiddenSize; i += WORKGROUP_SIZE) {
    output[baseIdx + i] = input[baseIdx + i] * scale * weight[i];
  }
}
`;

let layerNormPipeline: GPUComputePipeline | null = null;
let rmsNormPipeline: GPUComputePipeline | null = null;

/**
 * Layer Normalization
 */
export async function layerNorm(
  input: Tensor,
  weight: Tensor,
  bias: Tensor,
  eps = 1e-5
): Promise<Tensor> {
  if (!layerNormPipeline) {
    layerNormPipeline = createComputePipelineFromSource(LAYERNORM_SHADER, {
      label: 'layernorm',
      entryPoint: 'main',
    });
  }

  const hiddenSize = input.shape[input.ndim - 1];
  const batchSize = input.size / hiddenSize;

  const output = Tensor.empty(input.shape, { label: 'layernorm_output' });

  // Pack params correctly: batchSize (u32), hiddenSize (u32), eps (f32), pad (u32)
  const paramsData = new ArrayBuffer(16);
  const paramsView = new DataView(paramsData);
  paramsView.setUint32(0, batchSize, true);
  paramsView.setUint32(4, hiddenSize, true);
  paramsView.setFloat32(8, eps, true);
  paramsView.setUint32(12, 0, true);

  const params = createUniformBufferWithData(new Uint8Array(paramsData), 'layernorm_params');

  const bindGroup = createBindGroup(layerNormPipeline, 0, [
    { binding: 0, resource: input.getBuffer() },
    { binding: 1, resource: weight.getBuffer() },
    { binding: 2, resource: bias.getBuffer() },
    { binding: 3, resource: output.getBuffer() },
    { binding: 4, resource: params },
  ]);

  const usedBuffers = [input.getBuffer(), weight.getBuffer(), bias.getBuffer(), output.getBuffer(), params];
  await executeCompute(layerNormPipeline, [bindGroup], [batchSize, 1, 1], undefined, false, true, usedBuffers);

  requestBufferDestroy(params);
  return output;
}

/**
 * RMS Normalization (used in Llama)
 */
export async function rmsNorm(
  input: Tensor,
  weight: Tensor,
  eps = 1e-5
): Promise<Tensor> {
  if (!rmsNormPipeline) {
    rmsNormPipeline = createComputePipelineFromSource(RMSNORM_SHADER, {
      label: 'rmsnorm',
      entryPoint: 'main',
    });
  }

  const hiddenSize = input.shape[input.ndim - 1];
  const batchSize = input.size / hiddenSize;

  const output = Tensor.empty(input.shape, { label: 'rmsnorm_output' });

  // Pack params correctly
  const paramsData = new ArrayBuffer(16);
  const paramsView = new DataView(paramsData);
  paramsView.setUint32(0, batchSize, true);
  paramsView.setUint32(4, hiddenSize, true);
  paramsView.setFloat32(8, eps, true);
  paramsView.setUint32(12, 0, true);

  const params = createUniformBufferWithData(new Uint8Array(paramsData), 'rmsnorm_params');

  const bindGroup = createBindGroup(rmsNormPipeline, 0, [
    { binding: 0, resource: input.getBuffer() },
    { binding: 1, resource: weight.getBuffer() },
    { binding: 2, resource: output.getBuffer() },
    { binding: 3, resource: params },
  ]);

  const usedBuffers = [input.getBuffer(), weight.getBuffer(), output.getBuffer(), params];
  await executeCompute(rmsNormPipeline, [bindGroup], [batchSize, 1, 1], undefined, false, true, usedBuffers);

  requestBufferDestroy(params);
  return output;
}

// ============================================================================
// Rotary Position Embeddings (RoPE)
// ============================================================================

const ROPE_SHADER = `
struct Params {
  seqLen: u32,
  headDim: u32,
  numHeads: u32,
  base: f32,
  startPos: u32,  // Starting position for incremental inference
  _pad1: u32,
  _pad2: u32,
  _pad3: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let seqLen = params.seqLen;
  let headDim = params.headDim;
  let numHeads = params.numHeads;
  let base = params.base;
  let startPos = params.startPos;

  let totalPairs = seqLen * numHeads * (headDim / 2u);
  let pairIdx = gid.x;

  if (pairIdx >= totalPairs) {
    return;
  }

  // Decode indices
  let halfDim = headDim / 2u;
  let localPos = pairIdx / (numHeads * halfDim);  // Position within this batch
  let absPos = startPos + localPos;                // Absolute position in sequence
  let remainder = pairIdx % (numHeads * halfDim);
  let head = remainder / halfDim;
  let d = remainder % halfDim;

  // Compute rotation angle using ABSOLUTE position
  // theta = absPos * inv_freq[d] where inv_freq[d] = 1 / base^(2d/headDim)
  let theta = f32(absPos) / pow(base, 2.0 * f32(d) / f32(headDim));
  let cosTheta = cos(theta);
  let sinTheta = sin(theta);

  // Input indices - SPLIT HALVES pattern (used by Qwen2/LLaMA-style)
  // x1 = first half (dims 0..halfDim-1)
  // x2 = second half (dims halfDim..headDim-1)
  // Pairs are (0,64), (1,65), (2,66), ... for headDim=128
  let baseIdx = (localPos * numHeads + head) * headDim;
  let idx1 = baseIdx + d;           // First half: 0, 1, 2, ... 63
  let idx2 = baseIdx + d + halfDim; // Second half: 64, 65, 66, ... 127

  let x1 = input[idx1];
  let x2 = input[idx2];

  // Apply rotation using rotate_half pattern:
  // output[d] = x1 * cos - x2 * sin
  // output[d + halfDim] = x2 * cos + x1 * sin
  output[idx1] = x1 * cosTheta - x2 * sinTheta;
  output[idx2] = x2 * cosTheta + x1 * sinTheta;
}
`;

let ropePipeline: GPUComputePipeline | null = null;

/**
 * Apply Rotary Position Embeddings
 * @param input - Input tensor [seqLen, numHeads * headDim]
 * @param seqLen - Number of tokens being processed
 * @param numHeads - Number of attention heads
 * @param headDim - Dimension per head
 * @param base - RoPE base frequency (default 10000)
 * @param startPos - Starting position for incremental inference (default 0)
 */
export async function applyRope(
  input: Tensor,
  seqLen: number,
  numHeads: number,
  headDim: number,
  base = 10000.0,
  startPos = 0
): Promise<Tensor> {
  if (!ropePipeline) {
    ropePipeline = createComputePipelineFromSource(ROPE_SHADER, {
      label: 'rope',
      entryPoint: 'main',
    });
  }

  const output = Tensor.empty(input.shape, { label: 'rope_output' });

  // Params struct is 32 bytes (8 u32s, with padding for alignment)
  const paramsData = new ArrayBuffer(32);
  const paramsView = new DataView(paramsData);
  paramsView.setUint32(0, seqLen, true);
  paramsView.setUint32(4, headDim, true);
  paramsView.setUint32(8, numHeads, true);
  paramsView.setFloat32(12, base, true);
  paramsView.setUint32(16, startPos, true);
  // Padding at 20, 24, 28

  const params = createUniformBufferWithData(new Uint8Array(paramsData), 'rope_params');

  const bindGroup = createBindGroup(ropePipeline, 0, [
    { binding: 0, resource: input.getBuffer() },
    { binding: 1, resource: output.getBuffer() },
    { binding: 2, resource: params },
  ]);

  const totalPairs = seqLen * numHeads * (headDim / 2);
  const workgroups = calculateWorkgroups(totalPairs, 64);
  const usedBuffers = [input.getBuffer(), output.getBuffer(), params];
  await executeCompute(ropePipeline, [bindGroup], [workgroups, 1, 1], undefined, false, true, usedBuffers);

  requestBufferDestroy(params);
  return output;
}

// ============================================================================
// Multi-Head Attention
// ============================================================================

export interface AttentionConfig {
  numHeads: number;
  headDim: number;
  scale?: number;
  causal?: boolean;
}

/**
 * Multi-Head Attention
 *
 * Q, K, V: [batchSize, seqLen, numHeads * headDim]
 * Output: [batchSize, seqLen, numHeads * headDim]
 */
export async function attention(
  q: Tensor,
  k: Tensor,
  v: Tensor,
  config: AttentionConfig
): Promise<Tensor> {
  const { headDim, scale, causal = false } = config;
  // Note: numHeads would be used in a full multi-head implementation
  // that reshapes tensors to [batch, heads, seq, dim]
  const actualScale = scale ?? 1.0 / Math.sqrt(headDim);

  // For simplicity, we assume batch=1 and work with [seqLen, hiddenSize]
  const seqLen = q.shape[0];
  const hiddenSize = q.shape[1];

  // Reshape Q, K, V to [seqLen, numHeads, headDim]
  // Then compute attention per head

  // QK^T: [seqLen, seqLen] per head
  // For now, simplified implementation without explicit head separation
  // This is a basic implementation - production would need proper reshaping

  // Q @ K^T
  const kT = await transposeLastTwo(k, seqLen, hiddenSize);
  let scores = await matmul(q, kT);
  kT.destroy();

  // Scale
  const scaledScores = await mulScalar(scores, actualScale);
  scores.destroy();
  scores = scaledScores;

  // Apply causal mask if needed (simplified - sets future to -inf)
  if (causal) {
    scores = await applyCausalMask(scores, seqLen);
  }

  // Softmax
  const attnWeights = await softmax(scores);
  scores.destroy();

  // Attention @ V
  const output = await matmul(attnWeights, v);
  attnWeights.destroy();

  return output;
}

/**
 * Transpose last two dimensions
 */
async function transposeLastTwo(
  input: Tensor,
  rows: number,
  cols: number
): Promise<Tensor> {
  // Simple CPU transpose for now
  const data = await input.toArray();
  const result = new Float32Array(data.length);

  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      result[j * rows + i] = data[i * cols + j];
    }
  }

  return Tensor.fromData(result, [cols, rows], { label: 'transpose' });
}

/**
 * Apply causal mask to attention scores
 */
async function applyCausalMask(scores: Tensor, seqLen: number): Promise<Tensor> {
  const data = await scores.toArray();

  for (let i = 0; i < seqLen; i++) {
    for (let j = i + 1; j < seqLen; j++) {
      data[i * seqLen + j] = -Infinity;
    }
  }

  return Tensor.fromData(data, scores.shape, { label: 'masked_scores' });
}

// ============================================================================
// Feed-Forward Network
// ============================================================================

/**
 * Feed-Forward Network with SiLU activation (Llama-style)
 *
 * gate = x @ W_gate
 * up = x @ W_up
 * output = (silu(gate) * up) @ W_down
 */
export async function feedForward(
  x: Tensor,
  wGate: Tensor,
  wUp: Tensor,
  wDown: Tensor
): Promise<Tensor> {
  const { fusedSwiGLU } = await import('../ops/index.js');

  // Fused SwiGLU: silu(x @ wGate) * (x @ wUp) in single kernel
  const hidden = await fusedSwiGLU(x, wGate, wUp);

  // Down projection
  const output = await matmul(hidden, wDown);
  hidden.destroy();

  return output;
}

/**
 * Simple MLP with GELU activation
 */
export async function mlp(
  x: Tensor,
  w1: Tensor,
  w2: Tensor
): Promise<Tensor> {
  const { gelu } = await import('../ops/index.js');

  const hidden = await matmul(x, w1);
  const activated = await gelu(hidden);
  hidden.destroy();

  const output = await matmul(activated, w2);
  activated.destroy();

  return output;
}

/**
 * Reset all cached pipelines (useful after shader updates)
 */
export function resetLayersPipelines(): void {
  layerNormPipeline = null;
  rmsNormPipeline = null;
  ropePipeline = null;
}
