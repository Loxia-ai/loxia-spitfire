/**
 * WebGPU Tensor Operations
 * GPU-accelerated tensor operations using compute shaders
 */

/// <reference types="@webgpu/types" />

import { Tensor } from '../tensor.js';
import {
  createComputePipelineFromSource,
  createBindGroup,
  executeCompute,
  calculateWorkgroups,
  calculateWorkgroups2D,
  requestBufferDestroy,
} from '../shader.js';
import {
  createStorageBuffer,
  createStorageBufferWithData,
  createUniformBufferWithData,
} from '../buffer.js';

// ============================================================================
// Element-wise Operations
// ============================================================================

const ELEMENTWISE_SHADER = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: vec4<u32>;

fn relu(x: f32) -> f32 { return max(x, 0.0); }
fn silu(x: f32) -> f32 { return x / (1.0 + exp(-x)); }
fn gelu(x: f32) -> f32 {
  let c = 0.7978845608;
  let inner = c * (x + 0.044715 * x * x * x);
  return 0.5 * x * (1.0 + tanh(inner));
}

@compute @workgroup_size(256)
fn relu_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.x) { return; }
  output[idx] = relu(input[idx]);
}

@compute @workgroup_size(256)
fn silu_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.x) { return; }
  output[idx] = silu(input[idx]);
}

@compute @workgroup_size(256)
fn gelu_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.x) { return; }
  output[idx] = gelu(input[idx]);
}

@compute @workgroup_size(256)
fn neg_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.x) { return; }
  output[idx] = -input[idx];
}

@compute @workgroup_size(256)
fn exp_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.x) { return; }
  output[idx] = exp(input[idx]);
}

@compute @workgroup_size(256)
fn sqrt_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.x) { return; }
  output[idx] = sqrt(input[idx]);
}

@compute @workgroup_size(256)
fn rsqrt_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.x) { return; }
  output[idx] = inverseSqrt(input[idx]);
}

@compute @workgroup_size(256)
fn tanh_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.x) { return; }
  output[idx] = tanh(input[idx]);
}
`;

const BINARY_SHADER = `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

@compute @workgroup_size(256)
fn add_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.x) { return; }
  output[idx] = a[idx] + b[idx];
}

@compute @workgroup_size(256)
fn sub_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.x) { return; }
  output[idx] = a[idx] - b[idx];
}

@compute @workgroup_size(256)
fn mul_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.x) { return; }
  output[idx] = a[idx] * b[idx];
}

@compute @workgroup_size(256)
fn div_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.x) { return; }
  output[idx] = a[idx] / b[idx];
}
`;

const SCALAR_SHADER = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: vec4<f32>; // x=size, y=scalar

@compute @workgroup_size(256)
fn add_scalar_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= u32(params.x)) { return; }
  output[idx] = input[idx] + params.y;
}

@compute @workgroup_size(256)
fn mul_scalar_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= u32(params.x)) { return; }
  output[idx] = input[idx] * params.y;
}
`;

// Cached pipelines
let elementwisePipelines: Map<string, GPUComputePipeline> | null = null;
let binaryPipelines: Map<string, GPUComputePipeline> | null = null;
let scalarPipelines: Map<string, GPUComputePipeline> | null = null;

function getElementwisePipeline(op: string): GPUComputePipeline {
  if (!elementwisePipelines) {
    elementwisePipelines = new Map();
  }
  let pipeline = elementwisePipelines.get(op);
  if (!pipeline) {
    pipeline = createComputePipelineFromSource(ELEMENTWISE_SHADER, {
      label: `elementwise_${op}`,
      entryPoint: `${op}_kernel`,
    });
    elementwisePipelines.set(op, pipeline);
  }
  return pipeline;
}

function getBinaryPipeline(op: string): GPUComputePipeline {
  if (!binaryPipelines) {
    binaryPipelines = new Map();
  }
  let pipeline = binaryPipelines.get(op);
  if (!pipeline) {
    pipeline = createComputePipelineFromSource(BINARY_SHADER, {
      label: `binary_${op}`,
      entryPoint: `${op}_kernel`,
    });
    binaryPipelines.set(op, pipeline);
  }
  return pipeline;
}

function getScalarPipeline(op: string): GPUComputePipeline {
  if (!scalarPipelines) {
    scalarPipelines = new Map();
  }
  let pipeline = scalarPipelines.get(op);
  if (!pipeline) {
    pipeline = createComputePipelineFromSource(SCALAR_SHADER, {
      label: `scalar_${op}`,
      entryPoint: `${op}_kernel`,
    });
    scalarPipelines.set(op, pipeline);
  }
  return pipeline;
}

async function runElementwise(input: Tensor, op: string): Promise<Tensor> {
  const pipeline = getElementwisePipeline(op);
  const output = Tensor.empty(input.shape, { label: `${op}_output` });
  const params = createUniformBufferWithData(
    new Uint32Array([input.size, 0, 0, 0]),
    'params'
  );

  const bindGroup = createBindGroup(pipeline, 0, [
    { binding: 0, resource: input.getBuffer() },
    { binding: 1, resource: output.getBuffer() },
    { binding: 2, resource: params },
  ]);

  const workgroups = calculateWorkgroups(input.size, 256);
  // Track buffers used by this operation for proper disposal timing (including params!)
  const usedBuffers = [input.getBuffer(), output.getBuffer(), params];
  await executeCompute(pipeline, [bindGroup], [workgroups, 1, 1], undefined, false, true, usedBuffers);

  requestBufferDestroy(params);
  return output;
}

async function runBinary(a: Tensor, b: Tensor, op: string): Promise<Tensor> {
  if (a.size !== b.size) {
    throw new Error(`Tensor sizes must match: ${a.size} vs ${b.size}`);
  }

  const pipeline = getBinaryPipeline(op);
  const output = Tensor.empty(a.shape, { label: `${op}_output` });
  const params = createUniformBufferWithData(
    new Uint32Array([a.size, 0, 0, 0]),
    'params'
  );

  const bindGroup = createBindGroup(pipeline, 0, [
    { binding: 0, resource: a.getBuffer() },
    { binding: 1, resource: b.getBuffer() },
    { binding: 2, resource: output.getBuffer() },
    { binding: 3, resource: params },
  ]);

  const workgroups = calculateWorkgroups(a.size, 256);
  const usedBuffers = [a.getBuffer(), b.getBuffer(), output.getBuffer(), params];
  await executeCompute(pipeline, [bindGroup], [workgroups, 1, 1], undefined, false, true, usedBuffers);

  requestBufferDestroy(params);
  return output;
}

async function runScalar(input: Tensor, scalar: number, op: string): Promise<Tensor> {
  const pipeline = getScalarPipeline(op);
  const output = Tensor.empty(input.shape, { label: `${op}_scalar_output` });
  const params = createUniformBufferWithData(
    new Float32Array([input.size, scalar, 0, 0]),
    'params'
  );

  const bindGroup = createBindGroup(pipeline, 0, [
    { binding: 0, resource: input.getBuffer() },
    { binding: 1, resource: output.getBuffer() },
    { binding: 2, resource: params },
  ]);

  const workgroups = calculateWorkgroups(input.size, 256);
  const usedBuffers = [input.getBuffer(), output.getBuffer(), params];
  await executeCompute(pipeline, [bindGroup], [workgroups, 1, 1], undefined, false, true, usedBuffers);

  requestBufferDestroy(params);
  return output;
}

// Export element-wise ops
export const relu = (x: Tensor) => runElementwise(x, 'relu');
export const silu = (x: Tensor) => runElementwise(x, 'silu');
export const gelu = (x: Tensor) => runElementwise(x, 'gelu');
export const neg = (x: Tensor) => runElementwise(x, 'neg');
export const exp = (x: Tensor) => runElementwise(x, 'exp');
export const sqrt = (x: Tensor) => runElementwise(x, 'sqrt');
export const rsqrt = (x: Tensor) => runElementwise(x, 'rsqrt');
export const tanh = (x: Tensor) => runElementwise(x, 'tanh');

// Export binary ops
export const add = (a: Tensor, b: Tensor) => runBinary(a, b, 'add');
export const sub = (a: Tensor, b: Tensor) => runBinary(a, b, 'sub');
export const mul = (a: Tensor, b: Tensor) => runBinary(a, b, 'mul');
export const div = (a: Tensor, b: Tensor) => runBinary(a, b, 'div');

// Export scalar ops
export const addScalar = (x: Tensor, s: number) => runScalar(x, s, 'add_scalar');
export const mulScalar = (x: Tensor, s: number) => runScalar(x, s, 'mul_scalar');

// ============================================================================
// Broadcast Operations
// ============================================================================

const BROADCAST_ADD_SHADER = `
// Broadcast add: input[seqLen, dim] + bias[dim] -> output[seqLen, dim]
struct Params {
  seqLen: u32,
  dim: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let totalSize = params.seqLen * params.dim;

  if (idx >= totalSize) { return; }

  // idx = seqIdx * dim + dimIdx
  let dimIdx = idx % params.dim;

  output[idx] = input[idx] + bias[dimIdx];
}
`;

let broadcastAddPipeline: GPUComputePipeline | null = null;

/**
 * Broadcast add: input[seqLen, dim] + bias[dim] -> output[seqLen, dim]
 * Adds bias vector to each row of the input matrix
 */
export async function broadcastAdd(input: Tensor, bias: Tensor): Promise<Tensor> {
  if (input.ndim !== 2) {
    throw new Error('broadcastAdd: input must be 2D');
  }
  if (bias.ndim !== 1) {
    throw new Error('broadcastAdd: bias must be 1D');
  }

  const [seqLen, dim] = input.shape;
  if (bias.shape[0] !== dim) {
    throw new Error(`broadcastAdd: bias size ${bias.shape[0]} doesn't match input dim ${dim}`);
  }

  if (!broadcastAddPipeline) {
    broadcastAddPipeline = createComputePipelineFromSource(BROADCAST_ADD_SHADER, {
      label: 'broadcast_add',
      entryPoint: 'main',
    });
  }

  const output = Tensor.empty([seqLen, dim], { label: 'broadcast_add_output' });

  const params = createUniformBufferWithData(
    new Uint32Array([seqLen, dim, 0, 0]),
    'broadcast_add_params'
  );

  const bindGroup = createBindGroup(broadcastAddPipeline, 0, [
    { binding: 0, resource: input.getBuffer() },
    { binding: 1, resource: bias.getBuffer() },
    { binding: 2, resource: output.getBuffer() },
    { binding: 3, resource: params },
  ]);

  const workgroups = calculateWorkgroups(seqLen * dim, 256);
  const usedBuffers = [input.getBuffer(), bias.getBuffer(), output.getBuffer(), params];
  await executeCompute(broadcastAddPipeline, [bindGroup], [workgroups, 1, 1], undefined, false, true, usedBuffers);

  requestBufferDestroy(params);
  return output;
}

// ============================================================================
// Matrix Multiplication
// ============================================================================

const MATMUL_SHADER = `
// Tiled matrix multiplication with 32x32 output tiles
// A: [M, K], B: [K, N], C: [M, N]
// Workgroup: 16x16 = 256 threads, each computes a 2x2 output block
// Compute:load ratio = 16:1 (vs 4:1 with 8x8 tiles)

struct Params {
  M: u32,
  N: u32,
  K: u32,
  _pad: u32,
}

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const TILE_MN: u32 = 32u;
const TILE_K: u32 = 32u;

var<workgroup> tileA: array<f32, 1024>;  // 32 * 32
var<workgroup> tileB: array<f32, 1024>;  // 32 * 32

@compute @workgroup_size(16, 16)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let tx = lid.x;  // 0..15
  let ty = lid.y;  // 0..15
  let tid = ty * 16u + tx;

  let M = params.M;
  let N = params.N;
  let K = params.K;

  // Each thread computes a 2x2 block of outputs
  let rowBase = wid.y * TILE_MN + ty * 2u;
  let colBase = wid.x * TILE_MN + tx * 2u;

  var s00: f32 = 0.0;
  var s01: f32 = 0.0;
  var s10: f32 = 0.0;
  var s11: f32 = 0.0;

  let numTiles = (K + TILE_K - 1u) / TILE_K;

  for (var t = 0u; t < numTiles; t++) {
    let kOffset = t * TILE_K;

    // Cooperative load: 256 threads load 1024 elements (4 per thread)
    for (var i = tid; i < 1024u; i += 256u) {
      let tileRow = i / TILE_K;
      let tileCol = i % TILE_K;
      let aRow = wid.y * TILE_MN + tileRow;
      let aCol = kOffset + tileCol;
      if (aRow < M && aCol < K) {
        tileA[i] = A[aRow * K + aCol];
      } else {
        tileA[i] = 0.0;
      }
    }

    for (var i = tid; i < 1024u; i += 256u) {
      let tileRow = i / TILE_MN;
      let tileCol = i % TILE_MN;
      let bRow = kOffset + tileRow;
      let bCol = wid.x * TILE_MN + tileCol;
      if (bRow < K && bCol < N) {
        tileB[i] = B[bRow * N + bCol];
      } else {
        tileB[i] = 0.0;
      }
    }

    workgroupBarrier();

    // Accumulate 2x2 block from tile
    for (var k = 0u; k < TILE_K; k++) {
      let a0 = tileA[(ty * 2u) * TILE_K + k];
      let a1 = tileA[(ty * 2u + 1u) * TILE_K + k];
      let b0 = tileB[k * TILE_MN + tx * 2u];
      let b1 = tileB[k * TILE_MN + tx * 2u + 1u];

      s00 += a0 * b0;
      s01 += a0 * b1;
      s10 += a1 * b0;
      s11 += a1 * b1;
    }

    workgroupBarrier();
  }

  // Write 2x2 results
  if (rowBase < M && colBase < N) { C[rowBase * N + colBase] = s00; }
  if (rowBase < M && (colBase + 1u) < N) { C[rowBase * N + colBase + 1u] = s01; }
  if ((rowBase + 1u) < M && colBase < N) { C[(rowBase + 1u) * N + colBase] = s10; }
  if ((rowBase + 1u) < M && (colBase + 1u) < N) { C[(rowBase + 1u) * N + colBase + 1u] = s11; }
}
`;

let matmulPipeline: GPUComputePipeline | null = null;

// ---- GEMV: Specialized matrix-vector multiply for M=1 ----
// When M=1 the input vector A is tiny. Load it entirely into shared memory
// with ONE barrier, then each thread computes 4 output columns by streaming
// through B with fully coalesced memory access. No tiling barriers on B,
// no branch divergence, 256 threads all active.

const GEMV_SHADER = `
struct Params {
  N: u32,
  K: u32,
  _p1: u32,
  _p2: u32,
}

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const SHARED_K: u32 = 4096u;
var<workgroup> sharedA: array<f32, 4096>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let tid = lid.x;
  let N = params.N;
  let K = params.K;

  // 4 output columns per thread, 256 threads = 1024 columns per workgroup
  let baseCol = wid.x * 1024u + tid * 4u;

  let c0Valid = baseCol < N;
  let c1Valid = (baseCol + 1u) < N;
  let c2Valid = (baseCol + 2u) < N;
  let c3Valid = (baseCol + 3u) < N;

  var s0: f32 = 0.0;
  var s1: f32 = 0.0;
  var s2: f32 = 0.0;
  var s3: f32 = 0.0;

  // Tile the K dimension in chunks of 4096
  let numTilesK = (K + SHARED_K - 1u) / SHARED_K;

  for (var tileK = 0u; tileK < numTilesK; tileK++) {
    let kStart = tileK * SHARED_K;
    let kEnd = min(kStart + SHARED_K, K);
    let tileLen = kEnd - kStart;

    // Cooperative load of A into shared memory
    for (var i = tid; i < tileLen; i += 256u) {
      sharedA[i] = A[kStart + i];
    }
    workgroupBarrier();

    // Each thread accumulates dot product for its 4 output columns
    for (var kk = 0u; kk < tileLen; kk++) {
      let aVal = sharedA[kk];
      let bBase = (kStart + kk) * N + baseCol;
      if (c0Valid) { s0 += aVal * B[bBase]; }
      if (c1Valid) { s1 += aVal * B[bBase + 1u]; }
      if (c2Valid) { s2 += aVal * B[bBase + 2u]; }
      if (c3Valid) { s3 += aVal * B[bBase + 3u]; }
    }
    workgroupBarrier();
  }

  if (c0Valid) { C[baseCol] = s0; }
  if (c1Valid) { C[baseCol + 1u] = s1; }
  if (c2Valid) { C[baseCol + 2u] = s2; }
  if (c3Valid) { C[baseCol + 3u] = s3; }
}
`;

let gemvPipeline: GPUComputePipeline | null = null;

/**
 * GEMV: matrix-vector multiply for M=1
 * A: [1, K], B: [K, N] -> C: [1, N]
 */
async function gemv(a: Tensor, b: Tensor): Promise<Tensor> {
  const [, K] = a.shape;
  const [, N] = b.shape;

  if (!gemvPipeline) {
    gemvPipeline = createComputePipelineFromSource(GEMV_SHADER, {
      label: 'gemv',
      entryPoint: 'main',
    });
  }

  const output = Tensor.empty([1, N], { label: 'gemv_output' });

  const params = createUniformBufferWithData(
    new Uint32Array([N, K, 0, 0]),
    'gemv_params'
  );

  const bindGroup = createBindGroup(gemvPipeline, 0, [
    { binding: 0, resource: a.getBuffer() },
    { binding: 1, resource: b.getBuffer() },
    { binding: 2, resource: output.getBuffer() },
    { binding: 3, resource: params },
  ]);

  // 1024 columns per workgroup (256 threads × 4 cols)
  const wgX = Math.ceil(N / 1024);
  const usedBuffers = [a.getBuffer(), b.getBuffer(), output.getBuffer(), params];
  await executeCompute(gemvPipeline, [bindGroup], [wgX, 1, 1], undefined, false, true, usedBuffers);

  requestBufferDestroy(params);
  return output;
}

/**
 * Matrix multiplication: C = A @ B
 * A: [M, K], B: [K, N] -> C: [M, N]
 *
 * Uses specialized GEMV shader when M=1 for much better performance
 * on single-token inference (avoids wasteful 8x8 workgroups).
 */
export async function matmul(a: Tensor, b: Tensor): Promise<Tensor> {
  if (a.ndim !== 2 || b.ndim !== 2) {
    throw new Error('matmul requires 2D tensors');
  }

  const [M, K1] = a.shape;
  const [K2, N] = b.shape;

  if (K1 !== K2) {
    throw new Error(`Inner dimensions must match: ${K1} vs ${K2}`);
  }

  // Use specialized GEMV shader for M=1 (single-token inference)
  if (M === 1) {
    return gemv(a, b);
  }

  // Tiled GEMM for M > 1
  if (!matmulPipeline) {
    matmulPipeline = createComputePipelineFromSource(MATMUL_SHADER, {
      label: 'matmul',
      entryPoint: 'main',
    });
  }

  const K = K1;
  const output = Tensor.empty([M, N], { label: 'matmul_output' });

  const params = createUniformBufferWithData(
    new Uint32Array([M, N, K, 0]),
    'matmul_params'
  );

  const bindGroup = createBindGroup(matmulPipeline, 0, [
    { binding: 0, resource: a.getBuffer() },
    { binding: 1, resource: b.getBuffer() },
    { binding: 2, resource: output.getBuffer() },
    { binding: 3, resource: params },
  ]);

  const [wgX, wgY] = calculateWorkgroups2D(N, M, 32, 32);
  const usedBuffers = [a.getBuffer(), b.getBuffer(), output.getBuffer(), params];
  await executeCompute(matmulPipeline, [bindGroup], [wgX, wgY, 1], undefined, false, true, usedBuffers);

  requestBufferDestroy(params);
  return output;
}

// ============================================================================
// Reduction Operations
// ============================================================================

const REDUCE_SHADER = `
struct Params {
  size: u32,
  stride: u32, // For reducing along specific axis
  reduceSize: u32,
  _pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 256u;
var<workgroup> wg_data: array<f32, 256>;

@compute @workgroup_size(256)
fn sum_kernel(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let idx = gid.x;
  let localIdx = lid.x;

  // Load to shared memory
  if (idx < params.size) {
    wg_data[localIdx] = input[idx];
  } else {
    wg_data[localIdx] = 0.0;
  }
  workgroupBarrier();

  // Parallel reduction
  for (var s = WORKGROUP_SIZE / 2u; s > 0u; s >>= 1u) {
    if (localIdx < s) {
      wg_data[localIdx] += wg_data[localIdx + s];
    }
    workgroupBarrier();
  }

  // Write result
  if (localIdx == 0u) {
    output[wid.x] = wg_data[0];
  }
}

@compute @workgroup_size(256)
fn max_kernel(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let idx = gid.x;
  let localIdx = lid.x;

  // Load to shared memory
  if (idx < params.size) {
    wg_data[localIdx] = input[idx];
  } else {
    wg_data[localIdx] = -1.0e+38; // -FLT_MAX
  }
  workgroupBarrier();

  // Parallel reduction
  for (var s = WORKGROUP_SIZE / 2u; s > 0u; s >>= 1u) {
    if (localIdx < s) {
      wg_data[localIdx] = max(wg_data[localIdx], wg_data[localIdx + s]);
    }
    workgroupBarrier();
  }

  // Write result
  if (localIdx == 0u) {
    output[wid.x] = wg_data[0];
  }
}
`;

let reducePipelines: Map<string, GPUComputePipeline> | null = null;

function getReducePipeline(op: string): GPUComputePipeline {
  if (!reducePipelines) {
    reducePipelines = new Map();
  }
  let pipeline = reducePipelines.get(op);
  if (!pipeline) {
    pipeline = createComputePipelineFromSource(REDUCE_SHADER, {
      label: `reduce_${op}`,
      entryPoint: `${op}_kernel`,
    });
    reducePipelines.set(op, pipeline);
  }
  return pipeline;
}

async function runReduce(input: Tensor, op: string): Promise<number> {
  const pipeline = getReducePipeline(op);

  let current = input;
  let currentSize = input.size;

  // Iteratively reduce until we have a single value
  while (currentSize > 1) {
    const workgroups = Math.ceil(currentSize / 256);
    const outputBuffer = createStorageBuffer(workgroups * 4, 'reduce_output');

    const params = createUniformBufferWithData(
      new Uint32Array([currentSize, 0, 0, 0]),
      'reduce_params'
    );

    const bindGroup = createBindGroup(pipeline, 0, [
      { binding: 0, resource: current.getBuffer() },
      { binding: 1, resource: outputBuffer },
      { binding: 2, resource: params },
    ]);

    const usedBuffers = [current.getBuffer(), outputBuffer, params];
    await executeCompute(pipeline, [bindGroup], [workgroups, 1, 1], undefined, false, true, usedBuffers);

    requestBufferDestroy(params);

    if (current !== input) {
      current.destroy();
    }

    current = new Tensor([workgroups], outputBuffer, { label: 'reduce_intermediate' });
    currentSize = workgroups;
  }

  const result = await current.toArray();
  if (current !== input) {
    current.destroy();
  }

  return result[0];
}

export const sum = (x: Tensor) => runReduce(x, 'sum');
export const max = (x: Tensor) => runReduce(x, 'max');

/**
 * Compute mean of tensor
 */
export async function mean(x: Tensor): Promise<number> {
  const s = await sum(x);
  return s / x.size;
}

// ============================================================================
// Softmax
// ============================================================================

const SOFTMAX_SHADER = `
struct Params {
  size: u32,      // Total elements
  innerSize: u32, // Size of each softmax (last dimension)
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 256u;
var<workgroup> wg_data: array<f32, 256>;

@compute @workgroup_size(256)
fn softmax_kernel(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let batchIdx = wid.x;
  let localIdx = lid.x;
  let innerSize = params.innerSize;
  let baseIdx = batchIdx * innerSize;

  // Step 1: Find max (for numerical stability)
  var localMax: f32 = -1.0e+38;
  for (var i = localIdx; i < innerSize; i += WORKGROUP_SIZE) {
    localMax = max(localMax, input[baseIdx + i]);
  }
  wg_data[localIdx] = localMax;
  workgroupBarrier();

  // Reduce to find global max
  for (var s = WORKGROUP_SIZE / 2u; s > 0u; s >>= 1u) {
    if (localIdx < s) {
      wg_data[localIdx] = max(wg_data[localIdx], wg_data[localIdx + s]);
    }
    workgroupBarrier();
  }
  let maxVal = wg_data[0];
  workgroupBarrier();

  // Step 2: Compute exp(x - max) and sum
  var localSum: f32 = 0.0;
  for (var i = localIdx; i < innerSize; i += WORKGROUP_SIZE) {
    let expVal = exp(input[baseIdx + i] - maxVal);
    output[baseIdx + i] = expVal;
    localSum += expVal;
  }
  wg_data[localIdx] = localSum;
  workgroupBarrier();

  // Reduce to find total sum
  for (var s = WORKGROUP_SIZE / 2u; s > 0u; s >>= 1u) {
    if (localIdx < s) {
      wg_data[localIdx] += wg_data[localIdx + s];
    }
    workgroupBarrier();
  }
  let sumVal = wg_data[0];
  workgroupBarrier();

  // Step 3: Normalize
  for (var i = localIdx; i < innerSize; i += WORKGROUP_SIZE) {
    output[baseIdx + i] /= sumVal;
  }
}
`;

let softmaxPipeline: GPUComputePipeline | null = null;

/**
 * Softmax along the last dimension
 */
export async function softmax(x: Tensor): Promise<Tensor> {
  if (!softmaxPipeline) {
    softmaxPipeline = createComputePipelineFromSource(SOFTMAX_SHADER, {
      label: 'softmax',
      entryPoint: 'softmax_kernel',
    });
  }

  const innerSize = x.shape[x.ndim - 1];
  const batchSize = x.size / innerSize;

  const output = Tensor.empty(x.shape, { label: 'softmax_output' });
  const params = createUniformBufferWithData(
    new Uint32Array([x.size, innerSize, 0, 0]),
    'softmax_params'
  );

  const bindGroup = createBindGroup(softmaxPipeline, 0, [
    { binding: 0, resource: x.getBuffer() },
    { binding: 1, resource: output.getBuffer() },
    { binding: 2, resource: params },
  ]);

  const usedBuffers = [x.getBuffer(), output.getBuffer(), params];
  await executeCompute(softmaxPipeline, [bindGroup], [batchSize, 1, 1], undefined, false, true, usedBuffers);

  requestBufferDestroy(params);
  return output;
}

/**
 * Reset all cached pipelines (useful after shader updates)
 */
export function resetOpsPipelines(): void {
  elementwisePipelines = null;
  binaryPipelines = null;
  scalarPipelines = null;
  matmulPipeline = null;
  gemvPipeline = null;
  reducePipelines = null;
  softmaxPipeline = null;
  embeddingPipeline = null;
  broadcastAddPipeline = null;
  sliceLastRowPipeline = null;
  copyRowsPipeline = null;
  causalAttentionPipeline = null;
  fusedSwiGLUGemvPipeline = null;
  fusedQKVGemvPipeline = null;
}

// ============================================================================
// Embedding Lookup
// ============================================================================

const EMBEDDING_SHADER = `
struct Params {
  seqLen: u32,
  hiddenSize: u32,
  vocabSize: u32,
  transposed: u32,  // 1 if embeddings are [hiddenSize, vocabSize], 0 if [vocabSize, hiddenSize]
}

@group(0) @binding(0) var<storage, read> embeddings: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let seqLen = params.seqLen;
  let hiddenSize = params.hiddenSize;
  let vocabSize = params.vocabSize;

  // Each thread handles one element in the output
  let totalElements = seqLen * hiddenSize;
  if (idx >= totalElements) {
    return;
  }

  // Determine which token and which hidden dimension
  let tokenIdx = idx / hiddenSize;
  let dimIdx = idx % hiddenSize;

  // Get the token ID for this position
  let tokenId = indices[tokenIdx];

  // Look up the embedding based on storage format
  var embeddingIdx: u32;
  if (params.transposed == 1u) {
    // Embeddings stored as [hiddenSize, vocabSize]
    // To get embedding for token t, we need column t: element at [d, t] = d * vocabSize + t
    embeddingIdx = dimIdx * vocabSize + tokenId;
  } else {
    // Embeddings stored as [vocabSize, hiddenSize]
    // To get embedding for token t, we need row t: element at [t, d] = t * hiddenSize + d
    embeddingIdx = tokenId * hiddenSize + dimIdx;
  }
  output[idx] = embeddings[embeddingIdx];
}
`;

let embeddingPipeline: GPUComputePipeline | null = null;

/**
 * Embedding lookup: maps token indices to embedding vectors
 * embeddings: [vocabSize, hiddenSize] or [hiddenSize, vocabSize] (GGUF format)
 * indices: [seqLen] (token IDs)
 * vocabSize: optional, helps detect transposed format
 * returns: [seqLen, hiddenSize]
 */
export async function embeddingLookup(
  embeddings: Tensor,
  indices: number[],
  vocabSizeHint?: number
): Promise<Tensor> {
  if (embeddings.ndim !== 2) {
    throw new Error('Embeddings must be 2D');
  }

  if (indices.length === 0) {
    throw new Error('embeddingLookup: indices array is empty - no tokens to process');
  }

  if (!embeddingPipeline) {
    embeddingPipeline = createComputePipelineFromSource(EMBEDDING_SHADER, {
      label: 'embedding',
      entryPoint: 'main',
    });
  }

  const [dim0, dim1] = embeddings.shape;
  const seqLen = indices.length;

  // Detect if embeddings are transposed (GGUF stores as [hiddenSize, vocabSize])
  // If vocabSizeHint is provided, use it to detect; otherwise assume larger dim is vocabSize
  let vocabSize: number;
  let hiddenSize: number;
  let transposed: number;

  if (vocabSizeHint !== undefined) {
    // Use hint to determine format
    if (dim0 === vocabSizeHint) {
      // Standard format [vocabSize, hiddenSize]
      vocabSize = dim0;
      hiddenSize = dim1;
      transposed = 0;
    } else if (dim1 === vocabSizeHint) {
      // Transposed format [hiddenSize, vocabSize]
      hiddenSize = dim0;
      vocabSize = dim1;
      transposed = 1;
    } else {
      throw new Error(`Embedding shape [${dim0}, ${dim1}] doesn't match vocabSize ${vocabSizeHint}`);
    }
  } else {
    // Assume larger dimension is vocabSize
    if (dim0 > dim1) {
      vocabSize = dim0;
      hiddenSize = dim1;
      transposed = 0;
    } else {
      hiddenSize = dim0;
      vocabSize = dim1;
      transposed = 1;
    }
  }

  // Create indices buffer
  const indicesBuffer = createStorageBufferWithData(
    new Uint32Array(indices),
    'embedding_indices'
  );

  const output = Tensor.empty([seqLen, hiddenSize], { label: 'embedding_output' });

  const params = createUniformBufferWithData(
    new Uint32Array([seqLen, hiddenSize, vocabSize, transposed]),
    'embedding_params'
  );

  const bindGroup = createBindGroup(embeddingPipeline, 0, [
    { binding: 0, resource: embeddings.getBuffer() },
    { binding: 1, resource: indicesBuffer },
    { binding: 2, resource: output.getBuffer() },
    { binding: 3, resource: params },
  ]);

  const totalElements = seqLen * hiddenSize;
  const workgroups = calculateWorkgroups(totalElements, 256);
  const usedBuffers = [embeddings.getBuffer(), indicesBuffer, output.getBuffer(), params];
  await executeCompute(embeddingPipeline, [bindGroup], [workgroups, 1, 1], undefined, false, true, usedBuffers);

  requestBufferDestroy(params);
  requestBufferDestroy(indicesBuffer);

  return output;
}

// ============================================================================
// GPU Causal Attention (Fused)
// ============================================================================

const CAUSAL_ATTENTION_SHADER = `
// Fused causal attention with online softmax (Flash-Attention style)
// Computes: softmax((Q @ K^T) * scale) @ V with causal masking
// Supports GQA where numKVHeads <= numHeads
// No sequence length limit — processes keys in chunks of 1024

struct Params {
  numQPos: u32,      // Number of query positions to process
  numKeys: u32,      // Total number of key positions
  numHeads: u32,     // Number of query heads
  numKVHeads: u32,   // Number of KV heads (for GQA)
  headDim: u32,      // Dimension per head
  startPos: u32,     // Starting position for causal mask
  scale: f32,        // 1/sqrt(headDim)
  _pad: u32,
}

@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

// Shared memory (~5.5KB total)
var<workgroup> sharedQ: array<f32, 128>;       // Q vector (max headDim=128)
var<workgroup> sharedScores: array<f32, 1024>; // Scores for current chunk
var<workgroup> sharedTemp: array<f32, 256>;    // Reduction workspace
var<workgroup> wgMax: f32;
var<workgroup> wgSum: f32;

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let numQPos = params.numQPos;
  let numKeys = params.numKeys;
  let numHeads = params.numHeads;
  let numKVHeads = params.numKVHeads;
  let headDim = params.headDim;
  let startPos = params.startPos;
  let scale = params.scale;

  let workIdx = wid.x;
  let qPos = workIdx / numHeads;
  let head = workIdx % numHeads;

  if (qPos >= numQPos) { return; }

  let kvRatio = numHeads / numKVHeads;
  let kvHead = head / kvRatio;

  let absQPos = startPos + qPos;
  let validKeys = min(absQPos + 1u, numKeys);

  let tid = lid.x;
  let WG_SIZE = 256u;
  let KPT = 4u;        // keys per thread
  let CHUNK = 1024u;   // WG_SIZE * KPT

  // ===== Phase 0: Load Q into shared memory =====
  let qBase = qPos * numHeads * headDim + head * headDim;
  if (tid < headDim) {
    sharedQ[tid] = Q[qBase + tid];
  }
  workgroupBarrier();

  // ===== Online softmax running state =====
  var m_prev: f32 = -1.0e+38;  // running max
  var l_prev: f32 = 0.0;       // running exp-sum
  var oAccum: f32 = 0.0;       // per-dimension output (lane 0 threads, tid < headDim)

  // Process keys in chunks of CHUNK
  let numChunks = (validKeys + CHUNK - 1u) / CHUNK;

  for (var chunk = 0u; chunk < numChunks; chunk++) {
    let chunkStart = chunk * CHUNK;
    let chunkEnd = min(chunkStart + CHUNK, validKeys);
    let chunkLen = chunkEnd - chunkStart;

    // ===== Phase 1: Compute Q @ K^T scores for this chunk =====
    // Each thread computes KPT dot products with vec4 optimization
    for (var j = 0u; j < KPT; j++) {
      let ki = tid * KPT + j;
      if (ki < chunkLen) {
        let globalKi = chunkStart + ki;
        var score: f32 = 0.0;
        let kBase = globalKi * numKVHeads * headDim + kvHead * headDim;

        // Vectorized dot product: process 4 elements at a time
        // headDim is typically 64, 80, or 128 (divisible by 4)
        let headDim4 = headDim & ~3u;  // Round down to multiple of 4
        for (var d = 0u; d < headDim4; d += 4u) {
          let q4 = vec4<f32>(sharedQ[d], sharedQ[d+1u], sharedQ[d+2u], sharedQ[d+3u]);
          let k4 = vec4<f32>(K[kBase + d], K[kBase + d + 1u], K[kBase + d + 2u], K[kBase + d + 3u]);
          score += dot(q4, k4);
        }
        // Handle remainder (if headDim not divisible by 4)
        for (var d = headDim4; d < headDim; d++) {
          score += sharedQ[d] * K[kBase + d];
        }

        sharedScores[ki] = score * scale;
      } else if (ki < CHUNK) {
        sharedScores[ki] = -1.0e+38;
      }
    }
    workgroupBarrier();

    // ===== Phase 2: Max reduction =====
    var threadMax: f32 = -1.0e+38;
    for (var j = 0u; j < KPT; j++) {
      let ki = tid * KPT + j;
      if (ki < chunkLen) {
        threadMax = max(threadMax, sharedScores[ki]);
      }
    }
    sharedTemp[tid] = threadMax;
    workgroupBarrier();

    for (var s = WG_SIZE / 2u; s > 0u; s >>= 1u) {
      if (tid < s) {
        sharedTemp[tid] = max(sharedTemp[tid], sharedTemp[tid + s]);
      }
      workgroupBarrier();
    }

    if (tid == 0u) {
      wgMax = sharedTemp[0];
    }
    workgroupBarrier();
    let chunkMax = wgMax;

    // ===== Phase 3: Exp + sum, update online softmax state =====
    let m_new = max(m_prev, chunkMax);
    let alpha = exp(m_prev - m_new);

    var threadSum: f32 = 0.0;
    for (var j = 0u; j < KPT; j++) {
      let ki = tid * KPT + j;
      if (ki < chunkLen) {
        let expVal = exp(sharedScores[ki] - m_new);
        sharedScores[ki] = expVal;
        threadSum += expVal;
      } else if (ki < CHUNK) {
        sharedScores[ki] = 0.0;
      }
    }
    sharedTemp[tid] = threadSum;
    workgroupBarrier();

    for (var s = WG_SIZE / 2u; s > 0u; s >>= 1u) {
      if (tid < s) {
        sharedTemp[tid] += sharedTemp[tid + s];
      }
      workgroupBarrier();
    }

    if (tid == 0u) {
      wgSum = sharedTemp[0];
    }
    workgroupBarrier();
    let chunkSum = wgSum;

    // Update running state
    l_prev = l_prev * alpha + chunkSum;
    m_prev = m_new;

    // ===== Phase 4: V accumulation (lane-parallel, all 256 threads active) =====
    let numLanes = WG_SIZE / headDim;
    let myDim = tid % headDim;
    let myLane = tid / headDim;

    // Each lane processes a stripe of keys for dimension myDim
    var vSum: f32 = 0.0;
    if (myLane < numLanes) {
      for (var ki = myLane; ki < chunkLen; ki += numLanes) {
        let vIdx = (chunkStart + ki) * numKVHeads * headDim + kvHead * headDim + myDim;
        vSum += sharedScores[ki] * V[vIdx];
      }
    }
    sharedTemp[tid] = vSum;
    workgroupBarrier();

    // Lane 0 reduces across lanes and applies online correction
    if (tid < headDim) {
      var vTotal: f32 = 0.0;
      for (var lane = 0u; lane < numLanes; lane++) {
        vTotal += sharedTemp[lane * headDim + tid];
      }
      oAccum = oAccum * alpha + vTotal;
    }
    workgroupBarrier();
  }

  // ===== Final output: oAccum / l_prev =====
  if (tid < headDim) {
    let outIdx = qPos * numHeads * headDim + head * headDim + tid;
    if (l_prev > 0.0) {
      output[outIdx] = oAccum / l_prev;
    } else {
      output[outIdx] = 0.0;
    }
  }
}
`;

let causalAttentionPipeline: GPUComputePipeline | null = null;

/**
 * GPU-accelerated causal attention with GQA support
 *
 * @param Q - Query tensor [numQPos, numHeads * headDim]
 * @param K - Key tensor [bufferSize, numKVHeads * headDim] (may be larger than numKeys for cached buffers)
 * @param V - Value tensor [bufferSize, numKVHeads * headDim] (may be larger than numKeys for cached buffers)
 * @param numHeads - Number of query heads
 * @param numKVHeads - Number of KV heads (for GQA)
 * @param headDim - Dimension per head
 * @param startPos - Starting position for queries (for incremental inference)
 * @param numKeysOverride - Optional: actual number of valid keys in K/V (for KV cache with pre-allocated buffers)
 * @returns Output tensor [numQPos, numHeads * headDim]
 */
export async function causalAttention(
  Q: Tensor,
  K: Tensor,
  V: Tensor,
  numHeads: number,
  numKVHeads: number,
  headDim: number,
  startPos: number = 0,
  numKeysOverride?: number
): Promise<Tensor> {
  if (!causalAttentionPipeline) {
    causalAttentionPipeline = createComputePipelineFromSource(CAUSAL_ATTENTION_SHADER, {
      label: 'causal_attention',
      entryPoint: 'main',
    });
  }

  const numQPos = Q.shape[0];
  // Use override if provided (for KV cache), otherwise use K's shape
  const numKeys = numKeysOverride ?? K.shape[0];
  const scale = 1.0 / Math.sqrt(headDim);

  const output = Tensor.empty([numQPos, numHeads * headDim], { label: 'attention_output' });

  // Pack params: numQPos, numKeys, numHeads, numKVHeads, headDim, startPos, scale, pad
  const paramsData = new ArrayBuffer(32);
  const paramsView = new DataView(paramsData);
  paramsView.setUint32(0, numQPos, true);
  paramsView.setUint32(4, numKeys, true);
  paramsView.setUint32(8, numHeads, true);
  paramsView.setUint32(12, numKVHeads, true);
  paramsView.setUint32(16, headDim, true);
  paramsView.setUint32(20, startPos, true);
  paramsView.setFloat32(24, scale, true);
  paramsView.setUint32(28, 0, true);

  const params = createUniformBufferWithData(new Uint8Array(paramsData), 'attention_params');

  const bindGroup = createBindGroup(causalAttentionPipeline, 0, [
    { binding: 0, resource: Q.getBuffer() },
    { binding: 1, resource: K.getBuffer() },
    { binding: 2, resource: V.getBuffer() },
    { binding: 3, resource: output.getBuffer() },
    { binding: 4, resource: params },
  ]);

  // One workgroup per (qPos, head) pair
  const numWorkgroups = numQPos * numHeads;
  const usedBuffers = [Q.getBuffer(), K.getBuffer(), V.getBuffer(), output.getBuffer(), params];
  await executeCompute(causalAttentionPipeline, [bindGroup], [numWorkgroups, 1, 1], undefined, false, true, usedBuffers);

  requestBufferDestroy(params);
  return output;
}

// ============================================================================
// Slice Last Row (for efficient last token extraction)
// ============================================================================

const SLICE_LAST_ROW_SHADER = `
struct Params {
  numRows: u32,
  numCols: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let numCols = params.numCols;
  let numRows = params.numRows;

  if (idx >= numCols) { return; }

  // Copy from last row of input to output
  let lastRowStart = (numRows - 1u) * numCols;
  output[idx] = input[lastRowStart + idx];
}
`;

let sliceLastRowPipeline: GPUComputePipeline | null = null;

/**
 * Extract the last row of a 2D tensor on GPU
 * input: [rows, cols] -> output: [1, cols]
 * This avoids transferring the entire tensor to CPU just to get the last row
 */
export async function sliceLastRow(input: Tensor): Promise<Tensor> {
  if (input.ndim !== 2) {
    throw new Error('sliceLastRow requires 2D tensor');
  }

  if (!sliceLastRowPipeline) {
    sliceLastRowPipeline = createComputePipelineFromSource(SLICE_LAST_ROW_SHADER, {
      label: 'slice_last_row',
      entryPoint: 'main',
    });
  }

  const [rows, cols] = input.shape;
  const output = Tensor.empty([1, cols], { label: 'last_row' });

  const params = createUniformBufferWithData(
    new Uint32Array([rows, cols, 0, 0]),
    'slice_last_row_params'
  );

  const bindGroup = createBindGroup(sliceLastRowPipeline, 0, [
    { binding: 0, resource: input.getBuffer() },
    { binding: 1, resource: output.getBuffer() },
    { binding: 2, resource: params },
  ]);

  const workgroups = calculateWorkgroups(cols, 256);
  const usedBuffers = [input.getBuffer(), output.getBuffer(), params];
  await executeCompute(sliceLastRowPipeline, [bindGroup], [workgroups, 1, 1], undefined, false, true, usedBuffers);

  requestBufferDestroy(params);
  return output;
}

// ============================================================================
// Slice Last N Rows (for speculative decoding multi-position logits)
// ============================================================================

const SLICE_LAST_ROWS_SHADER = `
struct Params {
  totalRows: u32,    // Total rows in input
  numCols: u32,      // Number of columns
  numRowsToSlice: u32, // How many rows to extract from end
  _pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let numCols = params.numCols;
  let numRowsToSlice = params.numRowsToSlice;
  let totalElements = numRowsToSlice * numCols;

  if (idx >= totalElements) { return; }

  // Calculate which row and column this thread handles
  let outRow = idx / numCols;
  let col = idx % numCols;

  // Map to input row (last numRowsToSlice rows)
  let startRow = params.totalRows - numRowsToSlice;
  let inRow = startRow + outRow;

  output[idx] = input[inRow * numCols + col];
}
`;

let sliceLastRowsPipeline: GPUComputePipeline | null = null;

/**
 * Extract the last N rows of a 2D tensor on GPU
 * input: [rows, cols] -> output: [numRows, cols]
 * Used for speculative decoding to get logits at multiple positions
 */
export async function sliceLastRows(input: Tensor, numRows: number): Promise<Tensor> {
  if (input.ndim !== 2) {
    throw new Error('sliceLastRows requires 2D tensor');
  }

  const [totalRows, cols] = input.shape;
  if (numRows > totalRows) {
    throw new Error(`Cannot slice ${numRows} rows from tensor with ${totalRows} rows`);
  }

  // Optimize: if only 1 row, use the existing sliceLastRow
  if (numRows === 1) {
    return sliceLastRow(input);
  }

  if (!sliceLastRowsPipeline) {
    sliceLastRowsPipeline = createComputePipelineFromSource(SLICE_LAST_ROWS_SHADER, {
      label: 'slice_last_rows',
      entryPoint: 'main',
    });
  }

  const output = Tensor.empty([numRows, cols], { label: 'last_rows' });

  const params = createUniformBufferWithData(
    new Uint32Array([totalRows, cols, numRows, 0]),
    'slice_last_rows_params'
  );

  const bindGroup = createBindGroup(sliceLastRowsPipeline, 0, [
    { binding: 0, resource: input.getBuffer() },
    { binding: 1, resource: output.getBuffer() },
    { binding: 2, resource: params },
  ]);

  const totalElements = numRows * cols;
  const workgroups = calculateWorkgroups(totalElements, 256);
  const usedBuffers = [input.getBuffer(), output.getBuffer(), params];
  await executeCompute(sliceLastRowsPipeline, [bindGroup], [workgroups, 1, 1], undefined, false, true, usedBuffers);

  requestBufferDestroy(params);
  return output;
}

// ============================================================================
// KV Cache Operations
// ============================================================================

const COPY_ROWS_SHADER = `
// Copy rows from source to destination at specified position
// Used for updating KV cache: cache[startRow:startRow+numRows] = source[0:numRows]
struct Params {
  numRows: u32,      // Number of rows to copy
  numCols: u32,      // Number of columns (row width)
  startRow: u32,     // Starting row in destination
  _pad: u32,
}

@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read_write> dest: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let totalElements = params.numRows * params.numCols;

  if (idx >= totalElements) { return; }

  // Calculate source and dest indices
  let row = idx / params.numCols;
  let col = idx % params.numCols;
  let destRow = params.startRow + row;

  let srcIdx = row * params.numCols + col;
  let dstIdx = destRow * params.numCols + col;

  dest[dstIdx] = source[srcIdx];
}
`;

let copyRowsPipeline: GPUComputePipeline | null = null;

/**
 * Copy rows from source tensor to destination tensor at a specific position
 * Used for updating KV cache: dest[startRow:startRow+source.rows] = source
 *
 * @param source - Source tensor [numRows, numCols]
 * @param dest - Destination tensor (must be large enough)
 * @param startRow - Starting row position in destination
 */
export async function copyRows(
  source: Tensor,
  dest: Tensor,
  startRow: number
): Promise<void> {
  if (source.ndim !== 2 || dest.ndim !== 2) {
    throw new Error('copyRows requires 2D tensors');
  }

  const [srcRows, srcCols] = source.shape;
  const [, destCols] = dest.shape;

  if (srcCols !== destCols) {
    throw new Error(`Column mismatch: source has ${srcCols}, dest has ${destCols}`);
  }

  if (!copyRowsPipeline) {
    copyRowsPipeline = createComputePipelineFromSource(COPY_ROWS_SHADER, {
      label: 'copy_rows',
      entryPoint: 'main',
    });
  }

  const params = createUniformBufferWithData(
    new Uint32Array([srcRows, srcCols, startRow, 0]),
    'copy_rows_params'
  );

  const bindGroup = createBindGroup(copyRowsPipeline, 0, [
    { binding: 0, resource: source.getBuffer() },
    { binding: 1, resource: dest.getBuffer() },
    { binding: 2, resource: params },
  ]);

  const totalElements = srcRows * srcCols;
  const workgroups = calculateWorkgroups(totalElements, 256);
  const usedBuffers = [source.getBuffer(), dest.getBuffer(), params];
  await executeCompute(copyRowsPipeline, [bindGroup], [workgroups, 1, 1], undefined, false, true, usedBuffers);

  requestBufferDestroy(params);
}

/**
 * Create a pre-allocated cache tensor for KV cache
 * @param maxSeqLen - Maximum sequence length to support
 * @param dim - Dimension per row (numKVHeads * headDim)
 */
export function createCacheTensor(maxSeqLen: number, dim: number): Tensor {
  return Tensor.zeros([maxSeqLen, dim], { label: 'kv_cache' });
}

// ============================================================================
// Transpose
// ============================================================================

const TRANSPOSE_SHADER = `
struct Params {
  rows: u32,
  cols: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.y;
  let col = gid.x;
  let rows = params.rows;
  let cols = params.cols;

  if (row >= rows || col >= cols) {
    return;
  }

  // input[row, col] -> output[col, row]
  let inIdx = row * cols + col;
  let outIdx = col * rows + row;
  output[outIdx] = input[inIdx];
}
`;

let transposePipeline: GPUComputePipeline | null = null;

/**
 * Transpose a 2D tensor
 * input: [rows, cols] -> output: [cols, rows]
 */
export async function transpose(input: Tensor): Promise<Tensor> {
  if (input.ndim !== 2) {
    throw new Error('transpose requires 2D tensor');
  }

  if (!transposePipeline) {
    transposePipeline = createComputePipelineFromSource(TRANSPOSE_SHADER, {
      label: 'transpose',
      entryPoint: 'main',
    });
  }

  const [rows, cols] = input.shape;
  const output = Tensor.empty([cols, rows], { label: 'transpose_output' });

  const params = createUniformBufferWithData(
    new Uint32Array([rows, cols, 0, 0]),
    'transpose_params'
  );

  const bindGroup = createBindGroup(transposePipeline, 0, [
    { binding: 0, resource: input.getBuffer() },
    { binding: 1, resource: output.getBuffer() },
    { binding: 2, resource: params },
  ]);

  const [wgX, wgY] = calculateWorkgroups2D(cols, rows, 16, 16);
  const usedBuffers = [input.getBuffer(), output.getBuffer(), params];
  await executeCompute(transposePipeline, [bindGroup], [wgX, wgY, 1], undefined, false, true, usedBuffers);

  requestBufferDestroy(params);
  return output;
}

// ============================================================================
// Fused SwiGLU (Gate + Up + SiLU + Mul)
// Combines: silu(x @ W_gate) * (x @ W_up) into single kernel
// ============================================================================

const FUSED_SWIGLU_SHADER = `
// Fused SwiGLU: output = silu(x @ W_gate) * (x @ W_up)
// Reads input x only once, eliminates intermediate tensors

struct Params {
  M: u32,            // Batch/sequence dimension
  hidden: u32,       // Input hidden dimension
  intermediate: u32, // Output intermediate dimension
  _pad: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> W_gate: array<f32>;
@group(0) @binding(2) var<storage, read> W_up: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

// SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))
fn silu(val: f32) -> f32 {
  return val / (1.0 + exp(-val));
}

// Tile size for K dimension (input hidden)
const TILE_K: u32 = 64u;
var<workgroup> sharedX: array<f32, 64>;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let tid = lid.x;
  let M = params.M;
  let hidden = params.hidden;
  let intermediate = params.intermediate;

  // Each workgroup handles one row (m), threads handle output columns
  let m = wid.y;
  let outCol = wid.x * 256u + tid;

  // Check if this thread computes a valid output (but don't return early - must hit barriers)
  let isValid = (m < M) && (outCol < intermediate);

  var gateAccum: f32 = 0.0;
  var upAccum: f32 = 0.0;

  let numTiles = (hidden + TILE_K - 1u) / TILE_K;

  for (var tile = 0u; tile < numTiles; tile++) {
    // Cooperative load of x tile into shared memory
    // First 64 threads load the tile
    if (tid < TILE_K) {
      let loadIdx = tile * TILE_K + tid;
      if (loadIdx < hidden && m < M) {
        sharedX[tid] = x[m * hidden + loadIdx];
      } else {
        sharedX[tid] = 0.0;
      }
    }
    workgroupBarrier();

    // Compute partial dot products for both gate and up (only if valid)
    if (isValid) {
      let tileEnd = min(TILE_K, hidden - tile * TILE_K);
      for (var k = 0u; k < tileEnd; k++) {
        let globalK = tile * TILE_K + k;
        let xVal = sharedX[k];

        // Weight access: W[globalK, outCol] = W[globalK * intermediate + outCol]
        let gateWeight = W_gate[globalK * intermediate + outCol];
        let upWeight = W_up[globalK * intermediate + outCol];

        gateAccum += xVal * gateWeight;
        upAccum += xVal * upWeight;
      }
    }
    workgroupBarrier();
  }

  // Apply SiLU to gate and multiply with up (only write if valid)
  if (isValid) {
    let gateAct = silu(gateAccum);
    output[m * intermediate + outCol] = gateAct * upAccum;
  }
}
`;

let fusedSwiGLUPipeline: GPUComputePipeline | null = null;

// ---- GEMV variant of Fused SwiGLU for M=1 ----
// Loads entire x vector into shared memory (1 barrier per K tile),
// each thread computes 4 output columns for both gate and up projections,
// applies silu(gate) * up inline.

const FUSED_SWIGLU_GEMV_SHADER = `
struct Params {
  hidden: u32,
  intermediate: u32,
  _p1: u32,
  _p2: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> W_gate: array<f32>;
@group(0) @binding(2) var<storage, read> W_up: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

fn silu(val: f32) -> f32 {
  return val / (1.0 + exp(-val));
}

const SHARED_K: u32 = 4096u;
var<workgroup> sharedX: array<f32, 4096>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let tid = lid.x;
  let hidden = params.hidden;
  let intermediate = params.intermediate;

  let baseCol = wid.x * 1024u + tid * 4u;

  let c0Valid = baseCol < intermediate;
  let c1Valid = (baseCol + 1u) < intermediate;
  let c2Valid = (baseCol + 2u) < intermediate;
  let c3Valid = (baseCol + 3u) < intermediate;

  var g0: f32 = 0.0; var g1: f32 = 0.0; var g2: f32 = 0.0; var g3: f32 = 0.0;
  var u0: f32 = 0.0; var u1: f32 = 0.0; var u2: f32 = 0.0; var u3: f32 = 0.0;

  let numTilesK = (hidden + SHARED_K - 1u) / SHARED_K;

  for (var tileK = 0u; tileK < numTilesK; tileK++) {
    let kStart = tileK * SHARED_K;
    let kEnd = min(kStart + SHARED_K, hidden);
    let tileLen = kEnd - kStart;

    for (var i = tid; i < tileLen; i += 256u) {
      sharedX[i] = x[kStart + i];
    }
    workgroupBarrier();

    for (var kk = 0u; kk < tileLen; kk++) {
      let xVal = sharedX[kk];
      let wBase = (kStart + kk) * intermediate + baseCol;
      if (c0Valid) { g0 += xVal * W_gate[wBase];     u0 += xVal * W_up[wBase]; }
      if (c1Valid) { g1 += xVal * W_gate[wBase + 1u]; u1 += xVal * W_up[wBase + 1u]; }
      if (c2Valid) { g2 += xVal * W_gate[wBase + 2u]; u2 += xVal * W_up[wBase + 2u]; }
      if (c3Valid) { g3 += xVal * W_gate[wBase + 3u]; u3 += xVal * W_up[wBase + 3u]; }
    }
    workgroupBarrier();
  }

  if (c0Valid) { output[baseCol]      = silu(g0) * u0; }
  if (c1Valid) { output[baseCol + 1u] = silu(g1) * u1; }
  if (c2Valid) { output[baseCol + 2u] = silu(g2) * u2; }
  if (c3Valid) { output[baseCol + 3u] = silu(g3) * u3; }
}
`;

let fusedSwiGLUGemvPipeline: GPUComputePipeline | null = null;

/**
 * Fused SwiGLU: silu(x @ W_gate) * (x @ W_up)
 * Reads input x only once and fuses activation with multiply
 * Reduces 4 kernel calls to 1
 *
 * Uses GEMV variant when M=1 for single-token inference.
 *
 * @param x - Input tensor [M, hidden]
 * @param wGate - Gate weight [hidden, intermediate]
 * @param wUp - Up weight [hidden, intermediate]
 * @returns Output tensor [M, intermediate]
 */
export async function fusedSwiGLU(
  x: Tensor,
  wGate: Tensor,
  wUp: Tensor
): Promise<Tensor> {
  const M = x.shape[0];
  const hidden = x.shape[1];
  const intermediate = wGate.shape[1];

  // GEMV fast path for M=1 (single-token inference)
  if (M === 1) {
    if (!fusedSwiGLUGemvPipeline) {
      fusedSwiGLUGemvPipeline = createComputePipelineFromSource(FUSED_SWIGLU_GEMV_SHADER, {
        label: 'fused_swiglu_gemv',
        entryPoint: 'main',
      });
    }

    const output = Tensor.empty([1, intermediate], { label: 'swiglu_gemv_output' });

    const params = createUniformBufferWithData(
      new Uint32Array([hidden, intermediate, 0, 0]),
      'swiglu_gemv_params'
    );

    const bindGroup = createBindGroup(fusedSwiGLUGemvPipeline, 0, [
      { binding: 0, resource: x.getBuffer() },
      { binding: 1, resource: wGate.getBuffer() },
      { binding: 2, resource: wUp.getBuffer() },
      { binding: 3, resource: output.getBuffer() },
      { binding: 4, resource: params },
    ]);

    const wgX = Math.ceil(intermediate / 1024);
    const usedBuffers = [x.getBuffer(), wGate.getBuffer(), wUp.getBuffer(), output.getBuffer(), params];
    await executeCompute(fusedSwiGLUGemvPipeline, [bindGroup], [wgX, 1, 1], undefined, false, true, usedBuffers);

    requestBufferDestroy(params);
    return output;
  }

  // Tiled GEMM path for M > 1
  if (!fusedSwiGLUPipeline) {
    fusedSwiGLUPipeline = createComputePipelineFromSource(FUSED_SWIGLU_SHADER, {
      label: 'fused_swiglu',
      entryPoint: 'main',
    });
  }

  const output = Tensor.empty([M, intermediate], { label: 'swiglu_output' });

  // Pack params: M, hidden, intermediate, pad
  const paramsData = new ArrayBuffer(16);
  const view = new DataView(paramsData);
  view.setUint32(0, M, true);
  view.setUint32(4, hidden, true);
  view.setUint32(8, intermediate, true);
  view.setUint32(12, 0, true);

  const params = createUniformBufferWithData(new Uint8Array(paramsData), 'swiglu_params');

  const bindGroup = createBindGroup(fusedSwiGLUPipeline, 0, [
    { binding: 0, resource: x.getBuffer() },
    { binding: 1, resource: wGate.getBuffer() },
    { binding: 2, resource: wUp.getBuffer() },
    { binding: 3, resource: output.getBuffer() },
    { binding: 4, resource: params },
  ]);

  // Workgroups: X covers intermediate dimension, Y covers M
  const wgX = Math.ceil(intermediate / 256);
  const wgY = M;

  const usedBuffers = [x.getBuffer(), wGate.getBuffer(), wUp.getBuffer(), output.getBuffer(), params];
  await executeCompute(fusedSwiGLUPipeline, [bindGroup], [wgX, wgY, 1], undefined, false, true, usedBuffers);

  requestBufferDestroy(params);
  return output;
}

// ============================================================================
// Fused QKV Projection
// Combines: Q = x @ Wq, K = x @ Wk, V = x @ Wv into single kernel
// ============================================================================

const FUSED_QKV_SHADER = `
// Fused QKV Projection: outputs Q, K, V from single input read
// Reads input x only once instead of 3 times

struct Params {
  M: u32,       // Batch/sequence dimension
  hidden: u32,  // Input hidden dimension
  qDim: u32,    // Q output dimension (numHeads * headDim)
  kDim: u32,    // K output dimension (numKVHeads * headDim)
  vDim: u32,    // V output dimension (numKVHeads * headDim)
  _pad1: u32,
  _pad2: u32,
  _pad3: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> Wq: array<f32>;
@group(0) @binding(2) var<storage, read> Wk: array<f32>;
@group(0) @binding(3) var<storage, read> Wv: array<f32>;
@group(0) @binding(4) var<storage, read_write> Q: array<f32>;
@group(0) @binding(5) var<storage, read_write> K: array<f32>;
@group(0) @binding(6) var<storage, read_write> V: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;

const TILE_K: u32 = 64u;
var<workgroup> sharedX: array<f32, 64>;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let tid = lid.x;
  let M = params.M;
  let hidden = params.hidden;
  let qDim = params.qDim;
  let kDim = params.kDim;
  let vDim = params.vDim;

  // Row index (batch dimension)
  let m = wid.y;

  // Global output column across all three matrices
  let globalCol = wid.x * 256u + tid;
  let totalCols = qDim + kDim + vDim;

  // Check validity but don't return early (must hit barriers)
  let isValid = (m < M) && (globalCol < totalCols);

  // Determine which matrix (Q=0, K=1, V=2) and local column
  // Use safe defaults for invalid threads
  var targetMatrix: u32 = 0u;
  var localCol: u32 = 0u;

  if (isValid) {
    if (globalCol < qDim) {
      targetMatrix = 0u;
      localCol = globalCol;
    } else if (globalCol < qDim + kDim) {
      targetMatrix = 1u;
      localCol = globalCol - qDim;
    } else {
      targetMatrix = 2u;
      localCol = globalCol - qDim - kDim;
    }
  }

  var accum: f32 = 0.0;
  let numTiles = (hidden + TILE_K - 1u) / TILE_K;

  for (var tile = 0u; tile < numTiles; tile++) {
    // Cooperative load of x tile
    if (tid < TILE_K) {
      let loadIdx = tile * TILE_K + tid;
      if (loadIdx < hidden && m < M) {
        sharedX[tid] = x[m * hidden + loadIdx];
      } else {
        sharedX[tid] = 0.0;
      }
    }
    workgroupBarrier();

    // Compute partial dot product (only if valid)
    if (isValid) {
      let tileEnd = min(TILE_K, hidden - tile * TILE_K);
      for (var k = 0u; k < tileEnd; k++) {
        let globalK = tile * TILE_K + k;
        let xVal = sharedX[k];

        var weight: f32;
        if (targetMatrix == 0u) {
          weight = Wq[globalK * qDim + localCol];
        } else if (targetMatrix == 1u) {
          weight = Wk[globalK * kDim + localCol];
        } else {
          weight = Wv[globalK * vDim + localCol];
        }

        accum += xVal * weight;
      }
    }
    workgroupBarrier();
  }

  // Write output to appropriate matrix (only if valid)
  if (isValid) {
    if (targetMatrix == 0u) {
      Q[m * qDim + localCol] = accum;
    } else if (targetMatrix == 1u) {
      K[m * kDim + localCol] = accum;
    } else {
      V[m * vDim + localCol] = accum;
    }
  }
}
`;

let fusedQKVPipeline: GPUComputePipeline | null = null;

// ---- GEMV variant of Fused QKV Projection for M=1 ----
// Loads entire x vector into shared memory (1 barrier per K tile),
// each thread computes 4 output columns across the combined Q+K+V space.
// Determines target weight matrix by column range — no inner-loop branching
// since LLM dimensions are always multiples of headDim (>=64).

const FUSED_QKV_GEMV_SHADER = `
struct Params {
  hidden: u32,
  qDim: u32,
  kDim: u32,
  vDim: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> Wq: array<f32>;
@group(0) @binding(2) var<storage, read> Wk: array<f32>;
@group(0) @binding(3) var<storage, read> Wv: array<f32>;
@group(0) @binding(4) var<storage, read_write> Q: array<f32>;
@group(0) @binding(5) var<storage, read_write> K: array<f32>;
@group(0) @binding(6) var<storage, read_write> V: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;

const SHARED_K: u32 = 4096u;
var<workgroup> sharedX: array<f32, 4096>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let tid = lid.x;
  let hidden = params.hidden;
  let qDim = params.qDim;
  let kDim = params.kDim;
  let vDim = params.vDim;
  let totalCols = qDim + kDim + vDim;

  let baseCol = wid.x * 1024u + tid * 4u;

  let c0Valid = baseCol < totalCols;
  let c1Valid = (baseCol + 1u) < totalCols;
  let c2Valid = (baseCol + 2u) < totalCols;
  let c3Valid = (baseCol + 3u) < totalCols;

  // Determine target matrix and local column for baseCol.
  // All 4 cols are in the same matrix (dims are multiples of headDim >= 64).
  var targetMatrix: u32 = 0u;
  var localCol: u32 = baseCol;
  var matDim: u32 = qDim;
  if (baseCol >= qDim + kDim) {
    targetMatrix = 2u;
    localCol = baseCol - qDim - kDim;
    matDim = vDim;
  } else if (baseCol >= qDim) {
    targetMatrix = 1u;
    localCol = baseCol - qDim;
    matDim = kDim;
  }

  var a0: f32 = 0.0; var a1: f32 = 0.0; var a2: f32 = 0.0; var a3: f32 = 0.0;

  let numTilesK = (hidden + SHARED_K - 1u) / SHARED_K;

  for (var tileK = 0u; tileK < numTilesK; tileK++) {
    let kStart = tileK * SHARED_K;
    let kEnd = min(kStart + SHARED_K, hidden);
    let tileLen = kEnd - kStart;

    for (var i = tid; i < tileLen; i += 256u) {
      sharedX[i] = x[kStart + i];
    }
    workgroupBarrier();

    if (c0Valid) {
      for (var kk = 0u; kk < tileLen; kk++) {
        let xVal = sharedX[kk];
        let wIdx = (kStart + kk) * matDim + localCol;

        if (targetMatrix == 0u) {
          a0 += xVal * Wq[wIdx];
          if (c1Valid) { a1 += xVal * Wq[wIdx + 1u]; }
          if (c2Valid) { a2 += xVal * Wq[wIdx + 2u]; }
          if (c3Valid) { a3 += xVal * Wq[wIdx + 3u]; }
        } else if (targetMatrix == 1u) {
          a0 += xVal * Wk[wIdx];
          if (c1Valid) { a1 += xVal * Wk[wIdx + 1u]; }
          if (c2Valid) { a2 += xVal * Wk[wIdx + 2u]; }
          if (c3Valid) { a3 += xVal * Wk[wIdx + 3u]; }
        } else {
          a0 += xVal * Wv[wIdx];
          if (c1Valid) { a1 += xVal * Wv[wIdx + 1u]; }
          if (c2Valid) { a2 += xVal * Wv[wIdx + 2u]; }
          if (c3Valid) { a3 += xVal * Wv[wIdx + 3u]; }
        }
      }
    }
    workgroupBarrier();
  }

  // Write to appropriate output buffer
  if (targetMatrix == 0u) {
    if (c0Valid) { Q[localCol] = a0; }
    if (c1Valid) { Q[localCol + 1u] = a1; }
    if (c2Valid) { Q[localCol + 2u] = a2; }
    if (c3Valid) { Q[localCol + 3u] = a3; }
  } else if (targetMatrix == 1u) {
    if (c0Valid) { K[localCol] = a0; }
    if (c1Valid) { K[localCol + 1u] = a1; }
    if (c2Valid) { K[localCol + 2u] = a2; }
    if (c3Valid) { K[localCol + 3u] = a3; }
  } else {
    if (c0Valid) { V[localCol] = a0; }
    if (c1Valid) { V[localCol + 1u] = a1; }
    if (c2Valid) { V[localCol + 2u] = a2; }
    if (c3Valid) { V[localCol + 3u] = a3; }
  }
}
`;

let fusedQKVGemvPipeline: GPUComputePipeline | null = null;

/**
 * Fused QKV projection: computes Q, K, V from input x in a single kernel
 * Reads input x only once instead of 3 times
 *
 * Uses GEMV variant when M=1 for single-token inference.
 *
 * @param x - Input tensor [M, hidden]
 * @param wQ - Q weight [hidden, qDim]
 * @param wK - K weight [hidden, kDim]
 * @param wV - V weight [hidden, vDim]
 * @returns Object containing Q, K, V tensors
 */
export async function fusedQKVProjection(
  x: Tensor,
  wQ: Tensor,
  wK: Tensor,
  wV: Tensor
): Promise<{ Q: Tensor; K: Tensor; V: Tensor }> {
  const M = x.shape[0];
  const hidden = x.shape[1];
  const qDim = wQ.shape[1];
  const kDim = wK.shape[1];
  const vDim = wV.shape[1];

  // GEMV fast path for M=1 (single-token inference)
  if (M === 1) {
    if (!fusedQKVGemvPipeline) {
      fusedQKVGemvPipeline = createComputePipelineFromSource(FUSED_QKV_GEMV_SHADER, {
        label: 'fused_qkv_gemv',
        entryPoint: 'main',
      });
    }

    const Q = Tensor.empty([1, qDim], { label: 'Q_gemv_output' });
    const K = Tensor.empty([1, kDim], { label: 'K_gemv_output' });
    const V = Tensor.empty([1, vDim], { label: 'V_gemv_output' });

    const params = createUniformBufferWithData(
      new Uint32Array([hidden, qDim, kDim, vDim]),
      'qkv_gemv_params'
    );

    const bindGroup = createBindGroup(fusedQKVGemvPipeline, 0, [
      { binding: 0, resource: x.getBuffer() },
      { binding: 1, resource: wQ.getBuffer() },
      { binding: 2, resource: wK.getBuffer() },
      { binding: 3, resource: wV.getBuffer() },
      { binding: 4, resource: Q.getBuffer() },
      { binding: 5, resource: K.getBuffer() },
      { binding: 6, resource: V.getBuffer() },
      { binding: 7, resource: params },
    ]);

    const totalCols = qDim + kDim + vDim;
    const wgX = Math.ceil(totalCols / 1024);

    const usedBuffers = [
      x.getBuffer(), wQ.getBuffer(), wK.getBuffer(), wV.getBuffer(),
      Q.getBuffer(), K.getBuffer(), V.getBuffer(), params
    ];
    await executeCompute(fusedQKVGemvPipeline, [bindGroup], [wgX, 1, 1], undefined, false, true, usedBuffers);

    requestBufferDestroy(params);
    return { Q, K, V };
  }

  // Tiled GEMM path for M > 1
  if (!fusedQKVPipeline) {
    fusedQKVPipeline = createComputePipelineFromSource(FUSED_QKV_SHADER, {
      label: 'fused_qkv',
      entryPoint: 'main',
    });
  }

  const Q = Tensor.empty([M, qDim], { label: 'Q_output' });
  const K = Tensor.empty([M, kDim], { label: 'K_output' });
  const V = Tensor.empty([M, vDim], { label: 'V_output' });

  // Pack params: M, hidden, qDim, kDim, vDim, pad, pad, pad
  const paramsData = new ArrayBuffer(32);
  const view = new DataView(paramsData);
  view.setUint32(0, M, true);
  view.setUint32(4, hidden, true);
  view.setUint32(8, qDim, true);
  view.setUint32(12, kDim, true);
  view.setUint32(16, vDim, true);
  view.setUint32(20, 0, true);
  view.setUint32(24, 0, true);
  view.setUint32(28, 0, true);

  const params = createUniformBufferWithData(new Uint8Array(paramsData), 'qkv_params');

  const bindGroup = createBindGroup(fusedQKVPipeline, 0, [
    { binding: 0, resource: x.getBuffer() },
    { binding: 1, resource: wQ.getBuffer() },
    { binding: 2, resource: wK.getBuffer() },
    { binding: 3, resource: wV.getBuffer() },
    { binding: 4, resource: Q.getBuffer() },
    { binding: 5, resource: K.getBuffer() },
    { binding: 6, resource: V.getBuffer() },
    { binding: 7, resource: params },
  ]);

  // Workgroups: X covers all output columns (Q+K+V), Y covers M
  const totalCols = qDim + kDim + vDim;
  const wgX = Math.ceil(totalCols / 256);
  const wgY = M;

  const usedBuffers = [
    x.getBuffer(), wQ.getBuffer(), wK.getBuffer(), wV.getBuffer(),
    Q.getBuffer(), K.getBuffer(), V.getBuffer(), params
  ];
  await executeCompute(fusedQKVPipeline, [bindGroup], [wgX, wgY, 1], undefined, false, true, usedBuffers);

  requestBufferDestroy(params);
  return { Q, K, V };
}

// ============================================================================
// GPU Top-K Sampling
// ============================================================================

/**
 * GPU shader for top-k selection with softmax
 *
 * Algorithm:
 * 1. Each workgroup processes a chunk of the vocabulary
 * 2. Finds local top-k using parallel reduction
 * 3. Outputs local top-k values and indices
 * 4. CPU does final merge of workgroup results (small data)
 */
const TOPK_SHADER = `
struct Params {
  vocabSize: u32,
  k: u32,
  temperature: f32,
  _pad: u32,
}

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> outValues: array<f32>;
@group(0) @binding(2) var<storage, read_write> outIndices: array<f32>;  // Using f32 so toArray() works correctly
@group(0) @binding(3) var<storage, read_write> outMax: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const WG_SIZE: u32 = 256u;
const LOCAL_K: u32 = 64u;  // Each workgroup finds top-64

var<workgroup> sharedVals: array<f32, 64>;
var<workgroup> sharedIdxs: array<f32, 64>;  // Store as f32 for easy readback
var<workgroup> sharedMax: array<f32, 256>;
var<workgroup> localK: u32;

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(num_workgroups) numWGs: vec3<u32>
) {
  let tid = lid.x;
  let wgIdx = wid.x;
  let vocabSize = params.vocabSize;
  let k = min(params.k, LOCAL_K);
  let temperature = params.temperature;

  // Each workgroup handles a chunk of vocabulary
  let chunkSize = (vocabSize + numWGs.x - 1u) / numWGs.x;
  let chunkStart = wgIdx * chunkSize;
  let chunkEnd = min(chunkStart + chunkSize, vocabSize);

  // Step 1: Each thread finds its local max and top values
  var threadMax: f32 = -1.0e38;
  var threadTopVals: array<f32, 4>;
  var threadTopIdxs: array<f32, 4>;

  // Initialize thread's top-4
  for (var i = 0u; i < 4u; i++) {
    threadTopVals[i] = -1.0e38;
    threadTopIdxs[i] = 0.0;
  }

  // Process elements assigned to this thread
  let elementsPerThread = (chunkEnd - chunkStart + WG_SIZE - 1u) / WG_SIZE;
  for (var e = 0u; e < elementsPerThread; e++) {
    let idx = chunkStart + tid + e * WG_SIZE;
    if (idx < chunkEnd) {
      let val = logits[idx];
      threadMax = max(threadMax, val);

      // Insert into thread's top-4 (simple insertion)
      if (val > threadTopVals[3]) {
        threadTopVals[3] = val;
        threadTopIdxs[3] = f32(idx);
        // Bubble up
        for (var j = 3u; j > 0u; j--) {
          if (threadTopVals[j] > threadTopVals[j-1u]) {
            let tmpV = threadTopVals[j-1u];
            let tmpI = threadTopIdxs[j-1u];
            threadTopVals[j-1u] = threadTopVals[j];
            threadTopIdxs[j-1u] = threadTopIdxs[j];
            threadTopVals[j] = tmpV;
            threadTopIdxs[j] = tmpI;
          }
        }
      }
    }
  }

  // Step 2: Reduce to find workgroup max
  sharedMax[tid] = threadMax;
  workgroupBarrier();

  for (var s = WG_SIZE / 2u; s > 0u; s >>= 1u) {
    if (tid < s) {
      sharedMax[tid] = max(sharedMax[tid], sharedMax[tid + s]);
    }
    workgroupBarrier();
  }

  let wgMax = sharedMax[0];

  // Step 3: Merge thread top-4s into workgroup top-k
  // Thread 0 initializes shared arrays
  if (tid == 0u) {
    localK = 0u;
    for (var i = 0u; i < LOCAL_K; i++) {
      sharedVals[i] = -1.0e38;
      sharedIdxs[i] = 0.0;
    }
  }
  workgroupBarrier();

  // Each thread contributes its top values (sequentially to avoid races)
  // This is O(threads * 4) but threads = 256, so 1024 iterations total - acceptable
  for (var t = 0u; t < WG_SIZE; t++) {
    if (tid == t) {
      for (var i = 0u; i < 4u; i++) {
        if (threadTopVals[i] > -1.0e37) {
          // Find insertion point
          var insertPos = LOCAL_K;
          for (var j = 0u; j < LOCAL_K; j++) {
            if (threadTopVals[i] > sharedVals[j]) {
              insertPos = j;
              break;
            }
          }

          // Insert if within top-k
          if (insertPos < k) {
            // Shift down
            for (var j = k - 1u; j > insertPos; j--) {
              sharedVals[j] = sharedVals[j - 1u];
              sharedIdxs[j] = sharedIdxs[j - 1u];
            }
            sharedVals[insertPos] = threadTopVals[i];
            sharedIdxs[insertPos] = threadTopIdxs[i];
          }
        }
      }
    }
    workgroupBarrier();
  }

  // Step 4: Output raw logits (softmax done on CPU after merging workgroups)
  // This allows correct merging across workgroups by comparing raw logit values

  // Step 5: Write results (raw logits, not probabilities)
  let outOffset = wgIdx * LOCAL_K;
  if (tid < k) {
    outValues[outOffset + tid] = sharedVals[tid];
    outIndices[outOffset + tid] = sharedIdxs[tid];
  } else if (tid < LOCAL_K) {
    outValues[outOffset + tid] = 0.0;
    outIndices[outOffset + tid] = 0.0;
  }

  // Write workgroup max (for merging)
  if (tid == 0u) {
    outMax[wgIdx] = wgMax;
  }
}
`;

let topKPipeline: GPUComputePipeline | null = null;

export interface TopKResult {
  probs: Float32Array;
  indices: Uint32Array;
}

/**
 * GPU-accelerated top-k selection with softmax
 *
 * @param logits - Logits tensor [1, vocabSize] or [vocabSize]
 * @param k - Number of top elements to return
 * @param temperature - Temperature for softmax
 * @returns Top-k probabilities and their indices
 */
export async function topKSoftmaxGPU(
  logits: Tensor,
  k: number,
  temperature: number
): Promise<TopKResult> {
  const vocabSize = logits.shape[logits.ndim - 1];

  if (!topKPipeline) {
    topKPipeline = createComputePipelineFromSource(TOPK_SHADER, {
      label: 'topk_softmax',
      entryPoint: 'main',
    });
  }

  // Limit k to 64 (LOCAL_K in shader)
  const effectiveK = Math.min(k, 64);

  // Use multiple workgroups for large vocabularies
  // Each workgroup finds local top-k, outputs raw logits (not softmax)
  // CPU merges by raw logit value, then applies softmax
  const numWorkgroups = Math.min(Math.ceil(vocabSize / 4096), 32);
  const localK = 64;

  // Create output buffers
  const outValuesBuffer = createStorageBuffer(numWorkgroups * localK * 4, 'topk_values');
  const outIndicesBuffer = createStorageBuffer(numWorkgroups * localK * 4, 'topk_indices');
  const outMaxBuffer = createStorageBuffer(numWorkgroups * 4, 'topk_max');

  // Create params
  const paramsData = new ArrayBuffer(16);
  const view = new DataView(paramsData);
  view.setUint32(0, vocabSize, true);
  view.setUint32(4, effectiveK, true);
  view.setFloat32(8, temperature, true);
  view.setUint32(12, 0, true);
  const params = createUniformBufferWithData(new Uint8Array(paramsData), 'topk_params');

  const bindGroup = createBindGroup(topKPipeline, 0, [
    { binding: 0, resource: logits.getBuffer() },
    { binding: 1, resource: outValuesBuffer },
    { binding: 2, resource: outIndicesBuffer },
    { binding: 3, resource: outMaxBuffer },
    { binding: 4, resource: params },
  ]);

  const usedBuffers = [logits.getBuffer(), outValuesBuffer, outIndicesBuffer, outMaxBuffer, params];
  await executeCompute(topKPipeline, [bindGroup], [numWorkgroups, 1, 1], undefined, false, true, usedBuffers);

  // Read results from GPU (small amount of data)
  const valuesResult = new Tensor([numWorkgroups * localK], outValuesBuffer, { label: 'topk_values_tensor' });
  const indicesResult = new Tensor([numWorkgroups * localK], outIndicesBuffer, { label: 'topk_indices_tensor' });

  const allValues = await valuesResult.toArray();
  const allIndicesFloat = await indicesResult.toArray();
  // Indices are stored as f32 in shader, convert to integers
  const allIndices = new Uint32Array(allIndicesFloat.length);
  for (let i = 0; i < allIndicesFloat.length; i++) {
    allIndices[i] = allIndicesFloat[i];  // f32 -> u32 truncation
  }

  valuesResult.destroy();
  indicesResult.destroy();
  requestBufferDestroy(params);

  // Collect all candidates from all workgroups (raw logit values)
  const candidates: Array<{logit: number, idx: number}> = [];
  for (let wg = 0; wg < numWorkgroups; wg++) {
    for (let i = 0; i < localK; i++) {
      const logit = allValues[wg * localK + i];
      const idx = allIndices[wg * localK + i];
      // Filter out padding (very negative values)
      if (logit > -1e30) {
        candidates.push({ logit, idx });
      }
    }
  }

  // Sort by raw logit value (descending) and take global top-k
  candidates.sort((a, b) => b.logit - a.logit);
  const topK = candidates.slice(0, effectiveK);

  // Apply temperature scaling and softmax on CPU
  // Find max for numerical stability
  let maxLogit = -Infinity;
  for (const item of topK) {
    if (item.logit > maxLogit) maxLogit = item.logit;
  }

  // Compute exp((logit - max) / temperature) and sum
  const expValues: number[] = [];
  let sumExp = 0;
  for (const item of topK) {
    const scaled = (item.logit - maxLogit) / temperature;
    const exp = Math.exp(scaled);
    expValues.push(exp);
    sumExp += exp;
  }

  // Normalize to probabilities
  const finalProbs = new Float32Array(effectiveK);
  const finalIndices = new Uint32Array(effectiveK);
  for (let i = 0; i < topK.length; i++) {
    finalProbs[i] = expValues[i] / sumExp;
    finalIndices[i] = topK[i].idx;
  }
  // Fill remaining slots with zeros if fewer candidates than k
  for (let i = topK.length; i < effectiveK; i++) {
    finalProbs[i] = 0;
    finalIndices[i] = 0;
  }

  return { probs: finalProbs, indices: finalIndices };
}

/**
 * Sample a token from top-k probabilities
 */
export function sampleFromTopK(probs: Float32Array, indices: Uint32Array): number {
  const rand = Math.random();
  let cumsum = 0;
  for (let i = 0; i < probs.length; i++) {
    cumsum += probs[i];
    if (rand < cumsum) {
      return indices[i];
    }
  }
  // Fallback to last element
  return indices[indices.length - 1];
}

/**
 * Reset top-k pipeline (for testing)
 */
export function resetTopKPipeline(): void {
  topKPipeline = null;
}

/**
 * Reset all cached pipelines in ops module
 * Call this when switching GPU devices (e.g., loading a new model)
 */
export function resetAllOpsPipelines(): void {
  elementwisePipelines = null;
  binaryPipelines = null;
  scalarPipelines = null;
  broadcastAddPipeline = null;
  matmulPipeline = null;
  gemvPipeline = null;
  reducePipelines = null;
  softmaxPipeline = null;
  embeddingPipeline = null;
  causalAttentionPipeline = null;
  sliceLastRowPipeline = null;
  sliceLastRowsPipeline = null;
  copyRowsPipeline = null;
  transposePipeline = null;
  fusedSwiGLUPipeline = null;
  fusedSwiGLUGemvPipeline = null;
  fusedQKVPipeline = null;
  fusedQKVGemvPipeline = null;
  topKPipeline = null;
}
