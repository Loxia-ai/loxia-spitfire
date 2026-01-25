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
  await executeCompute(pipeline, [bindGroup], [workgroups, 1, 1]);

  params.destroy();
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
  await executeCompute(pipeline, [bindGroup], [workgroups, 1, 1]);

  params.destroy();
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
  await executeCompute(pipeline, [bindGroup], [workgroups, 1, 1]);

  params.destroy();
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
// Matrix Multiplication
// ============================================================================

const MATMUL_SHADER = `
// Tiled matrix multiplication for better cache performance
// A: [M, K], B: [K, N], C: [M, N]

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

const TILE_SIZE: u32 = 8u;

var<workgroup> tileA: array<array<f32, 8>, 8>;
var<workgroup> tileB: array<array<f32, 8>, 8>;

@compute @workgroup_size(8, 8)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let row = gid.y;
  let col = gid.x;
  let localRow = lid.y;
  let localCol = lid.x;

  let M = params.M;
  let N = params.N;
  let K = params.K;

  var sum: f32 = 0.0;

  let numTiles = (K + TILE_SIZE - 1u) / TILE_SIZE;

  for (var t = 0u; t < numTiles; t++) {
    // Load tile from A
    let aRow = row;
    let aCol = t * TILE_SIZE + localCol;
    if (aRow < M && aCol < K) {
      tileA[localRow][localCol] = A[aRow * K + aCol];
    } else {
      tileA[localRow][localCol] = 0.0;
    }

    // Load tile from B
    let bRow = t * TILE_SIZE + localRow;
    let bCol = col;
    if (bRow < K && bCol < N) {
      tileB[localRow][localCol] = B[bRow * N + bCol];
    } else {
      tileB[localRow][localCol] = 0.0;
    }

    workgroupBarrier();

    // Compute partial sum
    for (var k = 0u; k < TILE_SIZE; k++) {
      sum += tileA[localRow][k] * tileB[k][localCol];
    }

    workgroupBarrier();
  }

  // Write result
  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}
`;

let matmulPipeline: GPUComputePipeline | null = null;

/**
 * Matrix multiplication: C = A @ B
 * A: [M, K], B: [K, N] -> C: [M, N]
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

  const K = K1;

  if (!matmulPipeline) {
    matmulPipeline = createComputePipelineFromSource(MATMUL_SHADER, {
      label: 'matmul',
      entryPoint: 'main',
    });
  }

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

  const [wgX, wgY] = calculateWorkgroups2D(N, M, 8, 8);
  await executeCompute(matmulPipeline, [bindGroup], [wgX, wgY, 1]);

  params.destroy();
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

    await executeCompute(pipeline, [bindGroup], [workgroups, 1, 1]);

    params.destroy();

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

  await executeCompute(softmaxPipeline, [bindGroup], [batchSize, 1, 1]);

  params.destroy();
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
  reducePipelines = null;
  softmaxPipeline = null;
  embeddingPipeline = null;
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

  // Debug: print embedding lookup params
  if (seqLen <= 82) {
    console.log(`[EmbLookup] shape=[${dim0}, ${dim1}], vocabSize=${vocabSize}, hiddenSize=${hiddenSize}, transposed=${transposed}`);
    console.log(`[EmbLookup] First 5 token IDs: [${indices.slice(0, 5).join(', ')}]`);
    console.log(`[EmbLookup] Token 0 (${indices[0]}) index range: ${indices[0] * hiddenSize} to ${indices[0] * hiddenSize + hiddenSize - 1}`);
  }

  const bindGroup = createBindGroup(embeddingPipeline, 0, [
    { binding: 0, resource: embeddings.getBuffer() },
    { binding: 1, resource: indicesBuffer },
    { binding: 2, resource: output.getBuffer() },
    { binding: 3, resource: params },
  ]);

  const totalElements = seqLen * hiddenSize;
  const workgroups = calculateWorkgroups(totalElements, 256);
  await executeCompute(embeddingPipeline, [bindGroup], [workgroups, 1, 1]);

  // Debug: verify the output by reading back
  if (seqLen <= 82) {
    const outData = await output.toArray();
    console.log(`[EmbLookup] Output first 8: [${Array.from(outData.slice(0, 8)).map(v => v.toFixed(4)).join(', ')}]`);
    // Also verify by reading embedding directly
    const embData = await embeddings.toArray();
    const tokenId = indices[0];
    const directLookup = embData.slice(tokenId * hiddenSize, tokenId * hiddenSize + 8);
    console.log(`[EmbLookup] Direct lookup token ${tokenId} first 8: [${Array.from(directLookup).map(v => v.toFixed(4)).join(', ')}]`);
  }

  params.destroy();
  indicesBuffer.destroy();

  return output;
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
  await executeCompute(transposePipeline, [bindGroup], [wgX, wgY, 1]);

  params.destroy();
  return output;
}
