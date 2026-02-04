/**
 * WebGPU Shader Management
 * Handles WGSL shader compilation, caching, and compute pipeline creation
 */

/// <reference types="@webgpu/types" />

import { getWebGPUDevice } from './device.js';
import { getGPUProfiler } from './perf/gpu-profiler.js';

export interface ComputePipelineOptions {
  label?: string;
  entryPoint?: string;
  constants?: Record<string, number>;
}

export interface BindGroupEntry {
  binding: number;
  resource: GPUBuffer | GPUSampler | GPUTextureView;
}

/**
 * Shader cache for compiled shader modules
 */
const shaderCache = new Map<string, GPUShaderModule>();

/**
 * Reverse map from shader module to source string (for pipeline cache key)
 */
const moduleToSource = new WeakMap<GPUShaderModule, string>();

/**
 * Pipeline cache for compute pipelines
 */
const pipelineCache = new Map<string, GPUComputePipeline>();

/**
 * Command Batcher for reducing WebGPU/Dawn overhead
 * Batches multiple command buffers and submits them together in a single queue.submit() call
 * Based on TensorFlow.js research: batching ~15 command buffers provides 5%+ improvement
 *
 * Key insight from TensorFlow.js: track buffers owned by pending commands and defer their
 * destruction until after submit. Destroying a buffer while it's in use causes commands to fail.
 */
class CommandBatcher {
  private pendingCommandBuffers: GPUCommandBuffer[] = [];
  private readonly BATCH_SIZE = 15; // TensorFlow.js empirical value
  private totalBatches = 0;
  private totalPasses = 0;

  // Buffers that were requested to be destroyed while commands are pending
  private pendingDisposal: GPUBuffer[] = [];

  // Buffers from the PREVIOUS batch - safe to destroy since GPU has finished with them
  // GPU processes batches sequentially, so by the time we flush again, the previous batch is done
  private previousBatchDisposal: GPUBuffer[] = [];

  /**
   * Add a command buffer to the pending batch
   * Note: usedBuffers parameter kept for API compatibility but no longer tracked individually
   */
  addCommandBuffer(commandBuffer: GPUCommandBuffer, _usedBuffers: GPUBuffer[]): void {
    this.pendingCommandBuffers.push(commandBuffer);
    this.totalPasses++;

    // Auto-flush when batch is full
    if (this.pendingCommandBuffers.length >= this.BATCH_SIZE) {
      this.flush();
    }
  }

  /**
   * Request buffer destruction - defers if there are pending commands
   * Simplified: if ANY commands are pending, defer ALL destructions
   * Returns true if destruction was deferred, false if destroyed immediately
   */
  requestDestroy(buffer: GPUBuffer): boolean {
    if (this.pendingCommandBuffers.length > 0) {
      // Commands are pending - defer destruction
      this.pendingDisposal.push(buffer);
      return true;
    }
    // No pending commands - can destroy immediately
    return false;
  }

  /**
   * Flush all pending command buffers to the GPU
   *
   * Buffer destruction is deferred by one batch to ensure GPU has finished reading.
   * When we flush batch N, we destroy buffers from batch N-1 (which is guaranteed
   * complete since GPU processes sequentially).
   */
  flush(): void {
    // First, destroy buffers from the PREVIOUS batch (GPU has finished with them)
    for (const buffer of this.previousBatchDisposal) {
      buffer.destroy();
    }
    this.previousBatchDisposal = [];

    if (this.pendingCommandBuffers.length > 0) {
      const gpuDevice = getWebGPUDevice();
      // Submit all command buffers in a single call - this is the key optimization
      gpuDevice.submit(this.pendingCommandBuffers);
      this.totalBatches++;
      this.pendingCommandBuffers = [];

      // Move current pending disposal to previous batch disposal
      // These will be destroyed on the NEXT flush, after GPU finishes this batch
      this.previousBatchDisposal = this.pendingDisposal;
      this.pendingDisposal = [];
    } else if (this.pendingDisposal.length > 0) {
      // No commands submitted, but we have buffers to dispose
      // Move them to previous batch disposal for next flush
      this.previousBatchDisposal = this.pendingDisposal;
      this.pendingDisposal = [];
    }
  }

  /**
   * Flush and wait for GPU to complete all work
   * After sync, all buffers can be safely destroyed
   */
  async flushAndSync(): Promise<void> {
    // Submit any pending commands
    if (this.pendingCommandBuffers.length > 0) {
      const gpuDevice = getWebGPUDevice();
      gpuDevice.submit(this.pendingCommandBuffers);
      this.totalBatches++;
      this.pendingCommandBuffers = [];
    }

    // Wait for GPU to finish all work
    await getWebGPUDevice().sync();

    // Now safe to destroy ALL pending buffers since GPU is done
    for (const buffer of this.previousBatchDisposal) {
      buffer.destroy();
    }
    this.previousBatchDisposal = [];

    for (const buffer of this.pendingDisposal) {
      buffer.destroy();
    }
    this.pendingDisposal = [];
  }

  /**
   * Get statistics about batching
   */
  getStats(): { totalBatches: number; totalPasses: number; avgPassesPerBatch: number } {
    return {
      totalBatches: this.totalBatches,
      totalPasses: this.totalPasses,
      avgPassesPerBatch: this.totalBatches > 0 ? this.totalPasses / this.totalBatches : 0,
    };
  }

  /**
   * Reset statistics
   */
  resetStats(): void {
    this.totalBatches = 0;
    this.totalPasses = 0;
  }

  /**
   * Check if there are pending command buffers
   */
  hasPending(): boolean {
    return this.pendingCommandBuffers.length > 0;
  }
}

/**
 * Global command batcher instance
 */
let commandBatcher: CommandBatcher | null = null;

/**
 * Get the global command batcher (creates if needed)
 */
export function getCommandBatcher(): CommandBatcher {
  if (!commandBatcher) {
    commandBatcher = new CommandBatcher();
  }
  return commandBatcher;
}

/**
 * Flush the global command batcher
 */
export function flushCommandBatcher(): void {
  if (commandBatcher) {
    commandBatcher.flush();
  }
}

/**
 * Flush the global command batcher and sync with GPU
 */
export async function flushAndSyncCommandBatcher(): Promise<void> {
  if (commandBatcher) {
    await commandBatcher.flushAndSync();
  }
}

/**
 * Get command batcher statistics
 */
export function getCommandBatcherStats(): { totalBatches: number; totalPasses: number; avgPassesPerBatch: number } | null {
  return commandBatcher ? commandBatcher.getStats() : null;
}

/**
 * Reset command batcher statistics
 */
export function resetCommandBatcherStats(): void {
  if (commandBatcher) {
    commandBatcher.resetStats();
  }
}

/**
 * Request buffer destruction - defers if buffer is owned by pending commands
 * Based on TensorFlow.js approach: track owned buffers and defer disposal
 */
export function requestBufferDestroy(buffer: GPUBuffer): void {
  const batcher = getCommandBatcher();
  if (!batcher.requestDestroy(buffer)) {
    // Buffer not owned by pending commands - destroy immediately
    buffer.destroy();
  }
  // If owned, destruction is deferred until after flush
}

/**
 * Create a shader module from WGSL source
 */
export function createShaderModule(
  source: string,
  label?: string
): GPUShaderModule {
  // Check cache first
  const cacheKey = source;
  const cached = shaderCache.get(cacheKey);
  if (cached) {
    return cached;
  }

  const device = getWebGPUDevice().getDevice();
  const module = device.createShaderModule({
    label,
    code: source,
  });

  // Store reverse mapping for pipeline cache key generation
  moduleToSource.set(module, source);

  // Check for compilation errors asynchronously
  module.getCompilationInfo().then((info) => {
    for (const message of info.messages) {
      const type = message.type;
      const text = message.message;
      const line = message.lineNum;
      const col = message.linePos;
      if (type === 'error') {
        console.error(`WGSL Error in "${label || 'shader'}" at line ${line}:${col}: ${text}`);
      } else if (type === 'warning') {
        console.warn(`WGSL Warning in "${label || 'shader'}" at line ${line}:${col}: ${text}`);
      }
    }
  }).catch(() => {
    // Ignore if getCompilationInfo not supported
  });

  shaderCache.set(cacheKey, module);
  return module;
}

/**
 * Create a compute pipeline from a shader module
 */
export function createComputePipeline(
  shaderModule: GPUShaderModule,
  options: ComputePipelineOptions = {}
): GPUComputePipeline {
  const { label, entryPoint = 'main', constants } = options;

  // Create cache key using shader source (not label) for uniqueness
  const shaderSource = moduleToSource.get(shaderModule) || shaderModule.label || 'unknown';
  const cacheKey = `${shaderSource}-${entryPoint}-${JSON.stringify(constants || {})}`;
  const cached = pipelineCache.get(cacheKey);
  if (cached) {
    return cached;
  }

  const device = getWebGPUDevice().getDevice();

  const pipeline = device.createComputePipeline({
    label,
    layout: 'auto',
    compute: {
      module: shaderModule,
      entryPoint,
      constants,
    },
  });

  pipelineCache.set(cacheKey, pipeline);
  return pipeline;
}

/**
 * Create a compute pipeline directly from WGSL source
 */
export function createComputePipelineFromSource(
  source: string,
  options: ComputePipelineOptions = {}
): GPUComputePipeline {
  const module = createShaderModule(source, options.label);
  return createComputePipeline(module, options);
}

/**
 * Create a bind group for a compute pipeline
 */
export function createBindGroup(
  pipeline: GPUComputePipeline,
  groupIndex: number,
  entries: BindGroupEntry[],
  label?: string
): GPUBindGroup {
  const device = getWebGPUDevice().getDevice();
  const layout = pipeline.getBindGroupLayout(groupIndex);

  return device.createBindGroup({
    label,
    layout,
    entries: entries.map((entry) => ({
      binding: entry.binding,
      resource: isBuffer(entry.resource)
        ? { buffer: entry.resource }
        : entry.resource,
    })),
  });
}

/**
 * Check if a resource is a GPUBuffer
 */
function isBuffer(resource: GPUBuffer | GPUSampler | GPUTextureView): resource is GPUBuffer {
  return 'size' in resource && 'usage' in resource;
}

/**
 * Dispatch a compute shader
 */
export function dispatchCompute(
  pipeline: GPUComputePipeline,
  bindGroups: GPUBindGroup[],
  workgroupCounts: [number, number, number],
  label?: string
): GPUCommandBuffer {
  const device = getWebGPUDevice().getDevice();

  const encoder = device.createCommandEncoder({ label });

  // Inject GPU timestamp writes if profiler is active
  const profiler = getGPUProfiler();
  const opLabel = label || pipeline.label || 'unnamed';
  const timestampWrites = profiler.isEnabled() ? profiler.beginOperation(opLabel) : null;

  const passDescriptor: GPUComputePassDescriptor = { label };
  if (timestampWrites) {
    passDescriptor.timestampWrites = timestampWrites;
  }

  const pass = encoder.beginComputePass(passDescriptor);

  pass.setPipeline(pipeline);
  for (let i = 0; i < bindGroups.length; i++) {
    pass.setBindGroup(i, bindGroups[i]);
  }

  pass.dispatchWorkgroups(
    workgroupCounts[0],
    workgroupCounts[1],
    workgroupCounts[2]
  );

  pass.end();
  return encoder.finish();
}

/**
 * Execute a compute shader
 * By default, uses command buffer batching to reduce queue.submit() overhead
 * Command buffers are batched and submitted together (TensorFlow.js approach)
 * Set sync=true only when you need to read results immediately
 * Set batched=false to submit immediately (legacy behavior)
 *
 * @param usedBuffers - Array of GPUBuffers used by this operation (for tracking disposal)
 */
export async function executeCompute(
  pipeline: GPUComputePipeline,
  bindGroups: GPUBindGroup[],
  workgroupCounts: [number, number, number],
  label?: string,
  sync = false,
  batched = true,
  usedBuffers: GPUBuffer[] = []
): Promise<void> {
  // Create the command buffer (same as before)
  const commandBuffer = dispatchCompute(pipeline, bindGroups, workgroupCounts, label);

  if (batched && !sync) {
    // Add to batch with buffer tracking
    getCommandBatcher().addCommandBuffer(commandBuffer, usedBuffers);
  } else {
    // Direct execution - submit immediately
    const gpuDevice = getWebGPUDevice();
    gpuDevice.submit([commandBuffer]);
    // Only sync when explicitly requested (e.g., before reading buffer)
    if (sync) {
      await gpuDevice.sync();
    }
  }
}

/**
 * Clear shader and pipeline caches
 */
export function clearShaderCache(): void {
  shaderCache.clear();
  pipelineCache.clear();
}

/**
 * Common WGSL utility functions that can be included in shaders
 */
export const WGSL_UTILS = `
// Activation functions
fn relu(x: f32) -> f32 {
  return max(x, 0.0);
}

fn gelu(x: f32) -> f32 {
  // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
  let c = 0.7978845608; // sqrt(2/pi)
  let inner = c * (x + 0.044715 * x * x * x);
  return 0.5 * x * (1.0 + tanh(inner));
}

fn silu(x: f32) -> f32 {
  // SiLU (Swish): x * sigmoid(x)
  return x / (1.0 + exp(-x));
}

fn quick_gelu(x: f32) -> f32 {
  // QuickGELU: x * sigmoid(1.702 * x)
  return x / (1.0 + exp(-1.702 * x));
}

// Stable softmax helper
fn safe_exp(x: f32) -> f32 {
  return exp(clamp(x, -88.0, 88.0));
}
`;

/**
 * Generate a simple element-wise operation shader
 */
export function generateElementwiseShader(
  operation: string,
  workgroupSize = 256
): string {
  return `
${WGSL_UTILS}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: vec4<u32>;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  let size = params.x;

  if (idx >= size) {
    return;
  }

  let x = input[idx];
  output[idx] = ${operation};
}
`;
}

/**
 * Generate a binary operation shader (element-wise on two arrays)
 */
export function generateBinaryShader(
  operation: string,
  workgroupSize = 256
): string {
  return `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  let size = params.x;

  if (idx >= size) {
    return;
  }

  let x = a[idx];
  let y = b[idx];
  output[idx] = ${operation};
}
`;
}

/**
 * Calculate workgroup counts for a given size
 */
export function calculateWorkgroups(
  totalSize: number,
  workgroupSize: number
): number {
  return Math.ceil(totalSize / workgroupSize);
}

/**
 * Calculate 2D workgroup counts
 */
export function calculateWorkgroups2D(
  width: number,
  height: number,
  workgroupSizeX: number,
  workgroupSizeY: number
): [number, number] {
  return [
    Math.ceil(width / workgroupSizeX),
    Math.ceil(height / workgroupSizeY),
  ];
}

/**
 * Calculate 3D workgroup counts
 */
export function calculateWorkgroups3D(
  width: number,
  height: number,
  depth: number,
  workgroupSizeX: number,
  workgroupSizeY: number,
  workgroupSizeZ: number
): [number, number, number] {
  return [
    Math.ceil(width / workgroupSizeX),
    Math.ceil(height / workgroupSizeY),
    Math.ceil(depth / workgroupSizeZ),
  ];
}
