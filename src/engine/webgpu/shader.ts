/**
 * WebGPU Shader Management
 * Handles WGSL shader compilation, caching, and compute pipeline creation
 */

/// <reference types="@webgpu/types" />

import { getWebGPUDevice } from './device.js';

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
 * Pipeline cache for compute pipelines
 */
const pipelineCache = new Map<string, GPUComputePipeline>();

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

  // Create cache key
  const cacheKey = `${shaderModule.label || 'shader'}-${entryPoint}-${JSON.stringify(constants || {})}`;
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
  const pass = encoder.beginComputePass({ label });

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
 * Execute a compute shader and wait for completion
 */
export async function executeCompute(
  pipeline: GPUComputePipeline,
  bindGroups: GPUBindGroup[],
  workgroupCounts: [number, number, number],
  label?: string
): Promise<void> {
  const gpuDevice = getWebGPUDevice();
  const commandBuffer = dispatchCompute(pipeline, bindGroups, workgroupCounts, label);
  gpuDevice.submit([commandBuffer]);
  await gpuDevice.sync();
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
