/**
 * WebGPU Buffer Management
 * Utilities for creating and managing GPU buffers
 */

/// <reference types="@webgpu/types" />

import { getWebGPUDevice } from './device.js';
import { flushCommandBatcher } from './shader.js';

// Define GPUBufferUsage constants for Node.js compatibility
// These are the standard WebGPU buffer usage flags
const GPUBufferUsageFlags = {
  MAP_READ: 0x0001,
  MAP_WRITE: 0x0002,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  INDEX: 0x0010,
  VERTEX: 0x0020,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
  INDIRECT: 0x0100,
  QUERY_RESOLVE: 0x0200,
} as const;

// Define GPUMapMode constants for Node.js compatibility
const GPUMapModeFlags = {
  READ: 0x0001,
  WRITE: 0x0002,
} as const;

export type BufferUsage = 'storage' | 'uniform' | 'vertex' | 'index' | 'copy-src' | 'copy-dst';

export interface BufferOptions {
  label?: string;
  usage: BufferUsage[];
  mappedAtCreation?: boolean;
}

/**
 * Convert usage flags to GPUBufferUsageFlags
 */
function usageToFlags(usage: BufferUsage[]): GPUBufferUsageFlags {
  let flags = 0;
  for (const u of usage) {
    switch (u) {
      case 'storage':
        flags |= GPUBufferUsageFlags.STORAGE;
        break;
      case 'uniform':
        flags |= GPUBufferUsageFlags.UNIFORM;
        break;
      case 'vertex':
        flags |= GPUBufferUsageFlags.VERTEX;
        break;
      case 'index':
        flags |= GPUBufferUsageFlags.INDEX;
        break;
      case 'copy-src':
        flags |= GPUBufferUsageFlags.COPY_SRC;
        break;
      case 'copy-dst':
        flags |= GPUBufferUsageFlags.COPY_DST;
        break;
    }
  }
  return flags;
}

/**
 * Create a GPU buffer
 */
export function createBuffer(
  size: number,
  options: BufferOptions
): GPUBuffer {
  const device = getWebGPUDevice().getDevice();

  // Ensure size is aligned to 4 bytes
  const alignedSize = Math.ceil(size / 4) * 4;

  return device.createBuffer({
    label: options.label,
    size: alignedSize,
    usage: usageToFlags(options.usage),
    mappedAtCreation: options.mappedAtCreation ?? false,
  });
}

/**
 * Create a buffer and write data to it
 */
export function createBufferWithData(
  data: ArrayBufferView,
  options: BufferOptions
): GPUBuffer {
  const device = getWebGPUDevice().getDevice();

  // Ensure size is aligned to 4 bytes
  const alignedSize = Math.ceil(data.byteLength / 4) * 4;

  const buffer = device.createBuffer({
    label: options.label,
    size: alignedSize,
    usage: usageToFlags(options.usage),
    mappedAtCreation: true,
  });

  // Copy data
  const mapped = new Uint8Array(buffer.getMappedRange());
  mapped.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  buffer.unmap();

  return buffer;
}

/**
 * Create a storage buffer for compute shaders
 */
export function createStorageBuffer(
  size: number,
  label?: string
): GPUBuffer {
  return createBuffer(size, {
    label,
    usage: ['storage', 'copy-src', 'copy-dst'],
  });
}

/**
 * Create a storage buffer with initial data
 */
export function createStorageBufferWithData(
  data: ArrayBufferView,
  label?: string
): GPUBuffer {
  return createBufferWithData(data, {
    label,
    usage: ['storage', 'copy-src', 'copy-dst'],
  });
}

/**
 * Create a uniform buffer
 */
export function createUniformBuffer(
  size: number,
  label?: string
): GPUBuffer {
  return createBuffer(size, {
    label,
    usage: ['uniform', 'copy-dst'],
  });
}

/**
 * Create a uniform buffer with initial data
 */
export function createUniformBufferWithData(
  data: ArrayBufferView,
  label?: string
): GPUBuffer {
  return createBufferWithData(data, {
    label,
    usage: ['uniform', 'copy-dst'],
  });
}

/**
 * Create a staging buffer for reading data back from GPU
 */
export function createStagingBuffer(size: number, label?: string): GPUBuffer {
  const device = getWebGPUDevice().getDevice();

  // Ensure size is aligned to 4 bytes
  const alignedSize = Math.ceil(size / 4) * 4;

  return device.createBuffer({
    label,
    size: alignedSize,
    usage: GPUBufferUsageFlags.MAP_READ | GPUBufferUsageFlags.COPY_DST,
  });
}

/**
 * Write data to a buffer
 */
export function writeBuffer(
  buffer: GPUBuffer,
  data: ArrayBufferView,
  bufferOffset = 0
): void {
  const device = getWebGPUDevice().getDevice();
  device.queue.writeBuffer(buffer, bufferOffset, data);
}

/**
 * Read data from a GPU buffer
 */
export async function readBuffer(
  buffer: GPUBuffer,
  size?: number
): Promise<ArrayBuffer> {
  // Flush any pending batched commands before reading
  flushCommandBatcher();

  const gpuDevice = getWebGPUDevice();
  const device = gpuDevice.getDevice();

  const readSize = size ?? buffer.size;

  // Create staging buffer
  const stagingBuffer = createStagingBuffer(readSize, 'staging');

  // Copy to staging buffer
  const encoder = device.createCommandEncoder({ label: 'read buffer' });
  encoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, readSize);
  device.queue.submit([encoder.finish()]);

  // Wait for copy to complete
  await device.queue.onSubmittedWorkDone();

  // Map and read
  await stagingBuffer.mapAsync(GPUMapModeFlags.READ);
  const data = stagingBuffer.getMappedRange().slice(0);
  stagingBuffer.unmap();
  stagingBuffer.destroy();

  return data;
}

/**
 * Read buffer as Float32Array
 */
export async function readBufferFloat32(
  buffer: GPUBuffer,
  count?: number
): Promise<Float32Array> {
  const size = count ? count * 4 : undefined;
  const data = await readBuffer(buffer, size);
  return new Float32Array(data);
}

/**
 * Read buffer as Int32Array
 */
export async function readBufferInt32(
  buffer: GPUBuffer,
  count?: number
): Promise<Int32Array> {
  const size = count ? count * 4 : undefined;
  const data = await readBuffer(buffer, size);
  return new Int32Array(data);
}

/**
 * Read buffer as Uint32Array
 */
export async function readBufferUint32(
  buffer: GPUBuffer,
  count?: number
): Promise<Uint32Array> {
  const size = count ? count * 4 : undefined;
  const data = await readBuffer(buffer, size);
  return new Uint32Array(data);
}

/**
 * Copy between buffers
 */
export function copyBuffer(
  src: GPUBuffer,
  dst: GPUBuffer,
  size?: number,
  srcOffset = 0,
  dstOffset = 0
): GPUCommandBuffer {
  const device = getWebGPUDevice().getDevice();
  const copySize = size ?? Math.min(src.size - srcOffset, dst.size - dstOffset);

  const encoder = device.createCommandEncoder({ label: 'copy buffer' });
  encoder.copyBufferToBuffer(src, srcOffset, dst, dstOffset, copySize);
  return encoder.finish();
}

/**
 * Buffer pool for reusing buffers
 */
export class BufferPool {
  private pools: Map<string, GPUBuffer[]> = new Map();
  private inUse: Set<GPUBuffer> = new Set();

  /**
   * Get a buffer from the pool or create a new one
   */
  acquire(size: number, usage: BufferUsage[], label?: string): GPUBuffer {
    const key = `${size}-${usage.sort().join(',')}`;
    const pool = this.pools.get(key) || [];

    let buffer = pool.pop();
    if (!buffer) {
      buffer = createBuffer(size, { usage, label });
    }

    this.inUse.add(buffer);
    return buffer;
  }

  /**
   * Return a buffer to the pool
   */
  release(buffer: GPUBuffer): void {
    if (!this.inUse.has(buffer)) {
      return;
    }

    this.inUse.delete(buffer);
    const key = `${buffer.size}-storage,copy-src,copy-dst`; // Simplified key
    const pool = this.pools.get(key) || [];
    pool.push(buffer);
    this.pools.set(key, pool);
  }

  /**
   * Clear the pool and destroy all buffers
   */
  clear(): void {
    for (const pool of this.pools.values()) {
      for (const buffer of pool) {
        buffer.destroy();
      }
    }
    this.pools.clear();

    for (const buffer of this.inUse) {
      buffer.destroy();
    }
    this.inUse.clear();
  }
}

// Global buffer pool instance
let globalBufferPool: BufferPool | null = null;

/**
 * Get the global buffer pool
 */
export function getBufferPool(): BufferPool {
  if (!globalBufferPool) {
    globalBufferPool = new BufferPool();
  }
  return globalBufferPool;
}
