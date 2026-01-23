/**
 * WebGPU Tensor Implementation
 * GPU-backed tensor class for efficient numerical computations
 */

/// <reference types="@webgpu/types" />

import {
  createStorageBuffer,
  createStorageBufferWithData,
  readBufferFloat32,
  writeBuffer,
  getBufferPool,
} from './buffer.js';
import { getWebGPUDevice } from './device.js';

export type TensorDType = 'f32' | 'f16' | 'i32' | 'u32';

export interface TensorOptions {
  dtype?: TensorDType;
  label?: string;
  pooled?: boolean; // Use buffer pooling
}

/**
 * Compute the total number of elements from shape
 */
export function shapeToSize(shape: number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

/**
 * Compute strides for a contiguous tensor
 */
export function shapeToStrides(shape: number[]): number[] {
  const strides = new Array(shape.length);
  let stride = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

/**
 * Get the byte size for a dtype
 */
function dtypeToBytes(dtype: TensorDType): number {
  switch (dtype) {
    case 'f32':
    case 'i32':
    case 'u32':
      return 4;
    case 'f16':
      return 2;
  }
}

/**
 * GPU Tensor class
 */
export class Tensor {
  readonly shape: number[];
  readonly strides: number[];
  readonly size: number;
  readonly dtype: TensorDType;
  readonly label: string;

  private buffer: GPUBuffer;
  private readonly pooled: boolean;
  private destroyed = false;

  /**
   * Create a tensor with the given shape and buffer
   */
  constructor(
    shape: number[],
    buffer: GPUBuffer,
    options: TensorOptions = {}
  ) {
    this.shape = [...shape];
    this.strides = shapeToStrides(shape);
    this.size = shapeToSize(shape);
    this.dtype = options.dtype || 'f32';
    this.label = options.label || 'tensor';
    this.buffer = buffer;
    this.pooled = options.pooled || false;
  }

  /**
   * Get the number of dimensions
   */
  get ndim(): number {
    return this.shape.length;
  }

  /**
   * Get the total byte size
   */
  get byteSize(): number {
    return this.size * dtypeToBytes(this.dtype);
  }

  /**
   * Get the underlying GPU buffer
   */
  getBuffer(): GPUBuffer {
    if (this.destroyed) {
      throw new Error('Tensor has been destroyed');
    }
    return this.buffer;
  }

  /**
   * Read tensor data to CPU
   */
  async toArray(): Promise<Float32Array> {
    if (this.destroyed) {
      throw new Error('Tensor has been destroyed');
    }
    return readBufferFloat32(this.buffer, this.size);
  }

  /**
   * Read tensor data as a nested array (for easier debugging)
   */
  async toNestedArray(): Promise<number[] | number[][]> {
    const flat = await this.toArray();

    if (this.ndim === 1) {
      return Array.from(flat);
    }

    if (this.ndim === 2) {
      const result: number[][] = [];
      const [rows, cols] = this.shape;
      for (let i = 0; i < rows; i++) {
        result.push(Array.from(flat.slice(i * cols, (i + 1) * cols)));
      }
      return result;
    }

    // For higher dimensions, just return flat
    return Array.from(flat);
  }

  /**
   * Write data to tensor from CPU
   */
  async fromArray(data: Float32Array | number[]): Promise<void> {
    if (this.destroyed) {
      throw new Error('Tensor has been destroyed');
    }
    const arr = data instanceof Float32Array ? data : new Float32Array(data);
    if (arr.length !== this.size) {
      throw new Error(`Data size ${arr.length} doesn't match tensor size ${this.size}`);
    }
    writeBuffer(this.buffer, arr);
    await getWebGPUDevice().sync();
  }

  /**
   * Create a view with a different shape (same underlying buffer)
   */
  reshape(newShape: number[]): Tensor {
    const newSize = shapeToSize(newShape);
    if (newSize !== this.size) {
      throw new Error(`Cannot reshape tensor of size ${this.size} to shape [${newShape}] (size ${newSize})`);
    }
    return new Tensor(newShape, this.buffer, {
      dtype: this.dtype,
      label: `${this.label}.reshape`,
    });
  }

  /**
   * Release GPU memory
   */
  destroy(): void {
    if (this.destroyed) return;

    if (this.pooled) {
      getBufferPool().release(this.buffer);
    } else {
      this.buffer.destroy();
    }
    this.destroyed = true;
  }

  /**
   * Check if tensor is destroyed
   */
  isDestroyed(): boolean {
    return this.destroyed;
  }

  // Static factory methods

  /**
   * Create an empty tensor with given shape
   */
  static empty(shape: number[], options: TensorOptions = {}): Tensor {
    const size = shapeToSize(shape);
    const bytes = size * dtypeToBytes(options.dtype || 'f32');
    const buffer = createStorageBuffer(bytes, options.label);
    return new Tensor(shape, buffer, options);
  }

  /**
   * Create a tensor from data
   */
  static fromData(
    data: Float32Array | number[],
    shape: number[],
    options: TensorOptions = {}
  ): Tensor {
    const arr = data instanceof Float32Array ? data : new Float32Array(data);
    const size = shapeToSize(shape);

    if (arr.length !== size) {
      throw new Error(`Data size ${arr.length} doesn't match shape size ${size}`);
    }

    const buffer = createStorageBufferWithData(arr, options.label);
    return new Tensor(shape, buffer, options);
  }

  /**
   * Create a tensor filled with zeros
   */
  static zeros(shape: number[], options: TensorOptions = {}): Tensor {
    const size = shapeToSize(shape);
    const data = new Float32Array(size);
    return Tensor.fromData(data, shape, { ...options, label: options.label || 'zeros' });
  }

  /**
   * Create a tensor filled with ones
   */
  static ones(shape: number[], options: TensorOptions = {}): Tensor {
    const size = shapeToSize(shape);
    const data = new Float32Array(size).fill(1);
    return Tensor.fromData(data, shape, { ...options, label: options.label || 'ones' });
  }

  /**
   * Create a tensor with random values in [0, 1)
   */
  static random(shape: number[], options: TensorOptions = {}): Tensor {
    const size = shapeToSize(shape);
    const data = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      data[i] = Math.random();
    }
    return Tensor.fromData(data, shape, { ...options, label: options.label || 'random' });
  }

  /**
   * Create a tensor with values from a range
   */
  static arange(start: number, end: number, step = 1, options: TensorOptions = {}): Tensor {
    const size = Math.ceil((end - start) / step);
    const data = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      data[i] = start + i * step;
    }
    return Tensor.fromData(data, [size], { ...options, label: options.label || 'arange' });
  }

  /**
   * Create an identity matrix
   */
  static eye(n: number, options: TensorOptions = {}): Tensor {
    const data = new Float32Array(n * n);
    for (let i = 0; i < n; i++) {
      data[i * n + i] = 1;
    }
    return Tensor.fromData(data, [n, n], { ...options, label: options.label || 'eye' });
  }
}

/**
 * Utility to run multiple operations and clean up intermediate tensors
 */
export function withTensors<T>(
  tensors: Tensor[],
  fn: () => T
): T {
  try {
    return fn();
  } finally {
    for (const t of tensors) {
      if (!t.isDestroyed()) {
        t.destroy();
      }
    }
  }
}
