/**
 * QuantizedTensor - GPU-resident quantized weight storage
 *
 * Stores raw quantized bytes on GPU without dequantization.
 * Used for quantized matmul operations that dequantize on-the-fly.
 */

import { GGMLType } from '../../../types/model.js';
import { createStorageBufferWithData } from '../buffer.js';
import { getBlockSize, getBytesPerBlock } from './index.js';

/**
 * Supported quantization types for GPU-resident storage
 */
export type SupportedQuantType =
  | GGMLType.Q4_0
  | GGMLType.Q4_K
  | GGMLType.Q8_0;

/**
 * Options for creating a QuantizedTensor
 */
export interface QuantizedTensorOptions {
  label?: string;
}

/**
 * GPU-resident quantized tensor
 *
 * Unlike Tensor which stores f32 values, QuantizedTensor stores
 * raw quantized blocks that are dequantized on-the-fly during
 * compute operations.
 */
export class QuantizedTensor {
  /** Original shape in elements (e.g., [4096, 11008]) */
  readonly shape: readonly number[];

  /** Total number of f32 elements this represents */
  readonly numElements: number;

  /** Number of quantization blocks */
  readonly numBlocks: number;

  /** Quantization type */
  readonly quantType: SupportedQuantType;

  /** Block size (elements per block) */
  readonly blockSize: number;

  /** Bytes per block */
  readonly bytesPerBlock: number;

  /** Total bytes stored */
  readonly totalBytes: number;

  /** Debug label */
  readonly label: string;

  /** GPU buffer containing raw quantized data */
  private buffer: GPUBuffer;

  /** Whether this tensor has been destroyed */
  private destroyed = false;

  /**
   * Create a QuantizedTensor from raw quantized bytes
   *
   * @param data - Raw quantized bytes (will be copied to GPU)
   * @param shape - Logical shape in elements [rows, cols]
   * @param quantType - Quantization format
   * @param options - Additional options
   */
  constructor(
    data: Uint8Array,
    shape: number[],
    quantType: SupportedQuantType,
    options: QuantizedTensorOptions = {}
  ) {
    this.shape = Object.freeze([...shape]);
    this.numElements = shape.reduce((a, b) => a * b, 1);
    this.quantType = quantType;
    this.blockSize = getBlockSize(quantType);
    this.bytesPerBlock = getBytesPerBlock(quantType);
    this.numBlocks = Math.ceil(this.numElements / this.blockSize);
    this.totalBytes = data.length;
    this.label = options.label || 'quantized_tensor';

    // Validate data size
    const expectedBytes = this.numBlocks * this.bytesPerBlock;
    if (data.length < expectedBytes) {
      throw new Error(
        `QuantizedTensor: data size ${data.length} is less than expected ${expectedBytes} bytes ` +
        `(${this.numBlocks} blocks × ${this.bytesPerBlock} bytes/block)`
      );
    }

    // Align to 4 bytes for GPU buffer
    const alignedSize = Math.ceil(data.length / 4) * 4;
    const alignedData = new Uint8Array(alignedSize);
    alignedData.set(data);

    // Create GPU buffer with quantized data
    this.buffer = createStorageBufferWithData(alignedData, this.label);
  }

  /**
   * Get the underlying GPU buffer
   */
  getBuffer(): GPUBuffer {
    if (this.destroyed) {
      throw new Error('QuantizedTensor has been destroyed');
    }
    return this.buffer;
  }

  /**
   * Get number of dimensions
   */
  get ndim(): number {
    return this.shape.length;
  }

  /**
   * Check if this tensor has been destroyed
   */
  isDestroyed(): boolean {
    return this.destroyed;
  }

  /**
   * Destroy the tensor and release GPU memory
   */
  destroy(): void {
    if (!this.destroyed) {
      this.buffer.destroy();
      this.destroyed = true;
    }
  }

  /**
   * Get memory usage statistics
   */
  getMemoryStats(): {
    quantizedBytes: number;
    equivalentF32Bytes: number;
    compressionRatio: number;
  } {
    const equivalentF32Bytes = this.numElements * 4;
    return {
      quantizedBytes: this.totalBytes,
      equivalentF32Bytes,
      compressionRatio: equivalentF32Bytes / this.totalBytes,
    };
  }

  /**
   * Create a QuantizedTensor from raw bytes
   * Static factory method for consistency with Tensor.fromData()
   */
  static fromQuantizedData(
    data: Uint8Array,
    shape: number[],
    quantType: SupportedQuantType,
    options: QuantizedTensorOptions = {}
  ): QuantizedTensor {
    return new QuantizedTensor(data, shape, quantType, options);
  }

  /**
   * Check if a GGMLType is supported for quantized storage
   */
  static isSupported(type: GGMLType): type is SupportedQuantType {
    return (
      type === GGMLType.Q4_0 ||
      type === GGMLType.Q4_K ||
      type === GGMLType.Q8_0
    );
  }

  /**
   * Get human-readable quantization type name
   */
  static getTypeName(type: SupportedQuantType): string {
    switch (type) {
      case GGMLType.Q4_0: return 'Q4_0';
      case GGMLType.Q4_K: return 'Q4_K';
      case GGMLType.Q8_0: return 'Q8_0';
      default: return 'Unknown';
    }
  }
}
