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
  | GGMLType.Q6_K
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
      type === GGMLType.Q6_K ||
      type === GGMLType.Q8_0
    );
  }

  /**
   * Quantize f32 data to Q8_0 format and create a GPU-resident QuantizedTensor.
   *
   * Q8_0 format: 32 values per block, 34 bytes/block
   *   - 2 bytes: f16 scale (max_abs / 127)
   *   - 32 bytes: int8 quantized values
   *
   * @param f32Data - Float32Array of weights to quantize
   * @param shape - Logical shape [K, N] of the weight matrix
   * @param options - Additional options (label, etc.)
   */
  static quantizeFromF32(
    f32Data: Float32Array,
    shape: number[],
    options: QuantizedTensorOptions = {}
  ): QuantizedTensor {
    const blockSize = 32;
    const bytesPerBlock = 34; // 2 (f16 scale) + 32 (int8 values)
    const numBlocks = Math.ceil(f32Data.length / blockSize);
    const result = new Uint8Array(numBlocks * bytesPerBlock);

    for (let b = 0; b < numBlocks; b++) {
      const blockStart = b * blockSize;

      // Find max absolute value for scale
      let maxAbs = 0;
      for (let i = 0; i < blockSize; i++) {
        const idx = blockStart + i;
        if (idx < f32Data.length) {
          const abs = Math.abs(f32Data[idx]);
          if (abs > maxAbs) maxAbs = abs;
        }
      }

      const scale = maxAbs / 127;
      const invScale = scale > 0 ? 127 / maxAbs : 0;

      // Convert scale to f16
      const scaleF16 = QuantizedTensor._floatToHalf(scale);

      // Write scale (little-endian f16)
      const outBase = b * bytesPerBlock;
      result[outBase] = scaleF16 & 0xff;
      result[outBase + 1] = (scaleF16 >> 8) & 0xff;

      // Quantize values to int8
      for (let i = 0; i < blockSize; i++) {
        const srcIdx = blockStart + i;
        const value = srcIdx < f32Data.length ? f32Data[srcIdx] : 0;
        let quantized = Math.round(value * invScale);
        quantized = Math.max(-128, Math.min(127, quantized));
        result[outBase + 2 + i] = quantized & 0xff;
      }
    }

    return new QuantizedTensor(result, shape, GGMLType.Q8_0 as SupportedQuantType, options);
  }

  /**
   * Convert f32 to f16 (IEEE 754 half-precision float).
   */
  private static _floatToHalf(value: number): number {
    const floatView = new Float32Array(1);
    const int32View = new Int32Array(floatView.buffer);
    floatView[0] = value;
    const f = int32View[0];

    const sign = (f >> 16) & 0x8000;
    let exp = ((f >> 23) & 0xff) - 127 + 15;
    const mant = (f >> 13) & 0x3ff;

    if (exp <= 0) {
      if (exp < -10) return sign;
      const m = (mant | 0x400) >> (1 - exp);
      return sign | m;
    } else if (exp === 0xff - 127 + 15) {
      if (mant) return sign | 0x7c00 | mant;
      return sign | 0x7c00;
    } else if (exp > 30) {
      return sign | 0x7c00;
    }

    return sign | (exp << 10) | mant;
  }

  /**
   * Get human-readable quantization type name
   */
  static getTypeName(type: SupportedQuantType): string {
    switch (type) {
      case GGMLType.Q4_0: return 'Q4_0';
      case GGMLType.Q4_K: return 'Q4_K';
      case GGMLType.Q6_K: return 'Q6_K';
      case GGMLType.Q8_0: return 'Q8_0';
      default: return 'Unknown';
    }
  }
}
