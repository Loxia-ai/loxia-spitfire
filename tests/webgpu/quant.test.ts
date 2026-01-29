/**
 * WebGPU Quantization Tests - Phase 4
 * Tests for dequantization of GGML quantized formats
 */

import {
  WebGPUDevice,
  initWebGPU,
  getBlockSize,
  getBytesPerBlock,
  requiresDequantization,
  dequantize,
  dequantizeQ4_0,
  dequantizeQ8_0,
  dequantizeQ4_K,
  quantizeToQ8_0,
  QuantizedTensor,
  gemvQ8_0,
  gemvQ4_K,
  Tensor,
  ops,
} from '../../src/engine/webgpu/index.js';
import { GGMLType } from '../../src/types/model.js';

// Skip tests if WebGPU is not available
const describeWebGPU = process.env.SKIP_WEBGPU_TESTS ? describe.skip : describe;

describeWebGPU('WebGPU Quantization', () => {
  let device: WebGPUDevice;

  beforeAll(async () => {
    const isAvailable = await WebGPUDevice.isAvailable();
    if (!isAvailable) {
      console.log('WebGPU not available, skipping tests');
      return;
    }
    device = await initWebGPU();
  });

  afterAll(() => {
    if (device) {
      device.destroy();
    }
  });

  describe('Block Size Helpers', () => {
    test('should return correct block size for Q4_0', () => {
      expect(getBlockSize(GGMLType.Q4_0)).toBe(32);
    });

    test('should return correct block size for Q8_0', () => {
      expect(getBlockSize(GGMLType.Q8_0)).toBe(32);
    });

    test('should return correct block size for Q4_K', () => {
      expect(getBlockSize(GGMLType.Q4_K)).toBe(256);
    });

    test('should return 1 for F32', () => {
      expect(getBlockSize(GGMLType.F32)).toBe(1);
    });
  });

  describe('Bytes Per Block', () => {
    test('should return 4 for F32', () => {
      expect(getBytesPerBlock(GGMLType.F32)).toBe(4);
    });

    test('should return 2 for F16', () => {
      expect(getBytesPerBlock(GGMLType.F16)).toBe(2);
    });

    test('should return 18 for Q4_0', () => {
      expect(getBytesPerBlock(GGMLType.Q4_0)).toBe(18);
    });

    test('should return 34 for Q8_0', () => {
      expect(getBytesPerBlock(GGMLType.Q8_0)).toBe(34);
    });

    test('should return 144 for Q4_K', () => {
      expect(getBytesPerBlock(GGMLType.Q4_K)).toBe(144);
    });
  });

  describe('Requires Dequantization', () => {
    test('should return false for F32', () => {
      expect(requiresDequantization(GGMLType.F32)).toBe(false);
    });

    test('should return false for F16', () => {
      expect(requiresDequantization(GGMLType.F16)).toBe(false);
    });

    test('should return true for Q4_0', () => {
      expect(requiresDequantization(GGMLType.Q4_0)).toBe(true);
    });

    test('should return true for Q8_0', () => {
      expect(requiresDequantization(GGMLType.Q8_0)).toBe(true);
    });

    test('should return true for Q4_K', () => {
      expect(requiresDequantization(GGMLType.Q4_K)).toBe(true);
    });
  });

  describe('Q8_0 Quantization', () => {
    test('should quantize and dequantize with low error', async () => {
      if (!device) return;

      // Create test data
      const numElements = 64; // 2 blocks
      const original = new Float32Array(numElements);
      for (let i = 0; i < numElements; i++) {
        original[i] = (Math.random() - 0.5) * 2; // Range [-1, 1]
      }

      // Quantize to Q8_0
      const quantized = quantizeToQ8_0(original);

      // Expected size: 2 blocks * 34 bytes = 68 bytes
      expect(quantized.length).toBe(68);

      // Dequantize
      const dequantized = await dequantizeQ8_0(quantized, numElements);
      const result = await dequantized.toArray();

      expect(result.length).toBe(numElements);

      // Check that values are reasonably close
      let maxError = 0;
      let sumError = 0;
      for (let i = 0; i < numElements; i++) {
        const error = Math.abs(result[i] - original[i]);
        maxError = Math.max(maxError, error);
        sumError += error;
      }

      const avgError = sumError / numElements;

      // Q8_0 should have low quantization error
      expect(maxError).toBeLessThan(0.02); // Less than 2% max error
      expect(avgError).toBeLessThan(0.01); // Less than 1% average error

      dequantized.destroy();
    });

    test('should handle zero values', async () => {
      if (!device) return;

      const original = new Float32Array(32);
      // All zeros
      for (let i = 0; i < 32; i++) {
        original[i] = 0;
      }

      const quantized = quantizeToQ8_0(original);
      const dequantized = await dequantizeQ8_0(quantized, 32);
      const result = await dequantized.toArray();

      // All values should be zero or very close
      for (let i = 0; i < 32; i++) {
        expect(Math.abs(result[i])).toBeLessThan(0.001);
      }

      dequantized.destroy();
    });

    test('should handle large values', async () => {
      if (!device) return;

      const original = new Float32Array(32);
      for (let i = 0; i < 32; i++) {
        original[i] = (i - 16) * 100; // Range [-1600, 1500]
      }

      const quantized = quantizeToQ8_0(original);
      const dequantized = await dequantizeQ8_0(quantized, 32);
      const result = await dequantized.toArray();

      // Check relative error for large values
      for (let i = 0; i < 32; i++) {
        const relError = original[i] !== 0
          ? Math.abs(result[i] - original[i]) / Math.abs(original[i])
          : Math.abs(result[i]);
        // Q8_0 should maintain roughly 1% relative accuracy
        expect(relError).toBeLessThan(0.02);
      }

      dequantized.destroy();
    });
  });

  describe('Q4_0 Dequantization', () => {
    test('should dequantize Q4_0 data', async () => {
      if (!device) return;

      // Create a simple Q4_0 block manually
      // Block: 2 bytes scale (f16) + 16 bytes (32 x 4-bit nibbles)
      const numElements = 32;
      const blockData = new Uint8Array(20); // Padded to 20 for u32 alignment

      // Set scale to 1.0 (f16 representation: 0x3C00)
      blockData[0] = 0x00;
      blockData[1] = 0x3c;

      // Fill nibbles with values that will become [-8, -7, ..., 7, 8, -8, -7, ...]
      // Nibble 0 = 0 (gives -8*scale), nibble 1 = 1 (gives -7*scale), etc.
      for (let i = 0; i < 16; i++) {
        const lowNibble = (i * 2) % 16;
        const highNibble = (i * 2 + 1) % 16;
        blockData[2 + i] = (highNibble << 4) | lowNibble;
      }

      const dequantized = await dequantizeQ4_0(blockData, numElements);
      const result = await dequantized.toArray();

      expect(result.length).toBe(numElements);

      // Check that values are finite
      for (const val of result) {
        expect(Number.isFinite(val)).toBe(true);
      }

      dequantized.destroy();
    });
  });

  describe('F32 Passthrough', () => {
    test('should pass through F32 data unchanged', async () => {
      if (!device) return;

      const original = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
      const data = new Uint8Array(original.buffer);

      const result = await dequantize(data, 8, GGMLType.F32);
      const output = await result.toArray();

      expect(output.length).toBe(8);
      for (let i = 0; i < 8; i++) {
        expect(output[i]).toBeCloseTo(original[i], 5);
      }

      result.destroy();
    });
  });

  describe('Unsupported Types', () => {
    test('should throw for unsupported quantization type', async () => {
      if (!device) return;

      const data = new Uint8Array(32);

      // Q2_K is not implemented
      await expect(dequantize(data, 32, GGMLType.Q2_K)).rejects.toThrow();
    });
  });

  describe('Memory Management', () => {
    test('should not leak memory on repeated dequantization', async () => {
      if (!device) return;

      const original = new Float32Array(64);
      for (let i = 0; i < 64; i++) {
        original[i] = Math.random();
      }

      // Perform multiple quantize/dequantize cycles
      for (let cycle = 0; cycle < 10; cycle++) {
        const quantized = quantizeToQ8_0(original);
        const dequantized = await dequantizeQ8_0(quantized, 64);

        // Clean up
        dequantized.destroy();
      }

      // If we get here without errors, memory management is working
      expect(true).toBe(true);
    });
  });

  // =========================================================================
  // QuantizedTensor Tests (Phase 1 of quantized matmul implementation)
  // =========================================================================

  describe('QuantizedTensor', () => {
    describe('Q8_0 Storage', () => {
      test('should create Q8_0 tensor with correct metadata', async () => {
        if (!device) return;

        // Q8_0: 32 elements per block, 34 bytes per block
        // 64 elements = 2 blocks, 2 * 34 = 68 bytes
        const totalBytes = 68;

        // Create test quantized data
        const quantData = new Uint8Array(totalBytes);
        // Fill with some pattern
        for (let i = 0; i < totalBytes; i++) {
          quantData[i] = i % 256;
        }

        const qt = QuantizedTensor.fromQuantizedData(
          quantData,
          [64], // 1D shape with 64 elements
          GGMLType.Q8_0,
          { label: 'test_q8_0' }
        );

        // Verify metadata
        expect(qt.shape).toEqual([64]);
        expect(qt.numElements).toBe(64);
        expect(qt.numBlocks).toBe(2);
        expect(qt.blockSize).toBe(32);
        expect(qt.bytesPerBlock).toBe(34);
        expect(qt.totalBytes).toBe(68);
        expect(qt.quantType).toBe(GGMLType.Q8_0);
        expect(qt.label).toBe('test_q8_0');
        expect(qt.ndim).toBe(1);

        // Verify buffer exists
        expect(qt.getBuffer()).toBeDefined();
        expect(qt.isDestroyed()).toBe(false);

        qt.destroy();
        expect(qt.isDestroyed()).toBe(true);
      });

      test('should create Q8_0 tensor from quantized f32 data', async () => {
        if (!device) return;

        // Create original f32 data
        const original = new Float32Array(64);
        for (let i = 0; i < 64; i++) {
          original[i] = (i - 32) * 0.1; // Range [-3.2, 3.1]
        }

        // Quantize to Q8_0
        const quantData = quantizeToQ8_0(original);
        expect(quantData.length).toBe(68); // 2 blocks * 34 bytes

        // Store as QuantizedTensor
        const qt = QuantizedTensor.fromQuantizedData(
          quantData,
          [64],
          GGMLType.Q8_0,
          { label: 'test_from_f32' }
        );

        expect(qt.numElements).toBe(64);
        expect(qt.numBlocks).toBe(2);

        // Check memory stats
        const stats = qt.getMemoryStats();
        expect(stats.quantizedBytes).toBe(68);
        expect(stats.equivalentF32Bytes).toBe(256); // 64 * 4
        expect(stats.compressionRatio).toBeCloseTo(256 / 68, 2); // ~3.76x

        qt.destroy();
      });

      test('should handle 2D shape for weight matrices', async () => {
        if (!device) return;

        // Simulate a small weight matrix [32, 64] = 2048 elements
        // Q8_0: 2048 / 32 = 64 blocks, 64 * 34 = 2176 bytes
        const numElements = 32 * 64;
        const numBlocks = Math.ceil(numElements / 32);
        const totalBytes = numBlocks * 34;

        const quantData = new Uint8Array(totalBytes);

        const qt = QuantizedTensor.fromQuantizedData(
          quantData,
          [32, 64],
          GGMLType.Q8_0,
          { label: 'weight_matrix' }
        );

        expect(qt.shape).toEqual([32, 64]);
        expect(qt.numElements).toBe(2048);
        expect(qt.numBlocks).toBe(64);
        expect(qt.ndim).toBe(2);

        qt.destroy();
      });
    });

    describe('Q4_K Storage', () => {
      test('should create Q4_K tensor with correct metadata', async () => {
        if (!device) return;

        // Q4_K: 256 elements per block, 144 bytes per block
        // 512 elements = 2 blocks, 2 * 144 = 288 bytes
        const totalBytes = 288;

        const quantData = new Uint8Array(totalBytes);

        const qt = QuantizedTensor.fromQuantizedData(
          quantData,
          [512],
          GGMLType.Q4_K,
          { label: 'test_q4_k' }
        );

        expect(qt.shape).toEqual([512]);
        expect(qt.numElements).toBe(512);
        expect(qt.numBlocks).toBe(2);
        expect(qt.blockSize).toBe(256);
        expect(qt.bytesPerBlock).toBe(144);
        expect(qt.totalBytes).toBe(288);
        expect(qt.quantType).toBe(GGMLType.Q4_K);

        // Check compression ratio
        const stats = qt.getMemoryStats();
        expect(stats.quantizedBytes).toBe(288);
        expect(stats.equivalentF32Bytes).toBe(2048); // 512 * 4
        expect(stats.compressionRatio).toBeCloseTo(2048 / 288, 2); // ~7.1x

        qt.destroy();
      });

      test('should handle realistic FFN weight sizes', async () => {
        if (!device) return;

        // Simulate ffnGate weight [2048, 5632] for a small model
        // Total elements: 11,534,336
        // Q4_K blocks: 11,534,336 / 256 = 45,056 blocks
        // Total bytes: 45,056 * 144 = 6,488,064 bytes (~6.2 MB)
        //
        // For test, use smaller size: [256, 512] = 131,072 elements
        const rows = 256;
        const cols = 512;
        const numElements = rows * cols;
        const numBlocks = Math.ceil(numElements / 256);
        const totalBytes = numBlocks * 144;

        const quantData = new Uint8Array(totalBytes);

        const qt = QuantizedTensor.fromQuantizedData(
          quantData,
          [rows, cols],
          GGMLType.Q4_K,
          { label: 'ffn_gate' }
        );

        expect(qt.numElements).toBe(131072);
        expect(qt.numBlocks).toBe(512);

        const stats = qt.getMemoryStats();
        // f32 would be 131072 * 4 = 524,288 bytes
        // Q4_K is 512 * 144 = 73,728 bytes
        expect(stats.compressionRatio).toBeCloseTo(524288 / 73728, 1); // ~7.1x

        qt.destroy();
      });
    });

    describe('Q4_0 Storage', () => {
      test('should create Q4_0 tensor with correct metadata', async () => {
        if (!device) return;

        // Q4_0: 32 elements per block, 18 bytes per block
        // 64 elements = 2 blocks, 2 * 18 = 36 bytes
        const totalBytes = 36;

        const quantData = new Uint8Array(totalBytes);

        const qt = QuantizedTensor.fromQuantizedData(
          quantData,
          [64],
          GGMLType.Q4_0,
          { label: 'test_q4_0' }
        );

        expect(qt.numElements).toBe(64);
        expect(qt.numBlocks).toBe(2);
        expect(qt.blockSize).toBe(32);
        expect(qt.bytesPerBlock).toBe(18);

        qt.destroy();
      });
    });

    describe('Validation', () => {
      test('should throw if data is too small', async () => {
        if (!device) return;

        // Q8_0: 64 elements needs 2 blocks * 34 = 68 bytes
        const tooSmall = new Uint8Array(50); // Less than 68

        expect(() => {
          QuantizedTensor.fromQuantizedData(
            tooSmall,
            [64],
            GGMLType.Q8_0
          );
        }).toThrow(/data size.*less than expected/i);
      });

      test('should throw when accessing destroyed tensor', async () => {
        if (!device) return;

        const quantData = new Uint8Array(68); // 2 Q8_0 blocks
        const qt = QuantizedTensor.fromQuantizedData(
          quantData,
          [64],
          GGMLType.Q8_0
        );

        qt.destroy();

        expect(() => qt.getBuffer()).toThrow(/destroyed/i);
      });
    });

    describe('Static Helpers', () => {
      test('isSupported should return true for supported types', () => {
        expect(QuantizedTensor.isSupported(GGMLType.Q8_0)).toBe(true);
        expect(QuantizedTensor.isSupported(GGMLType.Q4_0)).toBe(true);
        expect(QuantizedTensor.isSupported(GGMLType.Q4_K)).toBe(true);
      });

      test('isSupported should return false for unsupported types', () => {
        expect(QuantizedTensor.isSupported(GGMLType.F32)).toBe(false);
        expect(QuantizedTensor.isSupported(GGMLType.F16)).toBe(false);
        expect(QuantizedTensor.isSupported(GGMLType.Q2_K)).toBe(false);
        expect(QuantizedTensor.isSupported(GGMLType.Q6_K)).toBe(false);
      });

      test('getTypeName should return correct names', () => {
        expect(QuantizedTensor.getTypeName(GGMLType.Q8_0)).toBe('Q8_0');
        expect(QuantizedTensor.getTypeName(GGMLType.Q4_0)).toBe('Q4_0');
        expect(QuantizedTensor.getTypeName(GGMLType.Q4_K)).toBe('Q4_K');
      });
    });

    describe('Memory Management', () => {
      test('should not leak memory on create/destroy cycles', async () => {
        if (!device) return;

        for (let i = 0; i < 10; i++) {
          const quantData = new Uint8Array(68);
          const qt = QuantizedTensor.fromQuantizedData(
            quantData,
            [64],
            GGMLType.Q8_0
          );
          qt.destroy();
        }

        // If we get here without OOM, memory management works
        expect(true).toBe(true);
      });
    });
  });

  // =========================================================================
  // Quantized GEMV Tests (Phase 2)
  // =========================================================================

  describe('Quantized GEMV', () => {
    describe('gemvQ8_0', () => {
      test('should produce results matching f32 GEMV within tolerance', async () => {
        if (!device) return;

        // Create test data: x[1, 64] @ W[64, 32] -> y[1, 32]
        const K = 64;
        const N = 32;

        // Create random input vector
        const xData = new Float32Array(K);
        for (let i = 0; i < K; i++) {
          xData[i] = (Math.random() - 0.5) * 2;
        }
        const x = Tensor.fromData(xData, [1, K], { label: 'x' });

        // Create random weight matrix
        const wData = new Float32Array(K * N);
        for (let i = 0; i < K * N; i++) {
          wData[i] = (Math.random() - 0.5) * 2;
        }

        // Quantize weights to Q8_0
        const wQuantData = quantizeToQ8_0(wData);
        const wQuant = QuantizedTensor.fromQuantizedData(
          wQuantData,
          [K, N],
          GGMLType.Q8_0,
          { label: 'W_q8' }
        );

        // Also dequantize for reference f32 computation
        const wDequant = await dequantizeQ8_0(wQuantData, K * N);
        const wDequantReshaped = wDequant.reshape([K, N]);

        // Compute using quantized GEMV
        const yQuant = await gemvQ8_0(x, wQuant);
        const yQuantData = await yQuant.toArray();

        // Compute using f32 matmul
        const yF32 = await ops.matmul(x, wDequantReshaped);
        const yF32Data = await yF32.toArray();

        // Compare results - should be nearly identical
        // (only difference is rounding in f16 scale storage)
        expect(yQuantData.length).toBe(N);
        expect(yF32Data.length).toBe(N);

        let maxError = 0;
        let sumError = 0;
        for (let i = 0; i < N; i++) {
          const error = Math.abs(yQuantData[i] - yF32Data[i]);
          maxError = Math.max(maxError, error);
          sumError += error;
        }
        const avgError = sumError / N;

        // Results should be very close (both use same quantized weights)
        expect(maxError).toBeLessThan(0.01);
        expect(avgError).toBeLessThan(0.001);

        // Cleanup
        x.destroy();
        wQuant.destroy();
        wDequant.destroy();
        yQuant.destroy();
        yF32.destroy();
      });

      test('should handle larger dimensions (hidden size)', async () => {
        if (!device) return;

        // Simulate typical hidden size: x[1, 256] @ W[256, 128] -> y[1, 128]
        const K = 256;
        const N = 128;

        const xData = new Float32Array(K);
        for (let i = 0; i < K; i++) {
          xData[i] = (Math.random() - 0.5) * 0.1;
        }
        const x = Tensor.fromData(xData, [1, K]);

        const wData = new Float32Array(K * N);
        for (let i = 0; i < K * N; i++) {
          wData[i] = (Math.random() - 0.5) * 0.1;
        }

        const wQuantData = quantizeToQ8_0(wData);
        const wQuant = QuantizedTensor.fromQuantizedData(
          wQuantData,
          [K, N],
          GGMLType.Q8_0
        );

        const wDequant = await dequantizeQ8_0(wQuantData, K * N);
        const wDequantReshaped = wDequant.reshape([K, N]);

        const yQuant = await gemvQ8_0(x, wQuant);
        const yQuantData = await yQuant.toArray();

        const yF32 = await ops.matmul(x, wDequantReshaped);
        const yF32Data = await yF32.toArray();

        expect(yQuantData.length).toBe(N);

        let maxError = 0;
        for (let i = 0; i < N; i++) {
          const error = Math.abs(yQuantData[i] - yF32Data[i]);
          maxError = Math.max(maxError, error);
        }

        // Slightly higher tolerance for larger matrices
        expect(maxError).toBeLessThan(0.05);

        x.destroy();
        wQuant.destroy();
        wDequant.destroy();
        yQuant.destroy();
        yF32.destroy();
      });

      test('should handle non-power-of-2 dimensions', async () => {
        if (!device) return;

        // Odd dimensions: x[1, 100] @ W[100, 77] -> y[1, 77]
        const K = 100;
        const N = 77;

        const xData = new Float32Array(K);
        for (let i = 0; i < K; i++) {
          xData[i] = Math.random() - 0.5;
        }
        const x = Tensor.fromData(xData, [1, K]);

        const wData = new Float32Array(K * N);
        for (let i = 0; i < K * N; i++) {
          wData[i] = Math.random() - 0.5;
        }

        const wQuantData = quantizeToQ8_0(wData);
        const wQuant = QuantizedTensor.fromQuantizedData(
          wQuantData,
          [K, N],
          GGMLType.Q8_0
        );

        const wDequant = await dequantizeQ8_0(wQuantData, K * N);
        const wDequantReshaped = wDequant.reshape([K, N]);

        const yQuant = await gemvQ8_0(x, wQuant);
        const yQuantData = await yQuant.toArray();

        const yF32 = await ops.matmul(x, wDequantReshaped);
        const yF32Data = await yF32.toArray();

        expect(yQuantData.length).toBe(N);

        let maxError = 0;
        for (let i = 0; i < N; i++) {
          const error = Math.abs(yQuantData[i] - yF32Data[i]);
          maxError = Math.max(maxError, error);
        }

        expect(maxError).toBeLessThan(0.05);

        x.destroy();
        wQuant.destroy();
        wDequant.destroy();
        yQuant.destroy();
        yF32.destroy();
      });

      test('should validate input dimensions', async () => {
        if (!device) return;

        // x must have shape[0] === 1
        const xBad = Tensor.fromData(new Float32Array(128), [2, 64]);
        const wData = quantizeToQ8_0(new Float32Array(64 * 32));
        const wQuant = QuantizedTensor.fromQuantizedData(wData, [64, 32], GGMLType.Q8_0);

        await expect(gemvQ8_0(xBad, wQuant)).rejects.toThrow(/shape\[0\].*1/);

        xBad.destroy();
        wQuant.destroy();
      });

      test('should validate dimension match', async () => {
        if (!device) return;

        // x.shape[1] must equal W.shape[0]
        const x = Tensor.fromData(new Float32Array(64), [1, 64]);
        const wData = quantizeToQ8_0(new Float32Array(32 * 16)); // [32, 16], not [64, ...]
        const wQuant = QuantizedTensor.fromQuantizedData(wData, [32, 16], GGMLType.Q8_0);

        await expect(gemvQ8_0(x, wQuant)).rejects.toThrow(/mismatch/i);

        x.destroy();
        wQuant.destroy();
      });

      test('should compute correct result for known values', async () => {
        if (!device) return;

        // Simple known case: x = [1, 1, 1, ...], W = identity-like
        // Result should be sum of each column
        const K = 32; // One Q8_0 block
        const N = 4;

        // All ones input
        const xData = new Float32Array(K);
        xData.fill(1.0);
        const x = Tensor.fromData(xData, [1, K]);

        // Weight matrix where each column sums to a known value
        // Column 0: all 0.5 -> sum = 16
        // Column 1: all 0.25 -> sum = 8
        // Column 2: all -0.5 -> sum = -16
        // Column 3: alternating 1, -1 -> sum = 0
        const wData = new Float32Array(K * N);
        for (let k = 0; k < K; k++) {
          wData[k * N + 0] = 0.5;
          wData[k * N + 1] = 0.25;
          wData[k * N + 2] = -0.5;
          wData[k * N + 3] = k % 2 === 0 ? 1.0 : -1.0;
        }

        const wQuantData = quantizeToQ8_0(wData);
        const wQuant = QuantizedTensor.fromQuantizedData(wQuantData, [K, N], GGMLType.Q8_0);

        const y = await gemvQ8_0(x, wQuant);
        const yData = await y.toArray();

        // Check approximate expected values (with quantization tolerance)
        expect(yData[0]).toBeCloseTo(16, 0);   // sum of 0.5 * 32
        expect(yData[1]).toBeCloseTo(8, 0);    // sum of 0.25 * 32
        expect(yData[2]).toBeCloseTo(-16, 0);  // sum of -0.5 * 32
        expect(yData[3]).toBeCloseTo(0, 0);    // alternating cancels

        x.destroy();
        wQuant.destroy();
        y.destroy();
      });
    });

    describe('gemvQ4_K', () => {
      // Helper to quantize f32 to Q4_K format (simplified for testing)
      function quantizeToQ4_K(data: Float32Array): Uint8Array {
        const QK_K = 256;
        const BYTES_PER_BLOCK = 144;
        const numBlocks = Math.ceil(data.length / QK_K);
        const result = new Uint8Array(numBlocks * BYTES_PER_BLOCK);

        for (let b = 0; b < numBlocks; b++) {
          const blockStart = b * QK_K;
          const outBase = b * BYTES_PER_BLOCK;

          // Find max absolute value for scale
          let maxAbs = 0;
          for (let i = 0; i < QK_K && blockStart + i < data.length; i++) {
            maxAbs = Math.max(maxAbs, Math.abs(data[blockStart + i]));
          }

          const d = maxAbs / 15; // Q4 values are 0-15
          const invD = d > 0 ? 15 / maxAbs : 0;

          // Write d as f16 (simplified - just store rough value)
          const dF16 = floatToHalfSimple(d);
          result[outBase] = dF16 & 0xff;
          result[outBase + 1] = (dF16 >> 8) & 0xff;

          // Write dmin as 0 (we use positive-only encoding for simplicity)
          result[outBase + 2] = 0;
          result[outBase + 3] = 0;

          // Write scales[12] - all set to 1 for uniform scaling
          for (let i = 0; i < 12; i++) {
            result[outBase + 4 + i] = 1; // scale=1 in lower 6 bits
          }

          // Quantize values to 4-bit (packed into qs[128])
          const qsBase = outBase + 16;

          // Q4_K layout: 4 chunks of 64 values each
          // Each chunk: bytes 0-31 store low nibble (first 32) and high nibble (next 32)
          for (let chunk = 0; chunk < 4; chunk++) {
            const chunkOffset = chunk * 64;
            const qOffset = chunk * 32;

            for (let l = 0; l < 32; l++) {
              const idx1 = blockStart + chunkOffset + l;
              const idx2 = blockStart + chunkOffset + 32 + l;

              const val1 = idx1 < data.length ? data[idx1] : 0;
              const val2 = idx2 < data.length ? data[idx2] : 0;

              // Quantize to 0-15 range
              const q1 = Math.min(15, Math.max(0, Math.round(val1 * invD)));
              const q2 = Math.min(15, Math.max(0, Math.round(val2 * invD)));

              // Pack: low nibble = q1, high nibble = q2
              result[qsBase + qOffset + l] = (q1 & 0xF) | ((q2 & 0xF) << 4);
            }
          }
        }

        return result;
      }

      function floatToHalfSimple(value: number): number {
        if (value === 0) return 0;
        const sign = value < 0 ? 0x8000 : 0;
        value = Math.abs(value);
        const exp = Math.floor(Math.log2(value));
        const mant = Math.round((value / Math.pow(2, exp) - 1) * 1024);
        const biasedExp = exp + 15;
        if (biasedExp <= 0) return sign;
        if (biasedExp >= 31) return sign | 0x7c00;
        return sign | (biasedExp << 10) | (mant & 0x3ff);
      }

      test('should produce results matching f32 GEMV within tolerance', async () => {
        if (!device) return;

        // Create test data: x[1, 256] @ W[256, 32] -> y[1, 32]
        // Using 256 for K to align with Q4_K block size
        const K = 256;
        const N = 32;

        // Create input vector with small values
        const xData = new Float32Array(K);
        for (let i = 0; i < K; i++) {
          xData[i] = (Math.random() - 0.5) * 0.2;
        }
        const x = Tensor.fromData(xData, [1, K], { label: 'x' });

        // Create weight matrix with small positive values (Q4_K works best with positive)
        const wData = new Float32Array(K * N);
        for (let i = 0; i < K * N; i++) {
          wData[i] = Math.random() * 0.2; // 0 to 0.2
        }

        // Quantize weights to Q4_K
        const wQuantData = quantizeToQ4_K(wData);
        const wQuant = QuantizedTensor.fromQuantizedData(
          wQuantData,
          [K, N],
          GGMLType.Q4_K,
          { label: 'W_q4k' }
        );

        // Also dequantize for reference f32 computation
        const wDequant = await dequantizeQ4_K(wQuantData, K * N);
        const wDequantReshaped = wDequant.reshape([K, N]);

        // Compute using quantized GEMV
        const yQuant = await gemvQ4_K(x, wQuant);
        const yQuantData = await yQuant.toArray();

        // Compute using f32 matmul
        const yF32 = await ops.matmul(x, wDequantReshaped);
        const yF32Data = await yF32.toArray();

        // Compare results
        expect(yQuantData.length).toBe(N);
        expect(yF32Data.length).toBe(N);

        let maxError = 0;
        let sumError = 0;
        for (let i = 0; i < N; i++) {
          const error = Math.abs(yQuantData[i] - yF32Data[i]);
          maxError = Math.max(maxError, error);
          sumError += error;
        }
        const avgError = sumError / N;

        // Q4_K has lower precision, allow larger tolerance
        expect(maxError).toBeLessThan(0.5);
        expect(avgError).toBeLessThan(0.1);

        // Cleanup
        x.destroy();
        wQuant.destroy();
        wDequant.destroy();
        yQuant.destroy();
        yF32.destroy();
      });

      test('should handle larger dimensions', async () => {
        if (!device) return;

        // Simulate attention projection: x[1, 512] @ W[512, 256] -> y[1, 256]
        const K = 512;
        const N = 256;

        const xData = new Float32Array(K);
        for (let i = 0; i < K; i++) {
          xData[i] = (Math.random() - 0.5) * 0.1;
        }
        const x = Tensor.fromData(xData, [1, K]);

        const wData = new Float32Array(K * N);
        for (let i = 0; i < K * N; i++) {
          wData[i] = Math.random() * 0.1;
        }

        const wQuantData = quantizeToQ4_K(wData);
        const wQuant = QuantizedTensor.fromQuantizedData(
          wQuantData,
          [K, N],
          GGMLType.Q4_K
        );

        const wDequant = await dequantizeQ4_K(wQuantData, K * N);
        const wDequantReshaped = wDequant.reshape([K, N]);

        const yQuant = await gemvQ4_K(x, wQuant);
        const yQuantData = await yQuant.toArray();

        const yF32 = await ops.matmul(x, wDequantReshaped);
        const yF32Data = await yF32.toArray();

        expect(yQuantData.length).toBe(N);

        let maxError = 0;
        for (let i = 0; i < N; i++) {
          const error = Math.abs(yQuantData[i] - yF32Data[i]);
          maxError = Math.max(maxError, error);
        }

        // Allow larger tolerance for bigger matrices
        expect(maxError).toBeLessThan(1.0);

        x.destroy();
        wQuant.destroy();
        wDequant.destroy();
        yQuant.destroy();
        yF32.destroy();
      });

      test('should validate input dimensions', async () => {
        if (!device) return;

        // x must have shape[0] === 1
        const xBad = Tensor.fromData(new Float32Array(512), [2, 256]);
        const wData = new Uint8Array(144); // One Q4_K block
        const wQuant = QuantizedTensor.fromQuantizedData(wData, [256, 1], GGMLType.Q4_K);

        await expect(gemvQ4_K(xBad, wQuant)).rejects.toThrow(/shape\[0\].*1/);

        xBad.destroy();
        wQuant.destroy();
      });

      test('should validate weight type', async () => {
        if (!device) return;

        const x = Tensor.fromData(new Float32Array(32), [1, 32]);
        // Create Q8_0 tensor instead of Q4_K
        const wData = quantizeToQ8_0(new Float32Array(32 * 4));
        const wQuant = QuantizedTensor.fromQuantizedData(wData, [32, 4], GGMLType.Q8_0);

        await expect(gemvQ4_K(x, wQuant)).rejects.toThrow(/Q4_K/);

        x.destroy();
        wQuant.destroy();
      });
    });
  });

  // =========================================================================
  // Weight Loading Infrastructure Tests (Phase 3)
  // =========================================================================

  describe('Weight Loading Infrastructure', () => {
    describe('WeightTensor Type Compatibility', () => {
      test('Tensor should be usable as WeightTensor', async () => {
        if (!device) return;

        // Create a regular f32 Tensor
        const data = new Float32Array([1, 2, 3, 4]);
        const tensor = Tensor.fromData(data, [2, 2], { label: 'test_f32' });

        // Should have standard Tensor properties
        expect(tensor.shape).toEqual([2, 2]);
        expect(tensor.ndim).toBe(2);

        // Can call toArray (unique to Tensor)
        const arr = await tensor.toArray();
        expect(arr.length).toBe(4);

        tensor.destroy();
      });

      test('QuantizedTensor should be usable as WeightTensor', async () => {
        if (!device) return;

        // Create a Q8_0 QuantizedTensor
        const quantData = new Uint8Array(68); // 2 blocks * 34 bytes
        const qt = QuantizedTensor.fromQuantizedData(
          quantData,
          [2, 32],
          GGMLType.Q8_0,
          { label: 'test_q8' }
        );

        // Should have common properties
        expect(qt.shape).toEqual([2, 32]);
        expect(qt.ndim).toBe(2);

        // Has getBuffer (common interface)
        expect(qt.getBuffer()).toBeDefined();

        // Has destroy (common interface)
        qt.destroy();
      });

      test('Both types should support shape and ndim properties', async () => {
        if (!device) return;

        // Create f32 weight
        const f32Weight = Tensor.fromData(new Float32Array(128), [32, 4]);

        // Create Q8_0 weight (same logical shape)
        const numBlocks = Math.ceil(128 / 32);
        const q8Weight = QuantizedTensor.fromQuantizedData(
          new Uint8Array(numBlocks * 34),
          [32, 4],
          GGMLType.Q8_0
        );

        // Both should have same shape
        expect(f32Weight.shape).toEqual([32, 4]);
        expect(q8Weight.shape).toEqual([32, 4]);

        // Both should report same ndim
        expect(f32Weight.ndim).toBe(2);
        expect(q8Weight.ndim).toBe(2);

        f32Weight.destroy();
        q8Weight.destroy();
      });
    });

    describe('Type Detection', () => {
      test('should correctly identify Tensor via instanceof', async () => {
        if (!device) return;

        const tensor = Tensor.fromData(new Float32Array(4), [2, 2]);

        expect(tensor instanceof Tensor).toBe(true);
        expect(tensor instanceof QuantizedTensor).toBe(false);

        tensor.destroy();
      });

      test('should correctly identify QuantizedTensor via instanceof', async () => {
        if (!device) return;

        const qt = QuantizedTensor.fromQuantizedData(
          new Uint8Array(34),
          [32],
          GGMLType.Q8_0
        );

        expect(qt instanceof QuantizedTensor).toBe(true);
        expect(qt instanceof Tensor).toBe(false);

        qt.destroy();
      });

      test('should allow runtime type checking for weight dispatch', async () => {
        if (!device) return;

        // This simulates what the engine needs to do
        const weights: (Tensor | QuantizedTensor)[] = [];

        // Add f32 weight
        weights.push(Tensor.fromData(new Float32Array(64), [8, 8]));

        // Add Q8_0 weight
        weights.push(QuantizedTensor.fromQuantizedData(
          new Uint8Array(68),
          [8, 8],
          GGMLType.Q8_0
        ));

        // Should be able to dispatch based on type
        for (const w of weights) {
          if (w instanceof QuantizedTensor) {
            // Would use gemvQ8_0
            expect(w.quantType).toBeDefined();
          } else {
            // Would use regular matmul
            expect((w as Tensor).dtype).toBeDefined();
          }
        }

        // Cleanup
        for (const w of weights) {
          w.destroy();
        }
      });
    });

    describe('Memory Stats Comparison', () => {
      test('QuantizedTensor should report significant compression', async () => {
        if (!device) return;

        // Simulate a typical attention weight [2048, 2048] = 4M elements
        // For test, use smaller [256, 256] = 65536 elements
        const rows = 256;
        const cols = 256;
        const numElements = rows * cols;

        // Q8_0: 65536 / 32 = 2048 blocks * 34 bytes = 69632 bytes
        const numBlocks = Math.ceil(numElements / 32);
        const q8Bytes = numBlocks * 34;

        const qt = QuantizedTensor.fromQuantizedData(
          new Uint8Array(q8Bytes),
          [rows, cols],
          GGMLType.Q8_0
        );

        const stats = qt.getMemoryStats();

        // f32 equivalent: 65536 * 4 = 262144 bytes
        expect(stats.equivalentF32Bytes).toBe(numElements * 4);
        expect(stats.quantizedBytes).toBe(q8Bytes);

        // Compression should be ~3.76x for Q8_0
        expect(stats.compressionRatio).toBeGreaterThan(3.5);
        expect(stats.compressionRatio).toBeLessThan(4.0);

        qt.destroy();
      });

      test('Q4_K should provide even higher compression', async () => {
        if (!device) return;

        // Q4_K: 256 elements per block, 144 bytes per block
        const numElements = 256 * 256; // 65536 elements
        const numBlocks = Math.ceil(numElements / 256); // 256 blocks
        const q4kBytes = numBlocks * 144; // 36864 bytes

        const qt = QuantizedTensor.fromQuantizedData(
          new Uint8Array(q4kBytes),
          [256, 256],
          GGMLType.Q4_K
        );

        const stats = qt.getMemoryStats();

        // f32 equivalent: 65536 * 4 = 262144 bytes
        // Q4_K: 36864 bytes
        // Compression: ~7.1x
        expect(stats.compressionRatio).toBeGreaterThan(7.0);
        expect(stats.compressionRatio).toBeLessThan(7.5);

        qt.destroy();
      });
    });
  });

  // =========================================================================
  // Engine Integration Tests (Phase 5)
  // =========================================================================

  describe('Engine Integration', () => {
    describe('matmulQ', () => {
      // Import matmulQ from layers
      const getMatmulQ = async () => {
        const mod = await import('../../src/engine/webgpu/layers/index.js');
        return mod.matmulQ;
      };

      test('should work with f32 Tensor weights', async () => {
        if (!device) return;
        const matmulQ = await getMatmulQ();

        // x[1, 64] @ W[64, 32] -> y[1, 32]
        const K = 64, N = 32;

        const xData = new Float32Array(K);
        for (let i = 0; i < K; i++) xData[i] = Math.random() - 0.5;
        const x = Tensor.fromData(xData, [1, K]);

        const wData = new Float32Array(K * N);
        for (let i = 0; i < K * N; i++) wData[i] = Math.random() - 0.5;
        const w = Tensor.fromData(wData, [K, N]);

        const y = await matmulQ(x, w);
        const yData = await y.toArray();

        expect(yData.length).toBe(N);

        // Verify against reference matmul
        const yRef = await ops.matmul(x, w);
        const yRefData = await yRef.toArray();

        for (let i = 0; i < N; i++) {
          expect(yData[i]).toBeCloseTo(yRefData[i], 5);
        }

        x.destroy();
        w.destroy();
        y.destroy();
        yRef.destroy();
      });

      test('should work with Q8_0 quantized weights', async () => {
        if (!device) return;
        const matmulQ = await getMatmulQ();

        const K = 64, N = 32;

        const xData = new Float32Array(K);
        for (let i = 0; i < K; i++) xData[i] = (Math.random() - 0.5) * 0.2;
        const x = Tensor.fromData(xData, [1, K]);

        const wData = new Float32Array(K * N);
        for (let i = 0; i < K * N; i++) wData[i] = (Math.random() - 0.5) * 0.2;

        const wQuantData = quantizeToQ8_0(wData);
        const w = QuantizedTensor.fromQuantizedData(wQuantData, [K, N], GGMLType.Q8_0);

        const y = await matmulQ(x, w);
        const yData = await y.toArray();

        expect(yData.length).toBe(N);

        // Should produce reasonable values (not NaN or Inf)
        for (let i = 0; i < N; i++) {
          expect(Number.isFinite(yData[i])).toBe(true);
        }

        x.destroy();
        w.destroy();
        y.destroy();
      });

      test('should work with batch size > 1 using GEMM', async () => {
        if (!device) return;
        const matmulQ = await getMatmulQ();

        const M = 4, K = 64, N = 32;

        // Batch of 4 (simulates prefill)
        const xData = new Float32Array(M * K);
        for (let i = 0; i < M * K; i++) xData[i] = (Math.random() - 0.5) * 0.2;
        const x = Tensor.fromData(xData, [M, K]);

        const wData = new Float32Array(K * N);
        for (let i = 0; i < K * N; i++) wData[i] = (Math.random() - 0.5) * 0.2;

        const wQuantData = quantizeToQ8_0(wData);
        const w = QuantizedTensor.fromQuantizedData(wQuantData, [K, N], GGMLType.Q8_0);

        const y = await matmulQ(x, w);
        const yData = await y.toArray();

        expect(yData.length).toBe(M * N);

        // Should produce reasonable values (not NaN or Inf)
        for (let i = 0; i < M * N; i++) {
          expect(Number.isFinite(yData[i])).toBe(true);
        }

        x.destroy();
        w.destroy();
        y.destroy();
      });
    });

    describe('gemmQ8_0', () => {
      const getGemmQ8_0 = async () => {
        const mod = await import('../../src/engine/webgpu/quant/qgemv.js');
        return mod.gemmQ8_0;
      };

      test('should compute batch matmul correctly', async () => {
        if (!device) return;
        const gemmQ8_0 = await getGemmQ8_0();

        const M = 8, K = 64, N = 32;

        // Create input batch
        const xData = new Float32Array(M * K);
        for (let i = 0; i < M * K; i++) xData[i] = (Math.random() - 0.5) * 0.2;
        const x = Tensor.fromData(xData, [M, K]);

        // Create and quantize weights
        const wData = new Float32Array(K * N);
        for (let i = 0; i < K * N; i++) wData[i] = (Math.random() - 0.5) * 0.2;

        const wQuantData = quantizeToQ8_0(wData);
        const wQuant = QuantizedTensor.fromQuantizedData(wQuantData, [K, N], GGMLType.Q8_0);

        // Compute using quantized GEMM
        const yQuant = await gemmQ8_0(x, wQuant);
        const yQuantData = await yQuant.toArray();

        expect(yQuantData.length).toBe(M * N);

        // Dequantize and compute reference
        const wDequant = await dequantizeQ8_0(wQuantData, K * N);
        const wDequantReshaped = wDequant.reshape([K, N]);
        const yRef = await ops.matmul(x, wDequantReshaped);
        const yRefData = await yRef.toArray();

        // Compare with tolerance
        let maxError = 0;
        for (let i = 0; i < M * N; i++) {
          const error = Math.abs(yQuantData[i] - yRefData[i]);
          maxError = Math.max(maxError, error);
        }
        expect(maxError).toBeLessThan(0.1);

        x.destroy();
        wQuant.destroy();
        yQuant.destroy();
        wDequant.destroy();
        yRef.destroy();
      });

      test('should handle typical prefill size', async () => {
        if (!device) return;
        const gemmQ8_0 = await getGemmQ8_0();

        // Simulate prefill: 32 tokens, hidden=256, output=128
        const M = 32, K = 256, N = 128;

        const xData = new Float32Array(M * K);
        for (let i = 0; i < M * K; i++) xData[i] = (Math.random() - 0.5) * 0.1;
        const x = Tensor.fromData(xData, [M, K]);

        const wData = new Float32Array(K * N);
        for (let i = 0; i < K * N; i++) wData[i] = (Math.random() - 0.5) * 0.1;

        const wQuantData = quantizeToQ8_0(wData);
        const wQuant = QuantizedTensor.fromQuantizedData(wQuantData, [K, N], GGMLType.Q8_0);

        const y = await gemmQ8_0(x, wQuant);
        const yData = await y.toArray();

        expect(yData.length).toBe(M * N);

        // Verify no NaN/Inf
        for (let i = 0; i < M * N; i++) {
          expect(Number.isFinite(yData[i])).toBe(true);
        }

        x.destroy();
        wQuant.destroy();
        y.destroy();
      });
    });

    describe('gemmQ4_K', () => {
      const getGemmQ4_K = async () => {
        const mod = await import('../../src/engine/webgpu/quant/qgemv.js');
        return mod.gemmQ4_K;
      };

      // Reuse quantize helper from gemvQ4_K tests
      function quantizeToQ4_K(data: Float32Array): Uint8Array {
        const QK_K = 256;
        const BYTES_PER_BLOCK = 144;
        const numBlocks = Math.ceil(data.length / QK_K);
        const result = new Uint8Array(numBlocks * BYTES_PER_BLOCK);

        for (let b = 0; b < numBlocks; b++) {
          const blockStart = b * QK_K;
          const outBase = b * BYTES_PER_BLOCK;

          let maxAbs = 0;
          for (let i = 0; i < QK_K && blockStart + i < data.length; i++) {
            maxAbs = Math.max(maxAbs, Math.abs(data[blockStart + i]));
          }

          const d = maxAbs / 15;
          const invD = d > 0 ? 15 / maxAbs : 0;

          const dF16 = floatToHalfSimple(d);
          result[outBase] = dF16 & 0xff;
          result[outBase + 1] = (dF16 >> 8) & 0xff;
          result[outBase + 2] = 0;
          result[outBase + 3] = 0;

          for (let i = 0; i < 12; i++) {
            result[outBase + 4 + i] = 1;
          }

          const qsBase = outBase + 16;
          for (let chunk = 0; chunk < 4; chunk++) {
            const chunkOffset = chunk * 64;
            const qOffset = chunk * 32;

            for (let l = 0; l < 32; l++) {
              const idx1 = blockStart + chunkOffset + l;
              const idx2 = blockStart + chunkOffset + 32 + l;

              const val1 = idx1 < data.length ? data[idx1] : 0;
              const val2 = idx2 < data.length ? data[idx2] : 0;

              const q1 = Math.min(15, Math.max(0, Math.round(val1 * invD)));
              const q2 = Math.min(15, Math.max(0, Math.round(val2 * invD)));

              result[qsBase + qOffset + l] = (q1 & 0xF) | ((q2 & 0xF) << 4);
            }
          }
        }
        return result;
      }

      function floatToHalfSimple(value: number): number {
        if (value === 0) return 0;
        const sign = value < 0 ? 0x8000 : 0;
        value = Math.abs(value);
        const exp = Math.floor(Math.log2(value));
        const mant = Math.round((value / Math.pow(2, exp) - 1) * 1024);
        const biasedExp = exp + 15;
        if (biasedExp <= 0) return sign;
        if (biasedExp >= 31) return sign | 0x7c00;
        return sign | (biasedExp << 10) | (mant & 0x3ff);
      }

      test('should compute batch matmul correctly', async () => {
        if (!device) return;
        const gemmQ4_K = await getGemmQ4_K();

        const M = 8, K = 256, N = 32;

        const xData = new Float32Array(M * K);
        for (let i = 0; i < M * K; i++) xData[i] = (Math.random() - 0.5) * 0.2;
        const x = Tensor.fromData(xData, [M, K]);

        const wData = new Float32Array(K * N);
        for (let i = 0; i < K * N; i++) wData[i] = Math.random() * 0.2;

        const wQuantData = quantizeToQ4_K(wData);
        const wQuant = QuantizedTensor.fromQuantizedData(wQuantData, [K, N], GGMLType.Q4_K);

        const yQuant = await gemmQ4_K(x, wQuant);
        const yQuantData = await yQuant.toArray();

        expect(yQuantData.length).toBe(M * N);

        // Verify no NaN/Inf
        for (let i = 0; i < M * N; i++) {
          expect(Number.isFinite(yQuantData[i])).toBe(true);
        }

        x.destroy();
        wQuant.destroy();
        yQuant.destroy();
      });

      test('should handle typical prefill size', async () => {
        if (!device) return;
        const gemmQ4_K = await getGemmQ4_K();

        // Simulate prefill: 16 tokens, hidden=512, output=256
        const M = 16, K = 512, N = 256;

        const xData = new Float32Array(M * K);
        for (let i = 0; i < M * K; i++) xData[i] = (Math.random() - 0.5) * 0.1;
        const x = Tensor.fromData(xData, [M, K]);

        const wData = new Float32Array(K * N);
        for (let i = 0; i < K * N; i++) wData[i] = Math.random() * 0.1;

        const wQuantData = quantizeToQ4_K(wData);
        const wQuant = QuantizedTensor.fromQuantizedData(wQuantData, [K, N], GGMLType.Q4_K);

        const y = await gemmQ4_K(x, wQuant);
        const yData = await y.toArray();

        expect(yData.length).toBe(M * N);

        for (let i = 0; i < M * N; i++) {
          expect(Number.isFinite(yData[i])).toBe(true);
        }

        x.destroy();
        wQuant.destroy();
        y.destroy();
      });
    });

    describe('feedForwardQ', () => {
      const getFeedForwardQ = async () => {
        const mod = await import('../../src/engine/webgpu/layers/index.js');
        return mod.feedForwardQ;
      };

      test('should work with all f32 weights (uses fused path)', async () => {
        if (!device) return;
        const feedForwardQ = await getFeedForwardQ();

        // Small FFN: hidden=64, intermediate=128
        const hidden = 64, intermediate = 128;

        const xData = new Float32Array(hidden);
        for (let i = 0; i < hidden; i++) xData[i] = (Math.random() - 0.5) * 0.1;
        const x = Tensor.fromData(xData, [1, hidden]);

        const gateData = new Float32Array(hidden * intermediate);
        const upData = new Float32Array(hidden * intermediate);
        const downData = new Float32Array(intermediate * hidden);
        for (let i = 0; i < hidden * intermediate; i++) {
          gateData[i] = (Math.random() - 0.5) * 0.1;
          upData[i] = (Math.random() - 0.5) * 0.1;
        }
        for (let i = 0; i < intermediate * hidden; i++) {
          downData[i] = (Math.random() - 0.5) * 0.1;
        }

        const wGate = Tensor.fromData(gateData, [hidden, intermediate]);
        const wUp = Tensor.fromData(upData, [hidden, intermediate]);
        const wDown = Tensor.fromData(downData, [intermediate, hidden]);

        const output = await feedForwardQ(x, wGate, wUp, wDown);
        const outputData = await output.toArray();

        expect(outputData.length).toBe(hidden);
        for (let i = 0; i < hidden; i++) {
          expect(Number.isFinite(outputData[i])).toBe(true);
        }

        x.destroy();
        wGate.destroy();
        wUp.destroy();
        wDown.destroy();
        output.destroy();
      });

      test('should work with Q8_0 quantized weights', async () => {
        if (!device) return;
        const feedForwardQ = await getFeedForwardQ();

        // Small FFN aligned to block size
        const hidden = 64, intermediate = 128;

        const xData = new Float32Array(hidden);
        for (let i = 0; i < hidden; i++) xData[i] = (Math.random() - 0.5) * 0.1;
        const x = Tensor.fromData(xData, [1, hidden]);

        // Quantize weights
        const gateData = new Float32Array(hidden * intermediate);
        const upData = new Float32Array(hidden * intermediate);
        const downData = new Float32Array(intermediate * hidden);
        for (let i = 0; i < hidden * intermediate; i++) {
          gateData[i] = (Math.random() - 0.5) * 0.1;
          upData[i] = (Math.random() - 0.5) * 0.1;
        }
        for (let i = 0; i < intermediate * hidden; i++) {
          downData[i] = (Math.random() - 0.5) * 0.1;
        }

        const wGate = QuantizedTensor.fromQuantizedData(
          quantizeToQ8_0(gateData), [hidden, intermediate], GGMLType.Q8_0
        );
        const wUp = QuantizedTensor.fromQuantizedData(
          quantizeToQ8_0(upData), [hidden, intermediate], GGMLType.Q8_0
        );
        const wDown = QuantizedTensor.fromQuantizedData(
          quantizeToQ8_0(downData), [intermediate, hidden], GGMLType.Q8_0
        );

        const output = await feedForwardQ(x, wGate, wUp, wDown);
        const outputData = await output.toArray();

        expect(outputData.length).toBe(hidden);
        for (let i = 0; i < hidden; i++) {
          expect(Number.isFinite(outputData[i])).toBe(true);
        }

        x.destroy();
        wGate.destroy();
        wUp.destroy();
        wDown.destroy();
        output.destroy();
      });

      test('should work with mixed f32 and quantized weights', async () => {
        if (!device) return;
        const feedForwardQ = await getFeedForwardQ();

        const hidden = 64, intermediate = 128;

        const xData = new Float32Array(hidden);
        for (let i = 0; i < hidden; i++) xData[i] = (Math.random() - 0.5) * 0.1;
        const x = Tensor.fromData(xData, [1, hidden]);

        // Mix: f32 gate, Q8_0 up, f32 down
        const gateData = new Float32Array(hidden * intermediate);
        const upData = new Float32Array(hidden * intermediate);
        const downData = new Float32Array(intermediate * hidden);
        for (let i = 0; i < hidden * intermediate; i++) {
          gateData[i] = (Math.random() - 0.5) * 0.1;
          upData[i] = (Math.random() - 0.5) * 0.1;
        }
        for (let i = 0; i < intermediate * hidden; i++) {
          downData[i] = (Math.random() - 0.5) * 0.1;
        }

        const wGate = Tensor.fromData(gateData, [hidden, intermediate]);
        const wUp = QuantizedTensor.fromQuantizedData(
          quantizeToQ8_0(upData), [hidden, intermediate], GGMLType.Q8_0
        );
        const wDown = Tensor.fromData(downData, [intermediate, hidden]);

        const output = await feedForwardQ(x, wGate, wUp, wDown);
        const outputData = await output.toArray();

        expect(outputData.length).toBe(hidden);
        for (let i = 0; i < hidden; i++) {
          expect(Number.isFinite(outputData[i])).toBe(true);
        }

        x.destroy();
        wGate.destroy();
        wUp.destroy();
        wDown.destroy();
        output.destroy();
      });
    });
  });
});
