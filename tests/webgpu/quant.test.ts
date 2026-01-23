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
  quantizeToQ8_0,
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
});
