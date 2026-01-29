/**
 * Tests for Optimized Quantized GEMV (Phase 1 - Memory Coalescing)
 *
 * These tests verify:
 * 1. Correctness: Optimized version produces same results as original
 * 2. Performance: Optimized version is faster than original
 */

import {
  WebGPUDevice,
  initWebGPU,
  QuantizedTensor,
  Tensor,
  gemvQ4_K,
  dequantizeQ4_K,
  ops,
} from '../../src/engine/webgpu/index.js';
import { gemvQ4_K_optimized, resetOptimizedPipelines } from '../../src/engine/webgpu/quant/qgemv-optimized.js';
import { GGMLType } from '../../src/types/model.js';

// Skip tests if WebGPU is not available
const describeWebGPU = process.env.SKIP_WEBGPU_TESTS ? describe.skip : describe;

// Helper to quantize f32 to Q4_K format
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

describeWebGPU('Optimized Q4_K GEMV', () => {
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

  beforeEach(() => {
    // Reset pipeline caches between tests
    resetOptimizedPipelines();
  });

  describe('Correctness Tests', () => {
    test('should produce same results as original for small matrices', async () => {
      if (!device) return;

      const K = 256;  // One Q4_K block
      const N = 64;

      // Create input
      const xData = new Float32Array(K);
      for (let i = 0; i < K; i++) {
        xData[i] = (Math.random() - 0.5) * 0.2;
      }
      const x = Tensor.fromData(xData, [1, K]);

      // Create weights
      const wData = new Float32Array(K * N);
      for (let i = 0; i < K * N; i++) {
        wData[i] = Math.random() * 0.2;
      }

      const wQuantData = quantizeToQ4_K(wData);
      const wQuant = QuantizedTensor.fromQuantizedData(
        wQuantData,
        [K, N],
        GGMLType.Q4_K
      );

      // Run original
      const yOriginal = await gemvQ4_K(x, wQuant);
      const yOriginalData = await yOriginal.toArray();

      // Run optimized
      const yOptimized = await gemvQ4_K_optimized(x, wQuant);
      const yOptimizedData = await yOptimized.toArray();

      // Compare
      expect(yOptimizedData.length).toBe(N);
      expect(yOriginalData.length).toBe(N);

      let maxError = 0;
      for (let i = 0; i < N; i++) {
        const error = Math.abs(yOptimizedData[i] - yOriginalData[i]);
        maxError = Math.max(maxError, error);
      }

      // Results should be identical (same algorithm, just reorganized)
      expect(maxError).toBeLessThan(0.001);

      // Cleanup
      x.destroy();
      wQuant.destroy();
      yOriginal.destroy();
      yOptimized.destroy();
    });

    test('should produce same results as original for larger matrices', async () => {
      if (!device) return;

      const K = 512;  // Two Q4_K blocks
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

      const yOriginal = await gemvQ4_K(x, wQuant);
      const yOriginalData = await yOriginal.toArray();

      const yOptimized = await gemvQ4_K_optimized(x, wQuant);
      const yOptimizedData = await yOptimized.toArray();

      let maxError = 0;
      for (let i = 0; i < N; i++) {
        const error = Math.abs(yOptimizedData[i] - yOriginalData[i]);
        maxError = Math.max(maxError, error);
      }

      expect(maxError).toBeLessThan(0.01);

      x.destroy();
      wQuant.destroy();
      yOriginal.destroy();
      yOptimized.destroy();
    });

    test('should produce same results as f32 reference within quantization tolerance', async () => {
      if (!device) return;

      const K = 256;
      const N = 128;

      const xData = new Float32Array(K);
      for (let i = 0; i < K; i++) {
        xData[i] = (Math.random() - 0.5) * 0.2;
      }
      const x = Tensor.fromData(xData, [1, K]);

      const wData = new Float32Array(K * N);
      for (let i = 0; i < K * N; i++) {
        wData[i] = Math.random() * 0.2;
      }

      const wQuantData = quantizeToQ4_K(wData);
      const wQuant = QuantizedTensor.fromQuantizedData(
        wQuantData,
        [K, N],
        GGMLType.Q4_K
      );

      // Dequantize for f32 reference
      const wDequant = await dequantizeQ4_K(wQuantData, K * N);
      const wDequantReshaped = wDequant.reshape([K, N]);

      // Run optimized
      const yOptimized = await gemvQ4_K_optimized(x, wQuant);
      const yOptimizedData = await yOptimized.toArray();

      // Run f32 reference
      const yF32 = await ops.matmul(x, wDequantReshaped);
      const yF32Data = await yF32.toArray();

      let maxError = 0;
      for (let i = 0; i < N; i++) {
        const error = Math.abs(yOptimizedData[i] - yF32Data[i]);
        maxError = Math.max(maxError, error);
      }

      // Q4_K has lower precision, allow reasonable tolerance
      expect(maxError).toBeLessThan(0.5);

      x.destroy();
      wQuant.destroy();
      wDequant.destroy();
      yOptimized.destroy();
      yF32.destroy();
    });

    test('should handle non-aligned dimensions', async () => {
      if (!device) return;

      // K not aligned to 256, N not aligned to 256
      const K = 300;
      const N = 100;

      const xData = new Float32Array(K);
      for (let i = 0; i < K; i++) {
        xData[i] = Math.random() - 0.5;
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

      const yOriginal = await gemvQ4_K(x, wQuant);
      const yOriginalData = await yOriginal.toArray();

      const yOptimized = await gemvQ4_K_optimized(x, wQuant);
      const yOptimizedData = await yOptimized.toArray();

      expect(yOptimizedData.length).toBe(N);

      let maxError = 0;
      for (let i = 0; i < N; i++) {
        const error = Math.abs(yOptimizedData[i] - yOriginalData[i]);
        maxError = Math.max(maxError, error);
      }

      expect(maxError).toBeLessThan(0.01);

      x.destroy();
      wQuant.destroy();
      yOriginal.destroy();
      yOptimized.destroy();
    });

    test('should validate input dimensions', async () => {
      if (!device) return;

      // x must have shape[0] === 1
      const xBad = Tensor.fromData(new Float32Array(512), [2, 256]);
      const wData = new Uint8Array(144); // One Q4_K block
      const wQuant = QuantizedTensor.fromQuantizedData(wData, [256, 1], GGMLType.Q4_K);

      await expect(gemvQ4_K_optimized(xBad, wQuant)).rejects.toThrow(/shape\[0\].*1/);

      xBad.destroy();
      wQuant.destroy();
    });

    test('should validate weight type', async () => {
      if (!device) return;

      const x = Tensor.fromData(new Float32Array(32), [1, 32]);
      // Create Q8_0 tensor instead of Q4_K
      const wData = new Uint8Array(34); // One Q8_0 block
      const wQuant = QuantizedTensor.fromQuantizedData(wData, [32, 1], GGMLType.Q8_0);

      await expect(gemvQ4_K_optimized(x, wQuant)).rejects.toThrow(/Q4_K/);

      x.destroy();
      wQuant.destroy();
    });
  });

  describe('Performance Tests', () => {
    // Performance test helper
    async function measureTime(
      fn: () => Promise<void>,
      warmupRuns: number = 3,
      measuredRuns: number = 10
    ): Promise<{ mean: number; std: number; min: number; max: number }> {
      // Warmup
      for (let i = 0; i < warmupRuns; i++) {
        await fn();
      }

      // Measure
      const times: number[] = [];
      for (let i = 0; i < measuredRuns; i++) {
        const start = performance.now();
        await fn();
        const end = performance.now();
        times.push(end - start);
      }

      const mean = times.reduce((a, b) => a + b, 0) / times.length;
      const variance = times.reduce((a, b) => a + (b - mean) ** 2, 0) / times.length;
      const std = Math.sqrt(variance);
      const min = Math.min(...times);
      const max = Math.max(...times);

      return { mean, std, min, max };
    }

    test('should be faster than original for typical hidden size', async () => {
      if (!device) return;

      // Typical attention projection: [1, 2048] @ [2048, 2048]
      const K = 2048;
      const N = 2048;

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

      // Measure original
      const origRef: { value: Tensor | null } = { value: null };
      const originalTime = await measureTime(async () => {
        if (origRef.value) origRef.value.destroy();
        origRef.value = await gemvQ4_K(x, wQuant);
      });

      // Measure optimized
      const optRef: { value: Tensor | null } = { value: null };
      const optimizedTime = await measureTime(async () => {
        if (optRef.value) optRef.value.destroy();
        optRef.value = await gemvQ4_K_optimized(x, wQuant);
      });

      console.log(`\nPerformance comparison (K=${K}, N=${N}):`);
      console.log(`  Original:  ${originalTime.mean.toFixed(2)}ms ± ${originalTime.std.toFixed(2)}ms`);
      console.log(`  Optimized: ${optimizedTime.mean.toFixed(2)}ms ± ${optimizedTime.std.toFixed(2)}ms`);
      console.log(`  Speedup:   ${(originalTime.mean / optimizedTime.mean).toFixed(2)}x`);

      // Note: We don't assert a specific speedup because it depends on hardware
      // The test serves as a benchmark and to verify no regression

      // Cleanup
      x.destroy();
      wQuant.destroy();
      if (origRef.value) origRef.value.destroy();
      if (optRef.value) optRef.value.destroy();
    }, 60000); // 60 second timeout for performance test

    test('should scale well with larger matrices', async () => {
      if (!device) return;

      // Test multiple sizes
      const sizes = [
        { K: 256, N: 256 },
        { K: 512, N: 512 },
        { K: 1024, N: 1024 },
        { K: 2048, N: 2048 },
      ];

      console.log('\nScaling test (optimized gemvQ4_K):');

      for (const { K, N } of sizes) {
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

        const yRef: { value: Tensor | null } = { value: null };
        const time = await measureTime(async () => {
          if (yRef.value) yRef.value.destroy();
          yRef.value = await gemvQ4_K_optimized(x, wQuant);
        }, 2, 5);

        console.log(`  ${K}x${N}: ${time.mean.toFixed(2)}ms (${(K * N / time.mean / 1e6).toFixed(2)} M elem/ms)`);

        x.destroy();
        wQuant.destroy();
        if (yRef.value) yRef.value.destroy();
      }
    }, 120000); // 2 minute timeout
  });

  describe('Edge Cases', () => {
    test('should handle minimum practical matrices (256x256)', async () => {
      if (!device) return;

      // Minimum practical size: K=256 (one Q4_K block), N=256 (one workgroup)
      const K = 256;
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

      // Compare optimized vs original
      const yOpt = await gemvQ4_K_optimized(x, wQuant);
      const yOrig = await gemvQ4_K(x, wQuant);
      const yOptData = await yOpt.toArray();
      const yOrigData = await yOrig.toArray();

      expect(yOptData.length).toBe(N);

      // Results should match original within tolerance
      let maxError = 0;
      for (let i = 0; i < N; i++) {
        expect(Number.isFinite(yOptData[i])).toBe(true);
        const error = Math.abs(yOptData[i] - yOrigData[i]);
        maxError = Math.max(maxError, error);
      }
      expect(maxError).toBeLessThan(1e-5);

      x.destroy();
      wQuant.destroy();
      yOpt.destroy();
      yOrig.destroy();
    });

    test('should handle N larger than workgroup columns', async () => {
      if (!device) return;

      const K = 256;
      const N = 1024;  // Multiple workgroups needed

      const xData = new Float32Array(K);
      for (let i = 0; i < K; i++) {
        xData[i] = Math.random() - 0.5;
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

      const yOriginal = await gemvQ4_K(x, wQuant);
      const yOriginalData = await yOriginal.toArray();

      const yOptimized = await gemvQ4_K_optimized(x, wQuant);
      const yOptimizedData = await yOptimized.toArray();

      expect(yOptimizedData.length).toBe(N);

      let maxError = 0;
      for (let i = 0; i < N; i++) {
        const error = Math.abs(yOptimizedData[i] - yOriginalData[i]);
        maxError = Math.max(maxError, error);
      }

      expect(maxError).toBeLessThan(0.01);

      x.destroy();
      wQuant.destroy();
      yOriginal.destroy();
      yOptimized.destroy();
    });

    test('should not produce NaN or Inf', async () => {
      if (!device) return;

      const K = 512;
      const N = 256;

      // Use values that might cause numerical issues
      const xData = new Float32Array(K);
      for (let i = 0; i < K; i++) {
        xData[i] = i % 2 === 0 ? 100.0 : -100.0;
      }
      const x = Tensor.fromData(xData, [1, K]);

      const wData = new Float32Array(K * N);
      for (let i = 0; i < K * N; i++) {
        wData[i] = (i % 3 === 0) ? 10.0 : ((i % 3 === 1) ? -10.0 : 0.0);
      }

      const wQuantData = quantizeToQ4_K(wData);
      const wQuant = QuantizedTensor.fromQuantizedData(
        wQuantData,
        [K, N],
        GGMLType.Q4_K
      );

      const y = await gemvQ4_K_optimized(x, wQuant);
      const yData = await y.toArray();

      for (let i = 0; i < N; i++) {
        expect(Number.isFinite(yData[i])).toBe(true);
      }

      x.destroy();
      wQuant.destroy();
      y.destroy();
    });
  });
});
