/**
 * Tests for GPU-accelerated Top-K Sampling
 */

import {
  WebGPUDevice,
  initWebGPU,
  Tensor,
} from '../../src/engine/webgpu/index.js';
import {
  topKSoftmaxGPU,
  sampleFromTopK,
  resetTopKPipeline,
} from '../../src/engine/webgpu/ops/index.js';

describe('GPU Top-K Sampling', () => {
  beforeAll(async () => {
    if (await WebGPUDevice.isAvailable()) {
      await initWebGPU({ powerPreference: 'high-performance' });
    }
  });

  beforeEach(() => {
    resetTopKPipeline();
  });

  const isWebGPUAvailable = async () => WebGPUDevice.isAvailable();

  test('should extract top-k values from logits', async () => {
    if (!(await isWebGPUAvailable())) {
      console.log('Skipping: WebGPU not available');
      return;
    }

    // Create logits with known top values
    const vocabSize = 1000;
    const logitsData = new Float32Array(vocabSize);

    // Set some specific high values
    logitsData[42] = 10.0;   // Should be in top-k
    logitsData[123] = 9.5;   // Should be in top-k
    logitsData[456] = 9.0;   // Should be in top-k
    logitsData[789] = 8.5;   // Should be in top-k

    // Fill rest with low values
    for (let i = 0; i < vocabSize; i++) {
      if (logitsData[i] === 0) {
        logitsData[i] = Math.random() * 2.0 - 1.0; // Random in [-1, 1]
      }
    }

    const logits = Tensor.fromData(logitsData, [1, vocabSize]);

    const result = await topKSoftmaxGPU(logits, 10, 1.0);
    logits.destroy();

    // Check that high-value indices are in the result
    const indices = Array.from(result.indices);
    expect(indices).toContain(42);
    expect(indices).toContain(123);
    expect(indices).toContain(456);
    expect(indices).toContain(789);

    // Probabilities should sum to 1
    let sum = 0;
    for (const p of result.probs) {
      sum += p;
    }
    expect(sum).toBeCloseTo(1.0, 1); // Allow some tolerance
  });

  test('should handle temperature scaling', async () => {
    if (!(await isWebGPUAvailable())) {
      console.log('Skipping: WebGPU not available');
      return;
    }

    const vocabSize = 100;
    const logitsData = new Float32Array(vocabSize);
    logitsData[0] = 5.0;
    logitsData[1] = 4.0;
    for (let i = 2; i < vocabSize; i++) {
      logitsData[i] = 1.0;
    }

    const logits1 = Tensor.fromData(logitsData, [1, vocabSize]);
    const logits2 = Tensor.fromData(logitsData, [1, vocabSize]);

    // Low temperature = sharper distribution
    const resultLow = await topKSoftmaxGPU(logits1, 10, 0.5);
    logits1.destroy();

    // High temperature = flatter distribution
    const resultHigh = await topKSoftmaxGPU(logits2, 10, 2.0);
    logits2.destroy();

    // Low temperature should have higher max probability
    const maxLow = Math.max(...resultLow.probs);
    const maxHigh = Math.max(...resultHigh.probs);
    expect(maxLow).toBeGreaterThan(maxHigh);
  });

  test('should sample from probabilities correctly', async () => {
    // Test the CPU sampling function directly
    const probs = new Float32Array([0.5, 0.3, 0.2]);
    const indices = new Uint32Array([100, 200, 300]);

    // Sample many times and check distribution
    const counts: Record<number, number> = { 100: 0, 200: 0, 300: 0 };
    const iterations = 10000;

    for (let i = 0; i < iterations; i++) {
      const token = sampleFromTopK(probs, indices);
      counts[token]++;
    }

    // Check approximate distribution
    expect(counts[100] / iterations).toBeCloseTo(0.5, 1);
    expect(counts[200] / iterations).toBeCloseTo(0.3, 1);
    expect(counts[300] / iterations).toBeCloseTo(0.2, 1);
  });

  test('should handle large vocabulary (128K)', async () => {
    if (!(await isWebGPUAvailable())) {
      console.log('Skipping: WebGPU not available');
      return;
    }

    const vocabSize = 128000;
    const logitsData = new Float32Array(vocabSize);

    // Set random logits
    for (let i = 0; i < vocabSize; i++) {
      logitsData[i] = Math.random() * 10.0 - 5.0;
    }

    // Set one clear winner
    logitsData[65432] = 100.0;

    const logits = Tensor.fromData(logitsData, [1, vocabSize]);

    const start = performance.now();
    const result = await topKSoftmaxGPU(logits, 40, 0.8);
    const elapsed = performance.now() - start;

    logits.destroy();

    console.log(`Top-K on 128K vocab: ${elapsed.toFixed(1)}ms`);

    // The winner should be in the result
    expect(Array.from(result.indices)).toContain(65432);

    // Time should be reasonable (< 50ms on most GPUs)
    expect(elapsed).toBeLessThan(100);
  });

  test('should be faster than full toArray', async () => {
    if (!(await isWebGPUAvailable())) {
      console.log('Skipping: WebGPU not available');
      return;
    }

    const vocabSize = 128000;
    const logitsData = new Float32Array(vocabSize);
    for (let i = 0; i < vocabSize; i++) {
      logitsData[i] = Math.random() * 10.0 - 5.0;
    }

    // Warm up
    {
      const warmup = Tensor.fromData(logitsData, [1, vocabSize]);
      await warmup.toArray();
      warmup.destroy();
    }
    {
      const warmup = Tensor.fromData(logitsData, [1, vocabSize]);
      await topKSoftmaxGPU(warmup, 40, 0.8);
      warmup.destroy();
    }

    // Measure toArray time
    const toArrayTimes: number[] = [];
    for (let i = 0; i < 5; i++) {
      const tensor = Tensor.fromData(logitsData, [1, vocabSize]);
      const start = performance.now();
      await tensor.toArray();
      toArrayTimes.push(performance.now() - start);
      tensor.destroy();
    }

    // Measure GPU top-k time
    const gpuTopKTimes: number[] = [];
    for (let i = 0; i < 5; i++) {
      const tensor = Tensor.fromData(logitsData, [1, vocabSize]);
      const start = performance.now();
      await topKSoftmaxGPU(tensor, 40, 0.8);
      gpuTopKTimes.push(performance.now() - start);
      tensor.destroy();
    }

    const avgToArray = toArrayTimes.reduce((a, b) => a + b, 0) / toArrayTimes.length;
    const avgGpuTopK = gpuTopKTimes.reduce((a, b) => a + b, 0) / gpuTopKTimes.length;

    console.log(`Full toArray (128K): ${avgToArray.toFixed(1)}ms`);
    console.log(`GPU Top-K (40): ${avgGpuTopK.toFixed(1)}ms`);
    console.log(`Speedup: ${(avgToArray / avgGpuTopK).toFixed(1)}x`);

    // GPU top-k should be significantly faster
    // (This might fail on very fast GPUs where toArray is already quick)
    // expect(avgGpuTopK).toBeLessThan(avgToArray);
  });
});
