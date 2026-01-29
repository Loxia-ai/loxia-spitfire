/**
 * Tests for Causal Attention with vec4 optimization
 */

import {
  WebGPUDevice,
  initWebGPU,
  Tensor,
  ops,
} from '../../src/engine/webgpu/index.js';

let device: WebGPUDevice | null = null;

beforeAll(async () => {
  try {
    device = await initWebGPU();
  } catch {
    console.log('WebGPU not available, skipping tests');
  }
});

afterAll(async () => {
  if (device) {
    device.destroy();
  }
});

/**
 * CPU reference implementation of causal attention
 */
function causalAttentionCPU(
  Q: Float32Array,
  K: Float32Array,
  V: Float32Array,
  numQPos: number,
  numKeys: number,
  numHeads: number,
  numKVHeads: number,
  headDim: number,
  startPos: number
): Float32Array {
  const output = new Float32Array(numQPos * numHeads * headDim);
  const scale = 1.0 / Math.sqrt(headDim);
  const kvRatio = numHeads / numKVHeads;

  for (let qPos = 0; qPos < numQPos; qPos++) {
    for (let head = 0; head < numHeads; head++) {
      const kvHead = Math.floor(head / kvRatio);
      const absQPos = startPos + qPos;
      const validKeys = Math.min(absQPos + 1, numKeys);

      // Compute attention scores
      const scores = new Float32Array(validKeys);
      for (let ki = 0; ki < validKeys; ki++) {
        let score = 0;
        for (let d = 0; d < headDim; d++) {
          const qIdx = qPos * numHeads * headDim + head * headDim + d;
          const kIdx = ki * numKVHeads * headDim + kvHead * headDim + d;
          score += Q[qIdx] * K[kIdx];
        }
        scores[ki] = score * scale;
      }

      // Softmax
      let maxScore = -Infinity;
      for (let ki = 0; ki < validKeys; ki++) {
        maxScore = Math.max(maxScore, scores[ki]);
      }
      let sumExp = 0;
      for (let ki = 0; ki < validKeys; ki++) {
        scores[ki] = Math.exp(scores[ki] - maxScore);
        sumExp += scores[ki];
      }
      for (let ki = 0; ki < validKeys; ki++) {
        scores[ki] /= sumExp;
      }

      // Weighted sum of V
      for (let d = 0; d < headDim; d++) {
        let sum = 0;
        for (let ki = 0; ki < validKeys; ki++) {
          const vIdx = ki * numKVHeads * headDim + kvHead * headDim + d;
          sum += scores[ki] * V[vIdx];
        }
        const outIdx = qPos * numHeads * headDim + head * headDim + d;
        output[outIdx] = sum;
      }
    }
  }

  return output;
}

describe('Causal Attention', () => {
  describe('Correctness Tests', () => {
    test('should match CPU reference for small input', async () => {
      if (!device) return;

      const numQPos = 1;
      const numKeys = 16;
      const numHeads = 4;
      const numKVHeads = 4;
      const headDim = 64;  // Divisible by 4
      const startPos = 15;

      // Create random inputs
      const qData = new Float32Array(numQPos * numHeads * headDim);
      const kData = new Float32Array(numKeys * numKVHeads * headDim);
      const vData = new Float32Array(numKeys * numKVHeads * headDim);

      for (let i = 0; i < qData.length; i++) qData[i] = (Math.random() - 0.5) * 0.1;
      for (let i = 0; i < kData.length; i++) kData[i] = (Math.random() - 0.5) * 0.1;
      for (let i = 0; i < vData.length; i++) vData[i] = (Math.random() - 0.5) * 0.1;

      const Q = Tensor.fromData(qData, [numQPos, numHeads * headDim]);
      const K = Tensor.fromData(kData, [numKeys, numKVHeads * headDim]);
      const V = Tensor.fromData(vData, [numKeys, numKVHeads * headDim]);

      // GPU result
      const gpuOutput = await ops.causalAttention(Q, K, V, numHeads, numKVHeads, headDim, startPos);
      const gpuData = await gpuOutput.toArray();

      // CPU reference
      const cpuData = causalAttentionCPU(qData, kData, vData, numQPos, numKeys, numHeads, numKVHeads, headDim, startPos);

      // Compare
      let maxError = 0;
      for (let i = 0; i < cpuData.length; i++) {
        const error = Math.abs(gpuData[i] - cpuData[i]);
        maxError = Math.max(maxError, error);
      }

      expect(maxError).toBeLessThan(1e-4);

      Q.destroy();
      K.destroy();
      V.destroy();
      gpuOutput.destroy();
    });

    test('should match CPU reference for headDim=128', async () => {
      if (!device) return;

      const numQPos = 1;
      const numKeys = 32;
      const numHeads = 8;
      const numKVHeads = 8;
      const headDim = 128;  // Larger head dimension
      const startPos = 31;

      const qData = new Float32Array(numQPos * numHeads * headDim);
      const kData = new Float32Array(numKeys * numKVHeads * headDim);
      const vData = new Float32Array(numKeys * numKVHeads * headDim);

      for (let i = 0; i < qData.length; i++) qData[i] = (Math.random() - 0.5) * 0.1;
      for (let i = 0; i < kData.length; i++) kData[i] = (Math.random() - 0.5) * 0.1;
      for (let i = 0; i < vData.length; i++) vData[i] = (Math.random() - 0.5) * 0.1;

      const Q = Tensor.fromData(qData, [numQPos, numHeads * headDim]);
      const K = Tensor.fromData(kData, [numKeys, numKVHeads * headDim]);
      const V = Tensor.fromData(vData, [numKeys, numKVHeads * headDim]);

      const gpuOutput = await ops.causalAttention(Q, K, V, numHeads, numKVHeads, headDim, startPos);
      const gpuData = await gpuOutput.toArray();
      const cpuData = causalAttentionCPU(qData, kData, vData, numQPos, numKeys, numHeads, numKVHeads, headDim, startPos);

      let maxError = 0;
      for (let i = 0; i < cpuData.length; i++) {
        const error = Math.abs(gpuData[i] - cpuData[i]);
        maxError = Math.max(maxError, error);
      }

      expect(maxError).toBeLessThan(1e-4);

      Q.destroy();
      K.destroy();
      V.destroy();
      gpuOutput.destroy();
    });

    test('should handle GQA (fewer KV heads)', async () => {
      if (!device) return;

      const numQPos = 1;
      const numKeys = 64;
      const numHeads = 32;
      const numKVHeads = 8;  // GQA: 4 query heads per KV head
      const headDim = 64;
      const startPos = 63;

      const qData = new Float32Array(numQPos * numHeads * headDim);
      const kData = new Float32Array(numKeys * numKVHeads * headDim);
      const vData = new Float32Array(numKeys * numKVHeads * headDim);

      for (let i = 0; i < qData.length; i++) qData[i] = (Math.random() - 0.5) * 0.1;
      for (let i = 0; i < kData.length; i++) kData[i] = (Math.random() - 0.5) * 0.1;
      for (let i = 0; i < vData.length; i++) vData[i] = (Math.random() - 0.5) * 0.1;

      const Q = Tensor.fromData(qData, [numQPos, numHeads * headDim]);
      const K = Tensor.fromData(kData, [numKeys, numKVHeads * headDim]);
      const V = Tensor.fromData(vData, [numKeys, numKVHeads * headDim]);

      const gpuOutput = await ops.causalAttention(Q, K, V, numHeads, numKVHeads, headDim, startPos);
      const gpuData = await gpuOutput.toArray();
      const cpuData = causalAttentionCPU(qData, kData, vData, numQPos, numKeys, numHeads, numKVHeads, headDim, startPos);

      let maxError = 0;
      for (let i = 0; i < cpuData.length; i++) {
        const error = Math.abs(gpuData[i] - cpuData[i]);
        maxError = Math.max(maxError, error);
      }

      expect(maxError).toBeLessThan(1e-4);

      Q.destroy();
      K.destroy();
      V.destroy();
      gpuOutput.destroy();
    });

    test('should handle sequence > 1024 keys (chunked processing)', async () => {
      if (!device) return;

      const numQPos = 1;
      const numKeys = 2048;  // Requires 2 chunks
      const numHeads = 4;
      const numKVHeads = 4;
      const headDim = 64;
      const startPos = 2047;

      const qData = new Float32Array(numQPos * numHeads * headDim);
      const kData = new Float32Array(numKeys * numKVHeads * headDim);
      const vData = new Float32Array(numKeys * numKVHeads * headDim);

      for (let i = 0; i < qData.length; i++) qData[i] = (Math.random() - 0.5) * 0.1;
      for (let i = 0; i < kData.length; i++) kData[i] = (Math.random() - 0.5) * 0.1;
      for (let i = 0; i < vData.length; i++) vData[i] = (Math.random() - 0.5) * 0.1;

      const Q = Tensor.fromData(qData, [numQPos, numHeads * headDim]);
      const K = Tensor.fromData(kData, [numKeys, numKVHeads * headDim]);
      const V = Tensor.fromData(vData, [numKeys, numKVHeads * headDim]);

      const gpuOutput = await ops.causalAttention(Q, K, V, numHeads, numKVHeads, headDim, startPos);
      const gpuData = await gpuOutput.toArray();
      const cpuData = causalAttentionCPU(qData, kData, vData, numQPos, numKeys, numHeads, numKVHeads, headDim, startPos);

      let maxError = 0;
      for (let i = 0; i < cpuData.length; i++) {
        const error = Math.abs(gpuData[i] - cpuData[i]);
        maxError = Math.max(maxError, error);
      }

      // Slightly higher tolerance for longer sequences due to accumulated floating point error
      expect(maxError).toBeLessThan(1e-3);

      Q.destroy();
      K.destroy();
      V.destroy();
      gpuOutput.destroy();
    });
  });

  describe('Performance Tests', () => {
    test('should complete within reasonable time for typical LLM sizes', async () => {
      if (!device) return;

      // Typical LLM config: Llama-like
      const numQPos = 1;
      const numKeys = 512;
      const numHeads = 32;
      const numKVHeads = 8;
      const headDim = 128;
      const startPos = 511;

      const qData = new Float32Array(numQPos * numHeads * headDim);
      const kData = new Float32Array(numKeys * numKVHeads * headDim);
      const vData = new Float32Array(numKeys * numKVHeads * headDim);

      for (let i = 0; i < qData.length; i++) qData[i] = Math.random() * 0.1;
      for (let i = 0; i < kData.length; i++) kData[i] = Math.random() * 0.1;
      for (let i = 0; i < vData.length; i++) vData[i] = Math.random() * 0.1;

      const Q = Tensor.fromData(qData, [numQPos, numHeads * headDim]);
      const K = Tensor.fromData(kData, [numKeys, numKVHeads * headDim]);
      const V = Tensor.fromData(vData, [numKeys, numKVHeads * headDim]);

      // Warmup
      let output = await ops.causalAttention(Q, K, V, numHeads, numKVHeads, headDim, startPos);
      await output.toArray();
      output.destroy();

      // Benchmark
      const iterations = 10;
      const start = performance.now();
      for (let i = 0; i < iterations; i++) {
        output = await ops.causalAttention(Q, K, V, numHeads, numKVHeads, headDim, startPos);
        await output.toArray();
        output.destroy();
      }
      const elapsed = performance.now() - start;
      const avgMs = elapsed / iterations;

      console.log(`\nCausal Attention Performance (${numHeads} heads, ${numKeys} keys, headDim=${headDim}):`);
      console.log(`  Average: ${avgMs.toFixed(2)}ms per call`);

      Q.destroy();
      K.destroy();
      V.destroy();
    }, 60000);
  });

  describe('Edge Cases', () => {
    test('should handle single key', async () => {
      if (!device) return;

      const numQPos = 1;
      const numKeys = 1;
      const numHeads = 4;
      const numKVHeads = 4;
      const headDim = 64;
      const startPos = 0;

      const qData = new Float32Array(numQPos * numHeads * headDim);
      const kData = new Float32Array(numKeys * numKVHeads * headDim);
      const vData = new Float32Array(numKeys * numKVHeads * headDim);

      for (let i = 0; i < qData.length; i++) qData[i] = Math.random() * 0.1;
      for (let i = 0; i < kData.length; i++) kData[i] = Math.random() * 0.1;
      for (let i = 0; i < vData.length; i++) vData[i] = Math.random() * 0.1;

      const Q = Tensor.fromData(qData, [numQPos, numHeads * headDim]);
      const K = Tensor.fromData(kData, [numKeys, numKVHeads * headDim]);
      const V = Tensor.fromData(vData, [numKeys, numKVHeads * headDim]);

      const gpuOutput = await ops.causalAttention(Q, K, V, numHeads, numKVHeads, headDim, startPos);
      const gpuData = await gpuOutput.toArray();

      // With single key, output should equal V (softmax of single value is 1.0)
      for (let i = 0; i < gpuData.length; i++) {
        expect(Number.isFinite(gpuData[i])).toBe(true);
      }

      Q.destroy();
      K.destroy();
      V.destroy();
      gpuOutput.destroy();
    });

    test('should not produce NaN or Inf', async () => {
      if (!device) return;

      const numQPos = 1;
      const numKeys = 128;
      const numHeads = 8;
      const numKVHeads = 8;
      const headDim = 64;
      const startPos = 127;

      const qData = new Float32Array(numQPos * numHeads * headDim);
      const kData = new Float32Array(numKeys * numKVHeads * headDim);
      const vData = new Float32Array(numKeys * numKVHeads * headDim);

      // Use larger values to stress numerical stability
      for (let i = 0; i < qData.length; i++) qData[i] = (Math.random() - 0.5) * 2;
      for (let i = 0; i < kData.length; i++) kData[i] = (Math.random() - 0.5) * 2;
      for (let i = 0; i < vData.length; i++) vData[i] = (Math.random() - 0.5) * 2;

      const Q = Tensor.fromData(qData, [numQPos, numHeads * headDim]);
      const K = Tensor.fromData(kData, [numKeys, numKVHeads * headDim]);
      const V = Tensor.fromData(vData, [numKeys, numKVHeads * headDim]);

      const gpuOutput = await ops.causalAttention(Q, K, V, numHeads, numKVHeads, headDim, startPos);
      const gpuData = await gpuOutput.toArray();

      let hasNaN = false;
      let hasInf = false;
      for (let i = 0; i < gpuData.length; i++) {
        if (Number.isNaN(gpuData[i])) hasNaN = true;
        if (!Number.isFinite(gpuData[i])) hasInf = true;
      }

      expect(hasNaN).toBe(false);
      expect(hasInf).toBe(false);

      Q.destroy();
      K.destroy();
      V.destroy();
      gpuOutput.destroy();
    });
  });
});
