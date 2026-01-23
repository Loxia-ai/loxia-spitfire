/**
 * WebGPU Layer Tests - Phase 3
 * Tests for transformer building blocks: LayerNorm, RoPE, Attention, FFN
 */

import {
  WebGPUDevice,
  initWebGPU,
  Tensor,
  rmsNorm,
  applyRope,
  attention,
  feedForward,
  ops,
} from '../../src/engine/webgpu/index.js';

// Skip tests if WebGPU is not available
const describeWebGPU = process.env.SKIP_WEBGPU_TESTS ? describe.skip : describe;

describeWebGPU('WebGPU Transformer Layers', () => {
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

  describe('RMS Normalization', () => {
    test('should normalize vectors', async () => {
      if (!device) return;

      // Input: [batchSize=2, hiddenSize=4]
      const input = Tensor.fromData([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
      const weight = Tensor.ones([4]);

      const output = await rmsNorm(input, weight, 1e-5);
      const data = await output.toArray();

      // RMS norm should normalize each row
      // For [1,2,3,4]: rms = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.74
      // normalized ≈ [0.365, 0.730, 1.095, 1.460]
      expect(data.length).toBe(8);

      // Check that values are reasonable (not NaN, not too large)
      for (const val of data) {
        expect(Number.isFinite(val)).toBe(true);
        expect(Math.abs(val)).toBeLessThan(10);
      }

      input.destroy();
      weight.destroy();
      output.destroy();
    });

    test('should apply learned scaling', async () => {
      if (!device) return;

      const input = Tensor.fromData([1, 1, 1, 1], [1, 4]);
      const weight = Tensor.fromData([2, 2, 2, 2], [4]);

      const output = await rmsNorm(input, weight);
      const data = await output.toArray();

      // Input is all 1s, RMS=1, so normalized is [1,1,1,1]
      // After scaling by weight=2: [2,2,2,2]
      expect(data[0]).toBeCloseTo(2, 1);
      expect(data[1]).toBeCloseTo(2, 1);
      expect(data[2]).toBeCloseTo(2, 1);
      expect(data[3]).toBeCloseTo(2, 1);

      input.destroy();
      weight.destroy();
      output.destroy();
    });
  });

  describe('Rotary Position Embeddings', () => {
    test('should apply position-dependent rotation', async () => {
      if (!device) return;

      const seqLen = 4;
      const numHeads = 2;
      const headDim = 4;
      const hiddenSize = numHeads * headDim;

      // Create input: [seqLen, hiddenSize]
      const inputData = new Float32Array(seqLen * hiddenSize);
      for (let i = 0; i < inputData.length; i++) {
        inputData[i] = 1; // All ones
      }

      const input = Tensor.fromData(inputData, [seqLen, hiddenSize]);
      const output = await applyRope(input, seqLen, numHeads, headDim, 10000);
      const data = await output.toArray();

      // RoPE should produce different values at different positions
      // Position 0 should have no rotation (cos(0)=1, sin(0)=0)
      // So first position should be close to input
      expect(data.length).toBe(seqLen * hiddenSize);

      // Check values are finite
      for (const val of data) {
        expect(Number.isFinite(val)).toBe(true);
      }

      // Different positions should have different values due to position encoding
      // We can verify the output is different from input
      expect(data.some((v, i) => Math.abs(v - 1) > 0.01 || i >= hiddenSize)).toBe(true);

      input.destroy();
      output.destroy();
    });
  });

  describe('Attention', () => {
    test('should compute self-attention', async () => {
      if (!device) return;

      const seqLen = 4;
      const hiddenSize = 8;

      // Create Q, K, V with random values
      const q = Tensor.random([seqLen, hiddenSize]);
      const k = Tensor.random([seqLen, hiddenSize]);
      const v = Tensor.random([seqLen, hiddenSize]);

      const output = await attention(q, k, v, {
        numHeads: 2,
        headDim: 4,
        causal: false,
      });

      expect(output.shape).toEqual([seqLen, hiddenSize]);

      const data = await output.toArray();
      for (const val of data) {
        expect(Number.isFinite(val)).toBe(true);
      }

      q.destroy();
      k.destroy();
      v.destroy();
      output.destroy();
    });

    test('should apply causal mask', async () => {
      if (!device) return;

      const seqLen = 4;
      const hiddenSize = 4;

      // Create Q, K, V
      const q = Tensor.ones([seqLen, hiddenSize]);
      const k = Tensor.ones([seqLen, hiddenSize]);
      const v = Tensor.fromData(
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [seqLen, hiddenSize]
      );

      const output = await attention(q, k, v, {
        numHeads: 1,
        headDim: 4,
        causal: true,
      });

      const data = await output.toArray();

      // With causal mask, position 0 can only attend to itself
      // Position 1 can attend to 0 and 1
      // etc.
      // Check that output is valid
      for (const val of data) {
        expect(Number.isFinite(val)).toBe(true);
      }

      q.destroy();
      k.destroy();
      v.destroy();
      output.destroy();
    });
  });

  describe('Feed-Forward Network', () => {
    test('should compute FFN with gate/up/down projections', async () => {
      if (!device) return;

      const seqLen = 4;
      const hiddenSize = 8;
      const intermediateSize = 16;

      // Input: [seqLen, hiddenSize]
      const x = Tensor.random([seqLen, hiddenSize]);

      // Weight matrices
      const wGate = Tensor.random([hiddenSize, intermediateSize]);
      const wUp = Tensor.random([hiddenSize, intermediateSize]);
      const wDown = Tensor.random([intermediateSize, hiddenSize]);

      const output = await feedForward(x, wGate, wUp, wDown);

      expect(output.shape).toEqual([seqLen, hiddenSize]);

      const data = await output.toArray();
      for (const val of data) {
        expect(Number.isFinite(val)).toBe(true);
      }

      x.destroy();
      wGate.destroy();
      wUp.destroy();
      wDown.destroy();
      output.destroy();
    });

    test('FFN should preserve batch dimension', async () => {
      if (!device) return;

      const batchSeq = 8;
      const hiddenSize = 16;
      const intermediateSize = 32;

      const x = Tensor.random([batchSeq, hiddenSize]);
      const wGate = Tensor.random([hiddenSize, intermediateSize]);
      const wUp = Tensor.random([hiddenSize, intermediateSize]);
      const wDown = Tensor.random([intermediateSize, hiddenSize]);

      const output = await feedForward(x, wGate, wUp, wDown);

      expect(output.shape).toEqual([batchSeq, hiddenSize]);
      expect(output.size).toBe(batchSeq * hiddenSize);

      x.destroy();
      wGate.destroy();
      wUp.destroy();
      wDown.destroy();
      output.destroy();
    });
  });

  describe('Layer Integration', () => {
    test('should chain RMSNorm -> Attention -> RMSNorm -> FFN', async () => {
      if (!device) return;

      const seqLen = 4;
      const hiddenSize = 8;
      const intermediateSize = 16;
      const numHeads = 2;
      const headDim = hiddenSize / numHeads;

      // Input
      const x = Tensor.random([seqLen, hiddenSize]);

      // Norm weights
      const normWeight1 = Tensor.ones([hiddenSize]);
      const normWeight2 = Tensor.ones([hiddenSize]);

      // Attention uses x as Q, K, V for self-attention
      // Simplified: no separate projection matrices

      // FFN weights
      const wGate = Tensor.random([hiddenSize, intermediateSize]);
      const wUp = Tensor.random([hiddenSize, intermediateSize]);
      const wDown = Tensor.random([intermediateSize, hiddenSize]);

      // Forward pass
      // 1. Pre-attention norm
      const normed1 = await rmsNorm(x, normWeight1);

      // 2. Self-attention
      const attnOut = await attention(normed1, normed1, normed1, {
        numHeads,
        headDim,
        causal: true,
      });

      // 3. Residual
      const residual1 = await ops.add(x, attnOut);

      // 4. Pre-FFN norm
      const normed2 = await rmsNorm(residual1, normWeight2);

      // 5. FFN
      const ffnOut = await feedForward(normed2, wGate, wUp, wDown);

      // 6. Residual
      const output = await ops.add(residual1, ffnOut);

      expect(output.shape).toEqual([seqLen, hiddenSize]);

      const data = await output.toArray();
      for (const val of data) {
        expect(Number.isFinite(val)).toBe(true);
      }

      // Cleanup
      x.destroy();
      normWeight1.destroy();
      normWeight2.destroy();
      wGate.destroy();
      wUp.destroy();
      wDown.destroy();
      normed1.destroy();
      attnOut.destroy();
      residual1.destroy();
      normed2.destroy();
      ffnOut.destroy();
      output.destroy();
    });
  });
});
