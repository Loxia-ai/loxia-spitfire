/**
 * WebGPU Tensor Tests - Phase 2
 * Tests for tensor operations: matmul, elementwise, reductions, softmax
 */

import {
  WebGPUDevice,
  initWebGPU,
  Tensor,
  ops,
} from '../../src/engine/webgpu/index.js';

// Skip tests if WebGPU is not available
const describeWebGPU = process.env.SKIP_WEBGPU_TESTS ? describe.skip : describe;

describeWebGPU('WebGPU Tensor Operations', () => {
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

  describe('Tensor Creation', () => {
    test('should create tensor from data', async () => {
      if (!device) return;

      const data = [1, 2, 3, 4, 5, 6];
      const t = Tensor.fromData(data, [2, 3]);

      expect(t.shape).toEqual([2, 3]);
      expect(t.size).toBe(6);
      expect(t.ndim).toBe(2);

      const result = await t.toArray();
      expect(Array.from(result)).toEqual(data);

      t.destroy();
    });

    test('should create zeros tensor', async () => {
      if (!device) return;

      const t = Tensor.zeros([3, 4]);
      const result = await t.toArray();

      expect(t.shape).toEqual([3, 4]);
      expect(result.every((v) => v === 0)).toBe(true);

      t.destroy();
    });

    test('should create ones tensor', async () => {
      if (!device) return;

      const t = Tensor.ones([2, 2]);
      const result = await t.toArray();

      expect(Array.from(result)).toEqual([1, 1, 1, 1]);

      t.destroy();
    });

    test('should reshape tensor', async () => {
      if (!device) return;

      const t = Tensor.fromData([1, 2, 3, 4, 5, 6], [2, 3]);
      const reshaped = t.reshape([3, 2]);

      expect(reshaped.shape).toEqual([3, 2]);

      const result = await reshaped.toArray();
      expect(Array.from(result)).toEqual([1, 2, 3, 4, 5, 6]);

      t.destroy();
    });
  });

  describe('Element-wise Operations', () => {
    test('relu should zero out negatives', async () => {
      if (!device) return;

      const t = Tensor.fromData([-2, -1, 0, 1, 2], [5]);
      const result = await ops.relu(t);
      const data = await result.toArray();

      expect(Array.from(data)).toEqual([0, 0, 0, 1, 2]);

      t.destroy();
      result.destroy();
    });

    test('silu should apply silu activation', async () => {
      if (!device) return;

      const t = Tensor.fromData([0, 1, 2], [3]);
      const result = await ops.silu(t);
      const data = await result.toArray();

      // silu(x) = x * sigmoid(x)
      expect(data[0]).toBeCloseTo(0, 4);
      expect(data[1]).toBeCloseTo(0.731, 2); // 1 * sigmoid(1)
      expect(data[2]).toBeCloseTo(1.762, 2); // 2 * sigmoid(2)

      t.destroy();
      result.destroy();
    });

    test('exp should compute exponential', async () => {
      if (!device) return;

      const t = Tensor.fromData([0, 1, 2], [3]);
      const result = await ops.exp(t);
      const data = await result.toArray();

      expect(data[0]).toBeCloseTo(1, 4);
      expect(data[1]).toBeCloseTo(Math.E, 4);
      expect(data[2]).toBeCloseTo(Math.E * Math.E, 4);

      t.destroy();
      result.destroy();
    });

    test('tanh should compute hyperbolic tangent', async () => {
      if (!device) return;

      const t = Tensor.fromData([-1, 0, 1], [3]);
      const result = await ops.tanh(t);
      const data = await result.toArray();

      expect(data[0]).toBeCloseTo(Math.tanh(-1), 4);
      expect(data[1]).toBeCloseTo(0, 4);
      expect(data[2]).toBeCloseTo(Math.tanh(1), 4);

      t.destroy();
      result.destroy();
    });
  });

  describe('Binary Operations', () => {
    test('add should add tensors element-wise', async () => {
      if (!device) return;

      const a = Tensor.fromData([1, 2, 3], [3]);
      const b = Tensor.fromData([4, 5, 6], [3]);
      const result = await ops.add(a, b);
      const data = await result.toArray();

      expect(Array.from(data)).toEqual([5, 7, 9]);

      a.destroy();
      b.destroy();
      result.destroy();
    });

    test('mul should multiply tensors element-wise', async () => {
      if (!device) return;

      const a = Tensor.fromData([1, 2, 3], [3]);
      const b = Tensor.fromData([2, 3, 4], [3]);
      const result = await ops.mul(a, b);
      const data = await result.toArray();

      expect(Array.from(data)).toEqual([2, 6, 12]);

      a.destroy();
      b.destroy();
      result.destroy();
    });

    test('sub should subtract tensors element-wise', async () => {
      if (!device) return;

      const a = Tensor.fromData([5, 6, 7], [3]);
      const b = Tensor.fromData([1, 2, 3], [3]);
      const result = await ops.sub(a, b);
      const data = await result.toArray();

      expect(Array.from(data)).toEqual([4, 4, 4]);

      a.destroy();
      b.destroy();
      result.destroy();
    });
  });

  describe('Scalar Operations', () => {
    test('addScalar should add constant to tensor', async () => {
      if (!device) return;

      const t = Tensor.fromData([1, 2, 3], [3]);
      const result = await ops.addScalar(t, 10);
      const data = await result.toArray();

      expect(Array.from(data)).toEqual([11, 12, 13]);

      t.destroy();
      result.destroy();
    });

    test('mulScalar should multiply tensor by constant', async () => {
      if (!device) return;

      const t = Tensor.fromData([1, 2, 3], [3]);
      const result = await ops.mulScalar(t, 2);
      const data = await result.toArray();

      expect(Array.from(data)).toEqual([2, 4, 6]);

      t.destroy();
      result.destroy();
    });
  });

  describe('Matrix Multiplication', () => {
    test('matmul should multiply 2D matrices', async () => {
      if (!device) return;

      // A = [[1, 2], [3, 4]]
      // B = [[5, 6], [7, 8]]
      // C = [[19, 22], [43, 50]]
      const a = Tensor.fromData([1, 2, 3, 4], [2, 2]);
      const b = Tensor.fromData([5, 6, 7, 8], [2, 2]);
      const c = await ops.matmul(a, b);
      const data = await c.toArray();

      expect(c.shape).toEqual([2, 2]);
      expect(data[0]).toBeCloseTo(19, 4);
      expect(data[1]).toBeCloseTo(22, 4);
      expect(data[2]).toBeCloseTo(43, 4);
      expect(data[3]).toBeCloseTo(50, 4);

      a.destroy();
      b.destroy();
      c.destroy();
    });

    test('matmul should handle non-square matrices', async () => {
      if (!device) return;

      // A = [[1, 2, 3], [4, 5, 6]] (2x3)
      // B = [[7, 8], [9, 10], [11, 12]] (3x2)
      // C = [[58, 64], [139, 154]] (2x2)
      const a = Tensor.fromData([1, 2, 3, 4, 5, 6], [2, 3]);
      const b = Tensor.fromData([7, 8, 9, 10, 11, 12], [3, 2]);
      const c = await ops.matmul(a, b);
      const data = await c.toArray();

      expect(c.shape).toEqual([2, 2]);
      expect(data[0]).toBeCloseTo(58, 4);
      expect(data[1]).toBeCloseTo(64, 4);
      expect(data[2]).toBeCloseTo(139, 4);
      expect(data[3]).toBeCloseTo(154, 4);

      a.destroy();
      b.destroy();
      c.destroy();
    });

    test('matmul should handle larger matrices', async () => {
      if (!device) return;

      const size = 64;
      const a = Tensor.random([size, size]);
      const b = Tensor.random([size, size]);

      const startTime = performance.now();
      const c = await ops.matmul(a, b);
      const endTime = performance.now();

      console.log(`${size}x${size} matmul time: ${endTime - startTime}ms`);

      expect(c.shape).toEqual([size, size]);

      // Verify a sample of results against CPU computation
      const aData = await a.toArray();
      const bData = await b.toArray();
      const cData = await c.toArray();

      // Check element [0, 0]
      let expected = 0;
      for (let k = 0; k < size; k++) {
        expected += aData[k] * bData[k * size];
      }
      expect(cData[0]).toBeCloseTo(expected, 3);

      a.destroy();
      b.destroy();
      c.destroy();
    });
  });

  describe('Reduction Operations', () => {
    test('sum should compute total sum', async () => {
      if (!device) return;

      const t = Tensor.fromData([1, 2, 3, 4, 5], [5]);
      const result = await ops.sum(t);

      expect(result).toBeCloseTo(15, 4);

      t.destroy();
    });

    test('max should find maximum', async () => {
      if (!device) return;

      const t = Tensor.fromData([3, 1, 4, 1, 5, 9, 2, 6], [8]);
      const result = await ops.max(t);

      expect(result).toBeCloseTo(9, 4);

      t.destroy();
    });

    test('mean should compute average', async () => {
      if (!device) return;

      const t = Tensor.fromData([2, 4, 6, 8], [4]);
      const result = await ops.mean(t);

      expect(result).toBeCloseTo(5, 4);

      t.destroy();
    });

    test('sum should work on large arrays', async () => {
      if (!device) return;

      const size = 10000;
      const data = new Float32Array(size);
      for (let i = 0; i < size; i++) {
        data[i] = 1;
      }

      const t = Tensor.fromData(data, [size]);
      const result = await ops.sum(t);

      expect(result).toBeCloseTo(size, 1);

      t.destroy();
    });
  });

  describe('Softmax', () => {
    test('softmax should produce probability distribution', async () => {
      if (!device) return;

      const t = Tensor.fromData([1, 2, 3], [3]);
      const result = await ops.softmax(t);
      const data = await result.toArray();

      // Probabilities should sum to 1
      const sum = data[0] + data[1] + data[2];
      expect(sum).toBeCloseTo(1, 4);

      // Larger values should have higher probability
      expect(data[2]).toBeGreaterThan(data[1]);
      expect(data[1]).toBeGreaterThan(data[0]);

      t.destroy();
      result.destroy();
    });

    test('softmax should be numerically stable with large values', async () => {
      if (!device) return;

      const t = Tensor.fromData([1000, 1001, 1002], [3]);
      const result = await ops.softmax(t);
      const data = await result.toArray();

      // Should not produce NaN or Inf
      expect(Number.isFinite(data[0])).toBe(true);
      expect(Number.isFinite(data[1])).toBe(true);
      expect(Number.isFinite(data[2])).toBe(true);

      const sum = data[0] + data[1] + data[2];
      expect(sum).toBeCloseTo(1, 4);

      t.destroy();
      result.destroy();
    });

    test('softmax should work on batches', async () => {
      if (!device) return;

      // 2 batches of 4 elements each
      const t = Tensor.fromData([1, 2, 3, 4, 10, 20, 30, 40], [2, 4]);
      const result = await ops.softmax(t);
      const data = await result.toArray();

      // Check that each row sums to 1
      const sum1 = data[0] + data[1] + data[2] + data[3];
      const sum2 = data[4] + data[5] + data[6] + data[7];

      expect(sum1).toBeCloseTo(1, 4);
      expect(sum2).toBeCloseTo(1, 4);

      t.destroy();
      result.destroy();
    });
  });

  describe('Performance Benchmarks', () => {
    test('benchmark matmul sizes', async () => {
      if (!device) return;

      const sizes = [32, 64, 128, 256];

      for (const size of sizes) {
        const a = Tensor.random([size, size]);
        const b = Tensor.random([size, size]);

        // Warmup
        let c = await ops.matmul(a, b);
        c.destroy();

        // Timed run
        const startTime = performance.now();
        const iterations = 5;
        for (let i = 0; i < iterations; i++) {
          c = await ops.matmul(a, b);
          c.destroy();
        }
        const endTime = performance.now();

        const avgTime = (endTime - startTime) / iterations;
        const gflops = (2 * size * size * size) / (avgTime * 1e6);

        console.log(`matmul ${size}x${size}: ${avgTime.toFixed(2)}ms, ${gflops.toFixed(2)} GFLOPS`);

        a.destroy();
        b.destroy();
      }
    });
  });

  describe('Edge Cases', () => {
    test('should handle single element tensor', async () => {
      if (!device) return;

      const t = Tensor.fromData([42], [1]);
      const result = await t.toArray();

      expect(result[0]).toBe(42);
      expect(t.size).toBe(1);
      expect(t.shape).toEqual([1]);

      t.destroy();
    });

    test('should handle empty-ish shapes', async () => {
      if (!device) return;

      const t = Tensor.ones([1, 1, 1]);
      expect(t.size).toBe(1);
      expect(t.ndim).toBe(3);

      t.destroy();
    });

    test('should handle negative values in relu', async () => {
      if (!device) return;

      const t = Tensor.fromData([-100, -1, -0.001, 0, 0.001, 1, 100], [7]);
      const result = await ops.relu(t);
      const data = await result.toArray();

      expect(data[0]).toBe(0);
      expect(data[1]).toBe(0);
      expect(data[2]).toBe(0);
      expect(data[3]).toBe(0);
      expect(data[4]).toBeCloseTo(0.001, 5);
      expect(data[5]).toBe(1);
      expect(data[6]).toBe(100);

      t.destroy();
      result.destroy();
    });

    test('should handle identity matrix in matmul', async () => {
      if (!device) return;

      // Create identity matrix
      const identity = Tensor.fromData([1, 0, 0, 0, 1, 0, 0, 0, 1], [3, 3]);
      const a = Tensor.fromData([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3]);

      const result = await ops.matmul(a, identity);
      const resultData = await result.toArray();
      const aData = await a.toArray();

      // A * I = A
      for (let i = 0; i < 9; i++) {
        expect(resultData[i]).toBeCloseTo(aData[i], 4);
      }

      identity.destroy();
      a.destroy();
      result.destroy();
    });

    test('should handle very small values in softmax', async () => {
      if (!device) return;

      const t = Tensor.fromData([-100, -99, -98], [3]);
      const result = await ops.softmax(t);
      const data = await result.toArray();

      // Should not produce NaN
      expect(Number.isFinite(data[0])).toBe(true);
      expect(Number.isFinite(data[1])).toBe(true);
      expect(Number.isFinite(data[2])).toBe(true);

      // Sum should still be 1
      const sum = data[0] + data[1] + data[2];
      expect(sum).toBeCloseTo(1, 4);

      t.destroy();
      result.destroy();
    });

    test('should handle equal values in softmax', async () => {
      if (!device) return;

      const t = Tensor.fromData([5, 5, 5, 5], [4]);
      const result = await ops.softmax(t);
      const data = await result.toArray();

      // All should be equal to 0.25
      for (let i = 0; i < 4; i++) {
        expect(data[i]).toBeCloseTo(0.25, 4);
      }

      t.destroy();
      result.destroy();
    });
  });

  describe('GELU Activation', () => {
    test('gelu should compute correctly', async () => {
      if (!device) return;

      const t = Tensor.fromData([-2, -1, 0, 1, 2], [5]);
      const result = await ops.gelu(t);
      const data = await result.toArray();

      // GELU(-2) ≈ -0.045, GELU(-1) ≈ -0.159
      // GELU(0) = 0, GELU(1) ≈ 0.841, GELU(2) ≈ 1.955
      expect(data[0]).toBeCloseTo(-0.045, 1);
      expect(data[1]).toBeCloseTo(-0.159, 1);
      expect(data[2]).toBeCloseTo(0, 4);
      expect(data[3]).toBeCloseTo(0.841, 1);
      expect(data[4]).toBeCloseTo(1.955, 1);

      t.destroy();
      result.destroy();
    });
  });

  describe('Chained Operations', () => {
    test('should chain multiple operations', async () => {
      if (!device) return;

      // (a + b) * 2
      const a = Tensor.fromData([1, 2, 3], [3]);
      const b = Tensor.fromData([4, 5, 6], [3]);

      const sum = await ops.add(a, b);
      const result = await ops.mulScalar(sum, 2);
      const data = await result.toArray();

      expect(Array.from(data)).toEqual([10, 14, 18]);

      a.destroy();
      b.destroy();
      sum.destroy();
      result.destroy();
    });

    test('should chain matmul operations', async () => {
      if (!device) return;

      // (A * B) * C
      const a = Tensor.fromData([1, 2, 3, 4], [2, 2]);
      const b = Tensor.fromData([5, 6, 7, 8], [2, 2]);
      const c = Tensor.fromData([1, 0, 0, 1], [2, 2]); // Identity

      const ab = await ops.matmul(a, b);
      const result = await ops.matmul(ab, c);
      const abData = await ab.toArray();
      const resultData = await result.toArray();

      // Result should equal ab since c is identity
      for (let i = 0; i < 4; i++) {
        expect(resultData[i]).toBeCloseTo(abData[i], 4);
      }

      a.destroy();
      b.destroy();
      c.destroy();
      ab.destroy();
      result.destroy();
    });
  });

  describe('3D Tensor Operations', () => {
    test('should create and manipulate 3D tensors', async () => {
      if (!device) return;

      const t = Tensor.fromData(
        [1, 2, 3, 4, 5, 6, 7, 8],
        [2, 2, 2]
      );

      expect(t.shape).toEqual([2, 2, 2]);
      expect(t.ndim).toBe(3);
      expect(t.size).toBe(8);

      const data = await t.toArray();
      expect(data.length).toBe(8);

      t.destroy();
    });

    test('should reshape between dimensions', async () => {
      if (!device) return;

      const t = Tensor.fromData(
        Array.from({ length: 24 }, (_, i) => i),
        [2, 3, 4]
      );

      const reshaped1 = t.reshape([4, 6]);
      expect(reshaped1.shape).toEqual([4, 6]);

      const reshaped2 = t.reshape([24]);
      expect(reshaped2.shape).toEqual([24]);

      const reshaped3 = t.reshape([1, 24]);
      expect(reshaped3.shape).toEqual([1, 24]);

      t.destroy();
    });
  });

  describe('Division Operations', () => {
    test('div should divide tensors element-wise', async () => {
      if (!device) return;

      const a = Tensor.fromData([10, 20, 30], [3]);
      const b = Tensor.fromData([2, 4, 5], [3]);
      const result = await ops.div(a, b);
      const data = await result.toArray();

      expect(data[0]).toBeCloseTo(5, 4);
      expect(data[1]).toBeCloseTo(5, 4);
      expect(data[2]).toBeCloseTo(6, 4);

      a.destroy();
      b.destroy();
      result.destroy();
    });
  });

  describe('Square Root and Inverse', () => {
    test('sqrt should compute square root', async () => {
      if (!device) return;

      const t = Tensor.fromData([1, 4, 9, 16, 25], [5]);
      const result = await ops.sqrt(t);
      const data = await result.toArray();

      expect(data[0]).toBeCloseTo(1, 4);
      expect(data[1]).toBeCloseTo(2, 4);
      expect(data[2]).toBeCloseTo(3, 4);
      expect(data[3]).toBeCloseTo(4, 4);
      expect(data[4]).toBeCloseTo(5, 4);

      t.destroy();
      result.destroy();
    });

    test('rsqrt should compute inverse square root', async () => {
      if (!device) return;

      const t = Tensor.fromData([1, 4, 9, 16], [4]);
      const result = await ops.rsqrt(t);
      const data = await result.toArray();

      expect(data[0]).toBeCloseTo(1, 4);
      expect(data[1]).toBeCloseTo(0.5, 4);
      expect(data[2]).toBeCloseTo(1/3, 4);
      expect(data[3]).toBeCloseTo(0.25, 4);

      t.destroy();
      result.destroy();
    });
  });

  describe('Negation', () => {
    test('neg should negate values', async () => {
      if (!device) return;

      const t = Tensor.fromData([-2, -1, 0, 1, 2], [5]);
      const result = await ops.neg(t);
      const data = await result.toArray();

      expect(Array.from(data)).toEqual([2, 1, 0, -1, -2]);

      t.destroy();
      result.destroy();
    });
  });
});
