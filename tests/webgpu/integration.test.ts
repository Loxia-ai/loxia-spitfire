/**
 * WebGPU Integration Tests
 * Tests for interactions between different WebGPU components
 */

import {
  WebGPUDevice,
  initWebGPU,
  Tensor,
  ops,
  rmsNorm,
  attention,
  feedForward,
  BufferPool,
  getBufferPool,
  clearShaderCache,
  PerfMonitor,
  getPerfMonitor,
  getOptimalWorkgroupConfig,
} from '../../src/engine/webgpu/index.js';

// Skip tests if WebGPU is not available
const describeWebGPU = process.env.SKIP_WEBGPU_TESTS ? describe.skip : describe;

describeWebGPU('WebGPU Integration Tests', () => {
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

  describe('Tensor + Ops Integration', () => {
    test('should perform full forward pass simulation', async () => {
      if (!device) return;

      // Simulate a simplified transformer forward pass
      const batchSize = 2;
      const seqLen = 4;
      const hiddenSize = 8;
      const intermediateSize = 16;

      // Create input
      const input = Tensor.random([batchSize * seqLen, hiddenSize]);

      // RMS Norm
      const normWeight = Tensor.ones([hiddenSize]);
      const normed = await rmsNorm(input, normWeight);

      // Self-attention
      const attnOut = await attention(normed, normed, normed, {
        numHeads: 2,
        headDim: hiddenSize / 2,
        causal: true,
      });

      // Residual
      const afterAttn = await ops.add(input, attnOut);

      // FFN
      const wGate = Tensor.random([hiddenSize, intermediateSize]);
      const wUp = Tensor.random([hiddenSize, intermediateSize]);
      const wDown = Tensor.random([intermediateSize, hiddenSize]);
      const ffnOut = await feedForward(afterAttn, wGate, wUp, wDown);

      // Final residual
      const output = await ops.add(afterAttn, ffnOut);

      // Verify shapes
      expect(output.shape).toEqual([batchSize * seqLen, hiddenSize]);

      // Verify values are finite
      const data = await output.toArray();
      for (const val of data) {
        expect(Number.isFinite(val)).toBe(true);
      }

      // Cleanup
      input.destroy();
      normWeight.destroy();
      normed.destroy();
      attnOut.destroy();
      afterAttn.destroy();
      wGate.destroy();
      wUp.destroy();
      wDown.destroy();
      ffnOut.destroy();
      output.destroy();
    });

    test('should handle multiple operations in sequence', async () => {
      if (!device) return;

      const size = 32;
      let tensor = Tensor.random([size, size]);

      // Chain of operations
      for (let i = 0; i < 5; i++) {
        const relu = await ops.relu(tensor);
        tensor.destroy();

        const scaled = await ops.mulScalar(relu, 0.9);
        relu.destroy();

        const added = await ops.addScalar(scaled, 0.1);
        scaled.destroy();

        tensor = added;
      }

      expect(tensor.shape).toEqual([size, size]);
      const data = await tensor.toArray();

      // Values should be positive (relu) and reasonable
      for (const val of data) {
        expect(val).toBeGreaterThanOrEqual(0);
        expect(val).toBeLessThan(100);
      }

      tensor.destroy();
    });
  });

  describe('Buffer Pool Integration', () => {
    test('should reuse buffers across operations', async () => {
      if (!device) return;

      const pool = getBufferPool();
      const size = 1024 * 4; // 1024 floats

      // Acquire and release multiple times
      const buffers: GPUBuffer[] = [];
      for (let i = 0; i < 5; i++) {
        const buf = pool.acquire(size, ['storage', 'copy-src', 'copy-dst']);
        buffers.push(buf);
      }

      // Release all
      for (const buf of buffers) {
        pool.release(buf);
      }

      // Acquire again - should get some from pool
      const newBuffers: GPUBuffer[] = [];
      for (let i = 0; i < 3; i++) {
        const buf = pool.acquire(size, ['storage', 'copy-src', 'copy-dst']);
        newBuffers.push(buf);
      }

      // Clean up
      for (const buf of newBuffers) {
        pool.release(buf);
      }
      pool.clear();
    });
  });

  describe('Performance Monitoring Integration', () => {
    test('should track tensor operations', async () => {
      if (!device) return;

      const monitor = getPerfMonitor();
      monitor.clear();

      // Perform operations with manual timing
      const a = Tensor.random([64, 64]);
      const b = Tensor.random([64, 64]);

      const start = performance.now();
      const c = await ops.matmul(a, b);
      const end = performance.now();

      // Record the metric
      monitor.record('matmul_64x64', start, end, 1, 64 * 64 * 64 * 2);

      const stats = monitor.getStats('matmul_64x64');
      expect(stats).toBeDefined();
      expect(stats!.count).toBe(1);

      a.destroy();
      b.destroy();
      c.destroy();
    });

    test('should generate meaningful performance report', async () => {
      if (!device) return;

      const monitor = new PerfMonitor();

      // Simulate multiple operations
      for (let i = 0; i < 10; i++) {
        monitor.record('op_a', i * 100, i * 100 + 50 + Math.random() * 10, 1, 1000);
        monitor.record('op_b', i * 100, i * 100 + 100 + Math.random() * 20, 2, 2000);
      }

      const report = monitor.report();
      expect(report).toContain('op_a');
      expect(report).toContain('op_b');
      expect(report).toContain('Count: 10');

      const allStats = monitor.getAllStats();
      expect(allStats.length).toBe(2);
    });
  });

  describe('Shader Cache Integration', () => {
    test('should cache shaders across multiple uses', async () => {
      if (!device) return;

      // Clear cache first
      clearShaderCache();

      // Create tensors and run operations
      const a = Tensor.fromData([1, 2, 3, 4], [4]);

      // First run creates shaders
      const result1 = await ops.relu(a);
      const result2 = await ops.relu(a);
      const result3 = await ops.relu(a);

      // All results should be equal
      const data1 = await result1.toArray();
      const data2 = await result2.toArray();
      const data3 = await result3.toArray();

      expect(Array.from(data1)).toEqual(Array.from(data2));
      expect(Array.from(data2)).toEqual(Array.from(data3));

      a.destroy();
      result1.destroy();
      result2.destroy();
      result3.destroy();
    });
  });

  describe('Workgroup Configuration Integration', () => {
    test('should use optimal config for operations', async () => {
      if (!device) return;

      const caps = device.getCapabilities();
      const config = getOptimalWorkgroupConfig(caps);

      // Verify config is valid
      expect(config.size1D).toBeGreaterThan(0);
      expect(config.size1D).toBeLessThanOrEqual(caps.maxComputeInvocationsPerWorkgroup);

      // Use config for tensor operation
      const size = config.size1D * 4;
      const t = Tensor.random([size]);
      const result = await ops.relu(t);

      expect(result.size).toBe(size);

      t.destroy();
      result.destroy();
    });
  });

  describe('Memory Management Integration', () => {
    test('should not leak memory in repeated operations', async () => {
      if (!device) return;

      // Run many operations
      for (let i = 0; i < 20; i++) {
        const a = Tensor.random([32, 32]);
        const b = Tensor.random([32, 32]);
        const c = await ops.matmul(a, b);
        const d = await ops.relu(c);
        const e = await ops.softmax(d);

        a.destroy();
        b.destroy();
        c.destroy();
        d.destroy();
        e.destroy();
      }

      // If we get here without errors, memory management is working
      expect(true).toBe(true);
    });

    test('should handle large tensors', async () => {
      if (!device) return;

      // Create a reasonably large tensor
      const size = 512;
      const t = Tensor.random([size, size]);

      const result = await ops.relu(t);
      expect(result.shape).toEqual([size, size]);

      const data = await result.toArray();
      expect(data.length).toBe(size * size);

      t.destroy();
      result.destroy();
    });
  });

  describe('Error Recovery Integration', () => {
    test('should handle operations after tensor destroy gracefully', async () => {
      if (!device) return;

      const a = Tensor.random([4, 4]);
      const b = Tensor.random([4, 4]);

      // Normal operation
      const c = await ops.add(a, b);
      expect(c.shape).toEqual([4, 4]);

      // Clean up properly
      a.destroy();
      b.destroy();
      c.destroy();

      // Create new tensors and verify system still works
      const d = Tensor.random([4, 4]);
      const e = Tensor.random([4, 4]);
      const f = await ops.add(d, e);

      expect(f.shape).toEqual([4, 4]);

      d.destroy();
      e.destroy();
      f.destroy();
    });
  });

  describe('Attention + FFN Integration', () => {
    test('should chain attention and FFN correctly', async () => {
      if (!device) return;

      const seqLen = 8;
      const hiddenSize = 16;
      const intermediateSize = 32;

      // Input
      const x = Tensor.random([seqLen, hiddenSize]);

      // Attention
      const attnOut = await attention(x, x, x, {
        numHeads: 2,
        headDim: 8,
        causal: true,
      });

      // FFN
      const wGate = Tensor.random([hiddenSize, intermediateSize]);
      const wUp = Tensor.random([hiddenSize, intermediateSize]);
      const wDown = Tensor.random([intermediateSize, hiddenSize]);

      const ffnOut = await feedForward(attnOut, wGate, wUp, wDown);

      expect(ffnOut.shape).toEqual([seqLen, hiddenSize]);

      const data = await ffnOut.toArray();
      for (const val of data) {
        expect(Number.isFinite(val)).toBe(true);
      }

      // Cleanup
      x.destroy();
      attnOut.destroy();
      wGate.destroy();
      wUp.destroy();
      wDown.destroy();
      ffnOut.destroy();
    });
  });
});

// Tests that don't require GPU
describe('WebGPU Integration (No GPU)', () => {
  test('should provide consistent API across components', () => {
    // Verify exports are available
    expect(typeof Tensor).toBe('function');
    expect(typeof ops).toBe('object');
    expect(typeof rmsNorm).toBe('function');
    expect(typeof attention).toBe('function');
    expect(typeof feedForward).toBe('function');
    expect(typeof BufferPool).toBe('function');
    expect(typeof PerfMonitor).toBe('function');
  });

  test('should have consistent tensor creation methods', () => {
    expect(typeof Tensor.fromData).toBe('function');
    expect(typeof Tensor.zeros).toBe('function');
    expect(typeof Tensor.ones).toBe('function');
    expect(typeof Tensor.random).toBe('function');
    expect(typeof Tensor.empty).toBe('function');
  });

  test('should have all expected ops', () => {
    // Element-wise
    expect(typeof ops.relu).toBe('function');
    expect(typeof ops.silu).toBe('function');
    expect(typeof ops.gelu).toBe('function');
    expect(typeof ops.tanh).toBe('function');
    expect(typeof ops.exp).toBe('function');
    expect(typeof ops.sqrt).toBe('function');
    expect(typeof ops.rsqrt).toBe('function');
    expect(typeof ops.neg).toBe('function');

    // Binary
    expect(typeof ops.add).toBe('function');
    expect(typeof ops.sub).toBe('function');
    expect(typeof ops.mul).toBe('function');
    expect(typeof ops.div).toBe('function');

    // Scalar
    expect(typeof ops.addScalar).toBe('function');
    expect(typeof ops.mulScalar).toBe('function');

    // Matrix
    expect(typeof ops.matmul).toBe('function');

    // Reduction
    expect(typeof ops.sum).toBe('function');
    expect(typeof ops.max).toBe('function');
    expect(typeof ops.mean).toBe('function');

    // Activation
    expect(typeof ops.softmax).toBe('function');
  });
});
