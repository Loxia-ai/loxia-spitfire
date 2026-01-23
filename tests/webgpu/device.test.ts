/**
 * WebGPU Device Tests - Phase 1
 * Tests for GPU device initialization, buffer operations, and basic compute
 */

import {
  WebGPUDevice,
  initWebGPU,
  createStorageBuffer,
  createStorageBufferWithData,
  createUniformBufferWithData,
  readBufferFloat32,
  writeBuffer,
  createComputePipelineFromSource,
  createBindGroup,
  executeCompute,
  calculateWorkgroups,
} from '../../src/engine/webgpu/index.js';

// Skip tests if WebGPU is not available
const describeWebGPU = process.env.SKIP_WEBGPU_TESTS ? describe.skip : describe;

describeWebGPU('WebGPU Device', () => {
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

  describe('Device Initialization', () => {
    test('should detect WebGPU availability', async () => {
      const available = await WebGPUDevice.isAvailable();
      expect(typeof available).toBe('boolean');
    });

    test('should initialize device', async () => {
      if (!device) return;
      expect(device.isInitialized()).toBe(true);
    });

    test('should have valid capabilities', async () => {
      if (!device) return;
      const caps = device.getCapabilities();

      expect(caps.maxBufferSize).toBeGreaterThan(0);
      expect(caps.maxStorageBufferBindingSize).toBeGreaterThan(0);
      expect(caps.maxComputeWorkgroupSizeX).toBeGreaterThanOrEqual(256);
      expect(caps.maxComputeWorkgroupSizeY).toBeGreaterThanOrEqual(256);
      expect(caps.maxComputeWorkgroupSizeZ).toBeGreaterThanOrEqual(64);
      expect(caps.maxComputeInvocationsPerWorkgroup).toBeGreaterThanOrEqual(256);

      console.log('GPU Capabilities:', caps);
    });
  });

  describe('Buffer Operations', () => {
    test('should create storage buffer', () => {
      if (!device) return;

      const buffer = createStorageBuffer(1024, 'test-buffer');
      expect(buffer).toBeDefined();
      expect(buffer.size).toBe(1024);

      buffer.destroy();
    });

    test('should create buffer with data', () => {
      if (!device) return;

      const data = new Float32Array([1, 2, 3, 4, 5]);
      const buffer = createStorageBufferWithData(data, 'data-buffer');

      expect(buffer).toBeDefined();
      expect(buffer.size).toBe(20); // 5 * 4 bytes

      buffer.destroy();
    });

    test('should read buffer data', async () => {
      if (!device) return;

      const data = new Float32Array([1.5, 2.5, 3.5, 4.5]);
      const buffer = createStorageBufferWithData(data, 'read-test');

      const result = await readBufferFloat32(buffer, 4);

      expect(result.length).toBe(4);
      expect(result[0]).toBeCloseTo(1.5);
      expect(result[1]).toBeCloseTo(2.5);
      expect(result[2]).toBeCloseTo(3.5);
      expect(result[3]).toBeCloseTo(4.5);

      buffer.destroy();
    });

    test('should write to buffer', async () => {
      if (!device) return;

      const buffer = createStorageBuffer(16, 'write-test');
      const data = new Float32Array([10, 20, 30, 40]);

      writeBuffer(buffer, data);
      await device.sync();

      const result = await readBufferFloat32(buffer, 4);

      expect(result[0]).toBeCloseTo(10);
      expect(result[1]).toBeCloseTo(20);
      expect(result[2]).toBeCloseTo(30);
      expect(result[3]).toBeCloseTo(40);

      buffer.destroy();
    });
  });

  describe('Compute Shader Execution', () => {
    test('should execute simple compute shader (double values)', async () => {
      if (!device) return;

      const size = 1024;
      const input = new Float32Array(size);
      for (let i = 0; i < size; i++) {
        input[i] = i;
      }

      // Create shader that doubles each value
      const shader = `
        @group(0) @binding(0) var<storage, read> input: array<f32>;
        @group(0) @binding(1) var<storage, read_write> output: array<f32>;
        @group(0) @binding(2) var<uniform> params: vec4<u32>;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let idx = global_id.x;
          let size = params.x;

          if (idx >= size) {
            return;
          }

          output[idx] = input[idx] * 2.0;
        }
      `;

      const inputBuffer = createStorageBufferWithData(input, 'input');
      const outputBuffer = createStorageBuffer(size * 4, 'output');
      const paramsBuffer = createUniformBufferWithData(
        new Uint32Array([size, 0, 0, 0]),
        'params'
      );

      const pipeline = createComputePipelineFromSource(shader, {
        label: 'double',
        entryPoint: 'main',
      });

      const bindGroup = createBindGroup(pipeline, 0, [
        { binding: 0, resource: inputBuffer },
        { binding: 1, resource: outputBuffer },
        { binding: 2, resource: paramsBuffer },
      ]);

      const workgroups = calculateWorkgroups(size, 256);
      await executeCompute(pipeline, [bindGroup], [workgroups, 1, 1]);

      const result = await readBufferFloat32(outputBuffer, size);

      // Verify results
      for (let i = 0; i < 10; i++) {
        expect(result[i]).toBeCloseTo(i * 2);
      }

      // Clean up
      inputBuffer.destroy();
      outputBuffer.destroy();
      paramsBuffer.destroy();
    });

    test('should execute element-wise add', async () => {
      if (!device) return;

      const size = 512;
      const a = new Float32Array(size);
      const b = new Float32Array(size);
      for (let i = 0; i < size; i++) {
        a[i] = i;
        b[i] = i * 0.5;
      }

      const shader = `
        @group(0) @binding(0) var<storage, read> a: array<f32>;
        @group(0) @binding(1) var<storage, read> b: array<f32>;
        @group(0) @binding(2) var<storage, read_write> output: array<f32>;
        @group(0) @binding(3) var<uniform> params: vec4<u32>;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let idx = global_id.x;
          if (idx >= params.x) { return; }
          output[idx] = a[idx] + b[idx];
        }
      `;

      const aBuffer = createStorageBufferWithData(a, 'a');
      const bBuffer = createStorageBufferWithData(b, 'b');
      const outputBuffer = createStorageBuffer(size * 4, 'output');
      const paramsBuffer = createUniformBufferWithData(
        new Uint32Array([size, 0, 0, 0]),
        'params'
      );

      const pipeline = createComputePipelineFromSource(shader);
      const bindGroup = createBindGroup(pipeline, 0, [
        { binding: 0, resource: aBuffer },
        { binding: 1, resource: bBuffer },
        { binding: 2, resource: outputBuffer },
        { binding: 3, resource: paramsBuffer },
      ]);

      const workgroups = calculateWorkgroups(size, 256);
      await executeCompute(pipeline, [bindGroup], [workgroups, 1, 1]);

      const result = await readBufferFloat32(outputBuffer, size);

      for (let i = 0; i < 10; i++) {
        expect(result[i]).toBeCloseTo(i + i * 0.5);
      }

      aBuffer.destroy();
      bBuffer.destroy();
      outputBuffer.destroy();
      paramsBuffer.destroy();
    });

    test('should handle large arrays', async () => {
      if (!device) return;

      const size = 1024 * 1024; // 1M elements
      const input = new Float32Array(size);
      for (let i = 0; i < size; i++) {
        input[i] = Math.random();
      }

      const shader = `
        @group(0) @binding(0) var<storage, read> input: array<f32>;
        @group(0) @binding(1) var<storage, read_write> output: array<f32>;
        @group(0) @binding(2) var<uniform> params: vec4<u32>;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let idx = global_id.x;
          if (idx >= params.x) { return; }
          output[idx] = input[idx] * input[idx];
        }
      `;

      const inputBuffer = createStorageBufferWithData(input, 'input');
      const outputBuffer = createStorageBuffer(size * 4, 'output');
      const paramsBuffer = createUniformBufferWithData(
        new Uint32Array([size, 0, 0, 0]),
        'params'
      );

      const pipeline = createComputePipelineFromSource(shader);
      const bindGroup = createBindGroup(pipeline, 0, [
        { binding: 0, resource: inputBuffer },
        { binding: 1, resource: outputBuffer },
        { binding: 2, resource: paramsBuffer },
      ]);

      const startTime = performance.now();
      const workgroups = calculateWorkgroups(size, 256);
      await executeCompute(pipeline, [bindGroup], [workgroups, 1, 1]);
      const endTime = performance.now();

      console.log(`Large array compute time: ${endTime - startTime}ms`);

      // Verify a sample
      const result = await readBufferFloat32(outputBuffer, 100);
      for (let i = 0; i < 100; i++) {
        expect(result[i]).toBeCloseTo(input[i] * input[i], 4);
      }

      inputBuffer.destroy();
      outputBuffer.destroy();
      paramsBuffer.destroy();
    });
  });
});
