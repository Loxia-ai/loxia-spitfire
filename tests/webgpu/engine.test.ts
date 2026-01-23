/**
 * WebGPU Engine Tests - Phase 8
 * Tests for the WebGPU inference engine
 */

import {
  WebGPUDevice,
} from '../../src/engine/webgpu/index.js';
import {
  WebGPUEngine,
  createWebGPUEngine,
} from '../../src/engine/webgpu-engine.js';
import {
  createEngine,
  detectBestEngine,
} from '../../src/engine/index.js';

// Skip tests if WebGPU is not available
const describeWebGPU = process.env.SKIP_WEBGPU_TESTS ? describe.skip : describe;

describe('WebGPU Engine Factory', () => {
  test('should create WebGPU engine via createEngine', () => {
    const engine = createEngine({ type: 'webgpu' });
    expect(engine).toBeInstanceOf(WebGPUEngine);
  });

  test('should create WebGPU engine via createWebGPUEngine', () => {
    const engine = createWebGPUEngine();
    expect(engine).toBeInstanceOf(WebGPUEngine);
  });

  test('should detect best engine type', async () => {
    const bestType = await detectBestEngine();
    // Should return either 'webgpu' or 'wasm'
    expect(['webgpu', 'wasm']).toContain(bestType);
  });
});

describeWebGPU('WebGPU Engine', () => {
  let engine: WebGPUEngine;
  let webgpuAvailable = false;

  beforeAll(async () => {
    webgpuAvailable = await WebGPUDevice.isAvailable();
    if (!webgpuAvailable) {
      console.log('WebGPU not available, skipping tests');
    }
  });

  beforeEach(() => {
    if (webgpuAvailable) {
      engine = createWebGPUEngine();
    }
  });

  afterEach(async () => {
    if (engine && webgpuAvailable) {
      await engine.shutdown();
    }
  });

  describe('Initialization', () => {
    test('should initialize successfully', async () => {
      if (!webgpuAvailable) return;
      await engine.init();
      expect(engine.isLoaded()).toBe(false); // No model loaded yet
    });

    test('should get GPU capabilities after init', async () => {
      if (!webgpuAvailable) return;
      await engine.init();
      const caps = engine.getCapabilities();

      if (caps) {
        expect(caps).toHaveProperty('maxBufferSize');
        expect(caps).toHaveProperty('maxComputeWorkgroupsPerDimension');
      }
    });

    test('should handle multiple init calls', async () => {
      if (!webgpuAvailable) return;
      await engine.init();
      await engine.init(); // Should not throw
      expect(engine.isLoaded()).toBe(false);
    });
  });

  describe('Event Emission', () => {
    test('should emit initialized event', async () => {
      if (!webgpuAvailable) return;
      const initPromise = new Promise<void>((resolve) => {
        engine.on('initialized', () => resolve());
      });

      await engine.init();
      await initPromise;
    });

    test('should emit shutdown event', async () => {
      if (!webgpuAvailable) return;
      await engine.init();

      const shutdownPromise = new Promise<void>((resolve) => {
        engine.on('shutdown', () => resolve());
      });

      await engine.shutdown();
      await shutdownPromise;
    });
  });

  describe('Model Loading', () => {
    test('should throw when generating without model', async () => {
      if (!webgpuAvailable) return;
      await engine.init();
      await expect(engine.generate('test')).rejects.toThrow('Model not loaded');
    });

    test('should throw when embedding without model', async () => {
      if (!webgpuAvailable) return;
      await engine.init();
      await expect(engine.embed('test')).rejects.toThrow('Model not loaded');
    });

    test('should report not loaded before loadModel', async () => {
      if (!webgpuAvailable) return;
      await engine.init();
      expect(engine.isLoaded()).toBe(false);
    });
  });

  describe('Shutdown', () => {
    test('should shutdown cleanly', async () => {
      if (!webgpuAvailable) return;
      await engine.init();
      await engine.shutdown();
      expect(engine.getCapabilities()).toBeNull();
    });

    test('should handle multiple shutdown calls', async () => {
      if (!webgpuAvailable) return;
      await engine.init();
      await engine.shutdown();
      await engine.shutdown(); // Should not throw
    });
  });
});

// Tests for engine interface compliance
describe('WebGPU Engine Interface', () => {
  test('should implement Engine interface methods', () => {
    const engine = createWebGPUEngine();

    expect(typeof engine.init).toBe('function');
    expect(typeof engine.loadModel).toBe('function');
    expect(typeof engine.generate).toBe('function');
    expect(typeof engine.generateStream).toBe('function');
    expect(typeof engine.embed).toBe('function');
    expect(typeof engine.isLoaded).toBe('function');
    expect(typeof engine.unload).toBe('function');
    expect(typeof engine.shutdown).toBe('function');
  });

  test('should extend EventEmitter', () => {
    const engine = createWebGPUEngine();

    expect(typeof engine.on).toBe('function');
    expect(typeof engine.emit).toBe('function');
    expect(typeof engine.removeListener).toBe('function');
  });
});
