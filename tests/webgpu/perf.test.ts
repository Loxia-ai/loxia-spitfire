/**
 * WebGPU Performance Utilities Tests - Phase 9
 * Tests for performance monitoring, benchmarking, and optimization
 */

import {
  WebGPUDevice,
  initWebGPU,
  PerfMonitor,
  getPerfMonitor,
  MemoryTracker,
  getMemoryTracker,
  getOptimalWorkgroupConfig,
  calculateOptimalWorkgroups1D,
  calculateOptimalWorkgroups2D,
  benchmark,
} from '../../src/engine/webgpu/index.js';

// Skip GPU-specific tests if WebGPU is not available
const describeWebGPU = process.env.SKIP_WEBGPU_TESTS ? describe.skip : describe;

describe('Performance Monitor', () => {
  let monitor: PerfMonitor;

  beforeEach(() => {
    monitor = new PerfMonitor();
  });

  test('should record metrics', () => {
    const metric = monitor.record('test_op', 0, 100, 10, 1000);

    expect(metric.name).toBe('test_op');
    expect(metric.duration).toBe(100);
    expect(metric.workgroupCount).toBe(10);
    expect(metric.elementsProcessed).toBe(1000);
    expect(metric.throughput).toBe(10000); // 1000 elements / 0.1 seconds
  });

  test('should accumulate statistics', () => {
    monitor.record('test_op', 0, 100, 10, 1000);
    monitor.record('test_op', 100, 250, 10, 1000);
    monitor.record('test_op', 250, 300, 10, 1000);

    const stats = monitor.getStats('test_op');
    expect(stats).toBeDefined();
    expect(stats!.count).toBe(3);
    expect(stats!.totalTime).toBe(300); // 100 + 150 + 50
    expect(stats!.minTime).toBe(50);
    expect(stats!.maxTime).toBe(150);
  });

  test('should track multiple operations', () => {
    monitor.record('op_a', 0, 100, 10, 1000);
    monitor.record('op_b', 0, 200, 20, 2000);

    const allStats = monitor.getAllStats();
    expect(allStats.length).toBe(2);
  });

  test('should get recent metrics', () => {
    for (let i = 0; i < 20; i++) {
      monitor.record(`op_${i}`, 0, 10, 1, 100);
    }

    const recent = monitor.getRecent(5);
    expect(recent.length).toBe(5);
    expect(recent[4].name).toBe('op_19');
  });

  test('should clear metrics', () => {
    monitor.record('test_op', 0, 100, 10, 1000);
    monitor.clear();

    expect(monitor.getAllStats().length).toBe(0);
    expect(monitor.getRecent().length).toBe(0);
  });

  test('should generate report', () => {
    monitor.record('test_op', 0, 100, 10, 1000);
    const report = monitor.report();

    expect(report).toContain('WebGPU Performance Report');
    expect(report).toContain('test_op');
    expect(report).toContain('Count: 1');
  });

  test('should disable monitoring', () => {
    monitor.setEnabled(false);
    monitor.record('test_op', 0, 100, 10, 1000);

    expect(monitor.getAllStats().length).toBe(0);
  });

  test('should use global monitor', () => {
    const global1 = getPerfMonitor();
    const global2 = getPerfMonitor();

    expect(global1).toBe(global2);
  });
});

describe('Memory Tracker', () => {
  let tracker: MemoryTracker;

  beforeEach(() => {
    tracker = new MemoryTracker();
  });

  test('should track allocations', () => {
    tracker.allocate('buffer_a', 1024);
    tracker.allocate('buffer_b', 2048);

    expect(tracker.getCurrentUsage()).toBe(3072);
  });

  test('should track deallocations', () => {
    tracker.allocate('buffer_a', 1024);
    tracker.allocate('buffer_b', 2048);
    tracker.deallocate('buffer_a');

    expect(tracker.getCurrentUsage()).toBe(2048);
  });

  test('should track peak usage', () => {
    tracker.allocate('buffer_a', 1024);
    tracker.allocate('buffer_b', 2048);
    tracker.deallocate('buffer_a');
    tracker.allocate('buffer_c', 512);

    expect(tracker.getPeakUsage()).toBe(3072);
    expect(tracker.getCurrentUsage()).toBe(2560);
  });

  test('should list allocations', () => {
    tracker.allocate('buffer_a', 1024);
    tracker.allocate('buffer_b', 2048);

    const allocations = tracker.getAllocations();
    expect(allocations.get('buffer_a')).toBe(1024);
    expect(allocations.get('buffer_b')).toBe(2048);
  });

  test('should reset tracking', () => {
    tracker.allocate('buffer_a', 1024);
    tracker.reset();

    expect(tracker.getCurrentUsage()).toBe(0);
    expect(tracker.getPeakUsage()).toBe(0);
    expect(tracker.getAllocations().size).toBe(0);
  });

  test('should generate report', () => {
    tracker.allocate('buffer_a', 1024 * 1024);
    const report = tracker.report();

    expect(report).toContain('WebGPU Memory Report');
    expect(report).toContain('buffer_a');
  });

  test('should use global tracker', () => {
    const global1 = getMemoryTracker();
    const global2 = getMemoryTracker();

    expect(global1).toBe(global2);
  });
});

describe('Workgroup Configuration', () => {
  test('should return default config without device', () => {
    const config = getOptimalWorkgroupConfig();

    expect(config.size1D).toBeGreaterThan(0);
    expect(config.size2D[0]).toBeGreaterThan(0);
    expect(config.size2D[1]).toBeGreaterThan(0);
    expect(config.tileSize).toBeGreaterThan(0);
  });

  test('should calculate 1D workgroups', () => {
    const count = calculateOptimalWorkgroups1D(1000);
    expect(count).toBeGreaterThan(0);
    expect(count).toBeLessThanOrEqual(1000);
  });

  test('should calculate 2D workgroups', () => {
    const [x, y] = calculateOptimalWorkgroups2D(100, 200);
    expect(x).toBeGreaterThan(0);
    expect(y).toBeGreaterThan(0);
    expect(x).toBeLessThanOrEqual(100);
    expect(y).toBeLessThanOrEqual(200);
  });

  test('should handle small sizes', () => {
    const count1D = calculateOptimalWorkgroups1D(1);
    expect(count1D).toBe(1);

    const [x, y] = calculateOptimalWorkgroups2D(1, 1);
    expect(x).toBe(1);
    expect(y).toBe(1);
  });

  test('should handle exact multiples', () => {
    const config = getOptimalWorkgroupConfig();
    const count = calculateOptimalWorkgroups1D(config.size1D * 4);
    expect(count).toBe(4);
  });
});

describe('Benchmark Utility', () => {
  test('should benchmark async function', async () => {
    let callCount = 0;
    const result = await benchmark(
      'test_fn',
      async () => {
        callCount++;
        return 42;
      },
      5, // iterations
      2  // warmup
    );

    expect(result.result).toBe(42);
    expect(callCount).toBe(7); // 2 warmup + 5 iterations
    expect(result.avgTime).toBeGreaterThanOrEqual(0);
    expect(result.minTime).toBeLessThanOrEqual(result.avgTime);
    expect(result.maxTime).toBeGreaterThanOrEqual(result.avgTime);
  });

  test('should handle slow functions', async () => {
    const result = await benchmark(
      'slow_fn',
      async () => {
        await new Promise((resolve) => setTimeout(resolve, 10));
        return 'done';
      },
      3,
      1
    );

    expect(result.result).toBe('done');
    expect(result.avgTime).toBeGreaterThanOrEqual(10);
  });
});

describeWebGPU('Performance with GPU', () => {
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

  test('should get config based on device capabilities', async () => {
    if (!device) return;

    const caps = device.getCapabilities();
    const config = getOptimalWorkgroupConfig(caps);

    expect(config.size1D).toBeGreaterThan(0);
    expect(config.size1D).toBeLessThanOrEqual(caps.maxComputeInvocationsPerWorkgroup);
  });
});
