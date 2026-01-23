/**
 * WebGPU Performance Utilities
 * Monitoring, benchmarking, and optimization helpers
 */

/// <reference types="@webgpu/types" />

import { getWebGPUDevice, type GPUCapabilities } from '../device.js';

/**
 * Performance metrics for a single operation
 */
export interface PerfMetrics {
  name: string;
  startTime: number;
  endTime: number;
  duration: number;
  workgroupCount: number;
  elementsProcessed: number;
  throughput: number; // elements per second
}

/**
 * Accumulated performance statistics
 */
export interface PerfStats {
  name: string;
  count: number;
  totalTime: number;
  avgTime: number;
  minTime: number;
  maxTime: number;
  totalElements: number;
  avgThroughput: number;
}

/**
 * Performance monitor for tracking operation metrics
 */
export class PerfMonitor {
  private metrics: PerfMetrics[] = [];
  private stats: Map<string, PerfStats> = new Map();
  private enabled = true;
  private maxHistory = 1000;

  /**
   * Enable or disable performance monitoring
   */
  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
  }

  /**
   * Record a performance metric
   */
  record(
    name: string,
    startTime: number,
    endTime: number,
    workgroupCount: number,
    elementsProcessed: number
  ): PerfMetrics {
    const duration = endTime - startTime;
    const throughput = elementsProcessed / (duration / 1000);

    const metric: PerfMetrics = {
      name,
      startTime,
      endTime,
      duration,
      workgroupCount,
      elementsProcessed,
      throughput,
    };

    if (this.enabled) {
      this.metrics.push(metric);
      this.updateStats(metric);

      // Limit history
      if (this.metrics.length > this.maxHistory) {
        this.metrics.shift();
      }
    }

    return metric;
  }

  /**
   * Update accumulated statistics
   */
  private updateStats(metric: PerfMetrics): void {
    let stats = this.stats.get(metric.name);
    if (!stats) {
      stats = {
        name: metric.name,
        count: 0,
        totalTime: 0,
        avgTime: 0,
        minTime: Infinity,
        maxTime: 0,
        totalElements: 0,
        avgThroughput: 0,
      };
      this.stats.set(metric.name, stats);
    }

    stats.count++;
    stats.totalTime += metric.duration;
    stats.avgTime = stats.totalTime / stats.count;
    stats.minTime = Math.min(stats.minTime, metric.duration);
    stats.maxTime = Math.max(stats.maxTime, metric.duration);
    stats.totalElements += metric.elementsProcessed;
    stats.avgThroughput = stats.totalElements / (stats.totalTime / 1000);
  }

  /**
   * Get statistics for a specific operation
   */
  getStats(name: string): PerfStats | undefined {
    return this.stats.get(name);
  }

  /**
   * Get all statistics
   */
  getAllStats(): PerfStats[] {
    return Array.from(this.stats.values());
  }

  /**
   * Get recent metrics
   */
  getRecent(count = 10): PerfMetrics[] {
    return this.metrics.slice(-count);
  }

  /**
   * Clear all metrics and statistics
   */
  clear(): void {
    this.metrics = [];
    this.stats.clear();
  }

  /**
   * Generate a performance report
   */
  report(): string {
    const lines: string[] = ['WebGPU Performance Report', '='.repeat(50)];

    for (const stats of this.getAllStats()) {
      lines.push(`\n${stats.name}:`);
      lines.push(`  Count: ${stats.count}`);
      lines.push(`  Total time: ${stats.totalTime.toFixed(2)}ms`);
      lines.push(`  Avg time: ${stats.avgTime.toFixed(2)}ms`);
      lines.push(`  Min time: ${stats.minTime.toFixed(2)}ms`);
      lines.push(`  Max time: ${stats.maxTime.toFixed(2)}ms`);
      lines.push(`  Total elements: ${stats.totalElements.toLocaleString()}`);
      lines.push(`  Avg throughput: ${(stats.avgThroughput / 1e6).toFixed(2)}M elem/s`);
    }

    return lines.join('\n');
  }
}

// Global performance monitor
let globalPerfMonitor: PerfMonitor | null = null;

/**
 * Get the global performance monitor
 */
export function getPerfMonitor(): PerfMonitor {
  if (!globalPerfMonitor) {
    globalPerfMonitor = new PerfMonitor();
  }
  return globalPerfMonitor;
}

/**
 * Workgroup size configuration
 */
export interface WorkgroupConfig {
  size1D: number;
  size2D: [number, number];
  tileSize: number;
}

/**
 * Optimal workgroup sizes based on device capabilities
 */
export function getOptimalWorkgroupConfig(caps?: GPUCapabilities): WorkgroupConfig {
  // Default configuration
  const config: WorkgroupConfig = {
    size1D: 256,
    size2D: [16, 16],
    tileSize: 8,
  };

  if (!caps) {
    try {
      caps = getWebGPUDevice().getCapabilities();
    } catch {
      return config;
    }
  }

  // Adjust based on max workgroup size
  const maxInvocations = caps.maxComputeInvocationsPerWorkgroup || 256;

  if (maxInvocations >= 1024) {
    config.size1D = 512;
    config.size2D = [32, 32];
    config.tileSize = 16;
  } else if (maxInvocations >= 256) {
    config.size1D = 256;
    config.size2D = [16, 16];
    config.tileSize = 8;
  } else {
    config.size1D = 64;
    config.size2D = [8, 8];
    config.tileSize = 4;
  }

  return config;
}

/**
 * Calculate optimal workgroup count for 1D operations
 */
export function calculateOptimalWorkgroups1D(
  totalSize: number,
  workgroupSize?: number
): number {
  const config = getOptimalWorkgroupConfig();
  const size = workgroupSize || config.size1D;
  return Math.ceil(totalSize / size);
}

/**
 * Calculate optimal workgroup count for 2D operations (like matmul)
 */
export function calculateOptimalWorkgroups2D(
  width: number,
  height: number,
  workgroupSize?: [number, number]
): [number, number] {
  const config = getOptimalWorkgroupConfig();
  const size = workgroupSize || config.size2D;
  return [
    Math.ceil(width / size[0]),
    Math.ceil(height / size[1]),
  ];
}

/**
 * Memory usage tracker
 */
export class MemoryTracker {
  private allocations: Map<string, number> = new Map();
  private totalAllocated = 0;
  private peakUsage = 0;

  /**
   * Track a buffer allocation
   */
  allocate(label: string, size: number): void {
    this.allocations.set(label, size);
    this.totalAllocated += size;
    this.peakUsage = Math.max(this.peakUsage, this.totalAllocated);
  }

  /**
   * Track a buffer deallocation
   */
  deallocate(label: string): void {
    const size = this.allocations.get(label);
    if (size) {
      this.totalAllocated -= size;
      this.allocations.delete(label);
    }
  }

  /**
   * Get current memory usage
   */
  getCurrentUsage(): number {
    return this.totalAllocated;
  }

  /**
   * Get peak memory usage
   */
  getPeakUsage(): number {
    return this.peakUsage;
  }

  /**
   * Get all allocations
   */
  getAllocations(): Map<string, number> {
    return new Map(this.allocations);
  }

  /**
   * Reset tracking
   */
  reset(): void {
    this.allocations.clear();
    this.totalAllocated = 0;
    this.peakUsage = 0;
  }

  /**
   * Generate memory report
   */
  report(): string {
    const lines: string[] = [
      'WebGPU Memory Report',
      '='.repeat(50),
      `Current usage: ${(this.totalAllocated / 1024 / 1024).toFixed(2)} MB`,
      `Peak usage: ${(this.peakUsage / 1024 / 1024).toFixed(2)} MB`,
      '',
      'Allocations:',
    ];

    const sorted = Array.from(this.allocations.entries())
      .sort((a, b) => b[1] - a[1]);

    for (const [label, size] of sorted.slice(0, 20)) {
      lines.push(`  ${label}: ${(size / 1024).toFixed(2)} KB`);
    }

    if (sorted.length > 20) {
      lines.push(`  ... and ${sorted.length - 20} more`);
    }

    return lines.join('\n');
  }
}

// Global memory tracker
let globalMemoryTracker: MemoryTracker | null = null;

/**
 * Get the global memory tracker
 */
export function getMemoryTracker(): MemoryTracker {
  if (!globalMemoryTracker) {
    globalMemoryTracker = new MemoryTracker();
  }
  return globalMemoryTracker;
}

/**
 * Benchmark a function
 */
export async function benchmark<T>(
  _name: string,
  fn: () => Promise<T>,
  iterations = 10,
  warmupIterations = 3
): Promise<{ result: T; avgTime: number; minTime: number; maxTime: number }> {
  // _name reserved for future logging/reporting
  // Warmup
  for (let i = 0; i < warmupIterations; i++) {
    await fn();
  }

  // Benchmark
  const times: number[] = [];
  let result!: T;

  for (let i = 0; i < iterations; i++) {
    const start = performance.now();
    result = await fn();
    const end = performance.now();
    times.push(end - start);
  }

  return {
    result,
    avgTime: times.reduce((a, b) => a + b, 0) / times.length,
    minTime: Math.min(...times),
    maxTime: Math.max(...times),
  };
}
