/**
 * GPU Timestamp Profiler
 * Uses WebGPU timestamp queries to measure actual GPU execution time per compute pass.
 * Unlike CPU-side performance.now(), this measures real GPU kernel duration in nanoseconds.
 *
 * Uses a small query set (4096 queries = 2048 operations) that auto-cycles:
 * when capacity is reached, it pauses recording until the next resolve() call
 * which reads timestamps and resets the query index for the next batch.
 *
 * Usage:
 *   const profiler = getGPUProfiler();
 *   profiler.enable();
 *   // ... run operations (call resolveAndAccumulate() periodically) ...
 *   await profiler.resolveAndAccumulate();
 *   const report = profiler.getReport();
 *   profiler.reset();
 */

/// <reference types="@webgpu/types" />

import { getWebGPUDevice } from '../device.js';

// Node.js compatible constants (not globals in Node WebGPU)
const GPUBufferUsageFlags = {
  MAP_READ: 0x0001,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  QUERY_RESOLVE: 0x0200,
} as const;

const GPUMapModeFlags = {
  READ: 0x0001,
} as const;

export interface ProfileEntry {
  label: string;
  beginIdx: number;
  endIdx: number;
}

export interface ProfileResult {
  label: string;
  durationMs: number;
  count: number;
}

// Dawn's max query set count is 8192. Use 4096 to be safe.
const MAX_QUERIES = 4096;
const MAX_OPERATIONS = MAX_QUERIES / 2; // 2048 ops per cycle

/**
 * GPU Timestamp Profiler using WebGPU timestamp queries.
 * Tracks per-compute-pass GPU execution time with nanosecond precision.
 *
 * Query set is limited to 4096 entries (Dawn limit is 8192).
 * Results accumulate across multiple resolve cycles.
 */
export class GPUProfiler {
  private querySet: GPUQuerySet | null = null;
  private resolveBuffer: GPUBuffer | null = null;
  private stagingBuffer: GPUBuffer | null = null;
  private nextQueryIndex = 0;
  private pendingOps: ProfileEntry[] = [];  // Current cycle's ops (not yet resolved)
  private enabled = false;
  private available = false;
  private capacityPaused = false;

  // Accumulated results across all resolve cycles
  private accumulated = new Map<string, ProfileResult>();
  private totalOpsRecorded = 0;
  private totalOpsMissed = 0;

  /**
   * Initialize GPU resources for timestamp profiling.
   * Must be called after WebGPU device is initialized.
   * Returns false if timestamp-query is not supported.
   */
  init(): boolean {
    const deviceWrapper = getWebGPUDevice();
    const device = deviceWrapper.getDevice();

    if (!device.features.has('timestamp-query')) {
      console.warn('[GPUProfiler] timestamp-query not supported by this device');
      this.available = false;
      return false;
    }

    try {
      this.querySet = device.createQuerySet({
        label: 'gpu-profiler-queries',
        type: 'timestamp',
        count: MAX_QUERIES,
      });

      this.resolveBuffer = device.createBuffer({
        label: 'profiler-resolve',
        size: MAX_QUERIES * 8,
        usage: GPUBufferUsageFlags.QUERY_RESOLVE | GPUBufferUsageFlags.COPY_SRC,
      });

      this.stagingBuffer = device.createBuffer({
        label: 'profiler-staging',
        size: MAX_QUERIES * 8,
        usage: GPUBufferUsageFlags.MAP_READ | GPUBufferUsageFlags.COPY_DST,
      });
    } catch (e) {
      console.warn(`[GPUProfiler] Failed to create GPU resources: ${e}`);
      this.available = false;
      return false;
    }

    this.available = true;
    console.log(`[GPUProfiler] Initialized (${MAX_OPERATIONS} ops/cycle, ${MAX_QUERIES} query slots)`);
    return true;
  }

  isAvailable(): boolean {
    return this.available;
  }

  enable(): void {
    if (!this.available) {
      console.warn('[GPUProfiler] Cannot enable - timestamp-query not available');
      return;
    }
    this.enabled = true;
  }

  disable(): void {
    this.enabled = false;
  }

  isEnabled(): boolean {
    return this.enabled && this.available;
  }

  /**
   * Begin a profiled operation. Returns timestampWrites config for beginComputePass().
   * Returns null if profiling is not enabled or current cycle is full (call resolveAndAccumulate).
   */
  beginOperation(label: string): GPUComputePassTimestampWrites | null {
    if (!this.enabled || !this.querySet) return null;

    if (this.nextQueryIndex + 2 > MAX_QUERIES) {
      // Current cycle is full — stop recording until next resolve
      if (!this.capacityPaused) {
        this.capacityPaused = true;
      }
      this.totalOpsMissed++;
      return null;
    }

    const beginIdx = this.nextQueryIndex;
    const endIdx = this.nextQueryIndex + 1;
    this.nextQueryIndex += 2;

    this.pendingOps.push({ label, beginIdx, endIdx });

    return {
      querySet: this.querySet,
      beginningOfPassWriteIndex: beginIdx,
      endOfPassWriteIndex: endIdx,
    };
  }

  /**
   * Resolve current cycle's timestamps, accumulate results, and reset for next cycle.
   * Call this periodically (e.g., between tokens) to capture all operations.
   * Safe to call even with no pending operations.
   */
  async resolveAndAccumulate(): Promise<void> {
    if (!this.querySet || !this.resolveBuffer || !this.stagingBuffer) return;
    if (this.nextQueryIndex === 0) return;

    const device = getWebGPUDevice().getDevice();

    // Ensure all GPU work using this cycle's timestamps is complete
    await getWebGPUDevice().sync();

    // Resolve timestamps → resolve buffer → staging buffer
    const encoder = device.createCommandEncoder({ label: 'profiler-resolve' });
    encoder.resolveQuerySet(this.querySet, 0, this.nextQueryIndex, this.resolveBuffer, 0);
    encoder.copyBufferToBuffer(this.resolveBuffer, 0, this.stagingBuffer, 0, this.nextQueryIndex * 8);
    device.queue.submit([encoder.finish()]);
    await getWebGPUDevice().sync();

    // Read timestamps from staging buffer
    await this.stagingBuffer.mapAsync(GPUMapModeFlags.READ);
    const data = new BigUint64Array(this.stagingBuffer.getMappedRange());

    for (const op of this.pendingOps) {
      const begin = data[op.beginIdx];
      const end = data[op.endIdx];
      const durationNs = Number(end - begin);
      const durationMs = durationNs / 1_000_000;

      const existing = this.accumulated.get(op.label);
      if (existing) {
        existing.durationMs += durationMs;
        existing.count++;
      } else {
        this.accumulated.set(op.label, { label: op.label, durationMs, count: 1 });
      }
      this.totalOpsRecorded++;
    }

    this.stagingBuffer.unmap();

    // Reset cycle for next batch
    this.nextQueryIndex = 0;
    this.pendingOps = [];
    this.capacityPaused = false;
  }

  /**
   * Dump raw timestamp values from the current (unresolved) cycle for debugging.
   * Call after GPU sync but BEFORE resolveAndAccumulate().
   */
  async diagnose(): Promise<string> {
    if (!this.querySet || !this.resolveBuffer || !this.stagingBuffer) {
      return '[GPUProfiler] No GPU resources initialized.';
    }
    if (this.pendingOps.length === 0) {
      return '[GPUProfiler] No pending operations to diagnose.';
    }

    const device = getWebGPUDevice().getDevice();

    // Resolve into staging
    await getWebGPUDevice().sync();
    const encoder = device.createCommandEncoder({ label: 'profiler-diagnose' });
    encoder.resolveQuerySet(this.querySet, 0, this.nextQueryIndex, this.resolveBuffer, 0);
    encoder.copyBufferToBuffer(this.resolveBuffer, 0, this.stagingBuffer, 0, this.nextQueryIndex * 8);
    device.queue.submit([encoder.finish()]);
    await getWebGPUDevice().sync();

    await this.stagingBuffer.mapAsync(GPUMapModeFlags.READ);
    const data = new BigUint64Array(this.stagingBuffer.getMappedRange());

    const lines: string[] = [
      '[GPUProfiler] Diagnostic Info:',
      `  Query set count: ${MAX_QUERIES} (Dawn max: 8192)`,
      `  Query slots used this cycle: ${this.nextQueryIndex} / ${MAX_QUERIES}`,
      `  Pending operations: ${this.pendingOps.length}`,
      `  Device features timestamp-query: ${device.features.has('timestamp-query')}`,
    ];

    const samplesToShow = Math.min(10, this.pendingOps.length);
    lines.push(`  First ${samplesToShow} raw timestamp pairs:`);
    let allZero = true;
    for (let i = 0; i < samplesToShow; i++) {
      const op = this.pendingOps[i];
      const begin = data[op.beginIdx];
      const end = data[op.endIdx];
      if (begin !== 0n || end !== 0n) allZero = false;
      const diffNs = end >= begin ? Number(end - begin) : -(Number(begin - end));
      lines.push(`    [${op.label}] begin=${begin}, end=${end}, diff=${diffNs}ns (${(diffNs / 1e6).toFixed(3)}ms)`);
    }

    if (allZero) {
      lines.push('');
      lines.push('  WARNING: ALL timestamps are zero!');
      lines.push('  Dawn may be quantizing timestamps to zero on this platform.');
    }

    this.stagingBuffer.unmap();
    return lines.join('\n');
  }

  /**
   * Get accumulated results across all resolved cycles.
   */
  getResults(): Map<string, ProfileResult> {
    return new Map(this.accumulated);
  }

  /**
   * Generate a formatted profiling report from accumulated results.
   */
  getReport(): string {
    if (this.accumulated.size === 0) {
      return '[GPUProfiler] No profiling data collected.';
    }

    const sorted = Array.from(this.accumulated.values()).sort((a, b) => b.durationMs - a.durationMs);
    const totalMs = sorted.reduce((sum, r) => sum + r.durationMs, 0);

    const lines: string[] = [
      '',
      '=== GPU Timestamp Profile ===',
      `Total GPU time: ${totalMs.toFixed(2)}ms`,
      `Operations profiled: ${this.totalOpsRecorded}`,
    ];

    if (this.totalOpsMissed > 0) {
      lines.push(`Operations missed (cycle overflow): ${this.totalOpsMissed}`);
    }

    lines.push('');
    lines.push('Operation                          | Total (ms) |  Count | Avg (ms) |   % ');
    lines.push('-----------------------------------|------------|--------|----------|-----');

    for (const r of sorted) {
      const name = r.label.padEnd(35);
      const total = r.durationMs.toFixed(2).padStart(10);
      const count = String(r.count).padStart(6);
      const avg = r.count > 0 ? (r.durationMs / r.count).toFixed(2).padStart(8) : '     N/A';
      const pct = totalMs > 0 ? ((r.durationMs / totalMs) * 100).toFixed(1).padStart(4) : ' N/A';
      lines.push(`${name}| ${total} | ${count} | ${avg} | ${pct}%`);
    }

    lines.push('');
    return lines.join('\n');
  }

  /**
   * Whether the current cycle is full and needs a resolve
   */
  needsResolve(): boolean {
    return this.capacityPaused;
  }

  getOperationCount(): number {
    return this.totalOpsRecorded;
  }

  /**
   * Reset all accumulated data for a fresh profiling session.
   */
  reset(): void {
    this.nextQueryIndex = 0;
    this.pendingOps = [];
    this.accumulated.clear();
    this.totalOpsRecorded = 0;
    this.totalOpsMissed = 0;
    this.capacityPaused = false;
  }

  destroy(): void {
    this.querySet?.destroy();
    this.resolveBuffer?.destroy();
    this.stagingBuffer?.destroy();
    this.querySet = null;
    this.resolveBuffer = null;
    this.stagingBuffer = null;
    this.available = false;
    this.enabled = false;
  }
}

// Global profiler instance
let globalProfiler: GPUProfiler | null = null;

export function getGPUProfiler(): GPUProfiler {
  if (!globalProfiler) {
    globalProfiler = new GPUProfiler();
  }
  return globalProfiler;
}

export function enableGPUProfiling(): boolean {
  const profiler = getGPUProfiler();
  const ok = profiler.init();
  if (ok) {
    profiler.enable();
  }
  return ok;
}

export function disableGPUProfiling(): void {
  if (globalProfiler) {
    globalProfiler.disable();
  }
}

export function destroyGPUProfiler(): void {
  if (globalProfiler) {
    globalProfiler.destroy();
    globalProfiler = null;
  }
}
