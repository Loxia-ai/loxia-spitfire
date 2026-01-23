/**
 * Utility functions
 */

import { homedir, platform, arch, cpus, totalmem, freemem } from 'os';
import { join } from 'path';
import type { Duration, SystemInfo } from '../types/index.js';

/**
 * Get the default models directory
 */
export function getDefaultModelsPath(): string {
  const envPath = process.env['OLLAMA_MODELS'];
  if (envPath) {
    return envPath;
  }

  const home = homedir();
  return join(home, '.ollama', 'models');
}

/**
 * Get the default host
 */
export function getDefaultHost(): string {
  return process.env['OLLAMA_HOST'] || 'http://127.0.0.1:11434';
}

/**
 * Parse duration string or number to milliseconds
 */
export function parseDuration(duration: Duration | undefined): number {
  if (duration === undefined) {
    return 5 * 60 * 1000; // 5 minutes default
  }

  if (typeof duration === 'number') {
    if (duration < 0) {
      return Number.MAX_SAFE_INTEGER; // Never expire
    }
    return duration * 1000; // Assume seconds, convert to ms
  }

  // Parse duration string like "5m", "1h", "30s"
  const match = duration.match(/^(-?\d+(?:\.\d+)?)(ns|us|µs|ms|s|m|h)?$/);
  if (!match) {
    return 5 * 60 * 1000; // Default 5 minutes
  }

  const value = parseFloat(match[1]);
  const unit = match[2] || 's';

  if (value < 0) {
    return Number.MAX_SAFE_INTEGER; // Never expire
  }

  switch (unit) {
    case 'ns':
      return value / 1_000_000;
    case 'us':
    case 'µs':
      return value / 1_000;
    case 'ms':
      return value;
    case 's':
      return value * 1_000;
    case 'm':
      return value * 60 * 1_000;
    case 'h':
      return value * 60 * 60 * 1_000;
    default:
      return value * 1_000;
  }
}

/**
 * Format duration in milliseconds to human readable string
 */
export function formatDuration(ms: number): string {
  if (ms < 1000) {
    return `${ms}ms`;
  }
  if (ms < 60000) {
    return `${(ms / 1000).toFixed(1)}s`;
  }
  if (ms < 3600000) {
    return `${(ms / 60000).toFixed(1)}m`;
  }
  return `${(ms / 3600000).toFixed(1)}h`;
}

/**
 * Format bytes to human readable string
 */
export function formatBytes(bytes: number): string {
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let unitIndex = 0;
  let value = bytes;

  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex++;
  }

  return `${value.toFixed(1)} ${units[unitIndex]}`;
}

/**
 * Get system information
 */
export function getSystemInfo(): SystemInfo {
  return {
    totalMemory: totalmem(),
    freeMemory: freemem(),
    cpuCount: cpus().length,
    platform: platform(),
    arch: arch(),
  };
}

/**
 * Sleep for a given number of milliseconds
 */
export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Generate a random port number
 */
export function getRandomPort(): number {
  return Math.floor(Math.random() * (65535 - 49152) + 49152);
}

/**
 * Compute SHA256 hash of a buffer
 */
export async function sha256(data: Buffer): Promise<string> {
  const { createHash } = await import('crypto');
  return createHash('sha256').update(data).digest('hex');
}

/**
 * Check if a file exists
 */
export async function fileExists(path: string): Promise<boolean> {
  const { access, constants } = await import('fs/promises');
  try {
    await access(path, constants.F_OK);
    return true;
  } catch {
    return false;
  }
}

/**
 * Create directory if it doesn't exist
 */
export async function ensureDir(path: string): Promise<void> {
  const { mkdir } = await import('fs/promises');
  await mkdir(path, { recursive: true });
}

/**
 * Deep merge objects
 */
export function deepMerge<T extends Record<string, unknown>>(
  target: T,
  source: Partial<T>
): T {
  const result = { ...target };

  for (const key in source) {
    const sourceValue = source[key];
    const targetValue = target[key];

    if (
      sourceValue !== undefined &&
      typeof sourceValue === 'object' &&
      sourceValue !== null &&
      !Array.isArray(sourceValue) &&
      typeof targetValue === 'object' &&
      targetValue !== null &&
      !Array.isArray(targetValue)
    ) {
      (result as Record<string, unknown>)[key] = deepMerge(
        targetValue as Record<string, unknown>,
        sourceValue as Record<string, unknown>
      );
    } else if (sourceValue !== undefined) {
      (result as Record<string, unknown>)[key] = sourceValue;
    }
  }

  return result;
}

/**
 * Convert snake_case to camelCase
 */
export function snakeToCamel(str: string): string {
  return str.replace(/_([a-z])/g, (_, letter: string) => letter.toUpperCase());
}

/**
 * Convert camelCase to snake_case
 */
export function camelToSnake(str: string): string {
  return str.replace(/[A-Z]/g, (letter) => `_${letter.toLowerCase()}`);
}

/**
 * Convert object keys from snake_case to camelCase
 */
export function keysToCamel<T>(obj: unknown): T {
  if (Array.isArray(obj)) {
    return obj.map(keysToCamel) as T;
  }

  if (obj !== null && typeof obj === 'object') {
    return Object.fromEntries(
      Object.entries(obj).map(([key, value]) => [
        snakeToCamel(key),
        keysToCamel(value),
      ])
    ) as T;
  }

  return obj as T;
}

/**
 * Convert object keys from camelCase to snake_case
 */
export function keysToSnake<T>(obj: unknown): T {
  if (Array.isArray(obj)) {
    return obj.map(keysToSnake) as T;
  }

  if (obj !== null && typeof obj === 'object') {
    return Object.fromEntries(
      Object.entries(obj).map(([key, value]) => [
        camelToSnake(key),
        keysToSnake(value),
      ])
    ) as T;
  }

  return obj as T;
}
