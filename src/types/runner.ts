/**
 * Runner Types
 * Types for communicating with the llama.cpp runner process
 */

import type { ModelOptions } from './api.js';

// Runner process status
export enum RunnerStatus {
  STARTING = 'starting',
  LOADING = 'loading',
  READY = 'ready',
  BUSY = 'busy',
  ERROR = 'error',
  STOPPED = 'stopped',
}

// Runner process info
export interface RunnerProcess {
  pid: number;
  port: number;
  status: RunnerStatus;
  modelPath: string;
  startedAt: Date;
  lastUsedAt: Date;
  vramSize: number;
  totalSize: number;
}

// Completion request to runner
export interface CompletionRequest {
  prompt: string;
  images?: string[]; // Base64 encoded
  grammar?: string;
  cachePrompt?: boolean;

  // Sampling parameters
  temperature?: number;
  topK?: number;
  topP?: number;
  minP?: number;
  typicalP?: number;
  repeatPenalty?: number;
  repeatLastN?: number;
  presencePenalty?: number;
  frequencyPenalty?: number;
  seed?: number;
  stop?: string[];
  nPredict?: number;
  nKeep?: number;

  // Logprobs
  logprobs?: boolean;
  topLogprobs?: number;

  // Stream
  stream?: boolean;
}

// Completion response from runner
export interface CompletionResponse {
  content: string;
  stop: boolean;
  stopReason?: 'stop' | 'length' | 'eos';

  // Metrics
  generationSettings?: Record<string, unknown>;
  model?: string;
  promptEvalCount?: number;
  promptEvalDuration?: number; // nanoseconds
  evalCount?: number;
  evalDuration?: number; // nanoseconds

  // Logprobs
  completionProbabilities?: TokenProbability[];
}

// Token probability info
export interface TokenProbability {
  content: string;
  probs: Array<{
    tok_str: string;
    prob: number;
  }>;
}

// Embedding request to runner
export interface EmbeddingRunnerRequest {
  content: string;
}

// Embedding response from runner
export interface EmbeddingRunnerResponse {
  embedding: number[];
  promptEvalCount: number;
}

// Tokenize request
export interface TokenizeRequest {
  content: string;
}

// Tokenize response
export interface TokenizeResponse {
  tokens: number[];
}

// Detokenize request
export interface DetokenizeRequest {
  tokens: number[];
}

// Detokenize response
export interface DetokenizeResponse {
  content: string;
}

// Health check response
export interface HealthResponse {
  status: 'ok' | 'loading' | 'error';
  progress?: number;
  slotsIdle?: number;
  slotsProcessing?: number;
}

// Runner load request
export interface LoadRequest {
  modelPath: string;
  projectorPath?: string;
  adapterPaths?: string[];
  options?: ModelOptions;
  systemInfo?: SystemInfo;
  gpus?: GPUInfo[];
}

// System information
export interface SystemInfo {
  totalMemory: number;
  freeMemory: number;
  cpuCount: number;
  platform: string;
  arch: string;
}

// GPU information
export interface GPUInfo {
  id: string;
  name: string;
  vendor: string;
  vram: number;
  freeVram: number;
  computeCapability?: string;
}

// Device type
export type DeviceType = 'cpu' | 'cuda' | 'metal' | 'rocm' | 'vulkan';

// Layer distribution across devices
export interface LayerAllocation {
  deviceId: string;
  deviceType: DeviceType;
  layers: number;
  vramUsed: number;
}

// Runner configuration
export interface RunnerConfig {
  // Paths
  runnerPath?: string;
  modelsPath: string;

  // Server settings
  host: string;
  port: number;

  // Resource limits
  maxLoadedModels: number;
  maxVram?: number;
  gpuOverhead: number;

  // Timeouts (ms)
  loadTimeout: number;
  requestTimeout: number;
  keepAliveTimeout: number;

  // Features
  flashAttention: boolean;
  kvCacheType: 'f16' | 'q8_0' | 'q4_0';

  // Environment
  cudaVisibleDevices?: string;
  rocmVisibleDevices?: string;
}

// Default runner configuration
export const DEFAULT_RUNNER_CONFIG: RunnerConfig = {
  modelsPath: '',
  host: '127.0.0.1',
  port: 0, // Auto-assign
  maxLoadedModels: 1,
  gpuOverhead: 0,
  loadTimeout: 300000, // 5 minutes
  requestTimeout: 120000, // 2 minutes
  keepAliveTimeout: 300000, // 5 minutes
  flashAttention: true,
  kvCacheType: 'f16',
};
