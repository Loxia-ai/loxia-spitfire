/**
 * Spitfire API Types
 * TypeScript definitions matching Ollama's API specification
 */

// Base types
export type ImageData = string; // Base64 encoded image data

export interface StatusError {
  statusCode: number;
  status: string;
  error: string;
}

export interface AuthorizationError {
  statusCode: number;
  status: string;
  signinUrl: string;
}

// Tool types for function calling
export interface ToolCallFunctionArguments {
  [key: string]: unknown;
}

export interface ToolCallFunction {
  index: number;
  name: string;
  arguments: ToolCallFunctionArguments;
}

export interface ToolCall {
  id?: string;
  function: ToolCallFunction;
}

export interface ToolProperty {
  anyOf?: ToolProperty[];
  type?: string | string[];
  items?: unknown;
  description?: string;
  enum?: unknown[];
}

export interface ToolFunctionParameters {
  type: string;
  $defs?: unknown;
  items?: unknown;
  required?: string[];
  properties: Record<string, ToolProperty>;
}

export interface ToolFunction {
  name: string;
  description?: string;
  parameters: ToolFunctionParameters;
}

export interface Tool {
  type: string;
  items?: unknown;
  function: ToolFunction;
}

// Message type for chat
export interface Message {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string;
  thinking?: string;
  images?: ImageData[];
  toolCalls?: ToolCall[];
  toolName?: string;
  toolCallId?: string;
}

// ThinkValue can be boolean or "high" | "medium" | "low"
export type ThinkValue = boolean | 'high' | 'medium' | 'low';

// Duration can be a number (seconds) or string like "5m"
export type Duration = number | string;

// Runner options set when model is loaded
export interface RunnerOptions {
  numCtx?: number;
  numBatch?: number;
  numGpu?: number;
  mainGpu?: number;
  useMmap?: boolean;
  numThread?: number;
}

// Model inference options
export interface ModelOptions extends RunnerOptions {
  numKeep?: number;
  seed?: number;
  numPredict?: number;
  topK?: number;
  topP?: number;
  minP?: number;
  typicalP?: number;
  repeatLastN?: number;
  temperature?: number;
  repeatPenalty?: number;
  presencePenalty?: number;
  frequencyPenalty?: number;
  stop?: string[];
}

// Token log probability
export interface TokenLogprob {
  token: string;
  logprob: number;
  bytes?: number[];
}

export interface Logprob extends TokenLogprob {
  topLogprobs?: TokenLogprob[];
}

// Metrics returned in responses
export interface Metrics {
  totalDuration?: number;
  loadDuration?: number;
  promptEvalCount?: number;
  promptEvalDuration?: number;
  evalCount?: number;
  evalDuration?: number;
}

// Debug info for template rendering
export interface DebugInfo {
  renderedTemplate: string;
  imageCount?: number;
}

// ============= Request Types =============

export interface GenerateRequest {
  model: string;
  prompt: string;
  suffix?: string;
  system?: string;
  template?: string;
  context?: number[];
  stream?: boolean;
  raw?: boolean;
  format?: unknown; // Can be "json" or JSON schema
  keepAlive?: Duration;
  images?: ImageData[];
  options?: ModelOptions;
  think?: ThinkValue;
  truncate?: boolean;
  shift?: boolean;
  logprobs?: boolean;
  topLogprobs?: number;
}

export interface ChatRequest {
  model: string;
  messages: Message[];
  stream?: boolean;
  format?: unknown;
  keepAlive?: Duration;
  tools?: Tool[];
  options?: ModelOptions;
  think?: ThinkValue;
  truncate?: boolean;
  shift?: boolean;
  logprobs?: boolean;
  topLogprobs?: number;
}

export interface EmbedRequest {
  model: string;
  input: string | string[];
  keepAlive?: Duration;
  truncate?: boolean;
  dimensions?: number;
  options?: ModelOptions;
}

export interface EmbeddingRequest {
  model: string;
  prompt: string;
  keepAlive?: Duration;
  options?: ModelOptions;
}

export interface CreateRequest {
  model: string;
  stream?: boolean;
  quantize?: string;
  from?: string;
  remoteHost?: string;
  files?: Record<string, string>;
  adapters?: Record<string, string>;
  template?: string;
  license?: string | string[];
  system?: string;
  parameters?: Record<string, unknown>;
  messages?: Message[];
  renderer?: string;
  parser?: string;
  info?: Record<string, unknown>;
}

export interface DeleteRequest {
  model: string;
}

export interface ShowRequest {
  model: string;
  system?: string;
  template?: string;
  verbose?: boolean;
  options?: ModelOptions;
}

export interface CopyRequest {
  source: string;
  destination: string;
}

export interface PullRequest {
  model: string;
  stream?: boolean;
}

export interface PushRequest {
  model: string;
  insecure?: boolean;
  stream?: boolean;
}

// ============= Response Types =============

export interface GenerateResponse extends Metrics {
  model: string;
  remoteModel?: string;
  remoteHost?: string;
  createdAt: string;
  response: string;
  thinking?: string;
  done: boolean;
  doneReason?: string;
  context?: number[];
  toolCalls?: ToolCall[];
  logprobs?: Logprob[];
}

export interface ChatResponse extends Metrics {
  model: string;
  remoteModel?: string;
  remoteHost?: string;
  createdAt: string;
  message: Message;
  done: boolean;
  doneReason?: string;
  logprobs?: Logprob[];
}

export interface EmbedResponse {
  model: string;
  embeddings: number[][];
  totalDuration?: number;
  loadDuration?: number;
  promptEvalCount?: number;
}

export interface EmbeddingResponse {
  embedding: number[];
}

export interface ProgressResponse {
  status: string;
  digest?: string;
  total?: number;
  completed?: number;
}

export interface ModelDetails {
  parentModel: string;
  format: string;
  family: string;
  families: string[];
  parameterSize: string;
  quantizationLevel: string;
}

export interface Tensor {
  name: string;
  type: string;
  shape: number[];
}

export interface ShowResponse {
  license?: string;
  modelfile?: string;
  parameters?: string;
  template?: string;
  system?: string;
  renderer?: string;
  parser?: string;
  details?: ModelDetails;
  messages?: Message[];
  remoteModel?: string;
  remoteHost?: string;
  modelInfo?: Record<string, unknown>;
  projectorInfo?: Record<string, unknown>;
  tensors?: Tensor[];
  capabilities?: string[];
  modifiedAt?: string;
}

export interface ListModelResponse {
  name: string;
  model: string;
  remoteModel?: string;
  remoteHost?: string;
  modifiedAt: string;
  size: number;
  digest: string;
  details?: ModelDetails;
}

export interface ListResponse {
  models: ListModelResponse[];
}

export interface ProcessModelResponse {
  name: string;
  model: string;
  size: number;
  digest: string;
  details?: ModelDetails;
  expiresAt: string;
  sizeVram: number;
  contextLength: number;
}

export interface ProcessResponse {
  models: ProcessModelResponse[];
}

// ============= Default Options =============

export const DEFAULT_OPTIONS: ModelOptions = {
  numPredict: -1,
  numKeep: 4,
  temperature: 0.8,
  topK: 40,
  topP: 0.9,
  typicalP: 1.0,
  repeatLastN: 64,
  repeatPenalty: 1.1,
  presencePenalty: 0.0,
  frequencyPenalty: 0.0,
  seed: -1,
  numCtx: 2048,
  numBatch: 512,
  numGpu: -1,
  numThread: 0,
};

export const DEFAULT_KEEP_ALIVE: Duration = '5m';
