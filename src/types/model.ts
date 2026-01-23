/**
 * Model and Manifest Types
 * Types for model storage, manifests, and GGUF format
 */

// Manifest layer media types (matching Ollama's OCI-like format)
export const MEDIA_TYPES = {
  MODEL: 'application/vnd.ollama.image.model',
  PROJECTOR: 'application/vnd.ollama.image.projector',
  ADAPTER: 'application/vnd.ollama.image.adapter',
  TEMPLATE: 'application/vnd.ollama.image.template',
  PARAMS: 'application/vnd.ollama.image.params',
  SYSTEM: 'application/vnd.ollama.image.system',
  LICENSE: 'application/vnd.ollama.image.license',
  MESSAGES: 'application/vnd.ollama.image.messages',
} as const;

export type MediaType = (typeof MEDIA_TYPES)[keyof typeof MEDIA_TYPES];

// Manifest layer
export interface ManifestLayer {
  mediaType: string;
  digest: string;
  size?: number;
}

// Manifest config
export interface ManifestConfig {
  digest: string;
  size?: number;
}

// Model manifest (OCI-like format)
export interface ModelManifest {
  schemaVersion?: number;
  mediaType?: string;
  config: ManifestConfig;
  layers: ManifestLayer[];
}

// Model name components
export interface ModelName {
  registry: string;
  namespace: string;
  model: string;
  tag: string;
}

// Model info stored in manifest
export interface ModelInfo {
  name: string;
  digest: string;
  size: number;
  modifiedAt: Date;
  manifest: ModelManifest;
  modelPath: string;
}

// GGUF file types
export enum GGUFValueType {
  UINT8 = 0,
  INT8 = 1,
  UINT16 = 2,
  INT16 = 3,
  UINT32 = 4,
  INT32 = 5,
  FLOAT32 = 6,
  BOOL = 7,
  STRING = 8,
  ARRAY = 9,
  UINT64 = 10,
  INT64 = 11,
  FLOAT64 = 12,
}

// GGUF tensor types (quantization formats)
export enum GGMLType {
  F32 = 0,
  F16 = 1,
  Q4_0 = 2,
  Q4_1 = 3,
  Q5_0 = 6,
  Q5_1 = 7,
  Q8_0 = 8,
  Q8_1 = 9,
  Q2_K = 10,
  Q3_K = 11,
  Q4_K = 12,
  Q5_K = 13,
  Q6_K = 14,
  Q8_K = 15,
  IQ2_XXS = 16,
  IQ2_XS = 17,
  IQ3_XXS = 18,
  IQ1_S = 19,
  IQ4_NL = 20,
  IQ3_S = 21,
  IQ2_S = 22,
  IQ4_XS = 23,
  I8 = 24,
  I16 = 25,
  I32 = 26,
  I64 = 27,
  F64 = 28,
  BF16 = 30,
}

// GGUF header
export interface GGUFHeader {
  magic: number;
  version: number;
  tensorCount: bigint;
  metadataKVCount: bigint;
}

// GGUF metadata key-value pair
export interface GGUFMetadata {
  [key: string]: unknown;
}

// GGUF tensor info
export interface GGUFTensorInfo {
  name: string;
  nDimensions: number;
  dimensions: bigint[];
  type: GGMLType;
  offset: bigint;
}

// Parsed GGUF file info
export interface GGUFFile {
  header: GGUFHeader;
  metadata: GGUFMetadata;
  tensors: GGUFTensorInfo[];
  tensorDataOffset: bigint;
}

// Common GGUF metadata keys
export const GGUF_KEYS = {
  GENERAL_ARCHITECTURE: 'general.architecture',
  GENERAL_NAME: 'general.name',
  GENERAL_AUTHOR: 'general.author',
  GENERAL_VERSION: 'general.version',
  GENERAL_DESCRIPTION: 'general.description',
  GENERAL_FILE_TYPE: 'general.file_type',
  GENERAL_QUANTIZATION_VERSION: 'general.quantization_version',

  // Context
  CONTEXT_LENGTH: '.context_length', // prefixed with architecture
  EMBEDDING_LENGTH: '.embedding_length',
  BLOCK_COUNT: '.block_count',
  FEED_FORWARD_LENGTH: '.feed_forward_length',
  ATTENTION_HEAD_COUNT: '.attention.head_count',
  ATTENTION_HEAD_COUNT_KV: '.attention.head_count_kv',

  // Tokenizer
  TOKENIZER_MODEL: 'tokenizer.ggml.model',
  TOKENIZER_LIST: 'tokenizer.ggml.tokens',
  TOKENIZER_TOKEN_TYPE: 'tokenizer.ggml.token_type',
  TOKENIZER_SCORES: 'tokenizer.ggml.scores',
  TOKENIZER_MERGES: 'tokenizer.ggml.merges',
  TOKENIZER_BOS_ID: 'tokenizer.ggml.bos_token_id',
  TOKENIZER_EOS_ID: 'tokenizer.ggml.eos_token_id',
  TOKENIZER_PAD_ID: 'tokenizer.ggml.padding_token_id',
  TOKENIZER_UNK_ID: 'tokenizer.ggml.unknown_token_id',

  // Chat template
  CHAT_TEMPLATE: 'tokenizer.chat_template',
} as const;

// Model architecture info extracted from GGUF
export interface ModelArchitecture {
  architecture: string;
  name?: string;
  contextLength: number;
  embeddingLength: number;
  blockCount: number;
  headCount: number;
  headCountKV: number;
  vocabSize: number;
  quantization: string;
}

// Parse model name into components
export function parseModelName(name: string): ModelName {
  // Default values
  let registry = 'registry.ollama.ai';
  let namespace = 'library';
  let model = name;
  let tag = 'latest';

  // Check for tag
  const tagIdx = name.lastIndexOf(':');
  if (tagIdx !== -1) {
    tag = name.substring(tagIdx + 1);
    name = name.substring(0, tagIdx);
  }

  // Check for registry/namespace/model format
  const parts = name.split('/');
  if (parts.length === 3) {
    registry = parts[0];
    namespace = parts[1];
    model = parts[2];
  } else if (parts.length === 2) {
    // Could be namespace/model or registry/model
    if (parts[0].includes('.')) {
      registry = parts[0];
      model = parts[1];
    } else {
      namespace = parts[0];
      model = parts[1];
    }
  } else {
    model = parts[0];
  }

  return { registry, namespace, model, tag };
}

// Format model name from components
export function formatModelName(name: ModelName, includeTag = true): string {
  let result = '';

  if (name.registry !== 'registry.ollama.ai') {
    result += name.registry + '/';
  }

  if (name.namespace !== 'library') {
    result += name.namespace + '/';
  }

  result += name.model;

  if (includeTag && name.tag !== 'latest') {
    result += ':' + name.tag;
  }

  return result;
}
