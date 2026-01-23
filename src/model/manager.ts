/**
 * Model Manager
 * High-level interface for managing models
 */

import { EventEmitter } from 'events';
import {
  parseModelName,
  formatModelName,
  type ModelName,
  type ModelManifest,
  type ModelInfo,
  type ModelArchitecture,
  type ModelDetails,
} from '../types/index.js';
import {
  getModelsPath,
  readManifest,
  listModels,
  modelExists,
  getModelInfo,
  getModelFilePath,
  getProjectorPath,
  getAdapterPaths,
  getTemplate,
  getSystemPrompt,
  getParameters,
  getModelTotalSize,
} from './manifest.js';
import { parseGGUF, extractArchitecture, getChatTemplate, isGGUF } from './gguf.js';

export interface ModelManagerOptions {
  modelsPath?: string;
}

export interface LoadedModelInfo {
  name: string;
  manifest: ModelManifest;
  modelPath: string;
  projectorPath?: string;
  adapterPaths: string[];
  template?: string;
  systemPrompt?: string;
  parameters: Record<string, unknown>;
  architecture?: ModelArchitecture;
}

export class ModelManager extends EventEmitter {
  private modelsPath: string;

  constructor(options: ModelManagerOptions = {}) {
    super();
    this.modelsPath = options.modelsPath || getModelsPath();
  }

  /**
   * Get the models directory path
   */
  getModelsPath(): string {
    return this.modelsPath;
  }

  /**
   * List all installed models
   */
  async list(): Promise<ModelInfo[]> {
    return listModels(this.modelsPath);
  }

  /**
   * Check if a model exists
   */
  async exists(name: string): Promise<boolean> {
    return modelExists(name, this.modelsPath);
  }

  /**
   * Get model info by name
   */
  async getInfo(name: string): Promise<ModelInfo | null> {
    return getModelInfo(name, this.modelsPath);
  }

  /**
   * Get detailed model information for display
   */
  async show(name: string): Promise<{
    license?: string;
    modelfile?: string;
    parameters?: string;
    template?: string;
    system?: string;
    details?: ModelDetails;
    modelInfo?: Record<string, unknown>;
  } | null> {
    const modelName = parseModelName(name);
    const manifest = await readManifest(modelName, this.modelsPath);

    if (!manifest) {
      return null;
    }

    const modelPath = getModelFilePath(manifest, this.modelsPath);
    if (!modelPath) {
      return null;
    }

    // Get template and system prompt
    const template = await getTemplate(manifest, this.modelsPath);
    const system = await getSystemPrompt(manifest, this.modelsPath);
    const parameters = await getParameters(manifest, this.modelsPath);

    // Parse GGUF for model info
    let modelInfo: Record<string, unknown> = {};
    let details: ModelDetails | undefined;

    try {
      const gguf = await parseGGUF(modelPath);
      const arch = extractArchitecture(gguf);

      modelInfo = {
        architecture: arch.architecture,
        context_length: arch.contextLength,
        embedding_length: arch.embeddingLength,
        block_count: arch.blockCount,
        attention_head_count: arch.headCount,
        attention_head_count_kv: arch.headCountKV,
        ...gguf.metadata,
      };

      details = {
        parentModel: '',
        format: 'gguf',
        family: arch.architecture,
        families: [arch.architecture],
        parameterSize: this.estimateParamSize(arch),
        quantizationLevel: arch.quantization,
      };
    } catch {
      // GGUF parsing failed, use basic info
    }

    // Format parameters as string
    let parametersStr = '';
    if (parameters) {
      parametersStr = Object.entries(parameters)
        .map(([k, v]) => `${k} ${v}`)
        .join('\n');
    }

    return {
      template: template || undefined,
      system: system || undefined,
      parameters: parametersStr || undefined,
      details,
      modelInfo,
    };
  }

  /**
   * Estimate parameter size string from architecture
   */
  private estimateParamSize(arch: ModelArchitecture): string {
    // Rough estimate based on embedding size and block count
    const embeddingSize = arch.embeddingLength || 4096;
    const blockCount = arch.blockCount || 32;
    const vocabSize = arch.vocabSize || 32000;

    // Very rough estimate: params = embedding * embedding * blockCount * 4 + vocab * embedding * 2
    const params =
      embeddingSize * embeddingSize * blockCount * 4 +
      vocabSize * embeddingSize * 2;

    if (params >= 70e9) return '70B+';
    if (params >= 30e9) return '34B';
    if (params >= 10e9) return '13B';
    if (params >= 6e9) return '7B';
    if (params >= 2e9) return '3B';
    if (params >= 1e9) return '1B';
    return '<1B';
  }

  /**
   * Load all information needed to run a model
   */
  async load(name: string): Promise<LoadedModelInfo | null> {
    const modelName = parseModelName(name);
    const manifest = await readManifest(modelName, this.modelsPath);

    if (!manifest) {
      this.emit('error', { name, error: 'Model not found' });
      return null;
    }

    const modelPath = getModelFilePath(manifest, this.modelsPath);
    if (!modelPath) {
      this.emit('error', { name, error: 'Model file not found in manifest' });
      return null;
    }

    // Check if the model file actually exists
    const { access, constants } = await import('fs/promises');
    try {
      await access(modelPath, constants.R_OK);
    } catch {
      this.emit('error', {
        name,
        error: `Model file not accessible: ${modelPath}`,
      });
      return null;
    }

    this.emit('loading', { name, modelPath });

    // Get optional components
    const projectorPath = getProjectorPath(manifest, this.modelsPath);
    const adapterPaths = getAdapterPaths(manifest, this.modelsPath);
    const template = await getTemplate(manifest, this.modelsPath);
    const systemPrompt = await getSystemPrompt(manifest, this.modelsPath);
    const parameters = (await getParameters(manifest, this.modelsPath)) || {};

    // Parse GGUF for architecture info
    let architecture: ModelArchitecture | undefined;
    try {
      const gguf = await parseGGUF(modelPath, 100); // Limit array size for speed
      architecture = extractArchitecture(gguf);

      // Also check for chat template in GGUF if not in manifest
      if (!template) {
        const ggufTemplate = getChatTemplate(gguf);
        if (ggufTemplate) {
          (
            await import('./manifest.js')
          ).getTemplate; // Just to ensure module is loaded
        }
      }
    } catch (error) {
      this.emit('warning', {
        name,
        warning: `Failed to parse GGUF metadata: ${error}`,
      });
    }

    const result: LoadedModelInfo = {
      name: formatModelName(modelName, true),
      manifest,
      modelPath,
      projectorPath: projectorPath || undefined,
      adapterPaths,
      template: template || undefined,
      systemPrompt: systemPrompt || undefined,
      parameters,
      architecture,
    };

    this.emit('loaded', { name, modelPath, architecture });

    return result;
  }

  /**
   * Get model architecture from GGUF file
   */
  async getArchitecture(
    nameOrPath: string
  ): Promise<ModelArchitecture | null> {
    let modelPath: string;

    // Check if it's a direct path or a model name
    if (
      nameOrPath.includes('/') &&
      !nameOrPath.includes(':') &&
      (await isGGUF(nameOrPath).catch(() => false))
    ) {
      modelPath = nameOrPath;
    } else {
      const info = await this.getInfo(nameOrPath);
      if (!info) return null;
      modelPath = info.modelPath;
    }

    try {
      const gguf = await parseGGUF(modelPath, 100);
      return extractArchitecture(gguf);
    } catch {
      return null;
    }
  }

  /**
   * Get total size of all model files
   */
  async getTotalSize(name: string): Promise<number> {
    const modelName = parseModelName(name);
    const manifest = await readManifest(modelName, this.modelsPath);

    if (!manifest) {
      return 0;
    }

    return getModelTotalSize(manifest, this.modelsPath);
  }

  /**
   * Parse a model name into components
   */
  parseModelName(name: string): ModelName {
    return parseModelName(name);
  }

  /**
   * Format a model name from components
   */
  formatModelName(name: ModelName, includeTag = true): string {
    return formatModelName(name, includeTag);
  }
}
