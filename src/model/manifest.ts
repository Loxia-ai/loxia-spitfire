/**
 * Model Manifest Parser
 * Parses Ollama model manifests and manages model storage
 */

import { readFile, readdir, stat } from 'fs/promises';
import { join } from 'path';
import { homedir } from 'os';
import {
  MEDIA_TYPES,
  parseModelName,
  formatModelName,
  type ModelManifest,
  type ModelName,
  type ModelInfo,
} from '../types/index.js';
import { fileExists } from '../utils/index.js';

/**
 * Get the default models directory
 */
export function getModelsPath(): string {
  const envPath = process.env['OLLAMA_MODELS'];
  if (envPath) {
    return envPath;
  }
  return join(homedir(), '.ollama', 'models');
}

/**
 * Get the manifests directory
 */
export function getManifestsPath(modelsPath?: string): string {
  return join(modelsPath || getModelsPath(), 'manifests');
}

/**
 * Get the blobs directory
 */
export function getBlobsPath(modelsPath?: string): string {
  return join(modelsPath || getModelsPath(), 'blobs');
}

/**
 * Get the path to a manifest file for a model
 */
export function getManifestPath(name: ModelName, modelsPath?: string): string {
  const manifestsDir = getManifestsPath(modelsPath);
  return join(manifestsDir, name.registry, name.namespace, name.model, name.tag);
}

/**
 * Get the path to a blob file by digest
 */
export function getBlobPath(digest: string, modelsPath?: string): string {
  // Digest format: sha256:abc123...
  // Blob path: blobs/sha256-abc123...
  const blobName = digest.replace(':', '-');
  return join(getBlobsPath(modelsPath), blobName);
}

/**
 * Read and parse a model manifest
 */
export async function readManifest(
  name: string | ModelName,
  modelsPath?: string
): Promise<ModelManifest | null> {
  const modelName = typeof name === 'string' ? parseModelName(name) : name;
  const manifestPath = getManifestPath(modelName, modelsPath);

  if (!(await fileExists(manifestPath))) {
    return null;
  }

  const content = await readFile(manifestPath, 'utf-8');
  return JSON.parse(content) as ModelManifest;
}

/**
 * Get the model file path from a manifest
 */
export function getModelFilePath(
  manifest: ModelManifest,
  modelsPath?: string
): string | null {
  const modelLayer = manifest.layers.find(
    (l) => l.mediaType === MEDIA_TYPES.MODEL
  );
  if (!modelLayer) {
    return null;
  }
  return getBlobPath(modelLayer.digest, modelsPath);
}

/**
 * Get the projector file path from a manifest (for multimodal models)
 */
export function getProjectorPath(
  manifest: ModelManifest,
  modelsPath?: string
): string | null {
  const projectorLayer = manifest.layers.find(
    (l) => l.mediaType === MEDIA_TYPES.PROJECTOR
  );
  if (!projectorLayer) {
    return null;
  }
  return getBlobPath(projectorLayer.digest, modelsPath);
}

/**
 * Get adapter (LoRA) paths from a manifest
 */
export function getAdapterPaths(
  manifest: ModelManifest,
  modelsPath?: string
): string[] {
  return manifest.layers
    .filter((l) => l.mediaType === MEDIA_TYPES.ADAPTER)
    .map((l) => getBlobPath(l.digest, modelsPath));
}

/**
 * Get the template from a manifest
 */
export async function getTemplate(
  manifest: ModelManifest,
  modelsPath?: string
): Promise<string | null> {
  const templateLayer = manifest.layers.find(
    (l) => l.mediaType === MEDIA_TYPES.TEMPLATE
  );
  if (!templateLayer) {
    return null;
  }

  const templatePath = getBlobPath(templateLayer.digest, modelsPath);
  if (!(await fileExists(templatePath))) {
    return null;
  }

  return readFile(templatePath, 'utf-8');
}

/**
 * Get the system prompt from a manifest
 */
export async function getSystemPrompt(
  manifest: ModelManifest,
  modelsPath?: string
): Promise<string | null> {
  const systemLayer = manifest.layers.find(
    (l) => l.mediaType === MEDIA_TYPES.SYSTEM
  );
  if (!systemLayer) {
    return null;
  }

  const systemPath = getBlobPath(systemLayer.digest, modelsPath);
  if (!(await fileExists(systemPath))) {
    return null;
  }

  return readFile(systemPath, 'utf-8');
}

/**
 * Get model parameters from a manifest
 */
export async function getParameters(
  manifest: ModelManifest,
  modelsPath?: string
): Promise<Record<string, unknown> | null> {
  const paramsLayer = manifest.layers.find(
    (l) => l.mediaType === MEDIA_TYPES.PARAMS
  );
  if (!paramsLayer) {
    return null;
  }

  const paramsPath = getBlobPath(paramsLayer.digest, modelsPath);
  if (!(await fileExists(paramsPath))) {
    return null;
  }

  const content = await readFile(paramsPath, 'utf-8');
  // Parameters are stored as lines of "key value" pairs
  const params: Record<string, unknown> = {};
  for (const line of content.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    const spaceIdx = trimmed.indexOf(' ');
    if (spaceIdx === -1) continue;

    const key = trimmed.slice(0, spaceIdx);
    const value = trimmed.slice(spaceIdx + 1);

    // Try to parse as number or boolean
    if (value === 'true') {
      params[key] = true;
    } else if (value === 'false') {
      params[key] = false;
    } else {
      const num = parseFloat(value);
      params[key] = isNaN(num) ? value : num;
    }
  }

  return params;
}

/**
 * List all installed models
 */
export async function listModels(modelsPath?: string): Promise<ModelInfo[]> {
  const manifestsDir = getManifestsPath(modelsPath);

  if (!(await fileExists(manifestsDir))) {
    return [];
  }

  const models: ModelInfo[] = [];

  // Walk the manifests directory
  // Structure: manifests/<registry>/<namespace>/<model>/<tag>
  const registries = await readdir(manifestsDir).catch(() => []);

  for (const registry of registries) {
    const registryPath = join(manifestsDir, registry);
    const registryStat = await stat(registryPath).catch(() => null);
    if (!registryStat?.isDirectory()) continue;

    const namespaces = await readdir(registryPath).catch(() => []);
    for (const namespace of namespaces) {
      const namespacePath = join(registryPath, namespace);
      const namespaceStat = await stat(namespacePath).catch(() => null);
      if (!namespaceStat?.isDirectory()) continue;

      const modelNames = await readdir(namespacePath).catch(() => []);
      for (const modelName of modelNames) {
        const modelPath = join(namespacePath, modelName);
        const modelStat = await stat(modelPath).catch(() => null);
        if (!modelStat?.isDirectory()) continue;

        const tags = await readdir(modelPath).catch(() => []);
        for (const tag of tags) {
          const tagPath = join(modelPath, tag);
          const tagStat = await stat(tagPath).catch(() => null);
          if (!tagStat?.isFile()) continue;

          try {
            const name: ModelName = {
              registry,
              namespace,
              model: modelName,
              tag,
            };

            const manifest = await readManifest(name, modelsPath);
            if (!manifest) continue;

            // Get model file size
            const modelFilePath = getModelFilePath(manifest, modelsPath);
            let size = 0;
            if (modelFilePath && (await fileExists(modelFilePath))) {
              const fileStat = await stat(modelFilePath);
              size = fileStat.size;
            }

            // Get digest from model layer
            const modelLayer = manifest.layers.find(
              (l) => l.mediaType === MEDIA_TYPES.MODEL
            );

            models.push({
              name: formatModelName(name, true),
              digest: modelLayer?.digest || manifest.config.digest,
              size,
              modifiedAt: tagStat.mtime,
              manifest,
              modelPath: modelFilePath || '',
            });
          } catch {
            // Skip invalid manifests
          }
        }
      }
    }
  }

  // Sort by modified time, newest first
  models.sort((a, b) => b.modifiedAt.getTime() - a.modifiedAt.getTime());

  return models;
}

/**
 * Check if a model exists
 */
export async function modelExists(
  name: string | ModelName,
  modelsPath?: string
): Promise<boolean> {
  const manifest = await readManifest(name, modelsPath);
  if (!manifest) return false;

  const modelPath = getModelFilePath(manifest, modelsPath);
  if (!modelPath) return false;

  return fileExists(modelPath);
}

/**
 * Get full model info
 */
export async function getModelInfo(
  name: string | ModelName,
  modelsPath?: string
): Promise<ModelInfo | null> {
  const modelName = typeof name === 'string' ? parseModelName(name) : name;
  const manifest = await readManifest(modelName, modelsPath);

  if (!manifest) {
    return null;
  }

  const modelFilePath = getModelFilePath(manifest, modelsPath);
  if (!modelFilePath || !(await fileExists(modelFilePath))) {
    return null;
  }

  const fileStat = await stat(modelFilePath);
  const manifestPath = getManifestPath(modelName, modelsPath);
  const manifestStat = await stat(manifestPath);

  const modelLayer = manifest.layers.find(
    (l) => l.mediaType === MEDIA_TYPES.MODEL
  );

  return {
    name: formatModelName(modelName, true),
    digest: modelLayer?.digest || manifest.config.digest,
    size: fileStat.size,
    modifiedAt: manifestStat.mtime,
    manifest,
    modelPath: modelFilePath,
  };
}

/**
 * Calculate total size of a model (all layers)
 */
export async function getModelTotalSize(
  manifest: ModelManifest,
  modelsPath?: string
): Promise<number> {
  let total = 0;

  for (const layer of manifest.layers) {
    const blobPath = getBlobPath(layer.digest, modelsPath);
    if (await fileExists(blobPath)) {
      const fileStat = await stat(blobPath);
      total += fileStat.size;
    }
  }

  return total;
}
