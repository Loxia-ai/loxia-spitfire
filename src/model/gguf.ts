/**
 * GGUF File Parser
 * Parses GGUF model file headers to extract metadata
 */

import { open, type FileHandle } from 'fs/promises';
import {
  GGUFValueType,
  GGMLType,
  GGUF_KEYS,
  type GGUFHeader,
  type GGUFMetadata,
  type GGUFTensorInfo,
  type GGUFFile,
  type ModelArchitecture,
} from '../types/index.js';

// GGUF magic number
const GGUF_MAGIC = 0x46554747; // 'GGUF' in little-endian

// Helper to read from a file handle
class BufferedReader {
  private handle: FileHandle;
  private buffer: Buffer;
  private bufferOffset: number;
  private bufferLength: number;
  private fileOffset: bigint;
  private readonly BUFFER_SIZE = 64 * 1024; // 64KB buffer

  constructor(handle: FileHandle) {
    this.handle = handle;
    this.buffer = Buffer.alloc(this.BUFFER_SIZE);
    this.bufferOffset = 0;
    this.bufferLength = 0;
    this.fileOffset = 0n;
  }

  async readBytes(count: number): Promise<Buffer> {
    const result = Buffer.alloc(count);
    let resultOffset = 0;

    while (resultOffset < count) {
      if (this.bufferOffset >= this.bufferLength) {
        // Refill buffer
        const { bytesRead } = await this.handle.read(
          this.buffer,
          0,
          this.BUFFER_SIZE,
          Number(this.fileOffset)
        );
        if (bytesRead === 0) {
          throw new Error('Unexpected end of file');
        }
        this.bufferOffset = 0;
        this.bufferLength = bytesRead;
        this.fileOffset += BigInt(bytesRead);
      }

      const available = this.bufferLength - this.bufferOffset;
      const toCopy = Math.min(available, count - resultOffset);
      this.buffer.copy(
        result,
        resultOffset,
        this.bufferOffset,
        this.bufferOffset + toCopy
      );
      this.bufferOffset += toCopy;
      resultOffset += toCopy;
    }

    return result;
  }

  async readUint8(): Promise<number> {
    const buf = await this.readBytes(1);
    return buf.readUInt8(0);
  }

  async readUint16(): Promise<number> {
    const buf = await this.readBytes(2);
    return buf.readUInt16LE(0);
  }

  async readUint32(): Promise<number> {
    const buf = await this.readBytes(4);
    return buf.readUInt32LE(0);
  }

  async readUint64(): Promise<bigint> {
    const buf = await this.readBytes(8);
    return buf.readBigUInt64LE(0);
  }

  async readInt8(): Promise<number> {
    const buf = await this.readBytes(1);
    return buf.readInt8(0);
  }

  async readInt16(): Promise<number> {
    const buf = await this.readBytes(2);
    return buf.readInt16LE(0);
  }

  async readInt32(): Promise<number> {
    const buf = await this.readBytes(4);
    return buf.readInt32LE(0);
  }

  async readInt64(): Promise<bigint> {
    const buf = await this.readBytes(8);
    return buf.readBigInt64LE(0);
  }

  async readFloat32(): Promise<number> {
    const buf = await this.readBytes(4);
    return buf.readFloatLE(0);
  }

  async readFloat64(): Promise<number> {
    const buf = await this.readBytes(8);
    return buf.readDoubleLE(0);
  }

  async readBool(): Promise<boolean> {
    const val = await this.readUint8();
    return val !== 0;
  }

  async readString(): Promise<string> {
    const length = await this.readUint64();
    const buf = await this.readBytes(Number(length));
    return buf.toString('utf-8');
  }

  getCurrentOffset(): bigint {
    return this.fileOffset - BigInt(this.bufferLength - this.bufferOffset);
  }
}

/**
 * Read a GGUF value based on its type
 */
async function readValue(
  reader: BufferedReader,
  type: GGUFValueType
): Promise<unknown> {
  switch (type) {
    case GGUFValueType.UINT8:
      return reader.readUint8();
    case GGUFValueType.INT8:
      return reader.readInt8();
    case GGUFValueType.UINT16:
      return reader.readUint16();
    case GGUFValueType.INT16:
      return reader.readInt16();
    case GGUFValueType.UINT32:
      return reader.readUint32();
    case GGUFValueType.INT32:
      return reader.readInt32();
    case GGUFValueType.FLOAT32:
      return reader.readFloat32();
    case GGUFValueType.BOOL:
      return reader.readBool();
    case GGUFValueType.STRING:
      return reader.readString();
    case GGUFValueType.UINT64:
      return reader.readUint64();
    case GGUFValueType.INT64:
      return reader.readInt64();
    case GGUFValueType.FLOAT64:
      return reader.readFloat64();
    case GGUFValueType.ARRAY: {
      const elemType = (await reader.readUint32()) as GGUFValueType;
      const count = await reader.readUint64();
      const arr: unknown[] = [];
      for (let i = 0n; i < count; i++) {
        arr.push(await readValue(reader, elemType));
      }
      return arr;
    }
    default:
      throw new Error(`Unknown GGUF value type: ${type}`);
  }
}

/**
 * Parse a GGUF file and extract header, metadata, and tensor info
 */
export async function parseGGUF(
  filePath: string,
  maxArraySize = 1024
): Promise<GGUFFile> {
  const handle = await open(filePath, 'r');
  const reader = new BufferedReader(handle);

  try {
    // Read header
    const magic = await reader.readUint32();
    if (magic !== GGUF_MAGIC) {
      throw new Error(
        `Invalid GGUF magic number: 0x${magic.toString(16)} (expected 0x${GGUF_MAGIC.toString(16)})`
      );
    }

    const version = await reader.readUint32();
    if (version < 2 || version > 3) {
      throw new Error(`Unsupported GGUF version: ${version}`);
    }

    const tensorCount = await reader.readUint64();
    const metadataKVCount = await reader.readUint64();

    const header: GGUFHeader = {
      magic,
      version,
      tensorCount,
      metadataKVCount,
    };

    // Read metadata
    const metadata: GGUFMetadata = {};
    for (let i = 0n; i < metadataKVCount; i++) {
      const key = await reader.readString();
      const valueType = (await reader.readUint32()) as GGUFValueType;

      // For arrays, check if we should skip large ones
      if (valueType === GGUFValueType.ARRAY) {
        const elemType = (await reader.readUint32()) as GGUFValueType;
        const count = await reader.readUint64();

        if (maxArraySize >= 0 && Number(count) > maxArraySize) {
          // Skip large arrays
          const elemSize = getElementSize(elemType);
          if (elemSize > 0) {
            await reader.readBytes(Number(count) * elemSize);
          } else {
            // Variable-size elements (strings), must read them
            for (let j = 0n; j < count; j++) {
              await readValue(reader, elemType);
            }
          }
          metadata[key] = `[array of ${count} elements]`;
        } else {
          const arr: unknown[] = [];
          for (let j = 0n; j < count; j++) {
            arr.push(await readValue(reader, elemType));
          }
          metadata[key] = arr;
        }
      } else {
        metadata[key] = await readValue(reader, valueType);
      }
    }

    // Read tensor info
    const tensors: GGUFTensorInfo[] = [];
    for (let i = 0n; i < tensorCount; i++) {
      const name = await reader.readString();
      const nDimensions = await reader.readUint32();

      const dimensions: bigint[] = [];
      for (let j = 0; j < nDimensions; j++) {
        dimensions.push(await reader.readUint64());
      }

      const type = (await reader.readUint32()) as GGMLType;
      const offset = await reader.readUint64();

      tensors.push({
        name,
        nDimensions,
        dimensions,
        type,
        offset,
      });
    }

    // Current position is where tensor data starts
    // GGUF v3 requires tensor data to be aligned to 32 bytes
    const rawOffset = reader.getCurrentOffset();
    const alignment = 32n;
    const tensorDataOffset = ((rawOffset + alignment - 1n) / alignment) * alignment;

    return {
      header,
      metadata,
      tensors,
      tensorDataOffset,
    };
  } finally {
    await handle.close();
  }
}

/**
 * Get the byte size of a fixed-size element type
 */
function getElementSize(type: GGUFValueType): number {
  switch (type) {
    case GGUFValueType.UINT8:
    case GGUFValueType.INT8:
    case GGUFValueType.BOOL:
      return 1;
    case GGUFValueType.UINT16:
    case GGUFValueType.INT16:
      return 2;
    case GGUFValueType.UINT32:
    case GGUFValueType.INT32:
    case GGUFValueType.FLOAT32:
      return 4;
    case GGUFValueType.UINT64:
    case GGUFValueType.INT64:
    case GGUFValueType.FLOAT64:
      return 8;
    default:
      return 0; // Variable size
  }
}

/**
 * Extract model architecture info from GGUF metadata
 */
export function extractArchitecture(gguf: GGUFFile): ModelArchitecture {
  const metadata = gguf.metadata;

  const architecture =
    (metadata[GGUF_KEYS.GENERAL_ARCHITECTURE] as string) || 'unknown';

  // Architecture-specific keys are prefixed with the architecture name
  const prefix = architecture;

  return {
    architecture,
    name: metadata[GGUF_KEYS.GENERAL_NAME] as string | undefined,
    contextLength: Number(
      metadata[`${prefix}${GGUF_KEYS.CONTEXT_LENGTH}`] || 2048
    ),
    embeddingLength: Number(
      metadata[`${prefix}${GGUF_KEYS.EMBEDDING_LENGTH}`] || 0
    ),
    blockCount: Number(metadata[`${prefix}${GGUF_KEYS.BLOCK_COUNT}`] || 0),
    headCount: Number(
      metadata[`${prefix}${GGUF_KEYS.ATTENTION_HEAD_COUNT}`] || 0
    ),
    headCountKV: Number(
      metadata[`${prefix}${GGUF_KEYS.ATTENTION_HEAD_COUNT_KV}`] || 0
    ),
    vocabSize: Array.isArray(metadata[GGUF_KEYS.TOKENIZER_LIST])
      ? (metadata[GGUF_KEYS.TOKENIZER_LIST] as unknown[]).length
      : 0,
    quantization: getQuantizationType(gguf),
  };
}

/**
 * Determine the quantization type from tensor types
 */
function getQuantizationType(gguf: GGUFFile): string {
  const typeCounts: Record<number, number> = {};

  for (const tensor of gguf.tensors) {
    typeCounts[tensor.type] = (typeCounts[tensor.type] || 0) + 1;
  }

  // Find the most common type (excluding f32 and f16 which are usually for special tensors)
  let maxCount = 0;
  let dominantType = GGMLType.F16;

  for (const [type, count] of Object.entries(typeCounts)) {
    const t = Number(type);
    if (t !== GGMLType.F32 && count > maxCount) {
      maxCount = count;
      dominantType = t;
    }
  }

  return GGMLType[dominantType] || `Q${dominantType}`;
}

/**
 * Get chat template from GGUF metadata if present
 */
export function getChatTemplate(gguf: GGUFFile): string | undefined {
  return gguf.metadata[GGUF_KEYS.CHAT_TEMPLATE] as string | undefined;
}

/**
 * Quick check if a file is a valid GGUF file
 */
export async function isGGUF(filePath: string): Promise<boolean> {
  const handle = await open(filePath, 'r');
  try {
    const buf = Buffer.alloc(4);
    await handle.read(buf, 0, 4, 0);
    return buf.readUInt32LE(0) === GGUF_MAGIC;
  } finally {
    await handle.close();
  }
}
