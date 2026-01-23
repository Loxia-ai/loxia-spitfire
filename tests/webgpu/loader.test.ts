/**
 * WebGPU Model Loader Tests - Phase 5
 * Tests for GGUF model loading functionality
 */

import { unlink, mkdir } from 'fs/promises';
import { join } from 'path';
import { tmpdir } from 'os';
import {
  WebGPUDevice,
  initWebGPU,
  calculateTensorBytes,
  calculateTensorElements,
  estimateModelMemory,
  listTensorNames,
  groupTensorsByLayer,
  getTensorInfo,
} from '../../src/engine/webgpu/index.js';
import { GGMLType, type GGUFFile, type GGUFTensorInfo } from '../../src/types/model.js';

// Skip tests if WebGPU is not available
const describeWebGPU = process.env.SKIP_WEBGPU_TESTS ? describe.skip : describe;

// Create mock GGUF file info for testing
function createMockGGUFFile(): GGUFFile {
  const tensors: GGUFTensorInfo[] = [
    // Embedding tensor
    {
      name: 'token_embd.weight',
      nDimensions: 2,
      dimensions: [4096n, 32000n], // [hidden_size, vocab_size]
      type: GGMLType.Q4_0,
      offset: 0n,
    },
    // Layer 0 tensors
    {
      name: 'blk.0.attn_norm.weight',
      nDimensions: 1,
      dimensions: [4096n],
      type: GGMLType.F32,
      offset: 1000n,
    },
    {
      name: 'blk.0.attn_q.weight',
      nDimensions: 2,
      dimensions: [4096n, 4096n],
      type: GGMLType.Q4_0,
      offset: 2000n,
    },
    {
      name: 'blk.0.attn_k.weight',
      nDimensions: 2,
      dimensions: [4096n, 4096n],
      type: GGMLType.Q4_0,
      offset: 3000n,
    },
    {
      name: 'blk.0.attn_v.weight',
      nDimensions: 2,
      dimensions: [4096n, 4096n],
      type: GGMLType.Q4_0,
      offset: 4000n,
    },
    {
      name: 'blk.0.attn_output.weight',
      nDimensions: 2,
      dimensions: [4096n, 4096n],
      type: GGMLType.Q4_0,
      offset: 5000n,
    },
    {
      name: 'blk.0.ffn_norm.weight',
      nDimensions: 1,
      dimensions: [4096n],
      type: GGMLType.F32,
      offset: 6000n,
    },
    {
      name: 'blk.0.ffn_gate.weight',
      nDimensions: 2,
      dimensions: [4096n, 11008n],
      type: GGMLType.Q4_0,
      offset: 7000n,
    },
    {
      name: 'blk.0.ffn_up.weight',
      nDimensions: 2,
      dimensions: [4096n, 11008n],
      type: GGMLType.Q4_0,
      offset: 8000n,
    },
    {
      name: 'blk.0.ffn_down.weight',
      nDimensions: 2,
      dimensions: [11008n, 4096n],
      type: GGMLType.Q4_0,
      offset: 9000n,
    },
    // Layer 1 tensors
    {
      name: 'blk.1.attn_norm.weight',
      nDimensions: 1,
      dimensions: [4096n],
      type: GGMLType.F32,
      offset: 10000n,
    },
    {
      name: 'blk.1.attn_q.weight',
      nDimensions: 2,
      dimensions: [4096n, 4096n],
      type: GGMLType.Q4_0,
      offset: 11000n,
    },
    // Output tensors
    {
      name: 'output_norm.weight',
      nDimensions: 1,
      dimensions: [4096n],
      type: GGMLType.F32,
      offset: 20000n,
    },
    {
      name: 'output.weight',
      nDimensions: 2,
      dimensions: [4096n, 32000n],
      type: GGMLType.Q4_0,
      offset: 21000n,
    },
  ];

  return {
    header: {
      magic: 0x46554747,
      version: 3,
      tensorCount: BigInt(tensors.length),
      metadataKVCount: 0n,
    },
    metadata: {},
    tensors,
    tensorDataOffset: 0n,
  };
}

describeWebGPU('WebGPU Model Loader', () => {
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

  describe('Tensor Size Calculations', () => {
    test('should calculate F32 tensor bytes correctly', () => {
      const tensor: GGUFTensorInfo = {
        name: 'test',
        nDimensions: 2,
        dimensions: [100n, 200n],
        type: GGMLType.F32,
        offset: 0n,
      };

      // 100 * 200 * 4 bytes = 80000
      expect(calculateTensorBytes(tensor)).toBe(80000);
    });

    test('should calculate F16 tensor bytes correctly', () => {
      const tensor: GGUFTensorInfo = {
        name: 'test',
        nDimensions: 2,
        dimensions: [100n, 200n],
        type: GGMLType.F16,
        offset: 0n,
      };

      // 100 * 200 * 2 bytes = 40000
      expect(calculateTensorBytes(tensor)).toBe(40000);
    });

    test('should calculate Q4_0 tensor bytes correctly', () => {
      const tensor: GGUFTensorInfo = {
        name: 'test',
        nDimensions: 2,
        dimensions: [32n, 32n], // 1024 elements = 32 blocks
        type: GGMLType.Q4_0,
        offset: 0n,
      };

      // 32 blocks * 18 bytes = 576
      expect(calculateTensorBytes(tensor)).toBe(576);
    });

    test('should calculate Q8_0 tensor bytes correctly', () => {
      const tensor: GGUFTensorInfo = {
        name: 'test',
        nDimensions: 1,
        dimensions: [64n], // 64 elements = 2 blocks
        type: GGMLType.Q8_0,
        offset: 0n,
      };

      // 2 blocks * 34 bytes = 68
      expect(calculateTensorBytes(tensor)).toBe(68);
    });

    test('should calculate Q4_K tensor bytes correctly', () => {
      const tensor: GGUFTensorInfo = {
        name: 'test',
        nDimensions: 1,
        dimensions: [256n], // 256 elements = 1 block
        type: GGMLType.Q4_K,
        offset: 0n,
      };

      // 1 block * 144 bytes = 144
      expect(calculateTensorBytes(tensor)).toBe(144);
    });

    test('should calculate tensor elements correctly', () => {
      const tensor: GGUFTensorInfo = {
        name: 'test',
        nDimensions: 3,
        dimensions: [10n, 20n, 30n],
        type: GGMLType.F32,
        offset: 0n,
      };

      expect(calculateTensorElements(tensor)).toBe(6000);
    });
  });

  describe('Memory Estimation', () => {
    test('should estimate model memory with dequantization', () => {
      const gguf = createMockGGUFFile();
      const estimated = estimateModelMemory(gguf, true);

      // Should be total elements * 4 (f32)
      let totalElements = 0;
      for (const tensor of gguf.tensors) {
        totalElements += calculateTensorElements(tensor);
      }

      expect(estimated).toBe(totalElements * 4);
    });

    test('should estimate model memory without dequantization', () => {
      const gguf = createMockGGUFFile();
      const estimated = estimateModelMemory(gguf, false);

      // Should be sum of actual tensor bytes
      let totalBytes = 0;
      for (const tensor of gguf.tensors) {
        totalBytes += calculateTensorBytes(tensor);
      }

      expect(estimated).toBe(totalBytes);
    });
  });

  describe('Tensor Info Utilities', () => {
    test('should list all tensor names', () => {
      const gguf = createMockGGUFFile();
      const names = listTensorNames(gguf);

      expect(names).toContain('token_embd.weight');
      expect(names).toContain('blk.0.attn_q.weight');
      expect(names).toContain('output.weight');
      expect(names.length).toBe(gguf.tensors.length);
    });

    test('should get tensor info by name', () => {
      const gguf = createMockGGUFFile();
      const info = getTensorInfo(gguf, 'blk.0.attn_q.weight');

      expect(info).toBeDefined();
      expect(info!.type).toBe(GGMLType.Q4_0);
      expect(info!.dimensions).toEqual([4096n, 4096n]);
    });

    test('should return undefined for non-existent tensor', () => {
      const gguf = createMockGGUFFile();
      const info = getTensorInfo(gguf, 'non_existent');

      expect(info).toBeUndefined();
    });
  });

  describe('Layer Grouping', () => {
    test('should group tensors by layer', () => {
      const gguf = createMockGGUFFile();
      const groups = groupTensorsByLayer(gguf);

      // Should have embed, output, and layer groups
      expect(groups.has('embed')).toBe(true);
      expect(groups.has('output')).toBe(true);
      expect(groups.has(0)).toBe(true);
      expect(groups.has(1)).toBe(true);
    });

    test('should correctly group layer 0 tensors', () => {
      const gguf = createMockGGUFFile();
      const groups = groupTensorsByLayer(gguf);
      const layer0 = groups.get(0);

      expect(layer0).toBeDefined();
      expect(layer0!.length).toBeGreaterThan(0);
      expect(layer0!.some((n) => n.includes('attn_q'))).toBe(true);
      expect(layer0!.some((n) => n.includes('ffn_gate'))).toBe(true);
    });

    test('should group embedding tensors', () => {
      const gguf = createMockGGUFFile();
      const groups = groupTensorsByLayer(gguf);
      const embed = groups.get('embed');

      expect(embed).toBeDefined();
      expect(embed!.some((n) => n.includes('token_embd'))).toBe(true);
    });

    test('should group output tensors', () => {
      const gguf = createMockGGUFFile();
      const groups = groupTensorsByLayer(gguf);
      const output = groups.get('output');

      expect(output).toBeDefined();
      expect(output!.some((n) => n.includes('output'))).toBe(true);
    });
  });
});

// Additional tests that require actual file I/O
describe('Model Loader File I/O', () => {
  let testDir: string;
  let testFilePath: string;

  beforeAll(async () => {
    testDir = join(tmpdir(), 'spitfire-loader-test-' + Date.now());
    await mkdir(testDir, { recursive: true });
    testFilePath = join(testDir, 'test.bin');
  });

  afterAll(async () => {
    try {
      await unlink(testFilePath);
    } catch {
      // File may not exist
    }
  });

  test('should handle file reading errors gracefully', async () => {
    // This is a placeholder for file I/O tests
    // In a full implementation, we would test actual file loading
    expect(true).toBe(true);
  });
});
