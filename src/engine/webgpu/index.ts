/**
 * WebGPU Engine Module Exports
 */

export {
  WebGPUDevice,
  getWebGPUDevice,
  initWebGPU,
  type GPUCapabilities,
  type DeviceOptions,
} from './device.js';

export {
  createBuffer,
  createBufferWithData,
  createStorageBuffer,
  createStorageBufferWithData,
  createUniformBuffer,
  createUniformBufferWithData,
  createStagingBuffer,
  writeBuffer,
  readBuffer,
  readBufferFloat32,
  readBufferInt32,
  readBufferUint32,
  copyBuffer,
  BufferPool,
  getBufferPool,
  type BufferUsage,
  type BufferOptions,
} from './buffer.js';

export {
  createShaderModule,
  createComputePipeline,
  createComputePipelineFromSource,
  createBindGroup,
  dispatchCompute,
  executeCompute,
  clearShaderCache,
  calculateWorkgroups,
  calculateWorkgroups2D,
  calculateWorkgroups3D,
  generateElementwiseShader,
  generateBinaryShader,
  WGSL_UTILS,
  getCommandBatcher,
  flushCommandBatcher,
  flushAndSyncCommandBatcher,
  getCommandBatcherStats,
  resetCommandBatcherStats,
  requestBufferDestroy,
  type ComputePipelineOptions,
  type BindGroupEntry,
} from './shader.js';

export {
  Tensor,
  shapeToSize,
  shapeToStrides,
  withTensors,
  type TensorDType,
  type TensorOptions,
} from './tensor.js';

export * as ops from './ops/index.js';

export {
  layerNorm,
  rmsNorm,
  addRmsNorm,
  applyRope,
  attention,
  feedForward,
  feedForwardQ,
  matmulQ,
  mlp,
  resetMatmulQDebugCount,
  type AttentionConfig,
  type WeightTensor as LayerWeightTensor,
} from './layers/index.js';

export {
  getBlockSize,
  getBytesPerBlock,
  requiresDequantization,
  dequantize,
  dequantizeQ4_0,
  dequantizeQ8_0,
  dequantizeQ4_K,
  quantizeToQ8_0,
  QuantizedTensor,
  gemvQ8_0,
  gemvQ4_K,
  gemmQ8_0,
  gemmQ4_K,
  type SupportedQuantType,
} from './quant/index.js';

export {
  loadTensor,
  loadWeight,
  loadTensors,
  loadTensorsByPattern,
  loadLlamaWeights,
  disposeLlamaWeights,
  calculateTensorBytes,
  calculateTensorElements,
  estimateModelMemory,
  getTensorInfo,
  listTensorNames,
  groupTensorsByLayer,
  type TensorLoadOptions,
  type LoadedTensor,
  type LoadedWeight,
  type WeightTensor,
  type LlamaWeights,
  type LlamaLayerWeights,
} from './model/index.js';

export {
  PerfMonitor,
  getPerfMonitor,
  MemoryTracker,
  getMemoryTracker,
  getOptimalWorkgroupConfig,
  calculateOptimalWorkgroups1D,
  calculateOptimalWorkgroups2D,
  benchmark,
  type PerfMetrics,
  type PerfStats,
  type WorkgroupConfig,
  GPUProfiler,
  getGPUProfiler,
  enableGPUProfiling,
  disableGPUProfiling,
  destroyGPUProfiler,
  type ProfileResult,
} from './perf/index.js';
