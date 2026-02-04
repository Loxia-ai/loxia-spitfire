/**
 * WebGPU Device Management
 * Handles GPU adapter discovery, device creation, and resource management
 */

/// <reference types="@webgpu/types" />

export interface GPUCapabilities {
  vendor: string;
  architecture: string;
  device: string;
  description: string;
  maxBufferSize: number;
  maxStorageBufferBindingSize: number;
  maxComputeWorkgroupSizeX: number;
  maxComputeWorkgroupSizeY: number;
  maxComputeWorkgroupSizeZ: number;
  maxComputeInvocationsPerWorkgroup: number;
  maxComputeWorkgroupsPerDimension: number;
}

export interface DeviceOptions {
  powerPreference?: 'low-power' | 'high-performance';
  requiredFeatures?: GPUFeatureName[];
  requiredLimits?: Record<string, number>;
  /** Dawn backend toggles (Node.js only). E.g. ['disable_timestamp_query_conversion'] */
  dawnToggles?: string[];
}

/**
 * WebGPU Device Manager
 * Singleton pattern for managing GPU device lifecycle
 */
export class WebGPUDevice {
  private static instance: WebGPUDevice | null = null;

  private gpu: GPU | null = null;
  private adapter: GPUAdapter | null = null;
  private device: GPUDevice | null = null;
  private capabilities: GPUCapabilities | null = null;
  private initialized = false;
  private initPromise: Promise<void> | null = null;

  private constructor() {}

  /**
   * Get the singleton instance
   */
  static getInstance(): WebGPUDevice {
    if (!WebGPUDevice.instance) {
      WebGPUDevice.instance = new WebGPUDevice();
    }
    return WebGPUDevice.instance;
  }

  /**
   * Check if WebGPU is available in this environment
   */
  static async isAvailable(): Promise<boolean> {
    try {
      const gpu = await WebGPUDevice.getGPU();
      if (!gpu) return false;

      const adapter = await gpu.requestAdapter();
      return adapter !== null;
    } catch {
      return false;
    }
  }

  /**
   * Get the GPU object (handles Node.js vs browser)
   */
  private static async getGPU(dawnToggles?: string[]): Promise<GPU | null> {
    // Try browser global first
    if (typeof globalThis !== 'undefined' && 'navigator' in globalThis) {
      const nav = (globalThis as { navigator?: { gpu?: GPU } }).navigator;
      if (nav?.gpu) {
        return nav.gpu;
      }
    }

    // Try Node.js WebGPU package
    try {
      const webgpu = await import('webgpu');
      // The webgpu package creates a GPU instance
      // Dawn toggles can be passed here (e.g. ['disable_timestamp_query_conversion'])
      const gpu = webgpu.create(dawnToggles || []);
      return gpu as GPU;
    } catch {
      return null;
    }
  }

  /**
   * Initialize the WebGPU device
   */
  async init(options: DeviceOptions = {}): Promise<void> {
    // Return existing promise if initialization is in progress
    if (this.initPromise) {
      return this.initPromise;
    }

    // Return immediately if already initialized
    if (this.initialized) {
      return;
    }

    this.initPromise = this.doInit(options);
    await this.initPromise;
  }

  private async doInit(options: DeviceOptions): Promise<void> {
    // Get GPU
    this.gpu = await WebGPUDevice.getGPU(options.dawnToggles);
    if (!this.gpu) {
      throw new Error('WebGPU not available in this environment');
    }

    // Request adapter
    this.adapter = await this.gpu.requestAdapter({
      powerPreference: options.powerPreference || 'high-performance',
    });

    if (!this.adapter) {
      throw new Error('Failed to get WebGPU adapter');
    }

    // Get adapter info (info is a property, not async method in newer API)
    const adapterInfo = this.adapter.info;

    // Request device with required features and limits
    // Request the maximum buffer size the adapter supports (up to 2GB)
    const adapterMaxBuffer = Number(this.adapter.limits.maxBufferSize) || 2 * 1024 * 1024 * 1024;
    const adapterMaxStorage = Number(this.adapter.limits.maxStorageBufferBindingSize) || 2 * 1024 * 1024 * 1024;

    const requiredLimits: Record<string, number> = {
      maxStorageBufferBindingSize: Math.min(adapterMaxStorage, 2 * 1024 * 1024 * 1024), // Up to 2GB
      maxBufferSize: Math.min(adapterMaxBuffer, 2 * 1024 * 1024 * 1024), // Up to 2GB
      ...options.requiredLimits,
    };

    // Clamp limits to what the adapter supports
    const limits = this.adapter.limits;
    for (const [key, value] of Object.entries(requiredLimits)) {
      const adapterLimit = limits[key as keyof GPUSupportedLimits];
      if (typeof adapterLimit === 'number') {
        requiredLimits[key] = Math.min(value, adapterLimit);
      }
    }

    // Request timestamp-query if adapter supports it (for GPU profiling)
    const features: GPUFeatureName[] = [...(options.requiredFeatures || [])];
    if (this.adapter.features.has('timestamp-query') && !features.includes('timestamp-query')) {
      features.push('timestamp-query');
    }

    this.device = await this.adapter.requestDevice({
      requiredFeatures: features,
      requiredLimits,
    });

    // Handle device lost
    this.device.lost.then((info: GPUDeviceLostInfo) => {
      console.error('WebGPU device lost:', info.message);
      this.handleDeviceLost(info);
    });

    // Store capabilities
    this.capabilities = {
      vendor: adapterInfo.vendor || 'unknown',
      architecture: adapterInfo.architecture || 'unknown',
      device: adapterInfo.device || 'unknown',
      description: adapterInfo.description || 'unknown',
      maxBufferSize: Number(this.device.limits.maxBufferSize),
      maxStorageBufferBindingSize: Number(
        this.device.limits.maxStorageBufferBindingSize
      ),
      maxComputeWorkgroupSizeX: this.device.limits.maxComputeWorkgroupSizeX,
      maxComputeWorkgroupSizeY: this.device.limits.maxComputeWorkgroupSizeY,
      maxComputeWorkgroupSizeZ: this.device.limits.maxComputeWorkgroupSizeZ,
      maxComputeInvocationsPerWorkgroup:
        this.device.limits.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupsPerDimension:
        this.device.limits.maxComputeWorkgroupsPerDimension,
    };

    this.initialized = true;
  }

  /**
   * Handle device lost event
   */
  private handleDeviceLost(_info: GPUDeviceLostInfo): void {
    this.initialized = false;
    this.device = null;
    this.adapter = null;
    this.initPromise = null;
  }

  /**
   * Get the GPU device
   */
  getDevice(): GPUDevice {
    if (!this.device) {
      throw new Error('WebGPU device not initialized. Call init() first.');
    }
    return this.device;
  }

  /**
   * Get GPU capabilities
   */
  getCapabilities(): GPUCapabilities {
    if (!this.capabilities) {
      throw new Error('WebGPU device not initialized. Call init() first.');
    }
    return this.capabilities;
  }

  /**
   * Check if device is initialized
   */
  isInitialized(): boolean {
    return this.initialized;
  }

  /**
   * Create a command encoder
   */
  createCommandEncoder(label?: string): GPUCommandEncoder {
    return this.getDevice().createCommandEncoder({ label });
  }

  /**
   * Submit commands to the GPU queue
   */
  submit(commandBuffers: GPUCommandBuffer[]): void {
    this.getDevice().queue.submit(commandBuffers);
  }

  /**
   * Wait for all pending GPU operations to complete
   */
  async sync(): Promise<void> {
    await this.getDevice().queue.onSubmittedWorkDone();
  }

  /**
   * Destroy the device and release resources
   */
  destroy(): void {
    if (this.device) {
      this.device.destroy();
    }
    this.device = null;
    this.adapter = null;
    this.gpu = null;
    this.capabilities = null;
    this.initialized = false;
    this.initPromise = null;
    WebGPUDevice.instance = null;
  }
}

/**
 * Get the global WebGPU device instance
 */
export function getWebGPUDevice(): WebGPUDevice {
  return WebGPUDevice.getInstance();
}

/**
 * Initialize WebGPU and return the device
 */
export async function initWebGPU(options?: DeviceOptions): Promise<WebGPUDevice> {
  const device = WebGPUDevice.getInstance();
  await device.init(options);
  return device;
}
