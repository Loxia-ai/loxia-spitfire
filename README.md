# Spitfire

**Standalone** Node.js module for running LLMs locally. No external dependencies required - includes both WebAssembly and WebGPU inference engines built from scratch.

## Features

- **Truly Standalone** - No Ollama or external runtime required
- **WebGPU Acceleration** - GPU-accelerated inference with automatic fallback
- **WebAssembly Engine** - Cross-platform CPU inference, runs anywhere Node.js runs
- **Ollama-compatible** - Works with existing GGUF models
- **Quantization Support** - Q4_0, Q8_0, Q4_K dequantization on GPU
- **HTTP API** - Drop-in replacement for Ollama's REST API
- **Programmatic API** - Clean TypeScript/JavaScript API
- **Streaming** - Full streaming support for text generation

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Your Application                         │
├─────────────────────────────────────────────────────────────┤
│    Spitfire API    │    SpitfireServer (HTTP/Fastify)       │
├────────────────────┴────────────────────────────────────────┤
│                      Engine Factory                          │
│              (Auto-detects best available)                   │
├──────────────────────────┬──────────────────────────────────┤
│      WebGPU Engine       │         WASM Engine              │
│   (GPU Acceleration)     │   (llama.cpp via Emscripten)     │
│  ┌────────────────────┐  │  ┌────────────────────────────┐  │
│  │ Tensor Ops (WGSL)  │  │  │    llama.cpp (WASM)        │  │
│  │ Attention/FFN      │  │  │    SIMD128 optimized       │  │
│  │ Quantization       │  │  └────────────────────────────┘  │
│  │ GGUF Loader        │  │                                  │
│  └────────────────────┘  │                                  │
├──────────────────────────┴──────────────────────────────────┤
│                      Node.js / V8                            │
└─────────────────────────────────────────────────────────────┘
```

**No subprocess, no external binaries** - everything runs in-process.

## Installation

```bash
npm install @loxia-labs/spitfire
```

## Quick Start

### Automatic Engine Selection

```typescript
import { createBestEngine } from '@loxia-labs/spitfire';

// Automatically uses WebGPU if available, falls back to WASM
const engine = await createBestEngine();

await engine.loadModel('/path/to/model.gguf');

const result = await engine.generate('Hello, how are you?', {
  maxTokens: 100,
  temperature: 0.8
});
console.log(result.text);

await engine.shutdown();
```

### Using WebGPU Engine (GPU Acceleration)

```typescript
import { createWebGPUEngine } from '@loxia-labs/spitfire';

const engine = createWebGPUEngine();

// Initialize GPU device
await engine.init();

// Check GPU capabilities
const caps = engine.getCapabilities();
console.log(`Max buffer size: ${caps.maxBufferSize / 1024 / 1024} MB`);

// Load and run model
await engine.loadModel('/path/to/model.gguf', {
  contextLength: 2048
});

const result = await engine.generate('Explain quantum computing', {
  maxTokens: 200,
  temperature: 0.7,
  topK: 40
});

console.log(result.text);
await engine.shutdown();
```

### Using WASM Engine (CPU)

```typescript
import { createWasmEngine } from '@loxia-labs/spitfire';

const engine = createWasmEngine();

await engine.init();
await engine.loadModel('/path/to/model.gguf', {
  contextLength: 2048,
  numThreads: 4
});

const result = await engine.generate('Hello!', {
  maxTokens: 100
});

await engine.shutdown();
```

### Using the High-Level API

```typescript
import { Spitfire } from '@loxia-labs/spitfire';

const spitfire = new Spitfire();

// Generate completion
const response = await spitfire.generate({
  model: 'llama3.2',
  prompt: 'What is the meaning of life?'
});
console.log(response.response);

// Chat
const chat = await spitfire.chat({
  model: 'llama3.2',
  messages: [
    { role: 'user', content: 'Hello!' }
  ]
});
console.log(chat.message.content);

await spitfire.shutdown();
```

### HTTP Server

```typescript
import { SpitfireServer } from '@loxia-labs/spitfire';

const server = new SpitfireServer({ port: 11434 });
await server.start();

// Now accessible at http://localhost:11434
// Compatible with Ollama API clients
```

### CLI

```bash
# List models
spitfire list

# Run inference
spitfire run /path/to/model.gguf "Hello!"

# Start HTTP server
spitfire serve --port 11434
```

## Building from Source

### Prerequisites

**For WASM Engine:**
Install [Emscripten SDK](https://emscripten.org/docs/getting_started/downloads.html):

```bash
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install latest
./emsdk activate latest
source ./emsdk_env.sh
```

**For WebGPU Engine:**
WebGPU support is included via the `webgpu` npm package (Dawn bindings).

### Build

```bash
cd spitfire
npm install
npm run build:wasm  # Build WASM engine
npm run build:ts    # Build TypeScript
```

## API Reference

### Engine Factory

```typescript
import {
  createEngine,
  createBestEngine,
  detectBestEngine
} from '@loxia-labs/spitfire';

// Create specific engine type
const engine = createEngine({ type: 'webgpu' }); // or 'wasm'

// Auto-detect and create best engine
const bestEngine = await createBestEngine();

// Just detect without creating
const engineType = await detectBestEngine(); // 'webgpu' or 'wasm'
```

### WebGPUEngine

GPU-accelerated inference engine.

```typescript
const engine = createWebGPUEngine();

await engine.init();
engine.getCapabilities();  // GPU limits and features

await engine.loadModel(path, {
  contextLength?: number;
  batchSize?: number;
});

await engine.generate(prompt, {
  maxTokens?: number;
  temperature?: number;
  topK?: number;
  topP?: number;
  stop?: string[];
});

await engine.embed(text);  // Get embeddings
await engine.unload();     // Unload model
await engine.shutdown();   // Release GPU
```

### WasmEngine

CPU-based inference engine.

```typescript
const engine = createWasmEngine({
  wasmPath?: string;
  numThreads?: number;
});

await engine.init();
await engine.loadModel(path, options);
await engine.generate(prompt, options);
await engine.embed(text);
await engine.shutdown();
```

### Spitfire (High-Level API)

```typescript
const spitfire = new Spitfire({
  modelsPath?: string;
  maxLoadedModels?: number;
  defaultKeepAlive?: string;
  engineType?: 'webgpu' | 'wasm' | 'auto';
});

await spitfire.generate(request);
await spitfire.chat(request);
await spitfire.embed(request);
await spitfire.list();
await spitfire.shutdown();
```

### SpitfireServer (HTTP API)

```typescript
const server = new SpitfireServer({
  host?: string;  // Default: '127.0.0.1'
  port?: number;  // Default: 11434
});

await server.start();
await server.stop();
```

## Model Compatibility

Spitfire works with GGUF model files. Supported quantization formats:

| Format | WebGPU | WASM |
|--------|--------|------|
| F32 | Yes | Yes |
| F16 | Yes | Yes |
| Q8_0 | Yes | Yes |
| Q4_0 | Yes | Yes |
| Q4_K | Yes | Yes |
| Q5_K | Partial | Yes |
| Q6_K | Partial | Yes |

**Get models from:**
1. **HuggingFace** - Many quantized models available
2. **Ollama models** - Copy from `~/.ollama/models/`
3. **Convert your own** - Use llama.cpp's conversion tools

## Performance

| Metric | WebGPU Engine | WASM Engine |
|--------|---------------|-------------|
| Speed | ~80% of native | ~70-85% of native |
| Memory | GPU VRAM | Up to 4GB |
| GPU | Required | Not used |
| Threading | GPU parallel | Multi-thread CPU |
| SIMD | GPU compute | WASM SIMD128 |

### WebGPU Optimizations

- Tiled matrix multiplication (8x8 tiles)
- Shader caching and precompilation
- Buffer pooling for memory reuse
- Numerically stable softmax
- Fused attention kernels

## Project Structure

```
spitfire/
├── src/
│   ├── engine/
│   │   ├── index.ts           # Engine factory
│   │   ├── wasm-engine.ts     # WASM inference
│   │   ├── webgpu-engine.ts   # WebGPU inference
│   │   └── webgpu/
│   │       ├── device.ts      # GPU device management
│   │       ├── buffer.ts      # Buffer utilities
│   │       ├── shader.ts      # WGSL shader compilation
│   │       ├── tensor.ts      # GPU Tensor class
│   │       ├── ops/           # Tensor operations
│   │       ├── layers/        # Transformer layers
│   │       ├── quant/         # Quantization support
│   │       ├── model/         # GGUF loading
│   │       └── perf/          # Performance monitoring
│   ├── types/                 # TypeScript definitions
│   ├── model/                 # Model management
│   ├── server/                # HTTP API
│   └── spitfire.ts            # Main API class
├── native/
│   ├── llama.cpp/             # llama.cpp source
│   ├── ggml/                  # ggml source
│   └── wasm/                  # WASM bindings
├── tests/
│   └── webgpu/                # WebGPU test suite (146 tests)
└── dist/
    └── wasm/                  # Compiled WASM files
```

## Testing

```bash
# Run all tests
npm test

# Run WebGPU tests only
npm test -- --testPathPattern=webgpu

# Run specific test file
npm test -- --testPathPattern=webgpu/tensor
```

**Test Coverage:** 228 tests across 10 test suites

## Requirements

- **Node.js** 18+
- **WebGPU** (for GPU acceleration): Supported in Node.js via Dawn bindings
- **Emscripten** (for building WASM): Only needed if building from source

## License

MIT

## Credits

**Written by Daniel Suissa, [Loxia.ai](https://loxia.ai)**

Visit us at **[https://autopilot.loxia.ai](https://autopilot.loxia.ai)**

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - The inference engine
- [ggml](https://github.com/ggerganov/ggml) - Tensor library
- [Ollama](https://github.com/ollama/ollama) - API design inspiration
- [WebLLM](https://github.com/mlc-ai/web-llm) - WebGPU LLM inspiration
- [Dawn](https://dawn.googlesource.com/dawn) - WebGPU implementation for Node.js
