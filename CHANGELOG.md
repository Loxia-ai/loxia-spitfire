# Changelog

All notable changes to Spitfire will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.1] - 2025-01-25

### Performance
- **GPU-Resident Token Extraction**: New `sliceLastRow` GPU shader extracts last token on GPU
  - Eliminates large GPU→CPU transfers (was transferring entire sequence every forward pass)
  - For 100-token sequences, reduces transfer from ~800KB to ~8KB per forward pass
- **Optimized Causal Attention Shader**: Rewrote attention kernel for efficiency
  - Q@K^T computed once and cached in shared memory (was computed 3x per workgroup)
  - Supports up to 1024 keys in shared memory for fast softmax
  - Proper workgroup synchronization eliminates race conditions
- **GPU Broadcast Add**: New `broadcastAdd` shader for bias operations
  - Replaces CPU-based bias addition that required GPU→CPU→GPU transfers
  - Eliminates 108 GPU-CPU transfers per token (3 biases × 36 layers)
- **Async GPU Execution**: `executeCompute` no longer syncs by default
  - GPU commands are batched and only synced when results are needed
  - Reduces synchronization overhead from ~720 syncs to ~2 per token

## [1.2.0] - 2025-01-25

### Added
- **GPU-Accelerated Attention**: Moved attention computation from CPU to GPU
  - New `causalAttention` WebGPU compute shader
  - Fused Q@K^T, causal masking, softmax, and @V operations
  - Supports Grouped Query Attention (GQA) for efficient KV sharing
  - 10-100x faster than previous CPU-based attention
- **KV Cache**: Implemented key-value caching for dramatically faster token generation
  - Prefill processes the full prompt once, caching K/V projections per layer
  - Subsequent tokens only compute for the new position, reusing cached K/V
  - Reduces computation from O(n²) to O(n) per token during generation
  - 10-50x speedup depending on sequence length
- **Configurable Debug Mode**: Added `debug` option to `WebGPUEngineOptions`
  - When `debug: false` (default), skips expensive GPU-CPU transfers for logging
  - Significantly reduces latency in production use

### Changed
- **RoPE with Position Offset**: `applyRope` now accepts `startPos` parameter
  - Enables correct position encoding during incremental inference
  - Essential for KV cache to produce correct attention patterns
- **Efficient Top-K Sampling**: Replaced O(vocab × k) scanning with O(vocab × log(k)) min-heap
  - More efficient selection of top-k tokens for large vocabularies

## [1.1.1] - 2025-01-25

### Fixed
- **Token Decoding**: GPT-2 byte characters now properly decoded back to readable text
  - `Ġ` decoded to space, `Ċ` decoded to newline
  - Output is now human-readable (no more "HelloĠworld")
  - Added `decodeGPT2Bytes()` method with cached lookup tables

## [1.1.0] - 2025-01-25

### Added
- **Chat Template Support**: Automatic Jinja2 chat template rendering using nunjucks
  - Reads chat templates from GGUF metadata
  - Supports Qwen2, Llama, and other HuggingFace-style templates
  - New `rawPrompt` option to skip automatic formatting
- **GPT-2 Byte-Level Tokenization**: Proper handling of special characters
  - Newlines encoded as `Ċ` (U+010A)
  - Spaces encoded as `Ġ` (U+0120)
  - Full compatibility with Qwen2 and similar tokenizers

### Fixed
- **Tokenizer Special Token Handling**: Fixed duplicate token bug in `preTokenize()`
  - `split()` with capturing groups now handled correctly
  - Special tokens no longer duplicated in output
- **Q6_K Dequantization**: Corrected scale indexing for proper weight decoding
- **Q4_K Dequantization**: Fixed block layout parsing and scale application
- **RoPE Implementation**: Verified correct rope_neox pattern matching llama.cpp

### Changed
- Improved tokenizer to handle GPT-2 style byte encoding
- Better error messages for template rendering failures
- Default stop sequences now include `<|im_end|>` for Qwen2 models

### Dependencies
- Added `nunjucks` ^3.2.4 for Jinja2 template rendering

## [1.0.0] - 2025-01-24

### Added
- Initial release
- WebGPU inference engine with GPU acceleration
- WebAssembly inference engine (llama.cpp via Emscripten)
- GGUF model file support
- Quantization support: F32, F16, Q8_0, Q4_0, Q4_K, Q5_K, Q6_K
- Streaming text generation
- HTTP API compatible with Ollama
- CLI for model management and inference
- TypeScript/JavaScript programmatic API
