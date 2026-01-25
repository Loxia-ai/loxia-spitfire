# Changelog

All notable changes to Spitfire will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
